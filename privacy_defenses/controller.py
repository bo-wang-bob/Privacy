"""Training and update hooks for independent membership defenses.

Most paper defenses target a frozen backbone with trainable soft prompts.
Record-DP additionally supports full ResNet and Transformer Adapter parameter
scopes. A run selects exactly one controller, so methods are never combined
silently.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from utils.performance import measure_stage
from utils.per_sample_gradients import (
    GRAD_SAMPLE_BACKENDS, clipped_sum_from_losses, resolve_grad_sample_backend,
)
from utils.privacy_accounting import (
    calibrate_gaussian_noise,
    calibrate_poisson_sampled_gaussian_noise,
    gaussian_rdp_epsilon,
    poisson_sampled_gaussian_epsilon,
    private_generator,
)

from privacy_attacks.code_poison import (
    compromised_prompt_loss,
    generate_membership_encoding_samples,
)
from privacy_defenses.cofedmid import CoFedMID
from privacy_defenses.www_dp import WWWPrivacy, ino_weights
from privacy_defenses.www import (
    encode_training_batches,
    infer_other_clients_state,
    rank_loss_differences,
)
from privacy_defenses.www_validation import (
    _class_adjusted_spearman,
    _low_fpr_hits,
    _pearson,
    _safe_mean,
    _spearman,
    _top_bottom_masks,
)


SUPPORTED_DEFENSES = {
    "none",
    "perturb",
    "sparse",
    "mixup",
    "sampling",
    "data_aug",
    "data_aug_sampling",
    "cofedmid",
    "prompt_dp",
    "record_dp",
    "local_client_dp",
    "mist",
    "soft",
    "hamp",
    "www",
}

FEDMIA_BASELINE_DEFENSES = {
    "perturb",
    "sparse",
    "mixup",
    "sampling",
    "data_aug",
    "data_aug_sampling",
}


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError(
            "A privacy defense requires at least one trainable parameter."
        )
    return parameters


def _cross_entropy_with_soft_targets(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)


def _soft_targets(
    labels: torch.Tensor, classes: int, true_probability: float
) -> torch.Tensor:
    if classes <= 1:
        raise ValueError("Soft-label defenses require at least two classes.")
    true_probability = float(max(1.0 / classes, min(0.999, true_probability)))
    other = (1.0 - true_probability) / (classes - 1)
    targets = torch.full(
        (labels.numel(), classes), other, device=labels.device, dtype=torch.float32
    )
    return targets.scatter_(1, labels.view(-1, 1), true_probability)


def _assign_flat_gradient(
    parameters: list[torch.nn.Parameter], flat_gradient: torch.Tensor
) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = (
            flat_gradient[offset : offset + count].view_as(parameter).clone()
        )
        offset += count
    if offset != flat_gradient.numel():
        raise ValueError("Processed gradient does not match trainable parameter size.")


def _clip_joint_gradient_and_add_noise(
    parameters: list[torch.nn.Parameter],
    max_norm: float,
    noise_multiplier: float,
    generator: torch.Generator,
) -> tuple[float, float]:
    """Privatize one complete client gradient in a single joint L2 space."""
    if max_norm <= 0:
        raise ValueError("Client gradient max_norm must be positive.")
    if noise_multiplier < 0:
        raise ValueError("Client gradient noise_multiplier cannot be negative.")
    squared_norm = sum(
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    )
    if not isinstance(squared_norm, torch.Tensor):
        squared_norm = torch.zeros((), device=parameters[0].device)
    raw_norm = torch.sqrt(squared_norm).item()
    clip_factor = min(1.0, max_norm / max(raw_norm, 1e-12))
    noise_std = noise_multiplier * max_norm
    with torch.no_grad():
        for parameter in parameters:
            gradient = (
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad.detach().mul(clip_factor)
            )
            if noise_std > 0:
                gradient.add_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ),
                    alpha=noise_std,
                )
            parameter.grad = gradient
    return float(raw_norm), float(clip_factor)


def _per_sample_gradients(
    losses: torch.Tensor, parameters: list[torch.nn.Parameter]
) -> torch.Tensor:
    rows = []
    for index in range(losses.numel()):
        gradients = torch.autograd.grad(
            losses[index],
            parameters,
            retain_graph=index + 1 < losses.numel(),
            allow_unused=False,
        )
        rows.append(torch.cat([gradient.reshape(-1) for gradient in gradients]))
    return torch.stack(rows)


def _loop_clipped_gradient_sum(
    losses: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    max_norm: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Reference per-record clipping without materializing a [B, P] matrix."""
    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    factors = []
    for index in range(losses.numel()):
        gradients = torch.autograd.grad(
            losses[index],
            parameters,
            retain_graph=index + 1 < losses.numel(),
            allow_unused=False,
        )
        norm = torch.sqrt(
            sum(gradient.detach().float().square().sum() for gradient in gradients)
        ).clamp_min(1e-12)
        factor = (max_norm / norm).clamp(max=1.0)
        with torch.no_grad():
            for destination, gradient in zip(accumulated, gradients):
                destination.add_(gradient * factor.to(gradient))
        factors.append(factor.detach())
    return accumulated, torch.stack(factors)


def _vmap_clipped_gradient_sum(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    max_norm: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Vectorized per-record clipping for ordinary registered PyTorch models."""
    named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not named_parameters:
        raise ValueError("Record-DP requires trainable parameters.")

    def sample_loss(parameters, image, label):
        logits = torch.func.functional_call(
            model,
            parameters,
            (image.unsqueeze(0),),
            strict=False,
        )
        return F.cross_entropy(logits, label.unsqueeze(0))

    gradient_function = torch.func.grad(sample_loss)
    per_sample = torch.func.vmap(
        gradient_function,
        in_dims=(None, 0, 0),
        randomness="different",
    )(named_parameters, images, labels)
    norm_squared = torch.zeros(
        labels.numel(), device=labels.device, dtype=torch.float32
    )
    for gradient in per_sample.values():
        norm_squared.add_(gradient.detach().float().flatten(1).square().sum(dim=1))
    factors = (max_norm / norm_squared.sqrt().clamp_min(1e-12)).clamp(max=1.0)
    accumulated = []
    for gradient in per_sample.values():
        factor_shape = (labels.numel(),) + (1,) * (gradient.ndim - 1)
        accumulated.append(
            (gradient * factors.to(gradient.dtype).view(factor_shape)).sum(dim=0)
        )
    return accumulated, factors


def _mean_loader_loss(model: torch.nn.Module, loader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            total += float(F.cross_entropy(model(images), labels, reduction="sum"))
            samples += labels.numel()
    model.train(was_training)
    return total / max(samples, 1)


def _cap_top_logit_margin(logits: torch.Tensor, margin: float) -> torch.Tensor:
    if margin < 0 or logits.ndim < 2 or logits.shape[-1] < 2:
        return logits
    values, indices = logits.topk(2, dim=-1)
    capped_top = torch.minimum(values[..., :1], values[..., 1:2] + margin)
    return logits.scatter(-1, indices[..., :1], capped_top)


def _output_temperature_hook(_module, _inputs, output):
    if _module.training:
        return output
    temperature = float(
        getattr(
            _module,
            "_privacy_output_temperature",
            getattr(_module, "_hamp_output_temperature", 1.0),
        )
    )
    margin = getattr(_module, "_privacy_output_margin", None)
    margin = None if margin is None else float(margin)
    if isinstance(output, tuple):
        logits = output[0] / temperature
        if margin is not None:
            logits = _cap_top_logit_margin(logits, margin)
        return (logits, *output[1:])
    logits = output / temperature
    if margin is not None:
        logits = _cap_top_logit_margin(logits, margin)
    return logits


def attach_output_temperature_transform(
    model: torch.nn.Module,
    temperature: float,
    margin: float | None = None,
) -> None:
    """Attach a differentiable, label-preserving low-confidence output map."""
    if temperature < 1.0:
        raise ValueError("output_temperature must be at least one.")
    if margin is not None and margin < 0.0:
        raise ValueError("output margin must be non-negative.")
    model._privacy_output_temperature = float(temperature)  # type: ignore[attr-defined]
    model._privacy_output_margin = None if margin is None else float(margin)  # type: ignore[attr-defined]
    if bool(getattr(model, "_supports_native_hamp_output", False)):
        model._hamp_output_temperature = float(temperature)  # type: ignore[attr-defined]
        return
    if not bool(getattr(model, "_privacy_output_hook_attached", False)):
        model.register_forward_hook(_output_temperature_hook)
        model._privacy_output_hook_attached = True  # type: ignore[attr-defined]


def attach_hamp_output_transform(model: torch.nn.Module, temperature: float) -> None:
    """Attach HAMP's differentiable, label-preserving low-confidence output map."""
    if temperature < 1.0:
        raise ValueError("HAMP output_temperature must be at least one.")
    model._hamp_output_temperature = float(temperature)  # type: ignore[attr-defined]
    attach_output_temperature_transform(model, temperature)


class DefenseController:
    """One selected defense spanning client training and upload processing."""

    def __init__(
        self,
        config: dict | None,
        device: torch.device,
        total_users: int,
        num_classes: int,
        total_rounds: int,
        samples_num: list[int] | None = None,
    ):
        self.config = dict(config or {})
        self.name = str(self.config.get("name", "none")).lower()
        if self.name not in SUPPORTED_DEFENSES:
            raise ValueError(f"Unsupported privacy defense: {self.name}")
        self.device = device
        self.total_users = int(total_users)
        self.num_classes = int(num_classes)
        self.total_rounds = int(total_rounds)
        self.samples_num = list(samples_num or [])
        self.seed = int(self.config.get("seed", 42))
        self.www_privacy = (
            WWWPrivacy(self.config, self.total_rounds, self.device, self.seed)
            if self.name == "www" else None
        )
        self.cofedmid = (
            CoFedMID(
                self.config, self.total_users, self.num_classes,
                self.total_rounds, self.seed,
            )
            if self.name == "cofedmid" else None
        )
        self.steps = defaultdict(int)
        self.batch_metrics = defaultdict(float)
        self.batch_counts = defaultdict(int)
        self._generator_calls = defaultdict(int)
        self._selected_by_round: dict[int, list[int]] = {}
        self._www_client_stats: dict[int, dict[str, int | float]] = {}
        self._www_pending_states: dict[
            int,
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        ] = {}
        self.www_analysis_interval = int(
            self.config.get("www_analysis_interval", 1)
        )
        self.www_analysis_timing = str(
            self.config.get("www_analysis_timing", "pre_update")
        ).lower()
        self.www_feature_statistics_enabled = bool(
            self.config.get("www_feature_statistics", False)
        )
        self._www_round_metrics: list[dict[str, int | float]] = []
        self._www_round_samples: list[dict[str, int | float]] = []
        self._www_projres_samples: list[dict[str, int | float]] = []
        self._www_projres_metrics: list[
            dict[str, int | float | None]
        ] = []
        if self.name == "www":
            if self.www_analysis_interval <= 0:
                raise ValueError("www_analysis_interval must be positive.")
            if self.www_analysis_timing not in {"pre_update", "post_round"}:
                raise ValueError(
                    "www_analysis_timing must be pre_update or post_round."
                )
        if self.name == "record_dp":
            adjacency = str(self.config.get("adjacency", "add_remove")).lower()
            sampling = str(self.config.get("sampling", "poisson")).lower()
            if adjacency != "add_remove":
                raise ValueError("Record-DP currently requires add_remove adjacency.")
            if sampling != "poisson":
                raise ValueError("Record-DP currently requires Poisson sampling.")
            if str(self.config.get("accountant", "rdp")).lower() != "rdp":
                raise ValueError("Record-DP currently requires accountant=rdp.")
            if self._record_dp_max_norm() <= 0:
                raise ValueError("Record-DP max_grad_norm must be positive.")
            if not 0 < self._record_dp_delta() < 1:
                raise ValueError("Record-DP delta must be in (0, 1).")
            backend = str(
                self.config.get("grad_sample_backend", "auto")
            ).lower()
            if backend not in GRAD_SAMPLE_BACKENDS:
                raise ValueError(
                    "Record-DP grad_sample_backend must be auto, loop, batched, or vmap."
                )
            chunk_size = self.config.get("microbatch_size", 4)
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("Record-DP microbatch_size must be a positive integer.")
        if self.name == "local_client_dp":
            privacy_unit = str(self.config.get("privacy_unit", "client")).lower()
            adjacency = str(self.config.get("adjacency", "add_remove")).lower()
            sampling = str(
                self.config.get("sampling", "full_participation")
            ).lower()
            if privacy_unit != "client":
                raise ValueError("Local client-DP requires privacy_unit=client.")
            if adjacency != "add_remove":
                raise ValueError(
                    "Local client-DP currently requires add_remove adjacency."
                )
            if sampling != "full_participation":
                raise ValueError(
                    "Local client-DP currently requires full_participation."
                )
            if str(self.config.get("accountant", "rdp")).lower() != "rdp":
                raise ValueError("Local client-DP currently requires accountant=rdp.")
            if self._local_client_dp_max_norm() <= 0:
                raise ValueError(
                    "Local client-DP max_update_norm must be positive."
                )
            if not 0 < self._local_client_dp_delta() < 1:
                raise ValueError("Local client-DP delta must be in (0, 1).")
        self.federated_method = "fedavg"
        self.method_config: dict = {}
        self.additional_private_steps = 0
        self.record_dp_noise_multiplier: float | None = None
        self.record_dp_sample_rates: dict[int, float] = {}
        self.record_dp_planned_steps: dict[int, int] = {}
        self.local_client_dp_noise_multiplier: float | None = None
        self.local_client_dp_planned_steps: dict[int, int] = {}

    def _record_dp_max_norm(self) -> float:
        return float(
            self.config.get(
                "max_grad_norm", self.config.get("dp_max_grad_norm", 1.0)
            )
        )

    def _record_dp_delta(self) -> float:
        return float(
            self.config.get("delta", self.config.get("dp_delta", 1e-5))
        )

    def _record_dp_reproducible(self) -> bool:
        return bool(
            self.config.get(
                "reproducible_noise",
                self.config.get("reproducible_dp_noise", False),
            )
        )

    def _local_client_dp_max_norm(self) -> float:
        return float(self.config.get("max_update_norm", 1.0))

    def _local_client_dp_delta(self) -> float:
        return float(
            self.config.get("delta", self.config.get("dp_delta", 1e-5))
        )

    def _local_client_dp_reproducible(self) -> bool:
        return bool(
            self.config.get(
                "reproducible_noise",
                self.config.get("reproducible_dp_noise", False),
            )
        )

    def configure_record_dp(self, users) -> None:
        """Resolve per-client sampling schedules and a shared noise multiplier."""
        if self.name != "record_dp":
            return
        self.record_dp_sample_rates = {
            int(user.id): float(user.record_dp_sample_rate) for user in users
        }
        self.record_dp_planned_steps = {
            int(user.id): int(self.total_rounds * user.record_dp_steps_per_update)
            + int(self.additional_private_steps)
            for user in users
        }
        schedules = [
            (
                self.record_dp_sample_rates[user_id],
                self.record_dp_planned_steps[user_id],
            )
            for user_id in sorted(self.record_dp_sample_rates)
        ]
        configured_noise = self.config.get(
            "noise_multiplier", self.config.get("dp_noise_multiplier")
        )
        target_epsilon = self.config.get("target_epsilon")
        if target_epsilon is not None and configured_noise not in {None, "auto"}:
            raise ValueError(
                "Record-DP must configure either target_epsilon or a numeric "
                "noise_multiplier, not both."
            )
        if target_epsilon is not None:
            self.record_dp_noise_multiplier = (
                calibrate_poisson_sampled_gaussian_noise(
                    target_epsilon=float(target_epsilon),
                    schedules=schedules,
                    delta=self._record_dp_delta(),
                )
            )
        else:
            if configured_noise in {None, "auto"}:
                raise ValueError(
                    "Record-DP requires target_epsilon or a numeric "
                    "noise_multiplier."
                )
            self.record_dp_noise_multiplier = float(configured_noise)
            if self.record_dp_noise_multiplier <= 0:
                raise ValueError("Record-DP noise_multiplier must be positive.")

    def configure_local_client_dp(self, users) -> None:
        """Calibrate one local Gaussian mechanism per client upload."""
        if self.name != "local_client_dp":
            return
        self.local_client_dp_planned_steps = {
            int(user.id): self.total_rounds + int(self.additional_private_steps)
            for user in users
        }
        configured_noise = self.config.get(
            "noise_multiplier", self.config.get("dp_noise_multiplier")
        )
        target_epsilon = self.config.get("target_epsilon")
        if target_epsilon is not None and configured_noise not in {None, "auto"}:
            raise ValueError(
                "Local client-DP must configure either target_epsilon or a "
                "numeric noise_multiplier, not both."
            )
        if target_epsilon is not None:
            planned_steps = max(self.local_client_dp_planned_steps.values())
            self.local_client_dp_noise_multiplier = calibrate_gaussian_noise(
                target_epsilon=float(target_epsilon),
                steps=planned_steps,
                delta=self._local_client_dp_delta(),
            )
        else:
            if configured_noise in {None, "auto"}:
                raise ValueError(
                    "Local client-DP requires target_epsilon or a numeric "
                    "noise_multiplier."
                )
            self.local_client_dp_noise_multiplier = float(configured_noise)
            if self.local_client_dp_noise_multiplier <= 0:
                raise ValueError(
                    "Local client-DP noise_multiplier must be positive."
                )

    @property
    def enabled(self) -> bool:
        return self.name != "none"

    def _generator(self, client_id: int, round_index: int, offset: int = 0):
        generator_device = (
            self.device if self.device.type in {"cpu", "cuda"} else torch.device("cpu")
        )
        key = (int(client_id), int(round_index), int(offset), str(generator_device))
        call_index = self._generator_calls[key]
        self._generator_calls[key] += 1
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(
            self.seed
            + 1000003 * int(client_id)
            + 1009 * int(round_index)
            + offset
            + 7919 * call_index
        )
        return generator

    def _record(self, key: str, value: float) -> None:
        self.batch_metrics[key] += float(value)
        self.batch_counts[key] += 1

    def validation_loss(self, model: torch.nn.Module, loader) -> float:
        return _mean_loader_loss(model, loader, self.device)

    def prepare_round(self, selected_ids: list[int], round_index: int) -> None:
        self._selected_by_round[int(round_index)] = list(selected_ids)
        if self.cofedmid is not None:
            self.cofedmid.prepare(selected_ids, round_index)

    def train_client(
        self,
        user,
        model: torch.nn.Module,
        round_index: int = 0,
        code_poison: bool = False,
        privacy_probe: bool = False,
    ) -> None:
        if privacy_probe and self.name in {"cofedmid", "www"}:
            raise ValueError(f"{self.name} does not support isolated active client probes.")
        model.to(self.device)
        model.train()
        parameters = _trainable_parameters(model)
        if self.name in {"record_dp", "local_client_dp", "www"}:
            optimizer_name = str(
                self.method_config.get("client_optimizer", "sgd")
            ).lower()
            if optimizer_name == "sgd":
                optimizer = torch.optim.SGD(
                    parameters,
                    lr=user.learning_rate,
                    momentum=float(self.method_config.get("momentum", 0.0)),
                    weight_decay=float(
                        self.method_config.get("weight_decay", 0.0)
                    ),
                )
            elif optimizer_name == "adamw":
                optimizer = torch.optim.AdamW(
                    parameters,
                    lr=user.learning_rate,
                    weight_decay=float(
                        self.method_config.get("weight_decay", 0.01)
                    ),
                )
            else:
                raise ValueError("DP client_optimizer must be sgd or adamw.")
        else:
            optimizer = torch.optim.SGD(parameters, lr=user.learning_rate)
        if user.federated_method == "fedsgd":
            optimizer.register_step_pre_hook(
                lambda _optimizer, _args, _kwargs: (
                    user.capture_protocol_gradients(model)
                )
            )
        if self.name == "www" and not privacy_probe:
            self._www_training(
                user, model, optimizer, round_index, code_poison
            )
        elif self.name == "prompt_dp":
            self._prompt_dp_training(
                user, model, optimizer, parameters, round_index, code_poison
            )
        elif self.name == "record_dp":
            self._record_dp_training(
                user, model, optimizer, parameters, round_index, code_poison
            )
        elif self.name == "local_client_dp":
            self._local_client_dp_training(
                user, model, optimizer, parameters, round_index, code_poison
            )
        elif self.name == "hamp":
            self._hamp_training(user, model, optimizer, round_index, code_poison)
        elif self.name == "soft":
            self._soft_training(user, model, optimizer, round_index, code_poison)
        elif self.name == "cofedmid":
            self._cofedmid_training(user, model, optimizer, round_index, code_poison)
        elif self.name in {"mixup", "sampling", "data_aug", "data_aug_sampling"}:
            self._fedmia_data_training(
                user, model, optimizer, round_index, code_poison
            )
        else:
            self._standard_training(
                user, model, optimizer, round_index, code_poison=code_poison
            )

    def _www_training(
        self,
        user,
        model: torch.nn.Module,
        optimizer,
        round_index: int,
        code_poison: bool,
    ) -> None:
        """Rank every real batch, apply INO clipping, then privatize the upload."""
        reference_states = self._www_pending_states.pop(user.id, None)
        generator = self.www_privacy.generator(user.id, round_index)
        sampling_generator = self.www_privacy.sampling_generator(user.id, round_index)
        for images, labels, local_indices in user.iter_www_local_batches(sampling_generator):
            with measure_stage(self, "train.www_ranking"):
                restore_state = user.get_parameters()
                own_state, other_state = (
                    reference_states if reference_states is not None
                    else (restore_state, restore_state)
                )
                ranking = rank_loss_differences(
                    model=model,
                    batches=[(images, labels)],
                    own_state=own_state,
                    other_state=other_state,
                    restore_state=restore_state,
                    device=self.device,
                    sample_indices=local_indices,
                )
            self._record_www_ranking(
                user=user,
                ranking=ranking,
                ranking_round=round_index,
                source_round=(
                    int(user.www_source_round) if reference_states is not None else -1
                ),
                aggregation_weight=(
                    float(user.www_aggregation_weight)
                    if reference_states is not None else 0.0
                ),
            )
            if reference_states is None:
                weights = torch.ones(labels.numel(), dtype=torch.float64)
                tail = torch.zeros(labels.numel(), dtype=torch.bool)
            else:
                weights, _, tail = ino_weights(
                    ranking.scores, float(self.config["www_tail_fraction"]),
                    float(self.config["www_beta_alpha"]),
                    float(self.config["www_beta_beta"]),
                    expected_batch_size=user.record_dp_expected_batch_size,
                )
            user.www_importance_weights = weights
            user.www_tail_mask = tail
            user.www_effective_clip_norms = weights * float(
                self.config["max_grad_norm"]
            )
            user.www_tail_local_indices = local_indices[tail].clone()
            extra_loss = (
                lambda inputs, targets: self._secret_losses(user, model, inputs, targets)
            ) if code_poison else None
            self.www_privacy.step(
                user, model, optimizer, images.to(self.device), labels.to(self.device),
                weights, self.steps[user.id], generator, extra_loss,
            )
            self.steps[user.id] += 1

    def _record_www_ranking(
        self,
        user,
        ranking,
        ranking_round: int,
        source_round: int,
        aggregation_weight: float,
    ) -> None:
        """Store one sample-aligned WWW ranking and its compact statistics."""
        user.www_ranking_round = int(ranking_round)
        user.www_source_round = int(source_round)
        user.www_aggregation_weight = float(aggregation_weight)
        user.www_own_losses = ranking.own_losses
        user.www_other_losses = ranking.other_losses
        user.www_scores = ranking.scores
        user.www_ranked_positions = ranking.ranked_positions
        user.www_ranked_scores = ranking.scores[ranking.ranked_positions]
        user.www_ranked_labels = ranking.labels[ranking.ranked_positions]
        user.www_local_indices = ranking.sample_indices
        user.www_ranked_local_indices = ranking.sample_indices[
            ranking.ranked_positions
        ]
        self._update_www_score_statistics(user, ranking, ranking_round)
        self._www_client_stats.setdefault(user.id, {}).update(
            {
                "latest_ranking_round": int(ranking_round),
                "latest_ranking_source_round": int(source_round),
                "latest_ranked_samples": int(ranking.scores.numel()),
                "latest_ranking_weight": float(aggregation_weight),
            }
        )
        if not bool(self.config.get("release_private_diagnostics", False)) or not ranking.scores.numel():
            return
        self._record("www_score_mean", float(ranking.scores.mean()))
        self._record("www_score_min", float(ranking.scores.min()))
        self._record("www_score_max", float(ranking.scores.max()))
        self._record(
            "www_positive_score_fraction",
            float((ranking.scores > 0).float().mean()),
        )

    def analyze_www_completed_round(
        self,
        users,
        global_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        aggregation_weights: dict[int, float],
        selected_ids: list[int],
        round_index: int,
    ) -> bool:
        """Analyze the exact FedSGD batches uploaded at a scheduled round."""
        completed_round = int(round_index) + 1
        if (
            self.name != "www"
            or not bool(self.config.get("release_private_diagnostics", False))
            or self.www_analysis_timing != "post_round"
            or completed_round % self.www_analysis_interval != 0
        ):
            return False
        if self.federated_method != "fedsgd":
            raise ValueError(
                "Post-round WWW analysis currently requires one-batch FedSGD."
            )

        for client_id in selected_ids:
            user = users[client_id]
            if user.last_train_batch is None or user.last_train_indices is None:
                raise RuntimeError(
                    f"Client {client_id} has no indexed FedSGD batch for WWW."
                )
            if not user.last_train_batch[1].numel():
                continue
            own_state = updated_states[client_id]
            weight = float(aggregation_weights[client_id])
            other_state = infer_other_clients_state(
                global_state=global_state,
                own_state=own_state,
                own_weight=weight,
            )
            shared_session = getattr(user.model, "use_shared_model", None)
            session = shared_session() if callable(shared_session) else nullcontext()
            with session:
                restore_state = user.get_parameters()
                ranking = rank_loss_differences(
                    model=user.model,
                    batches=[user.last_train_batch],
                    own_state=own_state,
                    other_state=other_state,
                    restore_state=restore_state,
                    device=self.device,
                    sample_indices=user.last_train_indices,
                )
            self._record_www_ranking(
                user=user,
                ranking=ranking,
                ranking_round=round_index,
                source_round=round_index,
                aggregation_weight=weight,
            )
            self._www_round_metrics.append(
                {
                    "communication_round": completed_round,
                    "client_id": int(client_id),
                    "sample_count": int(ranking.scores.numel()),
                    "score_mean": float(ranking.scores.mean()),
                    "score_min": float(ranking.scores.min()),
                    "score_max": float(ranking.scores.max()),
                    "positive_score_fraction": float(
                        (ranking.scores > 0).float().mean()
                    ),
                }
            )
            for position in range(ranking.scores.numel()):
                self._www_round_samples.append(
                    {
                        "communication_round": completed_round,
                        "client_id": int(client_id),
                        "batch_position": int(position),
                        "local_sample_index": int(
                            ranking.sample_indices[position]
                        ),
                        "class_label": int(ranking.labels[position]),
                        "own_loss": float(ranking.own_losses[position]),
                        "other_loss": float(ranking.other_losses[position]),
                        "www_score": float(ranking.scores[position]),
                    }
                )
        return any(row["communication_round"] == completed_round for row in self._www_round_metrics)

    def initialize_www_feature_statistics(self, users) -> None:
        """Compute fixed feature statistics from every complete local dataset."""
        if self.name != "www" or not self.www_feature_statistics_enabled:
            return
        for user in users:
            user.www_feature_seen = None
            user.www_class_feature_counts = None
            user.www_class_feature_means = None
            user.www_within_class_scatter = None
            user.www_within_class_covariance = None
            user.www_within_class_covariance_dof = 0
            for images, labels, local_indices in (
                user.iter_www_statistics_batches()
            ):
                encoded_features = encode_training_batches(
                    model=user.model,
                    batches=[(images, labels)],
                    device=self.device,
                )
                self._update_www_feature_statistics(
                    user=user,
                    encoded_features=encoded_features,
                    labels=labels,
                    sample_indices=local_indices,
                    num_classes=self.num_classes,
                )
            if user.www_class_feature_counts is None:
                raise ValueError(
                    f"Client {user.id} has no samples for WWW feature statistics."
                )
            encoded_samples = int(user.www_class_feature_counts.sum())
            if encoded_samples != int(user.train_samples):
                raise RuntimeError(
                    f"Client {user.id} WWW feature statistics cover "
                    f"{encoded_samples}/{user.train_samples} local samples."
                )
            self._www_client_stats.setdefault(user.id, {}).update(
                {
                    "encoded_feature_samples": encoded_samples,
                    "encoded_feature_classes": int(
                        (user.www_class_feature_counts > 0).sum()
                    ),
                    "encoded_feature_dimension": int(
                        user.www_class_feature_means.shape[1]
                    ),
                    "within_class_covariance_dof": int(
                        user.www_within_class_covariance_dof
                    ),
                }
            )

    @staticmethod
    def _update_www_score_statistics(user, ranking, round_index: int) -> None:
        """Maintain compact per-local-sample statistics across communication rounds."""
        sample_count = int(user.train_samples)
        if user.www_score_count is None:
            user.www_score_count = torch.zeros(sample_count, dtype=torch.long)
            user.www_score_sum = torch.zeros(sample_count, dtype=torch.float64)
            user.www_score_sum_sq = torch.zeros(sample_count, dtype=torch.float64)
            user.www_score_min = torch.full(
                (sample_count,), float("inf"), dtype=torch.float64
            )
            user.www_score_max = torch.full(
                (sample_count,), float("-inf"), dtype=torch.float64
            )
            user.www_score_last = torch.full(
                (sample_count,), float("nan"), dtype=torch.float64
            )
            user.www_score_last_round = torch.full(
                (sample_count,), -1, dtype=torch.long
            )

        indices = ranking.sample_indices.detach().cpu().long()
        scores = ranking.scores.detach().cpu().to(torch.float64)
        if indices.numel() == 0:
            return
        if int(indices.min()) < 0 or int(indices.max()) >= sample_count:
            raise IndexError("WWW local sample index is outside the client dataset.")
        ones = torch.ones(indices.numel(), dtype=torch.long)
        user.www_score_count.index_add_(0, indices, ones)
        user.www_score_sum.index_add_(0, indices, scores)
        user.www_score_sum_sq.index_add_(0, indices, scores.square())
        user.www_score_min.scatter_reduce_(
            0, indices, scores, reduce="amin", include_self=True
        )
        user.www_score_max.scatter_reduce_(
            0, indices, scores, reduce="amax", include_self=True
        )
        if torch.unique(indices).numel() == indices.numel():
            user.www_score_last[indices] = scores
            user.www_score_last_round[indices] = int(round_index)
        else:
            for index, score in zip(indices.tolist(), scores.tolist()):
                user.www_score_last[index] = score
                user.www_score_last_round[index] = int(round_index)

    @staticmethod
    def _update_www_feature_statistics(
        user,
        encoded_features: torch.Tensor,
        labels: torch.Tensor,
        sample_indices: torch.Tensor,
        num_classes: int,
    ) -> None:
        """Update unique-sample class means and pooled within-class covariance."""
        features = encoded_features.detach().cpu().to(torch.float64)
        labels = labels.detach().cpu().long().flatten()
        indices = sample_indices.detach().cpu().long().flatten()
        if features.ndim != 2:
            raise ValueError("WWW encoded features must be a matrix.")
        if features.shape[0] != labels.numel() or labels.numel() != indices.numel():
            raise ValueError(
                "WWW features, labels, and local indices must be sample-aligned."
            )
        if indices.numel() == 0:
            return
        sample_count = int(user.train_samples)
        if int(indices.min()) < 0 or int(indices.max()) >= sample_count:
            raise IndexError("WWW feature local index is outside the client dataset.")
        if int(labels.min()) < 0 or int(labels.max()) >= int(num_classes):
            raise ValueError("WWW feature label is outside the configured classes.")

        dimension = int(features.shape[1])
        if user.www_feature_seen is None:
            user.www_feature_seen = torch.zeros(sample_count, dtype=torch.bool)
            user.www_class_feature_counts = torch.zeros(
                num_classes, dtype=torch.long
            )
            user.www_class_feature_means = torch.zeros(
                (num_classes, dimension), dtype=torch.float64
            )
            user.www_within_class_scatter = torch.zeros(
                (dimension, dimension), dtype=torch.float64
            )
            user.www_within_class_covariance = torch.zeros(
                (dimension, dimension), dtype=torch.float64
            )
        elif user.www_class_feature_means.shape != (num_classes, dimension):
            raise ValueError(
                "WWW encoded feature dimension or class count changed during training."
            )

        new_positions = []
        batch_indices = set()
        for position, local_index in enumerate(indices.tolist()):
            if user.www_feature_seen[local_index] or local_index in batch_indices:
                continue
            batch_indices.add(local_index)
            new_positions.append(position)
        if not new_positions:
            return

        positions = torch.tensor(new_positions, dtype=torch.long)
        new_features = features[positions]
        new_labels = labels[positions]
        new_indices = indices[positions]
        for class_id in torch.unique(new_labels, sorted=True).tolist():
            class_mask = new_labels == int(class_id)
            class_features = new_features[class_mask]
            batch_count = int(class_features.shape[0])
            batch_mean = class_features.mean(dim=0)
            centered = class_features - batch_mean
            batch_scatter = centered.t().matmul(centered)

            previous_count = int(user.www_class_feature_counts[class_id])
            previous_mean = user.www_class_feature_means[class_id]
            combined_count = previous_count + batch_count
            if previous_count == 0:
                combined_mean = batch_mean
                merge_scatter = batch_scatter
            else:
                delta = batch_mean - previous_mean
                combined_mean = previous_mean + delta * (
                    batch_count / combined_count
                )
                correction = torch.outer(delta, delta) * (
                    previous_count * batch_count / combined_count
                )
                merge_scatter = batch_scatter + correction
            user.www_class_feature_counts[class_id] = combined_count
            user.www_class_feature_means[class_id] = combined_mean
            user.www_within_class_scatter.add_(merge_scatter)

        user.www_feature_seen[new_indices] = True
        populated_classes = int((user.www_class_feature_counts > 0).sum())
        total_samples = int(user.www_class_feature_counts.sum())
        degrees_of_freedom = total_samples - populated_classes
        user.www_within_class_covariance_dof = degrees_of_freedom
        if degrees_of_freedom > 0:
            covariance = user.www_within_class_scatter / degrees_of_freedom
            user.www_within_class_covariance.copy_(
                (covariance + covariance.t()) * 0.5
            )
        else:
            user.www_within_class_covariance.zero_()

    def prepare_client_training(
        self,
        user,
        global_state: dict[str, torch.Tensor],
        own_weight: float,
        source_round: int,
        own_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Build the two WWW references immediately before global overwrite."""
        if self.name != "www":
            return
        own_state = user.get_parameters() if own_state is None else own_state
        weight = float(own_weight)
        other_state = infer_other_clients_state(global_state, own_state, weight)
        self._www_pending_states[user.id] = (own_state, other_state)
        user.www_source_round = int(source_round)
        user.www_aggregation_weight = weight

    @staticmethod
    def _secret_losses(user, model, images, labels) -> torch.Tensor:
        synthetic = generate_membership_encoding_samples(
            images,
            mean=float(user.code_poison_config.get("synthetic_mean", 0.0)),
            std=float(user.code_poison_config.get("synthetic_std", 0.1)),
        )
        return float(user.code_poison_config.get("weight", 1.0)) * F.cross_entropy(
            model(synthetic), labels, reduction="none"
        )

    def _standard_training(
        self, user, model, optimizer, round_index: int, code_poison: bool = False
    ) -> None:
        for images, labels in user.iter_local_batches():
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            if code_poison:
                loss = compromised_prompt_loss(
                    model,
                    images,
                    labels,
                    weight=float(user.code_poison_config.get("weight", 1.0)),
                    mean=float(user.code_poison_config.get("synthetic_mean", 0.0)),
                    std=float(user.code_poison_config.get("synthetic_std", 0.1)),
                )
            else:
                loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            self.steps[user.id] += 1

    @staticmethod
    def _sample_beta(
        alpha: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if alpha <= 0:
            raise ValueError("MixUp alpha must be positive.")
        concentration = torch.tensor(alpha, device=device, dtype=dtype)
        left = torch._standard_gamma(concentration, generator=generator)
        right = torch._standard_gamma(concentration, generator=generator)
        return left / (left + right).clamp_min(torch.finfo(dtype).tiny)

    def _augment_images(
        self,
        images: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Tensor-space augmentation safe for preprocessed CLIP inputs."""
        if images.ndim != 4:
            raise ValueError("Data augmentation expects NCHW image tensors.")
        strength = float(self.config.get("data_aug_strength", 0.1))
        flip_probability = float(self.config.get("data_aug_flip_probability", 0.5))
        shortest_side = min(images.shape[-2:])
        maximum_shift = min(
            int(round(strength * shortest_side)),
            max(0, shortest_side - 1),
        )
        augmented = images.clone()
        flip = torch.rand(
            images.shape[0], generator=generator, device=images.device
        ) < flip_probability
        if bool(flip.any()):
            augmented[flip] = torch.flip(augmented[flip], dims=(-1,))
        if maximum_shift > 0:
            padded = F.pad(
                augmented,
                (maximum_shift,) * 4,
                mode="reflect",
            )
            offsets = torch.randint(
                -maximum_shift,
                maximum_shift + 1,
                (images.shape[0], 2),
                generator=generator,
                device=images.device,
            )
            shifted = []
            height, width = images.shape[-2:]
            for sample, (dy, dx) in zip(padded, offsets.tolist()):
                top = maximum_shift - int(dy)
                left = maximum_shift - int(dx)
                shifted.append(sample[:, top : top + height, left : left + width])
            augmented = torch.stack(shifted)
        jitter = float(self.config.get("data_aug_color_jitter", 0.1))
        if jitter > 0:
            mean = augmented.mean(dim=(-2, -1), keepdim=True)
            contrast = 1.0 + (
                2.0
                * torch.rand(
                    (images.shape[0], 1, 1, 1),
                    generator=generator,
                    device=images.device,
                    dtype=images.dtype,
                )
                - 1.0
            ) * jitter
            brightness = (
                2.0
                * torch.rand(
                    (images.shape[0], 1, 1, 1),
                    generator=generator,
                    device=images.device,
                    dtype=images.dtype,
                )
                - 1.0
            ) * jitter
            augmented = (augmented - mean) * contrast + mean + brightness
        return augmented

    def _fedmia_data_training(
        self,
        user,
        model: torch.nn.Module,
        optimizer,
        round_index: int,
        code_poison: bool,
    ) -> None:
        """FedMIA data-replacement baselines adapted to prompt-only training."""
        use_mixup = self.name == "mixup"
        use_sampling = self.name in {"sampling", "data_aug_sampling"}
        use_augmentation = self.name in {"data_aug", "data_aug_sampling"}
        sampling_ratio = float(self.config.get("sampling_ratio", 0.5))
        mixup_alpha = float(self.config.get("mixup_alpha", 1.0))
        for epoch in range(user.local_epochs):
            generator = self._generator(user.id, round_index, 1701 + epoch)
            for images, labels in user.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                original_count = labels.numel()
                if use_sampling and sampling_ratio < 1.0:
                    selected_count = max(
                        1, min(original_count, int(round(sampling_ratio * original_count)))
                    )
                    selected = torch.randperm(
                        original_count,
                        generator=generator,
                        device=labels.device,
                    )[:selected_count]
                    images = images[selected]
                    labels = labels[selected]
                    self._record("sampling_fraction", selected_count / original_count)
                if use_augmentation:
                    images = self._augment_images(images, generator)
                    self._record("data_aug_fraction", 1.0)
                optimizer.zero_grad(set_to_none=True)
                if use_mixup:
                    permutation = torch.randperm(
                        labels.numel(), generator=generator, device=labels.device
                    )
                    coefficient = self._sample_beta(
                        mixup_alpha, generator, images.device, images.dtype
                    )
                    mixed = (
                        coefficient * images
                        + (1.0 - coefficient) * images[permutation]
                    )
                    logits = model(mixed)
                    loss = (
                        coefficient
                        * F.cross_entropy(logits, labels, reduction="none")
                        + (1.0 - coefficient)
                        * F.cross_entropy(
                            logits, labels[permutation], reduction="none"
                        )
                    ).mean()
                    self._record("mixup_lambda", float(coefficient.detach()))
                    defended_images = mixed
                else:
                    defended_images = images
                    loss = F.cross_entropy(model(defended_images), labels)
                if code_poison:
                    loss = loss + self._secret_losses(
                        user, model, defended_images, labels
                    ).mean()
                loss.backward()
                optimizer.step()
                self.steps[user.id] += 1

    def _prompt_dp_training(
        self,
        user,
        model,
        optimizer,
        parameters,
        round_index: int,
        code_poison: bool,
    ) -> None:
        max_norm = float(self.config.get("dp_max_grad_norm", 1.0))
        noise_multiplier = float(self.config.get("dp_noise_multiplier", 1.0))
        if max_norm <= 0 or noise_multiplier <= 0:
            raise ValueError(
                "Prompt-DP clipping and noise multiplier must be positive."
            )
        generator = private_generator(
            self.device,
            bool(self.config.get("reproducible_dp_noise", False)),
            self.seed + 1000003 * user.id + 1009 * round_index + 17,
        )
        for _ in range(user.local_epochs):
            for images, labels in user.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                losses = F.cross_entropy(model(images), labels, reduction="none")
                if code_poison:
                    losses = losses + self._secret_losses(user, model, images, labels)
                sample_gradients = _per_sample_gradients(losses, parameters)
                norms = sample_gradients.norm(dim=1).clamp_min(1e-12)
                factors = (max_norm / norms).clamp(max=1.0)
                clipped = sample_gradients * factors.unsqueeze(1)
                gradient = clipped.mean(dim=0)
                if noise_multiplier > 0:
                    noise = torch.randn(
                        gradient.shape,
                        generator=generator,
                        device=gradient.device,
                        dtype=gradient.dtype,
                    )
                    gradient = gradient + noise * (
                        noise_multiplier * max_norm / labels.numel()
                    )
                _assign_flat_gradient(parameters, gradient)
                optimizer.step()
                self.steps[user.id] += 1
                self._record("dp_clip_fraction", float((factors < 1).float().mean()))

    def _record_dp_training(
        self,
        user,
        model,
        optimizer,
        parameters,
        round_index: int,
        code_poison: bool,
    ) -> None:
        """Run protocol-preserving, client-side record-level DP-SGD."""
        if self.record_dp_noise_multiplier is None:
            raise RuntimeError("Record-DP must be configured before training.")
        max_norm = self._record_dp_max_norm()
        expected_batch_size = int(user.record_dp_expected_batch_size)
        if expected_batch_size <= 0:
            raise ValueError("Record-DP expected batch size must be positive.")
        reproducible = self._record_dp_reproducible()
        sampling_generator = private_generator(
            torch.device("cpu"),
            reproducible,
            self.seed + 1000003 * user.id + 1009 * round_index + 17011,
        )
        noise_generator = private_generator(
            self.device,
            reproducible,
            self.seed + 1000003 * user.id + 1009 * round_index + 29009,
        )
        configured_backend = str(
            self.config.get("grad_sample_backend", "auto")
        ).lower()
        backend = resolve_grad_sample_backend(model, configured_backend)
        microbatch_size = int(self.config.get("microbatch_size", 4))

        for images, labels in user.iter_record_dp_batches(sampling_generator):
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            with measure_stage(self, "train.record_gradients"):
                clipped_sum = [torch.zeros_like(parameter) for parameter in parameters]
                all_factors = []
                for start in range(0, labels.numel(), microbatch_size):
                    stop = min(labels.numel(), start + microbatch_size)
                    batch_images = images[start:stop]
                    batch_labels = labels[start:stop]
                    if backend == "vmap" and not code_poison:
                        partial_sum, factors = _vmap_clipped_gradient_sum(
                            model,
                            batch_images,
                            batch_labels,
                            max_norm,
                        )
                    else:
                        losses = F.cross_entropy(
                            model(batch_images), batch_labels, reduction="none"
                        )
                        if code_poison:
                            losses = losses + self._secret_losses(
                                user, model, batch_images, batch_labels
                            )
                        if backend == "batched" and not code_poison:
                            partial_sum, factors = clipped_sum_from_losses(losses, parameters, max_norm)
                        else:
                            partial_sum, factors = _loop_clipped_gradient_sum(losses, parameters, max_norm)
                    with torch.no_grad():
                        for destination, partial in zip(clipped_sum, partial_sum):
                            destination.add_(partial)
                    all_factors.append(factors)

            with measure_stage(self, "train.noise_and_step"):
                with torch.no_grad():
                    for parameter, gradient_sum in zip(parameters, clipped_sum):
                        noise = torch.randn(
                            parameter.shape,
                            generator=noise_generator,
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                        parameter.grad = (
                            gradient_sum
                            + noise
                            * (self.record_dp_noise_multiplier * max_norm)
                        ) / expected_batch_size
                optimizer.step()
            self.steps[user.id] += 1
            if bool(self.config.get("release_private_diagnostics", False)):
                self._record("record_dp_batch_size", float(labels.numel()))
                self._record("record_dp_empty_batch", float(labels.numel() == 0))
            if all_factors and bool(
                self.config.get("release_private_diagnostics", False)
            ):
                factors = torch.cat(all_factors)
                self._record(
                    "record_dp_clip_fraction",
                    float((factors < 1).float().mean()),
                )

    def _local_client_dp_training(
        self,
        user,
        model,
        optimizer,
        parameters,
        round_index: int,
        code_poison: bool,
    ) -> None:
        """Clip and locally privatize the complete one-batch FedSGD upload."""
        if self.federated_method != "fedsgd":
            raise ValueError("Local client-DP currently requires one-batch FedSGD.")
        if self.local_client_dp_noise_multiplier is None:
            raise RuntimeError("Local client-DP must be configured before training.")
        noise_generator = private_generator(
            self.device,
            self._local_client_dp_reproducible(),
            self.seed + 1000003 * user.id + 1009 * round_index + 41011,
        )
        batches = list(user.iter_local_batches())
        if len(batches) != 1:
            raise RuntimeError(
                "Local client-DP requires exactly one client message per round."
            )
        images, labels = batches[0]
        images = images.to(self.device)
        labels = labels.to(self.device)
        optimizer.zero_grad(set_to_none=True)
        if code_poison:
            loss = compromised_prompt_loss(
                model,
                images,
                labels,
                weight=float(user.code_poison_config.get("weight", 1.0)),
                mean=float(user.code_poison_config.get("synthetic_mean", 0.0)),
                std=float(user.code_poison_config.get("synthetic_std", 0.1)),
            )
        else:
            loss = F.cross_entropy(model(images), labels)
        loss.backward()
        raw_norm, clip_factor = _clip_joint_gradient_and_add_noise(
            parameters,
            max_norm=self._local_client_dp_max_norm(),
            noise_multiplier=self.local_client_dp_noise_multiplier,
            generator=noise_generator,
        )
        optimizer.step()
        self.steps[user.id] += 1
        if bool(self.config.get("release_private_diagnostics", False)):
            self._record("local_client_dp_raw_update_norm", raw_norm)
            self._record("local_client_dp_clip_fraction", float(clip_factor < 1.0))

    def _hamp_training(
        self, user, model, optimizer, round_index: int, code_poison: bool
    ) -> None:
        true_probability = float(self.config.get("hamp_true_probability", 0.6))
        entropy_weight = float(self.config.get("hamp_entropy_weight", 0.05))
        for _ in range(user.local_epochs):
            for images, labels in user.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                targets = _soft_targets(labels, logits.shape[1], true_probability)
                soft_loss = _cross_entropy_with_soft_targets(logits, targets).mean()
                probabilities = torch.softmax(logits, dim=1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
                    dim=1
                )
                loss = soft_loss - entropy_weight * entropy.mean()
                if code_poison:
                    synthetic = generate_membership_encoding_samples(
                        images,
                        mean=float(user.code_poison_config.get("synthetic_mean", 0.0)),
                        std=float(user.code_poison_config.get("synthetic_std", 0.1)),
                    )
                    synthetic_logits = model(synthetic)
                    synthetic_targets = _soft_targets(
                        labels, synthetic_logits.shape[1], true_probability
                    )
                    synthetic_loss = _cross_entropy_with_soft_targets(
                        synthetic_logits, synthetic_targets
                    ).mean()
                    synthetic_probability = torch.softmax(synthetic_logits, dim=1)
                    synthetic_entropy = (
                        -(
                            synthetic_probability
                            * synthetic_probability.clamp_min(1e-12).log()
                        )
                        .sum(dim=1)
                        .mean()
                    )
                    loss = loss + float(user.code_poison_config.get("weight", 1.0)) * (
                        synthetic_loss - entropy_weight * synthetic_entropy
                    )
                loss.backward()
                optimizer.step()
                self.steps[user.id] += 1
                self._record("hamp_entropy", float(entropy.detach().mean()))

    def _soft_training(
        self, user, model, optimizer, round_index: int, code_poison: bool
    ) -> None:
        strength = float(self.config.get("soft_obfuscation_strength", 0.5))
        strength = max(0.0, min(1.0, strength))
        noise_std = float(self.config.get("soft_noise_std", 0.05))
        # SOFT phase 1 is a warm-up pass. Later global rounds perform iterative
        # selection against a held-out validation-loss threshold.
        for epoch in range(user.local_epochs):
            warmup = round_index == 0
            threshold = (
                math.inf
                if warmup
                else _mean_loader_loss(model, user.testloader, self.device)
            )
            generator = self._generator(user.id, round_index, 101 + epoch)
            for images, labels in user.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    original_losses = F.cross_entropy(
                        model(images), labels, reduction="none"
                    )
                    influential = original_losses < threshold
                    if warmup:
                        influential.zero_()
                transformed = torch.flip(images, dims=(-1,))
                if noise_std > 0:
                    transformed = (
                        transformed
                        + torch.randn(
                            transformed.shape,
                            generator=generator,
                            device=transformed.device,
                            dtype=transformed.dtype,
                        )
                        * noise_std
                    )
                mask = influential.view(-1, *([1] * (images.ndim - 1)))
                obfuscated = strength * transformed + (1.0 - strength) * images
                defended_images = torch.where(mask, obfuscated, images)
                loss = F.cross_entropy(model(defended_images), labels)
                if code_poison:
                    loss = (
                        loss
                        + self._secret_losses(
                            user, model, defended_images, labels
                        ).mean()
                    )
                loss.backward()
                optimizer.step()
                self.steps[user.id] += 1
                self._record(
                    "soft_selected_fraction", float(influential.float().mean())
                )


    def _cofedmid_classes(self, client_id: int, round_index: int) -> set[int]:
        if self.cofedmid.round_index != round_index:
            self.cofedmid.prepare(
                self._selected_by_round.get(round_index, list(range(self.total_users))),
                round_index,
            )
        return self.cofedmid.assignments[client_id]

    def _cofedmid_training(
        self, user, model, optimizer, round_index: int, code_poison: bool
    ) -> None:
        if user.id not in self.cofedmid.clients:
            self._standard_training(user, model, optimizer, round_index, code_poison)
            return
        if code_poison:
            raise ValueError("CoFedMID does not implement active code-poison training.")

        def record_step():
            self.steps[user.id] += 1

        self.cofedmid.train(user, model, optimizer, round_index, record_step)

    def after_local_training(
        self,
        users: list,
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        round_index: int,
        client_gradients: dict | None = None,
        aggregation_weights: dict | None = None,
        learning_rate: float | None = None,
    ) -> None:
        if self.name == "mist":
            self._mist_refinement(users, updated_states, selected_ids, round_index)
        elif self.name == "cofedmid":
            if self.federated_method == "fedsgd":
                if client_gradients is None or aggregation_weights is None:
                    raise ValueError("CoFedMID FedSGD requires uploaded gradients and actual weights.")
                self.cofedmid.perturb(
                    client_gradients, aggregation_weights, round_index,
                    learning_rate=learning_rate,
                )
                for user_id in selected_ids:
                    users[user_id].last_update_gradients = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in client_gradients[user_id].items()
                    }
            else:
                self._cofedmid_perturb(
                    updated_states, selected_ids, round_index, aggregation_weights
                )
        elif self.name == "perturb":
            self._fedmia_perturb_updates(
                base_state, updated_states, selected_ids, round_index
            )
        elif self.name == "sparse":
            self._fedmia_sparse_updates(base_state, updated_states, selected_ids)

    @staticmethod
    def _flatten_client_delta(
        base_state: dict[str, torch.Tensor],
        state: dict[str, torch.Tensor],
    ) -> tuple[list[str], torch.Tensor]:
        names = [name for name in state if name in base_state]
        if not names:
            raise ValueError("Update defense found no trainable prompt tensors.")
        flat = torch.cat(
            [
                (state[name].detach() - base_state[name].to(state[name].device)).reshape(
                    -1
                )
                for name in names
            ]
        )
        return names, flat

    @staticmethod
    def _assign_client_delta(
        base_state: dict[str, torch.Tensor],
        state: dict[str, torch.Tensor],
        names: list[str],
        flat: torch.Tensor,
    ) -> None:
        offset = 0
        for name in names:
            tensor = state[name]
            count = tensor.numel()
            delta = flat[offset : offset + count].view_as(tensor)
            state[name] = base_state[name].to(tensor.device) + delta
            offset += count
        if offset != flat.numel():
            raise ValueError("Defended update does not match prompt tensor sizes.")

    def _fedmia_perturb_updates(
        self,
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        round_index: int,
    ) -> None:
        """Client-level clipping and Gaussian perturbation from FedMIA."""
        clip_norm = float(self.config.get("perturb_clip_norm", 1.0))
        noise_std = float(self.config.get("perturb_noise_std", 0.05))
        for position, user_id in enumerate(selected_ids):
            state = updated_states[user_id]
            names, flat = self._flatten_client_delta(base_state, state)
            norm = flat.norm().clamp_min(1e-12)
            flat = flat * min(1.0, clip_norm / float(norm))
            generator = self._generator(user_id, round_index, 1801 + position)
            if noise_std > 0:
                flat = flat + torch.randn(
                    flat.shape,
                    generator=generator,
                    device=flat.device,
                    dtype=flat.dtype,
                ) * noise_std
            self._assign_client_delta(base_state, state, names, flat)
            self._record("perturb_preclip_norm", float(norm))
            self._record("perturb_upload_norm", float(flat.norm()))

    def _fedmia_sparse_updates(
        self,
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
    ) -> None:
        """Zero the configured fraction of smallest prompt-update elements."""
        ratio = float(self.config.get("sparse_ratio", 0.9))
        for user_id in selected_ids:
            state = updated_states[user_id]
            names, flat = self._flatten_client_delta(base_state, state)
            keep = max(1, int(math.ceil((1.0 - ratio) * flat.numel())))
            indices = flat.abs().topk(keep, sorted=False).indices
            sparse = torch.zeros_like(flat)
            sparse[indices] = flat[indices]
            self._assign_client_delta(base_state, state, names, sparse)
            self._record(
                "sparse_zero_fraction",
                float((sparse == 0).float().mean()),
            )


    def after_probe_training(
        self,
        model: torch.nn.Module,
        client_id: int,
        round_index: int = 0,
    ) -> None:
        """Single-client active probes cannot simulate coalition cancellation."""
        if self.name == "cofedmid":
            raise ValueError("CoFedMID does not support isolated active client probes.")

    def _mist_refinement(
        self,
        users: list,
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        round_index: int,
    ) -> None:
        if len(selected_ids) < 2:
            raise ValueError("MIST requires at least two selected client submodels.")
        steps = int(self.config.get("mist_cross_steps", 1))
        weight = float(self.config.get("mist_cross_weight", 1.0))
        phase_one_models = {}
        for user_id in selected_ids:
            snapshot = copy.deepcopy(users[user_id].model).to(self.device)
            snapshot.load_state_dict(updated_states[user_id], strict=False)
            snapshot.eval()
            for parameter in snapshot.parameters():
                parameter.requires_grad_(False)
            phase_one_models[user_id] = snapshot

        for user_id in selected_ids:
            user = users[user_id]
            model = user.model
            model.load_state_dict(updated_states[user_id], strict=False)
            model.train()
            optimizer = torch.optim.SGD(
                _trainable_parameters(model), lr=user.learning_rate
            )
            iterator = iter(user.trainloader)
            for _ in range(steps):
                try:
                    images, labels = next(iterator)
                except StopIteration:
                    iterator = iter(user.trainloader)
                    images, labels = next(iterator)
                images = images.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    peer_confidence = []
                    for peer_id, peer_model in phase_one_models.items():
                        if peer_id == user_id:
                            continue
                        peer_probability = torch.softmax(peer_model(images), dim=1)
                        peer_confidence.append(
                            peer_probability.gather(1, labels.view(-1, 1)).squeeze(1)
                        )
                    counterfactual = torch.stack(peer_confidence).mean(dim=0)
                optimizer.zero_grad(set_to_none=True)
                own_probability = torch.softmax(model(images), dim=1)
                own_confidence = own_probability.gather(1, labels.view(-1, 1)).squeeze(
                    1
                )
                cross_difference = (own_confidence - counterfactual).abs().mean()
                (weight * cross_difference).backward()
                optimizer.step()
                self._record("mist_cross_difference", float(cross_difference.detach()))
            updated_states[user_id] = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
                if name in updated_states[user_id]
            }

    def _cofedmid_perturb(
        self, updated_states, selected_ids, round_index, aggregation_weights=None
    ) -> None:
        if aggregation_weights is None:
            total = sum(self.samples_num[i] for i in selected_ids)
            aggregation_weights = {i: self.samples_num[i] / total for i in selected_ids}
        self.cofedmid.perturb(updated_states, aggregation_weights, round_index)

    def conservative_dp_epsilon(self) -> float | None:
        if self.www_privacy is not None:
            return self.www_privacy.summary(self.steps)["epsilon_upper_bound"]
        if self.name == "record_dp":
            epsilons = self.record_dp_epsilons()
            return max(epsilons.values()) if epsilons else None
        if self.name == "local_client_dp":
            epsilons = self.local_client_dp_epsilons()
            return max(epsilons.values()) if epsilons else None
        if self.name != "prompt_dp" or not self.steps:
            return None
        noise = float(self.config.get("dp_noise_multiplier", 1.0))
        mechanisms = 1
        delta = float(
            self.method_config.get("delta", self.config.get("dp_delta", 1e-5))
        )
        if noise <= 0:
            return math.inf
        steps = max(self.steps.values()) + int(self.additional_private_steps)
        candidates = []
        for order in (2, 3, 4, 8, 16, 32, 64):
            rdp = mechanisms * steps * order / (2.0 * noise * noise)
            candidates.append(rdp + math.log(1.0 / delta) / (order - 1))
        return min(candidates)

    def record_dp_epsilons(self) -> dict[int, float]:
        if self.name != "record_dp" or self.record_dp_noise_multiplier is None:
            return {}
        delta = self._record_dp_delta()
        return {
            client_id: poisson_sampled_gaussian_epsilon(
                noise_multiplier=self.record_dp_noise_multiplier,
                sample_rate=sample_rate,
                steps=int(self.steps.get(client_id, 0))
                + int(self.additional_private_steps),
                delta=delta,
            )
            for client_id, sample_rate in sorted(
                self.record_dp_sample_rates.items()
            )
        }

    def local_client_dp_epsilons(self) -> dict[int, float]:
        if (
            self.name != "local_client_dp"
            or self.local_client_dp_noise_multiplier is None
        ):
            return {}
        return {
            client_id: gaussian_rdp_epsilon(
                noise_multiplier=self.local_client_dp_noise_multiplier,
                steps=int(self.steps.get(client_id, 0))
                + int(self.additional_private_steps),
                delta=self._local_client_dp_delta(),
            )
            for client_id in sorted(self.local_client_dp_planned_steps)
        }

    def summary(self) -> dict:
        averaged = {
            key: self.batch_metrics[key] / max(1, self.batch_counts[key])
            for key in sorted(self.batch_metrics)
        }
        summary = {
            "defense": self.name,
            "enabled": self.enabled,
            "steps_per_client": {str(key): value for key, value in self.steps.items()},
            "metrics": averaged,
        }
        if self.cofedmid is not None:
            summary["cofedmid"] = self.cofedmid.summary()
        if self.name == "www":
            periodic_post_round = self.www_analysis_timing == "post_round"
            completed_rounds = (
                sorted(
                    {
                        int(row["communication_round"])
                        for row in self._www_round_metrics
                    }
                )
                if periodic_post_round
                else None
            )
            summary["www"] = {
                "score": "L(x; theta_-k) - L(x; theta_k)",
                "ranking": "ascending",
                "training_action": "ino_weighted_per_sample_clipping_and_gaussian_noise",
                "tail_fraction": self.config["www_tail_fraction"],
                "tail_count": "min(actual_batch_size, ceil(expected_batch_size * tail_fraction))",
                "tail_length": "fixed_per_client_expected_batch_size",
                "tail_selection": "largest_loss_differences",
                "initial_clip_norm": self.config["max_grad_norm"],
                "importance_function": "flipped_beta_cdf_interval_average",
                "beta_alpha": self.config["www_beta_alpha"],
                "beta_beta": self.config["www_beta_beta"],
                "defense_interval": 1,
                "reference": "previous_round_defended_local_and_other_client_models",
                "missing_reference": "uniform_clipping_and_noise",
                "analysis_timing": self.www_analysis_timing,
                "analysis_interval": self.www_analysis_interval,
                "scheduled_rounds": (
                    list(
                        range(
                            self.www_analysis_interval,
                            self.total_rounds + 1,
                            self.www_analysis_interval,
                        )
                    )
                    if periodic_post_round
                    else None
                ),
                "completed_rounds": completed_rounds,
                "round_metric_rows": len(self._www_round_metrics),
                "round_sample_rows": len(self._www_round_samples),
                "projres_alignment_sample_rows": len(
                    self._www_projres_samples
                ),
                "projres_alignment_relationship_rows": len(
                    self._www_projres_metrics
                ),
                "feature_statistics": {
                    "enabled": self.www_feature_statistics_enabled,
                    "feature_space": (
                        "frozen_clip_image_encoder_output"
                        if self.www_feature_statistics_enabled
                        else None
                    ),
                    "sample_weighting": "full_local_training_set_once",
                    "computation_stage": "before_federated_training",
                    "class_means": "per_client_per_class",
                    "within_class_covariance": "per_client_pooled_unbiased",
                },
                "client_state": {
                    str(client_id): values
                    for client_id, values in sorted(
                        self._www_client_stats.items()
                    )
                },
            }
        epsilon = self.conservative_dp_epsilon()
        if epsilon is not None:
            if self.www_privacy is not None:
                summary["privacy_accounting"] = self.www_privacy.summary(self.steps)
            elif self.name == "record_dp":
                epsilons = self.record_dp_epsilons()
                summary["privacy_accounting"] = {
                    "privacy_unit": "record",
                    "adjacency": "add_remove",
                    "accountant": "poisson_sampled_gaussian_rdp",
                    "sampling": "poisson",
                    "epsilon_upper_bound": epsilon,
                    "target_epsilon": self.config.get("target_epsilon"),
                    "delta": self._record_dp_delta(),
                    "noise_multiplier": self.record_dp_noise_multiplier,
                    "max_grad_norm": self._record_dp_max_norm(),
                    "normalization": "fixed_expected_batch_size",
                    "per_client": {
                        str(client_id): {
                            "sample_count": int(self.samples_num[client_id]),
                            "sample_rate": self.record_dp_sample_rates[client_id],
                            "actual_steps": int(self.steps.get(client_id, 0)),
                            "accounted_additional_private_steps": int(
                                self.additional_private_steps
                            ),
                            "planned_steps": self.record_dp_planned_steps[client_id],
                            "epsilon": epsilons[client_id],
                        }
                        for client_id in sorted(epsilons)
                    },
                    "federated_method": self.federated_method,
                    "client_upload_is_private": True,
                    "formal_dp_enabled": not self._record_dp_reproducible()
                    and not bool(
                        self.config.get("release_private_diagnostics", False)
                    ),
                    "private_diagnostics_released": bool(
                        self.config.get("release_private_diagnostics", False)
                    ),
                    "note": (
                        "Per-client sampled-Gaussian RDP composition; disjoint "
                        "client datasets compose in parallel, so the released "
                        "record-level epsilon is the client maximum."
                    ),
                }
            elif self.name == "local_client_dp":
                epsilons = self.local_client_dp_epsilons()
                summary["privacy_accounting"] = {
                    "privacy_unit": "client",
                    "adjacency": "add_remove",
                    "accountant": "gaussian_rdp",
                    "sampling": "full_participation",
                    "epsilon_upper_bound": epsilon,
                    "target_epsilon": self.config.get("target_epsilon"),
                    "delta": self._local_client_dp_delta(),
                    "noise_multiplier": self.local_client_dp_noise_multiplier,
                    "max_update_norm": self._local_client_dp_max_norm(),
                    "noise_std_per_upload_coordinate": (
                        self.local_client_dp_noise_multiplier
                        * self._local_client_dp_max_norm()
                    ),
                    "clipping_scope": "joint_trainable_client_gradient",
                    "per_client": {
                        str(client_id): {
                            "sample_count": int(self.samples_num[client_id]),
                            "sample_rate": 1.0,
                            "actual_steps": int(self.steps.get(client_id, 0)),
                            "accounted_additional_private_steps": int(
                                self.additional_private_steps
                            ),
                            "planned_steps": self.local_client_dp_planned_steps[
                                client_id
                            ],
                            "epsilon": epsilons[client_id],
                        }
                        for client_id in sorted(epsilons)
                    },
                    "federated_method": self.federated_method,
                    "client_upload_is_private": True,
                    "formal_dp_enabled": not self._local_client_dp_reproducible()
                    and not bool(
                        self.config.get("release_private_diagnostics", False)
                    ),
                    "private_diagnostics_released": bool(
                        self.config.get("release_private_diagnostics", False)
                    ),
                    "participation_metadata_is_private": False,
                    "note": (
                        "Each fixed client slot clips its complete one-batch "
                        "gradient and adds Gaussian noise before upload. Disjoint "
                        "clients compose in parallel, so the released client-level "
                        "epsilon is the client maximum. The guarantee hides a "
                        "client's data contribution, not network participation."
                    ),
                }
            else:
                summary["privacy_accounting"] = {
                    "epsilon_upper_bound": epsilon,
                    "delta": float(
                        self.method_config.get(
                            "delta", self.config.get("dp_delta", 1e-5)
                        )
                    ),
                    "note": "Conservative full-participation Gaussian composition; no subsampling amplification claimed.",
                    "federated_method": self.federated_method,
                    "formal_dp_enabled": not bool(
                        self.config.get("reproducible_dp_noise", False)
                    )
                    and not bool(
                        self.method_config.get("reproducible_dp_noise", False)
                    ),
                }
        return summary

    def save_summary(self, results_dir: str) -> dict:
        summary = self.summary()
        if self.cofedmid is not None:
            self.cofedmid.save(results_dir)
        if self.name == "www" and self._www_round_metrics:
            self.save_www_round_metrics(results_dir)
            summary["www"]["round_metrics_artifact"] = (
                "www_round_metrics.csv"
            )
            summary["www"]["round_samples_artifact"] = (
                "www_round_samples.csv"
            )
            summary["www"]["round_series_artifact"] = "www_series.json"
        if self.name == "www" and self._www_projres_samples:
            relationship_dir = os.path.join(results_dir, "privacy_audit")
            self._save_www_projres_relationship(relationship_dir)
            summary["www"]["projres_alignment_artifacts"] = {
                "samples": "privacy_audit/www_projres_samples.csv",
                "relationships": (
                    "privacy_audit/www_projres_relationship.csv"
                ),
                "summary": "privacy_audit/www_projres_relationship.json",
            }
        with measure_stage(self, "outputs.write"), open(
            os.path.join(results_dir, "defense_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2, allow_nan=False)
        return summary

    def save_www_round_metrics(self, results_dir: str) -> None:
        """Persist completed periodic WWW rows so long runs are resumable."""
        if self.name != "www" or not self._www_round_metrics:
            return
        os.makedirs(results_dir, exist_ok=True)
        metric_path = os.path.join(results_dir, "www_round_metrics.csv")
        with measure_stage(self, "outputs.write"), open(metric_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(self._www_round_metrics[0]),
            )
            writer.writeheader()
            writer.writerows(self._www_round_metrics)
        sample_path = os.path.join(results_dir, "www_round_samples.csv")
        with measure_stage(self, "outputs.write"), open(sample_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(self._www_round_samples[0]),
            )
            writer.writeheader()
            writer.writerows(self._www_round_samples)
        completed_rounds = sorted(
            {
                int(row["communication_round"])
                for row in self._www_round_metrics
            }
        )
        with measure_stage(self, "outputs.write"), open(
            os.path.join(results_dir, "www_series.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "experiment": "periodic_post_round_www",
                    "analysis_interval": self.www_analysis_interval,
                    "scheduled_rounds": list(
                        range(
                            self.www_analysis_interval,
                            self.total_rounds + 1,
                            self.www_analysis_interval,
                        )
                    ),
                    "completed_rounds": completed_rounds,
                    "metric_rows": len(self._www_round_metrics),
                    "sample_rows": len(self._www_round_samples),
                    "round_metrics": os.path.basename(metric_path),
                    "round_samples": os.path.basename(sample_path),
                },
                file,
                indent=2,
                allow_nan=False,
            )

    def record_www_projres_relationship(
        self,
        projres_payload: dict,
        output_dir: str,
        round_index: int,
    ) -> bool:
        """Strictly join one periodic ProjRes result to its WWW batch rows."""
        if self.name != "www" or self.www_analysis_timing != "post_round":
            return False
        completed_round = int(round_index) + 1
        payload_round = int(projres_payload.get("communication_round", -1))
        if payload_round != completed_round:
            raise ValueError(
                "WWW and ProjRes communication rounds do not match: "
                f"{completed_round} != {payload_round}."
            )
        if "result" in projres_payload:
            client_results = [projres_payload["result"]]
        else:
            client_results = list(projres_payload.get("per_client", []))
        if not client_results:
            raise ValueError("ProjRes payload contains no client results.")

        current_www = {
            (
                int(row["client_id"]),
                int(row["batch_position"]),
            ): row
            for row in self._www_round_samples
            if int(row["communication_round"]) == completed_round
        }
        if not current_www:
            raise ValueError(
                f"No WWW sample rows exist for communication round {completed_round}."
            )

        new_rows = []
        new_metrics = []
        top_fraction = float(
            self.config.get("www_validation_top_fraction", 0.2)
        )
        for result in client_results:
            client_id = int(result["client_id"])
            member_count = int(result["dimensions"]["member_candidate_count"])
            controls = dict(result.get("candidate_controls", {}))
            batch_positions = controls.get("member_batch_positions")
            local_indices = controls.get("member_local_indices")
            member_labels = controls.get("member_labels")
            if batch_positions is None or local_indices is None:
                raise ValueError(
                    "ProjRes result lacks explicit member batch/local indices."
                )
            if not (
                len(batch_positions)
                == len(local_indices)
                == len(member_labels)
                == member_count
            ):
                raise ValueError("ProjRes member alignment fields are inconsistent.")
            raw = dict(result["raw"])
            for key in ("labels", "scores", "l1_residuals"):
                if len(raw[key]) < member_count:
                    raise ValueError(
                        f"ProjRes raw field {key!r} is shorter than its members."
                    )
            if not (
                len(raw["labels"])
                == len(raw["scores"])
                == len(raw["l1_residuals"])
            ):
                raise ValueError("ProjRes raw candidate fields are inconsistent.")
            raw_labels = torch.tensor(raw["labels"], dtype=torch.long)
            all_projres_scores = torch.tensor(
                raw["scores"], dtype=torch.float64
            )
            nonmember_scores = all_projres_scores[raw_labels == 0]
            if nonmember_scores.numel() == 0:
                raise ValueError(
                    "WWW-ProjRes low-FPR analysis requires nonmember scores."
                )
            if int((raw_labels == 1).sum()) != member_count:
                raise ValueError(
                    "ProjRes member labels do not match member_candidate_count."
                )

            client_rows = []
            for offset in range(member_count):
                batch_position = int(batch_positions[offset])
                local_sample_index = int(local_indices[offset])
                www = current_www.get((client_id, batch_position))
                if www is None:
                    raise ValueError(
                        "Missing WWW row for ProjRes member "
                        f"round={completed_round}, client={client_id}, "
                        f"batch_position={batch_position}."
                    )
                if int(www["local_sample_index"]) != local_sample_index:
                    raise ValueError(
                        "WWW and ProjRes local sample indices do not match."
                    )
                class_label = int(member_labels[offset])
                if int(www["class_label"]) != class_label:
                    raise ValueError("WWW and ProjRes class labels do not match.")
                if int(raw["labels"][offset]) != 1:
                    raise ValueError("ProjRes aligned batch entry is not a member.")
                row = {
                    "communication_round": completed_round,
                    "client_id": client_id,
                    "batch_position": batch_position,
                    "local_sample_index": local_sample_index,
                    "class_label": class_label,
                    "www_score": float(www["www_score"]),
                    "www_own_loss": float(www["own_loss"]),
                    "www_other_loss": float(www["other_loss"]),
                    "projres_score": float(raw["scores"][offset]),
                    "projres_l1_residual": float(
                        raw["l1_residuals"][offset]
                    ),
                }
                client_rows.append(row)
                new_rows.append(row)

            www_scores = torch.tensor(
                [row["www_score"] for row in client_rows],
                dtype=torch.float64,
            )
            projres_scores = torch.tensor(
                [row["projres_score"] for row in client_rows],
                dtype=torch.float64,
            )
            negative_residuals = -torch.tensor(
                [row["projres_l1_residual"] for row in client_rows],
                dtype=torch.float64,
            )
            class_labels = torch.tensor(
                [row["class_label"] for row in client_rows],
                dtype=torch.long,
            )
            top, bottom, top_count = _top_bottom_masks(
                www_scores, top_fraction
            )
            projres_top = torch.zeros(member_count, dtype=torch.bool)
            projres_order = torch.argsort(
                projres_scores, descending=True, stable=True
            )
            projres_top[projres_order[:top_count]] = True
            overlap_count = int((top & projres_top).sum())
            expected_overlap = top_count * top_count / max(member_count, 1)
            adjusted, macro, adjusted_classes = _class_adjusted_spearman(
                www_scores,
                projres_scores,
                class_labels,
            )
            top_score = _safe_mean(projres_scores[top])
            bottom_score = _safe_mean(projres_scores[bottom])
            metric_row = {
                "communication_round": completed_round,
                "client_id": client_id,
                "aligned_member_samples": member_count,
                "projres_nonmember_samples": int(nonmember_scores.numel()),
                "top_fraction": top_fraction,
                "top_count": top_count,
                "pearson_www_projres_score": _pearson(
                    www_scores, projres_scores
                ),
                "spearman_www_projres_score": _spearman(
                    www_scores, projres_scores
                ),
                "class_adjusted_spearman": adjusted,
                "class_macro_spearman": macro,
                "class_adjusted_classes": adjusted_classes,
                "pearson_www_negative_l1_residual": _pearson(
                    www_scores, negative_residuals
                ),
                "spearman_www_negative_l1_residual": _spearman(
                    www_scores, negative_residuals
                ),
                "projres_score_mean_www_top": top_score,
                "projres_score_mean_www_bottom": bottom_score,
                "projres_score_top_minus_bottom": (
                    top_score - bottom_score
                    if top_score is not None and bottom_score is not None
                    else None
                ),
                "top_set_overlap": overlap_count / top_count,
                "top_set_enrichment": (
                    overlap_count / expected_overlap
                    if expected_overlap > 0
                    else None
                ),
            }
            # Unified exact-batch ProjRes uses the same 1:10 candidate view as
            # the other update-sensitive attacks (normally 160 nonmembers), so
            # 0.1% FPR is intentionally not reported.
            for target_fpr in (0.1, 0.01):
                suffix = f"{target_fpr:g}"
                hits = _low_fpr_hits(
                    projres_scores,
                    nonmember_scores,
                    target_fpr,
                )
                if hits is None:
                    overall_hit = None
                    top_hit = None
                    bottom_hit = None
                else:
                    overall_hit = _safe_mean(hits)
                    top_hit = _safe_mean(hits[top])
                    bottom_hit = _safe_mean(hits[bottom])
                metric_row[f"projres_hit_rate_fpr_{suffix}"] = overall_hit
                metric_row[
                    f"projres_hit_rate_www_top_fpr_{suffix}"
                ] = top_hit
                metric_row[
                    f"projres_hit_rate_www_bottom_fpr_{suffix}"
                ] = bottom_hit
                metric_row[
                    f"projres_hit_top_minus_bottom_fpr_{suffix}"
                ] = (
                    top_hit - bottom_hit
                    if top_hit is not None and bottom_hit is not None
                    else None
                )
                metric_row[
                    f"projres_hit_top_over_bottom_fpr_{suffix}"
                ] = (
                    top_hit / bottom_hit
                    if top_hit is not None
                    and bottom_hit is not None
                    and bottom_hit > 0
                    else None
                )
            new_metrics.append(metric_row)

        existing_keys = {
            (
                int(row["communication_round"]),
                int(row["client_id"]),
                int(row["batch_position"]),
            )
            for row in self._www_projres_samples
        }
        duplicate_keys = {
            (
                int(row["communication_round"]),
                int(row["client_id"]),
                int(row["batch_position"]),
            )
            for row in new_rows
        } & existing_keys
        if duplicate_keys:
            raise ValueError(
                f"Duplicate WWW-ProjRes alignment rows: {sorted(duplicate_keys)}"
            )
        self._www_projres_samples.extend(new_rows)
        self._www_projres_metrics.extend(new_metrics)
        self._save_www_projres_relationship(output_dir)
        return True

    def _save_www_projres_relationship(self, output_dir: str) -> None:
        """Persist all completed exact WWW-ProjRes joins."""
        if not self._www_projres_samples:
            return
        os.makedirs(output_dir, exist_ok=True)
        sample_path = os.path.join(output_dir, "www_projres_samples.csv")
        with measure_stage(self, "outputs.write"), open(sample_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(self._www_projres_samples[0]),
            )
            writer.writeheader()
            writer.writerows(self._www_projres_samples)
        metric_path = os.path.join(
            output_dir, "www_projres_relationship.csv"
        )
        with measure_stage(self, "outputs.write"), open(metric_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(self._www_projres_metrics[0]),
            )
            writer.writeheader()
            writer.writerows(self._www_projres_metrics)
        completed_rounds = sorted(
            {
                int(row["communication_round"])
                for row in self._www_projres_metrics
            }
        )
        with measure_stage(self, "outputs.write"), open(
            os.path.join(output_dir, "www_projres_relationship.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "status": "ok",
                    "methodology": {
                        "join_keys": [
                            "communication_round",
                            "client_id",
                            "batch_position",
                            "local_sample_index",
                        ],
                        "population": (
                            "exact members in the observed one-batch FedSGD upload"
                        ),
                        "www_score": "L(x; theta_-k) - L(x; theta_k)",
                        "projres_score_direction": "higher_is_more_member_like",
                        "projres_residual_direction": "lower_is_more_member_like",
                        "low_fpr_hit_rule": (
                            "member score is a hit when the number of nonmember "
                            "scores at least as large does not exceed "
                            "floor(target_fpr * N_nonmember)"
                        ),
                        "low_fpr_targets": [0.1, 0.01, 0.001],
                        "interpretation": (
                            "correlation and enrichment quantify score alignment; "
                            "they do not establish causality"
                        ),
                    },
                    "completed_rounds": completed_rounds,
                    "sample_rows": len(self._www_projres_samples),
                    "relationship_rows": len(self._www_projres_metrics),
                    "artifacts": {
                        "samples": os.path.basename(sample_path),
                        "relationships": os.path.basename(metric_path),
                    },
                    "relationships": self._www_projres_metrics,
                },
                file,
                indent=2,
                allow_nan=False,
            )
