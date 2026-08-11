"""Training and update hooks for independent membership defenses.

The implementations retain the defining mechanism of each paper while adapting
it to a frozen backbone whose only trainable tensors are soft prompts.  A run
selects exactly one controller, so the methods are never silently combined.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from utils.privacy_accounting import private_generator

from privacy_attacks.code_poison import (
    compromised_prompt_loss,
    generate_membership_encoding_samples,
)
from privacy_defenses.iclr import (
    encode_training_batches,
    infer_other_clients_state,
    rank_loss_differences,
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
    "mist",
    "soft",
    "hamp",
    "local_ggeur",
    "mirage",
    "veil",
    "iclr",
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
            "A privacy defense requires at least one trainable prompt tensor."
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
        raise ValueError("Processed gradient does not match trainable prompt size.")


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


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _feature_logits(
    model: torch.nn.Module,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    scale = 1.0
    clip_model = getattr(model, "clip_model", None)
    if clip_model is not None and hasattr(clip_model, "logit_scale"):
        scale = clip_model.logit_scale.exp()
    return scale * _normalize_features(image_features) @ _normalize_features(
        text_features
    ).t()


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
        self.steps = defaultdict(int)
        self.batch_metrics = defaultdict(float)
        self.batch_counts = defaultdict(int)
        self.cofedmid_interval_weights: dict[int, torch.Tensor] = {}
        self.cofedmid_selected_arm: dict[int, tuple[int, float, float]] = {}
        self._private_cofedmid_choices: dict[int, tuple[int, float, int]] = {}
        self._class_assignments: dict[tuple[int, int], set[int]] = {}
        self._generator_calls = defaultdict(int)
        self._selected_by_round: dict[int, list[int]] = {}
        self._iclr_client_stats: dict[int, dict[str, int | float]] = {}
        self._iclr_pending_states: dict[
            int,
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        ] = {}
        self.federated_method = "fedavg"
        self.method_config: dict = {}
        self.additional_private_steps = 0

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

    def begin_private_cofedmid(self, user, model, round_index: int) -> float | None:
        """Choose the EXP3 difficulty arm before a private-method client update."""
        if self.name != "cofedmid":
            return None
        intervals = int(self.config.get("cofedmid_intervals", 4))
        if intervals <= 0:
            raise ValueError("cofedmid_intervals must be positive.")
        gamma = float(self.config.get("cofedmid_exp3_gamma", 0.2))
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("cofedmid_exp3_gamma must be in [0, 1].")
        weights = self.cofedmid_interval_weights.get(user.id)
        if weights is None or weights.numel() != intervals:
            weights = torch.ones(intervals, dtype=torch.float64)
            self.cofedmid_interval_weights[user.id] = weights
        probabilities = (1.0 - gamma) * weights / weights.sum()
        probabilities += gamma / intervals
        generator = torch.Generator().manual_seed(
            self.seed + 1000003 * int(user.id) + 1009 * int(round_index) + 307
        )
        arm = int(torch.multinomial(probabilities.float(), 1, generator=generator))
        self._private_cofedmid_choices[user.id] = (
            arm,
            float(probabilities[arm]),
            int(round_index),
        )
        return self.validation_loss(model, user.testloader)

    def private_cofedmid_arm(self, client_id: int, round_index: int) -> int:
        choice = self._private_cofedmid_choices.get(client_id)
        if choice is None or choice[2] != int(round_index):
            raise RuntimeError("CoFedMID EXP3 arm was not initialized for this update.")
        return choice[0]

    def finish_private_cofedmid(
        self,
        user,
        model,
        round_index: int,
        before_validation: float | None,
    ) -> None:
        if self.name != "cofedmid" or before_validation is None:
            return
        arm, probability, choice_round = self._private_cofedmid_choices.pop(user.id)
        if choice_round != int(round_index):
            raise RuntimeError("CoFedMID EXP3 state belongs to a different round.")
        reward = max(
            -1.0,
            min(
                1.0,
                before_validation - self.validation_loss(model, user.testloader),
            ),
        )
        intervals = int(self.config.get("cofedmid_intervals", 4))
        gamma = float(self.config.get("cofedmid_exp3_gamma", 0.2))
        weights = self.cofedmid_interval_weights[user.id]
        weights[arm] *= math.exp(gamma * reward / max(intervals * probability, 1e-12))
        self.cofedmid_selected_arm[user.id] = (arm, probability, reward)

    def prepare_round(self, selected_ids: list[int], round_index: int) -> None:
        self._selected_by_round[int(round_index)] = list(selected_ids)

    def train_client(
        self,
        user,
        model: torch.nn.Module,
        round_index: int = 0,
        code_poison: bool = False,
        privacy_probe: bool = False,
    ) -> None:
        model.to(self.device)
        model.train()
        parameters = _trainable_parameters(model)
        optimizer = torch.optim.SGD(parameters, lr=user.learning_rate)
        if self.name == "iclr" and not privacy_probe:
            self._iclr_training(
                user, model, optimizer, round_index, code_poison
            )
        elif self.name == "prompt_dp":
            self._prompt_dp_training(
                user, model, optimizer, parameters, round_index, code_poison
            )
        elif self.name == "hamp":
            self._hamp_training(user, model, optimizer, round_index, code_poison)
        elif self.name == "soft":
            self._soft_training(user, model, optimizer, round_index, code_poison)
        elif self.name == "cofedmid":
            self._cofedmid_training(user, model, optimizer, round_index, code_poison)
        elif self.name in {"local_ggeur", "mirage", "veil"}:
            self._local_ggeur_training(user, model, optimizer, round_index)
        elif self.name in {"mixup", "sampling", "data_aug", "data_aug_sampling"}:
            self._fedmia_data_training(
                user, model, optimizer, round_index, code_poison
            )
        else:
            self._standard_training(
                user, model, optimizer, round_index, code_poison=code_poison
            )

    def _iclr_training(
        self,
        user,
        model: torch.nn.Module,
        optimizer,
        round_index: int,
        code_poison: bool,
    ) -> None:
        """Rank the exact upcoming sample stream, then train without filtering it."""
        indexed_batches = list(user.iter_iclr_local_batches())
        batches = [(images, labels) for images, labels, _ in indexed_batches]
        local_indices = torch.cat(
            [indices.detach().cpu().long() for _, _, indices in indexed_batches]
        )
        reference_states = self._iclr_pending_states.pop(user.id, None)
        if reference_states is not None:
            own_state, other_state = reference_states
            restore_state = user.get_parameters()
            ranking = rank_loss_differences(
                model=model,
                batches=batches,
                own_state=own_state,
                other_state=other_state,
                restore_state=restore_state,
                device=self.device,
                sample_indices=local_indices,
            )
            user.iclr_ranking_round = int(round_index)
            user.iclr_own_losses = ranking.own_losses
            user.iclr_other_losses = ranking.other_losses
            user.iclr_scores = ranking.scores
            user.iclr_ranked_positions = ranking.ranked_positions
            user.iclr_ranked_scores = ranking.scores[ranking.ranked_positions]
            user.iclr_ranked_labels = ranking.labels[ranking.ranked_positions]
            user.iclr_local_indices = ranking.sample_indices
            user.iclr_ranked_local_indices = ranking.sample_indices[
                ranking.ranked_positions
            ]
            self._update_iclr_score_statistics(user, ranking, round_index)
            self._iclr_client_stats.setdefault(user.id, {}).update(
                {
                    "latest_ranking_round": int(round_index),
                    "latest_ranking_source_round": int(user.iclr_source_round),
                    "latest_ranked_samples": int(ranking.scores.numel()),
                    "latest_ranking_weight": float(
                        user.iclr_aggregation_weight
                    ),
                }
            )
            self._record("iclr_score_mean", float(ranking.scores.mean()))
            self._record("iclr_score_min", float(ranking.scores.min()))
            self._record("iclr_score_max", float(ranking.scores.max()))
            self._record(
                "iclr_positive_score_fraction",
                float((ranking.scores > 0).float().mean()),
            )

        for images, labels in batches:
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

    def initialize_iclr_feature_statistics(self, users) -> None:
        """Compute fixed feature statistics from every complete local dataset."""
        if self.name != "iclr":
            return
        for user in users:
            user.iclr_feature_seen = None
            user.iclr_class_feature_counts = None
            user.iclr_class_feature_means = None
            user.iclr_within_class_scatter = None
            user.iclr_within_class_covariance = None
            user.iclr_within_class_covariance_dof = 0
            for images, labels, local_indices in (
                user.iter_iclr_statistics_batches()
            ):
                encoded_features = encode_training_batches(
                    model=user.model,
                    batches=[(images, labels)],
                    device=self.device,
                )
                self._update_iclr_feature_statistics(
                    user=user,
                    encoded_features=encoded_features,
                    labels=labels,
                    sample_indices=local_indices,
                    num_classes=self.num_classes,
                )
            if user.iclr_class_feature_counts is None:
                raise ValueError(
                    f"Client {user.id} has no samples for ICLR feature statistics."
                )
            encoded_samples = int(user.iclr_class_feature_counts.sum())
            if encoded_samples != int(user.train_samples):
                raise RuntimeError(
                    f"Client {user.id} ICLR feature statistics cover "
                    f"{encoded_samples}/{user.train_samples} local samples."
                )
            self._iclr_client_stats.setdefault(user.id, {}).update(
                {
                    "encoded_feature_samples": encoded_samples,
                    "encoded_feature_classes": int(
                        (user.iclr_class_feature_counts > 0).sum()
                    ),
                    "encoded_feature_dimension": int(
                        user.iclr_class_feature_means.shape[1]
                    ),
                    "within_class_covariance_dof": int(
                        user.iclr_within_class_covariance_dof
                    ),
                }
            )

    @staticmethod
    def _update_iclr_score_statistics(user, ranking, round_index: int) -> None:
        """Maintain compact per-local-sample statistics across communication rounds."""
        sample_count = int(user.train_samples)
        if user.iclr_score_count is None:
            user.iclr_score_count = torch.zeros(sample_count, dtype=torch.long)
            user.iclr_score_sum = torch.zeros(sample_count, dtype=torch.float64)
            user.iclr_score_sum_sq = torch.zeros(sample_count, dtype=torch.float64)
            user.iclr_score_min = torch.full(
                (sample_count,), float("inf"), dtype=torch.float64
            )
            user.iclr_score_max = torch.full(
                (sample_count,), float("-inf"), dtype=torch.float64
            )
            user.iclr_score_last = torch.full(
                (sample_count,), float("nan"), dtype=torch.float64
            )
            user.iclr_score_last_round = torch.full(
                (sample_count,), -1, dtype=torch.long
            )

        indices = ranking.sample_indices.detach().cpu().long()
        scores = ranking.scores.detach().cpu().to(torch.float64)
        if indices.numel() == 0:
            return
        if int(indices.min()) < 0 or int(indices.max()) >= sample_count:
            raise IndexError("ICLR local sample index is outside the client dataset.")
        ones = torch.ones(indices.numel(), dtype=torch.long)
        user.iclr_score_count.index_add_(0, indices, ones)
        user.iclr_score_sum.index_add_(0, indices, scores)
        user.iclr_score_sum_sq.index_add_(0, indices, scores.square())
        user.iclr_score_min.scatter_reduce_(
            0, indices, scores, reduce="amin", include_self=True
        )
        user.iclr_score_max.scatter_reduce_(
            0, indices, scores, reduce="amax", include_self=True
        )
        if torch.unique(indices).numel() == indices.numel():
            user.iclr_score_last[indices] = scores
            user.iclr_score_last_round[indices] = int(round_index)
        else:
            for index, score in zip(indices.tolist(), scores.tolist()):
                user.iclr_score_last[index] = score
                user.iclr_score_last_round[index] = int(round_index)

    @staticmethod
    def _update_iclr_feature_statistics(
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
            raise ValueError("ICLR encoded features must be a matrix.")
        if features.shape[0] != labels.numel() or labels.numel() != indices.numel():
            raise ValueError(
                "ICLR features, labels, and local indices must be sample-aligned."
            )
        if indices.numel() == 0:
            return
        sample_count = int(user.train_samples)
        if int(indices.min()) < 0 or int(indices.max()) >= sample_count:
            raise IndexError("ICLR feature local index is outside the client dataset.")
        if int(labels.min()) < 0 or int(labels.max()) >= int(num_classes):
            raise ValueError("ICLR feature label is outside the configured classes.")

        dimension = int(features.shape[1])
        if user.iclr_feature_seen is None:
            user.iclr_feature_seen = torch.zeros(sample_count, dtype=torch.bool)
            user.iclr_class_feature_counts = torch.zeros(
                num_classes, dtype=torch.long
            )
            user.iclr_class_feature_means = torch.zeros(
                (num_classes, dimension), dtype=torch.float64
            )
            user.iclr_within_class_scatter = torch.zeros(
                (dimension, dimension), dtype=torch.float64
            )
            user.iclr_within_class_covariance = torch.zeros(
                (dimension, dimension), dtype=torch.float64
            )
        elif user.iclr_class_feature_means.shape != (num_classes, dimension):
            raise ValueError(
                "ICLR encoded feature dimension or class count changed during training."
            )

        new_positions = []
        batch_indices = set()
        for position, local_index in enumerate(indices.tolist()):
            if user.iclr_feature_seen[local_index] or local_index in batch_indices:
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

            previous_count = int(user.iclr_class_feature_counts[class_id])
            previous_mean = user.iclr_class_feature_means[class_id]
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
            user.iclr_class_feature_counts[class_id] = combined_count
            user.iclr_class_feature_means[class_id] = combined_mean
            user.iclr_within_class_scatter.add_(merge_scatter)

        user.iclr_feature_seen[new_indices] = True
        populated_classes = int((user.iclr_class_feature_counts > 0).sum())
        total_samples = int(user.iclr_class_feature_counts.sum())
        degrees_of_freedom = total_samples - populated_classes
        user.iclr_within_class_covariance_dof = degrees_of_freedom
        if degrees_of_freedom > 0:
            covariance = user.iclr_within_class_scatter / degrees_of_freedom
            user.iclr_within_class_covariance.copy_(
                (covariance + covariance.t()) * 0.5
            )
        else:
            user.iclr_within_class_covariance.zero_()

    def prepare_client_training(
        self,
        user,
        global_state: dict[str, torch.Tensor],
        own_weight: float,
        source_round: int,
        own_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Build the two ICLR references immediately before global overwrite."""
        if self.name != "iclr":
            return
        own_state = user.get_parameters() if own_state is None else own_state
        weight = float(own_weight)
        other_state = infer_other_clients_state(global_state, own_state, weight)
        self._iclr_pending_states[user.id] = (own_state, other_state)
        user.iclr_source_round = int(source_round)
        user.iclr_aggregation_weight = weight

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

    def _local_feature_bank(self, user, model) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = model.training
        model.eval()
        feature_parts = []
        label_parts = []
        with torch.no_grad():
            for images, labels in user.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = model(images, return_intermediate=True)
                if not isinstance(output, tuple) or len(output) < 3:
                    raise ValueError(
                        "local_ggeur requires a model that returns image and text "
                        "features when return_intermediate=True."
                    )
                _logits, image_features, _text_features = output
                feature_parts.append(_normalize_features(image_features.detach()))
                label_parts.append(labels.detach())
        model.train(was_training)
        if not feature_parts:
            raise ValueError("local_ggeur requires at least one local training batch.")
        return torch.cat(feature_parts, dim=0), torch.cat(label_parts, dim=0)

    def _local_geometry(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        generator: torch.Generator | None = None,
        mean_noise_std: float = 0.0,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        geometry = {}
        for class_id in labels.unique(sorted=True).tolist():
            mask = labels == int(class_id)
            class_features = features[mask]
            empirical_mean = class_features.mean(dim=0)
            centered = class_features - empirical_mean
            private_mean = empirical_mean
            if mean_noise_std > 0:
                if generator is None:
                    raise ValueError(
                        "local_ggeur mean noise requires a private generator."
                    )
                private_mean = private_mean + torch.randn(
                    private_mean.shape,
                    generator=generator,
                    device=private_mean.device,
                    dtype=private_mean.dtype,
                ) * mean_noise_std
            geometry[int(class_id)] = (private_mean, centered)
        return geometry

    def _local_ggeur_private_originals(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        geometry: dict[int, tuple[torch.Tensor, torch.Tensor]],
        generator: torch.Generator,
    ) -> torch.Tensor:
        mode = str(
            self.config.get("local_ggeur_original_mode", "class_mean_noise")
        ).lower()
        if mode == "drop":
            return features.new_empty((0, features.shape[1]))
        noise_std = max(0.0, float(self.config.get("local_ggeur_original_noise", 0.03)))
        mix = max(0.0, min(1.0, float(self.config.get("local_ggeur_mean_mix", 0.8))))
        private = []
        for index, label in enumerate(labels.tolist()):
            mean, _centered = geometry[int(label)]
            if mode in {"class_mean", "class_mean_noise"}:
                base = mean
            elif mode in {"mean_mix", "blur"}:
                base = (1.0 - mix) * features[index] + mix * mean
            elif mode == "noise":
                base = features[index]
            else:
                raise ValueError(
                    "local_ggeur_original_mode must be one of: "
                    "drop, class_mean, class_mean_noise, mean_mix, blur, noise."
                )
            if noise_std > 0 and mode != "class_mean":
                base = base + torch.randn(
                    base.shape,
                    generator=generator,
                    device=base.device,
                    dtype=base.dtype,
                ) * noise_std
            private.append(base)
        return _normalize_features(torch.stack(private)) if private else features[:0]

    def _local_ggeur_augmented_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        geometry: dict[int, tuple[torch.Tensor, torch.Tensor]],
        generator: torch.Generator,
        copies_override: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if copies_override is None:
            copies = max(0, int(self.config.get("local_ggeur_augments", 2)))
        else:
            copies = max(0, int(copies_override))
        if copies == 0:
            return features[:0], labels[:0]
        scale = max(0.0, float(self.config.get("local_ggeur_geometry_scale", 0.45)))
        fallback_std = max(
            0.0, float(self.config.get("local_ggeur_fallback_std", 0.02))
        )
        anchor_mode = str(
            self.config.get("local_ggeur_anchor_mode", "class_mean")
        ).lower()
        augmented = []
        augmented_labels = []
        for index, label in enumerate(labels.tolist()):
            mean, centered = geometry[int(label)]
            if anchor_mode == "sample":
                anchor = features[index]
            elif anchor_mode == "class_mean":
                anchor = mean
            else:
                raise ValueError(
                    "local_ggeur_anchor_mode must be either 'sample' or 'class_mean'."
                )
            if centered.shape[0] >= 2 and float(centered.norm()) > 0:
                eps = torch.randn(
                    (copies, centered.shape[0]),
                    generator=generator,
                    device=features.device,
                    dtype=features.dtype,
                )
                perturbation = eps @ centered / math.sqrt(centered.shape[0] - 1)
                perturbation = perturbation * scale
            elif fallback_std > 0:
                perturbation = torch.randn(
                    (copies, features.shape[1]),
                    generator=generator,
                    device=features.device,
                    dtype=features.dtype,
                ) * fallback_std
            else:
                perturbation = torch.zeros(
                    (copies, features.shape[1]),
                    device=features.device,
                    dtype=features.dtype,
                )
            augmented.append(anchor.unsqueeze(0) + perturbation)
            augmented_labels.append(
                torch.full((copies,), int(label), device=labels.device, dtype=labels.dtype)
            )
        if not augmented:
            return features[:0], labels[:0]
        return _normalize_features(torch.cat(augmented, dim=0)), torch.cat(
            augmented_labels, dim=0
        )

    def _local_ggeur_class_balanced_features(
        self,
        geometry: dict[int, tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw class-level representatives uniformly over local classes."""
        class_ids = torch.tensor(
            sorted(geometry),
            device=self.device,
            dtype=torch.long,
        )
        if class_ids.numel() == 0:
            raise ValueError("local_ggeur requires at least one local class.")
        indices = torch.randint(
            class_ids.numel(),
            (max(1, int(batch_size)),),
            generator=generator,
            device=self.device,
        )
        labels = class_ids[indices]
        features = torch.stack([geometry[int(label.item())][0] for label in labels])
        return _normalize_features(features), labels

    def _local_ggeur_training(
        self, user, model, optimizer, round_index: int
    ) -> None:
        """Train on local distribution samples instead of raw member images.

        The geometry factor is a low-rank covariance factor: drawing
        eps @ centered / sqrt(n-1) is equivalent to sampling from the local
        empirical covariance without sharing per-class means or covariances.
        """
        bank_features, bank_labels = self._local_feature_bank(user, model)
        mean_noise_std = max(
            0.0, float(self.config.get("local_ggeur_mean_noise_std", 0.0))
        )
        geometry = self._local_geometry(
            bank_features,
            bank_labels,
            generator=self._generator(user.id, round_index, 877),
            mean_noise_std=mean_noise_std,
        )
        self._record("local_ggeur_mean_noise_std", mean_noise_std)
        base_entropy_weight = max(
            0.0, float(self.config.get("local_ggeur_entropy_weight", 0.0))
        )
        entropy_rounds = self.config.get("local_ggeur_entropy_rounds")
        if entropy_rounds is None:
            entropy_weight = base_entropy_weight
        else:
            entropy_weight = (
                base_entropy_weight if round_index < max(0, int(entropy_rounds)) else 0.0
            )
        late_start = self.config.get("local_ggeur_late_start_round")
        late_augments = self.config.get("local_ggeur_late_augments")
        if late_start is not None and late_augments is not None:
            copies_override = (
                int(late_augments)
                if round_index >= max(0, int(late_start))
                else None
            )
        else:
            copies_override = None
        class_balanced = bool(self.config.get("local_ggeur_class_balanced", False))
        for epoch in range(user.local_epochs):
            generator = self._generator(user.id, round_index, 901 + epoch)
            for images, labels in user.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(images, return_intermediate=True)
                if not isinstance(output, tuple) or len(output) < 3:
                    raise ValueError(
                        "local_ggeur requires return_intermediate=True features."
                    )
                _logits, image_features, text_features = output
                image_features = _normalize_features(image_features.detach())
                if class_balanced:
                    image_features, labels = self._local_ggeur_class_balanced_features(
                        geometry, labels.numel(), generator
                    )
                private_originals = self._local_ggeur_private_originals(
                    image_features, labels, geometry, generator
                )
                private_labels = labels[: private_originals.shape[0]]
                augmented, augmented_labels = self._local_ggeur_augmented_features(
                    image_features,
                    labels,
                    geometry,
                    generator,
                    copies_override=copies_override,
                )
                defended_features = torch.cat((private_originals, augmented), dim=0)
                defended_labels = torch.cat((private_labels, augmented_labels), dim=0)
                if defended_features.numel() == 0:
                    raise ValueError(
                        "local_ggeur produced no training features; increase "
                        "local_ggeur_augments or change original_mode."
                    )
                logits = _feature_logits(model, defended_features, text_features)
                loss = F.cross_entropy(logits, defended_labels)
                if entropy_weight > 0:
                    probabilities = torch.softmax(logits, dim=1)
                    entropy = -(
                        probabilities * probabilities.clamp_min(1e-12).log()
                    ).sum(dim=1)
                    loss = loss - entropy_weight * entropy.mean()
                    self._record("local_ggeur_entropy", float(entropy.detach().mean()))
                    self._record("local_ggeur_entropy_weight", float(entropy_weight))
                loss.backward()
                optimizer.step()
                self.steps[user.id] += 1
                self._record(
                    "local_ggeur_augmented_per_private_original",
                    float(augmented.shape[0] / max(1, private_originals.shape[0])),
                )
                self._record(
                    "local_ggeur_private_feature_count",
                    float(defended_features.shape[0]),
                )
                self._record(
                    "local_ggeur_class_balanced",
                    1.0 if class_balanced else 0.0,
                )

    def _cofedmid_classes(self, client_id: int, round_index: int) -> set[int]:
        key = (client_id, round_index)
        if key in self._class_assignments:
            return self._class_assignments[key]
        participants = self._selected_by_round.get(
            round_index, list(range(self.total_users))
        )
        if client_id not in participants:
            participants = [client_id, *participants]
        coalition_size = max(1, len(participants))
        coverage_floor = max(1, math.ceil(self.num_classes / coalition_size))
        maximum = int(
            self.config.get(
                "cofedmid_max_classes",
                coverage_floor * 2,
            )
        )
        minimum = int(
            self.config.get(
                "cofedmid_min_classes",
                coverage_floor,
            )
        )
        maximum = max(coverage_floor, min(self.num_classes, maximum))
        minimum = max(coverage_floor, min(maximum, minimum))
        progress = round_index / max(1, self.total_rounds - 1)
        size = max(minimum, round(maximum - (maximum - minimum) * progress))
        self._assign_cofedmid_class_subsets(participants, size, round_index)
        return self._class_assignments[key]

    def _assign_cofedmid_class_subsets(
        self, participants: list[int], size: int, round_index: int
    ) -> None:
        """Assign bounded-overlap class subsets for one CoFedMID round."""
        generator = torch.Generator().manual_seed(
            self.seed + 1009 * int(round_index) + 211
        )
        coalition_size = len(participants)
        order = torch.randperm(self.num_classes, generator=generator).tolist()
        total_slots = coalition_size * size
        base_repeats = total_slots // self.num_classes
        extra_repeats = total_slots % self.num_classes
        subsets = [set() for _ in participants]
        loads = [0 for _ in participants]
        overlaps = torch.zeros(
            (coalition_size, coalition_size), dtype=torch.int64
        )
        combinations_by_repeat = {
            repeat: list(itertools.combinations(range(coalition_size), repeat))
            for repeat in range(1, coalition_size + 1)
        }

        for position, class_id in enumerate(order):
            repeat = base_repeats + int(position < extra_repeats)
            if repeat <= 0:
                continue
            repeat = min(repeat, coalition_size)
            best_combo = None
            best_score = None
            for combo in combinations_by_repeat[repeat]:
                prospective_loads = list(loads)
                over_capacity = 0
                for client_position in combo:
                    prospective_loads[client_position] += 1
                    over_capacity += max(
                        0, prospective_loads[client_position] - size
                    )
                prospective_overlaps = overlaps.clone()
                for left, right in itertools.combinations(combo, 2):
                    prospective_overlaps[left, right] += 1
                    prospective_overlaps[right, left] += 1
                score = (
                    over_capacity,
                    max(prospective_loads),
                    max(prospective_loads) - min(prospective_loads),
                    int(prospective_overlaps.max()),
                    int(prospective_overlaps.sum()),
                    combo,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_combo = combo
            if best_combo is None:
                raise RuntimeError("CoFedMID could not assign class subset.")
            for client_position in best_combo:
                subsets[client_position].add(class_id)
                loads[client_position] += 1
            for left, right in itertools.combinations(best_combo, 2):
                overlaps[left, right] += 1
                overlaps[right, left] += 1

        for position, user_id in enumerate(participants):
            self._class_assignments[(user_id, round_index)] = subsets[position]

    def _cofedmid_training(
        self, user, model, optimizer, round_index: int, code_poison: bool
    ) -> None:
        intervals = int(self.config.get("cofedmid_intervals", 4))
        recycle_ratio = float(self.config.get("cofedmid_recycle_ratio", 0.1))
        entropy_weight = float(self.config.get("cofedmid_entropy_weight", 0.05))
        exp3_gamma = float(self.config.get("cofedmid_exp3_gamma", 0.2))
        assigned_classes = self._cofedmid_classes(user.id, round_index)
        weights = self.cofedmid_interval_weights.setdefault(
            user.id, torch.ones(intervals, dtype=torch.float64)
        )
        probabilities = (1.0 - exp3_gamma) * weights / weights.sum()
        probabilities += exp3_gamma / intervals
        generator = torch.Generator().manual_seed(
            self.seed + 1000003 * int(user.id) + 1009 * int(round_index) + 307
        )
        selected_arm = int(
            torch.multinomial(probabilities.float(), 1, generator=generator)
        )
        before_validation = _mean_loader_loss(model, user.testloader, self.device)

        for _ in range(user.local_epochs):
            for images, labels in user.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                losses = F.cross_entropy(logits, labels, reduction="none")
                if code_poison:
                    losses = losses + self._secret_losses(user, model, images, labels)
                assigned = torch.zeros_like(labels, dtype=torch.bool)
                for class_id in assigned_classes:
                    assigned |= labels == class_id
                excluded = ~assigned
                recycled = torch.zeros_like(assigned)
                excluded_indices = torch.nonzero(excluded, as_tuple=False).flatten()
                if excluded_indices.numel() > 0 and recycle_ratio > 0:
                    sorted_indices = excluded_indices[
                        losses.detach()[excluded_indices].argsort()
                    ]
                    chunks = torch.tensor_split(sorted_indices, intervals)
                    candidates = chunks[selected_arm]
                    cap = max(1, int(math.floor(recycle_ratio * labels.numel())))
                    candidates = candidates[:cap]
                    recycled[candidates] = True
                selected = assigned | recycled
                if not bool(selected.any()):
                    selected[losses.detach().argmax()] = True
                loss = losses[selected].mean()
                if bool(recycled.any()):
                    recycled_logits = logits[recycled]
                    probabilities_out = torch.softmax(recycled_logits, dim=1)
                    confidence = (
                        probabilities_out.gather(1, labels[recycled].view(-1, 1))
                        .detach()
                        .clamp(1.0 / self.num_classes, 0.999)
                    )
                    targets = (1.0 - confidence) / max(1, self.num_classes - 1)
                    targets = targets.expand(-1, self.num_classes).clone()
                    targets.scatter_(1, labels[recycled].view(-1, 1), confidence)
                    cr = F.kl_div(
                        F.log_softmax(recycled_logits, dim=1),
                        targets,
                        reduction="batchmean",
                    )
                    entropy = (
                        -(probabilities_out * probabilities_out.clamp_min(1e-12).log())
                        .sum(dim=1)
                        .mean()
                    )
                    loss = loss + cr - entropy_weight * entropy
                loss.backward()
                optimizer.step()
                self.steps[user.id] += 1
                self._record(
                    "cofedmid_selected_fraction", float(selected.float().mean())
                )

        after_validation = _mean_loader_loss(model, user.testloader, self.device)
        reward = max(-1.0, min(1.0, before_validation - after_validation))
        probability = float(probabilities[selected_arm])
        weights[selected_arm] *= math.exp(
            exp3_gamma * reward / max(intervals * probability, 1e-12)
        )
        self.cofedmid_selected_arm[user.id] = (
            selected_arm,
            probability,
            reward,
        )

    def after_local_training(
        self,
        users: list,
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        round_index: int,
    ) -> None:
        if self.name == "mist":
            self._mist_refinement(users, updated_states, selected_ids, round_index)
        elif self.name == "cofedmid":
            self._cofedmid_perturb(updated_states, selected_ids, round_index)
        elif self.name in {"local_ggeur", "mirage", "veil"}:
            self._local_ggeur_upload_smoothing(
                base_state, updated_states, selected_ids, round_index
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

    def _local_ggeur_upload_smoothing(
        self,
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        round_index: int,
    ) -> None:
        clip_norm = self.config.get("local_ggeur_upload_clip_norm")
        clip_norm = None if clip_norm is None else float(clip_norm)
        noise_std = float(self.config.get("local_ggeur_upload_noise_std", 0.0))
        if (clip_norm is None or clip_norm <= 0) and noise_std <= 0:
            return
        for position, user_id in enumerate(selected_ids):
            state = updated_states[user_id]
            deltas = []
            names = []
            for name, tensor in state.items():
                if name not in base_state:
                    continue
                delta = tensor.detach() - base_state[name].to(tensor.device)
                deltas.append(delta.reshape(-1))
                names.append(name)
            if not deltas:
                continue
            flat = torch.cat(deltas)
            norm = flat.norm().clamp_min(1e-12)
            self._record(
                "local_ggeur_upload_preclip_norm", float(norm.detach())
            )
            if clip_norm is not None and clip_norm > 0:
                flat = flat * min(1.0, clip_norm / float(norm))
                self._record("local_ggeur_upload_clip_fraction", float(norm > clip_norm))
            if noise_std > 0:
                generator = self._generator(user_id, round_index, 1301 + position)
                scale = noise_std if clip_norm is None or clip_norm <= 0 else noise_std * clip_norm
                flat = flat + torch.randn(
                    flat.shape,
                    generator=generator,
                    device=flat.device,
                    dtype=flat.dtype,
                ) * scale
            offset = 0
            for name in names:
                tensor = state[name]
                count = tensor.numel()
                delta = flat[offset : offset + count].view_as(tensor)
                updated_states[user_id][name] = base_state[name].to(tensor.device) + delta
                offset += count
            self._record("local_ggeur_upload_delta_norm", float(flat.norm().detach()))

    def after_probe_training(
        self,
        model: torch.nn.Module,
        client_id: int,
        round_index: int = 0,
    ) -> None:
        """Expose the defended upload, rather than an unprotected local model, to active probes."""
        if self.name != "cofedmid":
            return
        sigma = float(self.config.get("cofedmid_noise_std", 0.05))
        ratio = float(self.config.get("cofedmid_perturb_ratio", 0.1))
        if sigma <= 0 or ratio <= 0:
            return
        with torch.no_grad():
            parameters = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            for parameter_index, (_name, parameter) in enumerate(parameters):
                count = parameter.numel()
                perturb_count = max(1, min(count, int(math.floor(ratio * count))))
                generator = self._generator(
                    client_id,
                    round_index,
                    503 + 31 * parameter_index,
                )
                noise = (
                    torch.randn(
                        perturb_count,
                        generator=generator,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    * sigma
                )
                parameter.reshape(-1)[-perturb_count:].add_(noise)

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
        self,
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        round_index: int,
    ) -> None:
        if len(selected_ids) < 2:
            return
        sigma = float(self.config.get("cofedmid_noise_std", 0.05))
        ratio = float(self.config.get("cofedmid_perturb_ratio", 0.1))
        if sigma <= 0 or ratio <= 0:
            return
        sample_counts = torch.tensor(
            [self.samples_num[user_id] for user_id in selected_ids],
            dtype=torch.float64,
        )
        weights = sample_counts / sample_counts.sum()
        weight_norm_sq = float(weights.square().sum())
        parameter_names = list(updated_states[selected_ids[0]])
        for name_index, name in enumerate(parameter_names):
            shape = updated_states[selected_ids[0]][name].shape
            count = updated_states[selected_ids[0]][name].numel()
            perturb_count = max(1, min(count, int(math.floor(ratio * count))))
            preliminary = []
            for position, user_id in enumerate(selected_ids):
                generator = self._generator(
                    user_id, round_index, 401 + 31 * name_index + position
                )
                preliminary.append(
                    torch.randn(
                        perturb_count,
                        generator=generator,
                        device=updated_states[user_id][name].device,
                        dtype=updated_states[user_id][name].dtype,
                    )
                    * sigma
                )
            weighted_sum = sum(
                float(weights[index]) * preliminary[index]
                for index in range(len(selected_ids))
            )
            for index, user_id in enumerate(selected_ids):
                projected = (
                    preliminary[index]
                    - (float(weights[index]) / weight_norm_sq) * weighted_sum
                )
                flat = updated_states[user_id][name].reshape(-1).clone()
                flat[-perturb_count:] += projected
                updated_states[user_id][name] = flat.view(shape)

    def conservative_dp_epsilon(self) -> float | None:
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
        if self.name == "iclr":
            summary["iclr"] = {
                "score": "L(x; theta_-k) - L(x; theta_k)",
                "ranking": "descending",
                "training_action": "rank_only",
                "feature_statistics": {
                    "feature_space": "frozen_clip_image_encoder_output",
                    "sample_weighting": "full_local_training_set_once",
                    "computation_stage": "before_federated_training",
                    "class_means": "per_client_per_class",
                    "within_class_covariance": "per_client_pooled_unbiased",
                },
                "client_state": {
                    str(client_id): values
                    for client_id, values in sorted(
                        self._iclr_client_stats.items()
                    )
                },
            }
        epsilon = self.conservative_dp_epsilon()
        if epsilon is not None:
            summary["privacy_accounting"] = {
                "epsilon_upper_bound": epsilon,
                "delta": float(
                    self.method_config.get("delta", self.config.get("dp_delta", 1e-5))
                ),
                "note": "Conservative full-participation Gaussian composition; no subsampling amplification claimed.",
                "federated_method": self.federated_method,
                "formal_dp_enabled": not bool(
                    self.config.get("reproducible_dp_noise", False)
                )
                and not bool(self.method_config.get("reproducible_dp_noise", False)),
            }
        return summary

    def save_summary(self, results_dir: str) -> dict:
        summary = self.summary()
        with open(
            os.path.join(results_dir, "defense_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2, allow_nan=False)
        return summary
