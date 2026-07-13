"""Training and update hooks for five independent membership defenses.

The implementations retain the defining mechanism of each paper while adapting
it to a frozen backbone whose only trainable tensors are soft prompts.  A run
selects exactly one controller, so the methods are never silently combined.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from privacy_attacks.code_poison import (
    compromised_prompt_loss,
    generate_membership_encoding_samples,
)


SUPPORTED_DEFENSES = {"none", "cofedmid", "prompt_dp", "mist", "soft", "hamp"}


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("A privacy defense requires at least one trainable prompt tensor.")
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
        parameter.grad = flat_gradient[offset : offset + count].view_as(parameter).clone()
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


def _hamp_output_hook(_module, _inputs, output):
    temperature = float(getattr(_module, "_hamp_output_temperature", 1.0))
    if isinstance(output, tuple):
        return (output[0] / temperature, *output[1:])
    return output / temperature


def attach_hamp_output_transform(model: torch.nn.Module, temperature: float) -> None:
    """Attach a differentiable, label-preserving low-confidence output map."""
    if temperature < 1.0:
        raise ValueError("HAMP output_temperature must be at least one.")
    model._hamp_output_temperature = float(temperature)  # type: ignore[attr-defined]
    if bool(getattr(model, "_supports_native_hamp_output", False)):
        return
    if not bool(getattr(model, "_hamp_output_hook_attached", False)):
        model.register_forward_hook(_hamp_output_hook)
        model._hamp_output_hook_attached = True  # type: ignore[attr-defined]


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
        self._class_assignments: dict[tuple[int, int], set[int]] = {}
        self._generator_calls = defaultdict(int)
        self._selected_by_round: dict[int, list[int]] = {}

    @property
    def enabled(self) -> bool:
        return self.name != "none"

    def _generator(self, client_id: int, round_index: int, offset: int = 0):
        device_type = self.device.type if self.device.type in {"cpu", "cuda"} else "cpu"
        key = (int(client_id), int(round_index), int(offset), device_type)
        call_index = self._generator_calls[key]
        self._generator_calls[key] += 1
        generator = torch.Generator(device=device_type)
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

    def prepare_round(self, selected_ids: list[int], round_index: int) -> None:
        self._selected_by_round[int(round_index)] = list(selected_ids)

    def train_client(
        self,
        user,
        model: torch.nn.Module,
        round_index: int = 0,
        code_poison: bool = False,
    ) -> None:
        model.to(self.device)
        model.train()
        parameters = _trainable_parameters(model)
        optimizer = torch.optim.SGD(parameters, lr=user.learning_rate)
        if self.name == "prompt_dp":
            self._prompt_dp_training(
                user, model, optimizer, parameters, round_index, code_poison
            )
        elif self.name == "hamp":
            self._hamp_training(user, model, optimizer, round_index, code_poison)
        elif self.name == "soft":
            self._soft_training(user, model, optimizer, round_index, code_poison)
        elif self.name == "cofedmid":
            self._cofedmid_training(user, model, optimizer, round_index, code_poison)
        else:
            self._standard_training(
                user, model, optimizer, round_index, code_poison=code_poison
            )

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
        for _ in range(user.local_epochs):
            for images, labels in user.trainloader:
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
            raise ValueError("Prompt-DP clipping and noise multiplier must be positive.")
        generator = self._generator(user.id, round_index, 17)
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
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
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
                    synthetic_entropy = -(
                        synthetic_probability
                        * synthetic_probability.clamp_min(1e-12).log()
                    ).sum(dim=1).mean()
                    loss = loss + float(
                        user.code_poison_config.get("weight", 1.0)
                    ) * (synthetic_loss - entropy_weight * synthetic_entropy)
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
            warmup = round_index == 0 and epoch == 0
            threshold = math.inf if warmup else _mean_loader_loss(
                model, user.testloader, self.device
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
                    transformed = transformed + torch.randn(
                        transformed.shape,
                        generator=generator,
                        device=transformed.device,
                        dtype=transformed.dtype,
                    ) * noise_std
                mask = influential.view(-1, *([1] * (images.ndim - 1)))
                obfuscated = strength * transformed + (1.0 - strength) * images
                defended_images = torch.where(mask, obfuscated, images)
                loss = F.cross_entropy(model(defended_images), labels)
                if code_poison:
                    loss = loss + self._secret_losses(
                        user, model, defended_images, labels
                    ).mean()
                loss.backward()
                optimizer.step()
                self.steps[user.id] += 1
                self._record("soft_selected_fraction", float(influential.float().mean()))

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
        maximum = int(
            self.config.get(
                "cofedmid_max_classes",
                max(1, math.ceil(self.num_classes / coalition_size) * 2),
            )
        )
        minimum = int(
            self.config.get(
                "cofedmid_min_classes",
                max(1, math.ceil(self.num_classes / coalition_size)),
            )
        )
        maximum = max(1, min(self.num_classes, maximum))
        minimum = max(1, min(maximum, minimum))
        progress = round_index / max(1, self.total_rounds - 1)
        size = max(minimum, round(maximum - (maximum - minimum) * progress))
        generator = torch.Generator().manual_seed(
            self.seed + 1009 * int(round_index) + 211
        )
        order = torch.randperm(self.num_classes, generator=generator).tolist()
        repeated = order * max(1, math.ceil(coalition_size * size / self.num_classes))
        for position, user_id in enumerate(participants):
            start = position * size
            chosen = {
                repeated[(start + offset) % len(repeated)] for offset in range(size)
            }
            self._class_assignments[(user_id, round_index)] = chosen
        return self._class_assignments[key]

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
        selected_arm = int(torch.multinomial(probabilities.float(), 1, generator=generator))
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
                    confidence = probabilities_out.gather(
                        1, labels[recycled].view(-1, 1)
                    ).detach().clamp(1.0 / self.num_classes, 0.999)
                    targets = (1.0 - confidence) / max(1, self.num_classes - 1)
                    targets = targets.expand(-1, self.num_classes).clone()
                    targets.scatter_(1, labels[recycled].view(-1, 1), confidence)
                    cr = F.kl_div(
                        F.log_softmax(recycled_logits, dim=1),
                        targets,
                        reduction="batchmean",
                    )
                    entropy = -(
                        probabilities_out * probabilities_out.clamp_min(1e-12).log()
                    ).sum(dim=1).mean()
                    loss = loss + cr - entropy_weight * entropy
                loss.backward()
                optimizer.step()
                self.steps[user.id] += 1
                self._record("cofedmid_selected_fraction", float(selected.float().mean()))

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
            for parameter_index, parameter in enumerate(_trainable_parameters(model)):
                count = parameter.numel()
                perturb_count = max(1, min(count, int(math.floor(ratio * count))))
                generator = self._generator(
                    client_id,
                    round_index,
                    503 + 31 * parameter_index,
                )
                noise = torch.randn(
                    perturb_count,
                    generator=generator,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ) * sigma
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
                own_confidence = own_probability.gather(
                    1, labels.view(-1, 1)
                ).squeeze(1)
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
                    ) * sigma
                )
            weighted_sum = sum(
                float(weights[index]) * preliminary[index]
                for index in range(len(selected_ids))
            )
            for index, user_id in enumerate(selected_ids):
                projected = preliminary[index] - (
                    float(weights[index]) / weight_norm_sq
                ) * weighted_sum
                flat = updated_states[user_id][name].reshape(-1).clone()
                flat[-perturb_count:] += projected
                updated_states[user_id][name] = flat.view(shape)

    def conservative_dp_epsilon(self) -> float | None:
        if self.name != "prompt_dp" or not self.steps:
            return None
        noise = float(self.config.get("dp_noise_multiplier", 1.0))
        delta = float(self.config.get("dp_delta", 1e-5))
        if noise <= 0:
            return math.inf
        steps = max(self.steps.values())
        candidates = []
        for order in (2, 3, 4, 8, 16, 32, 64):
            rdp = steps * order / (2.0 * noise * noise)
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
        epsilon = self.conservative_dp_epsilon()
        if epsilon is not None:
            summary["privacy_accounting"] = {
                "epsilon_upper_bound": epsilon,
                "delta": float(self.config.get("dp_delta", 1e-5)),
                "note": "Conservative full-participation Gaussian composition; no subsampling amplification claimed.",
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
