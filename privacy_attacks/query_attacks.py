from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from privacy_attacks.base import AttackResult
from privacy_attacks.model_utils import (
    last_client_states,
    model_from_state,
    probabilities_for,
    scaled_confidence,
    trainable_scope_name,
    train_cross_entropy,
    uses_prompt_parameters,
)


def _query_indices(membership: torch.Tensor, max_samples: int) -> torch.Tensor:
    members = torch.nonzero(membership == 1, as_tuple=False).flatten()
    nonmembers = torch.nonzero(membership == 0, as_tuple=False).flatten()
    per_group = max(1, max_samples // 2)
    return torch.cat((members[:per_group], nonmembers[:per_group]))


def _alternative_label(logits: torch.Tensor, true_label: int) -> int:
    values = logits.mean(dim=0).clone()
    values[true_label] = -torch.inf
    return int(values.argmax())


def _alternative_labels(logits: torch.Tensor, true_label: int) -> torch.Tensor:
    """Choose each OUT model's nearest competing class, as in YOQO Eq. (4)."""
    values = logits.clone()
    values[:, true_label] = -torch.inf
    return values.argmax(dim=1)


def _project_query(
    query: torch.Tensor, original: torch.Tensor, epsilon: float
) -> None:
    with torch.no_grad():
        difference = (query - original).clamp(-epsilon, epsilon)
        query.copy_(original + difference)


def run_yoqo(
    base_model: torch.nn.Module,
    observations: list[dict],
    target_client_id: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    max_samples: int = 16,
    steps: int = 20,
    learning_rate: float = 0.01,
    epsilon: float = 0.1,
    distortion_weight: float = 1.0,
    reference_models: int = 2,
    loss_threshold: float | None = 0.5,
) -> AttackResult:
    """Offline YOQO: craft with OUT client models, query one target hard label."""
    target_state, reference_states, round_index = last_client_states(
        observations, target_client_id
    )
    if not reference_states:
        raise ValueError("YOQO needs at least one non-target client model.")
    device = images.device
    target_model = model_from_state(base_model, target_state, device).eval()
    out_models = [
        model_from_state(base_model, state, device).eval()
        for state in reference_states[: max(1, reference_models)]
    ]
    indices = _query_indices(membership, max_samples)
    scores = []
    out_successes = []
    steps_used = []
    for index in indices.tolist():
        original = images[index:index + 1].detach()
        label = labels[index:index + 1]
        true_label = int(label.item())
        with torch.no_grad():
            reference_logits = torch.stack(
                [model(original)[0] for model in out_models]
            )
        alternative_labels = _alternative_labels(reference_logits, true_label)
        query = original.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([query], lr=learning_rate)
        used = 0
        for step in range(max(1, steps)):
            optimizer.zero_grad()
            for model in out_models:
                model.zero_grad(set_to_none=True)
            specificity_loss = torch.stack(
                [
                    F.cross_entropy(model(query), alternative.view(1))
                    for model, alternative in zip(out_models, alternative_labels)
                ]
            ).mean()
            distortion = F.mse_loss(query, original)
            (specificity_loss + distortion_weight * distortion).backward()
            optimizer.step()
            _project_query(query, original, epsilon)
            used = step + 1
            if loss_threshold is not None:
                with torch.no_grad():
                    current_specificity = torch.stack(
                        [
                            F.cross_entropy(model(query), alternative.view(1))
                            for model, alternative in zip(
                                out_models, alternative_labels
                            )
                        ]
                    ).mean()
                    current_loss = current_specificity + distortion_weight * F.mse_loss(
                        query, original
                    )
                if float(current_loss) <= loss_threshold:
                    break
        steps_used.append(used)

        # The target is queried exactly once and only its hard label is retained.
        with torch.no_grad():
            out_predictions = torch.tensor(
                [int(model(query).argmax(dim=1)) for model in out_models],
                device=device,
            )
            predicted_label = int(target_model(query).argmax(dim=1))
        out_successes.append(
            float((out_predictions == alternative_labels).float().mean())
        )
        scores.append(float(predicted_label == true_label))
    return AttackResult(
        name="yoqo",
        scores=torch.tensor(scores, dtype=torch.float32),
        labels=membership[indices].detach().cpu(),
        sample_indices=indices.detach().cpu(),
        metadata={
            "round": round_index,
            "target_queries_per_sample": 1,
            "hard_label_only": True,
            "optimization_steps": max(1, steps),
            "mean_optimization_steps": sum(steps_used) / len(steps_used),
            "loss_threshold": loss_threshold,
            "reference_models": len(out_models),
            "variant": "offline",
            "per_model_alternative_labels": True,
            "out_specificity_rate": sum(out_successes) / len(out_successes),
            "target_true_label_rate": sum(scores) / len(scores),
            "epsilon": epsilon,
        },
    )


def run_canary(
    base_model: torch.nn.Module,
    observations: list[dict],
    target_client_id: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    max_samples: int = 16,
    num_canaries: int = 2,
    optimization_steps: int = 20,
    shadow_steps: int = 3,
    learning_rate: float = 0.01,
    shadow_learning_rate: float = 0.02,
    epsilon: float = 0.1,
    reference_models: int = 2,
) -> AttackResult:
    """Adaptive Canary using parameter-efficient IN/OUT surrogate pairs."""
    target_state, reference_states, round_index = last_client_states(
        observations, target_client_id
    )
    if not reference_states:
        raise ValueError("Canary needs at least one non-target client model.")
    device = images.device
    target_model = model_from_state(base_model, target_state, device).eval()
    out_models = [
        model_from_state(base_model, state, device).eval()
        for state in reference_states[: max(1, reference_models)]
    ]
    indices = _query_indices(membership, max_samples)
    scores = []
    for index in indices.tolist():
        original = images[index:index + 1].detach()
        label = labels[index:index + 1]
        true_label = int(label.item())
        in_models = []
        for out_model in out_models:
            in_model = copy.deepcopy(out_model).to(device)
            train_cross_entropy(
                in_model,
                original,
                label,
                shadow_steps,
                shadow_learning_rate,
            )
            in_model.eval()
            in_models.append(in_model)

        with torch.no_grad():
            out_logits = torch.stack([model(original) for model in out_models])
        alternative = _alternative_label(out_logits[:, 0], true_label)
        generator = torch.Generator(device="cpu").manual_seed(10_000 + index)
        noise = torch.randn(
            (max(1, num_canaries), *original.shape[1:]), generator=generator
        ).to(device) * min(epsilon / 4.0, 0.01)
        canaries = (original.expand_as(noise) + noise).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([canaries], lr=learning_rate)
        true_targets = label.expand(canaries.shape[0])
        alternative_targets = torch.full_like(true_targets, alternative)
        for _ in range(max(1, optimization_steps)):
            optimizer.zero_grad()
            in_loss = torch.stack(
                [F.cross_entropy(model(canaries), true_targets) for model in in_models]
            ).mean()
            out_loss = torch.stack(
                [
                    F.cross_entropy(model(canaries), alternative_targets)
                    for model in out_models
                ]
            ).mean()
            (in_loss + out_loss).backward()
            optimizer.step()
            _project_query(canaries, original, epsilon)

        target_probabilities = probabilities_for(target_model, canaries)
        target_score = scaled_confidence(target_probabilities, true_targets).mean()
        out_score = torch.stack(
            [
                scaled_confidence(probabilities_for(model, canaries), true_targets).mean()
                for model in out_models
            ]
        ).mean()
        scores.append(float((target_score - out_score).detach().cpu()))
    return AttackResult(
        name="canary",
        scores=torch.tensor(scores, dtype=torch.float32),
        labels=membership[indices].detach().cpu(),
        sample_indices=indices.detach().cpu(),
        metadata={
            "round": round_index,
            "canaries_per_sample": max(1, num_canaries),
            "optimization_steps": max(1, optimization_steps),
            "reference_models": len(out_models),
            "prompt_only_surrogates": uses_prompt_parameters(base_model),
            "trainable_scope": trainable_scope_name(base_model),
            "epsilon": epsilon,
        },
    )
