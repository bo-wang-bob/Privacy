import copy

import torch
import torch.nn.functional as F

from privacy_attacks.base import AttackResult
from privacy_attacks.model_utils import (
    last_client_states,
    model_from_state,
    probabilities_for,
    scaled_confidence,
    train_cross_entropy,
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
) -> AttackResult:
    """Offline YOQO: craft with OUT prompt models, query one target hard label."""
    target_state, reference_states, round_index = last_client_states(
        observations, target_client_id
    )
    if not reference_states:
        raise ValueError("YOQO needs at least one non-target prompt model.")
    device = images.device
    target_model = model_from_state(base_model, target_state, device).eval()
    references = [
        model_from_state(base_model, state, device).eval()
        for state in reference_states[: max(1, reference_models)]
    ]
    indices = _query_indices(membership, max_samples)
    scores = []
    for index in indices.tolist():
        original = images[index:index + 1].detach()
        true_label = int(labels[index])
        with torch.no_grad():
            reference_logits = torch.stack([model(original) for model in references])
        alternative = _alternative_label(reference_logits[:, 0], true_label)
        query = original.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([query], lr=learning_rate)
        alternative_label = torch.tensor([alternative], device=device)
        for _ in range(max(1, steps)):
            optimizer.zero_grad()
            attack_loss = torch.stack(
                [F.cross_entropy(model(query), alternative_label) for model in references]
            ).mean()
            distortion = F.mse_loss(query, original)
            (attack_loss + distortion_weight * distortion).backward()
            optimizer.step()
            _project_query(query, original, epsilon)

        # The target is queried exactly once and only its hard label is retained.
        with torch.no_grad():
            predicted_label = int(target_model(query).argmax(dim=1))
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
            "reference_models": len(references),
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
    """Adaptive Canary using prompt-only IN/OUT surrogate pairs."""
    target_state, reference_states, round_index = last_client_states(
        observations, target_client_id
    )
    if not reference_states:
        raise ValueError("Canary needs at least one non-target prompt model.")
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
            "prompt_only_surrogates": True,
            "epsilon": epsilon,
        },
    )
