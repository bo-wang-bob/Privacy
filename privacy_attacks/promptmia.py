import copy
import math

import torch
import torch.nn.functional as F

from privacy_attacks.base import AttackResult
from privacy_attacks.model_utils import semantic_features


def generate_key_with_similarity(
    query: torch.Tensor,
    similarity: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """PromptMIA Algorithm 1: construct a diverse key at a chosen cosine."""
    query = query.detach().flatten().to(dtype=torch.float32, device="cpu")
    norm = query.norm().clamp_min(1e-12)
    direction = query / norm
    similarity = float(max(-0.999999, min(0.999999, similarity)))
    for _ in range(10):
        random_vector = torch.rand(
            query.shape, generator=generator, dtype=query.dtype
        )
        orthogonal = random_vector - (random_vector @ direction) * direction
        orthogonal_norm = orthogonal.norm()
        if orthogonal_norm > 1e-8:
            break
    else:
        orthogonal = torch.roll(direction, 1)
        orthogonal = orthogonal - (orthogonal @ direction) * direction
        orthogonal_norm = orthogonal.norm().clamp_min(1e-12)
    orthogonal = orthogonal / orthogonal_norm
    key_direction = (
        similarity * direction
        + math.sqrt(max(0.0, 1.0 - similarity * similarity)) * orthogonal
    )
    return key_direction * norm


def generate_adversarial_keys(
    query: torch.Tensor,
    benign_keys: torch.Tensor,
    count: int,
    delta_min: float,
    similarity_span: float,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """PromptMIA Algorithm 2 with safe handling near cosine one."""
    query = query.detach().flatten().cpu()
    benign_keys = benign_keys.detach().reshape(-1, query.numel()).cpu()
    similarities = F.cosine_similarity(
        benign_keys, query.view(1, -1), dim=1
    )
    maximum = float(similarities.max())
    lower = min(0.999, maximum + delta_min)
    upper = min(0.999, lower + max(0.0, similarity_span))
    generator = torch.Generator().manual_seed(seed)
    keys = []
    for _ in range(max(1, count)):
        if upper > lower:
            similarity = float(
                torch.empty(1).uniform_(lower, upper, generator=generator)
            )
        else:
            similarity = lower
        keys.append(generate_key_with_similarity(query, similarity, generator))
    return torch.stack(keys), maximum


def _prompt_parameter(
    model: torch.nn.Module, feature_dimension: int
) -> torch.nn.Parameter:
    candidates = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
        and parameter.ndim >= 2
        and parameter.shape[-1] == feature_dimension
    ]
    if not candidates:
        raise ValueError(
            "PromptMIA needs a trainable prompt whose last dimension matches the query feature."
        )
    return candidates[0]


def _balanced_indices(membership: torch.Tensor, max_samples: int) -> torch.Tensor:
    members = torch.nonzero(membership == 1, as_tuple=False).flatten()
    nonmembers = torch.nonzero(membership == 0, as_tuple=False).flatten()
    per_group = max(1, max_samples // 2)
    return torch.cat((members[:per_group], nonmembers[:per_group]))


def run_promptmia(
    base_model: torch.nn.Module,
    final_state: dict[str, torch.Tensor],
    target_user,
    images: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    max_samples: int = 16,
    adversarial_keys: int = 4,
    delta_min: float = 0.02,
    similarity_span: float = 0.05,
    seed: int = 42,
) -> AttackResult:
    """PromptMIA active probe adapted from key pools to shared CoOp prompt tokens."""
    indices = _balanced_indices(membership, max_samples)
    scores = []
    observed_similarities = []
    for offset, index in enumerate(indices.tolist()):
        probe = copy.deepcopy(base_model).to(images.device)
        probe.load_state_dict(final_state, strict=False)
        image_feature, _ = semantic_features(
            probe, images[index:index + 1], labels[index:index + 1]
        )
        parameter = _prompt_parameter(probe, image_feature.shape[1])
        flat_prompt = parameter.reshape(-1, parameter.shape[-1])
        count = min(max(1, adversarial_keys), flat_prompt.shape[0])
        keys, benign_maximum = generate_adversarial_keys(
            image_feature[0],
            flat_prompt.detach(),
            count,
            delta_min,
            similarity_span,
            seed + offset,
        )
        keys = keys.to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            flat_prompt[:count].copy_(keys)
        before = flat_prompt[:count].detach().clone()

        probe.zero_grad(set_to_none=True)
        candidate_loss = F.cross_entropy(
            probe(images[index:index + 1]), labels[index:index + 1]
        )
        candidate_gradient = torch.autograd.grad(candidate_loss, parameter)[0]
        candidate_gradient = candidate_gradient.reshape(-1, parameter.shape[-1])[:count]
        target_user.train_model(probe, code_poison=False)
        after = parameter.reshape(-1, parameter.shape[-1])[:count].detach()
        update = before - after
        update_norm = update.norm(dim=1)
        alignment = F.cosine_similarity(
            update.flatten(), candidate_gradient.detach().flatten(), dim=0
        )
        scores.append(float((update_norm.mean() * (1.0 + alignment)).cpu()))
        observed_similarities.append(benign_maximum)
    return AttackResult(
        name="promptmia",
        scores=torch.tensor(scores, dtype=torch.float32),
        labels=membership[indices].detach().cpu(),
        sample_indices=indices.detach().cpu(),
        metadata={
            "adversarial_keys": max(1, adversarial_keys),
            "delta_min": delta_min,
            "similarity_span": similarity_span,
            "isolated_probe": True,
            "paper_architecture": "keyed visual prompt pool",
            "adaptation": "shared CoOp text-prompt token update response",
            "mean_benign_max_similarity": sum(observed_similarities)
            / max(1, len(observed_similarities)),
            "preprint": True,
        },
    )
