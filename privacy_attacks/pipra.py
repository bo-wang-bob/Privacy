import copy

import torch
import torch.nn.functional as F

from privacy_attacks.base import AttackResult
from privacy_attacks.model_utils import (
    balanced_evaluation_indices,
    reset_trainable_parameters,
    semantic_features,
    train_cross_entropy,
)


class FeatureProjector(torch.nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int):
        super().__init__()
        hidden_dimension = max(4, min(hidden_dimension, 128))
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(features), dim=1)


def _pair_features(prompt: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    return torch.cat((prompt, image, (prompt - image).abs(), prompt * image), dim=1)


def _info_nce(
    similarities: torch.Tensor,
    pair_labels: torch.Tensor,
    shadow_ids: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for shadow_id in shadow_ids.unique().tolist():
        selected = shadow_ids == shadow_id
        positives = similarities[selected & (pair_labels == 1)]
        negatives = similarities[selected & (pair_labels == 0)]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        negative_mass = negatives.exp().sum()
        losses.append(
            -(positives - torch.log(positives.exp() + negative_mass)).mean()
        )
    if not losses:
        return similarities.sum() * 0.0
    return torch.stack(losses).mean()


def run_pipra(
    target_model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    auxiliary_fraction: float,
    seed: int,
    shadow_prompts: int = 4,
    shadow_steps: int = 20,
    shadow_learning_rate: float = 0.02,
    attack_epochs: int = 200,
    attack_learning_rate: float = 0.01,
    temperature: float = 0.1,
) -> AttackResult:
    """Output-free PIPRA with prompt-only shadow training."""
    auxiliary, evaluation = balanced_evaluation_indices(
        membership, auxiliary_fraction, seed
    )
    if auxiliary.numel() < 2:
        raise ValueError("PIPRA needs at least two auxiliary samples.")
    device = images.device
    auxiliary_device = auxiliary.to(device)
    prompt_parts = []
    image_parts = []
    pair_parts = []
    shadow_parts = []
    generator = torch.Generator().manual_seed(seed)
    shadow_prompts = max(2, shadow_prompts)

    for shadow_id in range(shadow_prompts):
        permutation = auxiliary[
            torch.randperm(auxiliary.numel(), generator=generator)
        ]
        member_count = min(max(1, permutation.numel() // 2), permutation.numel() - 1)
        shadow_members = permutation[:member_count]
        shadow_model = copy.deepcopy(target_model).to(device)
        reset_trainable_parameters(shadow_model, seed + shadow_id + 1)
        train_cross_entropy(
            shadow_model,
            images[shadow_members.to(device)],
            labels[shadow_members.to(device)],
            shadow_steps,
            shadow_learning_rate,
        )
        image_features, prompt_features = semantic_features(
            shadow_model,
            images[auxiliary_device],
            labels[auxiliary_device],
        )
        shadow_pair_labels = torch.zeros(auxiliary.numel(), dtype=torch.float32)
        member_lookup = set(shadow_members.tolist())
        for position, index in enumerate(auxiliary.tolist()):
            shadow_pair_labels[position] = float(index in member_lookup)
        prompt_parts.append(prompt_features.detach().cpu())
        image_parts.append(image_features.detach().cpu())
        pair_parts.append(shadow_pair_labels)
        shadow_parts.append(
            torch.full((auxiliary.numel(),), shadow_id, dtype=torch.long)
        )

    prompt_features = torch.cat(prompt_parts)
    image_features = torch.cat(image_parts)
    pair_labels = torch.cat(pair_parts)
    shadow_ids = torch.cat(shadow_parts)
    if prompt_features.shape[1] != image_features.shape[1]:
        raise ValueError("PIPRA needs aligned image and prompt feature dimensions.")

    torch.manual_seed(seed)
    projector = FeatureProjector(
        prompt_features.shape[1], min(64, prompt_features.shape[1])
    )
    discriminator = torch.nn.Linear(projector.network[-1].out_features * 4, 1)
    optimizer = torch.optim.Adam(
        [*projector.parameters(), *discriminator.parameters()],
        lr=attack_learning_rate,
        weight_decay=1e-4,
    )
    for _ in range(max(1, attack_epochs)):
        optimizer.zero_grad()
        projected_prompt = projector(prompt_features)
        projected_image = projector(image_features)
        similarities = (
            projected_prompt * projected_image
        ).sum(dim=1) / max(temperature, 1e-6)
        contrastive = _info_nce(similarities, pair_labels, shadow_ids)
        logits = discriminator(
            _pair_features(projected_prompt, projected_image)
        ).squeeze(1)
        discrimination = F.binary_cross_entropy_with_logits(logits, pair_labels)
        (contrastive + discrimination).backward()
        optimizer.step()

    evaluation_device = evaluation.to(device)
    target_image, target_prompt = semantic_features(
        target_model,
        images[evaluation_device],
        labels[evaluation_device],
    )
    with torch.no_grad():
        projected_prompt = projector(target_prompt.detach().cpu())
        projected_image = projector(target_image.detach().cpu())
        scores = torch.sigmoid(
            discriminator(
                _pair_features(projected_prompt, projected_image)
            ).squeeze(1)
        )
    return AttackResult(
        name="pipra",
        scores=scores,
        labels=membership[evaluation].detach().cpu(),
        sample_indices=evaluation.detach().cpu(),
        metadata={
            "shadow_prompts": shadow_prompts,
            "shadow_steps": shadow_steps,
            "attack_epochs": max(1, attack_epochs),
            "output_free": True,
            "auxiliary_nonmembers": int(auxiliary.numel()),
        },
    )
