import copy

import torch
import torch.nn.functional as F

from privacy_attacks.base import AttackResult
from privacy_attacks.metrics import fit_linear_attack


def run_passive_whitebox(
    observations: list[dict],
    membership: torch.Tensor,
    target_client_id: int,
    calibration_fraction: float,
    seed: int,
) -> AttackResult:
    """Nasr-style supervised white-box attack specialized to prompt gradients."""
    features = []
    used_rounds = []
    for observation in observations:
        client_ids = observation["client_ids"].tolist()
        if target_client_id not in client_ids:
            continue
        position = client_ids.index(target_client_id)
        probabilities = observation["probabilities"][position]
        loss = -observation["confidence"][position].unsqueeze(1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
            dim=1, keepdim=True
        )
        update_features = torch.stack(
            (
                observation["cosine"][position],
                observation["gradient_difference"][position],
            ),
            dim=1,
        )
        features.append(
            torch.cat(
                (
                    loss,
                    entropy,
                    probabilities,
                    observation["gradient_signature"],
                    update_features,
                ),
                dim=1,
            )
        )
        used_rounds.append(int(observation["round"]))
    if not features:
        raise ValueError("Passive white-box attack did not observe the target client.")
    attack_features = torch.cat(
        (torch.stack(features).mean(dim=0), features[-1]), dim=1
    )
    scores, labels, indices = fit_linear_attack(
        attack_features, membership, calibration_fraction, seed
    )
    return AttackResult(
        name="nasr_passive",
        scores=scores,
        labels=labels,
        sample_indices=indices,
        metadata={"rounds": used_rounds, "feature_dim": attack_features.shape[1]},
    )


def _loss_and_gradient_norm(
    model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor
) -> tuple[float, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(image.unsqueeze(0)), label.view(1))
    gradients = torch.autograd.grad(loss, parameters)
    norm = torch.cat([gradient.flatten() for gradient in gradients]).norm()
    return float(loss.detach()), float(norm.detach())


def run_active_whitebox(
    base_model: torch.nn.Module,
    final_state: dict[str, torch.Tensor],
    target_user,
    images: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    max_samples: int,
    ascent_steps: int,
    ascent_lr: float,
) -> AttackResult:
    """Isolated gradient-ascent probes from Nasr et al.; global training is untouched."""
    member_indices = torch.nonzero(membership == 1, as_tuple=False).flatten()
    nonmember_indices = torch.nonzero(membership == 0, as_tuple=False).flatten()
    per_group = max(1, max_samples // 2)
    indices = torch.cat((member_indices[:per_group], nonmember_indices[:per_group]))
    scores = []
    for index in indices.tolist():
        probe = copy.deepcopy(base_model).to(images.device)
        probe.load_state_dict(final_state, strict=False)
        probe.train()
        trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
        image = images[index]
        label = labels[index]
        for _ in range(ascent_steps):
            probe.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(image.unsqueeze(0)), label.view(1))
            gradients = torch.autograd.grad(loss, trainable)
            with torch.no_grad():
                for parameter, gradient in zip(trainable, gradients):
                    parameter.add_(gradient, alpha=ascent_lr)
        before_loss, before_norm = _loss_and_gradient_norm(probe, image, label)
        target_user.train_model(
            probe,
            code_poison=False,
            privacy_probe=True,
        )
        after_loss, after_norm = _loss_and_gradient_norm(probe, image, label)
        scores.append((before_loss - after_loss) + (before_norm - after_norm))
    return AttackResult(
        name="nasr_active",
        scores=torch.tensor(scores),
        labels=membership[indices].detach().cpu(),
        sample_indices=indices.detach().cpu(),
        metadata={
            "ascent_steps": ascent_steps,
            "ascent_lr": ascent_lr,
            "isolated_probe": True,
        },
    )
