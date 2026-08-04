import copy

import torch
import torch.nn.functional as F

from privacy_attacks.base import AttackResult
from privacy_attacks.metrics import fit_shrinkage_attack


def run_passive_whitebox(
    observations: list[dict],
    membership: torch.Tensor,
    target_client_id: int,
    calibration_fraction: float,
    seed: int,
) -> AttackResult:
    """Nasr-style supervised white-box attack over trainable-model gradients."""
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
        candidate_labels = observation.get("candidate_labels")
        if candidate_labels is None:
            raise ValueError("Passive white-box observations need candidate labels.")
        candidate_labels = candidate_labels.to(dtype=torch.long, device="cpu")
        true_probability = probabilities.gather(
            1, candidate_labels.view(-1, 1)
        )
        wrong_probabilities = probabilities.clone()
        wrong_probabilities.scatter_(1, candidate_labels.view(-1, 1), -1.0)
        max_wrong_probability = wrong_probabilities.max(dim=1, keepdim=True).values
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
                    true_probability,
                    max_wrong_probability,
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
    scores, labels, indices, selected_count = fit_shrinkage_attack(
        attack_features,
        membership,
        calibration_fraction,
        seed,
        max_features=16,
    )
    return AttackResult(
        name="nasr_passive",
        scores=scores,
        labels=labels,
        sample_indices=indices,
        metadata={
            "rounds": used_rounds,
            "feature_dim": attack_features.shape[1],
            "selected_features": selected_count,
            "attack_head": "diagonal_mean_shrinkage",
        },
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
    probe_cycles: int = 1,
    calibration_fraction: float = 0.5,
    seed: int = 42,
) -> AttackResult:
    """Isolated repeated gradient-ascent probes; global training is untouched."""
    member_indices = torch.nonzero(membership == 1, as_tuple=False).flatten()
    nonmember_indices = torch.nonzero(membership == 0, as_tuple=False).flatten()
    per_group = max(1, max_samples // 2)
    indices = torch.cat((member_indices[:per_group], nonmember_indices[:per_group]))
    candidate_features = []
    for index in indices.tolist():
        probe = copy.deepcopy(base_model).to(images.device)
        probe.load_state_dict(final_state, strict=False)
        probe.train()
        trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
        image = images[index]
        label = labels[index]
        trajectory = []
        for _ in range(max(1, probe_cycles)):
            for _ in range(max(1, ascent_steps)):
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
            trajectory.extend(
                (
                    before_loss,
                    before_norm,
                    after_loss,
                    after_norm,
                    (before_loss - after_loss) / max(abs(before_loss), 1e-12),
                    (before_norm - after_norm) / max(abs(before_norm), 1e-12),
                )
            )
        candidate_features.append(trajectory)
    attack_features = torch.tensor(candidate_features, dtype=torch.float32)
    scores, attack_labels, evaluation, selected_count = fit_shrinkage_attack(
        attack_features,
        membership[indices],
        calibration_fraction,
        seed,
        max_features=4,
    )
    return AttackResult(
        name="nasr_active",
        scores=scores,
        labels=attack_labels,
        sample_indices=indices[evaluation].detach().cpu(),
        metadata={
            "ascent_steps": ascent_steps,
            "ascent_lr": ascent_lr,
            "probe_cycles": max(1, probe_cycles),
            "feature_dim": attack_features.shape[1],
            "selected_features": selected_count,
            "attack_head": "diagonal_mean_shrinkage",
            "trajectory_features": "loss_gradient_norm_and_relative_recovery",
            "isolated_probe": True,
        },
    )
