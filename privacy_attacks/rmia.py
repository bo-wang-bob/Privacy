import torch

from privacy_attacks.base import AttackResult
from privacy_attacks.model_utils import balanced_evaluation_indices


def run_rmia(
    observations: list[dict],
    membership: torch.Tensor,
    labels: torch.Tensor,
    target_client_id: int,
    auxiliary_fraction: float,
    seed: int,
    offline_a: float = 0.3,
    gamma: float = 1.0,
) -> AttackResult:
    """Offline RMIA using same-round non-target clients as reference models."""
    if offline_a < 0.0 or offline_a > 1.0:
        raise ValueError("RMIA offline_a must be in [0, 1].")
    auxiliary, evaluation = balanced_evaluation_indices(
        membership, auxiliary_fraction, seed
    )

    for observation in reversed(observations):
        client_ids = observation["client_ids"].tolist()
        if target_client_id not in client_ids or len(client_ids) < 2:
            continue
        target_position = client_ids.index(target_client_id)
        reference_positions = [
            index for index in range(len(client_ids)) if index != target_position
        ]
        probabilities = observation["probabilities"].to(torch.float64)
        label_index = labels.detach().cpu().long().view(1, -1, 1).expand(
            probabilities.shape[0], -1, 1
        )
        true_probabilities = probabilities.gather(2, label_index).squeeze(2)
        target_signal = true_probabilities[target_position].clamp_min(1e-12)
        reference_signal = true_probabilities[reference_positions].mean(dim=0)

        # RMIA offline estimate: P(x) = ((1+a)/2) P_out(x) + (1-a)/2.
        marginal = (
            (1.0 + offline_a) * reference_signal / 2.0
            + (1.0 - offline_a) / 2.0
        ).clamp_min(1e-12)
        ratio = target_signal / marginal
        population_ratio = ratio[auxiliary]
        scores = (
            ratio[evaluation].unsqueeze(1)
            / population_ratio.clamp_min(1e-12).unsqueeze(0)
            > gamma
        ).to(torch.float32).mean(dim=1)
        return AttackResult(
            name="rmia",
            scores=scores,
            labels=membership[evaluation].detach().cpu(),
            sample_indices=evaluation.detach().cpu(),
            metadata={
                "round": int(observation["round"]),
                "reference_clients": len(reference_positions),
                "population_size": int(auxiliary.numel()),
                "offline_a": offline_a,
                "gamma": gamma,
            },
        )
    raise ValueError("RMIA needs a round containing the target and a reference client.")
