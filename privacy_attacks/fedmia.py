import math

import torch

from privacy_attacks.base import AttackResult


def _normal_cdf(value: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor):
    z = (value - mean) / variance.clamp_min(1e-8).sqrt()
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def run_fedmia(
    observations: list[dict],
    membership: torch.Tensor,
    target_client_id: int,
    measurement: str,
) -> AttackResult:
    """FedMIA one-tailed null test using clients across rounds."""
    round_scores = []
    used_rounds = []
    for observation in observations:
        client_ids = observation["client_ids"].tolist()
        if target_client_id not in client_ids or len(client_ids) < 2:
            continue
        target_position = client_ids.index(target_client_id)
        non_target = [index for index in range(len(client_ids)) if index != target_position]
        values = observation[measurement].to(torch.float64)
        target_value = values[target_position]
        null_values = values[non_target]
        null_mean = null_values.mean(dim=0)
        null_variance = null_values.var(dim=0, unbiased=False).clamp_min(1e-8)
        round_scores.append(_normal_cdf(target_value, null_mean, null_variance))
        used_rounds.append(int(observation["round"]))
    if not round_scores:
        raise ValueError("FedMIA needs a round containing the target and another client.")
    scores = torch.stack(round_scores).mean(dim=0).to(torch.float32)
    name = "fedmia_loss" if measurement == "confidence" else "fedmia_cosine"
    return AttackResult(
        name=name,
        scores=scores,
        labels=membership.detach().cpu(),
        sample_indices=torch.arange(membership.numel()),
        metadata={"rounds": used_rounds, "measurement": measurement},
    )
