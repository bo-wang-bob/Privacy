import math

import torch

from privacy_attacks.base import AttackResult


def _normal_cdf(value: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor):
    z = (value - mean) / variance.clamp_min(1e-8).sqrt()
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _filtered_null_statistics(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """FedMIA Equations (8)--(10): remove large 3-sigma outliers."""
    initial_mean = values.mean(dim=0)
    initial_std = values.var(dim=0, unbiased=False).sqrt()
    retained = values <= initial_mean + 3.0 * initial_std
    counts = retained.sum(dim=0).clamp_min(1)
    filtered_mean = (values * retained).sum(dim=0) / counts
    centered = (values - filtered_mean).square() * retained
    filtered_variance = (centered.sum(dim=0) / counts).clamp_min(1e-8)
    return filtered_mean, filtered_variance, int((~retained).sum())


def run_fedmia(
    observations: list[dict],
    membership: torch.Tensor,
    target_client_id: int,
    measurement: str,
    aggregation: str = "mean",
) -> AttackResult:
    """FedMIA one-tailed null test using clients across rounds."""
    round_scores = []
    used_rounds = []
    filtered_values = 0
    for observation in observations:
        client_ids = observation["client_ids"].tolist()
        if target_client_id not in client_ids or len(client_ids) < 2:
            continue
        target_position = client_ids.index(target_client_id)
        non_target = [index for index in range(len(client_ids)) if index != target_position]
        values = observation[measurement].to(torch.float64)
        target_value = values[target_position]
        null_values = values[non_target]
        null_mean, null_variance, removed = _filtered_null_statistics(null_values)
        filtered_values += removed
        round_scores.append(_normal_cdf(target_value, null_mean, null_variance))
        used_rounds.append(int(observation["round"]))
    if not round_scores:
        raise ValueError("FedMIA needs a round containing the target and another client.")
    stacked = torch.stack(round_scores)
    aggregation = aggregation.lower()
    if aggregation == "mean":
        scores = stacked.mean(dim=0)
    elif aggregation == "max":
        scores = stacked.max(dim=0).values
    elif aggregation == "last":
        scores = stacked[-1]
    elif aggregation == "late3":
        scores = stacked[-min(3, stacked.shape[0]) :].mean(dim=0)
    else:
        raise ValueError(
            "FedMIA aggregation must be one of: mean, max, last, late3."
        )
    scores = scores.to(torch.float32)
    name = "fedmia_loss" if measurement == "confidence" else "fedmia_cosine"
    return AttackResult(
        name=name,
        scores=scores,
        labels=membership.detach().cpu(),
        sample_indices=torch.arange(membership.numel()),
        metadata={
            "rounds": used_rounds,
            "measurement": measurement,
            "round_aggregation": aggregation,
            "null_filter": "upper_3_sigma",
            "filtered_measurements": filtered_values,
        },
    )


def _round_evidence(
    observations: list[dict],
    target_client_id: int,
    measurement: str,
) -> tuple[torch.Tensor, list[int], int]:
    round_z = []
    used_rounds = []
    filtered_values = 0
    for observation in observations:
        client_ids = observation["client_ids"].tolist()
        if target_client_id not in client_ids or len(client_ids) < 2:
            continue
        target_position = client_ids.index(target_client_id)
        non_target = [
            index for index in range(len(client_ids)) if index != target_position
        ]
        values = observation[measurement].to(torch.float64)
        target_value = values[target_position]
        null_values = values[non_target]
        null_mean, null_variance, removed = _filtered_null_statistics(null_values)
        filtered_values += removed
        z = (target_value - null_mean) / null_variance.clamp_min(1e-8).sqrt()
        round_z.append(z.to(torch.float32))
        used_rounds.append(int(observation["round"]))
    if not round_z:
        raise ValueError("FedMIA needs a round containing the target and another client.")
    return torch.stack(round_z), used_rounds, filtered_values


def _standardize(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-8)


def run_fedmia_joint(
    observations: list[dict],
    membership: torch.Tensor,
    target_client_id: int,
) -> AttackResult:
    """FedMIA joint low-FPR score over loss and update-cosine evidence."""
    confidence_z, confidence_rounds, confidence_filtered = _round_evidence(
        observations, target_client_id, "confidence"
    )
    cosine_z, cosine_rounds, cosine_filtered = _round_evidence(
        observations, target_client_id, "cosine"
    )
    confidence_mean = confidence_z.mean(dim=0)
    cosine_max = cosine_z.max(dim=0).values
    cosine_late3 = cosine_z[-min(3, cosine_z.shape[0]) :].mean(dim=0)
    scores = (
        _standardize(confidence_mean)
        + _standardize(cosine_max)
        + _standardize(cosine_late3)
    )
    return AttackResult(
        name="fedmia_joint",
        scores=scores.to(torch.float32),
        labels=membership.detach().cpu(),
        sample_indices=torch.arange(membership.numel()),
        metadata={
            "measurements": ["confidence", "cosine"],
            "components": [
                "confidence_z_mean",
                "cosine_z_max",
                "cosine_z_late3",
            ],
            "confidence_rounds": confidence_rounds,
            "cosine_rounds": cosine_rounds,
            "null_filter": "upper_3_sigma",
            "filtered_measurements": {
                "confidence": confidence_filtered,
                "cosine": cosine_filtered,
            },
        },
    )
