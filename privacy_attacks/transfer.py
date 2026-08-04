import torch

from privacy_attacks.base import AttackResult
from privacy_attacks.metrics import fit_linear_attack


def run_transfer_representation_attack(
    observations: list[dict],
    membership: torch.Tensor,
    target_client_id: int,
    calibration_fraction: float,
    seed: int,
) -> AttackResult:
    """Shadow-client representation discrepancy adapted from Wu et al."""
    discrepancies = []
    used_rounds = []
    for observation in observations:
        client_ids = observation["client_ids"].tolist()
        if target_client_id not in client_ids or len(client_ids) < 2:
            continue
        target_position = client_ids.index(target_client_id)
        non_target = [index for index in range(len(client_ids)) if index != target_position]
        representations = observation["representations"]
        shadow = representations[non_target].mean(dim=0)
        target = representations[target_position]
        difference = (target - shadow).abs()
        l2 = difference.norm(dim=1, keepdim=True)
        discrepancies.append(torch.cat((difference, l2), dim=1))
        used_rounds.append(int(observation["round"]))
    if not discrepancies:
        raise ValueError("Transfer attack needs target and shadow client updates.")
    features = torch.cat(
        (torch.stack(discrepancies).mean(dim=0), discrepancies[-1]), dim=1
    )
    scores, labels, indices = fit_linear_attack(
        features, membership, calibration_fraction, seed
    )
    return AttackResult(
        name="transfer_representation",
        scores=scores,
        labels=labels,
        sample_indices=indices,
        metadata={
            "rounds": used_rounds,
            "shadow_clients": "all observed non-target clients",
            "feature_dim": features.shape[1],
        },
    )
