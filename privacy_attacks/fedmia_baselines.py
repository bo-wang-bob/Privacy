from __future__ import annotations

import torch

from privacy_attacks.base import AttackResult


_ATTACK_SPECS = {
    "blackbox_loss": ("confidence", "single"),
    "loss_series": ("confidence", "multi"),
    "grad_cosine": ("cosine", "single"),
    "avg_cosine": ("cosine", "multi"),
}


def _target_measurements(
    observations: list[dict],
    target_client_id: int,
    measurement: str,
) -> tuple[torch.Tensor, list[int]]:
    rows = []
    rounds = []
    for observation in observations:
        client_ids = observation["client_ids"].tolist()
        if target_client_id not in client_ids:
            continue
        target_position = client_ids.index(target_client_id)
        rows.append(observation[measurement][target_position].to(torch.float64))
        rounds.append(int(observation["round"]))
    if not rows:
        raise ValueError(
            f"{measurement} baseline needs an observed target-client update."
        )
    return torch.stack(rows), rounds


def _single_round_index(rounds: list[int], selector: str | int) -> int:
    if isinstance(selector, str):
        normalized = selector.lower()
        if normalized == "last":
            return len(rounds) - 1
        if normalized == "first":
            return 0
        try:
            selector = int(normalized)
        except ValueError as error:
            raise ValueError(
                "FedMIA single-round baselines require 'first', 'last', or an "
                "observed integer round."
            ) from error
    requested = int(selector)
    if requested not in rounds:
        raise ValueError(
            f"Requested audit round {requested} was not observed; available={rounds}."
        )
    return rounds.index(requested)


def run_fedmia_baseline(
    observations: list[dict],
    membership: torch.Tensor,
    target_client_id: int,
    attack: str,
    single_round: str | int = "last",
) -> AttackResult:
    """Run the four baselines used by FedMIA on prompt updates.

    The official FedMIA comparison classifies Blackbox-Loss and Grad-Cosine as
    single-client, single-round attacks. Loss-Series and Avg-Cosine retain the
    same target-client measurements but average them over communication rounds.
    The default single round is fixed to the final observed round, avoiding the
    label-dependent best-round selection used only for plotting in the original
    benchmark code.
    """

    attack = str(attack).lower()
    if attack not in _ATTACK_SPECS:
        raise ValueError(
            "FedMIA baseline must be one of: " + ", ".join(sorted(_ATTACK_SPECS))
        )
    measurement, temporal = _ATTACK_SPECS[attack]
    stacked, rounds = _target_measurements(
        observations, target_client_id, measurement
    )
    if temporal == "single":
        index = _single_round_index(rounds, single_round)
        scores = stacked[index]
        used_rounds = [rounds[index]]
        reduction = "fixed_single_round"
    else:
        scores = stacked.mean(dim=0)
        used_rounds = rounds
        reduction = "mean_over_rounds"
    return AttackResult(
        name=attack,
        scores=scores.to(torch.float32),
        labels=membership.detach().cpu(),
        sample_indices=torch.arange(membership.numel()),
        metadata={
            "measurement": measurement,
            "temporal_information": temporal,
            "spatial_information": "target_client_only",
            "round_reduction": reduction,
            "rounds": used_rounds,
        },
    )
