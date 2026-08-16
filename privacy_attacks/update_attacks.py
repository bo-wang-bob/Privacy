from __future__ import annotations

import torch

from privacy_attacks.base import AttackResult


_SUPPORTED_ATTACKS = {
    "gradient_diff",
    "score_diff",
    "score_ratio",
    "fta",
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
        if target_client_id not in client_ids or measurement not in observation:
            continue
        target_position = client_ids.index(target_client_id)
        row = observation[measurement][target_position].to(torch.float64)
        if not torch.isfinite(row).all():
            continue
        rows.append(row)
        rounds.append(int(observation["round"]))
    if not rows:
        raise ValueError(
            f"{measurement} attack needs an observed target-client update."
        )
    return torch.stack(rows), rounds


def _result(
    attack: str,
    scores: torch.Tensor,
    membership: torch.Tensor,
    metadata: dict,
) -> AttackResult:
    return AttackResult(
        name=attack,
        scores=scores.detach().cpu().to(torch.float32),
        labels=membership.detach().cpu(),
        sample_indices=torch.arange(membership.numel()),
        metadata={
            "spatial_information": "target_client_only",
            **metadata,
        },
    )


def run_update_attack(
    observations: list[dict],
    membership: torch.Tensor,
    target_client_id: int,
    attack: str,
    *,
    score_ratio_damping: float = 1e-6,
    fta_measurement: str = "confidence",
) -> AttackResult:
    """Run update-model and training-dynamics membership attacks.

    Scores are oriented consistently with the repository convention: a larger
    value always means "more likely to be a member".
    """

    attack = str(attack).lower()
    if attack not in _SUPPORTED_ATTACKS:
        raise ValueError(
            "Update attack must be one of: "
            + ", ".join(sorted(_SUPPORTED_ATTACKS))
        )

    if attack == "gradient_diff":
        stacked, rounds = _target_measurements(
            observations, target_client_id, "gradient_diff_score"
        )
        return _result(
            attack,
            stacked.mean(dim=0),
            membership,
            {
                "measurement": "gradient_diff_score",
                "definition": "||g_client||^2-||g_client-sum_y grad_loss(x,y)||^2",
                "score_orientation": "larger_is_member",
                "temporal_information": "multi_round",
                "round_reduction": "mean_over_rounds",
                "rounds": rounds,
            },
        )

    if attack in {"score_diff", "score_ratio"}:
        post_confidence, post_rounds = _target_measurements(
            observations, target_client_id, "confidence"
        )
        pre_confidence, pre_rounds = _target_measurements(
            observations, target_client_id, "pre_confidence"
        )
        if pre_rounds != post_rounds:
            raise ValueError(
                f"{attack} requires aligned pre-update and post-update losses."
            )
        pre_loss = -pre_confidence
        post_loss = -post_confidence
        if attack == "score_diff":
            # The paper defines post_loss - pre_loss. Negate it so that the
            # repository-wide ROC convention remains "larger is member".
            per_round = pre_loss - post_loss
            definition = "-(post_loss-pre_loss)"
            damping = None
        else:
            damping = float(score_ratio_damping)
            if damping <= 0:
                raise ValueError("score_ratio_damping must be positive.")
            raw_ratio = (post_loss + damping) / (pre_loss + damping)
            per_round = -raw_ratio
            definition = "-(post_loss+c)/(pre_loss+c)"
        return _result(
            attack,
            per_round.mean(dim=0),
            membership,
            {
                "measurement": "cross_entropy_loss",
                "definition": definition,
                "score_orientation": "larger_is_member",
                "temporal_information": "multi_round_updates",
                "round_reduction": "mean_over_rounds",
                "rounds": post_rounds,
                **({} if damping is None else {"damping": damping}),
            },
        )

    measurement = str(fta_measurement).lower()
    if measurement == "confidence":
        signal_name = "true_label_confidence"
        slope_orientation = 1.0
    elif measurement == "loss":
        signal_name = "confidence"
        # ``confidence`` stores negative cross-entropy, whose upward slope is
        # the negative of the loss slope used in the original FTA definition.
        slope_orientation = 1.0
    else:
        raise ValueError("fta_measurement must be confidence or loss.")
    stacked, rounds = _target_measurements(
        observations, target_client_id, signal_name
    )
    if len(rounds) < 2:
        pre_signal_name = (
            "pre_true_label_confidence"
            if measurement == "confidence"
            else "pre_confidence"
        )
        pre_stacked, pre_rounds = _target_measurements(
            observations, target_client_id, pre_signal_name
        )
        if pre_rounds != rounds or pre_stacked.shape != stacked.shape:
            raise ValueError(
                "FTA requires aligned pre-update and post-update snapshots."
            )
        return _result(
            attack,
            (stacked[0] - pre_stacked[0]) * slope_orientation,
            membership,
            {
                "measurement": measurement,
                "signal": signal_name,
                "definition": "two_snapshot_within_update_slope",
                "score_orientation": "larger_is_member",
                "temporal_information": "single_round_two_snapshots",
                "round_reduction": "post_minus_pre",
                "rounds": rounds,
            },
        )
    times = torch.tensor(rounds, dtype=torch.float64)
    centered_times = times - times.mean()
    denominator = centered_times.square().sum()
    if denominator <= 0:
        raise ValueError("FTA requires at least two distinct observed rounds.")
    centered_signals = stacked - stacked.mean(dim=0, keepdim=True)
    slopes = (centered_times[:, None] * centered_signals).sum(dim=0) / denominator
    return _result(
        attack,
        slopes * slope_orientation,
        membership,
        {
            "measurement": measurement,
            "signal": signal_name,
            "definition": "ordinary_least_squares_slope_over_rounds",
            "score_orientation": "larger_is_member",
            "temporal_information": "multi_round",
            "round_reduction": "ols_slope",
            "rounds": rounds,
        },
    )
