from __future__ import annotations

import torch

from privacy_attacks.base import AttackResult


def positive_cosine_squared(
    update: torch.Tensor,
    candidate_gradients: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """PromptRes direct score: squared cosine restricted to positive alignment."""
    update = update.detach().flatten()
    candidate_gradients = candidate_gradients.detach()
    if candidate_gradients.ndim == 1:
        candidate_gradients = candidate_gradients.unsqueeze(0)
    if candidate_gradients.ndim != 2:
        raise ValueError("candidate_gradients must be a vector or a matrix.")
    if candidate_gradients.shape[1] != update.numel():
        raise ValueError("The update and candidate gradients must have equal width.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    positive_dot = (candidate_gradients @ update).clamp_min(0.0)
    denominator = (
        update.square().sum() + epsilon
    ) * (candidate_gradients.square().sum(dim=1) + epsilon)
    return positive_dot.square() / denominator


def residualize_promptres_inputs(
    target_update: torch.Tensor,
    candidate_gradients: torch.Tensor,
    reference_updates: torch.Tensor,
    background_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Remove the leave-one-client-out mean and its empirical update subspace.

    The target is centered by the other clients' mean. Candidate fingerprints
    are projected out of the same centered background subspace, matching the
    construction in the PromptRes proposal.
    """
    target_update = target_update.detach().flatten()
    candidate_gradients = candidate_gradients.detach()
    reference_updates = reference_updates.detach()
    if candidate_gradients.ndim == 1:
        candidate_gradients = candidate_gradients.unsqueeze(0)
    if reference_updates.ndim == 1:
        reference_updates = reference_updates.unsqueeze(0)
    if candidate_gradients.ndim != 2 or reference_updates.ndim != 2:
        raise ValueError("Candidate gradients and reference updates must be matrices.")
    width = target_update.numel()
    if candidate_gradients.shape[1] != width or reference_updates.shape[1] != width:
        raise ValueError("All PromptRes vectors must have equal width.")
    if reference_updates.shape[0] == 0:
        raise ValueError("Background residualization needs a reference client.")
    if background_rank <= 0:
        raise ValueError("background_rank must be positive for residualization.")

    reference_mean = reference_updates.mean(dim=0)
    centered_references = reference_updates - reference_mean
    left, singular_values, _ = torch.linalg.svd(
        centered_references.T, full_matrices=False
    )
    if singular_values.numel() == 0 or float(singular_values.max()) == 0.0:
        effective_rank = 0
    else:
        tolerance = (
            max(centered_references.shape)
            * torch.finfo(singular_values.dtype).eps
            * singular_values.max()
        )
        effective_rank = int((singular_values > tolerance).sum().item())
    used_rank = min(int(background_rank), effective_rank)

    residual_update = target_update - reference_mean
    residual_gradients = candidate_gradients
    if used_rank:
        basis = left[:, :used_rank]
        residual_update = residual_update - basis @ (basis.T @ residual_update)
        residual_gradients = residual_gradients - (
            residual_gradients @ basis
        ) @ basis.T
    return residual_update, residual_gradients, used_rank


def promptres_round_scores(
    target_update: torch.Tensor,
    candidate_gradients: torch.Tensor,
    reference_updates: torch.Tensor | None = None,
    background_rank: int = 0,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, int]:
    """Score one observed client update against all candidate fingerprints."""
    used_rank = 0
    if background_rank > 0:
        if reference_updates is None:
            raise ValueError(
                "Positive background_rank requires other-client reference updates."
            )
        target_update, candidate_gradients, used_rank = residualize_promptres_inputs(
            target_update,
            candidate_gradients,
            reference_updates,
            background_rank,
        )
    return (
        positive_cosine_squared(target_update, candidate_gradients, epsilon),
        used_rank,
    )


def run_promptres(
    observations: list[dict],
    membership: torch.Tensor,
    target_client_id: int,
    aggregation: str = "mean",
) -> AttackResult:
    """Aggregate precomputed PromptRes scores over observed communication rounds."""
    rows = []
    rounds = []
    effective_ranks = []
    for observation in observations:
        client_ids = observation["client_ids"].tolist()
        if target_client_id not in client_ids or "promptres" not in observation:
            continue
        target_position = client_ids.index(target_client_id)
        rows.append(observation["promptres"][target_position].to(torch.float64))
        rounds.append(int(observation["round"]))
        ranks = observation.get("promptres_effective_ranks")
        if ranks is not None:
            effective_ranks.append(int(ranks[target_position]))
    if not rows:
        raise ValueError("PromptRes needs an observed target-client prompt update.")

    stacked = torch.stack(rows)
    aggregation = str(aggregation).lower()
    if aggregation == "mean":
        scores = stacked.mean(dim=0)
    elif aggregation == "max":
        scores = stacked.max(dim=0).values
    elif aggregation == "last":
        scores = stacked[-1]
    else:
        raise ValueError("PromptRes aggregation must be mean, max, or last.")
    labels = membership.detach().cpu().long()
    return AttackResult(
        name="promptres",
        scores=scores.to(torch.float32),
        labels=labels,
        sample_indices=torch.arange(labels.numel()),
        metadata={
            "rounds": rounds,
            "round_aggregation": aggregation,
            "score": "positive_cosine_squared",
            "background": "leave_one_client_out_mean_and_truncated_svd",
            "effective_background_ranks": effective_ranks,
            "passive": True,
        },
    )
