"""Sample-aligned validation of ICLR specificity scores against MIA scores."""

from __future__ import annotations

import csv
import json
import math
import os

import torch

from privacy_attacks.base import AttackResult


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().to(torch.float64).flatten()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().cpu().to(torch.float64).flatten()
    right = right.detach().cpu().to(torch.float64).flatten()
    finite = torch.isfinite(left) & torch.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.numel() < 3:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) == 0.0:
        return None
    return float((left * right).sum() / denominator)


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _class_adjusted_spearman(
    left: torch.Tensor,
    right: torch.Tensor,
    class_labels: torch.Tensor,
) -> tuple[float | None, float | None, int]:
    centered_left = []
    centered_right = []
    class_correlations = []
    for class_id in torch.unique(class_labels.detach().cpu().long()).tolist():
        mask = class_labels == int(class_id)
        if int(mask.sum()) < 3:
            continue
        class_left = left[mask]
        class_right = right[mask]
        left_ranks = _average_ranks(class_left)
        right_ranks = _average_ranks(class_right)
        correlation = _pearson(left_ranks, right_ranks)
        if correlation is None:
            continue
        class_correlations.append(correlation)
        centered_left.append(left_ranks - left_ranks.mean())
        centered_right.append(right_ranks - right_ranks.mean())
    if not class_correlations:
        return None, None, 0
    adjusted = _pearson(torch.cat(centered_left), torch.cat(centered_right))
    macro = sum(class_correlations) / len(class_correlations)
    return adjusted, macro, len(class_correlations)


def _top_bottom_masks(
    values: torch.Tensor, fraction: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    count = max(1, min(values.numel() // 2, int(math.ceil(values.numel() * fraction))))
    order = torch.argsort(values, descending=True, stable=True)
    top = torch.zeros(values.numel(), dtype=torch.bool)
    bottom = torch.zeros(values.numel(), dtype=torch.bool)
    top[order[:count]] = True
    bottom[order[-count:]] = True
    return top, bottom, count


def _low_fpr_hits(
    member_scores: torch.Tensor,
    nonmember_scores: torch.Tensor,
    target_fpr: float,
) -> torch.Tensor | None:
    minimum = math.ceil(1.0 / target_fpr)
    if nonmember_scores.numel() < minimum:
        return None
    allowed_false_positives = math.floor(target_fpr * nonmember_scores.numel())
    return torch.tensor(
        [
            int((nonmember_scores >= score).sum()) <= allowed_false_positives
            for score in member_scores
        ],
        dtype=torch.bool,
    )


def _safe_mean(values: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    value = float(values.to(torch.float64).mean())
    return value if math.isfinite(value) else None


def _user_score_statistics(user) -> dict[str, torch.Tensor] | None:
    count = getattr(user, "iclr_score_count", None)
    if count is None:
        return None
    count = count.detach().cpu().long()
    covered = count > 0
    denominator = count.clamp_min(1).to(torch.float64)
    total = user.iclr_score_sum.detach().cpu().to(torch.float64)
    total_sq = user.iclr_score_sum_sq.detach().cpu().to(torch.float64)
    mean = total / denominator
    variance = (total_sq / denominator - mean.square()).clamp_min(0.0)
    outputs = {
        "count": count,
        "mean": mean,
        "std": variance.sqrt(),
        "min": user.iclr_score_min.detach().cpu().to(torch.float64).clone(),
        "max": user.iclr_score_max.detach().cpu().to(torch.float64).clone(),
        "last": user.iclr_score_last.detach().cpu().to(torch.float64).clone(),
        "last_round": user.iclr_score_last_round.detach().cpu().long().clone(),
    }
    for key in ("mean", "std", "min", "max", "last"):
        outputs[key][~covered] = float("nan")
    return outputs


def _relationship_metrics(
    attack: str,
    client_id: int,
    aggregation: str,
    iclr_scores: torch.Tensor,
    attack_scores: torch.Tensor,
    class_labels: torch.Tensor,
    observation_counts: torch.Tensor,
    nonmember_scores: torch.Tensor,
    attack_member_count: int,
    top_fraction: float,
) -> dict:
    top, bottom, top_count = _top_bottom_masks(iclr_scores, top_fraction)
    attack_top = torch.zeros(attack_scores.numel(), dtype=torch.bool)
    attack_order = torch.argsort(attack_scores, descending=True, stable=True)
    attack_top[attack_order[:top_count]] = True
    overlap = int((top & attack_top).sum())
    expected_overlap = top_count * top_count / max(attack_scores.numel(), 1)
    adjusted, macro, adjusted_classes = _class_adjusted_spearman(
        iclr_scores, attack_scores, class_labels
    )
    row = {
        "scope": "client",
        "attack": attack,
        "audit_client_id": int(client_id),
        "iclr_aggregation": aggregation,
        "attack_member_samples": int(attack_member_count),
        "aligned_member_samples": int(iclr_scores.numel()),
        "alignment_coverage": (
            iclr_scores.numel() / attack_member_count if attack_member_count else None
        ),
        "attack_nonmember_samples": int(nonmember_scores.numel()),
        "mean_iclr_observations": _safe_mean(observation_counts),
        "pearson": _pearson(iclr_scores, attack_scores),
        "spearman": _spearman(iclr_scores, attack_scores),
        "class_adjusted_spearman": adjusted,
        "class_macro_spearman": macro,
        "class_adjusted_classes": adjusted_classes,
        "top_fraction": float(top_fraction),
        "top_count": top_count,
        "attack_score_mean_iclr_top": _safe_mean(attack_scores[top]),
        "attack_score_mean_iclr_bottom": _safe_mean(attack_scores[bottom]),
        "attack_score_top_minus_bottom": (
            _safe_mean(attack_scores[top]) - _safe_mean(attack_scores[bottom])
        ),
        "top_set_overlap": overlap / top_count,
        "top_set_enrichment": (
            overlap / expected_overlap if expected_overlap > 0 else None
        ),
    }
    for target in (0.1, 0.01, 0.001):
        suffix = f"{target:g}"
        hits = _low_fpr_hits(attack_scores, nonmember_scores, target)
        if hits is None:
            row[f"attack_hit_rate_fpr_{suffix}"] = None
            row[f"attack_hit_rate_iclr_top_fpr_{suffix}"] = None
            row[f"attack_hit_rate_iclr_bottom_fpr_{suffix}"] = None
            row[f"attack_hit_top_minus_bottom_fpr_{suffix}"] = None
            row[f"attack_hit_top_over_bottom_fpr_{suffix}"] = None
            continue
        overall = _safe_mean(hits)
        top_rate = _safe_mean(hits[top])
        bottom_rate = _safe_mean(hits[bottom])
        row[f"attack_hit_rate_fpr_{suffix}"] = overall
        row[f"attack_hit_rate_iclr_top_fpr_{suffix}"] = top_rate
        row[f"attack_hit_rate_iclr_bottom_fpr_{suffix}"] = bottom_rate
        row[f"attack_hit_top_minus_bottom_fpr_{suffix}"] = top_rate - bottom_rate
        row[f"attack_hit_top_over_bottom_fpr_{suffix}"] = (
            top_rate / bottom_rate if bottom_rate > 0 else None
        )
    return row


def validate_iclr_attack_relationships(
    results: list[AttackResult],
    users: list,
    candidate_labels: torch.Tensor,
    candidate_membership: torch.Tensor,
    candidate_client_ids: torch.Tensor,
    candidate_local_indices: torch.Tensor | None,
    output_dir: str,
    top_fraction: float = 0.2,
) -> dict:
    """Join ICLR scores to attack outputs and write reproducible diagnostics."""
    if not 0.0 < top_fraction <= 0.5:
        raise ValueError("ICLR validation top_fraction must be in (0, 0.5].")
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "iclr_attack_relationship.json")
    if candidate_local_indices is None:
        summary = {
            "status": "unavailable",
            "reason": (
                "Candidate local indices are unavailable. Use "
                "audit.candidate_sampling=low_fpr_full."
            ),
        }
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, allow_nan=False)
        return summary

    labels = candidate_labels.detach().cpu().long()
    membership = candidate_membership.detach().cpu().long()
    client_ids = candidate_client_ids.detach().cpu().long()
    local_indices = candidate_local_indices.detach().cpu().long()
    expected = labels.numel()
    if not (
        membership.numel()
        == client_ids.numel()
        == local_indices.numel()
        == expected
    ):
        raise ValueError("ICLR validation candidate tensors are not aligned.")

    user_statistics = {
        int(user.id): _user_score_statistics(user) for user in users
    }
    metric_rows = []
    sample_rows = []
    for result in results:
        result_indices = result.sample_indices.detach().cpu().long()
        result_labels = result.labels.detach().cpu().long()
        result_scores = result.scores.detach().cpu().to(torch.float64)
        if not (
            result_indices.numel()
            == result_labels.numel()
            == result_scores.numel()
        ):
            raise ValueError(f"Attack {result.name} returned misaligned tensors.")
        if result_indices.numel() and (
            int(result_indices.min()) < 0 or int(result_indices.max()) >= expected
        ):
            raise IndexError(f"Attack {result.name} returned an invalid sample index.")
        if not torch.equal(result_labels, membership[result_indices]):
            raise ValueError(
                f"Attack {result.name} labels disagree with candidate membership."
            )

        result_clients = client_ids[result_indices]
        for client_id in torch.unique(result_clients).tolist():
            client_mask = result_clients == int(client_id)
            member_mask = client_mask & (result_labels == 1)
            nonmember_mask = client_mask & (result_labels == 0)
            attack_member_count = int(member_mask.sum())
            if attack_member_count == 0:
                continue
            statistics = user_statistics.get(int(client_id))
            if statistics is None:
                continue
            member_result_indices = result_indices[member_mask]
            member_local_indices = local_indices[member_result_indices]
            valid_local = (
                (member_local_indices >= 0)
                & (member_local_indices < statistics["count"].numel())
            )
            covered = torch.zeros_like(valid_local)
            covered[valid_local] = (
                statistics["count"][member_local_indices[valid_local]] > 0
            )
            aligned_mask = valid_local & covered
            if int(aligned_mask.sum()) < 2:
                continue
            aligned_candidate_indices = member_result_indices[aligned_mask]
            aligned_local_indices = member_local_indices[aligned_mask]
            aligned_attack_scores = result_scores[member_mask][aligned_mask]
            aligned_class_labels = labels[aligned_candidate_indices]
            observation_counts = statistics["count"][aligned_local_indices]
            nonmember_scores = result_scores[nonmember_mask]

            for position in range(aligned_local_indices.numel()):
                local_index = int(aligned_local_indices[position])
                sample_rows.append(
                    {
                        "attack": result.name,
                        "sample_index": int(aligned_candidate_indices[position]),
                        "audit_client_id": int(client_id),
                        "local_sample_index": local_index,
                        "class_label": int(aligned_class_labels[position]),
                        "membership": 1,
                        "attack_score": float(aligned_attack_scores[position]),
                        "iclr_observations": int(statistics["count"][local_index]),
                        "iclr_mean": float(statistics["mean"][local_index]),
                        "iclr_std": float(statistics["std"][local_index]),
                        "iclr_min": float(statistics["min"][local_index]),
                        "iclr_max": float(statistics["max"][local_index]),
                        "iclr_last": float(statistics["last"][local_index]),
                        "iclr_last_round": int(
                            statistics["last_round"][local_index]
                        ),
                    }
                )

            for aggregation in ("mean", "last", "max"):
                iclr_scores = statistics[aggregation][aligned_local_indices]
                finite = torch.isfinite(iclr_scores) & torch.isfinite(
                    aligned_attack_scores
                )
                if int(finite.sum()) < 2:
                    continue
                metric_rows.append(
                    _relationship_metrics(
                        attack=result.name,
                        client_id=int(client_id),
                        aggregation=aggregation,
                        iclr_scores=iclr_scores[finite],
                        attack_scores=aligned_attack_scores[finite],
                        class_labels=aligned_class_labels[finite],
                        observation_counts=observation_counts[finite],
                        nonmember_scores=nonmember_scores[
                            torch.isfinite(nonmember_scores)
                        ],
                        attack_member_count=attack_member_count,
                        top_fraction=top_fraction,
                    )
                )

    sample_path = os.path.join(output_dir, "iclr_attack_samples.csv")
    sample_fields = (
        list(sample_rows[0])
        if sample_rows
        else [
            "attack",
            "sample_index",
            "audit_client_id",
            "local_sample_index",
            "class_label",
            "membership",
            "attack_score",
            "iclr_observations",
            "iclr_mean",
            "iclr_std",
            "iclr_min",
            "iclr_max",
            "iclr_last",
            "iclr_last_round",
        ]
    )
    with open(sample_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample_rows)

    metric_path = os.path.join(output_dir, "iclr_attack_relationship.csv")
    leading_fields = [
        "scope",
        "attack",
        "audit_client_id",
        "iclr_aggregation",
    ]
    extra_fields = sorted(
        {key for row in metric_rows for key in row} - set(leading_fields)
    )
    with open(metric_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=[*leading_fields, *extra_fields]
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    attacks = sorted({row["attack"] for row in metric_rows})
    clients = sorted({int(row["audit_client_id"]) for row in metric_rows})
    summary = {
        "status": "ok" if metric_rows else "no_aligned_samples",
        "methodology": {
            "population": "strictly aligned audited member samples",
            "attack_score_direction": "higher means more member-like",
            "iclr_score": "L(x; theta_-k) - L(x; theta_k)",
            "iclr_aggregations": ["mean", "last", "max"],
            "top_fraction": top_fraction,
            "class_control": (
                "within-class rank centering plus macro per-class Spearman"
            ),
            "low_fpr_hit_rule": (
                "member score is a hit when the number of nonmember scores at "
                "least as large does not exceed floor(target_fpr * N_nonmember)"
            ),
            "interpretation": (
                "association and enrichment measure specificity alignment; "
                "they do not establish causality or defense effectiveness"
            ),
        },
        "attacks": attacks,
        "audit_client_ids": clients,
        "relationship_rows": len(metric_rows),
        "sample_rows": len(sample_rows),
        "artifacts": {
            "relationships": os.path.basename(metric_path),
            "samples": os.path.basename(sample_path),
        },
        "relationships": metric_rows,
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    return summary
