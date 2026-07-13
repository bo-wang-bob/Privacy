import csv
import logging
import math
import os
from typing import Optional

import torch

from context.context import Context

logger = logging.getLogger(__name__)

RAW_TOP1_EPS = 1e-12
ROBUST_MAD_SCALE = 1.4826


def _median_float(values: list[float]) -> float:
    sorted_values = sorted(float(value) for value in values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return 0.5 * (sorted_values[middle - 1] + sorted_values[middle])


def _format_optional_float(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.10g}"


def _collect_text_features(
    ctx: Context,
    user_ids: list[int],
) -> dict[int, torch.Tensor]:
    cached_features = getattr(ctx, "text_feature_dict", {})
    collected = {}
    for user_id in user_ids:
        text_features = cached_features.get(user_id)
        if text_features is None:
            try:
                text_features = ctx.users[user_id].get_text_features(normalize=True)
            except Exception:
                logger.exception(
                    "SeismographAggregator: failed to collect text features for user %s "
                    "in round %s.",
                    user_id,
                    ctx.glob_iter,
                )
                continue
        collected[user_id] = text_features.detach()
    return collected


def collect_text_feature_raw_top1_singular_values(
    ctx: Context,
    user_ids: list[int],
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    """Collect the raw top-1 singular value used by the seismograph detector."""
    if not ctx.fpl:
        raise ValueError("SeismographAggregator requires federated prompt learning.")
    if not user_ids:
        logger.warning(
            "SeismographAggregator: no users selected for text-feature analysis."
        )
        return [], torch.empty(0, device=device, dtype=torch.float32)

    text_feature_dict = _collect_text_features(ctx, user_ids)
    valid_user_ids = []
    raw_top1_values = []
    first_shape = None
    for user_id in user_ids:
        text_features = text_feature_dict.get(user_id)
        if text_features is None or not torch.is_tensor(text_features):
            continue
        if text_features.dim() != 2:
            logger.warning(
                "SeismographAggregator: user %s has invalid text-feature shape %s.",
                user_id,
                tuple(text_features.shape),
            )
            continue
        if text_features.shape[0] != ctx.num_classes:
            logger.warning(
                "SeismographAggregator: user %s text-feature class count mismatch: "
                "expected=%s, got=%s.",
                user_id,
                ctx.num_classes,
                text_features.shape[0],
            )
            continue
        if first_shape is None:
            first_shape = tuple(text_features.shape)
        elif tuple(text_features.shape) != first_shape:
            logger.warning(
                "SeismographAggregator: user %s text-feature shape %s differs from %s.",
                user_id,
                tuple(text_features.shape),
                first_shape,
            )
            continue

        features = text_features.to(device=device, dtype=torch.float32)
        if not bool(torch.isfinite(features).all().item()):
            logger.warning(
                "SeismographAggregator: user %s text features contain non-finite values.",
                user_id,
            )
            continue
        try:
            singular_values = torch.linalg.svdvals(features)
        except RuntimeError:
            logger.exception(
                "SeismographAggregator: SVD failed for user %s in round %s.",
                user_id,
                ctx.glob_iter,
            )
            continue
        if singular_values.numel() == 0:
            continue
        valid_user_ids.append(user_id)
        raw_top1_values.append(singular_values[0])

    if not raw_top1_values:
        return [], torch.empty(0, device=device, dtype=torch.float32)

    values = torch.stack(raw_top1_values).to(device=device, dtype=torch.float32)
    logger.info(
        "SeismographAggregator: round %s raw top-1 singular values: %s",
        ctx.glob_iter,
        [
            (user_id, round(float(value), 10))
            for user_id, value in zip(
                valid_user_ids,
                values.detach().cpu().tolist(),
            )
        ],
    )
    return valid_user_ids, values


def _save_raw_top1_history_scores(ctx: Context, rows: list[dict]) -> None:
    csv_path = os.path.join(
        ctx.results_dir,
        "text_feature_raw_top1_history",
        f"round_{ctx.glob_iter}.csv",
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "round",
        "user_id",
        "raw_singular_value_top1",
        "log_raw_top1",
        "history_count",
        "history_median",
        "baseline",
        "baseline_type",
        "delta",
        "score_median",
        "score_scale",
        "robust_z",
        "threshold",
        "seismograph_score",
        "seismograph_alarmed",
        "is_suspicious",
        "is_excluded",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "round": ctx.glob_iter,
                    "user_id": row["user_id"],
                    "raw_singular_value_top1": f"{row['raw_singular_value_top1']:.10g}",
                    "log_raw_top1": f"{row['log_raw_top1']:.10g}",
                    "history_count": row["history_count"],
                    "history_median": _format_optional_float(
                        row.get("history_median")
                    ),
                    "baseline": f"{row['baseline']:.10g}",
                    "baseline_type": row["baseline_type"],
                    "delta": f"{row['delta']:.10g}",
                    "score_median": f"{row['score_median']:.10g}",
                    "score_scale": f"{row['score_scale']:.10g}",
                    "robust_z": f"{row['robust_z']:.10g}",
                    "threshold": f"{row['threshold']:.10g}",
                    "seismograph_score": f"{row['seismograph_score']:.10g}",
                    "seismograph_alarmed": row["seismograph_alarmed"],
                    "is_suspicious": row["is_suspicious"],
                    "is_excluded": row["is_excluded"],
                }
            )


def filter_users_by_raw_top1_svd_history(
    ctx: Context,
    selected_ids: list[int],
    valid_user_ids: list[int],
    raw_top1_values: torch.Tensor,
    device: torch.device,
    *,
    raw_top1_log_history: dict[int, list[float]],
    raw_top1_seismograph_state: dict[int, float],
    seismograph_k: float = 1.0,
    seismograph_h: float = 5.0,
) -> list[int]:
    """Apply the SEISMOGRAPH historical robust-z client filter."""
    if not valid_user_ids or raw_top1_values.numel() == 0:
        return selected_ids

    values = raw_top1_values.detach().to(device=device, dtype=torch.float32)
    finite_mask = torch.isfinite(values)
    if not bool(finite_mask.any().item()):
        logger.warning(
            "SeismographAggregator: no finite raw top-1 values in round %s; using all users.",
            ctx.glob_iter,
        )
        return selected_ids

    finite_user_ids = [
        user_id
        for user_id, is_finite in zip(
            valid_user_ids,
            finite_mask.detach().cpu().tolist(),
        )
        if is_finite
    ]
    finite_values = values[finite_mask]
    log_values = torch.log(finite_values.clamp_min(0.0) + RAW_TOP1_EPS)
    if finite_values.numel() < 2:
        for user_id, log_value in zip(
            finite_user_ids,
            log_values.detach().cpu().tolist(),
        ):
            raw_top1_log_history.setdefault(user_id, []).append(float(log_value))
            raw_top1_seismograph_state.setdefault(user_id, 0.0)
        return selected_ids

    current_log_median = float(log_values.median().detach().cpu().item())
    score_values = []
    raw_rows = []
    for user_id, raw_value, log_value in zip(
        finite_user_ids,
        finite_values.detach().cpu().tolist(),
        log_values.detach().cpu().tolist(),
    ):
        history = raw_top1_log_history.get(user_id, [])
        history_median = _median_float(history) if history else None
        baseline = history_median if history else current_log_median
        baseline_type = "history_median" if history else "round_median_fallback"
        delta = float(log_value) - float(baseline)
        score_values.append(delta)
        raw_rows.append(
            {
                "user_id": user_id,
                "raw_singular_value_top1": float(raw_value),
                "log_raw_top1": float(log_value),
                "history_count": len(history),
                "history_median": history_median,
                "baseline": float(baseline),
                "baseline_type": baseline_type,
                "delta": delta,
            }
        )

    score_tensor = torch.tensor(score_values, device=device, dtype=torch.float32)
    score_median_tensor = score_tensor.median()
    absolute_deviation = (score_tensor - score_median_tensor).abs()
    score_scale_tensor = ROBUST_MAD_SCALE * absolute_deviation.median()
    if float(score_scale_tensor.detach().cpu().item()) <= RAW_TOP1_EPS:
        score_scale_tensor = absolute_deviation.mean()

    threshold = math.sqrt(2.0 * math.log(max(2, len(finite_user_ids))))
    if float(score_scale_tensor.detach().cpu().item()) <= RAW_TOP1_EPS:
        robust_z = torch.zeros_like(score_tensor)
    else:
        robust_z = (score_tensor - score_median_tensor) / score_scale_tensor
    z_values = robust_z.detach().cpu().tolist()

    suspicious_ids = {
        row["user_id"]
        for row, z_value in zip(raw_rows, z_values)
        if float(z_value) > threshold
    }
    seismograph_scores = {}
    for row, z_value in zip(raw_rows, z_values):
        user_id = row["user_id"]
        score = max(
            0.0,
            raw_top1_seismograph_state.get(user_id, 0.0)
            + float(z_value)
            - seismograph_k,
        )
        raw_top1_seismograph_state[user_id] = score
        seismograph_scores[user_id] = score

    alarmed_ids = {
        user_id
        for user_id, score in seismograph_scores.items()
        if score > seismograph_h
    }
    excluded_ids = set(alarmed_ids)
    aggregation_ids = [
        user_id for user_id in selected_ids if user_id not in excluded_ids
    ]
    if not aggregation_ids:
        logger.warning(
            "SeismographAggregator: filtering removed every selected user in round %s; "
            "falling back to all selected users.",
            ctx.glob_iter,
        )
        excluded_ids = set()
        aggregation_ids = selected_ids

    score_median = float(score_median_tensor.detach().cpu().item())
    score_scale = float(score_scale_tensor.detach().cpu().item())
    rows = []
    for row, z_value in zip(raw_rows, z_values):
        output_row = dict(row)
        user_id = row["user_id"]
        output_row.update(
            {
                "score_median": score_median,
                "score_scale": score_scale,
                "robust_z": float(z_value),
                "threshold": threshold,
                "seismograph_score": seismograph_scores.get(user_id, 0.0),
                "seismograph_alarmed": user_id in alarmed_ids,
                "is_suspicious": user_id in suspicious_ids,
                "is_excluded": user_id in excluded_ids,
            }
        )
        rows.append(output_row)
    _save_raw_top1_history_scores(ctx, rows)

    for row in rows:
        if not row["seismograph_alarmed"]:
            raw_top1_log_history.setdefault(row["user_id"], []).append(
                float(row["log_raw_top1"])
            )

    logger.info(
        "SEISMOGRAPH: round %s k=%.4g h=%.4g, excluded=%s, aggregate=%s",
        ctx.glob_iter,
        seismograph_k,
        seismograph_h,
        sorted(excluded_ids),
        aggregation_ids,
    )
    return aggregation_ids
