"""Stream WWW risk-loss diagnostics; schema v2 never implies gradient clipping."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata


NORM_FIELDS = ("raw_grad_norm", "regularizer_grad_norm", "total_grad_norm")
LOSS_FIELDS = (
    "ce_loss", "current_probability", "reference_probability", "confidence_gap",
    "cross_difference", "regularization_loss", "additional_loss", "total_loss",
    "ce_gradient_factor",
)
SAMPLE_FIELDS = (
    "communication_round", "client_id", "client_step", "local_sample_index",
    "label", "batch_position", "batch_size", "configured_batch_size",
    "risk_available", "risk_source_round", "risk_score", "risk_rank",
    "risk_percentile", "own_loss", "other_loss", "group", "tail_selected",
    "regularization_weight", "risk_weight", *LOSS_FIELDS,
    *NORM_FIELDS, "normalized_contribution_norm",
)
BATCH_FIELDS = (
    "communication_round", "client_id", "client_step", "batch_size",
    "configured_batch_size", "risk_available", "risk_source_round", "group",
    "sample_count", "regularization_weight", "mean_risk_weight", "ce_direction_reversed_count",
    "risk_min", "risk_mean", "risk_max",
    *(f"{name}_mean" for name in LOSS_FIELDS),
    *(f"{name}_{stat}" for name in NORM_FIELDS for stat in ("mean", "median", "p90", "p99", "max")),
    *(f"risk_{method}_{name}" for name in NORM_FIELDS for method in ("pearson", "spearman")),
)


def correlation(left, right):
    """Require at least three nonconstant paired observations; undefined is null."""
    if len(left) < 3:
        return None
    x, y = left - left.mean(), right - right.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(x @ y / denominator, -1, 1)) if denominator > 0 else None


class WWWGradientRecorder:
    """One row per actual training visit; files flush after every client batch.

    Correlations and quantiles are computed within client/batch/group, avoiding
    an implicit mix of different client models or rounds. Memory is O(batch),
    independent of training duration. Existing result artifacts are not reused.
    """

    def __init__(self, results_dir):
        self.directory = Path(results_dir) / "www_diagnostics"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.sample_rows = self.batch_count = self.summary_rows = 0
        self.ranked_rows = self.low_risk_rows = self.reversed_rows = 0
        self.closed = False
        self._samples = (self.directory / "sample_gradients.csv").open("x", newline="", encoding="utf-8")
        try:
            self._batches = (self.directory / "batch_summary.csv").open("x", newline="", encoding="utf-8")
        except BaseException:
            self._samples.close()
            raise
        self._sample_writer = csv.DictWriter(self._samples, fieldnames=SAMPLE_FIELDS)
        self._batch_writer = csv.DictWriter(self._batches, fieldnames=BATCH_FIELDS)
        self._sample_writer.writeheader()
        self._batch_writer.writeheader()
        self._samples.flush()
        self._batches.flush()

    def record(self, *, user, ranking, diagnostics, weights, tail,
               round_index, client_step, source_round, has_reference, regularization_weight):
        if self.closed:
            raise RuntimeError("WWW gradient diagnostics are already closed.")
        n = ranking.labels.numel()
        names = (*NORM_FIELDS, *LOSS_FIELDS)
        if any(diagnostics[name].shape != (n,) for name in names):
            raise ValueError("WWW gradient diagnostics must align with ranked samples.")
        # One small device-to-host transfer per real batch, never per record or
        # parameter. Full per-record gradients remain bounded by compute chunks.
        values = torch.stack([diagnostics[name].detach().double() for name in names]).cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("WWW gradient diagnostics contain non-finite values.")
        data = dict(zip(names, values))
        scores = ranking.scores.detach().cpu().double().numpy()
        risk_weights = weights.detach().cpu().double().numpy()
        tail_values = tail.detach().cpu().numpy()
        ranks = np.empty(n, dtype=np.int64)
        ranks[ranking.ranked_positions.numpy()] = np.arange(1, n + 1)
        groups = np.where(tail_values, "high_risk", "low_risk") if has_reference else np.full(n, "warmup")
        metadata = {
            "communication_round": int(round_index) + 1,
            "client_id": int(user.id), "client_step": int(client_step),
            "batch_size": n, "configured_batch_size": int(user.batch_size),
            "risk_available": int(has_reference),
            "risk_source_round": int(source_round) + 1 if has_reference else None,
            "regularization_weight": float(regularization_weight),
        }
        reversed_ce = data["ce_gradient_factor"] < 0
        reference_fields = {"reference_probability", "confidence_gap", "cross_difference"}
        for i in range(n):
            self._sample_writer.writerow({
                **metadata,
                "local_sample_index": int(ranking.sample_indices[i]),
                "label": int(ranking.labels[i]), "batch_position": i,
                "risk_score": float(scores[i]) if has_reference else None,
                "risk_rank": int(ranks[i]) if has_reference else None,
                "risk_percentile": float((ranks[i] - .5) / n) if has_reference else None,
                "own_loss": float(ranking.own_losses[i]) if has_reference else None,
                "other_loss": float(ranking.other_losses[i]) if has_reference else None,
                "group": str(groups[i]), "tail_selected": int(tail_values[i]),
                **{name: (None if not has_reference and name in reference_fields else float(data[name][i]))
                   for name in names},
                "normalized_contribution_norm": float(data["total_grad_norm"][i] / n),
                "risk_weight": float(risk_weights[i]),
            })
        # "all" is a summary view, not additional sample observations.
        for group in ("all", *(str(g) for g in np.unique(groups))):
            mask = np.ones(n, dtype=bool) if group == "all" else groups == group
            count = int(mask.sum())
            risk = scores[mask]
            row = {
                **metadata, "group": group, "sample_count": count,
                "ce_direction_reversed_count": int(reversed_ce[mask].sum()),
                "mean_risk_weight": float(risk_weights[mask].mean()) if count else None,
                **{f"{name}_mean": (None if not has_reference and name in reference_fields
                                     else float(data[name][mask].mean())) for name in LOSS_FIELDS},
                "risk_min": float(risk.min()) if count and has_reference else None,
                "risk_mean": float(risk.mean()) if count and has_reference else None,
                "risk_max": float(risk.max()) if count and has_reference else None,
            }
            risk_ranks = rankdata(risk) if count and has_reference else None
            for name in NORM_FIELDS:
                norms = data[name][mask]
                stats = (float(norms.mean()), *np.quantile(norms, [.5, .9, .99]), float(norms.max())) if count else [None] * 5
                row.update({f"{name}_{key}": None if value is None else float(value)
                            for key, value in zip(("mean", "median", "p90", "p99", "max"), stats)})
                row[f"risk_pearson_{name}"] = correlation(risk, norms) if has_reference else None
                row[f"risk_spearman_{name}"] = correlation(risk_ranks, rankdata(norms)) if count and has_reference else None
            self._batch_writer.writerow(row)
            self.summary_rows += 1
        self._samples.flush()
        self._batches.flush()
        self.sample_rows += n
        self.batch_count += 1
        self.ranked_rows += n if has_reference else 0
        self.low_risk_rows += int((groups == "low_risk").sum())
        self.reversed_rows += int(reversed_ce.sum())

    def summary(self):
        return {
            "enabled": True, "schema_version": 2, "mechanism": "risk_controlled_loss",
            "sample_rows": self.sample_rows,
            "client_batches": self.batch_count, "batch_summary_rows": self.summary_rows,
            "risk_available_rows": self.ranked_rows, "low_risk_ce_only_rows": self.low_risk_rows,
            "ce_direction_reversed_rows": self.reversed_rows,
            "files": {"samples": "www_diagnostics/sample_gradients.csv",
                      "batch_summary": "www_diagnostics/batch_summary.csv",
                      "summary": "www_diagnostics/summary.json"},
            "sample_identity": "(client_id, local_sample_index); each row is one training visit",
            "round_indexing": "communication_round, risk_source_round and client_step are one-based; batch_position is zero-based",
            "norm_scope": "Joint L2 over all trainable parameters, before batch averaging; frozen parameters excluded",
            "clipping_enabled": False,
            "loss": "CE + lambda * risk_weight * abs(p_y - stopgrad(q_y)); averaged over actual batch",
            "gradient_norms": {
                "raw_grad_norm": "Gradient of current CE only, measured on the actual student forward graph",
                "regularizer_grad_norm": "Gradient of lambda*r*abs(p_y-q_y), derived from its exact CE multiplier",
                "total_grad_norm": "Gradient of complete per-record training loss, including optional additional losses",
                "ce_gradient_factor": "Signed 1-lambda*r*p_y*sign(p_y-q_y); applies to CE+regularizer, before any additional loss",
            },
            "teacher": "Frozen previous-round other-client parameter aggregate; not an exact leave-one-out model or mean of peer probabilities",
            "risk": "Previous-round loss(other model) - loss(own model); not a privacy probability",
            "correlations": "Pearson and tie-averaged Spearman within each client/batch/group; warmup, fewer than 3 samples or constant vectors are null",
            "quantiles": "Exact per-client/batch/group linear-interpolated quantiles; not pooled across rounds",
            "missing_values": "Empty CSV fields / JSON null mean unavailable or not applicable, never zero",
            "formal_dp_enabled": False,
        }

    def close(self, status):
        if self.closed:
            return
        self.closed = True
        try:
            self._samples.close()
        finally:
            self._batches.close()
        payload = {**self.summary(), "status": status}
        (self.directory / "summary.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
