"""Human-readable terminal summaries; source metrics and artifacts stay raw."""
from __future__ import annotations

import math
from typing import Any


def format_number(value: Any, *, percent: bool = False, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{100 * number:.2f}%" if percent else f"{number:.{digits}f}"


def format_table(headers: list[str], rows: list[list[str]], *, numeric=()) -> str:
    """ASCII borders work in redirected logs without terminal/color dependencies."""
    cells = [[" ".join(str(value).split()) for value in row] for row in rows]
    widths = [max([len(label), *(len(row[i]) for row in cells)])
              for i, label in enumerate(headers)]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def line(row):
        return "| " + " | ".join(
            value.rjust(width) if i in numeric else value.ljust(width)
            for i, (value, width) in enumerate(zip(row, widths))
        ) + " |"

    return "\n".join([border, line(headers), border, *(line(row) for row in cells), border])


def reportable_metric(summary: dict, key: str):
    availability = summary.get("metric_availability", {}).get(key, {})
    if availability.get("resolvable") is False:
        return None
    reportable = summary.get("reportable_metrics")
    if isinstance(reportable, dict):
        # An unavailable reportable score must never fall back to a raw TPR.
        return reportable.get(key)
    if key.startswith("tpr_at_fpr_") and summary.get("nonmember_count") is not None:
        minimum = math.ceil(1 / float(key.removeprefix("tpr_at_fpr_")))
        if int(summary["nonmember_count"]) < minimum:
            return None
    if key == summary.get("primary_metric") and "primary_score" in summary:
        return summary["primary_score"]
    return summary.get(key)


def format_attack_table(summaries: list[dict]) -> str:
    if not summaries:
        return "No reported attack results."
    rows = []
    for summary in summaries:
        name = str(summary.get("attack", "unknown"))
        if summary.get("score_degenerate"):
            name += "*"
        rows.append([
            name,
            format_number(reportable_metric(summary, "auc")),
            *(format_number(reportable_metric(summary, f"tpr_at_fpr_{fpr}"), percent=True)
              for fpr in ("0.01", "0.001", "0.1")),
            str(summary.get("member_count", "N/A")),
            str(summary.get("nonmember_count", "N/A")),
        ])
    result = format_table(
        ["Attack", "AUC", "TPR@1%FPR", "TPR@0.1%FPR", "TPR@10%FPR", "Members", "Nonmembers"],
        rows, numeric=range(1, 7),
    )
    result += "\nTPR@1%FPR: primary attack metric. N/A: unavailable or insufficient nonmembers."
    if any(summary.get("score_degenerate") for summary in summaries):
        result += "\n* Constant scores; see the audit metadata."
    return result


def format_run_summary(
    *, model: str, dataset: str, method: str, metrics: dict,
    total_rounds: int, defense: dict, attacks: list[dict], results_dir: str,
    audit_enabled: bool = True, audit_errors: dict | None = None,
) -> str:
    """A bounded display, independent of per-client/state dictionary sizes."""
    lines = [
        "=" * 88,
        "RUN RESULTS",
        f"Model: {model}  |  Dataset: {dataset}  |  Defense: {defense.get('defense', 'none')}  |  Method: {method}",
        "", "Task metrics (final evaluation)",
        format_table(
            ["Round", "Loss", "Accuracy", "MCC", "Learning rate"],
            [[f"{metrics.get('round', 'N/A')}/{total_rounds}",
              format_number(metrics.get("loss")),
              format_number(metrics.get("accuracy"), percent=True),
              format_number(metrics.get("mcc")),
              format_number(metrics.get("learning_rate"), digits=6)]],
            numeric=range(5),
        ),
    ]
    accounting = defense.get("privacy_accounting")
    if isinstance(accounting, dict):
        delta = accounting.get("delta")
        try:
            delta_text = f"{float(delta):.2g}" if math.isfinite(float(delta)) else "N/A"
        except (TypeError, ValueError):
            delta_text = "N/A"
        lines.extend([
            "", f"Privacy accounting (unit: {accounting.get('privacy_unit', 'N/A')})",
            format_table(
                ["Epsilon (max)", "Target epsilon", "Delta", "Clip norm", "Noise mult.", "Formal DP"],
                [[format_number(accounting.get("epsilon_upper_bound")),
                  format_number(accounting.get("target_epsilon")), delta_text,
                  format_number(accounting.get("max_grad_norm")),
                  format_number(accounting.get("noise_multiplier")),
                  {True: "yes", False: "no"}.get(accounting.get("formal_dp_enabled"), "N/A")]],
                numeric=range(5),
            ),
        ])
    lines.extend(["", "Membership attacks", format_attack_table(attacks) if audit_enabled else "Disabled."])
    if audit_errors:
        names = sorted(audit_errors)
        lines.append(f"Audit errors: {len(names)} ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''}); see summary.json.")
    lines.extend([
        "", f"Results: {results_dir}",
        "  Task: training_metrics.csv  |  Defense: defense_summary.json",
    ])
    if audit_enabled:
        lines.append("  Attacks: privacy_audit/summary.json  |  Predictions: privacy_audit/predictions.csv")
    lines.append("=" * 88)
    return "\n".join(lines)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "N/A"
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_sweep_summary(records: list[dict]) -> str:
    rows = []
    for index, record in enumerate(records, 1):
        metric = record.get("metric")
        rows.append([
            str(index), record["model"], record["dataset"], record["defense"],
            f"{record['seed']}/{record['target_client_id']}", record["status"],
            {"accuracy": "Accuracy", "test_accuracy": "Accuracy", "mcc": "MCC"}.get(metric, metric or "N/A"),
            format_number(record.get("value"), percent=metric in {"accuracy", "test_accuracy"}),
            format_duration(record.get("seconds")),
        ])
    return "\n".join([
        "EXPERIMENT OVERVIEW",
        format_table(["#", "Model", "Dataset", "Defense", "Seed/Client", "Status", "Metric", "Final value", "Elapsed"],
                     rows, numeric=(0, 4, 7, 8)),
        "Final value: task utility. FAILED/PARTIAL rows may contain incomplete results.",
    ])
