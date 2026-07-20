#!/usr/bin/env python3
"""Replay and visualize a saved FedMIA attack trajectory.

The launcher log is used to identify the exact sweep job.  The actual attack
replay uses that job's ``privacy_audit/signals.pt`` because the text log only
contains collection milestones and final metrics, not the per-round signals.

Example
-------
MPLCONFIGDIR=/tmp/matplotlib-fedmia-replay \
python analysis_scripts/replay_fedmia_attack.py \
  results/fedmia_prompt_methods_fewshot/launcher_logs/\
cifar100_promptfl_none_seed42_target0_b9cda3ae45.log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fedmia-replay")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from privacy_attacks.fedmia import run_fedmia  # noqa: E402
from privacy_attacks.metrics import membership_metrics  # noqa: E402


ATTACKS = {
    "fedmia_loss": "confidence",
    "fedmia_cosine": "cosine",
}
COLORS = {
    "fedmia_loss": "#2563EB",
    "fedmia_cosine": "#D97706",
}
DISPLAY_NAMES = {
    "fedmia_loss": "FedMIA loss",
    "fedmia_cosine": "FedMIA cosine",
}
MEMBERSHIP_COLORS = {
    1: "#2563EB",
    0: "#D97706",
}
MEMBERSHIP_NAMES = {
    1: "Member",
    0: "Non-member",
}


@dataclass
class ReplayResult:
    attack: str
    scores: torch.Tensor
    labels: torch.Tensor
    sample_indices: torch.Tensor
    client_ids: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay pooled-client FedMIA from the signals saved by a launcher log."
        )
    )
    parser.add_argument(
        "launcher_log",
        type=Path,
        help="Path to one launcher_logs/<job-id>.log file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Defaults to <resolved-run-dir>/privacy_audit/replay."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution (default: 180).",
    )
    return parser.parse_args()


def _launcher_config_path(launcher_log: Path) -> Path | None:
    first_line = launcher_log.read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("COMMAND "):
        return None
    command = shlex.split(first_line.removeprefix("COMMAND "))
    try:
        index = command.index("--config")
    except ValueError:
        return None
    return Path(command[index + 1]).resolve()


def resolve_run_dir(launcher_log: Path) -> tuple[Path, Path]:
    """Resolve the generated config and timestamped result directory."""
    launcher_log = launcher_log.resolve()
    if not launcher_log.is_file():
        raise FileNotFoundError(f"Launcher log does not exist: {launcher_log}")

    sweep_root = launcher_log.parent.parent
    state_path = sweep_root / "sweep_state.json"
    job_id = launcher_log.stem
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        run_state = state.get("runs", {}).get(job_id, {})
        result_dir = run_state.get("result_dir")
        if result_dir:
            config_path = _launcher_config_path(launcher_log)
            if config_path is None:
                config_path = sweep_root / "configs" / f"{job_id}.yaml"
            return config_path.resolve(), Path(result_dir).resolve()

    config_path = _launcher_config_path(launcher_log)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(
            "Could not resolve the generated config from the launcher log."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_root = Path(config["results_dir"])
    candidates = sorted(
        path.parent.parent
        for path in result_root.glob("*/privacy_audit/signals.pt")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one completed run below {result_root}, found {len(candidates)}. "
            "Keep sweep_state.json or pass a launcher log for an unambiguous job."
        )
    return config_path.resolve(), candidates[0].resolve()


def launcher_rounds(launcher_log: Path) -> list[int]:
    pattern = re.compile(r"Collected privacy signals for round (\d+)")
    return [
        int(match.group(1))
        for match in pattern.finditer(launcher_log.read_text(encoding="utf-8"))
    ]


def load_inputs(
    launcher_log: Path,
) -> tuple[dict, dict, Path, Path]:
    config_path, run_dir = resolve_run_dir(launcher_log)
    if not config_path.is_file():
        raise FileNotFoundError(f"Generated config does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    signals_path = run_dir / "privacy_audit" / "signals.pt"
    if not signals_path.is_file():
        raise FileNotFoundError(
            f"The run has no completed signal archive yet: {signals_path}"
        )
    # This is a tensor-and-primitive payload written by MembershipAuditor.
    payload = torch.load(signals_path, map_location="cpu", weights_only=True)
    required = {
        "candidate_labels",
        "candidate_client_ids",
        "membership",
        "observations",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(f"signals.pt is missing fields: {sorted(missing)}")
    return config, payload, config_path, run_dir


def _usable_clients(
    observations: list[dict], candidate_client_ids: torch.Tensor
) -> list[int]:
    candidates = sorted(int(value) for value in torch.unique(candidate_client_ids))
    return [
        client_id
        for client_id in candidates
        if any(
            client_id in observation["client_ids"].tolist()
            and observation["client_ids"].numel() >= 2
            for observation in observations
        )
    ]


def replay_pooled(
    observations: list[dict],
    payload: dict,
    attack: str,
    aggregation: str,
    tail: str,
    calibration_fraction: float,
    seed: int,
) -> ReplayResult:
    """Replay MembershipAuditor._run_fedmia_signal for pooled clients."""
    measurement = ATTACKS[attack]
    membership = payload["membership"].detach().cpu().long()
    candidate_client_ids = payload["candidate_client_ids"].detach().cpu().long()

    score_parts = []
    label_parts = []
    index_parts = []
    client_parts = []
    for client_id in _usable_clients(observations, candidate_client_ids):
        indices = torch.nonzero(
            candidate_client_ids == client_id, as_tuple=False
        ).flatten()
        result = run_fedmia(
            observations=observations,
            membership=membership[indices],
            target_client_id=client_id,
            measurement=measurement,
            aggregation=aggregation,
            tail=tail,
            calibration_fraction=calibration_fraction,
            seed=seed + 1009 * client_id,
            candidate_indices=indices,
        )
        global_indices = indices[result.sample_indices]
        score_parts.append(result.scores.detach().cpu())
        label_parts.append(result.labels.detach().cpu().long())
        index_parts.append(global_indices)
        client_parts.append(candidate_client_ids[global_indices])

    if not score_parts:
        raise ValueError("No client has a usable FedMIA round.")
    return ReplayResult(
        attack=attack,
        scores=torch.cat(score_parts),
        labels=torch.cat(label_parts),
        sample_indices=torch.cat(index_parts),
        client_ids=torch.cat(client_parts),
    )


def metric_row(
    result: ReplayResult,
    round_index: int,
    trajectory: str,
) -> dict[str, object]:
    metrics = membership_metrics(result.labels, result.scores)
    members = result.scores[result.labels == 1].to(torch.float64)
    nonmembers = result.scores[result.labels == 0].to(torch.float64)
    return {
        "attack": result.attack,
        "round": round_index,
        "trajectory": trajectory,
        "auc": metrics["auc"],
        "tpr_at_fpr_0.1": metrics["tpr_at_fpr_0.1"],
        "tpr_at_fpr_0.01": metrics["tpr_at_fpr_0.01"],
        "tpr_at_fpr_0.001": metrics["tpr_at_fpr_0.001"],
        "member_mean_score": float(members.mean()),
        "nonmember_mean_score": float(nonmembers.mean()),
        "mean_score_gap": float(members.mean() - nonmembers.mean()),
        "member_count": int(members.numel()),
        "nonmember_count": int(nonmembers.numel()),
        "client_count": int(torch.unique(result.client_ids).numel()),
    }


def replay_trajectory(
    observations: list[dict], payload: dict, audit: dict, seed: int
) -> tuple[list[dict[str, object]], dict[str, ReplayResult]]:
    rows: list[dict[str, object]] = []
    final_results = {}
    tail = str(audit.get("fedmia_tail", "upper"))
    calibration_fraction = float(
        audit.get("fedmia_tail_calibration_fraction", 0.25)
    )

    for attack in ATTACKS:
        aggregation = str(audit.get(f"{attack}_aggregation", "mean"))
        for position, observation in enumerate(observations):
            current_round = int(observation["round"])
            single = replay_pooled(
                [observation],
                payload,
                attack,
                aggregation="mean",
                tail=tail,
                calibration_fraction=calibration_fraction,
                seed=seed,
            )
            rows.append(metric_row(single, current_round, "single_round"))

            cumulative = replay_pooled(
                observations[: position + 1],
                payload,
                attack,
                aggregation=aggregation,
                tail=tail,
                calibration_fraction=calibration_fraction,
                seed=seed,
            )
            rows.append(metric_row(cumulative, current_round, "cumulative"))
            if position == len(observations) - 1:
                final_results[attack] = cumulative
    return rows, final_results


def per_client_metrics(final_results: dict[str, ReplayResult]) -> list[dict[str, object]]:
    rows = []
    for attack, result in final_results.items():
        for client_id in sorted(int(value) for value in torch.unique(result.client_ids)):
            mask = result.client_ids == client_id
            metrics = membership_metrics(result.labels[mask], result.scores[mask])
            rows.append(
                {
                    "attack": attack,
                    "audit_client_id": client_id,
                    "auc": metrics["auc"],
                    "tpr_at_fpr_0.1": metrics["tpr_at_fpr_0.1"],
                    "tpr_at_fpr_0.01": metrics["tpr_at_fpr_0.01"],
                    "tpr_at_fpr_0.001": metrics["tpr_at_fpr_0.001"],
                    "member_count": int((result.labels[mask] == 1).sum()),
                    "nonmember_count": int((result.labels[mask] == 0).sum()),
                }
            )
    return rows


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean, mean
    # Client is the uncertainty unit.  This normal interval is descriptive,
    # not a claim that repeated candidate samples are independent.
    margin = float(1.96 * array.std(ddof=1) / np.sqrt(array.size))
    return mean, mean - margin, mean + margin


def raw_signal_curves(
    observations: list[dict], payload: dict
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize target-client loss/cosine by membership for every round."""
    membership = payload["membership"].detach().cpu().long()
    owners = payload["candidate_client_ids"].detach().cpu().long()
    group_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []

    signal_fields = {
        "loss": "confidence",
        "cosine": "cosine",
    }
    for observation in observations:
        round_index = int(observation["round"])
        selected = observation["client_ids"].tolist()
        for signal, field in signal_fields.items():
            client_group_means: dict[int, list[float]] = {0: [], 1: []}
            pooled_values: dict[int, list[torch.Tensor]] = {0: [], 1: []}
            client_gaps = []
            for client_id in sorted(int(value) for value in torch.unique(owners)):
                if client_id not in selected:
                    continue
                indices = torch.nonzero(owners == client_id, as_tuple=False).flatten()
                position = selected.index(client_id)
                values = observation[field][position, indices].detach().cpu().float()
                if signal == "loss":
                    # The compact archive stores confidence=-cross_entropy.
                    values = -values

                means = {}
                for member_value in (0, 1):
                    group = values[membership[indices] == member_value]
                    if group.numel() == 0:
                        continue
                    means[member_value] = float(group.mean())
                    client_group_means[member_value].append(means[member_value])
                    pooled_values[member_value].append(group)
                if set(means) == {0, 1}:
                    favorable_gap = (
                        means[0] - means[1]
                        if signal == "loss"
                        else means[1] - means[0]
                    )
                    client_gaps.append(favorable_gap)

            for member_value in (0, 1):
                client_means = client_group_means[member_value]
                if not client_means:
                    continue
                pooled = torch.cat(pooled_values[member_value]).to(torch.float64)
                macro_mean, ci_low, ci_high = _mean_ci(client_means)
                group_rows.append(
                    {
                        "signal": signal,
                        "round": round_index,
                        "membership": member_value,
                        "membership_name": MEMBERSHIP_NAMES[member_value],
                        "client_balanced_mean": macro_mean,
                        "client_95ci_low": ci_low,
                        "client_95ci_high": ci_high,
                        "pooled_mean": float(pooled.mean()),
                        "pooled_median": float(pooled.median()),
                        "pooled_q10": float(torch.quantile(pooled, 0.10)),
                        "pooled_q90": float(torch.quantile(pooled, 0.90)),
                        "candidate_count": int(pooled.numel()),
                        "client_count": len(client_means),
                    }
                )

            gap_mean, gap_low, gap_high = _mean_ci(client_gaps)
            gap_rows.append(
                {
                    "signal": signal,
                    "round": round_index,
                    "favorable_gap": gap_mean,
                    "client_95ci_low": gap_low,
                    "client_95ci_high": gap_high,
                    "client_count": len(client_gaps),
                    "definition": (
                        "nonmember_loss_minus_member_loss"
                        if signal == "loss"
                        else "member_cosine_minus_nonmember_cosine"
                    ),
                }
            )
    return group_rows, gap_rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, final_results: dict[str, ReplayResult]) -> None:
    rows = []
    for attack, result in final_results.items():
        for index, client_id, label, score in zip(
            result.sample_indices.tolist(),
            result.client_ids.tolist(),
            result.labels.tolist(),
            result.scores.tolist(),
        ):
            rows.append(
                {
                    "attack": attack,
                    "sample_index": index,
                    "audit_client_id": client_id,
                    "membership": label,
                    "score": score,
                }
            )
    write_csv(path, rows)


def _series(
    rows: list[dict[str, object]], attack: str, trajectory: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (
            row
            for row in rows
            if row["attack"] == attack and row["trajectory"] == trajectory
        ),
        key=lambda row: int(row["round"]),
    )
    return (
        np.asarray([int(row["round"]) for row in selected]),
        np.asarray([float(row[metric]) for row in selected]),
    )


def plot_replay(
    path: Path,
    rows: list[dict[str, object]],
    final_results: dict[str, ReplayResult],
    config: dict,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#64748B",
            "axes.linewidth": 0.8,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    figure.patch.set_facecolor("white")

    for attack in ATTACKS:
        color = COLORS[attack]
        label = DISPLAY_NAMES[attack]
        rounds, cumulative_auc = _series(rows, attack, "cumulative", "auc")
        _, single_auc = _series(rows, attack, "single_round", "auc")
        axes[0, 0].plot(
            rounds,
            single_auc,
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.55,
        )
        axes[0, 0].plot(
            rounds,
            cumulative_auc,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            markevery=2,
            label=label,
        )

        _, cumulative_tpr = _series(
            rows, attack, "cumulative", "tpr_at_fpr_0.01"
        )
        _, single_tpr = _series(
            rows, attack, "single_round", "tpr_at_fpr_0.01"
        )
        axes[0, 1].plot(
            rounds,
            single_tpr,
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.55,
        )
        axes[0, 1].plot(
            rounds,
            cumulative_tpr,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            markevery=2,
            label=label,
        )

    axes[0, 0].axhline(0.5, color="#334155", linestyle="--", linewidth=1.1)
    axes[0, 0].text(
        0.5,
        0.502,
        "random AUC = 0.50",
        color="#334155",
        fontsize=8,
        ha="left",
        va="bottom",
    )
    axes[0, 0].set_title("A. ROC AUC over communication rounds (focused scale)")
    axes[0, 0].set_ylabel("ROC AUC")
    all_auc = [float(row["auc"]) for row in rows]
    auc_low = min(0.48, min(all_auc) - 0.01)
    auc_high = max(0.55, max(all_auc) + 0.01)
    axes[0, 0].set_ylim(max(0.0, auc_low), min(1.0, auc_high))

    axes[0, 1].axhline(0.01, color="#334155", linestyle="--", linewidth=1.1)
    axes[0, 1].text(
        0.5,
        0.011,
        "random reference = 1%",
        color="#334155",
        fontsize=8,
        ha="left",
        va="bottom",
    )
    axes[0, 1].set_title("B. TPR at 1% FPR over communication rounds")
    axes[0, 1].set_ylabel("TPR at FPR <= 1%")
    axes[0, 1].set_ylim(bottom=0.0)

    for axis in axes[0]:
        axis.set_xlabel("Communication round")
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncols=2, loc="upper left")
        axis.text(
            0.99,
            0.02,
            "solid: cumulative  |  dotted: single round",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="#64748B",
            fontsize=8,
        )

    bins = np.linspace(0.0, 1.0, 31)
    for column, attack in enumerate(ATTACKS):
        axis = axes[1, column]
        result = final_results[attack]
        member = result.scores[result.labels == 1].numpy()
        nonmember = result.scores[result.labels == 0].numpy()
        axis.hist(
            nonmember,
            bins=bins,
            density=True,
            histtype="step",
            color="#475569",
            linewidth=1.8,
            linestyle="--",
            label=f"Non-member (n={nonmember.size:,})",
        )
        axis.hist(
            member,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=COLORS[attack],
            edgecolor=COLORS[attack],
            linewidth=1.1,
            alpha=0.24,
            label=f"Member (n={member.size:,})",
        )
        final_metrics = membership_metrics(result.labels, result.scores)
        axis.set_title(
            f"{chr(ord('C') + column)}. Final {DISPLAY_NAMES[attack]} score distribution"
        )
        axis.set_xlabel("Final FedMIA membership score")
        axis.set_ylabel("Density")
        axis.set_xlim(0.0, 1.0)
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper left")
        axis.text(
            0.98,
            0.84,
            (
                f"AUC {final_metrics['auc']:.3f}\n"
                f"TPR@1%FPR {final_metrics['tpr_at_fpr_0.01']:.2%}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#334155",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.92,
            },
        )

    audit = config.get("audit", {})
    title = (
        "FedMIA attack replay — "
        f"{str(config.get('dataset_name', 'unknown')).upper()} / "
        f"{str(config.get('aggregator', 'unknown')).upper()} / "
        f"seed {config.get('seed', 'unknown')}"
    )
    subtitle = (
        f"{len({int(row['round']) for row in rows})} observed rounds; "
        f"pooled client audit; tail={audit.get('fedmia_tail', 'upper')}; "
        "curves recomputed from compact signals.pt"
    )
    figure.suptitle(title, x=0.07, y=0.985, ha="left", fontsize=16, fontweight="bold")
    figure.text(0.07, 0.948, subtitle, ha="left", va="top", color="#64748B")
    figure.tight_layout(rect=(0.04, 0.03, 0.98, 0.91), h_pad=2.4, w_pad=2.0)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _curve_arrays(
    rows: list[dict[str, object]], signal: str, member_value: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        (
            row
            for row in rows
            if row["signal"] == signal
            and int(row["membership"]) == member_value
        ),
        key=lambda row: int(row["round"]),
    )
    return (
        np.asarray([int(row["round"]) for row in selected]),
        np.asarray([float(row["client_balanced_mean"]) for row in selected]),
        np.asarray([float(row["client_95ci_low"]) for row in selected]),
        np.asarray([float(row["client_95ci_high"]) for row in selected]),
    )


def _gap_arrays(
    rows: list[dict[str, object]], signal: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if row["signal"] == signal),
        key=lambda row: int(row["round"]),
    )
    return (
        np.asarray([int(row["round"]) for row in selected]),
        np.asarray([float(row["favorable_gap"]) for row in selected]),
        np.asarray([float(row["client_95ci_low"]) for row in selected]),
        np.asarray([float(row["client_95ci_high"]) for row in selected]),
    )


def plot_raw_signal_curves(
    path: Path,
    group_rows: list[dict[str, object]],
    gap_rows: list[dict[str, object]],
    config: dict,
    dpi: int,
) -> None:
    """Plot member/non-member target-client loss and cosine trajectories."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#64748B",
            "axes.linewidth": 0.8,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex="col")
    figure.patch.set_facecolor("white")

    signal_specs = {
        "loss": {
            "column": 0,
            "title": "A. Cross-entropy loss by membership (focused scale)",
            "ylabel": "Cross-entropy loss",
            "gap_title": "C. Loss separation favorable to membership inference",
            "gap_ylabel": "Non-member loss − member loss",
        },
        "cosine": {
            "column": 1,
            "title": "B. Gradient/update cosine by membership (focused scale)",
            "ylabel": "Cosine similarity",
            "gap_title": "D. Cosine separation favorable to membership inference",
            "gap_ylabel": "Member cosine − non-member cosine",
        },
    }

    for signal, spec in signal_specs.items():
        column = int(spec["column"])
        axis = axes[0, column]
        for member_value in (1, 0):
            rounds, means, lows, highs = _curve_arrays(
                group_rows, signal, member_value
            )
            color = MEMBERSHIP_COLORS[member_value]
            linestyle = "-" if member_value == 1 else "--"
            marker = "o" if member_value == 1 else "s"
            marker_face = color if member_value == 1 else "white"
            axis.fill_between(
                rounds,
                lows,
                highs,
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            axis.plot(
                rounds,
                means,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker=marker,
                markerfacecolor=marker_face,
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=3.7,
                markevery=2,
                label=MEMBERSHIP_NAMES[member_value],
            )
        axis.set_title(str(spec["title"]))
        axis.set_ylabel(str(spec["ylabel"]))
        axis.legend(frameon=False, ncols=2, loc="best")
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

        gap_axis = axes[1, column]
        rounds, gaps, lows, highs = _gap_arrays(gap_rows, signal)
        gap_axis.axhline(0.0, color="#334155", linestyle="--", linewidth=1.1)
        gap_axis.fill_between(
            rounds,
            lows,
            highs,
            color="#2563EB",
            alpha=0.13,
            linewidth=0,
        )
        gap_axis.plot(
            rounds,
            gaps,
            color="#2563EB",
            linewidth=2.0,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#2563EB",
            markersize=3.8,
            markevery=2,
        )
        gap_axis.set_title(str(spec["gap_title"]))
        gap_axis.set_ylabel(str(spec["gap_ylabel"]))
        gap_axis.set_xlabel("Communication round")
        gap_axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        gap_axis.spines[["top", "right"]].set_visible(False)
        gap_axis.text(
            0.99,
            0.04,
            f"final gap {gaps[-1]:+.4f}",
            transform=gap_axis.transAxes,
            ha="right",
            va="bottom",
            color="#334155",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.92,
            },
        )

    title = (
        "Member vs non-member FedMIA input signals — "
        f"{str(config.get('dataset_name', 'unknown')).upper()} / "
        f"{str(config.get('aggregator', 'unknown')).upper()} / "
        f"seed {config.get('seed', 'unknown')}"
    )
    subtitle = (
        "Target-client signal; client-balanced mean with approximate 95% CI "
        "across clients; positive gap means membership-favorable separation"
    )
    figure.suptitle(title, x=0.07, y=0.985, ha="left", fontsize=16, fontweight="bold")
    figure.text(0.07, 0.948, subtitle, ha="left", va="top", color="#64748B")
    figure.tight_layout(rect=(0.04, 0.03, 0.98, 0.91), h_pad=2.5, w_pad=2.2)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def validate_against_summary(
    run_dir: Path, final_results: dict[str, ReplayResult]
) -> dict[str, dict[str, float | bool]]:
    summary_path = run_dir / "privacy_audit" / "summary.json"
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {item["attack"]: item for item in summary.get("attacks", [])}
    checks = {}
    for attack, result in final_results.items():
        actual = membership_metrics(result.labels, result.scores)
        reference = expected.get(attack, {})
        auc_diff = actual["auc"] - float(reference.get("auc", float("nan")))
        tpr_diff = actual["tpr_at_fpr_0.01"] - float(
            reference.get("tpr_at_fpr_0.01", float("nan"))
        )
        checks[attack] = {
            "replayed_auc": actual["auc"],
            "summary_auc": float(reference.get("auc", float("nan"))),
            "auc_absolute_difference": abs(auc_diff),
            "replayed_tpr_at_fpr_0.01": actual["tpr_at_fpr_0.01"],
            "summary_tpr_at_fpr_0.01": float(
                reference.get("tpr_at_fpr_0.01", float("nan"))
            ),
            "tpr_absolute_difference": abs(tpr_diff),
            "exact_within_1e-12": abs(auc_diff) <= 1e-12 and abs(tpr_diff) <= 1e-12,
        }
    return checks


def main() -> int:
    args = parse_args()
    launcher_log = args.launcher_log.resolve()
    config, payload, config_path, run_dir = load_inputs(launcher_log)
    observations = sorted(payload["observations"], key=lambda item: int(item["round"]))
    log_rounds = launcher_rounds(launcher_log)
    signal_rounds = [int(item["round"]) for item in observations]
    if log_rounds and log_rounds != signal_rounds:
        raise RuntimeError(
            "Launcher log and signals.pt contain different observed rounds: "
            f"log={log_rounds}, signals={signal_rounds}"
        )

    audit = dict(config.get("audit", {}))
    seed = int(audit.get("seed", config.get("seed", 42)))
    rows, final_results = replay_trajectory(observations, payload, audit, seed)
    signal_rows, signal_gap_rows = raw_signal_curves(observations, payload)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / "privacy_audit" / "replay"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "fedmia_round_metrics.csv"
    client_path = output_dir / "fedmia_client_metrics.csv"
    predictions_path = output_dir / "fedmia_predictions_replayed.csv"
    figure_path = output_dir / "fedmia_attack_replay.png"
    signal_curve_path = output_dir / "fedmia_member_nonmember_signal_curves.csv"
    signal_gap_path = output_dir / "fedmia_member_nonmember_signal_gaps.csv"
    signal_figure_path = output_dir / "fedmia_member_nonmember_signal_curves.png"
    summary_path = output_dir / "fedmia_replay_summary.json"

    write_csv(trajectory_path, rows)
    write_csv(client_path, per_client_metrics(final_results))
    write_predictions(predictions_path, final_results)
    write_csv(signal_curve_path, signal_rows)
    write_csv(signal_gap_path, signal_gap_rows)
    plot_replay(figure_path, rows, final_results, config, args.dpi)
    plot_raw_signal_curves(
        signal_figure_path,
        signal_rows,
        signal_gap_rows,
        config,
        args.dpi,
    )

    validation = validate_against_summary(run_dir, final_results)
    replay_summary = {
        "launcher_log": str(launcher_log),
        "generated_config": str(config_path),
        "resolved_run_dir": str(run_dir),
        "signals_path": str(run_dir / "privacy_audit" / "signals.pt"),
        "observed_rounds": signal_rounds,
        "storage_mode": payload.get("storage_mode"),
        "attack_configuration": {
            "tail": audit.get("fedmia_tail", "upper"),
            "fedmia_loss_aggregation": audit.get(
                "fedmia_loss_aggregation", "mean"
            ),
            "fedmia_cosine_aggregation": audit.get(
                "fedmia_cosine_aggregation", "mean"
            ),
            "audit_scope": "pooled_clients",
        },
        "validation_against_original_summary": validation,
        "outputs": {
            "round_metrics": str(trajectory_path),
            "client_metrics": str(client_path),
            "predictions": str(predictions_path),
            "figure": str(figure_path),
            "member_nonmember_signal_curves": str(signal_curve_path),
            "member_nonmember_signal_gaps": str(signal_gap_path),
            "member_nonmember_signal_figure": str(signal_figure_path),
        },
    }
    summary_path.write_text(
        json.dumps(replay_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Resolved run: {run_dir}")
    print(f"Replayed {len(signal_rounds)} rounds from: {run_dir / 'privacy_audit/signals.pt'}")
    for attack, check in validation.items():
        print(
            f"{attack}: AUC={check['replayed_auc']:.12f}, "
            f"TPR@1%FPR={check['replayed_tpr_at_fpr_0.01']:.12f}, "
            f"matches summary={check['exact_within_1e-12']}"
        )
    print(f"Figure: {figure_path}")
    print(f"Member/non-member signal figure: {signal_figure_path}")
    print(f"Round metrics: {trajectory_path}")
    print(f"Member/non-member signal curves: {signal_curve_path}")
    print(f"Member/non-member signal gaps: {signal_gap_path}")
    print(f"Client metrics: {client_path}")
    print(f"Replay summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
