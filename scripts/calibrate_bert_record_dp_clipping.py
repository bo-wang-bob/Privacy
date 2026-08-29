#!/usr/bin/env python3
"""Measure per-record BERT Adapter gradient norms for DP clipping calibration.

This is an explicitly non-private analysis.  It never clips gradients, adds
noise, or updates the model.  A fixed sample of local training records is
evaluated at one or more model states so that candidate Record-DP clipping
thresholds can be chosen before the private sweep starts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trainmodel.transformer_adapter import TransformerAdapterClassifier
from utils.text_data_loader import load_federated_text_classification


LOGGER = logging.getLogger(__name__)
DEFAULT_CANDIDATE_NORMS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class Checkpoint:
    label: str
    communication_round: int
    path: Path | None


@dataclass(frozen=True)
class SampleReference:
    client_id: int
    local_index: int
    global_index: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-private BERT/SST-5 per-record gradient-norm calibration "
            "using the exact Adapter and classifier parameter scope of Record-DP."
        )
    )
    parser.add_argument(
        "--run-dir",
        help=(
            "Completed non-private BERT run. Its run_config.yaml, initial state, "
            "saved_models/global_round_*.pt, and final checkpoint are discovered."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/models/bert_adapter.yaml",
        help="Configuration used when --run-dir is omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Additional trainable-state checkpoint. Repeat for multiple states. "
            "The reserved value 'initial' selects the initialized model."
        ),
    )
    parser.add_argument(
        "--checkpoint-rounds",
        default="available",
        help=(
            "Comma-separated completed rounds to analyze from --run-dir. Round 0 "
            "means the initialized model; default: every available state."
        ),
    )
    parser.add_argument(
        "--client-ids",
        default="all",
        help="Comma-separated federated clients or 'all' (default).",
    )
    parser.add_argument(
        "--samples-per-client",
        type=int,
        default=16,
        help="Fixed random records sampled without replacement per client.",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=4,
        help="Records sharing one forward graph; gradients remain per record.",
    )
    parser.add_argument(
        "--candidate-norms",
        default=",".join(str(value) for value in DEFAULT_CANDIDATE_NORMS),
        help="Comma-separated C values for counterfactual clipping statistics.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Client-cluster bootstrap replicates for 95%% intervals.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete torch device such as cuda:0.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory. Defaults to a timestamped directory under "
            "analysis_scripts/."
        ),
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _parse_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parsed) != len(set(parsed)):
        raise ValueError("Integer lists must not contain duplicates.")
    return parsed


def _parse_positive_floats(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(not math.isfinite(item) or item <= 0 for item in parsed):
        raise ValueError("Candidate clipping norms must be finite and positive.")
    return sorted(set(parsed))


def _parse_checkpoint_argument(value: str) -> Checkpoint:
    if value.strip().lower() == "initial":
        return Checkpoint("round_0000_initial", 0, None)
    if "=" not in value:
        raise ValueError("--checkpoint must be 'initial' or LABEL=PATH.")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label or not raw_path.strip():
        raise ValueError("--checkpoint requires a non-empty label and path.")
    round_matches = re.findall(r"\d+", label)
    communication_round = int(round_matches[-1]) if round_matches else -1
    path = _resolve_repo_path(raw_path.strip())
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {path}")
    return Checkpoint(label, communication_round, path)


def discover_run_checkpoints(
    run_dir: Path, config: dict, checkpoint_rounds: str
) -> list[Checkpoint]:
    available: dict[int, Checkpoint] = {
        0: Checkpoint("round_0000_initial", 0, None)
    }
    saved_dir = run_dir / "saved_models"
    if saved_dir.is_dir():
        for path in sorted(saved_dir.glob("global_round_*.pt")):
            match = re.fullmatch(r"global_round_(\d+)\.pt", path.name)
            if match:
                round_number = int(match.group(1))
                available[round_number] = Checkpoint(
                    f"round_{round_number:04d}", round_number, path.resolve()
                )
    final_round = int(config.get("num_global_iters", 0))
    final_path = run_dir / "final_transformer_adapter.pt"
    if final_round > 0 and final_path.is_file():
        available[final_round] = Checkpoint(
            f"round_{final_round:04d}_final", final_round, final_path.resolve()
        )
    if checkpoint_rounds.strip().lower() == "available":
        selected_rounds = sorted(available)
    else:
        selected_rounds = _parse_ints(checkpoint_rounds)
        missing = sorted(set(selected_rounds) - set(available))
        if missing:
            raise ValueError(
                f"Requested checkpoint rounds are unavailable: {missing}; "
                f"available={sorted(available)}"
            )
    return [available[round_number] for round_number in selected_rounds]


def resolve_experiment(args: argparse.Namespace) -> tuple[dict, list[Checkpoint], Path | None]:
    run_dir = _resolve_repo_path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        config_path = run_dir / "run_config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"run_config.yaml was not found in {run_dir}")
    else:
        config_path = _resolve_repo_path(args.config)
    config = _read_yaml(config_path)
    if str(config.get("model_type", "")).lower() != "bert_adapter":
        raise ValueError("Calibration currently requires model_type=bert_adapter.")
    if str(config.get("dataset_name", "")).lower() != "sst5":
        raise ValueError("Calibration currently requires dataset_name=sst5.")
    checkpoints = (
        discover_run_checkpoints(run_dir, config, args.checkpoint_rounds)
        if run_dir is not None
        else []
    )
    checkpoints.extend(_parse_checkpoint_argument(value) for value in args.checkpoint)
    if not checkpoints:
        checkpoints = [Checkpoint("round_0000_initial", 0, None)]
    labels = [checkpoint.label for checkpoint in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint labels must be unique.")
    return config, checkpoints, run_dir


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def select_sample_references(
    train_sets,
    client_ids: Iterable[int],
    samples_per_client: int,
    seed: int,
) -> list[SampleReference]:
    if samples_per_client <= 0:
        raise ValueError("samples_per_client must be positive.")
    references: list[SampleReference] = []
    for client_id in client_ids:
        dataset = train_sets[client_id]
        if samples_per_client > len(dataset):
            raise ValueError(
                f"Client {client_id} has only {len(dataset)} records, fewer than "
                f"samples_per_client={samples_per_client}."
            )
        generator = torch.Generator().manual_seed(
            int(seed) + 1000003 * int(client_id) + 43019
        )
        local_indices = torch.randperm(len(dataset), generator=generator)[
            :samples_per_client
        ].tolist()
        source_indices = getattr(dataset, "indices", None)
        for local_index in local_indices:
            global_index = (
                int(source_indices[local_index])
                if source_indices is not None
                else None
            )
            references.append(
                SampleReference(int(client_id), int(local_index), global_index)
            )
    return references


def parameter_group(name: str) -> str:
    if ".adapter.down." in name:
        return "adapter_down"
    if ".adapter.up." in name:
        return "adapter_up"
    if name.startswith("classifier."):
        return "classifier"
    return "other_trainable"


def per_sample_norm_rows(
    *,
    model: torch.nn.Module,
    train_sets,
    collate_fn,
    references: list[SampleReference],
    checkpoint: Checkpoint,
    device: torch.device,
    microbatch_size: int,
) -> list[dict]:
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive.")
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("The model has no trainable parameters.")
    parameters = [parameter for _, parameter in named_parameters]
    groups = [parameter_group(name) for name, _ in named_parameters]
    group_names = sorted(set(groups))
    rows: list[dict] = []
    model.train()
    for offset in range(0, len(references), microbatch_size):
        batch_references = references[offset : offset + microbatch_size]
        samples = [
            train_sets[reference.client_id][reference.local_index]
            for reference in batch_references
        ]
        inputs, labels = collate_fn(samples)
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        losses = F.cross_entropy(model(inputs), labels, reduction="none")
        token_counts = inputs[:, 1].sum(dim=1).detach().cpu().tolist()
        for sample_offset, reference in enumerate(batch_references):
            gradients = torch.autograd.grad(
                losses[sample_offset],
                parameters,
                retain_graph=sample_offset + 1 < len(batch_references),
                allow_unused=False,
            )
            group_squares = {name: 0.0 for name in group_names}
            for gradient, group in zip(gradients, groups):
                group_squares[group] += float(
                    gradient.detach().float().square().sum().cpu()
                )
            joint_square = sum(group_squares.values())
            row = {
                "checkpoint": checkpoint.label,
                "communication_round": checkpoint.communication_round,
                "client_id": reference.client_id,
                "local_index": reference.local_index,
                "global_index": reference.global_index,
                "label": int(labels[sample_offset].detach().cpu()),
                "token_count": int(token_counts[sample_offset]),
                "loss": float(losses[sample_offset].detach().cpu()),
                "joint_grad_norm": math.sqrt(max(joint_square, 0.0)),
            }
            for group in group_names:
                row[f"{group}_grad_norm"] = math.sqrt(
                    max(group_squares[group], 0.0)
                )
            rows.append(row)
        del inputs, labels, losses
        if len(rows) % max(16, microbatch_size) == 0 or len(rows) == len(references):
            LOGGER.info(
                "%s: measured %d/%d records",
                checkpoint.label,
                len(rows),
                len(references),
            )
    return rows


def _metric_values(norms: np.ndarray, current_c: float = 1.0) -> dict[str, float]:
    values: dict[str, float] = {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms, ddof=1)) if norms.size > 1 else 0.0,
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "current_c": float(current_c),
        "current_c_clip_fraction": float(np.mean(norms > current_c)),
        "current_c_mean_clip_factor": float(
            np.mean(np.minimum(1.0, current_c / np.maximum(norms, 1e-30)))
        ),
    }
    for quantile in QUANTILES:
        values[f"p{int(100 * quantile):02d}"] = float(
            np.quantile(norms, quantile)
        )
    return values


def _cluster_bootstrap_intervals(
    rows: list[dict], replicates: int, seed: int
) -> dict[str, tuple[float, float]]:
    if replicates <= 0:
        return {}
    clients = sorted({int(row["client_id"]) for row in rows})
    if len(clients) < 2:
        return {}
    by_client = {
        client_id: np.asarray(
            [
                float(row["joint_grad_norm"])
                for row in rows
                if int(row["client_id"]) == client_id
            ],
            dtype=np.float64,
        )
        for client_id in clients
    }
    generator = np.random.default_rng(int(seed))
    metrics = {
        "p50": [],
        "p75": [],
        "p90": [],
        "p95": [],
        "current_c_clip_fraction": [],
    }
    for _ in range(replicates):
        selected = generator.choice(clients, size=len(clients), replace=True)
        norms = np.concatenate([by_client[int(client)] for client in selected])
        values = _metric_values(norms)
        for name in metrics:
            metrics[name].append(values[name])
    return {
        name: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for name, values in metrics.items()
    }


def summarize_checkpoints(
    raw_rows: list[dict], bootstrap_replicates: int, seed: int
) -> list[dict]:
    summaries = []
    checkpoints = list(dict.fromkeys(row["checkpoint"] for row in raw_rows))
    for index, checkpoint in enumerate(checkpoints):
        rows = [row for row in raw_rows if row["checkpoint"] == checkpoint]
        norms = np.asarray(
            [float(row["joint_grad_norm"]) for row in rows], dtype=np.float64
        )
        summary = {
            "checkpoint": checkpoint,
            "communication_round": int(rows[0]["communication_round"]),
            "records": len(rows),
            "clients": len({int(row["client_id"]) for row in rows}),
            **_metric_values(norms),
        }
        intervals = _cluster_bootstrap_intervals(
            rows, bootstrap_replicates, seed + 7919 * index
        )
        for metric, (lower, upper) in intervals.items():
            summary[f"{metric}_ci95_low"] = lower
            summary[f"{metric}_ci95_high"] = upper
        summaries.append(summary)
    return summaries


def clipping_grid(raw_rows: list[dict], candidates: Iterable[float]) -> list[dict]:
    output = []
    checkpoints = list(dict.fromkeys(row["checkpoint"] for row in raw_rows))
    for checkpoint in checkpoints:
        rows = [row for row in raw_rows if row["checkpoint"] == checkpoint]
        norms = np.asarray(
            [float(row["joint_grad_norm"]) for row in rows], dtype=np.float64
        )
        for candidate in candidates:
            factors = np.minimum(1.0, float(candidate) / np.maximum(norms, 1e-30))
            clipped_norms = norms * factors
            output.append(
                {
                    "checkpoint": checkpoint,
                    "communication_round": int(rows[0]["communication_round"]),
                    "max_grad_norm": float(candidate),
                    "clip_fraction": float(np.mean(norms > candidate)),
                    "mean_clip_factor": float(np.mean(factors)),
                    "mean_unclipped_norm": float(np.mean(norms)),
                    "mean_clipped_norm": float(np.mean(clipped_norms)),
                    "norm_magnitude_retained": float(
                        np.sum(clipped_norms) / max(np.sum(norms), 1e-30)
                    ),
                }
            )
    return output


def candidate_shortlist(summary_rows: list[dict]) -> list[dict]:
    median_p95 = float(np.median([float(row["p95"]) for row in summary_rows]))
    upper = max(1.0, 2.0 ** math.ceil(math.log2(max(median_p95, 1.0))))
    candidates = []
    candidate = 1.0
    while candidate <= upper:
        candidates.append(candidate)
        candidate *= 2.0
    return [
        {
            "basis": (
                "current_baseline"
                if candidate == 1.0
                else "geometric_tradeoff_grid"
            ),
            "max_grad_norm": candidate,
            "grid_upper_basis": "median_checkpoint_p95",
            "grid_upper_raw_value": median_p95,
        }
        for candidate in candidates
    ]


def summarize_parameter_groups(raw_rows: list[dict]) -> list[dict]:
    group_fields = [
        field
        for field in raw_rows[0]
        if field.endswith("_grad_norm") and field != "joint_grad_norm"
    ]
    output = []
    checkpoints = list(dict.fromkeys(row["checkpoint"] for row in raw_rows))
    for checkpoint in checkpoints:
        rows = [row for row in raw_rows if row["checkpoint"] == checkpoint]
        joint_squares = np.asarray(
            [float(row["joint_grad_norm"]) ** 2 for row in rows],
            dtype=np.float64,
        )
        for field in group_fields:
            values = np.asarray(
                [float(row[field]) for row in rows], dtype=np.float64
            )
            squared_share = np.square(values) / np.maximum(joint_squares, 1e-30)
            output.append(
                {
                    "checkpoint": checkpoint,
                    "communication_round": int(rows[0]["communication_round"]),
                    "parameter_group": field.removesuffix("_grad_norm"),
                    "mean_norm": float(np.mean(values)),
                    "p50_norm": float(np.quantile(values, 0.5)),
                    "p90_norm": float(np.quantile(values, 0.9)),
                    "mean_squared_norm_share": float(np.mean(squared_share)),
                }
            )
    return output


def sampling_diagnostics(train_sets, references: list[SampleReference]) -> dict:
    source_counts: dict[int, int] = {}
    for dataset in train_sets:
        for local_index in range(len(dataset)):
            _, label = dataset[local_index]
            source_counts[int(label)] = source_counts.get(int(label), 0) + 1
    sample_counts: dict[int, int] = {}
    for reference in references:
        _, label = train_sets[reference.client_id][reference.local_index]
        sample_counts[int(label)] = sample_counts.get(int(label), 0) + 1
    labels = sorted(set(source_counts) | set(sample_counts))
    source_total = sum(source_counts.values())
    sample_total = sum(sample_counts.values())
    source_proportions = {
        label: source_counts.get(label, 0) / source_total for label in labels
    }
    sample_proportions = {
        label: sample_counts.get(label, 0) / sample_total for label in labels
    }
    label_tv = 0.5 * sum(
        abs(source_proportions[label] - sample_proportions[label])
        for label in labels
    )
    return {
        "source_records": source_total,
        "sample_records": sample_total,
        "source_label_counts": {
            str(key): value for key, value in source_counts.items()
        },
        "sample_label_counts": {
            str(key): value for key, value in sample_counts.items()
        },
        "source_label_proportions": {
            str(key): value for key, value in source_proportions.items()
        },
        "sample_label_proportions": {
            str(key): value for key, value in sample_proportions.items()
        },
        "label_total_variation": label_tv,
        "unique_global_indices": len(
            {
                reference.global_index
                for reference in references
                if reference.global_index is not None
            }
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_figure(
    path: Path, raw_rows: list[dict], grid_rows: list[dict]
) -> None:
    import matplotlib.pyplot as plt

    checkpoints = list(dict.fromkeys(row["checkpoint"] for row in raw_rows))
    colors = ["#2F6BFF", "#E88927", "#7A9A32", "#C55A9B", "#8566C7"]
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for index, checkpoint in enumerate(checkpoints):
        color = colors[index % len(colors)]
        norms = np.sort(
            np.asarray(
                [
                    float(row["joint_grad_norm"])
                    for row in raw_rows
                    if row["checkpoint"] == checkpoint
                ],
                dtype=np.float64,
            )
        )
        cumulative = np.arange(1, norms.size + 1) / norms.size
        axes[0].plot(norms, cumulative, label=checkpoint, color=color, linewidth=2)
        selected_grid = [row for row in grid_rows if row["checkpoint"] == checkpoint]
        axes[1].plot(
            [float(row["max_grad_norm"]) for row in selected_grid],
            [float(row["clip_fraction"]) for row in selected_grid],
            marker="o",
            label=checkpoint,
            color=color,
            linewidth=2,
        )
    axes[0].axvline(1.0, color="#333333", linestyle="--", linewidth=1.5, label="current C=1")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("Per-record joint gradient norm (log scale)")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[0].set_title("Gradient-norm empirical CDF")
    axes[0].grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Candidate clipping threshold C (log scale)")
    axes[1].set_ylabel("Fraction of records clipped")
    axes[1].set_title("Counterfactual clipping rate")
    axes[1].grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "BERT Adapter / SST-5 non-private clipping calibration",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_markdown_report(
    path: Path,
    *,
    summary_rows: list[dict],
    parameter_group_rows: list[dict],
    sample_diagnostics: dict,
    shortlist: list[dict],
    config: dict,
    run_dir: Path | None,
    output_dir: Path,
) -> None:
    lines = [
        "# BERT Record-DP 裁剪阈值非隐私校准",
        "",
        "## 技术摘要",
        "",
        (
            "本实验在不裁剪、不加噪、也不更新模型的条件下，统计每条 SST-5 "
            "序列对全部可训练 BERT Adapter 与分类头参数的联合梯度范数。"
        ),
        "当前 `C=1` 仅作为对照；候选值应结合下面的裁剪比例和固定隐私预算下随 `C` 线性增长的绝对噪声共同选择。",
        "",
        "## 分阶段统计",
        "",
        "| 模型状态 | 样本数 | P50 | P75 | P90 | P95 | C=1 裁剪率 | 平均裁剪因子 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {checkpoint} | {records} | {p50:.4g} | {p75:.4g} | "
            "{p90:.4g} | {p95:.4g} | {clip:.2%} | {factor:.4f} |".format(
                checkpoint=row["checkpoint"],
                records=int(row["records"]),
                p50=float(row["p50"]),
                p75=float(row["p75"]),
                p90=float(row["p90"]),
                p95=float(row["p95"]),
                clip=float(row["current_c_clip_fraction"]),
                factor=float(row["current_c_mean_clip_factor"]),
            )
        )
    lines.extend(
        [
            "",
            "## 建议进入固定预算消融的候选值",
            "",
            "先在一个代表性固定预算（建议 `epsilon=3`）上运行以下对数网格；它覆盖当前基线到跨检查点 P95 的典型量级，但不是已经证明最优的阈值：",
            "",
        ]
    )
    for row in shortlist:
        lines.append(f"- `C={float(row['max_grad_norm']):g}`")
    lines.extend(
        [
            "",
            "下一步应在相同的 `epsilon`、`delta`、采样率、轮数和随机种子下逐个运行这些 `C`。噪声乘数 `sigma` 不变，但实际噪声标准差 `sigma*C` 会随 `C` 线性变化。",
            "",
            "## 联合范数由哪些参数贡献",
            "",
            "| 模型状态 | 参数组 | 平均范数 | P50 | 平均平方范数占比 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in parameter_group_rows:
        lines.append(
            "| {checkpoint} | {group} | {mean:.4g} | {p50:.4g} | {share:.2%} |".format(
                checkpoint=row["checkpoint"],
                group=row["parameter_group"],
                mean=float(row["mean_norm"]),
                p50=float(row["p50_norm"]),
                share=float(row["mean_squared_norm_share"]),
            )
        )
    lines.extend(
        [
            "",
            "初始化时 Adapter down 投影因为 up 投影的零初始化而没有梯度；训练后梯度贡献会重新分配。该分组核对也验证了联合范数确实覆盖全部可训练参数。",
            "",
            "## 抽样质量",
            "",
            f"- 训练集记录数：`{sample_diagnostics['source_records']}`；校准样本：`{sample_diagnostics['sample_records']}`。",
            f"- 唯一全局样本索引：`{sample_diagnostics['unique_global_indices']}`；不存在跨客户端重复抽样。",
            f"- 校准样本与完整训练集的标签分布 TV 距离：`{float(sample_diagnostics['label_total_variation']):.4f}`。",
            "",
            "## 统计口径与限制",
            "",
            f"- 模型：`{config.get('model_type')}`；数据集：`{config.get('dataset_name')}`；客户端数：`{config.get('total_users')}`。",
            "- 每个客户端使用固定的无放回随机样本；不同检查点复用同一批记录，因此阶段差异可以配对解释。",
            "- 置信区间采用客户端聚类 bootstrap，避免把同一客户端内的记录错误地视为完全独立。",
            "- 这是非隐私校准，不能把由私有训练集得到并对外发布的阈值选择过程直接视为零隐私成本。正式保证应使用公开代理集、预先注册网格或另行计入选择预算。",
            "- 梯度范数分位数只控制裁剪强度，不单独决定最终效用；仍需结合准确率、ProjRes 和噪声幅度完成消融。",
            "",
            "## 可复现产物",
            "",
            f"- 原始逐记录统计：`{(output_dir / 'per_record_gradient_norms.csv').name}`",
            f"- 检查点汇总：`{(output_dir / 'summary_by_checkpoint.csv').name}`",
            f"- 参数组贡献：`{(output_dir / 'summary_by_parameter_group.csv').name}`",
            f"- 候选阈值裁剪率：`{(output_dir / 'clipping_grid.csv').name}`",
            f"- 图：`{(output_dir / 'gradient_norm_calibration.png').name}`",
        ]
    )
    if run_dir is not None:
        lines.append(f"- 非隐私训练来源：`{run_dir}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_trainable_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint is not a non-empty state dictionary: {path}")
    return state


def main() -> None:
    args = parse_args()
    if args.microbatch_size <= 0:
        raise ValueError("--microbatch-size must be positive.")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative.")
    config, checkpoints, run_dir = resolve_experiment(args)
    seed = int(config.get("seed", 42) if args.seed is None else args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(args.device)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        _resolve_repo_path(args.output_dir)
        if args.output_dir
        else REPOSITORY_ROOT
        / "analysis_scripts"
        / f"bert_record_dp_clipping_calibration_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        ],
    )
    LOGGER.info("Calibration output: %s", output_dir)
    LOGGER.info("Device: %s", device)

    model_path = _resolve_repo_path(config["model_path"])
    dataset_path = _resolve_repo_path(config["dataset_path"])
    data = load_federated_text_classification(
        dataset_name=str(config["dataset_name"]),
        dataset_path=dataset_path,
        model_path=model_path,
        num_users=int(config["total_users"]),
        seed=seed,
        max_length=int(config.get("max_length", 128)),
    )
    if args.client_ids.strip().lower() == "all":
        client_ids = list(range(int(config["total_users"])))
    else:
        client_ids = _parse_ints(args.client_ids)
        invalid = [
            client_id
            for client_id in client_ids
            if not 0 <= client_id < int(config["total_users"])
        ]
        if invalid:
            raise ValueError(f"Client IDs are out of range: {invalid}")
    references = select_sample_references(
        data.train_sets, client_ids, args.samples_per_client, seed
    )
    adapter = dict(config.get("adapter", {}))
    model = TransformerAdapterClassifier(
        model_path=str(model_path),
        architecture="bert",
        num_classes=len(data.class_names),
        reduction=int(adapter.get("reduction", 2)),
        activation=str(adapter.get("activation", "relu")),
        classifier_dropout=float(adapter.get("classifier_dropout", 0.0)),
        gradient_checkpointing=bool(adapter.get("gradient_checkpointing", False)),
        zero_init_up=bool(adapter.get("zero_init_up", True)),
        device=device,
    )
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.export_trainable_state().items()
    }
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    LOGGER.info(
        "Selected %d records from %d clients; trainable parameters=%d",
        len(references),
        len(client_ids),
        trainable_count,
    )

    raw_rows: list[dict] = []
    for checkpoint in checkpoints:
        state = (
            initial_state
            if checkpoint.path is None
            else _load_trainable_state(checkpoint.path)
        )
        model.load_trainable_state(state, strict=True)
        raw_rows.extend(
            per_sample_norm_rows(
                model=model,
                train_sets=data.train_sets,
                collate_fn=data.collate_fn,
                references=references,
                checkpoint=checkpoint,
                device=device,
                microbatch_size=args.microbatch_size,
            )
        )

    summary_rows = summarize_checkpoints(
        raw_rows, args.bootstrap_replicates, seed
    )
    parameter_group_rows = summarize_parameter_groups(raw_rows)
    sample_diagnostics = sampling_diagnostics(data.train_sets, references)
    candidates = _parse_positive_floats(args.candidate_norms)
    shortlist = candidate_shortlist(summary_rows)
    full_candidates = sorted(
        set(candidates) | {float(row["max_grad_norm"]) for row in shortlist}
    )
    grid_rows = clipping_grid(raw_rows, full_candidates)
    write_csv(output_dir / "per_record_gradient_norms.csv", raw_rows)
    write_csv(output_dir / "summary_by_checkpoint.csv", summary_rows)
    write_csv(output_dir / "summary_by_parameter_group.csv", parameter_group_rows)
    write_csv(output_dir / "clipping_grid.csv", grid_rows)
    render_figure(output_dir / "gradient_norm_calibration.png", raw_rows, grid_rows)

    resolved = {
        "experiment": "non_private_record_dp_clipping_calibration",
        "privacy_status": "non_private",
        "run_dir": str(run_dir) if run_dir is not None else None,
        "model_type": config["model_type"],
        "dataset_name": config["dataset_name"],
        "seed": seed,
        "device": str(device),
        "trainable_parameters": trainable_count,
        "client_ids": client_ids,
        "samples_per_client": args.samples_per_client,
        "sample_records_per_checkpoint": len(references),
        "microbatch_size": args.microbatch_size,
        "bootstrap_replicates": args.bootstrap_replicates,
        "sampling_diagnostics": sample_diagnostics,
        "checkpoints": [
            {
                "label": checkpoint.label,
                "communication_round": checkpoint.communication_round,
                "path": str(checkpoint.path) if checkpoint.path is not None else None,
            }
            for checkpoint in checkpoints
        ],
        "summary_by_checkpoint": summary_rows,
        "summary_by_parameter_group": parameter_group_rows,
        "candidate_shortlist": shortlist,
        "candidate_grid": full_candidates,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "report.md",
        summary_rows=summary_rows,
        parameter_group_rows=parameter_group_rows,
        sample_diagnostics=sample_diagnostics,
        shortlist=shortlist,
        config=config,
        run_dir=run_dir,
        output_dir=output_dir,
    )
    LOGGER.info("Calibration complete: %s", output_dir)


if __name__ == "__main__":
    main()
