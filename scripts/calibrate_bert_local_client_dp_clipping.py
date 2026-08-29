#!/usr/bin/env python3
"""Run non-private BERT FedSGD and calibrate client-upload clipping norm S.

The experiment follows the ordinary full-participation BERT Adapter training
protocol.  It observes, but never changes, each client's exact one-batch mean
gradient before aggregation.  The resulting statistics are explicitly
non-private research artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aggregator.aggregator_builder import build_aggregator
from scripts.run_fedllm_adapter import validate_config as validate_text_config
from servers.serverbase import ServerBase
from trainmodel.transformer_adapter import TransformerAdapterClassifier
from utils.privacy_accounting import calibrate_gaussian_noise
from utils.text_data_loader import load_federated_text_classification


LOGGER = logging.getLogger(__name__)
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
RECOMMENDATION_QUANTILES = (
    ("p50", 0.5, "aggressive"),
    ("p75", 0.75, "balanced"),
    ("p90", 0.9, "conservative"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train non-private BERT/SST-5 FedSGD while measuring every "
            "selected client's exact batch-gradient joint L2 norm."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/bert_base_sst5_adapter.yaml",
        help="Base BERT configuration; defense and audit are forced off.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Non-private training rounds; defaults to the base configuration.",
    )
    parser.add_argument(
        "--accounting-rounds",
        type=int,
        help=(
            "Rounds used only to report DP noise magnitudes. Defaults to the "
            "unmodified base configuration, normally 500."
        ),
    )
    parser.add_argument(
        "--client-ids",
        default="all",
        help=(
            "Clients included in norm summaries, as comma-separated ids or "
            "'all'. All clients still train to preserve the protocol."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:1.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--target-epsilon", type=float, default=3.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Client-cluster bootstrap replicates for pooled quantiles.",
    )
    parser.add_argument(
        "--output-dir",
        help="Defaults to a timestamped directory under analysis_scripts/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved experiment without training.",
    )
    return parser.parse_args()


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _parse_client_ids(value: str, total_users: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(total_users))
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--client-ids must not be empty.")
    if len(parsed) != len(set(parsed)):
        raise ValueError("--client-ids must not contain duplicates.")
    invalid = [client_id for client_id in parsed if not 0 <= client_id < total_users]
    if invalid:
        raise ValueError(f"Client IDs are out of range: {invalid}")
    return sorted(parsed)


def resolve_device(value: str, gpu: int, require_cuda: bool) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(normalized)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required; pass --no-require-cuda for a pilot.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def resolve_config(args: argparse.Namespace) -> tuple[dict, int, list[int]]:
    config_path = _resolve_repo_path(args.config)
    config = _read_yaml(config_path)
    if str(config.get("model_type", "")).lower() != "bert_adapter":
        raise ValueError("Calibration requires model_type=bert_adapter.")
    if str(config.get("dataset_name", "")).lower() != "sst5":
        raise ValueError("Calibration currently requires dataset_name=sst5.")
    configured_rounds = int(config.get("num_global_iters", 0))
    accounting_rounds = int(args.accounting_rounds or configured_rounds)
    if args.rounds is not None:
        config["num_global_iters"] = int(args.rounds)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if int(config.get("num_global_iters", 0)) <= 0:
        raise ValueError("Training rounds must be positive.")
    if accounting_rounds <= 0:
        raise ValueError("--accounting-rounds must be positive.")
    if args.target_epsilon <= 0:
        raise ValueError("--target-epsilon must be positive.")
    if not 0 < args.delta < 1:
        raise ValueError("--delta must be in (0, 1).")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative.")

    config["config_path"] = str(config_path)
    config["save_models"] = False
    config["defense"] = {"name": "none"}
    config["audit"] = {
        "enabled": False,
        "strict": False,
        "attacks": [],
        "exact_batch_membership_attacks": [],
        "target_client_id": 0,
        "audit_client_ids": [0],
        "candidate_sampling": "legacy",
        "training_health_check": False,
    }
    config["projres"] = {
        "enabled": False,
        "evaluation_round": "last",
        "decision_mode": "ranking",
        "threshold": None,
    }
    validate_text_config(config)
    client_ids = _parse_client_ids(args.client_ids, int(config["total_users"]))
    return config, accounting_rounds, client_ids


def parameter_group(name: str) -> str:
    if ".adapter.down." in name:
        return "adapter_down"
    if ".adapter.up." in name:
        return "adapter_up"
    if name.startswith("classifier."):
        return "classifier"
    return "other_trainable"


def client_gradient_norm_row(
    *,
    round_index: int,
    client_id: int,
    gradients: dict[str, torch.Tensor],
    sample_count: int,
    learning_rate: float,
) -> dict:
    group_squares: dict[str, float] = {}
    for name, gradient in gradients.items():
        group = parameter_group(name)
        group_squares[group] = group_squares.get(group, 0.0) + float(
            gradient.detach().float().square().sum()
        )
    joint_square = sum(group_squares.values())
    row = {
        "communication_round": int(round_index) + 1,
        "client_id": int(client_id),
        "batch_size": int(sample_count),
        "learning_rate": float(learning_rate),
        "joint_grad_norm": math.sqrt(max(joint_square, 0.0)),
    }
    for group in sorted(group_squares):
        row[f"{group}_grad_norm"] = math.sqrt(max(group_squares[group], 0.0))
    return row


class ClientGradientNormObserver:
    def __init__(self, client_ids: Iterable[int], expected_messages: int):
        self.client_ids = set(int(client_id) for client_id in client_ids)
        self.expected_messages = int(expected_messages)
        self.rows: list[dict] = []

    def __call__(self, **payload) -> None:
        if int(payload["client_id"]) not in self.client_ids:
            return
        self.rows.append(client_gradient_norm_row(**payload))
        count = len(self.rows)
        log_interval = max(len(self.client_ids) * 10, 1)
        if count % log_interval == 0 or count == self.expected_messages:
            LOGGER.info(
                "Measured %d/%d client uploads",
                count,
                self.expected_messages,
            )


def _metric_values(norms: np.ndarray) -> dict[str, float]:
    if norms.size <= 0:
        raise ValueError("Gradient-norm summaries require at least one message.")
    values = {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms, ddof=1)) if norms.size > 1 else 0.0,
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
    }
    for quantile in QUANTILES:
        values[f"p{int(quantile * 100):02d}"] = float(
            np.quantile(norms, quantile)
        )
    return values


def _summarize_rows(rows: list[dict], **identity) -> dict:
    norms = np.asarray(
        [float(row["joint_grad_norm"]) for row in rows], dtype=np.float64
    )
    return {
        **identity,
        "messages": len(rows),
        "clients": len({int(row["client_id"]) for row in rows}),
        "rounds": len({int(row["communication_round"]) for row in rows}),
        **_metric_values(norms),
    }


def summarize_by_round(rows: list[dict]) -> list[dict]:
    return [
        _summarize_rows(
            [row for row in rows if int(row["communication_round"]) == round_id],
            communication_round=round_id,
        )
        for round_id in sorted({int(row["communication_round"]) for row in rows})
    ]


def summarize_by_client(rows: list[dict]) -> list[dict]:
    return [
        _summarize_rows(
            [row for row in rows if int(row["client_id"]) == client_id],
            client_id=client_id,
        )
        for client_id in sorted({int(row["client_id"]) for row in rows})
    ]


def phase_name(communication_round: int, total_rounds: int) -> str:
    phase_index = min(2, (max(1, communication_round) - 1) * 3 // total_rounds)
    return ("early", "middle", "late")[phase_index]


def summarize_by_phase(rows: list[dict], total_rounds: int) -> list[dict]:
    output = []
    for phase in ("early", "middle", "late"):
        selected = [
            row
            for row in rows
            if phase_name(int(row["communication_round"]), total_rounds) == phase
        ]
        if selected:
            output.append(_summarize_rows(selected, phase=phase))
    return output


def client_cluster_bootstrap(
    rows: list[dict], replicates: int, seed: int
) -> dict[str, tuple[float, float]]:
    if replicates <= 0:
        return {}
    client_ids = sorted({int(row["client_id"]) for row in rows})
    if len(client_ids) < 2:
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
        for client_id in client_ids
    }
    generator = np.random.default_rng(seed)
    bootstrap_values = {"p50": [], "p75": [], "p90": []}
    for _ in range(replicates):
        selected = generator.choice(client_ids, size=len(client_ids), replace=True)
        norms = np.concatenate([by_client[int(client_id)] for client_id in selected])
        metrics = _metric_values(norms)
        for metric in bootstrap_values:
            bootstrap_values[metric].append(metrics[metric])
    return {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in bootstrap_values.items()
    }


def round_significant(value: float, digits: int = 3) -> float:
    if value <= 0 or not math.isfinite(value):
        raise ValueError("Recommended clipping norms must be finite and positive.")
    decimals = digits - int(math.floor(math.log10(abs(value)))) - 1
    return float(round(value, decimals))


def clipping_statistics(norms: np.ndarray, threshold: float) -> dict[str, float]:
    factors = np.minimum(1.0, threshold / np.maximum(norms, 1e-30))
    clipped = norms * factors
    return {
        "clip_fraction": float(np.mean(norms > threshold)),
        "mean_clip_factor": float(np.mean(factors)),
        "mean_clipped_norm": float(np.mean(clipped)),
        "norm_magnitude_retained": float(
            np.sum(clipped) / max(float(np.sum(norms)), 1e-30)
        ),
    }


def recommend_thresholds(
    rows: list[dict],
    *,
    target_epsilon: float,
    delta: float,
    accounting_rounds: int,
    total_users: int,
) -> tuple[list[dict], list[dict], float]:
    norms = np.asarray(
        [float(row["joint_grad_norm"]) for row in rows], dtype=np.float64
    )
    noise_multiplier = calibrate_gaussian_noise(
        target_epsilon=target_epsilon,
        steps=accounting_rounds,
        delta=delta,
    )
    recommendations = []
    raw_thresholds = {}
    for label, quantile, role in RECOMMENDATION_QUANTILES:
        raw_threshold = float(np.quantile(norms, quantile))
        threshold = round_significant(raw_threshold)
        raw_thresholds[label] = raw_threshold
        recommendations.append(
            {
                "quantile": label.upper(),
                "role": role,
                "raw_threshold": raw_threshold,
                "recommended_s": threshold,
                **clipping_statistics(norms, threshold),
                "target_epsilon": target_epsilon,
                "delta": delta,
                "accounting_rounds": accounting_rounds,
                "noise_multiplier": noise_multiplier,
                "local_noise_std_per_coordinate": noise_multiplier * threshold,
                "aggregate_noise_std_per_coordinate": (
                    noise_multiplier * threshold / math.sqrt(total_users)
                ),
            }
        )
    grid_values = {
        round_significant(0.5 * raw_thresholds["p50"]),
        *(float(row["recommended_s"]) for row in recommendations),
        round_significant(2.0 * raw_thresholds["p90"]),
    }
    grid = [
        {
            "max_update_norm": threshold,
            **clipping_statistics(norms, threshold),
            "local_noise_std_per_coordinate": noise_multiplier * threshold,
            "aggregate_noise_std_per_coordinate": (
                noise_multiplier * threshold / math.sqrt(total_users)
            ),
        }
        for threshold in sorted(grid_values)
    ]
    return recommendations, grid, noise_multiplier


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_figure(
    path: Path,
    rows: list[dict],
    phase_rows: list[dict],
    recommendations: list[dict],
) -> None:
    import matplotlib.pyplot as plt

    norms = np.sort(
        np.asarray([float(row["joint_grad_norm"]) for row in rows], dtype=np.float64)
    )
    cumulative = np.arange(1, norms.size + 1) / norms.size
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(norms, cumulative, color="#2F6BFF", linewidth=2)
    colors = {"P50": "#2F6BFF", "P75": "#E88927", "P90": "#C55A9B"}
    for row in recommendations:
        label = str(row["quantile"])
        axes[0].axvline(
            float(row["recommended_s"]),
            color=colors[label],
            linestyle="--",
            linewidth=1.5,
            label=f"{label}: S={float(row['recommended_s']):g}",
        )
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("Client batch-mean joint gradient norm (log scale)")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[0].set_title("Exact client-upload gradient norms")
    axes[0].grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axes[0].legend(frameon=False, fontsize=8)

    phases = [str(row["phase"]) for row in phase_rows]
    positions = np.arange(len(phases))
    for metric, color in (("p50", "#2F6BFF"), ("p75", "#E88927"), ("p90", "#C55A9B")):
        axes[1].plot(
            positions,
            [float(row[metric]) for row in phase_rows],
            marker="o",
            linewidth=2,
            color=color,
            label=metric.upper(),
        )
    axes[1].set_xticks(positions, phases)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training phase")
    axes[1].set_ylabel("Gradient-norm quantile (log scale)")
    axes[1].set_title("Norm drift across training")
    axes[1].grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("BERT Adapter / SST-5 local client-DP clipping calibration")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    path: Path,
    *,
    overall: dict,
    phase_rows: list[dict],
    recommendations: list[dict],
    grid_rows: list[dict],
    config: dict,
    accounting_rounds: int,
    client_ids: list[int],
) -> None:
    lines = [
        "# BERT 本地客户端级 DP 裁剪阈值非隐私校准",
        "",
        "## 结论口径",
        "",
        (
            "本实验运行原始无 DP、全参与、one-batch FedSGD，并在聚合前只读观察每个"
            "客户端实际上传的 batch-mean 梯度。联合范数覆盖所有 BERT Adapter 与分类头"
            "参数；训练过程没有裁剪、没有加噪，也没有成员推理审计。"
        ),
        "",
        "## 汇总分位数",
        "",
        "| 消息数 | 客户端 | 轮数 | P50 | P75 | P90 | P95 | P99 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| {messages} | {clients} | {rounds} | {p50:.5g} | {p75:.5g} | "
            "{p90:.5g} | {p95:.5g} | {p99:.5g} |"
        ).format(**overall),
        "",
        "## 建议的正式 S 候选",
        "",
        "| 依据 | 角色 | 推荐 S | 反事实裁剪率 | 平均裁剪因子 | 本地单坐标噪声 std | 聚合后噪声 std |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in recommendations:
        lines.append(
            "| {quantile} | {role} | {recommended_s:.5g} | {clip_fraction:.2%} | "
            "{mean_clip_factor:.4f} | {local_noise_std_per_coordinate:.5g} | "
            "{aggregate_noise_std_per_coordinate:.5g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "默认建议先把 `P75` 作为 balanced 候选，再用 `P50/P90` 检查强裁剪和弱裁剪两侧。它们不是效用最优性的证明，最终仍需在相同 epsilon、delta、轮数和 seed 下比较准确率与 ProjRes。",
            "",
            "## 训练阶段漂移",
            "",
            "| 阶段 | 消息数 | P50 | P75 | P90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in phase_rows:
        lines.append(
            "| {phase} | {messages} | {p50:.5g} | {p75:.5g} | {p90:.5g} |".format(
                **row
            )
        )
    grid_values = ",".join(f"{float(row['max_update_norm']):g}" for row in grid_rows)
    lines.extend(
        [
            "",
            "## 下一步命令",
            "",
            "```bash",
            "bash scripts/run_bert_local_client_dp_projres_sweep.sh \\",
            f"  --epsilons {float(recommendations[0]['target_epsilon']):g} \\",
            f"  --max-client-update-norms {grid_values} \\",
            "  --seeds 42 --no-nondp --dry-run",
            "```",
            "",
            "## 限制",
            "",
            f"- 实际训练轮数：`{config['num_global_iters']}`；噪声幅度按 `{accounting_rounds}` 轮预算计算。",
            f"- 汇总客户端：`{','.join(map(str, client_ids))}`；训练仍使用全部 `{config['total_users']}` 个客户端。",
            "- 这是非隐私统计。如果这些阈值来自敏感训练集并被公开，阈值选择本身不能宣称零隐私成本；正式论文应使用公开代理数据、预注册候选网格，或另行核算选择过程。",
            "- 客户端上传可被服务器观察，因此表中的本地噪声 std 是威胁模型中的直接噪声；除以 sqrt(K) 的聚合值只描述模型效用，不削弱单客户端上传所需的本地保护。",
            "",
            "## 产物",
            "",
            "- `client_batch_gradient_norms.csv`：逐轮逐客户端原始联合范数。",
            "- `summary_by_round.csv`、`summary_by_client.csv`、`summary_by_phase.csv`：分层汇总。",
            "- `recommended_s.csv`、`clipping_grid.csv`：正式候选与反事实裁剪率。",
            "- `client_update_norm_calibration.png`：经验 CDF 与训练阶段漂移。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config, accounting_rounds, client_ids = resolve_config(args)
    seed = int(config.get("seed", 42))
    model_path = _resolve_repo_path(config["model_path"])
    dataset_path = _resolve_repo_path(config["dataset_path"])
    plan = {
        "experiment": "non_private_local_client_dp_clipping_calibration",
        "privacy_status": "non_private",
        "config": str(config["config_path"]),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "training_rounds": int(config["num_global_iters"]),
        "accounting_rounds": accounting_rounds,
        "total_users": int(config["total_users"]),
        "sample_users": int(config["sample_users"]),
        "batch_size": int(config["batch_size"]),
        "observed_client_ids": client_ids,
        "target_epsilon": args.target_epsilon,
        "delta": args.delta,
        "seed": seed,
        "device": args.device if args.device != "auto" else f"cuda:{args.gpu}",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    device = resolve_device(args.device, args.gpu, args.require_cuda)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        _resolve_repo_path(args.output_dir)
        if args.output_dir
        else REPOSITORY_ROOT
        / "analysis_scripts"
        / f"bert_local_client_dp_clipping_calibration_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        ],
        force=True,
    )
    LOGGER.info("Calibration output: %s", output_dir)
    LOGGER.info("Resolved plan: %s", plan)

    data = load_federated_text_classification(
        dataset_name=str(config["dataset_name"]),
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        num_users=int(config["total_users"]),
        seed=seed,
        max_length=int(config.get("max_length", 128)),
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
    model.classnames = list(data.class_names)
    expected_messages = int(config["num_global_iters"]) * len(client_ids)
    observer = ClientGradientNormObserver(client_ids, expected_messages)
    optimization = dict(config.get("optimization", {}))
    method_config = {
        "client_optimizer": str(optimization.get("client_optimizer", "sgd")).lower(),
        "momentum": float(optimization.get("momentum", 0.0)),
        "weight_decay": float(optimization.get("weight_decay", 0.0)),
        "max_grad_norm": float(optimization.get("max_grad_norm", 0.0)),
        "seed": seed,
    }
    resolved_config = dict(config)
    resolved_config.update(
        {
            "model_path": str(model_path),
            "dataset_path": str(dataset_path),
            "device": str(device),
            "analysis": plan,
        }
    )
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_config, file, sort_keys=False, allow_unicode=True)

    server = ServerBase(
        device=device,
        dataset_name=str(config["dataset_name"]),
        train_sets=data.train_sets,
        test_sets=data.test_sets,
        class_names=data.class_names,
        model=model,
        batch_size=int(config["batch_size"]),
        eval_batch_size=int(config["eval_batch_size"]),
        learning_rate=float(config["learning_rate"]),
        learning_rate_decay=float(config.get("learning_rate_decay", 1.0)),
        learning_rate_decay_interval=int(
            config.get("learning_rate_decay_interval", 1)
        ),
        num_glob_iters=int(config["num_global_iters"]),
        local_epochs=1,
        total_users=int(config["total_users"]),
        results_dir=str(output_dir),
        user_per_round=int(config["sample_users"]),
        aggregator=build_aggregator(
            "fedsgd", device=device, aggregation_weighting="uniform"
        ),
        save_models=False,
        collate_fn=data.collate_fn,
        eval_interval=int(config["eval_interval"]),
        audit_config=config["audit"],
        projres_config=config["projres"],
        defense_config=config["defense"],
        method_config=method_config,
        client_gradient_observer=observer,
    )
    server.train()
    if len(observer.rows) != expected_messages:
        raise RuntimeError(
            f"Expected {expected_messages} observed uploads, got {len(observer.rows)}."
        )

    overall = _summarize_rows(observer.rows, scope="all_messages")
    intervals = client_cluster_bootstrap(
        observer.rows, args.bootstrap_replicates, seed
    )
    for metric, (lower, upper) in intervals.items():
        overall[f"{metric}_ci95_low"] = lower
        overall[f"{metric}_ci95_high"] = upper
    round_rows = summarize_by_round(observer.rows)
    client_rows = summarize_by_client(observer.rows)
    phase_rows = summarize_by_phase(
        observer.rows, int(config["num_global_iters"])
    )
    recommendations, grid_rows, noise_multiplier = recommend_thresholds(
        observer.rows,
        target_epsilon=args.target_epsilon,
        delta=args.delta,
        accounting_rounds=accounting_rounds,
        total_users=int(config["total_users"]),
    )
    write_csv(output_dir / "client_batch_gradient_norms.csv", observer.rows)
    write_csv(output_dir / "summary_overall.csv", [overall])
    write_csv(output_dir / "summary_by_round.csv", round_rows)
    write_csv(output_dir / "summary_by_client.csv", client_rows)
    write_csv(output_dir / "summary_by_phase.csv", phase_rows)
    write_csv(output_dir / "recommended_s.csv", recommendations)
    write_csv(output_dir / "clipping_grid.csv", grid_rows)
    render_figure(
        output_dir / "client_update_norm_calibration.png",
        observer.rows,
        phase_rows,
        recommendations,
    )
    summary = {
        **plan,
        "device": str(device),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "message_definition": "one_batch_mean_joint_trainable_gradient",
        "noise_multiplier": noise_multiplier,
        "overall": overall,
        "by_phase": phase_rows,
        "recommendations": recommendations,
        "recommended_default": next(
            row for row in recommendations if row["quantile"] == "P75"
        ),
        "candidate_grid": grid_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir / "report.md",
        overall=overall,
        phase_rows=phase_rows,
        recommendations=recommendations,
        grid_rows=grid_rows,
        config=config,
        accounting_rounds=accounting_rounds,
        client_ids=client_ids,
    )
    LOGGER.info(
        "Calibration complete | P50=%g | P75=%g | P90=%g | output=%s",
        overall["p50"],
        overall["p75"],
        overall["p90"],
        output_dir,
    )


if __name__ == "__main__":
    main()
