"""Validate and summarize like-for-like privacy experiment result directories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import yaml


FAIR_CONFIG_PATHS = (
    "train_mode",
    "dataset_name",
    "batch_size",
    "eval_batch_size",
    "learning_rate",
    "num_global_iters",
    "local_epochs",
    "total_users",
    "sample_users",
    "dirichlet_alpha",
    "seed",
    "fpl_shots",
    "n_ctx",
    "class_specific_ctx",
    "audit.enabled",
    "audit.audit_view",
    "audit.strict",
    "audit.target_client_id",
    "audit.ensure_target_participation",
    "audit.attacks",
    "audit.max_samples_per_group",
    "audit.audit_interval",
    "audit.calibration_fraction",
    "audit.auxiliary_fraction",
    "audit.fedmia_loss_aggregation",
    "audit.fedmia_cosine_aggregation",
    "audit.fedmia_joint_components",
    "audit.rmia_offline_a",
    "audit.rmia_gamma",
    "audit.qmia_quantile",
    "audit.qmia_epochs",
    "audit.qmia_learning_rate",
)

FAIR_CONFIG_DEFAULTS = {
    "audit.fedmia_loss_aggregation": "mean",
    "audit.fedmia_cosine_aggregation": "mean",
    "audit.fedmia_joint_components": [
        "confidence_z_mean",
        "cosine_z_max",
        "cosine_z_late3",
    ],
}


def _nested(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            if path in FAIR_CONFIG_DEFAULTS:
                return FAIR_CONFIG_DEFAULTS[path]
            raise KeyError(f"Missing fair-comparison field {path!r}.")
        value = value[part]
    return value


def validate_fair_configs(
    configs: list[dict[str, Any]], ignored_paths: tuple[str, ...] = ()
) -> None:
    """Raise when any scenario or audit-budget field differs."""
    if len(configs) < 2:
        raise ValueError("Fair comparison requires at least two runs.")
    reference = configs[0]
    mismatches = []
    for index, config in enumerate(configs[1:], start=1):
        for path in FAIR_CONFIG_PATHS:
            if path in ignored_paths:
                continue
            expected = _nested(reference, path)
            actual = _nested(config, path)
            if actual != expected:
                mismatches.append(
                    f"run {index} {path}: expected {expected!r}, got {actual!r}"
                )
    if mismatches:
        raise ValueError("Runs are not like-for-like:\n" + "\n".join(mismatches))


def _method_name(config: dict[str, Any]) -> str:
    aggregator = str(config["aggregator"]).lower()
    defense = str(config.get("defense", {}).get("name", "none")).lower()
    if aggregator == "fedavg" and defense == "local_ggeur":
        return "Local-GGEUR"
    if defense == "none":
        return {"fedavg": "FedAvg", "dpfpl": "DP-FPL", "fedask": "FedASK"}.get(
            aggregator, aggregator
        )
    return f"{aggregator}+{defense}"


def load_run(result_dir: str | Path) -> dict[str, Any]:
    path = Path(result_dir)
    with (path / "run_config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with (path / "training_metrics.csv").open(encoding="utf-8", newline="") as file:
        training = list(csv.DictReader(file))
    if not training:
        raise ValueError(f"No training metrics in {path}.")
    with (path / "privacy_audit" / "summary.json").open(encoding="utf-8") as file:
        audit = json.load(file)
    attacks = {
        item["attack"]: {
            "tpr_at_fpr_0.01": float(item["tpr_at_fpr_0.01"]),
            "auc": float(item["auc"]),
            "num_samples": int(item["num_samples"]),
        }
        for item in audit["attacks"]
    }
    last = training[-1]
    return {
        "path": str(path),
        "config": config,
        "method": _method_name(config),
        "accuracy": float(last["accuracy"]),
        "loss": float(last["loss"]),
        "round": int(last["round"]),
        "test_samples": int(last["samples"]),
        "attacks": attacks,
        "worst_tpr_at_fpr_0.01": max(
            item["tpr_at_fpr_0.01"] for item in attacks.values()
        ),
        "mean_tpr_at_fpr_0.01": sum(
            item["tpr_at_fpr_0.01"] for item in attacks.values()
        )
        / len(attacks),
    }


def markdown_table(runs: list[dict[str, Any]]) -> str:
    attack_names = list(runs[0]["attacks"])
    rows = [
        "| Method | Accuracy | Worst TPR@1%FPR | Mean TPR@1%FPR | "
        + " | ".join(attack_names)
        + " |",
        "|---|---:|---:|---:|" + "---:|" * len(attack_names),
    ]
    for run in runs:
        metrics = [run["attacks"][name]["tpr_at_fpr_0.01"] for name in attack_names]
        rows.append(
            f"| {run['method']} | {run['accuracy']:.4f} | "
            f"{run['worst_tpr_at_fpr_0.01']:.4f} | "
            f"{run['mean_tpr_at_fpr_0.01']:.4f} | "
            + " | ".join(f"{value:.4f}" for value in metrics)
            + " |"
        )
    return "\n".join(rows)


def aggregate_markdown_table(runs: list[dict[str, Any]]) -> str:
    """Aggregate repeated, seed-matched runs as mean +/- sample standard deviation."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["method"], []).append(run)
    attack_names = list(runs[0]["attacks"])

    def cell(values: list[float]) -> str:
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{statistics.mean(values):.4f} +/- {deviation:.4f}"

    rows = [
        "| Method | Seeds | Accuracy | Worst TPR@1%FPR | Mean TPR@1%FPR | "
        + " | ".join(attack_names)
        + " |",
        "|---|---:|---:|---:|---:|" + "---:|" * len(attack_names),
    ]
    for method, method_runs in grouped.items():
        attack_cells = [
            cell([run["attacks"][name]["tpr_at_fpr_0.01"] for run in method_runs])
            for name in attack_names
        ]
        rows.append(
            f"| {method} | {len(method_runs)} | "
            f"{cell([run['accuracy'] for run in method_runs])} | "
            f"{cell([run['worst_tpr_at_fpr_0.01'] for run in method_runs])} | "
            f"{cell([run['mean_tpr_at_fpr_0.01'] for run in method_runs])} | "
            + " | ".join(attack_cells)
            + " |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify shared settings and summarize privacy result directories."
    )
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Validate method matching within each seed and report mean +/- std.",
    )
    args = parser.parse_args()
    runs = [load_run(path) for path in args.result_dirs]
    if args.aggregate_seeds:
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for run in runs:
            by_seed.setdefault(int(run["config"]["seed"]), []).append(run)
        expected_methods = None
        for seed, seed_runs in by_seed.items():
            methods = {run["method"] for run in seed_runs}
            if expected_methods is None:
                expected_methods = methods
            elif methods != expected_methods:
                raise ValueError(
                    f"Seed {seed} methods {sorted(methods)} do not match "
                    f"{sorted(expected_methods)}."
                )
            validate_fair_configs([run["config"] for run in seed_runs])
        validate_fair_configs(
            [run["config"] for run in runs], ignored_paths=("seed",)
        )
        print(aggregate_markdown_table(runs))
    else:
        validate_fair_configs([run["config"] for run in runs])
        print(markdown_table(runs))


if __name__ == "__main__":
    main()
