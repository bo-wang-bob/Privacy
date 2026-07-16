"""Build validated, source-backed tables for the AAAI 2027 VEIL paper.

The script intentionally ignores incomplete runs and pre-fairness experiments.
It requires exact member/non-member label-histogram matching in every selected
audit and emits only aggregate CSV artifacts; raw results remain git-ignored.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from utils.fair_comparison import load_run, validate_fair_configs


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "paper" / "aaai2027" / "evidence"
DATASETS = ("flowers", "caltech101", "dtd")
SEEDS = (42, 43, 44)
ATTACKS = (
    "fedmia_loss",
    "fedmia_cosine",
    "fedmia_joint",
    "nasr_passive",
    "rmia",
    "quantile_mia",
)
METHOD_ORDER = (
    "FedAvg",
    "Prompt-DP",
    "HAMP",
    "VEIL",
    "DP-FPL",
    "FedASK",
)
CUTOFFS = {
    "flowers": datetime(2026, 7, 16, 19, 23).timestamp(),
    "caltech101": datetime(2026, 7, 16, 21, 35).timestamp(),
    # Exclude the pre-alias DTD sentinel that did not apply VEIL's final-query
    # temperature map.  All formal DTD runs are intentionally newer.
    "dtd": datetime(2026, 7, 16, 21, 52).timestamp(),
}
OFFICIAL_CONFIG = yaml.safe_load(
    (ROOT / "configs" / "veil_multidataset.yaml").read_text(encoding="utf-8")
)


def selected_values_match(
    actual: dict[str, Any], expected: dict[str, Any], keys: tuple[str, ...]
) -> bool:
    return all(actual.get(key) == expected.get(key) for key in keys)


def matches_official_method(config: dict[str, Any], method: str) -> bool:
    """Reject sweeps/ablations that happen to share a paper method label."""

    defense = config.get("defense", {})
    expected_defense = OFFICIAL_CONFIG["defense"]
    if int(config.get("gpu", -1)) not in {0, 1}:
        return False
    if method == "FedAvg":
        return config.get("aggregator") == "fedavg" and defense.get("name") == "none"
    if method == "Prompt-DP":
        keys = (
            "dp_max_grad_norm",
            "dp_noise_multiplier",
            "dp_delta",
            "reproducible_dp_noise",
        )
        return (
            config.get("aggregator") == "fedavg"
            and defense.get("name") == "prompt_dp"
            and selected_values_match(defense, expected_defense, keys)
        )
    if method == "HAMP":
        keys = (
            "hamp_true_probability",
            "hamp_entropy_weight",
            "hamp_output_temperature",
        )
        return (
            config.get("aggregator") == "fedavg"
            and defense.get("name") == "hamp"
            and selected_values_match(defense, expected_defense, keys)
        )
    if method == "VEIL":
        keys = tuple(
            key for key in expected_defense if key.startswith("local_ggeur_")
        )
        return (
            config.get("aggregator") == "fedavg"
            and defense.get("name") in {"local_ggeur", "mirage", "veil"}
            and selected_values_match(defense, expected_defense, keys)
        )
    if method == "DP-FPL":
        return (
            config.get("aggregator") == "dpfpl"
            and defense.get("name") == "none"
            and selected_values_match(
                config.get("dpfpl", {}),
                OFFICIAL_CONFIG["dpfpl"],
                tuple(OFFICIAL_CONFIG["dpfpl"]),
            )
        )
    if method == "FedASK":
        return (
            config.get("aggregator") == "fedask"
            and defense.get("name") == "none"
            and selected_values_match(
                config.get("fedask", {}),
                OFFICIAL_CONFIG["fedask"],
                tuple(OFFICIAL_CONFIG["fedask"]),
            )
        )
    return False


def method_name(config: dict[str, Any]) -> str | None:
    aggregator = str(config["aggregator"]).lower()
    defense = str(config.get("defense", {}).get("name", "none")).lower()
    if aggregator == "fedavg":
        return {
            "none": "FedAvg",
            "prompt_dp": "Prompt-DP",
            "hamp": "HAMP",
            "local_ggeur": "VEIL",
            "mirage": "VEIL",
            "veil": "VEIL",
        }.get(defense)
    if defense != "none":
        return None
    return {"dpfpl": "DP-FPL", "fedask": "FedASK"}.get(aggregator)


def audit_is_valid(summary_path: Path) -> bool:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate = summary.get("candidate_sampling", {})
    attacks = summary.get("attacks", [])
    member_histogram = candidate.get("member_label_histogram", [])
    return bool(
        candidate.get("label_histograms_matched")
        and member_histogram == candidate.get("nonmember_label_histogram")
        and sum(member_histogram) == 64
        # Earlier Flowers runs predate the explicit provenance field but used
        # the same target/other-test pools and passed exact histogram matching.
        and candidate.get("nonmember_source_priority")
        in (None, ["target_test", "other_client_test", "other_client_train"])
        and [item.get("attack") for item in attacks] == list(ATTACKS)
        and all(
            0.0 <= float(item.get("tpr_at_fpr_0.01", -1.0)) <= 1.0
            and 0.0 <= float(item.get("auc", -1.0)) <= 1.0
            and int(item.get("num_samples", 0)) > 0
            for item in attacks
        )
        and not summary.get("errors")
    )


def discover() -> list[dict[str, Any]]:
    candidates: dict[tuple[str, int, str], list[Path]] = defaultdict(list)
    for config_path in RESULTS.glob("*/run_config.yaml"):
        result_dir = config_path.parent
        summary_path = result_dir / "privacy_audit" / "summary.json"
        training_path = result_dir / "training_metrics.csv"
        if not summary_path.exists() or not training_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset = str(config.get("dataset_name", "")).lower()
        seed = int(config.get("seed", -1))
        method = method_name(config)
        if dataset not in DATASETS or seed not in SEEDS or method not in METHOD_ORDER:
            continue
        if not matches_official_method(config, method):
            continue
        if summary_path.stat().st_mtime < CUTOFFS[dataset]:
            continue
        if dataset != "flowers" and not bool(config.get("require_cuda", False)):
            continue
        if not bool(config.get("audit", {}).get("match_candidate_labels", False)):
            continue
        if not audit_is_valid(summary_path):
            continue
        candidates[(dataset, seed, method)].append(result_dir)

    expected = {
        (dataset, seed, method)
        for dataset in DATASETS
        for seed in SEEDS
        for method in METHOD_ORDER
    }
    missing = sorted(expected - set(candidates))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} completed paper runs: {missing}")

    runs = []
    for key in sorted(expected):
        selected = max(
            candidates[key],
            key=lambda path: (path / "privacy_audit" / "summary.json").stat().st_mtime,
        )
        run = load_run(selected)
        run["dataset"] = key[0]
        run["seed"] = key[1]
        run["method"] = key[2]
        runs.append(run)

    for dataset in DATASETS:
        for seed in SEEDS:
            matched = [
                run
                for run in runs
                if run["dataset"] == dataset and run["seed"] == seed
            ]
            validate_fair_configs([run["config"] for run in matched])
    return runs


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(runs: list[dict[str, Any]]) -> None:
    run_rows = []
    accounting_rows = []
    for run in runs:
        row = {
            "dataset": run["dataset"],
            "seed": run["seed"],
            "method": run["method"],
            "accuracy": run["accuracy"],
            "worst_tpr": run["worst_tpr_at_fpr_0.01"],
            "mean_tpr": run["mean_tpr_at_fpr_0.01"],
            "result_dir": Path(run["path"]).name,
        }
        row.update(
            {attack: run["attacks"][attack]["tpr_at_fpr_0.01"] for attack in ATTACKS}
        )
        run_rows.append(row)
        result_dir = Path(run["path"])
        defense_summary = json.loads(
            (result_dir / "defense_summary.json").read_text(encoding="utf-8")
        )
        method_summary = json.loads(
            (result_dir / "federated_method_summary.json").read_text(encoding="utf-8")
        )
        accounts = []
        if run["method"] == "Prompt-DP":
            account = defense_summary.get("privacy_accounting", {})
            accounts.append(("epsilon", account.get("epsilon_upper_bound"), account))
        elif run["method"] == "DP-FPL":
            account = method_summary.get("privacy_accounting", {})
            accounts.extend(
                (
                    ("local_epsilon", account.get("local_epsilon_upper_bound"), account),
                    ("global_epsilon", account.get("global_epsilon_upper_bound"), account),
                )
            )
        elif run["method"] == "FedASK":
            account = method_summary.get("privacy_accounting", {})
            accounts.append(("epsilon", account.get("epsilon_upper_bound"), account))
        for scope, value, account in accounts:
            if value is None:
                raise RuntimeError(f"Missing {scope} accounting in {result_dir}")
            accounting_rows.append(
                {
                    "dataset": run["dataset"],
                    "seed": run["seed"],
                    "method": run["method"],
                    "scope": scope,
                    "epsilon_upper_bound": value,
                    "delta": account.get("delta"),
                    "formal_dp_enabled": account.get("formal_dp_enabled"),
                }
            )
    write_csv(
        OUTPUT / "run_level.csv",
        [
            "dataset",
            "seed",
            "method",
            "accuracy",
            "worst_tpr",
            "mean_tpr",
            *ATTACKS,
            "result_dir",
        ],
        run_rows,
    )
    write_csv(
        OUTPUT / "privacy_accounting.csv",
        [
            "dataset",
            "seed",
            "method",
            "scope",
            "epsilon_upper_bound",
            "delta",
            "formal_dp_enabled",
        ],
        accounting_rows,
    )
    accounting_summary_rows = []
    accounting_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accounting_rows:
        accounting_groups[(row["method"], row["scope"])].append(row)
    for (method, scope), group in sorted(accounting_groups.items()):
        deltas = {float(row["delta"]) for row in group}
        formal_flags = {bool(row["formal_dp_enabled"]) for row in group}
        if len(deltas) != 1 or formal_flags != {True}:
            raise RuntimeError(
                f"Inconsistent privacy accounting for {(method, scope)}: "
                f"deltas={deltas}, formal_flags={formal_flags}"
            )
        epsilon = [float(row["epsilon_upper_bound"]) for row in group]
        accounting_summary_rows.append(
            {
                "method": method,
                "scope": scope,
                "runs": len(group),
                "epsilon_min": min(epsilon),
                "epsilon_max": max(epsilon),
                "delta": deltas.pop(),
                "formal_dp_enabled": True,
            }
        )
    write_csv(
        OUTPUT / "privacy_accounting_summary.csv",
        [
            "method",
            "scope",
            "runs",
            "epsilon_min",
            "epsilon_max",
            "delta",
            "formal_dp_enabled",
        ],
        accounting_summary_rows,
    )

    aggregate_rows = []
    attack_rows = []
    for dataset in DATASETS:
        for method in METHOD_ORDER:
            group = [
                row
                for row in run_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            accuracy_mean, accuracy_std = mean_std([row["accuracy"] for row in group])
            worst_mean, worst_std = mean_std([row["worst_tpr"] for row in group])
            privacy_mean, privacy_std = mean_std([row["mean_tpr"] for row in group])
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seeds": len(group),
                    "accuracy_mean": accuracy_mean,
                    "accuracy_std": accuracy_std,
                    "worst_tpr_mean": worst_mean,
                    "worst_tpr_std": worst_std,
                    "mean_tpr_mean": privacy_mean,
                    "mean_tpr_std": privacy_std,
                }
            )
            for attack in ATTACKS:
                attack_mean, attack_std = mean_std([row[attack] for row in group])
                attack_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "attack": attack,
                        "tpr_mean": attack_mean,
                        "tpr_std": attack_std,
                    }
                )
    write_csv(
        OUTPUT / "aggregate.csv",
        [
            "dataset",
            "method",
            "seeds",
            "accuracy_mean",
            "accuracy_std",
            "worst_tpr_mean",
            "worst_tpr_std",
            "mean_tpr_mean",
            "mean_tpr_std",
        ],
        aggregate_rows,
    )
    write_csv(
        OUTPUT / "attack_aggregate.csv",
        ["dataset", "method", "attack", "tpr_mean", "tpr_std"],
        attack_rows,
    )


def main() -> None:
    runs = discover()
    build_outputs(runs)
    print(f"Validated {len(runs)} runs and wrote tables to {OUTPUT}")


if __name__ == "__main__":
    main()
