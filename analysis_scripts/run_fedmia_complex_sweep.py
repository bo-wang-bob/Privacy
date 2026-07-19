#!/usr/bin/env python3
"""Run and summarize resumable multi-dataset FedMIA prompt experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepJob:
    run_id: str
    config: dict[str, Any]
    dataset: str
    method: str
    seed: int
    target_client_id: int
    defense: str
    defense_parameters: dict[str, Any]
    run_root: Path
    config_path: Path


@dataclass
class ActiveRun:
    job: SweepJob
    gpu: int
    process: subprocess.Popen
    log_file: Any


@dataclass(frozen=True)
class GPUStatus:
    index: int
    free_memory_mb: int
    utilization_percent: int


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def _resolve_path(value: str | Path, base: Path = REPOSITORY_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _grid_rows(grid: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = []
    for key in keys:
        candidates = grid[key]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Defense grid {key!r} must be a non-empty list.")
        values.append(candidates)
    return [dict(zip(keys, row)) for row in itertools.product(*values)]


def _stable_run_id(
    defense: str,
    seed: int,
    target_client_id: int,
    config: dict[str, Any],
    dataset_prefix: str | None = None,
    method_prefix: str | None = None,
) -> str:
    canonical = copy.deepcopy(config)
    canonical.pop("gpu", None)
    canonical.pop("results_dir", None)
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    prefixes = [
        value for value in (dataset_prefix, method_prefix) if value is not None
    ]
    prefix = "" if not prefixes else f"{'_'.join(prefixes)}_"
    return f"{prefix}{defense}_seed{seed}_target{target_client_id}_{digest}"


def build_jobs(
    spec: dict[str, Any],
    spec_path: Path,
    data_root: str | None = None,
    cache_dir: str | None = None,
) -> tuple[list[SweepJob], Path]:
    base_config_path = _resolve_path(
        spec.get("base_config", "configs/fedmia_prompt_benchmark.yaml")
    )
    base_config = _load_yaml(base_config_path)
    common = spec.get("common", {})
    if not isinstance(common, dict):
        raise ValueError("sweep.common must be a mapping.")
    base_config = _deep_merge(base_config, common)
    sweep_name = str(spec.get("name", spec_path.stem))
    results_root = _resolve_path(spec.get("results_root", f"results/{sweep_name}"))
    seeds = [int(value) for value in spec.get("seeds", [42])]
    targets = [int(value) for value in spec.get("target_client_ids", [0])]
    defenses = spec.get("defenses", [])
    if not seeds or not targets or not isinstance(defenses, list) or not defenses:
        raise ValueError("The sweep needs seeds, target_client_ids, and defenses.")

    dataset_entries = spec.get("datasets")
    multi_dataset_spec = dataset_entries is not None
    if dataset_entries is None:
        dataset_entries = [
            {"name": str(base_config.get("dataset_name", "dataset")), "overrides": {}}
        ]
    if not isinstance(dataset_entries, list) or not dataset_entries:
        raise ValueError("sweep.datasets must be a non-empty list when provided.")

    method_entries = spec.get("methods")
    multi_method_spec = method_entries is not None
    if method_entries is not None and (
        not isinstance(method_entries, list) or not method_entries
    ):
        raise ValueError("sweep.methods must be a non-empty list when provided.")

    jobs: list[SweepJob] = []
    seen_ids: set[str] = set()
    for dataset_entry in dataset_entries:
        if not isinstance(dataset_entry, dict) or "name" not in dataset_entry:
            raise ValueError("Each dataset entry must be a mapping with a name.")
        dataset_overrides = dataset_entry.get("overrides", {})
        if not isinstance(dataset_overrides, dict):
            raise ValueError("Dataset overrides must be a mapping.")
        dataset_config = _deep_merge(base_config, dataset_overrides)
        if "dataset_name" not in dataset_overrides:
            dataset_config["dataset_name"] = str(dataset_entry["name"])
        dataset = str(dataset_config["dataset_name"]).lower()
        if data_root is not None:
            dataset_config["data_root"] = str(_resolve_path(data_root))
        if cache_dir is not None:
            dataset_config["cache_dir"] = str(_resolve_path(cache_dir))

        expanded_methods = method_entries or [
            {"name": str(dataset_config.get("aggregator", "fedavg"))}
        ]
        for method_entry in expanded_methods:
            if isinstance(method_entry, str):
                method = method_entry.lower()
                method_overrides: dict[str, Any] = {}
            elif isinstance(method_entry, dict) and "name" in method_entry:
                method = str(method_entry["name"]).lower()
                method_overrides = method_entry.get("overrides", {})
                if not isinstance(method_overrides, dict):
                    raise ValueError("Method overrides must be a mapping.")
            else:
                raise ValueError(
                    "Each method entry must be a name or a mapping with a name."
                )
            method_config = _deep_merge(dataset_config, method_overrides)
            method_config["aggregator"] = method

            for entry in defenses:
                if not isinstance(entry, dict) or "name" not in entry:
                    raise ValueError("Each defense entry must be a mapping with a name.")
                defense = str(entry["name"]).lower()
                fixed = entry.get("fixed", {})
                if not isinstance(fixed, dict):
                    raise ValueError(
                        f"Defense {defense} fixed parameters must be a mapping."
                    )
                for grid_parameters in _grid_rows(entry.get("grid")):
                    defense_parameters = {**fixed, **grid_parameters}
                    for seed, target_client_id in itertools.product(seeds, targets):
                        config = copy.deepcopy(method_config)
                        config["seed"] = seed
                        config.setdefault("audit", {})[
                            "target_client_id"
                        ] = target_client_id
                        config["defense"] = _deep_merge(
                            config.get("defense", {}),
                            {"name": defense, **defense_parameters},
                        )
                        run_id = _stable_run_id(
                            defense,
                            seed,
                            target_client_id,
                            config,
                            dataset_prefix=dataset if multi_dataset_spec else None,
                            method_prefix=method if multi_method_spec else None,
                        )
                        if run_id in seen_ids:
                            raise ValueError(f"Duplicate sweep job generated: {run_id}")
                        seen_ids.add(run_id)
                        run_root = results_root / "runs" / run_id
                        config["results_dir"] = str(run_root)
                        jobs.append(
                            SweepJob(
                                run_id=run_id,
                                config=config,
                                dataset=dataset,
                                method=method,
                                seed=seed,
                                target_client_id=target_client_id,
                                defense=defense,
                                defense_parameters=defense_parameters,
                                run_root=run_root,
                                config_path=results_root
                                / "configs"
                                / f"{run_id}.yaml",
                            )
                        )
    return jobs, results_root


def filter_jobs_by_dataset(
    jobs: list[SweepJob], dataset_names: str | None
) -> list[SweepJob]:
    """Select complete dataset groups after expanding a sweep specification."""
    if dataset_names is None:
        return jobs
    requested = {
        value.strip().lower() for value in dataset_names.split(",") if value.strip()
    }
    if not requested:
        raise ValueError("--datasets must contain at least one dataset name.")
    available = {job.dataset for job in jobs}
    unknown = requested - available
    if unknown:
        raise ValueError(
            "Unknown dataset(s): "
            f"{', '.join(sorted(unknown))}. Available: {', '.join(sorted(available))}."
        )
    return [job for job in jobs if job.dataset in requested]


def filter_jobs_by_method(
    jobs: list[SweepJob], method_names: str | None
) -> list[SweepJob]:
    """Select complete method groups after expanding a sweep specification."""
    if method_names is None:
        return jobs
    requested = {
        value.strip().lower() for value in method_names.split(",") if value.strip()
    }
    if not requested:
        raise ValueError("--methods must contain at least one method name.")
    available = {job.method for job in jobs}
    unknown = requested - available
    if unknown:
        raise ValueError(
            "Unknown method(s): "
            f"{', '.join(sorted(unknown))}. Available: {', '.join(sorted(available))}."
        )
    return [job for job in jobs if job.method in requested]


def _completed_result(job: SweepJob) -> Path | None:
    if not job.run_root.exists():
        return None
    summaries = sorted(
        job.run_root.glob("*/privacy_audit/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary in summaries:
        result_dir = summary.parents[1]
        if not (result_dir / "training_metrics.csv").is_file():
            continue
        if bool(job.config.get("audit", {}).get("training_health_check", False)):
            health_path = result_dir / "training_health.json"
            if not health_path.is_file():
                continue
            try:
                with health_path.open("r", encoding="utf-8") as file:
                    if not bool(json.load(file).get("passed", False)):
                        continue
            except (OSError, json.JSONDecodeError):
                continue
        try:
            with summary.open("r", encoding="utf-8") as file:
                audit = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if audit.get("errors"):
            continue
        completed_attacks = {
            item.get("attack") for item in audit.get("attacks", [])
        }
        expected_attacks = set(job.config.get("audit", {}).get("attacks", []))
        if completed_attacks != expected_attacks:
            continue
        return result_dir
    return None


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"runs": {}}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.setdefault("runs", {})
    return payload


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True)
    temporary.replace(path)


def _write_job_config(job: SweepJob) -> None:
    job.config_path.parent.mkdir(parents=True, exist_ok=True)
    with job.config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(job.config, file, sort_keys=False, allow_unicode=True)


def _parse_gpus(raw: str) -> list[int]:
    gpus = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("--gpus must contain unique non-negative GPU indices.")
    return gpus


def _query_gpu_status(candidate_gpus: list[int]) -> list[GPUStatus]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Automatic GPU selection requires a working nvidia-smi command."
        ) from error
    candidates = set(candidate_gpus)
    statuses = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        index, free_memory, utilization = map(int, fields)
        if index in candidates:
            statuses.append(
                GPUStatus(
                    index=index,
                    free_memory_mb=free_memory,
                    utilization_percent=utilization,
                )
            )
    missing = candidates - {status.index for status in statuses}
    if missing:
        raise ValueError(f"GPU indices not reported by nvidia-smi: {sorted(missing)}")
    return statuses


def _wait_for_best_gpu(
    candidate_gpus: list[int],
    minimum_free_memory_mb: int,
    poll_seconds: int = 30,
) -> GPUStatus:
    while True:
        statuses = _query_gpu_status(candidate_gpus)
        eligible = [
            status
            for status in statuses
            if status.free_memory_mb >= minimum_free_memory_mb
        ]
        if eligible:
            return max(
                eligible,
                key=lambda status: (
                    status.free_memory_mb,
                    -status.utilization_percent,
                ),
            )
        description = ", ".join(
            f"gpu:{status.index} free={status.free_memory_mb}MiB "
            f"util={status.utilization_percent}%"
            for status in statuses
        )
        print(
            f"No GPU has {minimum_free_memory_mb}MiB free; {description}. "
            f"Retrying in {poll_seconds}s.",
            flush=True,
        )
        time.sleep(poll_seconds)


def _best_available_gpu(
    candidate_gpus: list[int],
    busy_gpus: set[int],
    minimum_free_memory_mb: int,
) -> GPUStatus | None:
    """Return the best eligible GPU that is not already running a sweep job."""
    available = [gpu for gpu in candidate_gpus if gpu not in busy_gpus]
    if not available:
        return None
    eligible = [
        status
        for status in _query_gpu_status(available)
        if status.free_memory_mb >= minimum_free_memory_mb
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda status: (
            status.free_memory_mb,
            -status.utilization_percent,
        ),
    )


def _launch(job: SweepJob, gpu: int, logs_root: Path) -> ActiveRun:
    _write_job_config(job)
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"{job.run_id}.log"
    log_file = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "main.py"),
        "--config",
        str(job.config_path),
        "--gpu",
        str(gpu),
    ]
    log_file.write("COMMAND " + " ".join(command) + "\n")
    log_file.flush()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return ActiveRun(job=job, gpu=gpu, process=process, log_file=log_file)


def summarize(jobs: list[SweepJob], results_root: Path) -> tuple[int, int]:
    detailed_rows: list[dict[str, Any]] = []
    complete_runs = 0
    for job in jobs:
        result_dir = _completed_result(job)
        if result_dir is None:
            continue
        complete_runs += 1
        with (result_dir / "training_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as file:
            metric_rows = list(csv.DictReader(file))
        if not metric_rows:
            continue
        final_accuracy = float(metric_rows[-1]["accuracy"])
        epsilon_upper_bound = None
        dp_delta = None
        defense_summary_path = result_dir / "defense_summary.json"
        if defense_summary_path.is_file():
            with defense_summary_path.open("r", encoding="utf-8") as file:
                defense_summary = json.load(file)
            accounting = defense_summary.get("privacy_accounting", {})
            if accounting:
                epsilon_upper_bound = float(accounting["epsilon_upper_bound"])
                dp_delta = float(accounting["delta"])
        with (result_dir / "privacy_audit" / "summary.json").open(
            "r", encoding="utf-8"
        ) as file:
            audit = json.load(file)
        for attack in audit.get("attacks", []):
            reportable = attack.get("reportable_metrics", attack)

            def optional_metric(key: str) -> float | None:
                value = reportable.get(key)
                return None if value is None else float(value)

            tpr10 = optional_metric("tpr_at_fpr_0.1")
            tpr1 = optional_metric("tpr_at_fpr_0.01")
            tpr0_1 = optional_metric("tpr_at_fpr_0.001")
            detailed_rows.append(
                {
                    "run_id": job.run_id,
                    "dataset": job.dataset,
                    "method": job.method,
                    "seed": job.seed,
                    "target_client_id": job.target_client_id,
                    "defense": job.defense,
                    "defense_parameters": json.dumps(
                        job.defense_parameters, sort_keys=True
                    ),
                    "attack": attack["attack"],
                    "final_accuracy": final_accuracy,
                    "tpr_at_fpr_0.1": tpr10,
                    "tpr_at_fpr_0.01": tpr1,
                    "tpr_at_fpr_0.001": tpr0_1,
                    "tpr_pct_at_fpr_10pct": (
                        None if tpr10 is None else 100.0 * tpr10
                    ),
                    "tpr_pct_at_fpr_1pct": (
                        None if tpr1 is None else 100.0 * tpr1
                    ),
                    "tpr_pct_at_fpr_0_1pct": (
                        None if tpr0_1 is None else 100.0 * tpr0_1
                    ),
                    "auc": float(attack["auc"]),
                    "member_count": attack.get("member_count"),
                    "nonmember_count": attack.get("nonmember_count"),
                    "fpr_resolution": attack.get("fpr_resolution"),
                    "epsilon_upper_bound": epsilon_upper_bound,
                    "dp_delta": dp_delta,
                    "num_samples": int(attack["num_samples"]),
                    "result_dir": str(result_dir),
                }
            )

    results_root.mkdir(parents=True, exist_ok=True)
    detailed_path = results_root / "summary_by_run.csv"
    detailed_fields = [
        "run_id",
        "dataset",
        "method",
        "seed",
        "target_client_id",
        "defense",
        "defense_parameters",
        "attack",
        "final_accuracy",
        "tpr_at_fpr_0.1",
        "tpr_at_fpr_0.01",
        "tpr_at_fpr_0.001",
        "tpr_pct_at_fpr_10pct",
        "tpr_pct_at_fpr_1pct",
        "tpr_pct_at_fpr_0_1pct",
        "auc",
        "member_count",
        "nonmember_count",
        "fpr_resolution",
        "epsilon_upper_bound",
        "dp_delta",
        "num_samples",
        "result_dir",
    ]
    with detailed_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=detailed_fields)
        writer.writeheader()
        writer.writerows(detailed_rows)

    grouped: dict[tuple[str, str, int, str, str, str], list[dict[str, Any]]] = {}
    for row in detailed_rows:
        key = (
            str(row["dataset"]),
            str(row["method"]),
            int(row["target_client_id"]),
            row["defense"],
            row["defense_parameters"],
            row["attack"],
        )
        grouped.setdefault(key, []).append(row)
    aggregate_rows = []
    for (
        dataset,
        method,
        target_client_id,
        defense,
        parameters,
        attack,
    ), rows in sorted(grouped.items()):
        def values(key: str) -> list[float]:
            return [
                float(row[key])
                for row in rows
                if row.get(key) is not None and row.get(key) != ""
            ]

        def mean(key: str) -> float | None:
            selected = values(key)
            return statistics.fmean(selected) if selected else None

        def sample_std(key: str) -> float | None:
            selected = values(key)
            return (
                statistics.stdev(selected)
                if len(selected) > 1
                else 0.0 if selected else None
            )

        epsilon_values = [
            float(row["epsilon_upper_bound"])
            for row in rows
            if row["epsilon_upper_bound"] is not None
        ]
        member_values = [
            int(row["member_count"])
            for row in rows
            if row["member_count"] is not None
        ]
        nonmember_values = [
            int(row["nonmember_count"])
            for row in rows
            if row["nonmember_count"] is not None
        ]
        resolution_values = [
            float(row["fpr_resolution"])
            for row in rows
            if row["fpr_resolution"] is not None
        ]

        aggregate_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "target_client_id": target_client_id,
                "defense": defense,
                "defense_parameters": parameters,
                "attack": attack,
                "runs": len(rows),
                "accuracy_mean": mean("final_accuracy"),
                "accuracy_std": sample_std("final_accuracy"),
                "tpr_at_fpr_0.1_mean": mean("tpr_at_fpr_0.1"),
                "tpr_at_fpr_0.1_std": sample_std("tpr_at_fpr_0.1"),
                "tpr_at_fpr_0.01_mean": mean("tpr_at_fpr_0.01"),
                "tpr_at_fpr_0.01_std": sample_std("tpr_at_fpr_0.01"),
                "tpr_at_fpr_0.001_mean": mean("tpr_at_fpr_0.001"),
                "tpr_at_fpr_0.001_std": sample_std("tpr_at_fpr_0.001"),
                "tpr_pct_at_fpr_10pct_mean": mean("tpr_pct_at_fpr_10pct"),
                "tpr_pct_at_fpr_10pct_std": sample_std("tpr_pct_at_fpr_10pct"),
                "tpr_pct_at_fpr_1pct_mean": mean("tpr_pct_at_fpr_1pct"),
                "tpr_pct_at_fpr_1pct_std": sample_std("tpr_pct_at_fpr_1pct"),
                "tpr_pct_at_fpr_0_1pct_mean": mean(
                    "tpr_pct_at_fpr_0_1pct"
                ),
                "tpr_pct_at_fpr_0_1pct_std": sample_std(
                    "tpr_pct_at_fpr_0_1pct"
                ),
                "auc_mean": mean("auc"),
                "auc_std": sample_std("auc"),
                "member_count_min": min(member_values) if member_values else None,
                "nonmember_count_min": (
                    min(nonmember_values) if nonmember_values else None
                ),
                "fpr_resolution_max": (
                    max(resolution_values) if resolution_values else None
                ),
                "epsilon_upper_bound_mean": (
                    statistics.fmean(epsilon_values) if epsilon_values else None
                ),
                "epsilon_upper_bound_std": (
                    statistics.stdev(epsilon_values)
                    if len(epsilon_values) > 1
                    else 0.0 if epsilon_values else None
                ),
                "dp_delta": next(
                    (row["dp_delta"] for row in rows if row["dp_delta"] is not None),
                    None,
                ),
            }
        )
    aggregate_path = results_root / "summary_aggregate.csv"
    aggregate_fields = [
        "dataset",
        "method",
        "target_client_id",
        "defense",
        "defense_parameters",
        "attack",
        "runs",
        "accuracy_mean",
        "accuracy_std",
        "tpr_at_fpr_0.1_mean",
        "tpr_at_fpr_0.1_std",
        "tpr_at_fpr_0.01_mean",
        "tpr_at_fpr_0.01_std",
        "tpr_at_fpr_0.001_mean",
        "tpr_at_fpr_0.001_std",
        "tpr_pct_at_fpr_10pct_mean",
        "tpr_pct_at_fpr_10pct_std",
        "tpr_pct_at_fpr_1pct_mean",
        "tpr_pct_at_fpr_1pct_std",
        "tpr_pct_at_fpr_0_1pct_mean",
        "tpr_pct_at_fpr_0_1pct_std",
        "auc_mean",
        "auc_std",
        "member_count_min",
        "nonmember_count_min",
        "fpr_resolution_max",
        "epsilon_upper_bound_mean",
        "epsilon_upper_bound_std",
        "dp_delta",
    ]
    with aggregate_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    tpr_matrix_path = results_root / "summary_tpr_matrix.csv"
    tpr_fields = [
        "dataset",
        "method",
        "target_client_id",
        "defense",
        "defense_parameters",
        "attack",
        "runs",
        "accuracy_mean",
        "accuracy_std",
        "tpr_at_fpr_0.1_mean",
        "tpr_at_fpr_0.1_std",
        "tpr_at_fpr_0.01_mean",
        "tpr_at_fpr_0.01_std",
        "tpr_at_fpr_0.001_mean",
        "tpr_at_fpr_0.001_std",
    ]
    with tpr_matrix_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tpr_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in tpr_fields} for row in aggregate_rows
        )

    privacy_metrics_path = results_root / "summary_privacy_metrics.csv"
    privacy_metric_fields = [
        "dataset",
        "method",
        "target_client_id",
        "attack",
        "runs",
        "accuracy_mean",
        "accuracy_std",
        "tpr_pct_at_fpr_0_1pct_mean",
        "tpr_pct_at_fpr_0_1pct_std",
        "tpr_pct_at_fpr_1pct_mean",
        "tpr_pct_at_fpr_1pct_std",
        "tpr_pct_at_fpr_10pct_mean",
        "tpr_pct_at_fpr_10pct_std",
        "auc_mean",
        "auc_std",
        "member_count_min",
        "nonmember_count_min",
        "fpr_resolution_max",
    ]
    with privacy_metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=privacy_metric_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in privacy_metric_fields}
            for row in aggregate_rows
        )

    pareto_rows = []
    by_attack: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in aggregate_rows:
        by_attack.setdefault(
            (
                str(row["dataset"]),
                str(row["method"]),
                int(row["target_client_id"]),
                str(row["attack"]),
            ),
            [],
        ).append(row)
    for rows in by_attack.values():
        comparable = [
            row for row in rows if row["tpr_at_fpr_0.01_mean"] is not None
        ]
        for candidate in comparable:
            dominated = any(
                other is not candidate
                and float(other["accuracy_mean"])
                >= float(candidate["accuracy_mean"])
                and float(other["tpr_at_fpr_0.01_mean"])
                <= float(candidate["tpr_at_fpr_0.01_mean"])
                and (
                    float(other["accuracy_mean"])
                    > float(candidate["accuracy_mean"])
                    or float(other["tpr_at_fpr_0.01_mean"])
                    < float(candidate["tpr_at_fpr_0.01_mean"])
                )
                for other in comparable
            )
            if not dominated:
                pareto_rows.append(candidate)
    pareto_path = results_root / "privacy_utility_pareto.csv"
    with pareto_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(pareto_rows)
    return complete_runs, len(detailed_rows)


def run_sweep(
    jobs: list[SweepJob],
    results_root: Path,
    gpus: list[int],
    force: bool,
    minimum_free_memory_mb: int,
    max_parallel_jobs: int = 1,
) -> int:
    if max_parallel_jobs <= 0:
        raise ValueError("jobs must be positive.")
    effective_parallel_jobs = min(max_parallel_jobs, len(gpus))
    state_path = results_root / "sweep_state.json"
    state = _load_state(state_path)
    pending = []
    for job in jobs:
        completed = _completed_result(job)
        if completed is not None and not force:
            state["runs"][job.run_id] = {
                "status": "succeeded",
                "result_dir": str(completed),
            }
        else:
            pending.append(job)
    _save_state(state_path, state)
    print(
        f"Sweep jobs={len(jobs)}, pending={len(pending)}, "
        f"already_complete={len(jobs) - len(pending)}, candidate_gpus={gpus}, "
        f"requested_parallel_jobs={max_parallel_jobs}, "
        f"effective_parallel_jobs={effective_parallel_jobs}, "
        f"minimum_free_memory_mb={minimum_free_memory_mb}",
        flush=True,
    )

    failures = 0
    completed_count = len(jobs) - len(pending)
    active: dict[int, ActiveRun] = {}
    try:
        while pending or active:
            while pending and len(active) < effective_parallel_jobs:
                status = _best_available_gpu(
                    gpus, set(active), minimum_free_memory_mb
                )
                if status is None:
                    if active:
                        break
                    available = [gpu for gpu in gpus if gpu not in active]
                    status = _wait_for_best_gpu(
                        available, minimum_free_memory_mb
                    )
                job = pending.pop(0)
                run = _launch(job, status.index, results_root / "launcher_logs")
                active[status.index] = run
                state["runs"][job.run_id] = {
                    "status": "running",
                    "gpu": status.index,
                    "gpu_free_memory_mb_at_start": status.free_memory_mb,
                    "gpu_utilization_at_start": status.utilization_percent,
                    "started_at": time.time(),
                }
                _save_state(state_path, state)
                print(
                    f"[{completed_count}/{len(jobs)}] started {job.run_id} on "
                    f"gpu:{status.index} (free={status.free_memory_mb}MiB, "
                    f"util={status.utilization_percent}%, "
                    f"active={len(active)}/{effective_parallel_jobs})",
                    flush=True,
                )

            finished_gpus = [
                gpu for gpu, run in active.items() if run.process.poll() is not None
            ]
            if not finished_gpus:
                time.sleep(2)
                continue
            for gpu in finished_gpus:
                run = active.pop(gpu)
                return_code = int(run.process.returncode)
                run.log_file.close()
                result_dir = _completed_result(run.job)
                succeeded = return_code == 0 and result_dir is not None
                state["runs"][run.job.run_id] = {
                    "status": "succeeded" if succeeded else "failed",
                    "gpu": gpu,
                    "return_code": return_code,
                    "finished_at": time.time(),
                    "result_dir": None if result_dir is None else str(result_dir),
                }
                _save_state(state_path, state)
                completed_count += 1
                if not succeeded:
                    failures += 1
                print(
                    f"[{completed_count}/{len(jobs)}] "
                    f"{'finished' if succeeded else 'FAILED'} {run.job.run_id} "
                    f"on gpu:{gpu} (active={len(active)}/{effective_parallel_jobs})",
                    flush=True,
                )
    except KeyboardInterrupt:
        print(
            f"Interrupt received; terminating {len(active)} active experiment(s)...",
            flush=True,
        )
        for gpu, run in active.items():
            run.process.terminate()
        for gpu, run in active.items():
            try:
                run.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                run.process.kill()
            run.log_file.close()
            state["runs"][run.job.run_id] = {
                "status": "interrupted",
                "gpu": gpu,
                "finished_at": time.time(),
            }
        _save_state(state_path, state)
        return 130

    complete_runs, attack_rows = summarize(jobs, results_root)
    print(
        f"Summary complete: runs={complete_runs}/{len(jobs)}, "
        f"attack_rows={attack_rows}, failures={failures}",
        flush=True,
    )
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a multi-dataset FedMIA defense grid, run experiments "
            "on automatically selected single GPUs, resume completed jobs, "
            "and aggregate privacy metrics."
        )
    )
    parser.add_argument(
        "--spec",
        default="configs/fedmia_complex_sweep.yaml",
        help="Sweep YAML specification relative to the repository root.",
    )
    parser.add_argument(
        "--gpus",
        help=(
            "Comma-separated candidate GPU indices. Each job automatically "
            "uses an eligible candidate with the most free memory."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help=(
            "Maximum concurrent tasks. Defaults to 1; concurrency cannot exceed "
            "the number of candidate GPUs, and each GPU runs at most one task."
        ),
    )
    parser.add_argument(
        "--min-free-memory-mb",
        type=int,
        help="Wait until a candidate GPU has at least this much free memory.",
    )
    parser.add_argument("--data-root", help="Override the dataset root.")
    parser.add_argument("--cache-dir", help="Override the local CLIP cache.")
    parser.add_argument(
        "--datasets",
        help=(
            "Comma-separated dataset names to run after expansion, for example "
            "caltech101,oxfordpets."
        ),
    )
    parser.add_argument(
        "--methods",
        help=(
            "Comma-separated federated methods to run after expansion, for "
            "example promptfl,fedotp."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded jobs without starting training.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild CSV summaries without launching experiments.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun jobs even when a complete result already exists.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Limit expanded jobs for a smoke test; applied after expansion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = _resolve_path(args.spec)
    spec = _load_yaml(spec_path)
    jobs, results_root = build_jobs(
        spec,
        spec_path,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
    )
    jobs = filter_jobs_by_dataset(jobs, args.datasets)
    jobs = filter_jobs_by_method(jobs, args.methods)
    if args.max_runs is not None:
        if args.max_runs <= 0:
            raise ValueError("--max-runs must be positive.")
        jobs = jobs[: args.max_runs]
    if args.dry_run:
        print(f"Expanded {len(jobs)} jobs; results_root={results_root}")
        for job in jobs:
            print(
                job.run_id,
                f"dataset={job.dataset}",
                f"method={job.method}",
                f"seed={job.seed}",
                f"target={job.target_client_id}",
                f"defense={job.defense}",
                json.dumps(job.defense_parameters, sort_keys=True),
            )
        return 0
    if args.summarize_only:
        complete_runs, attack_rows = summarize(jobs, results_root)
        print(
            f"Summarized {complete_runs}/{len(jobs)} runs and {attack_rows} attack rows."
        )
        return 0
    gpu_text = args.gpus or ",".join(str(gpu) for gpu in spec.get("gpus", [0]))
    minimum_free_memory_mb = (
        int(args.min_free_memory_mb)
        if args.min_free_memory_mb is not None
        else int(spec.get("minimum_free_memory_mb", 0))
    )
    if minimum_free_memory_mb < 0:
        raise ValueError("minimum free GPU memory must be non-negative.")
    max_parallel_jobs = (
        int(args.jobs) if args.jobs is not None else int(spec.get("jobs", 1))
    )
    if max_parallel_jobs <= 0:
        raise ValueError("--jobs must be positive.")
    return run_sweep(
        jobs,
        results_root,
        _parse_gpus(gpu_text),
        args.force,
        minimum_free_memory_mb,
        max_parallel_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
