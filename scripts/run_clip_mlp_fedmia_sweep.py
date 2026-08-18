#!/usr/bin/env python3
"""Run multi-dataset frozen-CLIP FedMIA experiments and optional ProjRes."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_LOCK = threading.Lock()
_CHILD_TIMESTAMP = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3,6})?)\s+"
)


@dataclass(frozen=True)
class SweepJob:
    run_id: str
    config: dict[str, Any]
    dataset: str
    method: str
    seed: int
    target_client_id: int
    defense: str
    run_root: Path
    config_path: Path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def _audit_client_label(config: dict[str, Any]) -> str:
    """Return a stable human-readable label for the clients audited in one run."""
    audit = config.get("audit", {})
    configured = audit.get("audit_client_ids")
    if isinstance(configured, str) and configured.lower() == "all":
        return "all"
    if isinstance(configured, list) and len(configured) > 1:
        return "clients" + "-".join(str(int(value)) for value in configured)
    if isinstance(configured, list) and configured:
        return str(int(configured[0]))
    return str(int(audit.get("target_client_id", 0)))


def _timestamped_run_id(
    config: dict[str, Any],
    dataset: str,
    seed: int,
    target: int | str,
    started_at: datetime.datetime,
) -> str:
    canonical = copy.deepcopy(config)
    canonical.pop("gpu", None)
    canonical.pop("results_dir", None)
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    timestamp = started_at.strftime("%Y-%m-%d_%H-%M-%S-%f")
    model_type = str(config.get("model_type", "clip_mlp"))
    method = str(config.get("aggregator", "fedavg")).lower()
    return (
        f"{timestamp}_{model_type}_{dataset}_{method}_seed{seed}_target{target}_{digest}"
    )


def build_jobs(
    spec: dict[str, Any],
    spec_path: Path,
    data_root: str | None = None,
    cache_dir: str | None = None,
    num_global_iters: int | None = None,
    dirichlet_alpha: float | None = None,
    learning_rate: float | None = None,
    learning_rate_decay: float | None = None,
    learning_rate_decay_interval: int | None = None,
    partition_mode: str | None = None,
    started_at: datetime.datetime | None = None,
) -> tuple[list[SweepJob], Path]:
    del spec_path
    base_path = _resolve_path(
        spec.get("base_config", "configs/clip_mlp_low_fpr_attacks.yaml")
    )
    base = _deep_merge(_load_yaml(base_path), spec.get("common", {}))
    base["projres"] = _deep_merge(
        base.get("projres", {}), spec.get("projres", {})
    )
    results_root = _resolve_path(spec.get("results_root", "results"))
    sweep_name = str(spec.get("name", "clip_mlp_fedmia_attacks"))
    datasets = spec.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("sweep.datasets must be a non-empty list.")
    seeds = [int(value) for value in spec.get("seeds", [42])]
    targets = [int(value) for value in spec.get("target_client_ids", [0])]
    if not seeds or not targets:
        raise ValueError("sweep seeds and target_client_ids must be non-empty.")
    configured_audit_clients = base.get("audit", {}).get("audit_client_ids")
    pooled_audit = (
        isinstance(configured_audit_clients, str)
        and configured_audit_clients.lower() == "all"
    ) or (
        isinstance(configured_audit_clients, list)
        and len(configured_audit_clients) > 1
    )
    if pooled_audit and len(targets) != 1:
        raise ValueError(
            "A pooled audit uses one training job; target_client_ids must contain "
            "only one compatibility anchor."
        )
    defenses = spec.get("defenses", [{"name": "none"}])
    if len(defenses) != 1 or not isinstance(defenses[0], dict):
        raise ValueError("The frozen-CLIP attack sweep requires one defense entry.")
    defense_config = dict(defenses[0])
    defense_name = str(defense_config.get("name", "none")).lower()
    if defense_name not in {"none", "iclr"}:
        raise ValueError(
            "The frozen-CLIP attack sweep supports defense=none or iclr."
        )
    defense_config["name"] = defense_name

    jobs = []
    invocation_time = started_at or datetime.datetime.now()
    for entry, seed, target in itertools.product(datasets, seeds, targets):
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("Each dataset entry must be a mapping with a name.")
        config = _deep_merge(base, entry.get("overrides", {}))
        config["dataset_name"] = str(
            entry.get("overrides", {}).get("dataset_name", entry["name"])
        ).lower()
        if data_root is not None:
            config["data_root"] = str(_resolve_path(data_root))
        if cache_dir is not None:
            config["cache_dir"] = str(_resolve_path(cache_dir))
        if num_global_iters is not None:
            if num_global_iters <= 0:
                raise ValueError("--rounds must be positive.")
            config["num_global_iters"] = num_global_iters
        if dirichlet_alpha is not None:
            if dirichlet_alpha <= 0:
                raise ValueError("--dirichlet-alpha must be positive.")
            config["dirichlet_alpha"] = dirichlet_alpha
            config["partition_mode"] = "dirichlet"
        if partition_mode is not None:
            normalized_partition = str(partition_mode).lower()
            if normalized_partition not in {"iid", "dirichlet"}:
                raise ValueError("--partition-mode must be iid or dirichlet.")
            config["partition_mode"] = normalized_partition
        if learning_rate is not None:
            if learning_rate <= 0:
                raise ValueError("--learning-rate must be positive.")
            config["learning_rate"] = learning_rate
        if learning_rate_decay is not None:
            if not 0 < learning_rate_decay <= 1:
                raise ValueError("--learning-rate-decay must be in (0, 1].")
            config["learning_rate_decay"] = learning_rate_decay
        if learning_rate_decay_interval is not None:
            if learning_rate_decay_interval <= 0:
                raise ValueError(
                    "--learning-rate-decay-interval must be positive."
                )
            config["learning_rate_decay_interval"] = learning_rate_decay_interval
        config["seed"] = seed
        config["aggregator"] = (
            "fedsgd"
            if str(config.get("model_type", "clip_mlp")).lower()
            in {"visual_adapter", "clip_lora"}
            else "fedavg"
        )
        if str(config.get("model_type", "clip_mlp")).lower() in {
            "clip_mlp",
            "visual_adapter",
            "clip_lora",
        }:
            config["aggregation_weighting"] = "uniform"
        config["sweep_name"] = sweep_name
        config["defense"] = _deep_merge(
            config.get("defense", {}), defense_config
        )
        config.setdefault("audit", {})["target_client_id"] = target
        dataset = str(config["dataset_name"])
        run_id = _timestamped_run_id(
            config, dataset, seed, _audit_client_label(config), invocation_time
        )
        run_root = results_root / run_id
        config["results_dir"] = str(run_root)
        config["results_dir_is_run_dir"] = True
        jobs.append(
            SweepJob(
                run_id=run_id,
                config=config,
                dataset=dataset,
                method=str(config["aggregator"]),
                seed=seed,
                target_client_id=target,
                defense=defense_name,
                run_root=run_root,
                config_path=run_root / "run_config.yaml",
            )
        )
    return jobs, results_root


def discover_existing_jobs(
    results_root: Path, sweep_name: str | None = None
) -> list[SweepJob]:
    """Reconstruct prior timestamped jobs for summarize-only runs."""
    jobs = []
    config_paths = list(results_root.glob("*/run_config.yaml"))
    if sweep_name:
        legacy_root = results_root / sweep_name / "runs"
        config_paths.extend(legacy_root.glob("*/run_config.yaml"))
    for config_path in sorted(set(config_paths)):
        config = _load_yaml(config_path)
        configured_sweep = config.get("sweep_name")
        is_legacy_sweep_path = bool(
            sweep_name
            and config_path.parent.parent == results_root / sweep_name / "runs"
        )
        if sweep_name and configured_sweep != sweep_name and not is_legacy_sweep_path:
            continue
        run_root = config_path.parent
        audit = config.get("audit", {})
        jobs.append(
            SweepJob(
                run_id=run_root.name,
                config=config,
                dataset=str(config.get("dataset_name", "unknown")).lower(),
                method=str(config.get("aggregator", "fedavg")).lower(),
                seed=int(config.get("seed", 42)),
                target_client_id=int(audit.get("target_client_id", 0)),
                defense=str(config.get("defense", {}).get("name", "none")),
                run_root=run_root,
                config_path=config_path,
            )
        )
    return jobs


def filter_jobs_by_dataset(jobs: list[SweepJob], names: str | None) -> list[SweepJob]:
    if names is None:
        return jobs
    requested = {name.lower() for name in _parse_csv(names)}
    available = {job.dataset for job in jobs}
    unknown = requested - available
    if unknown:
        raise ValueError(
            f"Unknown datasets: {sorted(unknown)}; available: {sorted(available)}."
        )
    return [job for job in jobs if job.dataset in requested]


def _write_job_config(job: SweepJob) -> None:
    job.config_path.parent.mkdir(parents=True, exist_ok=True)
    with job.config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(job.config, file, sort_keys=False, allow_unicode=True)


def _completed_result(job: SweepJob) -> Path | None:
    summary = job.run_root / "privacy_audit" / "summary.json"
    metrics = job.run_root / "training_metrics.csv"
    if not summary.is_file() or not metrics.is_file():
        return None
    try:
        with summary.open("r", encoding="utf-8") as file:
            audit = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    completed = {row.get("attack") for row in audit.get("attacks", [])}
    expected = set(job.config.get("audit", {}).get("attacks", []))
    exact_batch_attacks = set(
        job.config.get("audit", {}).get(
            "exact_batch_membership_attacks", []
        )
    )
    unified_projres = "projres" in exact_batch_attacks
    projres_complete = not bool(
        job.config.get("projres", {}).get("enabled", False)
    ) or unified_projres or _projres_path(job).is_file()
    iclr_complete = True
    if str(job.defense).lower() == "iclr":
        validation = audit.get("iclr_validation") or {}
        privacy_audit = job.run_root / "privacy_audit"
        iclr_complete = (
            validation.get("status") == "ok"
            and (privacy_audit / "iclr_attack_samples.csv").is_file()
            and (privacy_audit / "iclr_attack_relationship.csv").is_file()
            and (privacy_audit / "iclr_attack_relationship.json").is_file()
        )
    return (
        job.run_root
        if not audit.get("errors")
        and completed == expected
        and projres_complete
        and iclr_complete
        else None
    )


def _display(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(_display(item) for item in value)
    return str(value)


def _flatten(value: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            rows.extend(_flatten(item, name))
        else:
            rows.append((name, item))
    return rows


def _job_hyperparameters_block(job: SweepJob) -> str:
    config = job.config
    model_type = str(config.get("model_type", "clip_mlp"))
    selected = {
        "model_type": model_type,
        "dataset": job.dataset,
        "seed": job.seed,
        "data": {
            "partition_mode": config.get("partition_mode"),
            "dirichlet_alpha": config.get("dirichlet_alpha"),
            "use_full_dataset": config.get("use_full_dataset"),
            "data_root": config.get("data_root"),
        },
        "federated": {
            "aggregator": config.get("aggregator"),
            "aggregation_weighting": config.get("aggregation_weighting"),
            "total_users": config.get("total_users"),
            "sample_users": config.get("sample_users"),
            "num_global_iters": config.get("num_global_iters"),
            "local_epochs": config.get("local_epochs"),
        },
        "optimization": {
            "learning_rate": config.get("learning_rate"),
            "learning_rate_decay": config.get("learning_rate_decay", 1.0),
            "learning_rate_decay_interval": config.get(
                "learning_rate_decay_interval", 1
            ),
            "batch_size": config.get("batch_size"),
            "eval_batch_size": config.get("eval_batch_size"),
            "eval_interval": config.get("eval_interval"),
        },
        model_type: config.get(model_type, {}),
        "privacy_audit": config.get("audit", {}),
        "projres": config.get("projres", {}),
        "privacy_defense": config.get("defense", {}),
    }
    rows = _flatten(selected)
    width = max(len(key) for key, _ in rows)
    divider = "=" * 88
    body = "\n".join(f"  {key:<{width}} : {_display(value)}" for key, value in rows)
    return f"{divider}\nHYPERPARAMETERS | {job.run_id}\n{divider}\n{body}\n{divider}"


def summarize(
    jobs: list[SweepJob], results_root: Path, summary_prefix: str
) -> tuple[int, int]:
    rows = []
    client_rows = []
    complete = 0
    for job in jobs:
        result_dir = _completed_result(job)
        if result_dir is None:
            continue
        complete += 1
        with (result_dir / "training_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as file:
            training_rows = list(csv.DictReader(file))
        with (result_dir / "privacy_audit" / "summary.json").open(
            "r", encoding="utf-8"
        ) as file:
            audit = json.load(file)
        accuracy = float(training_rows[-1]["accuracy"]) if training_rows else None
        for attack in audit.get("attacks", []):
            reportable = attack.get("reportable_metrics", attack)
            metadata = attack.get("metadata", {})
            macro = metadata.get("macro_metrics", {})
            macro_value = lambda key: (
                macro.get(key, {}).get("mean")
                if isinstance(macro.get(key), dict)
                else None
            )
            rows.append(
                {
                    "run_id": job.run_id,
                    "dataset": job.dataset,
                    "seed": job.seed,
                    "target_client_id": _audit_client_label(job.config),
                    "attack": attack["attack"],
                    "final_accuracy": accuracy,
                    "auc": attack.get("auc"),
                    "tpr_at_fpr_0.1": reportable.get("tpr_at_fpr_0.1"),
                    "tpr_at_fpr_0.01": reportable.get("tpr_at_fpr_0.01"),
                    "tpr_at_fpr_0.001": reportable.get("tpr_at_fpr_0.001"),
                    "member_count": attack.get("member_count"),
                    "nonmember_count": attack.get("nonmember_count"),
                    "fpr_resolution": attack.get("fpr_resolution"),
                    "client_macro_auc": macro_value("auc"),
                    "client_macro_tpr_at_fpr_0.1": macro_value(
                        "tpr_at_fpr_0.1"
                    ),
                    "client_macro_tpr_at_fpr_0.01": macro_value(
                        "tpr_at_fpr_0.01"
                    ),
                    "client_macro_tpr_at_fpr_0.001": macro_value(
                        "tpr_at_fpr_0.001"
                    ),
                    "result_dir": str(result_dir),
                }
            )
            for client_id, client in metadata.get(
                "per_client_metrics", {}
            ).items():
                client_reportable = client.get("reportable_metrics", client)
                nonmember_count = client.get("nonmember_count")
                client_rows.append(
                    {
                        "run_id": job.run_id,
                        "dataset": job.dataset,
                        "seed": job.seed,
                        "target_client_id": client_id,
                        "attack": attack["attack"],
                        "final_accuracy": accuracy,
                        "auc": client.get("auc"),
                        "tpr_at_fpr_0.1": client_reportable.get(
                            "tpr_at_fpr_0.1"
                        ),
                        "tpr_at_fpr_0.01": client_reportable.get(
                            "tpr_at_fpr_0.01"
                        ),
                        "tpr_at_fpr_0.001": client_reportable.get(
                            "tpr_at_fpr_0.001"
                        ),
                        "member_count": client.get("member_count"),
                        "nonmember_count": nonmember_count,
                        "fpr_resolution": (
                            1.0 / int(nonmember_count)
                            if nonmember_count
                            else None
                        ),
                        "result_dir": str(result_dir),
                    }
                )
    results_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id", "dataset", "seed", "target_client_id", "attack",
        "final_accuracy", "auc", "tpr_at_fpr_0.1", "tpr_at_fpr_0.01",
        "tpr_at_fpr_0.001", "member_count", "nonmember_count",
        "fpr_resolution", "client_macro_auc",
        "client_macro_tpr_at_fpr_0.1", "client_macro_tpr_at_fpr_0.01",
        "client_macro_tpr_at_fpr_0.001", "result_dir",
    ]
    with (results_root / f"{summary_prefix}_summary_by_run.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    client_fields = [
        "run_id", "dataset", "seed", "target_client_id", "attack",
        "final_accuracy", "auc", "tpr_at_fpr_0.1", "tpr_at_fpr_0.01",
        "tpr_at_fpr_0.001", "member_count", "nonmember_count",
        "fpr_resolution", "result_dir",
    ]
    with (results_root / f"{summary_prefix}_summary_by_client.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=client_fields)
        writer.writeheader()
        writer.writerows(client_rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["attack"]), []).append(row)
    aggregate_rows = []
    for (dataset, attack), group in sorted(grouped.items()):
        result = {"dataset": dataset, "attack": attack, "runs": len(group)}
        for metric in (
            "final_accuracy",
            "auc",
            "tpr_at_fpr_0.001",
            "client_macro_auc",
            "client_macro_tpr_at_fpr_0.001",
        ):
            values = [float(row[metric]) for row in group if row[metric] is not None]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else None
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        aggregate_rows.append(result)
    aggregate_fields = [
        "dataset", "attack", "runs", "final_accuracy_mean", "final_accuracy_std",
        "auc_mean", "auc_std", "tpr_at_fpr_0.001_mean", "tpr_at_fpr_0.001_std",
        "client_macro_auc_mean", "client_macro_auc_std",
        "client_macro_tpr_at_fpr_0.001_mean",
        "client_macro_tpr_at_fpr_0.001_std",
    ]
    with (results_root / f"{summary_prefix}_summary_aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return complete, len(rows)


DEFAULT_ATTACKS = (
    "blackbox_loss,loss_series,grad_cosine,avg_cosine,"
    "fedmia_loss,fedmia_cosine"
)


def _parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("A comma-separated option must contain at least one value.")
    return items


def _projres_path(job) -> Path:
    return job.run_root / "privacy_audit" / "projres_strict.json"


def _commands(job, gpu: int) -> list[list[str]]:
    main_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "main.py"),
        "--config",
        str(job.config_path),
        "--gpu",
        str(gpu),
    ]
    return [main_command]


def _command_text(command: list[str]) -> str:
    import shlex

    return shlex.join(command)


def _log_context(job: SweepJob, phase: str, gpu: int) -> str:
    model_type = str(job.config.get("model_type", "clip_mlp"))
    return (
        f"model={model_type} | dataset={job.dataset} | method={job.method} | "
        f"run={job.run_id} | phase={phase} | gpu={gpu}"
    )


def _format_scoped_log_line(
    line: str,
    context: str,
    *,
    now: datetime.datetime | None = None,
) -> str:
    """Attach experiment identity to every child-process output line."""
    text = line.rstrip("\r\n")
    match = _CHILD_TIMESTAMP.match(text)
    if match is not None:
        timestamp = match.group("timestamp")
        text = text[match.end() :]
    else:
        timestamp = (now or datetime.datetime.now()).strftime(
            "%Y-%m-%d %H:%M:%S,%f"
        )[:-3]
    return f"{timestamp} | {context} | {text}\n"


def _projres_parameters_block(job, projres: dict[str, Any]) -> str:
    divider = "=" * 88
    model_type = str(job.config.get("model_type", "clip_mlp"))
    attacked_layer = {
        "clip_mlp": "classifier.0.weight",
        "visual_adapter": "adapter.net.0.weight",
        "clip_lora": (
            "clip_model.vision_model.encoder.layers.0.self_attn."
            "q_proj.lora_A"
        ),
    }.get(model_type, "first_trainable_linear_weight")
    return "\n".join(
        (
            divider,
            f"PROJRES ON OBSERVED CLIENT UPDATE | {job.run_id}",
            divider,
            f"  evaluation_round   : {projres.get('evaluation_round', 'last')}",
            f"  decision_mode      : {projres.get('decision_mode', 'ranking')}",
            f"  max_candidates     : {projres.get('max_candidates', 32)}",
            f"  min_nonmembers     : {projres.get('min_nonmembers', 1000)}",
            f"  max_nonmembers     : {projres.get('max_nonmembers', 20000)}",
            f"  attacked_layer     : {attacked_layer}",
            "  update_source      : observed client upload from that real round",
            divider,
        )
    )


def _run_logged(
    command: list[str],
    log_path: Path,
    *,
    stream_output: bool = False,
    context: str = "experiment=unknown",
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    # The sweep runner already tees/captures the child stream into run.log.
    # Prevent main.py from opening the same log file with a second handler.
    environment["FEDMIA_LAUNCHER_LOG_CAPTURE"] = "1"
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            _format_scoped_log_line(
                "COMMAND " + _command_text(command), context
            )
        )
        log_file.flush()
        if not stream_output:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            return int(completed.returncode)

        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to capture child-process output.")
        for line in process.stdout:
            formatted = _format_scoped_log_line(line, context)
            log_file.write(formatted)
            log_file.flush()
            with _OUTPUT_LOCK:
                sys.stdout.write(formatted)
                sys.stdout.flush()
        return int(process.wait())


def _announce_job_start(
    job: SweepJob,
    gpu: int,
    projres: dict[str, Any],
    skip_projres: bool,
) -> None:
    blocks = [
        (
            f"STARTING JOB | {_log_context(job, 'all', gpu)} | "
            f"seed={job.seed} | audit_clients={_audit_client_label(job.config)}"
        ),
        _job_hyperparameters_block(job),
    ]
    unified_projres = "projres" in set(
        job.config.get("audit", {}).get(
            "exact_batch_membership_attacks", []
        )
    )
    if (
        not unified_projres
        and not skip_projres
        and bool(projres.get("enabled", True))
    ):
        blocks.append(_projres_parameters_block(job, projres))
    # One print under a lock keeps concurrent jobs from interleaving parameter
    # blocks while still announcing each job only when its worker really starts.
    with _OUTPUT_LOCK:
        print("\n".join(blocks), flush=True)


def _announce_phase(
    job: SweepJob, phase: str, phase_key: str, gpu: int, log_path: Path
) -> None:
    with _OUTPUT_LOCK:
        print(
            f"RUNNING {phase} | {_log_context(job, phase_key, gpu)} | log={log_path}",
            flush=True,
        )


def _run_job(
    job: SweepJob,
    gpu: int,
    projres: dict[str, Any],
    skip_projres: bool,
    force: bool,
    stream_output: bool = False,
):
    _announce_job_start(job, gpu, projres, skip_projres)
    _write_job_config(job)
    main_command = _commands(job, gpu)[0]
    main_complete = _completed_result(job) is not None
    if force or not main_complete:
        if force:
            stale = job.run_root / "privacy_audit" / "summary.json"
            if stale.is_file():
                stale.unlink()
        main_log = job.run_root / "run.log"
        main_phase = "training+membership_audit"
        protocol = str(job.config.get("aggregator", "fedavg")).upper()
        _announce_phase(
            job,
            f"{protocol} + GENERIC AUDIT",
            main_phase,
            gpu,
            main_log,
        )
        return_code = _run_logged(
            main_command,
            main_log,
            stream_output=stream_output,
            context=_log_context(job, main_phase, gpu),
        )
        if return_code != 0 or _completed_result(job) is None:
            return job.run_id, False, "main.py failed"
    return job.run_id, True, "complete"


def _projres_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if "result" in payload:
        return payload["result"]["attack"]["metrics"]
    return payload["pooled_metrics"]


def summarize_projres(jobs, results_root: Path, summary_prefix: str) -> int:
    rows = []
    for job in jobs:
        path = _projres_path(job)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        metrics = _projres_metrics(payload)
        reportable = metrics.get("reportable_metrics", metrics)
        rows.append(
            {
                "run_id": job.run_id,
                "dataset": job.dataset,
                "seed": job.seed,
                "target_client_id": _audit_client_label(job.config),
                "attack": "projres",
                "auc": metrics.get("auc"),
                "tpr_at_fpr_0.1": reportable.get("tpr_at_fpr_0.1"),
                "tpr_at_fpr_0.01": reportable.get("tpr_at_fpr_0.01"),
                "tpr_at_fpr_0.001": reportable.get("tpr_at_fpr_0.001"),
                "fpr_resolution": metrics.get("fpr_resolution"),
                "result_path": str(path),
            }
        )
    results_root.mkdir(parents=True, exist_ok=True)
    path = results_root / f"{summary_prefix}_summary_projres.csv"
    fields = [
        "run_id",
        "dataset",
        "seed",
        "target_client_id",
        "attack",
        "auc",
        "tpr_at_fpr_0.1",
        "tpr_at_fpr_0.01",
        "tpr_at_fpr_0.001",
        "fpr_resolution",
        "result_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", default="configs/clip_mlp_fedmia_attacks_sweep.yaml"
    )
    parser.add_argument("--datasets", help="Comma-separated dataset names.")
    parser.add_argument("--attacks", default=None, help="Generic attack CSV.")
    parser.add_argument(
        "--defense",
        choices=["none", "iclr"],
        help="Run the plain baseline or ICLR specificity validation.",
    )
    parser.add_argument(
        "--target-client",
        help="Audit one client ID, or use 'all' for every client in one training run.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", help="Comma-separated GPU indices.")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--rounds", "--num-global-iters", type=int)
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument(
        "--partition-mode",
        choices=["iid", "dirichlet"],
        help="Explicit client partition mode; defaults to the sweep specification.",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--learning-rate-decay", type=float)
    parser.add_argument("--learning-rate-decay-interval", type=int)
    parser.add_argument(
        "--projres-round",
        help=(
            "One-based communication round for standalone ProjRes, or "
            "'last'; unavailable to unified Visual Adapter ProjRes."
        ),
    )
    parser.add_argument("--projres-max-candidates", type=int)
    parser.add_argument("--projres-min-out", type=int)
    parser.add_argument("--projres-max-out", type=int)
    parser.add_argument("--skip-projres", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--dry-run", "--print-command", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = _resolve_path(args.spec)
    spec = copy.deepcopy(_load_yaml(spec_path))
    sweep_name = str(spec.get("name", "clip_mlp_fedmia_attacks"))
    if args.attacks is not None:
        audit_override = spec.setdefault("common", {}).setdefault("audit", {})
        audit_override["attacks"] = _parse_csv(args.attacks)
        if "exact_batch_membership_attacks" in audit_override:
            audit_override["exact_batch_membership_attacks"] = [
                attack
                for attack in audit_override[
                    "exact_batch_membership_attacks"
                ]
                if attack in audit_override["attacks"] or attack == "projres"
            ]
    if args.defense is not None:
        spec["defenses"] = [{"name": args.defense}]
    if args.target_client is not None:
        target_text = str(args.target_client).strip().lower()
        audit_override = spec.setdefault("common", {}).setdefault("audit", {})
        if target_text == "all":
            spec["target_client_ids"] = [0]
            audit_override["target_client_id"] = 0
            audit_override["audit_client_ids"] = "all"
        else:
            try:
                target_client_id = int(target_text)
            except ValueError as error:
                raise ValueError(
                    "--target-client must be a non-negative integer or 'all'."
                ) from error
            if target_client_id < 0:
                raise ValueError(
                    "--target-client must be a non-negative integer or 'all'."
                )
            spec["target_client_ids"] = [target_client_id]
            audit_override["target_client_id"] = target_client_id
            audit_override["audit_client_ids"] = [target_client_id]
    if args.seed is not None:
        spec["seeds"] = [args.seed]
    projres = dict(spec.get("projres", {}))
    projres_enabled = bool(projres.get("enabled", True)) and not args.skip_projres
    overrides = {
        "evaluation_round": args.projres_round,
        "max_candidates": args.projres_max_candidates,
        "min_nonmembers": args.projres_min_out,
        "max_nonmembers": args.projres_max_out,
    }
    projres.update({key: value for key, value in overrides.items() if value is not None})
    if projres.get("threshold") is not None:
        raise ValueError("ProjRes is ranking-only; threshold must be null.")
    if str(projres.get("decision_mode", "ranking")).lower() != "ranking":
        raise ValueError("ProjRes decision_mode must be ranking.")
    projres["enabled"] = projres_enabled
    spec["projres"] = projres
    audit_override = spec.setdefault("common", {}).setdefault("audit", {})
    configured_exact_batch_attacks = list(
        audit_override.get("exact_batch_membership_attacks", [])
    )
    unified_projres = "projres" in configured_exact_batch_attacks
    if args.skip_projres:
        audit_override["attacks"] = [
            attack
            for attack in audit_override.get("attacks", [])
            if attack != "projres"
        ]
        audit_override["exact_batch_membership_attacks"] = [
            attack
            for attack in configured_exact_batch_attacks
            if attack != "projres"
        ]
        unified_projres = False
    elif unified_projres and projres_enabled:
        attacks = audit_override.setdefault("attacks", [])
        if "projres" not in attacks:
            attacks.append("projres")
    if unified_projres and args.projres_round is not None:
        raise ValueError(
            "Unified Visual Adapter ProjRes uses the shared attack interval; "
            "--projres-round is only available to standalone ProjRes."
        )
    if projres_enabled:
        if unified_projres:
            batch_size = int(spec.get("common", {}).get("batch_size", 0))
            ratio = int(
                audit_override.get(
                    "exact_batch_nonmember_to_member_ratio",
                    audit_override.get("nonmember_to_member_ratio", 1),
                )
            )
            expected_nonmembers = batch_size * ratio
            if int(projres.get("max_candidates", 0)) != batch_size or (
                int(projres.get("min_nonmembers", 0))
                != expected_nonmembers
                or int(projres.get("max_nonmembers", 0))
                != expected_nonmembers
            ):
                raise ValueError(
                    "Unified Visual Adapter ProjRes must use the shared full "
                    "batch and exact 1:N nonmember count."
                )
        else:
            if int(projres.get("min_nonmembers", 1000)) < 1000:
                raise ValueError(
                    "Standalone ProjRes needs at least 1000 non-members for "
                    "0.1% FPR."
                )
            projres.setdefault("max_nonmembers", 20000)
    standalone_projres_enabled = projres_enabled and not unified_projres

    jobs, results_root = build_jobs(
        spec,
        spec_path,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        num_global_iters=args.rounds,
        dirichlet_alpha=args.dirichlet_alpha,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        learning_rate_decay_interval=args.learning_rate_decay_interval,
        partition_mode=args.partition_mode,
    )
    jobs = filter_jobs_by_dataset(jobs, args.datasets)
    if args.max_runs is not None:
        if args.max_runs <= 0:
            raise ValueError("--max-runs must be positive.")
        jobs = jobs[: args.max_runs]
    gpu_text = args.gpus or ",".join(str(value) for value in spec.get("gpus", [0]))
    gpus = [int(value.strip()) for value in gpu_text.split(",") if value.strip()]
    if not gpus or min(gpus) < 0:
        raise ValueError("--gpus must contain non-negative GPU indices.")

    if args.dry_run:
        for index, job in enumerate(jobs):
            print(_job_hyperparameters_block(job))
            if standalone_projres_enabled:
                print(_projres_parameters_block(job, projres))
            for command in _commands(job, gpus[index % len(gpus)]):
                print("COMMAND " + _command_text(command))
        print(f"Expanded {len(jobs)} jobs; results_root={results_root}")
        return 0
    if args.summarize_only:
        jobs = filter_jobs_by_dataset(
            discover_existing_jobs(results_root, sweep_name), args.datasets
        )
        if args.defense is not None:
            jobs = [job for job in jobs if job.defense == args.defense]
        if args.max_runs is not None:
            jobs = jobs[-args.max_runs :]
        complete, attack_rows = summarize(jobs, results_root, sweep_name)
        projres_rows = (
            summarize_projres(jobs, results_root, sweep_name)
            if standalone_projres_enabled
            else 0
        )
        print(
            f"Summarized runs={complete}/{len(jobs)}, generic_rows={attack_rows}, "
            f"projres_rows={projres_rows}."
        )
        return 0

    workers = args.jobs if args.jobs is not None else int(spec.get("jobs", 1))
    if workers <= 0:
        raise ValueError("--jobs must be positive.")
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_job,
                job,
                gpus[index % len(gpus)],
                projres,
                args.skip_projres,
                args.force,
                True,
            ): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            run_id, succeeded, message = future.result()
            print(f"{'finished' if succeeded else 'FAILED'} {run_id}: {message}")
            failures += int(not succeeded)
    complete, attack_rows = summarize(jobs, results_root, sweep_name)
    projres_rows = (
        summarize_projres(jobs, results_root, sweep_name)
        if standalone_projres_enabled
        else 0
    )
    print(
        f"Summary complete: runs={complete}/{len(jobs)}, "
        f"generic_rows={attack_rows}, projres_rows={projres_rows}, failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
