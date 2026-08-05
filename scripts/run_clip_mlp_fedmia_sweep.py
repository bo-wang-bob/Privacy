#!/usr/bin/env python3
"""Run multi-dataset frozen-CLIP FedMIA experiments and optional ProjRes."""

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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_LOCK = threading.Lock()


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


def _stable_run_id(config: dict[str, Any], dataset: str, seed: int, target: int) -> str:
    canonical = copy.deepcopy(config)
    canonical.pop("gpu", None)
    canonical.pop("results_dir", None)
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"{dataset}_none_seed{seed}_target{target}_{digest}"


def build_jobs(
    spec: dict[str, Any],
    spec_path: Path,
    data_root: str | None = None,
    cache_dir: str | None = None,
    num_global_iters: int | None = None,
    dirichlet_alpha: float | None = None,
    learning_rate: float | None = None,
) -> tuple[list[SweepJob], Path]:
    del spec_path
    base_path = _resolve_path(
        spec.get("base_config", "configs/clip_mlp_low_fpr_attacks.yaml")
    )
    base = _deep_merge(_load_yaml(base_path), spec.get("common", {}))
    results_root = _resolve_path(
        spec.get("results_root", "results/clip_mlp_fedmia_attacks")
    )
    datasets = spec.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("sweep.datasets must be a non-empty list.")
    seeds = [int(value) for value in spec.get("seeds", [42])]
    targets = [int(value) for value in spec.get("target_client_ids", [0])]
    if not seeds or not targets:
        raise ValueError("sweep seeds and target_client_ids must be non-empty.")
    defenses = spec.get("defenses", [{"name": "none"}])
    if len(defenses) != 1 or str(defenses[0].get("name", "none")).lower() != "none":
        raise ValueError("The frozen-CLIP attack sweep supports defense=none only.")

    jobs = []
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
        if learning_rate is not None:
            if learning_rate <= 0:
                raise ValueError("--learning-rate must be positive.")
            config["learning_rate"] = learning_rate
        config["seed"] = seed
        config["aggregator"] = "fedavg"
        config["defense"] = _deep_merge(config.get("defense", {}), {"name": "none"})
        config.setdefault("audit", {})["target_client_id"] = target
        dataset = str(config["dataset_name"])
        run_id = _stable_run_id(config, dataset, seed, target)
        run_root = results_root / "runs" / run_id
        config["results_dir"] = str(run_root)
        config["results_dir_is_run_dir"] = True
        jobs.append(
            SweepJob(
                run_id=run_id,
                config=config,
                dataset=dataset,
                method="fedavg",
                seed=seed,
                target_client_id=target,
                defense="none",
                run_root=run_root,
                config_path=run_root / "run_config.yaml",
            )
        )
    return jobs, results_root


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
    return job.run_root if not audit.get("errors") and completed == expected else None


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
            "total_users": config.get("total_users"),
            "sample_users": config.get("sample_users"),
            "num_global_iters": config.get("num_global_iters"),
            "local_epochs": config.get("local_epochs"),
        },
        "optimization": {
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"),
            "eval_batch_size": config.get("eval_batch_size"),
            "eval_interval": config.get("eval_interval"),
        },
        model_type: config.get(model_type, {}),
        "privacy_audit": config.get("audit", {}),
    }
    rows = _flatten(selected)
    width = max(len(key) for key, _ in rows)
    divider = "=" * 88
    body = "\n".join(f"  {key:<{width}} : {_display(value)}" for key, value in rows)
    return f"{divider}\nHYPERPARAMETERS | {job.run_id}\n{divider}\n{body}\n{divider}"


def summarize(jobs: list[SweepJob], results_root: Path) -> tuple[int, int]:
    rows = []
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
            rows.append(
                {
                    "run_id": job.run_id,
                    "dataset": job.dataset,
                    "seed": job.seed,
                    "target_client_id": job.target_client_id,
                    "attack": attack["attack"],
                    "final_accuracy": accuracy,
                    "auc": attack.get("auc"),
                    "tpr_at_fpr_0.1": reportable.get("tpr_at_fpr_0.1"),
                    "tpr_at_fpr_0.01": reportable.get("tpr_at_fpr_0.01"),
                    "tpr_at_fpr_0.001": reportable.get("tpr_at_fpr_0.001"),
                    "member_count": attack.get("member_count"),
                    "nonmember_count": attack.get("nonmember_count"),
                    "fpr_resolution": attack.get("fpr_resolution"),
                    "result_dir": str(result_dir),
                }
            )
    results_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id", "dataset", "seed", "target_client_id", "attack",
        "final_accuracy", "auc", "tpr_at_fpr_0.1", "tpr_at_fpr_0.01",
        "tpr_at_fpr_0.001", "member_count", "nonmember_count",
        "fpr_resolution", "result_dir",
    ]
    with (results_root / "summary_by_run.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["attack"]), []).append(row)
    aggregate_rows = []
    for (dataset, attack), group in sorted(grouped.items()):
        result = {"dataset": dataset, "attack": attack, "runs": len(group)}
        for metric in ("final_accuracy", "auc", "tpr_at_fpr_0.001"):
            values = [float(row[metric]) for row in group if row[metric] is not None]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else None
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        aggregate_rows.append(result)
    aggregate_fields = [
        "dataset", "attack", "runs", "final_accuracy_mean", "final_accuracy_std",
        "auc_mean", "auc_std", "tpr_at_fpr_0.001_mean", "tpr_at_fpr_0.001_std",
    ]
    with (results_root / "summary_aggregate.csv").open(
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


def _commands(job, gpu: int, projres: dict[str, Any], skip_projres: bool):
    main_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "main.py"),
        "--config",
        str(job.config_path),
        "--gpu",
        str(gpu),
    ]
    projres_command = None
    if not skip_projres and bool(projres.get("enabled", True)):
        projres_command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/validate_projres_mlp_real.py"),
            "--config",
            str(job.config_path),
            "--gpu",
            str(gpu),
            "--target-client",
            str(job.target_client_id),
            "--threshold",
            str(projres.get("threshold", 0.01)),
            "--max-candidates",
            str(projres.get("max_candidates", 32)),
            "--min-nonmembers",
            str(projres.get("min_nonmembers", 1000)),
            "--max-nonmembers",
            str(projres.get("max_nonmembers", 0)),
            "--output",
            str(_projres_path(job)),
        ]
    return main_command, projres_command


def _command_text(command: list[str]) -> str:
    import shlex

    return shlex.join(command)


def _projres_parameters_block(job, projres: dict[str, Any]) -> str:
    divider = "=" * 88
    model_type = str(job.config.get("model_type", "clip_mlp"))
    attacked_layer = {
        "clip_mlp": "classifier.0.weight",
        "visual_adapter": "adapter.net.0.weight",
    }.get(model_type, "first_trainable_linear_weight")
    return "\n".join(
        (
            divider,
            f"STRICT PROJRES | {job.run_id}",
            divider,
            f"  threshold          : {projres.get('threshold', 0.01)}",
            f"  max_candidates     : {projres.get('max_candidates', 32)}",
            f"  min_nonmembers     : {projres.get('min_nonmembers', 1000)}",
            f"  max_nonmembers     : {projres.get('max_nonmembers', 0)}",
            f"  attacked_layer     : {attacked_layer}",
            "  threat_model       : one batch, one vanilla FedSGD step",
            divider,
        )
    )


def _run_logged(
    command: list[str],
    log_path: Path,
    *,
    stream_output: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    # The sweep runner already tees/captures the child stream into run.log.
    # Prevent main.py from opening the same log file with a second handler.
    environment["FEDMIA_LAUNCHER_LOG_CAPTURE"] = "1"
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND " + _command_text(command) + "\n")
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
            log_file.write(line)
            log_file.flush()
            with _OUTPUT_LOCK:
                sys.stdout.write(line)
                sys.stdout.flush()
        return int(process.wait())


def _announce_job_start(
    job: SweepJob,
    gpu: int,
    projres: dict[str, Any],
    skip_projres: bool,
) -> None:
    blocks = [
        f"STARTING JOB | {job.run_id} | gpu:{gpu}",
        _job_hyperparameters_block(job),
    ]
    if not skip_projres and bool(projres.get("enabled", True)):
        blocks.append(_projres_parameters_block(job, projres))
    # One print under a lock keeps concurrent jobs from interleaving parameter
    # blocks while still announcing each job only when its worker really starts.
    with _OUTPUT_LOCK:
        print("\n".join(blocks), flush=True)


def _announce_phase(job: SweepJob, phase: str, log_path: Path) -> None:
    with _OUTPUT_LOCK:
        print(
            f"RUNNING {phase} | {job.run_id} | log={log_path}",
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
    main_command, projres_command = _commands(job, gpu, projres, skip_projres)
    main_complete = _completed_result(job) is not None
    if force or not main_complete:
        if force:
            stale = job.run_root / "privacy_audit" / "summary.json"
            if stale.is_file():
                stale.unlink()
        main_log = job.run_root / "run.log"
        _announce_phase(job, "FEDAVG + GENERIC AUDIT", main_log)
        return_code = _run_logged(
            main_command,
            main_log,
            stream_output=stream_output,
        )
        if return_code != 0 or _completed_result(job) is None:
            return job.run_id, False, "main.py failed"
    if projres_command is not None and (force or not _projres_path(job).is_file()):
        projres_log = job.run_root / "projres_strict.log"
        _announce_phase(job, "STRICT PROJRES", projres_log)
        return_code = _run_logged(
            projres_command,
            projres_log,
            stream_output=stream_output,
        )
        if return_code != 0 or not _projres_path(job).is_file():
            return job.run_id, False, "strict ProjRes failed"
    return job.run_id, True, "complete"


def _projres_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if "result" in payload:
        return payload["result"]["attack"]["metrics"]
    return payload["pooled_metrics"]


def summarize_projres(jobs, results_root: Path) -> int:
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
                "target_client_id": job.target_client_id,
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
    path = results_root / "summary_projres.csv"
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
    parser.add_argument("--target-client", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", help="Comma-separated GPU indices.")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--rounds", "--num-global-iters", type=int)
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--projres-threshold", type=float)
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
    if args.attacks is not None:
        spec.setdefault("common", {}).setdefault("audit", {})["attacks"] = (
            _parse_csv(args.attacks)
        )
    if args.target_client is not None:
        spec["target_client_ids"] = [args.target_client]
    if args.seed is not None:
        spec["seeds"] = [args.seed]
    projres = dict(spec.get("projres", {}))
    projres_enabled = bool(projres.get("enabled", True)) and not args.skip_projres
    overrides = {
        "threshold": args.projres_threshold,
        "max_candidates": args.projres_max_candidates,
        "min_nonmembers": args.projres_min_out,
        "max_nonmembers": args.projres_max_out,
    }
    projres.update({key: value for key, value in overrides.items() if value is not None})
    if float(projres.get("threshold", 0.01)) < 0:
        raise ValueError("ProjRes threshold must be non-negative.")
    if int(projres.get("min_nonmembers", 1000)) < 1000:
        raise ValueError("ProjRes needs at least 1000 non-members for 0.1% FPR.")

    jobs, results_root = build_jobs(
        spec,
        spec_path,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        num_global_iters=args.rounds,
        dirichlet_alpha=args.dirichlet_alpha,
        learning_rate=args.learning_rate,
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
            if projres_enabled:
                print(_projres_parameters_block(job, projres))
            for command in _commands(
                job, gpus[index % len(gpus)], projres, args.skip_projres
            ):
                if command is not None:
                    print("COMMAND " + _command_text(command))
        print(f"Expanded {len(jobs)} jobs; results_root={results_root}")
        return 0
    if args.summarize_only:
        complete, attack_rows = summarize(jobs, results_root)
        projres_rows = summarize_projres(jobs, results_root) if projres_enabled else 0
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
                workers == 1,
            ): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            run_id, succeeded, message = future.result()
            print(f"{'finished' if succeeded else 'FAILED'} {run_id}: {message}")
            failures += int(not succeeded)
    complete, attack_rows = summarize(jobs, results_root)
    projres_rows = summarize_projres(jobs, results_root) if projres_enabled else 0
    print(
        f"Summary complete: runs={complete}/{len(jobs)}, "
        f"generic_rows={attack_rows}, projres_rows={projres_rows}, failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
