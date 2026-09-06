#!/usr/bin/env python3
"""Expand and run the repository's supported privacy experiment matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from typing import Any, Iterable

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPOSITORY_ROOT / "configs" / "experiment_catalog.yaml"
_PRINT_LOCK = threading.Lock()
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.result_formatting import format_sweep_summary


@dataclass(frozen=True)
class ExperimentTask:
    run_id: str
    model: str
    runner: str
    dataset: str
    attacks: tuple[str, ...]
    defense: str
    seed: int
    target_client_id: int
    config: dict[str, Any]
    run_dir: Path
    config_path: Path
    gpu: int | None = None


@dataclass(frozen=True)
class TaskResult:
    task: ExperimentTask
    returncode: int
    started_at: str
    finished_at: str


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (REPOSITORY_ROOT / path).resolve()
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 顶层必须是映射: {resolved}")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    keys = [item.strip() for item in path.split(".") if item.strip()]
    if not keys:
        raise ValueError("--set 的配置路径不能为空。")
    cursor = config
    for key in keys[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"无法覆盖 {path!r}: {key!r} 不是映射。")
        cursor = child
    cursor[keys[-1]] = value


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("逗号分隔参数不能为空。")
    return list(dict.fromkeys(values))


def parse_int_csv(value: str) -> list[int]:
    values = parse_csv(value)
    try:
        parsed = [int(item) for item in values]
    except ValueError as error:
        raise ValueError(f"需要逗号分隔的整数，实际为 {value!r}。") from error
    return list(dict.fromkeys(parsed))


def normalize_values(
    values: Iterable[str], aliases: dict[str, str]
) -> list[str]:
    return list(dict.fromkeys(aliases.get(value, value) for value in values))


def _selection(
    raw: str,
    *,
    defaults: list[str],
    supported: list[str],
    aliases: dict[str, str],
    allow_none: bool = False,
) -> list[str]:
    requested = parse_csv(raw)
    if len(requested) == 1 and requested[0] == "default":
        return list(defaults)
    if len(requested) == 1 and requested[0] == "all":
        return list(supported)
    if allow_none and len(requested) == 1 and requested[0] == "none":
        return []
    reserved = {"default", "all"}
    if allow_none:
        reserved.add("none")
    if any(item in reserved for item in requested):
        raise ValueError("default、all 和 none 不能与具体名称混用。")
    return normalize_values(requested, aliases)


def _defense_override(
    catalog: dict[str, Any], model: str, defense: str
) -> dict[str, Any]:
    entry = catalog.get("defense_overrides", {}).get(defense)
    if not isinstance(entry, dict):
        raise ValueError(f"catalog 缺少防御配置: {defense}")
    common = entry.get("common", {})
    model_override = entry.get("models", {}).get(model, {})
    return deep_merge(common, model_override)


def _disable_resnet_paper_protocol(config: dict[str, Any]) -> None:
    config.setdefault("resnet18", {})["paper_protocol"] = False
    config.setdefault("audit", {})["www_candidate_scoring"] = False


def _resolve_projres_candidate_defaults(
    config: dict[str, Any], explicit_keys: set[str]
) -> None:
    """Derive unified ProjRes bounds from the final batch/sampling protocol.

    Model YAMLs describe their baseline batch, so their candidate counts become
    stale after --set batch_size. Explicit ProjRes overrides remain subject to
    the runner's protocol validation; independent/disabled ProjRes is untouched.
    """
    audit = config.get("audit", {})
    projres = config.get("projres", {})
    if (
        "projres" not in audit.get("exact_batch_membership_attacks", [])
        or not projres.get("enabled", False)
    ):
        return
    defense = str(config.get("defense", {}).get("name", "none")).lower()
    if defense in {"record_dp", "www"}:
        members = nonmembers = 0
    else:
        members = int(config["batch_size"])
        nonmembers = members * int(audit.get("exact_batch_nonmember_to_member_ratio", 10))
    for key, value in {
        "max_candidates": members,
        "min_nonmembers": nonmembers,
        "max_nonmembers": nonmembers,
    }.items():
        if key not in explicit_keys:
            projres[key] = value


def resolve_model_config(
    catalog: dict[str, Any],
    *,
    model: str,
    dataset: str,
    attacks: list[str],
    defense: str,
    seed: int,
    target_client_id: int,
    results_dir: str | Path,
    rounds: int | None = None,
    learning_rate: float | None = None,
    partition_mode: str | None = None,
    dirichlet_alpha: float | None = None,
    require_cuda: bool | None = None,
    dotted_overrides: list[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve one runnable configuration; exposed for focused regression tests."""
    profile = catalog["models"][model]
    config = load_yaml(profile["config"])
    config = deep_merge(config, _defense_override(catalog, model, defense))
    config["dataset_name"] = dataset
    if profile["runner"] == "text":
        config["dataset_path"] = f"./data/huggingface/{dataset}"

    audit = config.setdefault("audit", {})
    if profile["runner"] == "text":
        audit.setdefault("grad_sample_backend", "auto")
        audit.setdefault("grad_sample_chunk_size", 4)
        audit.setdefault("gradient_update_cache_mb", 2048)
    exact_supported = set(catalog["attacks"]["exact_batch"])
    audit["enabled"] = bool(attacks)
    audit["attacks"] = list(attacks)
    audit["target_client_id"] = target_client_id
    audit["audit_client_ids"] = [target_client_id]
    audit["exact_batch_membership_attacks"] = [
        attack for attack in attacks if attack in exact_supported
    ]
    intervals = dict(audit.get("attack_audit_intervals", {}))
    audit["attack_audit_intervals"] = {
        attack: intervals[attack] for attack in attacks if attack in intervals
    }
    projres = config.setdefault("projres", {})
    projres["enabled"] = "projres" in attacks

    config["defense"] = deep_merge(
        config.get("defense", {}), {"name": defense}
    )
    config["seed"] = seed
    config["results_dir"] = str(Path(results_dir).resolve())
    config["results_dir_is_run_dir"] = True

    if rounds is not None:
        if rounds <= 0:
            raise ValueError("--rounds 必须为正整数。")
        config["num_global_iters"] = rounds
    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("--learning-rate 必须为正数。")
        config["learning_rate"] = learning_rate
    if partition_mode is not None:
        config["partition_mode"] = partition_mode
    if dirichlet_alpha is not None:
        if dirichlet_alpha <= 0:
            raise ValueError("--dirichlet-alpha 必须为正数。")
        config["dirichlet_alpha"] = dirichlet_alpha
        if partition_mode is None:
            config["partition_mode"] = "dirichlet"
    if require_cuda is not None:
        config["require_cuda"] = require_cuda

    if model == "resnet18" and (
        not attacks
        or defense != "none"
        or rounds is not None
        or learning_rate is not None
        or partition_mode is not None
        or dirichlet_alpha is not None
    ):
        _disable_resnet_paper_protocol(config)

    explicit_projres_keys: set[str] = set()
    for path, value in dotted_overrides or []:
        set_dotted(config, path, copy.deepcopy(value))
        if path == "projres" and isinstance(value, dict):
            explicit_projres_keys = set(value)
        elif path.startswith("projres."):
            explicit_projres_keys.add(path.split(".")[1])
    _resolve_projres_candidate_defaults(config, explicit_projres_keys)
    return config


def validate_resolved_config(config: dict[str, Any], runner: str) -> None:
    if runner == "vision":
        from main import validate_config
    else:
        from scripts.run_fedllm_adapter import validate_config
    validate_config(config)


def _task_id(
    config: dict[str, Any],
    model: str,
    dataset: str,
    defense: str,
    started: dt.datetime,
) -> str:
    canonical = copy.deepcopy(config)
    canonical.pop("results_dir", None)
    canonical.pop("gpu", None)
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    method = (
        "fedsgd"
        if "architecture" in config
        else config.get("aggregator", "fedsgd")
    )
    target = config.get("audit", {}).get("target_client_id", 0)
    return (
        f"{started:%Y-%m-%d_%H-%M-%S-%f}_{model}_{dataset}_{method}_"
        f"{defense}_seed{config['seed']}_target{target}_{digest}"
    )


def _model_selection(catalog: dict[str, Any], raw: str) -> list[str]:
    profiles = list(catalog["models"])
    aliases = catalog.get("aliases", {}).get("models", {})
    defaults = catalog.get("defaults", {}).get("models", profiles)
    selected = _selection(
        raw, defaults=defaults, supported=profiles, aliases=aliases
    )
    unknown = [value for value in selected if value not in catalog["models"]]
    if unknown:
        raise ValueError("未知模型: " + ", ".join(unknown))
    return selected


def build_tasks(
    catalog: dict[str, Any], args: argparse.Namespace
) -> tuple[list[ExperimentTask], list[str]]:
    models = _model_selection(catalog, args.models)
    aliases = catalog.get("aliases", {})
    targets = parse_int_csv(args.target_clients)
    selected_seeds = (
        None
        if args.seeds.strip().lower() == "default"
        else parse_int_csv(args.seeds)
    )
    if (
        selected_seeds is not None and min(selected_seeds) < 0
    ) or min(targets) < 0:
        raise ValueError("seed 和 target client 均不能为负数。")
    dotted_overrides = []
    for expression in args.set_values:
        if "=" not in expression:
            raise ValueError(f"--set 需要 path=value，实际为 {expression!r}。")
        path, raw_value = expression.split("=", 1)
        dotted_overrides.append((path.strip(), yaml.safe_load(raw_value)))

    results_root = resolve_path(args.results_root)
    started = args.started_at or dt.datetime.now()
    tasks: list[ExperimentTask] = []
    skipped: list[str] = []
    for model in models:
        profile = catalog["models"][model]
        seeds = (
            list(profile.get("default_seeds", [42]))
            if selected_seeds is None
            else selected_seeds
        )
        datasets = _selection(
            args.datasets,
            defaults=profile["default_datasets"],
            supported=profile["datasets"],
            aliases=aliases.get("datasets", {}),
        )
        unsupported_datasets = [
            item for item in datasets if item not in profile["datasets"]
        ]
        datasets = [item for item in datasets if item in profile["datasets"]]
        if unsupported_datasets:
            skipped.append(
                f"{model}: 跳过不兼容数据集 {','.join(unsupported_datasets)}"
            )
        if not datasets:
            skipped.append(f"{model}: 没有兼容的数据集，未生成任务")
            continue

        attacks = _selection(
            args.attacks,
            defaults=profile["default_attacks"],
            supported=profile["supported_attacks"],
            aliases={},
            allow_none=True,
        )
        unsupported_attacks = [
            item for item in attacks if item not in profile["supported_attacks"]
        ]
        if unsupported_attacks:
            skipped.append(
                f"{model}: 不支持攻击 {','.join(unsupported_attacks)}，未生成该模型任务"
            )
            continue

        defenses = _selection(
            args.defenses,
            defaults=profile["default_defenses"],
            supported=profile["supported_defenses"],
            aliases=aliases.get("defenses", {}),
        )
        unsupported_defenses = [
            item for item in defenses if item not in profile["supported_defenses"]
        ]
        defenses = [
            item for item in defenses if item in profile["supported_defenses"]
        ]
        if unsupported_defenses:
            skipped.append(
                f"{model}: 跳过不兼容防御 {','.join(unsupported_defenses)}"
            )
        if not defenses:
            skipped.append(f"{model}: 没有兼容的防御，未生成任务")
            continue

        # A sweep containing CoFedMID must use the same holdout reservation
        # for its controls. Explicit --set applies to every defense as usual.
        study_overrides = list(dotted_overrides)
        split_path = "defense.cofedmid_validation_fraction"
        if "cofedmid" in defenses and not any(path == split_path for path, _ in study_overrides):
            cofedmid_config = _defense_override(catalog, model, "cofedmid")
            fraction = cofedmid_config.get("defense", {}).get("cofedmid_validation_fraction", 0.1)
            study_overrides.append((split_path, fraction))

        for dataset, defense, seed, target in itertools.product(
            datasets, defenses, seeds, targets
        ):
            provisional = results_root / "__pending__"
            config = resolve_model_config(
                catalog,
                model=model,
                dataset=dataset,
                attacks=attacks,
                defense=defense,
                seed=seed,
                target_client_id=target,
                results_dir=provisional,
                rounds=args.rounds,
                learning_rate=args.learning_rate,
                partition_mode=args.partition_mode,
                dirichlet_alpha=args.dirichlet_alpha,
                require_cuda=args.require_cuda,
                dotted_overrides=study_overrides,
            )
            run_id = _task_id(config, model, dataset, defense, started)
            run_dir = results_root / run_id
            config["results_dir"] = str(run_dir)
            validate_resolved_config(config, profile["runner"])
            tasks.append(
                ExperimentTask(
                    run_id=run_id,
                    model=model,
                    runner=profile["runner"],
                    dataset=dataset,
                    attacks=tuple(attacks),
                    defense=defense,
                    seed=seed,
                    target_client_id=target,
                    config=config,
                    run_dir=run_dir,
                    config_path=run_dir / "run_config.yaml",
                )
            )
    if args.max_runs is not None:
        if args.max_runs <= 0:
            raise ValueError("--max-runs 必须为正整数。")
        tasks = tasks[: args.max_runs]
    if not tasks:
        raise ValueError("所选条件没有生成任何兼容任务。")
    return tasks, skipped


def task_command(task: ExperimentTask) -> list[str]:
    entry = (
        "main.py"
        if task.runner == "vision"
        else "scripts/run_fedllm_adapter.py"
    )
    return [
        sys.executable,
        str(REPOSITORY_ROOT / entry),
        "--config",
        str(task.config_path),
    ]


def print_catalog(catalog: dict[str, Any]) -> None:
    print("MODEL\tRUNNER\tDATASETS\tATTACKS\tDEFENSES")
    for name, profile in catalog["models"].items():
        print(
            "\t".join(
                (
                    name,
                    profile["runner"],
                    ",".join(profile["datasets"]),
                    ",".join(profile["supported_attacks"]),
                    ",".join(profile["supported_defenses"]),
                )
            )
        )


def print_plan(tasks: list[ExperimentTask], skipped: list[str]) -> None:
    for message in skipped:
        print(f"SKIP | {message}")
    print(f"TASKS | {len(tasks)}")
    for index, task in enumerate(tasks, 1):
        attacks = ",".join(task.attacks) if task.attacks else "none"
        config = task.config
        method = (
            "fedsgd"
            if task.runner == "text"
            else config.get("aggregator", "unknown")
        )
        partition = config.get("partition_mode", "iid")
        if config.get("use_full_dataset"):
            data_view = "full"
        elif config.get("fpl_shots") is not None:
            data_view = f"{config['fpl_shots']}-shot"
        else:
            data_view = "full"
        print(
            f"[{index:03d}] model={task.model} dataset={task.dataset} "
            f"defense={task.defense} attacks={attacks} seed={task.seed} "
            f"target={task.target_client_id}"
        )
        print(
            f"      protocol=method:{method} "
            f"weighting:{config.get('aggregation_weighting', 'uniform')} "
            f"users:{config.get('sample_users')}/{config.get('total_users')} "
            f"rounds:{config.get('num_global_iters')} "
            f"batch:{config.get('batch_size')} lr:{config.get('learning_rate')} "
            f"partition:{partition} data:{data_view}"
        )
        defense_config = config.get("defense", {})
        if "projres" in config.get("audit", {}).get("exact_batch_membership_attacks", []):
            bounds = config["projres"]
            print(
                f"      projres=members:{bounds['max_candidates']} "
                f"nonmembers:{bounds['min_nonmembers']}/{bounds['max_nonmembers']} "
                "(0=dynamic actual batch)"
            )
        if task.defense in {"www", "record_dp"}:
            print(
                f"      per_record_gradients=backend:{defense_config.get('grad_sample_backend', 'auto')} "
                f"chunk:{defense_config.get('microbatch_size', 4)}"
            )
        if task.defense == "www":
            print(
                f"      www=epsilon:{defense_config['target_epsilon']} "
                f"clip:{defense_config['max_grad_norm']} delta:{defense_config['delta']} "
                f"tail:{defense_config['www_tail_fraction']}*expected_batch ranking:ascending "
                f"noise:{defense_config['noise_multiplier']} "
                "sampling:poisson adjacency:add_remove accounting:poisson_sampled_gaussian_rdp"
            )
        if task.defense == "cofedmid":
            print(
                f"      cofedmid=clients:{defense_config['cofedmid_clients']} "
                f"coverage:{defense_config['cofedmid_coverage']} "
                f"modules:{defense_config['cofedmid_partition']}/"
                f"{defense_config['cofedmid_compensation']}/"
                f"{defense_config['cofedmid_perturbation']} "
                f"first_recycling_round:{defense_config['cofedmid_init_round'] + 1} "
                f"recycle_ratio:{defense_config['cofedmid_recycle_ratio']} "
                f"noise_std:{defense_config['cofedmid_noise_std']} "
                f"noise_space:{defense_config['cofedmid_noise_space']} "
                f"parameter_tail:{defense_config['cofedmid_perturb_ratio']}"
            )
        validation_fraction = defense_config.get("cofedmid_validation_fraction", 0)
        if validation_fraction:
            print(f"      defense_validation_fraction={validation_fraction}")
        print(f"      run_dir={task.run_dir}")
        print(f"      command={shlex.join(task_command(task))}")


def _timestamped_line(message: str) -> str:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    return f"{timestamp} | {message.rstrip()}\n"


def _task_header(task: ExperimentTask) -> str:
    """Identify one task once instead of prefixing every child log line."""
    context = (
        f"model={task.model} | dataset={task.dataset} | defense={task.defense} | "
        f"run={task.run_id} | phase=train | gpu={task.gpu}"
    )
    return _timestamped_line(f"TASK | {context}")


def _forward_child_line(line: str) -> str:
    """Preserve the child's own timestamp, level, logger, and message."""
    return line if line.endswith("\n") else line + "\n"


def run_task(task: ExperimentTask) -> TaskResult:
    started = dt.datetime.now(dt.timezone.utc)
    task.run_dir.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(task.config)
    if task.gpu is not None:
        config["gpu"] = task.gpu
    with task.config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    env = os.environ.copy()
    env["FEDMIA_LAUNCHER_LOG_CAPTURE"] = "1"
    command = task_command(task)
    with (task.run_dir / "run.log").open("w", encoding="utf-8") as log:
        header = _task_header(task)
        command_line = _timestamped_line("COMMAND | " + shlex.join(command))
        log.write(header)
        log.write(command_line)
        with _PRINT_LOCK:
            sys.stdout.write(header)
            sys.stdout.write(command_line)
            sys.stdout.flush()
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            formatted = _forward_child_line(raw_line)
            log.write(formatted)
            log.flush()
            with _PRINT_LOCK:
                sys.stdout.write(formatted)
                sys.stdout.flush()
        returncode = process.wait()
        footer = _timestamped_line(f"EXIT | returncode={returncode}")
        log.write(footer)
        with _PRINT_LOCK:
            sys.stdout.write(footer)
            sys.stdout.flush()
    finished = dt.datetime.now(dt.timezone.utc)
    return TaskResult(
        task=task,
        returncode=returncode,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
    )


def _assign_gpus(tasks: list[ExperimentTask], gpus: list[int]) -> list[ExperimentTask]:
    if not gpus:
        return tasks
    return [
        replace(task, gpu=gpus[index % len(gpus)])
        for index, task in enumerate(tasks)
    ]


def _failed_result(task: ExperimentTask, error: Exception) -> TaskResult:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with _PRINT_LOCK:
        print(f"FAILED | {task.run_id} | {error}", file=sys.stderr)
    return TaskResult(task, 1, now, now)


def run_tasks(
    tasks: list[ExperimentTask], jobs: int, gpus: list[int]
) -> list[TaskResult]:
    if jobs <= 0:
        raise ValueError("--jobs 必须为正整数。")
    assigned = _assign_gpus(tasks, gpus)
    results: list[TaskResult] = []
    if jobs == 1:
        for task in assigned:
            try:
                results.append(run_task(task))
            except Exception as error:
                results.append(_failed_result(task, error))
        return results
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_task = {executor.submit(run_task, task): task for task in assigned}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append(_failed_result(task, error))
    order = {task.run_id: index for index, task in enumerate(assigned)}
    return sorted(results, key=lambda result: order[result.task.run_id])


def _last_training_metric(
    run_dir: Path, primary_metric: str | None = None
) -> tuple[str | None, float | None]:
    path = run_dir / "training_metrics.csv"
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, None
    metric_order = [primary_metric] if primary_metric else []
    metric_order.extend(["accuracy", "mcc", "test_accuracy"])
    for key in dict.fromkeys(metric_order):
        value = rows[-1].get(key)
        if value not in {None, ""}:
            return key, float(value)
    return None, None


def print_result_overview(results: list[TaskResult]) -> None:
    records = []
    for result in results:
        task = result.task
        metric, value = _last_training_metric(task.run_dir, task.config.get("primary_metric"))
        status = "OK" if result.returncode == 0 else "FAILED"
        audit_file = task.run_dir / "privacy_audit" / "summary.json"
        if status == "OK" and audit_file.exists():
            with audit_file.open(encoding="utf-8") as handle:
                if (json.load(handle) or {}).get("errors"):
                    status = "PARTIAL"
        try:
            seconds = (dt.datetime.fromisoformat(result.finished_at)
                       - dt.datetime.fromisoformat(result.started_at)).total_seconds()
        except (TypeError, ValueError):
            seconds = None
        records.append({
            "model": task.model, "dataset": task.dataset, "defense": task.defense,
            "seed": task.seed, "target_client_id": task.target_client_id,
            "status": status, "metric": metric, "value": value, "seconds": seconds,
        })
    print("\n" + format_sweep_summary(records) + "\n")


def write_outputs(
    results: list[TaskResult], results_root: Path, invocation_id: str
) -> tuple[Path, Path]:
    manifest_path = results_root / f"experiment_manifest_{invocation_id}.json"
    summary_path = results_root / f"experiment_summary_{invocation_id}.csv"
    results_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "invocation_id": invocation_id,
        "tasks": [
            {
                "run_id": result.task.run_id,
                "model": result.task.model,
                "dataset": result.task.dataset,
                "attacks": list(result.task.attacks),
                "defense": result.task.defense,
                "seed": result.task.seed,
                "target_client_id": result.task.target_client_id,
                "gpu": result.task.gpu,
                "returncode": result.returncode,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "result_dir": str(result.task.run_dir),
            }
            for result in results
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    rows: list[dict[str, Any]] = []
    for result in results:
        metric_name, metric_value = _last_training_metric(
            result.task.run_dir,
            result.task.config.get("primary_metric"),
        )
        summary_file = result.task.run_dir / "privacy_audit" / "summary.json"
        attacks = []
        if summary_file.exists():
            with summary_file.open("r", encoding="utf-8") as handle:
                attacks = (json.load(handle) or {}).get("attacks", [])
        if not attacks:
            attacks = [{"attack": "none"}]
        for attack in attacks:
            reportable = attack.get("reportable_metrics", attack)
            rows.append(
                {
                    "run_id": result.task.run_id,
                    "model": result.task.model,
                    "dataset": result.task.dataset,
                    "defense": result.task.defense,
                    "seed": result.task.seed,
                    "target_client_id": result.task.target_client_id,
                    "attack": attack.get("attack", "none"),
                    "final_metric": metric_name,
                    "final_metric_value": metric_value,
                    "auc": reportable.get("auc", attack.get("auc")),
                    "tpr_at_fpr_0.001": reportable.get("tpr_at_fpr_0.001"),
                    "returncode": result.returncode,
                    "result_dir": str(result.task.run_dir),
                }
            )
    fields = list(rows[0]) if rows else []
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return manifest_path, summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一展开并运行模型 × 数据集 × 防御 × seed × 客户端隐私实验。"
    )
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument(
        "--models", default="default", help="default、all 或逗号分隔模型。"
    )
    parser.add_argument(
        "--datasets", default="default", help="default、all 或逗号分隔数据集。"
    )
    parser.add_argument(
        "--attacks", default="default", help="default、all、none 或逗号分隔攻击。"
    )
    parser.add_argument(
        "--defenses", default="default", help="default、all 或逗号分隔防御。"
    )
    parser.add_argument("--seeds", default="default", help="default 或逗号分隔整数。")
    parser.add_argument("--target-clients", default="0")
    parser.add_argument("--gpus", default="0", help="逗号分隔 GPU；空字符串表示 CPU。")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--partition-mode", choices=("iid", "dirichlet", "pathological")
    )
    parser.add_argument("--dirichlet-alpha", type=float)
    cuda = parser.add_mutually_exclusive_group()
    cuda.add_argument("--require-cuda", dest="require_cuda", action="store_true")
    cuda.add_argument("--no-require-cuda", dest="require_cuda", action="store_false")
    parser.set_defaults(require_cuda=None)
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="对最终配置做 YAML 类型的点路径覆盖，可重复。",
    )
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="列出兼容矩阵后退出。")
    args = parser.parse_args(argv)
    args.started_at = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog = load_yaml(args.catalog)
    if args.list:
        print_catalog(catalog)
        return 0
    args.started_at = dt.datetime.now()
    tasks, skipped = build_tasks(catalog, args)
    print_plan(tasks, skipped)
    if args.dry_run:
        print("DRY-RUN | 未创建结果目录，未启动训练。")
        return 0
    gpus = parse_int_csv(args.gpus) if args.gpus.strip() else []
    results = run_tasks(tasks, args.jobs, gpus)
    invocation_id = args.started_at.strftime("%Y%m%d_%H%M%S_%f")
    manifest, summary = write_outputs(
        results, resolve_path(args.results_root), invocation_id
    )
    failed = [result for result in results if result.returncode != 0]
    not_started = len(tasks) - len(results)
    print_result_overview(results)
    print(
        f"DONE | completed={len(results) - len(failed)} failed={len(failed)} "
        f"not_started={not_started}"
    )
    print(f"MANIFEST | {manifest}")
    print(f"SUMMARY | {summary}")
    return 1 if failed or not_started else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as error:
        print(f"ERROR | {error}", file=sys.stderr)
        raise SystemExit(2) from error
