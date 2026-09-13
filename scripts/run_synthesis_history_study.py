"""Frozen 100-round validation of the compact local history/selection proposal."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import comparison_protocol, digest
from scripts.run_privacy_experiments import build_tasks, load_yaml, parse_args
from utils.confirmation_split import read_manifest, validate_mapping

STUDY = ROOT / "analysis_scripts/synthesis_history_compact_study_20260913"
PYTHON = "/root/.local/share/mamba/envs/pfedba/bin/python"
MANIFEST = ROOT / "analysis_scripts/risk_synthesis_confirmation_data_20260912/split.json"
MANIFEST_SHA = "00941791be9727022bb09c4b9d55f8b06d05a1cc3489a99753cbdb393151c178"
ARMS = {
    "baseline": dict(risk_history="none", candidate_selection="first_semantic", mode="risk"),
    "history": dict(risk_history="zero_risk_frequency", candidate_selection="first_semantic", mode="risk"),
    "selection": dict(risk_history="none", candidate_selection="least_local_similarity", mode="risk"),
    "combined": dict(risk_history="zero_risk_frequency", candidate_selection="least_local_similarity", mode="risk"),
    "combined_shuffle": dict(risk_history="zero_risk_frequency", candidate_selection="least_local_similarity", mode="shuffled_risk"),
}
CONTROLS = {
    "none_43": "2026-09-12_05-13-17-869543_clip_adapter_cifar100_fedavg_none_seed43_target0_027f42ccf1",
    "none_44": "2026-09-12_05-51-11-379403_clip_adapter_cifar100_fedavg_none_seed44_target0_79a17f7782",
    "none_45": "2026-09-12_07-34-10-909703_clip_adapter_cifar100_fedavg_none_seed45_target0_3c297fb21e",
    "baseline_43": "2026-09-12_17-45-01-206560_clip_adapter_cifar100_fedavg_risk_synthesis_seed43_target0_e61674f8b2",
    "baseline_shuffle_43": "2026-09-12_17-45-03-178555_clip_adapter_cifar100_fedavg_risk_synthesis_seed43_target0_9f2c452b24",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, state, *, exclusive=False):
    state["updated_at_utc"] = now()
    if exclusive:
        with path.open("x") as handle:
            json.dump(state, handle, indent=2)
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)


def normalized(args):
    import main as entry
    import yaml
    tasks, skipped = build_tasks(load_yaml(ROOT / "configs/experiment_catalog.yaml"), parse_args(args))
    if len(tasks) != 1 or skipped:
        raise ValueError("Expected exactly one supported experiment.")
    with tempfile.TemporaryDirectory(prefix="compact-synthesis-") as temporary:
        path = Path(temporary) / "config.yaml"
        path.write_text(yaml.safe_dump(tasks[0].config))
        argv = sys.argv
        try:
            sys.argv = ["main.py", "--config", str(path)]
            config = entry.parse_args()
        finally:
            sys.argv = argv
    entry.normalize_clip_adapter_config(config)
    entry.validate_config(config)
    config.setdefault(config["aggregator"], {}).setdefault("seed", config["seed"])
    return config


def prepare(study):
    read_manifest(MANIFEST, MANIFEST_SHA)
    controls = {}
    for key, name in CONTROLS.items():
        directory = ROOT / "results" / name
        config = load_yaml(directory / "run_config.yaml")
        if json.loads((directory / "performance_summary.json").read_text())["status"] != "completed":
            raise ValueError(f"Incomplete reused control: {key}")
        files = [directory / relative for relative in (
            "run_config.yaml", "training_metrics.csv", "performance_summary.json", "confirmation_split.json",
            "data_partition.json", "privacy_audit/summary.json", "privacy_audit/predictions.csv",
            "privacy_audit/signals.pt", "privacy_audit/candidate_selection.pt",
            "privacy_audit/client_train_update_candidate_selection.pt")]
        if key.startswith("baseline"):
            files += [directory / "risk_synthesis" / relative for relative in
                      ("synthesis_summary.json", "synthetic_exposure.csv", "source_exposure.pt")]
        controls[key] = dict(directory=str(directory), protocol=comparison_protocol(config),
                             defense=config["defense"], sources={str(p): digest(p) for p in files})
    # The original frozen experiment proves what code produced the controls.
    old = json.loads((ROOT / "analysis_scripts/risk_synthesis_all_study_plan_20260912.json").read_text())
    changed = [p for p, h in old["source_hashes"].items() if digest(ROOT / p) != h]
    expected = {"configs/experiment_catalog.yaml", "privacy_defenses/risk_synthesis.py", "privacy_defenses/controller.py"}
    if set(changed) != expected:
        raise ValueError(f"Unexpected source drift from controls: {changed}")
    jobs = []
    schedule = [(arm, 43) for arm in ("history", "selection", "combined", "combined_shuffle")]
    schedule += [(arm, seed) for seed in (44, 45) for arm in ("baseline", "combined", "combined_shuffle")]
    for arm, seed in schedule:
        args = ["--models", "clip_adapter", "--datasets", "cifar100", "--methods", "fedavg",
                "--defenses", "risk_synthesis", "--attacks", "all", "--seeds", str(seed), "--rounds", "100",
                "--set", f"confirmation_split_manifest={MANIFEST}",
                "--set", f"confirmation_split_sha256={MANIFEST_SHA}"]
        for key, value in ARMS[arm].items():
            args += ["--set", f"defense.synthesis.{key}={value}"]
        config = normalized(args)
        protocol = comparison_protocol(config)
        if protocol != controls[f"none_{seed}"]["protocol"]:
            differences = [k for k in protocol.keys() | controls[f"none_{seed}"]["protocol"].keys()
                           if protocol.get(k) != controls[f"none_{seed}"]["protocol"].get(k)]
            raise ValueError(f"Unmatched {arm}/{seed} protocol: {differences}")
        options = config["defense"]["synthesis"]
        assert options["attempts"] == 2 and options["replacement_policy"] == "all"
        assert len(config["audit"]["attacks"]) == 11 and config["audit"]["audit_client_ids"] == [0]
        jobs.append(dict(id=f"{arm}_{seed}", arm=arm, seed=seed, arguments=args,
                         protocol=protocol, defense=config["defense"]))
    paths = list(old["source_hashes"])
    paths += ["privacy_defenses/synthesis_history.py", "scripts/run_synthesis_history_study.py",
              "scripts/verify_synthesis_history.py"]
    plan = dict(schema_version=1, prepared_at_utc=now(), jobs=jobs, controls=controls,
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={p: digest(ROOT / p) for p in sorted(set(paths))}, baseline_source_changes=changed,
        baseline_reuse_reason="Only synthesis-specific controller branch, synthesis implementation and defaults changed; "
                              "shared training, models, optimizer, aggregation, data and attacks match the frozen baseline code.",
        confirmation_manifest=str(MANIFEST), confirmation_manifest_sha256=MANIFEST_SHA,
        scope="CLIP transformer Adapter/CIFAR100, FedAvg100, original100/class,10IID clients, target0. "
              "seed43 separates components; seeds44/45 repeat the predetermined combination and full shuffle. "
              "The source partition and seeds were previously studied; these are new method runs, not untouched-data confirmation.",
        hypothesis="Past zero-risk visits identify underprotected records; symmetric historical/loss ranks and "
                   "semantic-feasible least similarity to any local original may reduce residual membership leakage.",
        hyperparameters=dict(new_tunable_numeric=0, removed_proposal_parameters=["EMA beta", "retention exponent p", "rank mixture lambda"],
            history="cumulative mean of per-round zero-risk frequency", combination="equal midrank sum; loss gap breaks ties",
            candidates="reuse attempts=2; evaluate both, train one", unchanged_numeric_defaults={k: options[k] for k in (
                "noise_scale", "class_rank", "pooled_rank", "shrinkage", "margin_tolerance", "norm_ratio_min",
                "norm_ratio_max", "min_class_samples", "attempts")},
            caveat="Fixed equal rank contributions and zero-risk event are design assumptions. Existing geometry, noise "
                   "and semantic settings remain; this is not a wholly hyperparameter-free method."),
        acceptance=dict(
            overall_vs_none="For each seed: max11 AUC decrease>=0.02, max11 TPR@1% decrease>0, accuracy loss<=0.02.",
            added_value_vs_v4="Report each seed and mean; consistent added benefit requires lower max11 AUC and "
                              "max11 TPR@1% in all three seeds, with <=0.02 accuracy loss in each.",
            ordering_vs_combined_shuffle="Same per-seed added-value checks; conditional candidate intervals must be "
                                          "reported separately from seed variability, not substituted for it.",
            component_ablation="History-only and selection-only are exploratory seed43; do not claim multi-seed component efficacy.",
            selection="All100round checkpoints, all11 attacks and allscheduled jobs; no best-checkpoint or post-hoc winner substitution."),
        environment={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "PYTHONUNBUFFERED": "1"})
    study.mkdir(exist_ok=False)
    save(study / "plan.json", plan, exclusive=True)
    print(json.dumps(dict(plan=str(study / "plan.json"), sha256=digest(study / "plan.json"),
                         new_jobs=len(jobs), reused_controls=len(controls)), indent=2))


def check_sources(plan):
    for path, expected in plan["source_hashes"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen source changed: {path}")
    for control in plan["controls"].values():
        for path, expected in control["sources"].items():
            if digest(Path(path)) != expected:
                raise RuntimeError(f"Reused result changed: {path}")
    read_manifest(plan["confirmation_manifest"], plan["confirmation_manifest_sha256"])


def gpu_state(gpu):
    result = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    index, memory, utilization = map(int, result.split(","))
    return dict(index=index, memory_used_mib=memory, utilization_percent=utilization)


def worker(study, gpu, wait_for_gpu):
    plan_path = study / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan_sha = digest(plan_path)
    path = study / f"worker_gpu{gpu}.json"
    state = dict(status="starting", pid=os.getpid(), gpu=gpu, plan_sha256=plan_sha, started_at_utc=now())
    save(path, state, exclusive=True)
    active_execution = None
    try:
        check_sources(plan)
        while True:
            if digest(plan_path) != plan_sha:
                raise RuntimeError("Frozen plan changed.")
            pending = [job for job in plan["jobs"] if not (study / f'{job["id"]}.json').exists()]
            if not pending:
                state["status"] = "finished_claimed_queue"
                save(path, state)
                return
            resource = gpu_state(gpu)
            # Scheduling guard, not a defense hyperparameter. Never terminate or
            # crowd other GPU jobs. Recheck before each new training process.
            if resource["memory_used_mib"] >= 1024 or resource["utilization_percent"] > 5:
                state.update(status="waiting_for_idle_gpu", resource=resource, pending_jobs=len(pending))
                save(path, state)
                if not wait_for_gpu:
                    raise RuntimeError(f"GPU {gpu} is occupied; use --wait-for-gpu to queue safely.")
                time.sleep(30)
                continue
            if shutil.disk_usage(ROOT).free < 10 * 1024**3:
                raise RuntimeError("Less than 10 GiB free; preserving all prior user results.")
            claimed = False
            for job in pending:
                job_path = study / f'{job["id"]}.json'
                execution = dict(status="claimed", worker_pid=os.getpid(), gpu=gpu, job=job,
                                 plan_sha256=plan_sha, started_at_utc=now())
                try:
                    save(job_path, execution, exclusive=True)
                    claimed = True
                    break
                except FileExistsError:
                    continue
            if not claimed:
                continue
            active_execution = (job_path, execution)
            check_sources(plan)
            state.update(status="running", job=job["id"])
            save(path, state)
            command = [PYTHON, "scripts/run_privacy_experiments.py", *job["arguments"], "--gpus", str(gpu)]
            execution.update(status="running", command=command)
            directories = set()
            with (study / f'{job["id"]}_console.log').open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **plan["environment"]},
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                execution["runner_pid"] = process.pid
                state["runner_pid"] = process.pid
                save(job_path, execution)
                save(path, state)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if "Progress" in line or "RUN RESULTS" in line or "Traceback" in line:
                        print(f'[{job["id"]}] {line}', end="", flush=True)
                    match = re.match(r"\s*run_dir=(.+)\s*$", line)
                    if match:
                        directories.add(match.group(1).strip())
                        execution["result_directories"] = sorted(directories)
                        save(job_path, execution)
                execution["returncode"] = process.wait()
            if execution["returncode"] or len(directories) != 1:
                execution.update(status="failed", error="Training failure or ambiguous result directory")
                save(job_path, execution)
                raise RuntimeError(execution["error"])
            execution["status"] = "checking_artifacts"
            save(job_path, execution)
            check_sources(plan)
            directory = Path(next(iter(directories)))
            config = load_yaml(directory / "run_config.yaml")
            if comparison_protocol(config) != job["protocol"] or config["defense"] != job["defense"]:
                raise RuntimeError("Actual run differs from frozen configuration.")
            manifest, sha = read_manifest(directory / "confirmation_split.json", MANIFEST_SHA)
            mapping = json.loads((directory / "data_partition.json").read_text())
            validate_mapping(mapping, manifest)
            if mapping["seed"] != job["seed"] or mapping["manifest_sha256"] != sha:
                raise RuntimeError("Unexpected original data mapping.")
            for relative in ("performance_summary.json", "risk_synthesis/synthesis_summary.json"):
                if json.loads((directory / relative).read_text())["status"] != "completed":
                    raise RuntimeError(f"Incomplete {relative}")
            summary = json.loads((directory / "privacy_audit/summary.json").read_text())
            if summary.get("errors") or {a["attack"] for a in summary.get("attacks", [])} != set(config["audit"]["attacks"]):
                raise RuntimeError("Incomplete attack set.")
            if job["arm"] != "baseline":
                from scripts.verify_synthesis_history import verify
                validation = verify(directory / "risk_synthesis")
                save(study / f'{job["id"]}_mechanism.json', validation, exclusive=True)
            execution.update(status="completed", finished_at_utc=now())
            save(job_path, execution)
            active_execution = None
            print(f'Completed {job["id"]}: {directory}', flush=True)
    except BaseException as error:
        if active_execution is not None:
            job_path, execution = active_execution
            execution.update(status="failed", error=repr(error))
            save(job_path, execution)
        state.update(status="failed", error=repr(error))
        save(path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status"))
    parser.add_argument("--study-dir", type=Path, default=STUDY)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--wait-for-gpu", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "prepare":
        prepare(args.study_dir.resolve())
    elif args.action == "run":
        worker(args.study_dir.resolve(), args.gpu, args.wait_for_gpu)
    else:
        plan = json.loads((args.study_dir / "plan.json").read_text())
        for job in plan["jobs"]:
            path = args.study_dir / f'{job["id"]}.json'
            state = json.loads(path.read_text()) if path.exists() else {"status": "unclaimed"}
            print(json.dumps(dict(job=job["id"], status=state["status"], gpu=state.get("gpu"),
                                 result_directories=state.get("result_directories"))))
        for path in args.study_dir.glob("worker_gpu*.json"):
            print(json.dumps(json.loads(path.read_text())))


if __name__ == "__main__":
    main()
