"""Freeze and execute the bounded full-replacement Adapter effect comparison."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import comparison_protocol, digest
from scripts.run_privacy_experiments import build_tasks, load_yaml, parse_args
from utils.confirmation_split import read_manifest, validate_mapping

PLAN = ROOT / "analysis_scripts/risk_synthesis_all_study_plan_20260912.json"
PYTHON = "/root/.local/share/mamba/envs/pfedba/bin/python"
MANIFEST = ROOT / "analysis_scripts/risk_synthesis_confirmation_data_20260912/split.json"
MANIFEST_SHA = "00941791be9727022bb09c4b9d55f8b06d05a1cc3489a99753cbdb393151c178"
CONTROLS = {
    "none": "2026-09-12_05-13-17-869543_clip_adapter_cifar100_fedavg_none_seed43_target0_027f42ccf1",
    "legacy_partial_risk": "2026-09-12_05-44-18-841952_clip_adapter_cifar100_fedavg_risk_synthesis_seed43_target0_9e3d39dbbd",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    import main as entry
    import yaml
    read_manifest(MANIFEST, MANIFEST_SHA)
    catalog = load_yaml(ROOT / "configs/experiment_catalog.yaml")
    controls = {}
    for arm, name in CONTROLS.items():
        directory = ROOT / "results" / name
        config = load_yaml(directory / "run_config.yaml")
        assert json.loads((directory / "performance_summary.json").read_text())["status"] == "completed"
        files = [directory / relative for relative in (
            "run_config.yaml", "training_metrics.csv", "performance_summary.json", "confirmation_split.json",
            "data_partition.json", "privacy_audit/summary.json", "privacy_audit/predictions.csv",
            "privacy_audit/signals.pt", "privacy_audit/candidate_selection.pt",
            "privacy_audit/client_train_update_candidate_selection.pt")]
        controls[arm] = dict(directory=str(directory), protocol=comparison_protocol(config),
                             defense=config["defense"], sources={str(p): digest(p) for p in files})
    assert controls["none"]["protocol"] == controls["legacy_partial_risk"]["protocol"]
    jobs = []
    for arm, gpu in (("risk", "1"), ("shuffled_risk", "0")):
        args = ["--models", "clip_adapter", "--datasets", "cifar100", "--methods", "fedavg",
                "--defenses", "risk_synthesis", "--attacks", "all", "--seeds", "43",
                "--rounds", "100", "--gpus", gpu,
                "--set", f"confirmation_split_manifest={MANIFEST}",
                "--set", f"confirmation_split_sha256={MANIFEST_SHA}",
                "--set", f"defense.synthesis.mode={arm}"]
        tasks, skipped = build_tasks(catalog, parse_args(args))
        assert len(tasks) == 1 and not skipped
        with tempfile.TemporaryDirectory(prefix="synthesis-all-study-") as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(yaml.safe_dump(tasks[0].config))
            old_argv = sys.argv
            try:
                sys.argv = ["main.py", "--config", str(path)]
                config = entry.parse_args()
            finally:
                sys.argv = old_argv
        entry.normalize_clip_adapter_config(config)
        entry.validate_config(config)
        config.setdefault(config["aggregator"], {}).setdefault("seed", config["seed"])
        protocol = comparison_protocol(config)
        if protocol != controls["none"]["protocol"]:
            differing = [k for k in protocol.keys() | controls["none"]["protocol"].keys()
                         if protocol.get(k) != controls["none"]["protocol"].get(k)]
            raise ValueError(f"Unmatched baseline protocol: {differing}")
        synthesis = config["defense"]["synthesis"]
        assert synthesis["replacement_policy"] == "all" and synthesis["warmup_rounds"] == 0
        assert synthesis["replacement_fraction"] == 1 and synthesis["center_weighting"] == "uniform"
        assert len(config["audit"]["attacks"]) == 11 and config["audit"]["audit_client_ids"] == [0]
        jobs.append(dict(arm=arm, gpu=gpu, protocol=protocol, defense=config["defense"],
                         command=[PYTHON, "scripts/run_privacy_experiments.py", *args], status="prepared"))
    old_plan = json.loads((ROOT / "analysis_scripts/risk_synthesis_confirmation_plan_20260912.json").read_text())
    changed = [p for p, h in old_plan["source_hashes"].items() if digest(ROOT / p) != h]
    allowed = {"configs/experiment_catalog.yaml", "privacy_defenses/risk_synthesis.py",
               "scripts/run_privacy_experiments.py"}
    if set(changed) != allowed:
        raise ValueError(f"Unexpected source drift from matched baseline: {changed}")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    roots = {"privacy_defenses", "privacy_attacks", "trainmodel", "utils", "users", "servers", "aggregator", "configs"}
    paths = [p for p in tracked if (Path(p).parts[0] in roots and Path(p).suffix in {".py", ".yaml"})
             or p in {"main.py", "scripts/run_privacy_experiments.py", "scripts/run_fedllm_adapter.py"}]
    paths += [str(Path(__file__).relative_to(ROOT))]
    plan = dict(schema_version=1, prepared_at_utc=now(), jobs=jobs, controls=controls,
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        confirmation_manifest=str(MANIFEST), confirmation_manifest_sha256=MANIFEST_SHA,
        scope="Exploratory 100-round full-replacement Adapter/CIFAR100, seed43, client0. "
              "Reuse exactly matched NONE and historical partial replacement; two fresh risk/shuffle runs. "
              "This source partition was already assessed; this is not an untouched-data confirmation.",
        baseline_source_changes=changed,
        baseline_reuse_reason="Complete non-defense parsed configurations match; shared trainers, model, "
                              "optimizer, aggregation, data and attack implementations are unchanged. "
                              "Changed sources implement new defense defaults, full replacement and dry-run display.",
        acceptance=dict(maximum_all_attack_auc_delta_at_most=-0.02,
                        maximum_tpr_at_1pct_delta_strictly_below=0, accuracy_delta_at_least=-0.02),
        interpretation="Report all 11 attacks, direction and class-conditional diagnostics, original source "
                       "identity and exact-duplicate sensitivity. Risk versus shuffle tests ordering separately. "
                       "Comparison to historical partial policy is a composite protocol change. No formal DP.",
        source_hashes={p: digest(ROOT / p) for p in paths},
        environment={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    with PLAN.open("x") as handle:
        json.dump(plan, handle, indent=2)
    print(json.dumps(dict(plan=str(PLAN), sha256=digest(PLAN), source_files=len(paths),
                         arms=[j["arm"] for j in jobs], baseline_protocol_match=True), indent=2))


def run_arm(arm):
    plan = json.loads(PLAN.read_text())
    fingerprint = digest(PLAN)
    job = next(j for j in plan["jobs"] if j["arm"] == arm)
    state_path = ROOT / f"analysis_scripts/risk_synthesis_all_study_{arm}_execution_20260912.json"
    state = dict(status="prepared", pid=os.getpid(), plan=str(PLAN), plan_sha256=fingerprint,
                 started_at_utc=now(), job=job)
    with state_path.open("x") as handle:
        json.dump(state, handle, indent=2)

    def save():
        state["updated_at_utc"] = now()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)

    def check_sources():
        if digest(PLAN) != fingerprint:
            raise RuntimeError("Frozen study plan changed.")
        for p, h in plan["source_hashes"].items():
            if digest(ROOT / p) != h:
                raise RuntimeError(f"Frozen source changed: {p}")
        for control in plan["controls"].values():
            for p, h in control["sources"].items():
                if digest(Path(p)) != h:
                    raise RuntimeError(f"Reused control artifact changed: {p}")
        read_manifest(plan["confirmation_manifest"], plan["confirmation_manifest_sha256"])

    try:
        check_sources()
        if shutil.disk_usage(ROOT).free < 10 * 1024**3:
            raise RuntimeError("Less than 10 GiB free; preserving existing results.")
        state["status"] = job["status"] = "running"
        save()
        process = subprocess.Popen(job["command"], cwd=ROOT,
            env={**os.environ, **plan["environment"], "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        job["runner_pid"] = process.pid
        save()
        directories = set()
        for line in process.stdout:
            print(line, end="", flush=True)
            match = re.match(r"\s*run_dir=(.+)\s*$", line)
            if match:
                directories.add(match.group(1).strip())
                job["result_directories"] = sorted(directories)
                save()
        job["returncode"] = process.wait()
        state["status"] = "validating_artifacts"
        save()
        if job["returncode"] or len(directories) != 1:
            raise RuntimeError("Training failed or result directory is ambiguous.")
        check_sources()
        directory = Path(next(iter(directories)))
        config = load_yaml(directory / "run_config.yaml")
        if comparison_protocol(config) != job["protocol"] or config["defense"] != job["defense"]:
            raise RuntimeError("Actual protocol differs from frozen study.")
        manifest, fingerprint = read_manifest(directory / "confirmation_split.json", MANIFEST_SHA)
        mapping = json.loads((directory / "data_partition.json").read_text())
        if mapping["seed"] != 43 or mapping["manifest_sha256"] != fingerprint:
            raise RuntimeError("Original source mapping differs from frozen study.")
        validate_mapping(mapping, manifest)
        for relative in ("performance_summary.json", "risk_synthesis/synthesis_summary.json"):
            if json.loads((directory / relative).read_text())["status"] != "completed":
                raise RuntimeError(f"Incomplete result: {relative}")
        for relative in ("summary.json", "predictions.csv", "signals.pt", "candidate_selection.pt",
                         "client_train_update_candidate_selection.pt"):
            if not (directory / "privacy_audit" / relative).is_file():
                raise RuntimeError(f"Missing attack artifact: {relative}")
        state["status"] = job["status"] = "completed"
        state["finished_at_utc"] = now()
        save()
    except BaseException as error:
        state["status"] = job["status"] = "failed"
        state["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "risk", "shuffled_risk"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run_arm(args.action)
