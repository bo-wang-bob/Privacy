from __future__ import annotations

import csv
import json
from pathlib import Path

import analysis_scripts.run_fedmia_complex_sweep as sweep
from analysis_scripts.run_fedmia_complex_sweep import (
    build_jobs,
    filter_jobs_by_dataset,
    filter_jobs_by_method,
    summarize,
)
from main import validate_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_sequential_launcher_selects_gpu_with_most_free_memory(monkeypatch):
    statuses = [
        sweep.GPUStatus(index=0, free_memory_mb=9000, utilization_percent=5),
        sweep.GPUStatus(index=1, free_memory_mb=15000, utilization_percent=30),
    ]
    monkeypatch.setattr(sweep, "_query_gpu_status", lambda _candidates: statuses)
    selected = sweep._wait_for_best_gpu([0, 1], minimum_free_memory_mb=7000)
    assert selected.index == 1


def test_parallel_gpu_selection_keeps_all_eligible_devices(monkeypatch):
    statuses = {
        0: sweep.GPUStatus(index=0, free_memory_mb=9000, utilization_percent=5),
        1: sweep.GPUStatus(index=1, free_memory_mb=15000, utilization_percent=30),
    }
    monkeypatch.setattr(
        sweep,
        "_query_gpu_status",
        lambda candidates: [statuses[gpu] for gpu in candidates],
    )
    selected = sweep._best_available_gpu([0, 1], minimum_free_memory_mb=7000)
    assert selected is not None and selected.index == 1


def test_parallel_scheduler_allows_multiple_jobs_on_one_gpu(tmp_path, monkeypatch):
    spec_path = REPOSITORY_ROOT / "configs" / "fedmia_prompt_methods_sweep.yaml"
    import yaml

    with spec_path.open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    jobs, _ = build_jobs(spec, spec_path)
    jobs = jobs[:3]
    launched: set[str] = set()
    launch_gpus: list[int] = []

    class FinishedProcess:
        returncode = 0

        @staticmethod
        def poll():
            return 0

    class LogFile:
        @staticmethod
        def close():
            return None

    def launch(job, gpu, _logs_root):
        launched.add(job.run_id)
        launch_gpus.append(gpu)
        return sweep.ActiveRun(job, gpu, FinishedProcess(), LogFile())

    monkeypatch.setattr(sweep, "_launch", launch)
    monkeypatch.setattr(
        sweep,
        "_completed_result",
        lambda job: tmp_path / job.run_id if job.run_id in launched else None,
    )
    monkeypatch.setattr(
        sweep,
        "_best_available_gpu",
        lambda candidates, _minimum: sweep.GPUStatus(
            index=candidates[0],
            free_memory_mb=10000,
            utilization_percent=0,
        ),
    )
    monkeypatch.setattr(sweep, "summarize", lambda _jobs, _root: (3, 6))

    result = sweep.run_sweep(
        jobs,
        tmp_path,
        gpus=[0],
        force=False,
        minimum_free_memory_mb=7000,
        max_parallel_jobs=3,
    )
    assert result == 0
    assert launch_gpus == [0, 0, 0]


def test_complex_fedmia_spec_expands_stable_seventy_eight_run_grid():
    spec_path = REPOSITORY_ROOT / "configs" / "fedmia_complex_sweep.yaml"
    import yaml

    with spec_path.open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    jobs, results_root = build_jobs(spec, spec_path)

    assert len(jobs) == 78
    assert len({job.run_id for job in jobs}) == 78
    assert jobs[0].run_id == "none_seed42_target0_4cfe584014"
    assert results_root == REPOSITORY_ROOT / "results" / "fedmia_complex_tpr1"
    assert {job.seed for job in jobs} == {42, 1337, 2027}
    assert {job.target_client_id for job in jobs} == {0}
    assert sum(job.defense == "none" for job in jobs) == 3
    assert sum(job.defense == "data_aug_sampling" for job in jobs) == 12
    assert sum(job.defense == "prompt_dp" for job in jobs) == 9
    assert sum(job.defense == "hamp" for job in jobs) == 9
    for job in jobs:
        assert job.config["aggregator"] == "fedavg"
        assert job.config["num_global_iters"] == 20
        assert job.config["local_epochs"] == 2
        assert job.config["audit"]["max_member_samples"] == 128
        assert job.config["audit"]["max_nonmember_samples"] == 2048
        assert "nasr_active" not in job.config["audit"]["attacks"]
        assert job.config["results_dir"] == str(job.run_root)


def test_pathological_full_spec_expands_all_dataset_specific_schedules():
    spec_path = REPOSITORY_ROOT / "configs" / "fedmia_pathological_full_sweep.yaml"
    import yaml

    with spec_path.open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    jobs, results_root = build_jobs(spec, spec_path)

    assert len(jobs) == 390
    assert len({job.run_id for job in jobs}) == 390
    assert results_root == REPOSITORY_ROOT / "results" / "fedmia_pathological_full"
    assert {job.dataset for job in jobs} == {
        "caltech101",
        "oxfordpets",
        "flowers",
        "food101",
        "cifar100",
    }
    assert all(job.config["partition_mode"] == "pathological" for job in jobs)
    assert all(job.config["use_full_dataset"] is True for job in jobs)
    assert all(job.config["fpl_shots"] is None for job in jobs)
    assert all(job.config["local_epochs"] == 1 for job in jobs)
    assert all(job.config["learning_rate"] == 0.0001 for job in jobs)
    for job in jobs:
        if job.dataset == "cifar100":
            assert job.config["total_users"] == 50
            assert job.config["sample_users"] == 10
            assert job.config["num_global_iters"] == 400
        else:
            assert job.config["total_users"] == 10
            assert job.config["sample_users"] == 10
            assert job.config["num_global_iters"] == 100

    food_jobs = filter_jobs_by_dataset(jobs, "food101")
    assert len(food_jobs) == 78
    assert {job.dataset for job in food_jobs} == {"food101"}


def test_prompt_method_fedmia_spec_expands_three_methods_and_two_attacks():
    spec_path = REPOSITORY_ROOT / "configs" / "fedmia_prompt_methods_sweep.yaml"
    import yaml

    with spec_path.open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    jobs, results_root = build_jobs(spec, spec_path)

    assert len(jobs) == 45
    assert len({job.run_id for job in jobs}) == 45
    assert results_root == REPOSITORY_ROOT / "results" / "fedmia_prompt_methods"
    assert spec["jobs"] == 1
    assert spec["gpus"] == [0]
    assert {job.method for job in jobs} == {"promptfl", "fedotp", "fedpgp"}
    assert all(
        job.config["audit"]["attacks"] == ["fedmia_loss", "fedmia_cosine"]
        for job in jobs
    )
    assert all(job.config["defense"]["name"] == "none" for job in jobs)
    assert all(job.method in job.run_id for job in jobs)
    assert len(filter_jobs_by_method(jobs, "fedotp")) == 15
    assert len(filter_jobs_by_dataset(jobs, "cifar100")) == 9
    for job in jobs:
        validate_config(job.config)
        if job.dataset == "cifar100":
            assert (job.config["total_users"], job.config["sample_users"]) == (50, 10)
            assert job.config["num_global_iters"] == 400
            assert job.config["audit"]["audit_client_ids"] == list(range(10))
        else:
            assert (job.config["total_users"], job.config["sample_users"]) == (10, 10)
            assert job.config["num_global_iters"] == 100
            assert job.config["audit"]["audit_client_ids"] == "all"


def test_sweep_summary_uses_tpr_at_one_percent_as_primary_table(tmp_path: Path):
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "name": "test_sweep",
        "base_config": str(
            REPOSITORY_ROOT / "configs" / "fedmia_prompt_benchmark.yaml"
        ),
        "results_root": str(tmp_path / "results"),
        "seeds": [1, 2],
        "target_client_ids": [0],
        "common": {"audit": {"attacks": ["blackbox_loss"]}},
        "defenses": [{"name": "none"}],
    }
    jobs, results_root = build_jobs(spec, spec_path)
    for index, job in enumerate(jobs):
        result_dir = job.run_root / f"completed_{index}"
        (result_dir / "privacy_audit").mkdir(parents=True)
        with (result_dir / "training_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(("round", "loss", "accuracy", "samples"))
            writer.writerow((19, 1.0, 0.7 + index * 0.1, 100))
        payload = {
            "attacks": [
                {
                    "attack": "blackbox_loss",
                    "tpr_at_fpr_0.1": 0.3 + index * 0.1,
                    "tpr_at_fpr_0.01": 0.1 + index * 0.1,
                    "tpr_at_fpr_0.001": 0.01 + index * 0.01,
                    "auc": 0.6 + index * 0.1,
                    "num_samples": 200,
                }
            ]
        }
        with (result_dir / "privacy_audit" / "summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(payload, file)

    complete_runs, attack_rows = summarize(jobs, results_root)
    assert (complete_runs, attack_rows) == (2, 2)
    with (results_root / "summary_aggregate.csv").open(
        "r", encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert int(rows[0]["runs"]) == 2
    assert float(rows[0]["accuracy_mean"]) == 0.75
    assert abs(float(rows[0]["tpr_at_fpr_0.1_mean"]) - 0.35) < 1e-12
    assert abs(float(rows[0]["tpr_at_fpr_0.01_mean"]) - 0.15) < 1e-12
    assert (results_root / "summary_tpr_matrix.csv").is_file()
    assert (results_root / "summary_privacy_metrics.csv").is_file()
    with (results_root / "summary_privacy_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as file:
        privacy_rows = list(csv.DictReader(file))
    assert float(privacy_rows[0]["tpr_pct_at_fpr_1pct_mean"]) == 15.0
    assert (results_root / "privacy_utility_pareto.csv").is_file()
