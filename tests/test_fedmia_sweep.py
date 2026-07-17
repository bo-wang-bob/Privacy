from __future__ import annotations

import csv
import json
from pathlib import Path

import analysis_scripts.run_fedmia_complex_sweep as sweep
from analysis_scripts.run_fedmia_complex_sweep import build_jobs, summarize


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_sequential_launcher_selects_gpu_with_most_free_memory(monkeypatch):
    statuses = [
        sweep.GPUStatus(index=0, free_memory_mb=9000, utilization_percent=5),
        sweep.GPUStatus(index=1, free_memory_mb=15000, utilization_percent=30),
    ]
    monkeypatch.setattr(sweep, "_query_gpu_status", lambda _candidates: statuses)
    selected = sweep._wait_for_best_gpu([0, 1], minimum_free_memory_mb=7000)
    assert selected.index == 1


def test_complex_fedmia_spec_expands_stable_seventy_eight_run_grid():
    spec_path = REPOSITORY_ROOT / "configs" / "fedmia_complex_sweep.yaml"
    import yaml

    with spec_path.open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    jobs, results_root = build_jobs(spec, spec_path)

    assert len(jobs) == 78
    assert len({job.run_id for job in jobs}) == 78
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
    assert (results_root / "privacy_utility_pareto.csv").is_file()
