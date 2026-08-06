from __future__ import annotations

import datetime
from pathlib import Path
import shlex
import subprocess

import yaml

from main import validate_config
from scripts import run_clip_mlp_fedmia_sweep as sweep


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "run_visual_adapter_fedmia_attacks.sh"
SPEC = REPOSITORY_ROOT / "configs" / "visual_adapter_fedmia_attacks_sweep.yaml"


def _dry_run(*arguments: str) -> tuple[str, list[list[str]]]:
    completed = subprocess.run(
        ["bash", str(SCRIPT), *arguments, "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [
        shlex.split(line.removeprefix("COMMAND "))
        for line in completed.stdout.splitlines()
        if line.startswith("COMMAND ")
    ]
    return completed.stdout, commands


def test_visual_adapter_attack_sweep_has_five_16shot_jobs_with_projres():
    output, commands = _dry_run()

    assert "Expanded 5 jobs" in output
    assert len(commands) == 10
    main_commands = [command for command in commands if command[1].endswith("main.py")]
    projres_commands = [
        command
        for command in commands
        if command[1].endswith("validate_projres_mlp_real.py")
    ]
    assert len(main_commands) == len(projres_commands) == 5
    assert "STRICT PROJRES" in output
    assert "adapter.net.0.weight" in output
    assert "model_type" in output and "visual_adapter" in output
    assert "visual_adapter.precompute_features" in output
    assert "data.partition_mode" in output and ": iid" in output
    assert "privacy_audit.low_fpr_max_members" in output and ": 5000" in output
    assert (
        "privacy_audit.low_fpr_max_nonmembers" in output
        and ": 20000" in output
    )
    assert "blackbox_loss, loss_series, grad_cosine, avg_cosine" in output
    for dataset in ("caltech101", "oxfordpets", "flowers", "food101", "cifar100"):
        assert dataset in output


def test_visual_adapter_sweep_builds_valid_fpl_16shot_configs():
    with SPEC.open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    started_at = datetime.datetime(2026, 8, 5, 14, 30, 52, 123456)
    jobs, results_root = sweep.build_jobs(spec, SPEC, started_at=started_at)

    assert len(jobs) == 5
    assert results_root == REPOSITORY_ROOT / "results"
    for job in jobs:
        assert job.run_id.startswith(
            "2026-08-05_14-30-52-123456_visual_adapter_"
        )
        assert f"_{job.dataset}_fedavg_seed42_target0_" in job.run_id
        assert job.run_root.name == job.run_id
        assert job.run_root.parent == REPOSITORY_ROOT / "results"
        assert job.config["sweep_name"] == "visual_adapter_fedmia_attacks"
        assert job.config["model_type"] == "visual_adapter"
        assert job.config["aggregator"] == "fedavg"
        assert job.config["fpl_shots"] == 16
        assert job.config["use_full_dataset"] is False
        assert job.config["partition_mode"] == "iid"
        assert job.config["audit"]["candidate_sampling"] == "low_fpr_full"
        assert job.config["audit"]["low_fpr_max_members"] == 5000
        assert job.config["audit"]["low_fpr_max_nonmembers"] == 20000
        assert job.config["batch_size"] == 128
        assert job.config["eval_batch_size"] == 512
        assert job.config["eval_interval"] == 5
        assert job.config["visual_adapter"]["precompute_batch_size"] == 64
        assert job.config["audit"]["audit_batch_size"] == 512
        assert job.config["audit"]["audit_interval"] == 5
        validate_config(job.config)


def test_summarize_only_can_rediscover_timestamped_training_tasks(tmp_path):
    run_root = (
        tmp_path
        / "2026-08-05_14-30-52-123456_clip_mlp_flowers_fedavg_seed7_target2_deadbeef00"
    )
    run_root.mkdir(parents=True)
    config = {
        "model_type": "clip_mlp",
        "dataset_name": "flowers",
        "aggregator": "fedavg",
        "seed": 7,
        "defense": {"name": "none"},
        "audit": {"target_client_id": 2, "attacks": ["fedmia_loss"]},
    }
    with (run_root / "run_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    jobs = sweep.discover_existing_jobs(tmp_path)

    assert len(jobs) == 1
    assert jobs[0].run_root == run_root
    assert jobs[0].dataset == "flowers"
    assert jobs[0].seed == 7
    assert jobs[0].target_client_id == 2


def test_summarize_only_can_rediscover_legacy_grouped_tasks(tmp_path):
    run_root = tmp_path / "clip_mlp_fedmia_attacks" / "runs" / "legacy_run"
    run_root.mkdir(parents=True)
    config = {
        "model_type": "clip_mlp",
        "dataset_name": "caltech101",
        "aggregator": "fedavg",
        "seed": 42,
        "defense": {"name": "none"},
        "audit": {"target_client_id": 0, "attacks": ["fedmia_loss"]},
    }
    with (run_root / "run_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    jobs = sweep.discover_existing_jobs(tmp_path, "clip_mlp_fedmia_attacks")

    assert len(jobs) == 1
    assert jobs[0].run_root == run_root


def test_visual_adapter_attack_sweep_supports_common_overrides():
    output, commands = _dry_run(
        "--datasets",
        "cifar100",
        "--attacks",
        "fedmia_loss,fedmia_cosine",
        "--target-client",
        "3",
        "--rounds",
        "2",
    )

    assert len(commands) == 2
    assert "dataset" in output and "cifar100" in output
    assert "federated.num_global_iters" in output and ": 2" in output
    assert "privacy_audit.target_client_id" in output and ": 3" in output
    assert "fedmia_loss, fedmia_cosine" in output
