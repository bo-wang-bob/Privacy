from __future__ import annotations

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
    assert "blackbox_loss, loss_series, grad_cosine, avg_cosine" in output
    for dataset in ("caltech101", "oxfordpets", "flowers", "food101", "cifar100"):
        assert dataset in output


def test_visual_adapter_sweep_builds_valid_fpl_16shot_configs():
    with SPEC.open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    jobs, results_root = sweep.build_jobs(spec, SPEC)

    assert len(jobs) == 5
    assert results_root == REPOSITORY_ROOT / "results" / "visual_adapter_fedmia_attacks"
    for job in jobs:
        assert job.config["model_type"] == "visual_adapter"
        assert job.config["aggregator"] == "fedavg"
        assert job.config["fpl_shots"] == 16
        assert job.config["use_full_dataset"] is False
        assert job.config["audit"]["candidate_sampling"] == "low_fpr_full"
        assert job.config["batch_size"] == 128
        assert job.config["eval_batch_size"] == 512
        assert job.config["eval_interval"] == 5
        assert job.config["visual_adapter"]["precompute_batch_size"] == 64
        assert job.config["audit"]["audit_batch_size"] == 512
        assert job.config["audit"]["audit_interval"] == 5
        validate_config(job.config)


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
