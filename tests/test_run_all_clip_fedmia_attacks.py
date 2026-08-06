from __future__ import annotations

from pathlib import Path
import re
import shlex
import subprocess


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_all_clip_fedmia_attacks.sh"
)


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


def test_ultimate_sweep_expands_both_models_all_datasets_and_mlp_projres():
    output, commands = _dry_run()

    assert output.count("Expanded 5 jobs") == 2
    assert "model=clip_mlp" in output
    assert "model=visual_adapter" in output
    assert output.count("optimization.learning_rate") == 10
    assert len(re.findall(r"optimization\.learning_rate\s+: 0\.1$", output, re.M)) == 5
    assert len(re.findall(r"optimization\.learning_rate\s+: 0\.01$", output, re.M)) == 5
    assert len(re.findall(r"data\.partition_mode\s+: iid$", output, re.M)) == 10
    assert output.count("privacy_audit.low_fpr_max_members") == 10
    assert output.count("privacy_audit.low_fpr_max_nonmembers") == 10
    assert output.count("privacy_audit.audit_interval") == 10
    assert output.count("privacy_audit.attack_audit_intervals.loss_series") == 10
    assert output.count("privacy_audit.attack_audit_intervals.avg_cosine") == 10
    assert output.count("privacy_audit.attack_audit_intervals.fedmia_loss") == 10
    assert output.count("privacy_audit.attack_audit_intervals.fedmia_cosine") == 10
    assert len(commands) == 20
    main_commands = [command for command in commands if command[1].endswith("main.py")]
    projres_commands = [
        command
        for command in commands
        if command[1].endswith("validate_projres_mlp_real.py")
    ]
    assert len(main_commands) == 10
    assert len(projres_commands) == 10


def test_ultimate_sweep_filters_model_dataset_attacks_and_learning_rate():
    output, commands = _dry_run(
        "--models",
        "adapter",
        "--datasets",
        "food101",
        "--attacks",
        "fedmia_loss,fedmia_cosine",
        "--learning-rate",
        "0.0005",
    )

    assert "model=visual_adapter" in output
    assert "model=clip_mlp" not in output
    assert "optimization.learning_rate" in output and ": 0.0005" in output
    assert "fedmia_loss, fedmia_cosine" in output
    assert len(commands) == 2
    assert commands[0][1].endswith("main.py")
    assert commands[1][1].endswith("validate_projres_mlp_real.py")
    assert "STRICT PROJRES" in output


def test_ultimate_sweep_rejects_unknown_model():
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--models", "unknown", "--dry-run"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Unknown model" in completed.stderr
