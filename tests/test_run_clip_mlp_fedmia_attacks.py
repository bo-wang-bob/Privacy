from __future__ import annotations

from pathlib import Path
import datetime
import shlex
import subprocess
import sys

from scripts import run_clip_mlp_fedmia_sweep as sweep


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_clip_mlp_fedmia_attacks.sh"
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


def test_mlp_attack_sweep_has_five_datasets_and_two_phases_per_dataset():
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
    for dataset in ("caltech101", "oxfordpets", "flowers", "food101", "cifar100"):
        assert f"dataset" in output and dataset in output
        assert any(dataset in command[command.index("--config") + 1] for command in commands)
    assert "clip_mlp.precompute_features" in output
    assert "data.partition_mode" in output and ": iid" in output
    assert "optimization.batch_size" in output and ": 128" in output
    assert "optimization.learning_rate" in output and ": 0.001" in output
    assert "optimization.eval_batch_size" in output and ": 512" in output
    assert "optimization.eval_interval" in output and ": 5" in output
    assert "clip_mlp.precompute_batch_size" in output and ": 64" in output
    assert "privacy_audit.audit_interval" in output
    assert "privacy_audit.audit_batch_size" in output
    assert "privacy_audit.low_fpr_max_members" in output and ": 5000" in output
    assert (
        "privacy_audit.low_fpr_max_nonmembers" in output
        and ": 20000" in output
    )
    assert "blackbox_loss, loss_series, grad_cosine, avg_cosine" in output


def test_mlp_attack_sweep_supports_filters_overrides_and_prints_parameters():
    output, commands = _dry_run(
        "--datasets",
        "flowers",
        "--attacks",
        "fedmia_loss,fedmia_cosine",
        "--target-client",
        "3",
        "--projres-threshold",
        "0.001",
        "--rounds",
        "3",
    )

    assert len(commands) == 2
    assert "dataset" in output and "flowers" in output
    assert "federated.num_global_iters" in output and ": 3" in output
    assert "fedmia_loss, fedmia_cosine" in output
    projres = commands[1]
    assert projres[projres.index("--target-client") + 1] == "3"
    assert projres[projres.index("--threshold") + 1] == "0.001"


def test_mlp_attack_sweep_can_skip_the_standalone_projres_phase():
    output, commands = _dry_run("--datasets", "caltech101", "--skip-projres")

    assert len(commands) == 1
    assert commands[0][1].endswith("main.py")
    assert "STRICT PROJRES" not in output


def test_normal_run_prints_parameters_only_when_each_job_starts(
    tmp_path, monkeypatch, capsys
):
    def job(name: str) -> sweep.SweepJob:
        root = tmp_path / name
        return sweep.SweepJob(
            run_id=f"{name}_run",
            config={
                "dataset_name": name,
                "audit": {"attacks": ["fedmia_loss"]},
                "clip_mlp": {"hidden_dim": 8},
            },
            dataset=name,
            method="fedavg",
            seed=42,
            target_client_id=0,
            defense="none",
            run_root=root,
            config_path=root / "run_config.yaml",
        )

    monkeypatch.setattr(sweep, "_completed_result", lambda _job: tmp_path)
    first = job("first_dataset")
    second = job("second_dataset")

    sweep._run_job(first, 0, {"enabled": False}, True, False)
    first_output = capsys.readouterr().out
    assert "model=clip_mlp | dataset=first_dataset" in first_output
    assert "run=first_dataset_run | phase=all | gpu=0" in first_output
    assert "first_dataset" in first_output
    assert "second_dataset" not in first_output

    sweep._run_job(second, 1, {"enabled": False}, True, False)
    second_output = capsys.readouterr().out
    assert "model=clip_mlp | dataset=second_dataset" in second_output
    assert "run=second_dataset_run | phase=all | gpu=1" in second_output
    assert "second_dataset" in second_output
    assert "first_dataset" not in second_output


def test_single_job_log_runner_tees_child_output_to_terminal_and_file(
    tmp_path, capsys
):
    log_path = tmp_path / "run.log"
    return_code = sweep._run_logged(
        [sys.executable, "-c", "print('live training progress')"],
        log_path,
        stream_output=True,
        context="model=clip_mlp | dataset=flowers | run=test | phase=train | gpu=0",
    )

    assert return_code == 0
    assert "live training progress" in capsys.readouterr().out
    log_text = log_path.read_text(encoding="utf-8")
    assert "| model=clip_mlp | dataset=flowers | run=test | phase=train | gpu=0 | COMMAND " in log_text
    assert "live training progress" in log_text


def test_scoped_log_line_reuses_child_timestamp_and_adds_experiment_identity():
    line = sweep._format_scoped_log_line(
        "2026-08-05 12:34:56,789 INFO trainer: round=1\n",
        "model=visual_adapter | dataset=food101 | run=abc | phase=train | gpu=1",
    )

    assert line == (
        "2026-08-05 12:34:56,789 | model=visual_adapter | dataset=food101 | "
        "run=abc | phase=train | gpu=1 | INFO trainer: round=1\n"
    )


def test_scoped_log_line_timestamps_plain_child_output():
    line = sweep._format_scoped_log_line(
        "plain progress\n",
        "model=clip_mlp | dataset=cifar100 | run=xyz | phase=projres | gpu=0",
        now=datetime.datetime(2026, 8, 5, 12, 34, 56, 123000),
    )

    assert line == (
        "2026-08-05 12:34:56,123 | model=clip_mlp | dataset=cifar100 | "
        "run=xyz | phase=projres | gpu=0 | plain progress\n"
    )
