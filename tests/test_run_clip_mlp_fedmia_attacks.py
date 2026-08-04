from __future__ import annotations

from pathlib import Path
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
    assert "STARTING JOB | first_dataset_run | gpu:0" in first_output
    assert "first_dataset" in first_output
    assert "second_dataset" not in first_output

    sweep._run_job(second, 1, {"enabled": False}, True, False)
    second_output = capsys.readouterr().out
    assert "STARTING JOB | second_dataset_run | gpu:1" in second_output
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
    )

    assert return_code == 0
    assert "live training progress" in capsys.readouterr().out
    log_text = log_path.read_text(encoding="utf-8")
    assert log_text.startswith("COMMAND ")
    assert "live training progress" in log_text
