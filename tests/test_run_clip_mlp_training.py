from __future__ import annotations

from pathlib import Path
import shlex
import subprocess


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_clip_mlp_train_only.sh"
)


def _dry_run(*arguments: str) -> list[str]:
    completed = subprocess.run(
        ["bash", str(SCRIPT), *arguments, "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    line = next(
        item for item in completed.stdout.splitlines() if item.startswith("COMMAND ")
    )
    return shlex.split(line.removeprefix("COMMAND "))


def test_clip_mlp_training_script_has_complete_plain_training_defaults():
    command = _dry_run()

    assert command[1].endswith("main.py")
    assert command[command.index("--config") + 1] == "configs/clip_mlp_fedavg.yaml"
    assert command[command.index("--learning_rate") + 1] == "0.1"
    assert command[command.index("--dataset_name") + 1] == "caltech101"
    assert command[command.index("--num_global_iters") + 1] == "100"
    assert command[command.index("--local_epochs") + 1] == "2"
    assert command[command.index("--dirichlet_alpha") + 1] == "0.1"
    assert command[command.index("--gpu") + 1] == "0"
    assert command[command.index("--model_type") + 1] == "clip_mlp"
    assert command[command.index("--attack") + 1] == "none"
    assert command[command.index("--defense") + 1] == "none"
