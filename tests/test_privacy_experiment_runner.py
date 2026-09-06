import datetime as dt
from pathlib import Path

from scripts.run_privacy_experiments import (
    _forward_child_line,
    _task_header,
    _timestamped_line,
    build_tasks,
    load_yaml,
    parse_args,
    resolve_model_config,
    run_task,
    validate_resolved_config,
)
from servers.serverbase import _format_round_progress


CATALOG = load_yaml("configs/experiment_catalog.yaml")


def _args(*values: str):
    args = parse_args(list(values))
    args.started_at = dt.datetime(2026, 8, 29, 12, 0, 0)
    return args


def test_matrix_bundles_attacks_and_expands_defenses_seeds_and_targets(tmp_path):
    args = _args(
        "--models",
        "clip_mlp,clip_adapter",
        "--datasets",
        "caltech101",
        "--attacks",
        "blackbox_loss,projres",
        "--defenses",
        "none,www",
        "--seeds",
        "7,8",
        "--target-clients",
        "0,1",
        "--results-root",
        str(tmp_path),
    )

    tasks, skipped = build_tasks(CATALOG, args)

    assert skipped == []
    assert len(tasks) == 2 * 1 * 2 * 2 * 2
    assert {task.attacks for task in tasks} == {("blackbox_loss", "projres")}
    assert {task.defense for task in tasks} == {"none", "www"}
    assert {task.config["projres"]["enabled"] for task in tasks} == {True}
    assert list(tmp_path.iterdir()) == [], "task expansion must not write results"


def test_all_attacks_are_resolved_per_model_and_incompatible_model_is_skipped(tmp_path):
    all_args = _args(
        "--models",
        "resnet18,bert_lora",
        "--datasets",
        "default",
        "--attacks",
        "all",
        "--defenses",
        "none",
        "--results-root",
        str(tmp_path),
    )
    tasks, skipped = build_tasks(CATALOG, all_args)
    assert skipped == []
    by_model = {task.model: task for task in tasks}
    assert by_model["resnet18"].attacks == ("fedmia_loss",)
    assert len(by_model["bert_lora"].attacks) == 11

    projres_args = _args(
        "--models",
        "resnet18,clip_lora",
        "--datasets",
        "default",
        "--attacks",
        "projres",
        "--results-root",
        str(tmp_path),
    )
    tasks, skipped = build_tasks(CATALOG, projres_args)
    assert {task.model for task in tasks} == {"clip_lora"}
    assert any("resnet18" in message and "不支持攻击" in message for message in skipped)


def test_bert_lora_defaults_match_the_cola_rank16_protocol(tmp_path):
    tasks, skipped = build_tasks(
        CATALOG,
        _args(
            "--models",
            "bert_lora",
            "--results-root",
            str(tmp_path),
        ),
    )

    assert skipped == []
    assert len(tasks) == 1
    task = tasks[0]
    assert task.dataset == "cola"
    assert task.defense == "none"
    assert len(task.attacks) == 11
    assert task.config["num_global_iters"] == 500
    assert task.config["learning_rate"] == 0.015
    assert task.config["lora"]["rank"] == 16
    assert task.config["lora"]["alpha"] == 32.0
    assert list(tmp_path.iterdir()) == []


def test_catalog_defenses_deep_merge_to_valid_configs(tmp_path):
    cases = (
        ("resnet18", "cifar100", "record_dp", "vision"),
        ("bert_adapter", "sst5", "record_dp", "text"),
        ("bert_adapter", "sst5", "local_client_dp", "text"),
        ("bert_lora", "cola", "www", "text"),
    )
    for model, dataset, defense, runner in cases:
        attacks = CATALOG["models"][model]["supported_attacks"]
        config = resolve_model_config(
            CATALOG,
            model=model,
            dataset=dataset,
            attacks=attacks,
            defense=defense,
            seed=42,
            target_client_id=0,
            results_dir=tmp_path / model,
        )
        validate_resolved_config(config, runner)
        assert config["defense"]["name"] == defense
        assert config["results_dir_is_run_dir"] is True
    assert list(tmp_path.iterdir()) == []


def test_training_only_filters_attack_specific_configuration(tmp_path):
    config = resolve_model_config(
        CATALOG,
        model="clip_adapter",
        dataset="flowers",
        attacks=[],
        defense="none",
        seed=1,
        target_client_id=0,
        results_dir=tmp_path,
    )
    validate_resolved_config(config, "vision")
    assert config["audit"]["enabled"] is False
    assert config["audit"]["attacks"] == []
    assert config["audit"]["exact_batch_membership_attacks"] == []
    assert config["audit"]["attack_audit_intervals"] == {}
    assert config["projres"]["enabled"] is False


def test_run_ids_are_first_level_task_directories(tmp_path):
    tasks, _ = build_tasks(
        CATALOG,
        _args(
            "--models",
            "bert_lora",
            "--datasets",
            "imdb",
            "--attacks",
            "loss_series",
            "--defenses",
            "none,www",
            "--results-root",
            str(tmp_path),
        ),
    )
    assert len(tasks) == 2
    assert all(task.run_dir.parent == Path(tmp_path) for task in tasks)
    assert all(task.config_path == task.run_dir / "run_config.yaml" for task in tasks)


def test_task_identity_is_logged_once_and_child_lines_are_not_reprefixed(tmp_path):
    tasks, _ = build_tasks(
        CATALOG,
        _args(
            "--models",
            "bert_lora",
            "--datasets",
            "imdb",
            "--attacks",
            "none",
            "--results-root",
            str(tmp_path),
        ),
    )
    task = tasks[0]

    header = _task_header(task)
    assert "TASK | model=bert_lora | dataset=imdb" in header
    assert header.count("model=") == 1
    assert header.count("dataset=") == 1
    assert header.count("run=") == 1
    assert "phase=train" in header
    assert "gpu=" in header

    child_line = (
        "2026-09-01 00:00:01,000 INFO server: "
        "Progress | round=50/500 | loss=0.5142\n"
    )
    assert _forward_child_line(child_line) == child_line
    assert "model=" not in _forward_child_line(child_line)
    assert _forward_child_line("traceback line") == "traceback line\n"

    footer = _timestamped_line("EXIT | returncode=0")
    assert "EXIT | returncode=0" in footer
    assert "model=" not in footer


def test_run_log_keeps_task_identity_only_in_header(tmp_path, monkeypatch):
    tasks, _ = build_tasks(
        CATALOG,
        _args(
            "--models",
            "bert_lora",
            "--datasets",
            "cola",
            "--attacks",
            "none",
            "--results-root",
            str(tmp_path),
        ),
    )
    task = tasks[0]
    child_line = (
        "2026-09-01 00:00:01,000 INFO servers.serverbase: "
        "Progress | round=50/500 | loss=0.5142 | lr=0.01\n"
    )

    class FakeProcess:
        stdout = iter((child_line,))

        @staticmethod
        def wait():
            return 0

    monkeypatch.setattr(
        "scripts.run_privacy_experiments.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    result = run_task(task)

    assert result.returncode == 0
    log = (task.run_dir / "run.log").read_text(encoding="utf-8")
    assert log.count("model=bert_lora") == 1
    assert log.count("dataset=cola") == 1
    assert child_line in log
    assert "model=bert_lora | " + child_line not in log
    assert "EXIT | returncode=0" in log


def test_evaluation_progress_keeps_metrics_and_partial_client_ids():
    line = _format_round_progress(
        round_index=49,
        total_rounds=500,
        loss=0.5142,
        accuracy=0.775647,
        selected_ids=[0, 2],
        total_users=30,
        audit_snapshots=55,
        learning_rate=0.005,
        mcc=0.429646,
    )

    assert "Progress | round=50/500" in line
    assert "loss=0.5142" in line
    assert "mcc=0.4296" in line
    assert "accuracy=77.56%" in line
    assert "lr=0.005" in line
    assert "selected=[0,2]" in line
    assert "audit_snapshots=55" in line
