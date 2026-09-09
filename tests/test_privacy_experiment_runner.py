import datetime as dt
from pathlib import Path

import pytest

from scripts.run_privacy_experiments import (
    _forward_child_line,
    _task_header,
    _timestamped_line,
    build_tasks,
    load_yaml,
    parse_args,
    print_result_overview,
    resolve_model_config,
    run_task,
    validate_resolved_config,
    TaskResult,
    write_outputs,
)
from servers.serverbase import _format_round_progress


CATALOG = load_yaml("configs/experiment_catalog.yaml")


def test_method_sweep_resolves_all_peft_models_and_candidate_protocols(tmp_path):
    args = _args("--models", "all", "--methods", "fedsgd,fedavg",
                 "--defenses", "none", "--results-root", str(tmp_path))
    tasks, skipped = build_tasks(CATALOG, args)
    assert any("resnet18" in item and "fedsgd" in item for item in skipped)
    for task in tasks:
        config = task.config
        method = config["aggregator"]
        assert f"_{method}_" in task.run_id
        baseline = load_yaml(CATALOG["models"][task.model]["config"])
        assert config["num_global_iters"] == (100 if method == "fedavg" else baseline["num_global_iters"])
        if task.model == "resnet18":
            assert config["resnet18"]["paper_protocol"] is False
            continue
        assert config["aggregation_weighting"] == ("uniform" if method == "fedsgd" else "sample_count")
        audit = config["audit"]
        assert bool(audit["exact_batch_membership_attacks"]) == (method == "fedsgd")
        assert bool(audit["client_train_membership_attacks"]) == (method == "fedavg")
        if method == "fedavg":
            assert config["projres"]["max_candidates"] == 0
            assert config["projres"]["candidate_scope"] == "full_client_train"
        if task.model.startswith("clip_"):
            expected_intervals = (
                load_yaml(CATALOG["models"]["bert_adapter"]["config"])["audit"]["attack_audit_intervals"]
                if method == "fedavg" else baseline["audit"]["attack_audit_intervals"]
            )
            assert audit["attack_audit_intervals"] == expected_intervals
            assert config["projres"]["evaluation_interval"] == expected_intervals["projres"]
        assert "models" not in config
    assert len({task.run_id for task in tasks}) == len(tasks)
    assert not list(tmp_path.iterdir())


def test_fedavg_epochs_and_weighting_override(tmp_path):
    tasks, _ = build_tasks(CATALOG, _args(
        "--models", "bert_lora", "--methods", "fedavg", "--local-epochs", "3",
        "--rounds", "80", "--aggregation-weighting", "uniform", "--results-root", str(tmp_path)))
    assert tasks[0].config["local_epochs"] == 3
    assert tasks[0].config["aggregation_weighting"] == "uniform"
    assert tasks[0].config["num_global_iters"] == 80
    with pytest.raises(ValueError, match="local_epochs"):
        build_tasks(CATALOG, _args("--models", "bert_lora", "--methods", "fedsgd",
                                  "--local-epochs", "3"))


@pytest.mark.parametrize("method", ["fedsgd", "fedavg"])
def test_clip_lora_projres_uses_last_cls_and_rejects_constant_cls(method, tmp_path):
    import copy
    args = _args("--models", "clip_lora", "--datasets", "cifar100", "--methods", method,
                 "--attacks", "projres", "--defenses", "none", "--results-root", str(tmp_path))
    tasks, skipped = build_tasks(CATALOG, args)
    assert not skipped and len(tasks) == 1
    config = tasks[0].config
    assert config["projres"]["token_reduction"] == "cls"
    assert config["projres"]["attacked_parameter"] is None
    for value in [None, "auto"]:
        automatic = copy.deepcopy(config)
        if value is None:
            automatic["projres"].pop("token_reduction")
        else:
            automatic["projres"]["token_reduction"] = value
        validate_resolved_config(automatic, "vision")
    legacy = copy.deepcopy(config)
    legacy["projres"].update(
        attacked_parameter="clip_model.vision_model.encoder.layers.0.self_attn.q_proj.lora_A",
        token_reduction="mean",
    )
    validate_resolved_config(legacy, "vision")
    for value in ["cls", "auto"]:
        invalid = copy.deepcopy(legacy)
        invalid["projres"]["token_reduction"] = value
        with pytest.raises(ValueError, match="constant across images"):
            validate_resolved_config(invalid, "vision")
    for value in ["last", "invalid"]:
        invalid = copy.deepcopy(config)
        invalid["projres"]["token_reduction"] = value
        with pytest.raises(ValueError, match="constant across images|token_reduction"):
            validate_resolved_config(invalid, "vision")


def test_resolved_config_can_switch_back_to_fedsgd_without_stale_candidates(tmp_path):
    tasks, _ = build_tasks(CATALOG, _args("--models", "bert_lora", "--methods", "fedavg",
                                        "--results-root", str(tmp_path)))
    config = tasks[0].config
    config["aggregator"] = "fedsgd"
    validate_resolved_config(config, "text")
    assert config["audit"]["client_train_membership_attacks"] == []
    assert "projres" in config["audit"]["exact_batch_membership_attacks"]
    assert config["projres"]["max_candidates"] == 16
    assert config["projres"]["min_nonmembers"] == 160


def _args(*values: str):
    args = parse_args(list(values))
    args.started_at = dt.datetime(2026, 8, 29, 12, 0, 0)
    return args


@pytest.mark.parametrize("model", ["clip_adapter", "clip_lora"])
@pytest.mark.parametrize("method", ["fedsgd", "fedavg"])
@pytest.mark.parametrize("shots", [16, 32, None])
def test_clip_training_data_regimes_resolve_and_validate(model, method, shots, tmp_path):
    overrides = [] if shots == 16 else [
        "--set", f"use_full_dataset={'true' if shots is None else 'false'}",
        "--set", f"fpl_shots={'null' if shots is None else shots}",
    ]
    tasks, skipped = build_tasks(CATALOG, _args(
        "--models", model, "--datasets", "cifar100,food101", "--methods", method,
        "--defenses", "none", "--results-root", str(tmp_path), *overrides,
    ))
    assert not skipped and len(tasks) == 2
    for task in tasks:
        validate_resolved_config(task.config, "vision")
        assert task.config["fpl_shots"] == shots
        assert task.config["use_full_dataset"] is (shots is None)
        assert task.config["partition_mode"] == "iid"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model", ["clip_adapter", "clip_lora"])
@pytest.mark.parametrize("full,shots", [(True, 16), (False, None), (False, 0),
                                        (False, -1), (False, 1.5), (False, True)])
def test_clip_training_data_regimes_reject_ambiguous_or_invalid_caps(model, full, shots):
    config = load_yaml(f"configs/models/{model}.yaml")
    config.update(use_full_dataset=full, fpl_shots=shots)
    with pytest.raises(ValueError, match="fpl_shots"):
        validate_resolved_config(config, "vision")


@pytest.mark.parametrize("model", ["clip_adapter", "clip_lora"])
@pytest.mark.parametrize("shots", [16, 32, None])
def test_direct_clip_cli_preserves_configured_data_regime(model, shots, tmp_path, monkeypatch):
    import sys
    import yaml
    from main import parse_args as parse_main_args

    config = load_yaml(f"configs/models/{model}.yaml")
    config.update(use_full_dataset=shots is None, fpl_shots=shots)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(path), "--model_type", model])
    resolved = parse_main_args()
    validate_resolved_config(resolved, "vision")
    assert resolved["use_full_dataset"] is (shots is None)
    assert resolved["fpl_shots"] == shots
    assert resolved["partition_mode"] == "iid"


@pytest.mark.parametrize("model", ["clip_adapter", "clip_lora"])
@pytest.mark.parametrize("options,expected_shots", [(["--shots", "32"], 32),
                                                   (["--use_full_dataset"], None)])
def test_direct_clip_cli_can_override_data_regime(model, options, expected_shots, monkeypatch):
    import sys
    from main import parse_args as parse_main_args

    monkeypatch.setattr(sys, "argv", [
        "main.py", "--config", f"configs/models/{model}.yaml", "--model_type", model, *options,
    ])
    resolved = parse_main_args()
    validate_resolved_config(resolved, "vision")
    assert resolved["fpl_shots"] == expected_shots
    assert resolved["use_full_dataset"] is (expected_shots is None)
    assert resolved["partition_mode"] == "iid"


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


def test_batch32_bert_sweep_automatically_resolves_each_defense_projres(tmp_path):
    tasks, skipped = build_tasks(
        CATALOG,
        _args(
            "--models", "bert_adapter", "--datasets", "cola,imdb",
            "--defenses", "none,record_dp,www", "--attacks", "all",
            "--set", "batch_size=32", "--set", "defense.target_epsilon=16",
            "--set", "defense.max_grad_norm=8", "--results-root", str(tmp_path),
        ),
    )
    assert not skipped and len(tasks) == 6
    for task in tasks:
        assert task.config["batch_size"] == 32 and len(task.attacks) == 11
        bounds = task.config["projres"]
        actual = tuple(bounds[k] for k in ("max_candidates", "min_nonmembers", "max_nonmembers"))
        assert actual == ((0, 0, 0) if task.defense == "record_dp" else (32, 320, 320))
        if task.defense == "www":
            assert task.config["defense"]["target_epsilon"] is None
            assert task.config["defense"]["noise_multiplier"] == 0
        else:
            assert task.config["defense"]["target_epsilon"] == 16
    assert list(tmp_path.iterdir()) == [], "config resolution must not create runs"


@pytest.mark.parametrize("model", ["clip_mlp", "clip_adapter", "clip_lora", "bert_lora", "gpt2_adapter"])
def test_projres_defaults_follow_final_batch_and_ratio_for_other_peft_models(model, tmp_path):
    tasks, skipped = build_tasks(
        CATALOG,
        _args(
            "--models", model, "--datasets", CATALOG["models"][model]["default_datasets"][0],
            "--defenses", "none,cofedmid", "--attacks", "all",
            "--set", "batch_size=7",
            "--set", "audit.exact_batch_nonmember_to_member_ratio=5",
            "--results-root", str(tmp_path),
        ),
    )
    assert not skipped and len(tasks) == 2
    for task in tasks:
        bounds = task.config["projres"]
        assert bounds["max_candidates"] == 7
        assert bounds["min_nonmembers"] == bounds["max_nonmembers"] == 35


@pytest.mark.parametrize("override", [
    ("projres.max_candidates", 16),
    ("projres", {"enabled": True, "max_candidates": 16}),
])
def test_explicit_projres_bounds_are_preserved_and_invalid_protocol_is_rejected(override, tmp_path):
    config = resolve_model_config(
        CATALOG, model="bert_adapter", dataset="cola", attacks=["projres"],
        defense="none", seed=42, target_client_id=0, results_dir=tmp_path,
        dotted_overrides=[("batch_size", 32), override],
    )
    assert config["projres"]["max_candidates"] == 16
    assert config["projres"]["min_nonmembers"] == config["projres"]["max_nonmembers"] == 320
    with pytest.raises(ValueError, match="complete real training batch"):
        validate_resolved_config(config, "text")


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


def test_overview_uses_final_task_metric_and_marks_failed_and_partial_runs(tmp_path, capsys):
    tasks, _ = build_tasks(CATALOG, _args(
        "--models", "bert_adapter", "--datasets", "cola,imdb",
        "--defenses", "none", "--attacks", "none", "--results-root", str(tmp_path),
    ))
    task = tasks[0]
    (task.run_dir / "privacy_audit").mkdir(parents=True)
    metrics = "round,loss,accuracy,mcc\n50,0.7,0.8,0.5\n100,0.8,0.7,-0.03\n"
    (task.run_dir / "training_metrics.csv").write_text(metrics)
    (task.run_dir / "privacy_audit/summary.json").write_text('{"errors":{"projres":"failed"}}')
    results = [TaskResult(t, code, "2026-09-06T10:00:00+00:00", "2026-09-06T11:02:03+00:00")
               for t, code in zip(tasks, [0, 1])]
    print_result_overview(results)
    display = capsys.readouterr().out
    assert "MCC" in display and "-0.0300" in display and "0.5000" not in display
    assert "PARTIAL" in display and "FAILED" in display and "N/A" in display
    assert "01:02:03" in display
    assert (task.run_dir / "training_metrics.csv").read_text() == metrics


def test_sweep_csv_preserves_reportable_primary_scores_and_resolution(tmp_path):
    import csv
    import json

    tasks, _ = build_tasks(CATALOG, _args(
        "--models", "bert_adapter", "--datasets", "cola", "--attacks", "all",
        "--results-root", str(tmp_path),
    ))
    task = tasks[0]
    (task.run_dir / "privacy_audit").mkdir(parents=True)
    (task.run_dir / "training_metrics.csv").write_text("round,accuracy,mcc\n500,0.8,0.6\n")
    attacks = [{
        "attack": name, "primary_metric": "tpr_at_fpr_0.01", "primary_score": .99,
        "auc": .97, "tpr_at_fpr_0.001": .98,
        "member_count": 32, "nonmember_count": 320, "fpr_resolution": 1/320,
        "reportable_metrics": {"auc": .61, "tpr_at_fpr_0.01": value,
                               "tpr_at_fpr_0.1": .4, "tpr_at_fpr_0.001": None},
    } for name, value in [("projres", .125), ("grad_cosine", 0.), ("gradient_diff", None)]]
    path = task.run_dir / "privacy_audit/summary.json"
    original = json.dumps({"attacks": attacks})
    path.write_text(original)
    _, output = write_outputs([TaskResult(task, 0, "", "")], tmp_path, "test")
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["primary_score"] for row in rows] == ["0.125", "0.0", ""]
    assert [row["primary_metric_reportable"] for row in rows] == ["True", "True", "False"]
    for row in rows:
        assert row["primary_score"] == row["tpr_at_fpr_0.01"]
        assert row["primary_metric"] == "tpr_at_fpr_0.01"
        assert row["auc"] == "0.61" and row["tpr_at_fpr_0.001"] == ""
        assert row["member_count"] == "32" and row["nonmember_count"] == "320"
        assert float(row["fpr_resolution"]) == 1/320
        assert row["final_metric"] == "mcc" and row["final_metric_value"] == "0.6"
    assert path.read_text() == original
