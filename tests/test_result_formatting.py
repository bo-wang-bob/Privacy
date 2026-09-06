import copy
import logging

import pytest
import torch

from utils.result_formatting import (
    format_attack_table, format_number, format_run_summary, format_sweep_summary,
)


def attack_summary():
    return {
        "attack": "projres", "auc": .543219, "primary_metric": "tpr_at_fpr_0.01",
        "primary_score": .99, "tpr_at_fpr_0.001": .98,
        "member_count": 32, "nonmember_count": 320,
        "reportable_metrics": {"auc": .543219, "tpr_at_fpr_0.01": .03125,
                               "tpr_at_fpr_0.001": None, "tpr_at_fpr_0.1": 0.},
        "metadata": {"private_array": ["DO_NOT_PRINT"] * 100},
    }


def test_attack_display_uses_reportable_scores_preserves_zero_and_omits_metadata():
    summary = attack_summary()
    before = copy.deepcopy(summary)
    display = format_attack_table([summary])
    assert "0.5432" in display and "3.12%" in display and "0.00%" in display
    assert "N/A" in display and "98.00%" not in display and "99.00%" not in display
    assert "DO_NOT_PRINT" not in display and summary == before
    # A historical raw-only summary must still respect nonmember resolution.
    del summary["reportable_metrics"]
    assert "98.00%" not in format_attack_table([summary])


@pytest.mark.parametrize("value", [None, "", float("nan"), float("inf"), "invalid"])
def test_missing_or_nonfinite_numbers_are_never_displayed_as_zero(value):
    assert format_number(value) == "N/A"
    assert format_number(value, percent=True) == "N/A"


def test_run_summary_has_bounded_privacy_display_and_preserves_mcc_units():
    defense = {
        "defense": "www", "steps_per_client": {str(i): 500 for i in range(100)},
        "privacy_accounting": {
            "privacy_unit": "record", "epsilon_upper_bound": 15.99,
            "target_epsilon": 16, "delta": 1e-5, "max_grad_norm": 8,
            "noise_multiplier": 1.234567, "formal_dp_enabled": False,
            "per_client": {str(i): {"private_array": "DO_NOT_PRINT"} for i in range(100)},
        },
    }
    display = format_run_summary(
        model="bert_adapter", dataset="cola", method="fedsgd", total_rounds=500,
        metrics={"round": 500, "loss": .123456, "accuracy": .71234, "mcc": -.034567, "learning_rate": .005},
        defense=defense, attacks=[attack_summary()], results_dir="/tmp/toy",
    )
    assert "71.23%" in display and "-0.0346" in display and "-3.46%" not in display
    assert "15.9900" in display and "16.0000" in display and "1e-05" in display
    assert "Formal DP" in display and "no" in display
    assert "DO_NOT_PRINT" not in display and "steps_per_client" not in display
    assert "defense_summary.json" in display and len(display) < 3000


def test_empty_results_and_constant_scores_are_explicit():
    assert "EXPERIMENT OVERVIEW" in format_sweep_summary([])
    summary = attack_summary()
    summary["score_degenerate"] = True
    display = format_attack_table([summary])
    assert "projres*" in display and "Constant scores" in display
    assert "No reported attack results" in format_attack_table([])


@pytest.mark.parametrize("model_type", ["clip_mlp", "bert_adapter"])
def test_completed_server_emits_one_shared_result_block(model_type, tmp_path, monkeypatch, caplog):
    from aggregator.aggregator_builder import build_aggregator
    from servers.serverbase import ServerBase
    from test_cofedmid import tiny_model, toy_dataset

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        server = ServerBase(
            device=torch.device("cpu"), dataset_name="toy", model=tiny_model(model_type, monkeypatch),
            train_sets=[toy_dataset(model_type, 4, 10+i) for i in range(2)],
            test_sets=[toy_dataset(model_type, 4, 20+i) for i in range(2)],
            class_names=["0", "1", "2"], batch_size=2, eval_batch_size=32,
            learning_rate=.01, num_glob_iters=1, local_epochs=1, total_users=2,
            results_dir=str(tmp_path), user_per_round=2, eval_interval=1,
            aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"),
            audit_config={"enabled": False}, projres_config={"enabled": False},
            defense_config={"name": "none"},
        )
        with caplog.at_level(logging.INFO, logger="servers.serverbase"):
            assert server.train() == []
        assert caplog.text.count("RUN RESULTS") == 1
        assert "Task metrics (final evaluation)" in caplog.text
        assert "Disabled." in caplog.text
        assert "Privacy defense completed: {" not in caplog.text
        assert (tmp_path / "defense_summary.json").exists()
    finally:
        torch.set_num_threads(previous)
