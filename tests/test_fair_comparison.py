import copy
import datetime

import pytest

from utils.fair_comparison import (
    aggregate_markdown_table,
    markdown_table,
    validate_fair_configs,
)
from main import _result_run_id
from analysis_scripts.veil_paper_results import (
    OFFICIAL_CONFIG,
    matches_official_method,
)


def _config():
    return {
        "train_mode": "centralized",
        "dataset_name": "flowers",
        "batch_size": 16,
        "eval_batch_size": 64,
        "learning_rate": 0.001,
        "num_global_iters": 5,
        "local_epochs": 1,
        "total_users": 10,
        "sample_users": 10,
        "dirichlet_alpha": 0.1,
        "seed": 42,
        "fpl_shots": 16,
        "n_ctx": 16,
        "class_specific_ctx": False,
        "audit": {
            "enabled": True,
            "audit_view": "protocol_plus_released_prompts",
            "strict": True,
            "target_client_id": 0,
            "ensure_target_participation": True,
            "attacks": ["fedmia_loss", "nasr_passive"],
            "max_samples_per_group": 32,
            "audit_interval": 2,
            "calibration_fraction": 0.5,
            "match_candidate_labels": True,
            "auxiliary_fraction": 0.5,
            "fedmia_loss_aggregation": "mean",
            "fedmia_cosine_aggregation": "mean",
            "rmia_offline_a": 0.3,
            "rmia_gamma": 2.0,
            "qmia_quantile": 0.9,
            "qmia_epochs": 200,
            "qmia_learning_rate": 0.01,
        },
    }


def test_validate_fair_configs_allows_method_specific_fields():
    first = _config()
    first["aggregator"] = "dpfpl"
    second = copy.deepcopy(first)
    second["aggregator"] = "fedask"
    second["fedask"] = {"rank": 16}
    validate_fair_configs([first, second])


def test_result_run_id_separates_concurrent_processes():
    now = datetime.datetime(2026, 7, 16, 18, 0, 0, 123456)
    assert _result_run_id(now, 10) != _result_run_id(now, 11)
    assert _result_run_id(now, 10).endswith("123456_10")


def test_validate_fair_configs_normalizes_known_audit_defaults():
    first = _config()
    second = copy.deepcopy(first)
    del second["audit"]["fedmia_loss_aggregation"]
    del second["audit"]["fedmia_cosine_aggregation"]
    validate_fair_configs([first, second])


def test_validate_fair_configs_rejects_audit_budget_mismatch():
    first = _config()
    second = copy.deepcopy(first)
    second["audit"]["max_samples_per_group"] = 64
    with pytest.raises(ValueError, match="max_samples_per_group"):
        validate_fair_configs([first, second])


def test_markdown_table_reports_worst_and_per_attack_metrics():
    run = {
        "method": "Local-GGEUR",
        "accuracy": 0.62,
        "worst_tpr_at_fpr_0.01": 0.125,
        "mean_tpr_at_fpr_0.01": 0.0625,
        "attacks": {
            "fedmia_loss": {"tpr_at_fpr_0.01": 0.0},
            "nasr_passive": {"tpr_at_fpr_0.01": 0.125},
        },
    }
    table = markdown_table([run])
    assert "Local-GGEUR" in table
    assert "0.6200" in table
    assert "0.1250" in table


def test_aggregate_markdown_table_reports_mean_and_sample_std():
    base = {
        "method": "Local-GGEUR",
        "accuracy": 0.62,
        "worst_tpr_at_fpr_0.01": 0.125,
        "mean_tpr_at_fpr_0.01": 0.0625,
        "attacks": {"nasr_passive": {"tpr_at_fpr_0.01": 0.125}},
    }
    other = copy.deepcopy(base)
    other["accuracy"] = 0.64
    other["worst_tpr_at_fpr_0.01"] = 0.0
    other["mean_tpr_at_fpr_0.01"] = 0.0
    other["attacks"]["nasr_passive"]["tpr_at_fpr_0.01"] = 0.0
    table = aggregate_markdown_table([base, other])
    assert "| Local-GGEUR | 2 |" in table
    assert "0.6300 +/- 0.0141" in table


def test_paper_selector_accepts_full_veil_and_rejects_ablation():
    config = copy.deepcopy(OFFICIAL_CONFIG)
    config["gpu"] = 0
    config["aggregator"] = "fedavg"
    config["defense"]["name"] = "veil"
    assert matches_official_method(config, "VEIL")
    config["defense"]["local_ggeur_augments"] = 0
    assert not matches_official_method(config, "VEIL")


def test_paper_selector_allows_runtime_seed_in_private_method_config():
    config = copy.deepcopy(OFFICIAL_CONFIG)
    config["gpu"] = 1
    config["aggregator"] = "dpfpl"
    config["defense"]["name"] = "none"
    config["dpfpl"]["seed"] = 42
    assert matches_official_method(config, "DP-FPL")
