from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
import yaml

from aggregator.aggregator_builder import build_aggregator
from main import validate_config
from servers.serverbase import ServerBase
from trainmodel.clip_adapter import CLIPAdapter
from trainmodel.clip_mlp import CLIPImageMLP


ATTACKS = {
    "blackbox_loss",
    "loss_series",
    "grad_cosine",
    "avg_cosine",
    "fedmia_loss",
    "fedmia_cosine",
    "gradient_diff",
    "score_diff",
    "score_ratio",
    "fta",
    "projres",
}
EXACT_BATCH_ATTACKS = {
    "blackbox_loss",
    "grad_cosine",
    "gradient_diff",
    "projres",
    "score_diff",
    "score_ratio",
}


class _TinyCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=4)
        self.encoder = nn.Linear(6, 4)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values.flatten(1))


def _model(model_type: str) -> nn.Module:
    if model_type == "clip_mlp":
        return CLIPImageMLP(
            clip_model=_TinyCLIP(),
            num_classes=2,
            hidden_dim=4,
            dropout=0.0,
            device=torch.device("cpu"),
        )
    if model_type == "clip_adapter":
        return CLIPAdapter(
            clip_model=_TinyCLIP(),
            text_features=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            ),
            classnames=["zero", "one"],
            reduction=2,
            alpha=0.2,
            output_relu=False,
            text_adapter_enabled=True,
            device=torch.device("cpu"),
        )
    raise AssertionError(model_type)


def _dataset(samples_per_class: int, offset: float) -> TensorDataset:
    labels = torch.arange(samples_per_class * 2) % 2
    features = torch.linspace(
        -1.0 + offset,
        1.0 + offset,
        steps=labels.numel() * 4,
    ).reshape(labels.numel(), 4)
    return TensorDataset(features, labels)


def _audit_config() -> dict:
    return {
        "enabled": True,
        "strict": True,
        "target_client_id": 0,
        "audit_client_ids": [0],
        "ensure_target_participation": True,
        "attacks": sorted(ATTACKS),
        "candidate_sampling": "balanced_global_holdout",
        "require_full_target_train_members": True,
        "nonmember_to_member_ratio": 1,
        "exact_batch_membership_attacks": sorted(EXACT_BATCH_ATTACKS),
        "exact_batch_nonmember_to_member_ratio": 10,
        "paper_balanced_evaluation_size": 0,
        "low_fpr_min_nonmembers": 2,
        "low_fpr_max_members": 0,
        "low_fpr_max_nonmembers": 0,
        "audit_batch_size": 32,
        "audit_interval": 1,
        "attack_audit_intervals": {attack: 1 for attack in ATTACKS},
        "calibration_fraction": 0.5,
        "auxiliary_fraction": 0.5,
        "qmia_epochs": 2,
        "pipra_shadow_prompts": 2,
        "pipra_shadow_steps": 1,
        "pipra_attack_epochs": 2,
        "imia_models": 1,
        "imia_warmup_steps": 1,
        "imia_imitation_steps": 1,
        "imia_pivot_steps": 1,
        "query_max_samples": 4,
        "query_reference_models": 1,
        "yoqo_steps": 1,
        "canary_num_queries": 1,
        "canary_steps": 1,
        "canary_shadow_steps": 1,
        "promptmia_max_samples": 4,
        "promptmia_keys": 2,
        "promptres_background_rank": 1,
        "training_health_check": True,
    }


@pytest.mark.parametrize("model_type", ["clip_mlp", "clip_adapter"])
def test_clip_peft_runs_all_attacks_with_exact_batch_fedsgd(
    model_type, tmp_path
):
    train_sets = [_dataset(4, index * 0.02) for index in range(2)]
    test_sets = [_dataset(24, 0.4 + index * 0.02) for index in range(2)]
    server = ServerBase(
        device=torch.device("cpu"),
        dataset_name="toy",
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=["zero", "one"],
        model=_model(model_type),
        batch_size=2,
        eval_batch_size=32,
        learning_rate=0.05,
        num_glob_iters=2,
        local_epochs=1,
        total_users=2,
        results_dir=str(tmp_path / model_type),
        user_per_round=2,
        aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"),
        eval_interval=1,
        audit_config=_audit_config(),
        projres_config={
            "enabled": True,
            "evaluation_interval": 1,
            "decision_mode": "ranking",
            "threshold": None,
            "max_candidates": 2,
            "min_nonmembers": 20,
            "max_nonmembers": 20,
        },
        defense_config={"name": "none"},
        method_config={
            "client_optimizer": "sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "max_grad_norm": 0.0,
            "seed": 7,
        },
    )

    summaries = server.train()

    assert server.auditor.errors == {}
    assert {summary["attack"] for summary in summaries} == ATTACKS
    assert all(
        summary["metadata"]["model_type"] == model_type
        for summary in summaries
    )
    assert server.ctx.aggregation_weights == {0: 0.5, 1: 0.5}
    assert all(
        message["kind"] == "gradient"
        for message in server.ctx.protocol_messages.values()
    )
    assert (
        tmp_path / model_type / "privacy_audit" / "candidate_selection.pt"
    ).exists()
    assert (
        tmp_path
        / model_type
        / "privacy_audit"
        / "exact_batch_candidate_selection.pt"
    ).exists()


@pytest.mark.parametrize(
    "path",
    [
        "configs/clip_mlp_low_fpr_attacks.yaml",
        "configs/clip_adapter_low_fpr_attacks.yaml",
    ],
)
def test_clip_peft_configs_enforce_fedsgd_fewshot_and_bert_candidates(path):
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    validate_config(config)

    assert config["aggregator"] == "fedsgd"
    assert config["aggregation_weighting"] == "uniform"
    assert config["local_epochs"] == 1
    assert config["use_full_dataset"] is False
    assert config["fpl_shots"] == 16
    assert set(config["audit"]["attacks"]) == ATTACKS
    assert config["audit"]["candidate_sampling"] == "balanced_global_holdout"
    assert config["audit"]["require_full_target_train_members"] is True
    assert config["audit"]["nonmember_to_member_ratio"] == 1
    assert set(config["audit"]["exact_batch_membership_attacks"]) == (
        EXACT_BATCH_ATTACKS
    )
    assert config["audit"]["exact_batch_nonmember_to_member_ratio"] == 10
    assert config["projres"]["max_candidates"] == 32
    assert config["projres"]["min_nonmembers"] == 320
    assert config["projres"]["max_nonmembers"] == 320
