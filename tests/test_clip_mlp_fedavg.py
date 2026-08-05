from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from aggregator.aggregator_builder import build_aggregator
from aggregator.fedavg_aggregator import aggregate_fedavg_model_states
from context.context import Context
from main import default_config, validate_config
from privacy_attacks.auditor import MembershipAuditor
from servers.serverbase import ServerBase
from trainmodel.clip_feature_cache import (
    collate_clip_features,
    precompute_federated_clip_features,
)
from trainmodel.clip_mlp import CLIPImageMLP


class TinyCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=4)
        self.encoder = nn.Linear(6, 4)
        self.encoded_samples = 0

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.encoded_samples += int(pixel_values.shape[0])
        return self.encoder(pixel_values.flatten(1))


def _model() -> CLIPImageMLP:
    return CLIPImageMLP(TinyCLIP(), num_classes=3, hidden_dim=5)


def test_clip_mlp_freezes_clip_and_backpropagates_only_through_two_linear_layers():
    model = _model()
    model.train()
    logits = model(torch.randn(4, 1, 2, 3))
    nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 1])).backward()

    assert not model.clip_model.training
    assert all(
        not parameter.requires_grad for parameter in model.clip_model.parameters()
    )
    assert all(parameter.grad is None for parameter in model.clip_model.parameters())
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert [name for name, _ in trainable] == [
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
    ]
    assert all(parameter.grad is not None for _, parameter in trainable)

    query = torch.randn(1, 1, 2, 3, requires_grad=True)
    model(query).sum().backward()
    assert query.grad is not None
    assert float(query.grad.norm()) > 0


def test_clip_mlp_client_copy_shares_frozen_encoder_but_clones_mlp_state():
    model = _model()
    clone = model.clone_head_only()
    assert clone.clip_model is model.clip_model
    assert clone.classifier is not model.classifier
    assert all(
        torch.equal(clone.classifier.state_dict()[name], tensor)
        for name, tensor in model.classifier.state_dict().items()
    )


def test_clip_mlp_precomputes_all_images_and_reuses_vectors_for_train_and_test(
    tmp_path,
):
    images = torch.randn(12, 1, 2, 3)
    labels = torch.arange(12) % 3
    raw_train = [
        TensorDataset(images[:3], labels[:3]),
        TensorDataset(images[3:6], labels[3:6]),
    ]
    raw_test = [
        TensorDataset(images[6:9], labels[6:9]),
        TensorDataset(images[9:], labels[9:]),
    ]
    model = _model()
    encoded_train, encoded_test, summary = precompute_federated_clip_features(
        model,
        raw_train,
        raw_test,
        collate_fn=None,
        batch_size=2,
    )

    assert model.clip_model.encoded_samples == 12
    assert summary.train_samples == summary.test_samples == 6
    assert summary.feature_dimension == 4
    assert all(dataset.tensors[0].shape[1] == 4 for dataset in encoded_train)

    server = ServerBase(
        train_mode="centralized",
        device=torch.device("cpu"),
        dataset_name="toy",
        train_sets=encoded_train,
        test_sets=encoded_test,
        class_names=["a", "b", "c"],
        model=model,
        batch_size=2,
        learning_rate=0.1,
        num_glob_iters=2,
        local_epochs=1,
        total_users=2,
        results_dir=str(tmp_path),
        user_per_round=2,
        aggregator=build_aggregator("fedavg"),
        collate_fn=collate_clip_features,
        eval_interval=1,
        audit_config={
            "enabled": False,
            "attacks": [],
            "target_client_id": 0,
            "training_health_check": True,
        },
        defense_config={"name": "none"},
    )
    server.train()

    assert model.clip_model.encoded_samples == 12
    assert server.auditor.total_rounds == 2


def test_fedavg_sample_weights_all_mlp_parameters():
    model = _model()
    ctx = Context(2, model, ["a", "b", "c"])
    ctx.samples_num = [1, 3]
    ctx.updated_model_state = {
        0: {
            name: torch.ones_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        1: {
            name: torch.full_like(parameter, 5.0)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
    }

    aggregate_fedavg_model_states(ctx, [0, 1])

    assert set(ctx.new_model_state[0]) == set(ctx.trainable_param_names)
    assert all(
        torch.equal(tensor, torch.full_like(tensor, 4.0))
        for tensor in ctx.new_model_state[0].values()
    )


def test_clip_mlp_config_requires_plain_full_dataset_fedavg():
    config = default_config()
    config.update(
        {
            "model_type": "clip_mlp",
            "aggregator": "fedavg",
            "train_mode": "centralized",
            "use_full_dataset": True,
            "fpl_shots": None,
        }
    )
    config["audit"]["enabled"] = False
    config["audit"]["attacks"] = []
    validate_config(config)

    config["audit"]["enabled"] = True
    config["audit"]["attacks"] = ["fedmia_cosine", "promptmia"]
    validate_config(config)

    invalid = dict(config)
    invalid["fpl_shots"] = 1
    invalid["use_full_dataset"] = False
    with pytest.raises(ValueError, match="full dataset"):
        validate_config(invalid)


def test_clip_mlp_runs_complete_fedavg_training_and_saves_only_mlp(tmp_path):
    images = torch.randn(12, 1, 2, 3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    train_sets = [
        TensorDataset(images[:6], labels[:6]),
        TensorDataset(images[6:], labels[6:]),
    ]
    test_sets = [
        TensorDataset(images[:6], labels[:6]),
        TensorDataset(images[6:], labels[6:]),
    ]
    model = _model()
    initial_backbone = {
        name: tensor.detach().clone()
        for name, tensor in model.clip_model.state_dict().items()
    }
    initial_head = {
        name: tensor.detach().clone()
        for name, tensor in model.classifier.state_dict().items()
    }
    server = ServerBase(
        train_mode="centralized",
        device=torch.device("cpu"),
        dataset_name="toy",
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=["a", "b", "c"],
        model=model,
        batch_size=3,
        learning_rate=0.2,
        num_glob_iters=2,
        local_epochs=1,
        total_users=2,
        results_dir=str(tmp_path),
        user_per_round=2,
        aggregator=build_aggregator("fedavg"),
        eval_interval=1,
        audit_config={
            "enabled": False,
            "attacks": [],
            "target_client_id": 0,
            "training_health_check": True,
        },
        defense_config={"name": "none"},
    )

    assert server.train() == []
    saved = torch.load(tmp_path / "final_mlp.pt", weights_only=True)
    assert set(saved) == {
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
    }
    assert not (tmp_path / "final_prompt.pt").exists()
    assert any(
        not torch.equal(model.classifier.state_dict()[name], tensor)
        for name, tensor in initial_head.items()
    )
    assert all(
        torch.equal(model.clip_model.state_dict()[name], tensor)
        for name, tensor in initial_backbone.items()
    )


def test_clip_mlp_low_fpr_full_uses_all_candidates_and_caches_clip(tmp_path):
    def dataset(count: int, offset: float) -> TensorDataset:
        images = torch.linspace(
            -1.0 + offset,
            1.0 + offset,
            steps=count * 6,
        ).reshape(count, 1, 2, 3)
        labels = torch.arange(count) % 3
        return TensorDataset(images, labels)

    model = _model()
    users = [
        SimpleNamespace(
            id=0,
            train_data=dataset(7, 0.0),
            test_data=dataset(500, 0.1),
        ),
        SimpleNamespace(
            id=1,
            train_data=dataset(20, 0.2),
            test_data=dataset(500, 0.3),
        ),
    ]
    auditor = MembershipAuditor(
        model=model,
        users=users,
        target_client_id=0,
        device=torch.device("cpu"),
        results_dir=str(tmp_path),
        config={
            "enabled": True,
            "attacks": ["blackbox_loss", "grad_cosine", "fedmia_cosine"],
            "candidate_sampling": "low_fpr_full",
            "low_fpr_min_nonmembers": 1000,
            "audit_batch_size": 64,
            "low_fpr_gradient_batch_size": 8,
        },
        defense_config={"name": "none"},
        federated_method="fedavg",
        num_classes=3,
    )

    assert auditor.candidate_inputs_are_features
    assert auditor.images.shape == (1027, 4)
    assert int((auditor.membership == 1).sum()) == 7
    assert int((auditor.membership == 0).sum()) == 1020
    assert model.clip_model.encoded_samples == 1027
    encoded_before_outputs = model.clip_model.encoded_samples
    logits, _, losses = auditor._candidate_outputs(
        auditor.model, require_representation=False
    )
    assert logits.shape == (1027, 3)
    assert losses.shape == (1027,)
    assert model.clip_model.encoded_samples == encoded_before_outputs

    base_state = {
        name: parameter.detach().clone()
        for name, parameter in auditor.model.named_parameters()
        if parameter.requires_grad
    }
    updated_states = {
        client_id: {
            name: tensor - (client_id + 1) * 0.001 * torch.ones_like(tensor)
            for name, tensor in base_state.items()
        }
        for client_id in (0, 1)
    }
    auditor.observe_round(
        round_index=0,
        base_state=base_state,
        updated_states=updated_states,
        selected_ids=[0, 1],
    )
    observation = auditor.observations[0]
    assert observation["confidence"].shape == (2, 1027)
    assert observation["cosine"].shape == (2, 1027)
    assert model.clip_model.encoded_samples == encoded_before_outputs


def test_clip_mlp_low_fpr_reuses_precomputed_cpu_feature_tensors(tmp_path):
    def features(count: int) -> TensorDataset:
        return TensorDataset(
            torch.randn(count, 4),
            torch.arange(count) % 3,
        )

    model = _model()
    users = [
        SimpleNamespace(id=0, train_data=features(7), test_data=features(500)),
        SimpleNamespace(id=1, train_data=features(20), test_data=features(500)),
    ]
    auditor = MembershipAuditor(
        model=model,
        users=users,
        target_client_id=0,
        device=torch.device("cpu"),
        results_dir=str(tmp_path),
        config={
            "enabled": True,
            "attacks": ["blackbox_loss"],
            "candidate_sampling": "low_fpr_full",
            "low_fpr_min_nonmembers": 1000,
            "audit_batch_size": 512,
            "audit_interval": 5,
            "total_rounds": 12,
        },
        defense_config={"name": "none"},
        federated_method="fedavg",
        num_classes=3,
    )

    assert auditor.images.shape == (1027, 4)
    assert model.clip_model.encoded_samples == 0
    assert [auditor.should_observe(index) for index in (0, 1, 5, 10, 11)] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_attack_specific_audit_intervals_schedule_only_required_rounds():
    auditor = MembershipAuditor.__new__(MembershipAuditor)
    auditor.enabled = True
    auditor.total_rounds = 10
    auditor.audit_interval = 5
    auditor.config = {}
    auditor.signal_storage = "compact"
    auditor.pooled_client_audit = False
    auditor.target_client_id = 0
    auditor.attacks = [
        "blackbox_loss",
        "loss_series",
        "grad_cosine",
        "avg_cosine",
        "fedmia_loss",
        "fedmia_cosine",
    ]
    auditor.attack_audit_intervals = {
        "loss_series": 1,
        "avg_cosine": 1,
        "fedmia_loss": 1,
        "fedmia_cosine": 1,
    }
    auditor.observations = [{"round": round_index} for round_index in range(10)]

    assert [
        round_index
        for round_index in range(10)
        if auditor.should_observe(round_index)
    ] == list(range(10))
    assert [
        observation["round"]
        for observation in auditor._observations_for_attack("blackbox_loss")
    ] == [9]
    assert [
        observation["round"]
        for observation in auditor._observations_for_attack("grad_cosine")
    ] == [9]
    for attack in ("loss_series", "avg_cosine"):
        assert [
            observation["round"]
            for observation in auditor._observations_for_attack(attack)
        ] == list(range(10))
    for attack in ("fedmia_loss", "fedmia_cosine"):
        assert [
            observation["round"]
            for observation in auditor._observations_for_attack(attack)
        ] == list(range(10))
    assert auditor._attacks_for_round(1) == [
        "loss_series",
        "avg_cosine",
        "fedmia_loss",
        "fedmia_cosine",
    ]
    assert auditor._attacks_for_round(5) == auditor._attacks_for_round(1)
    assert auditor._attacks_for_round(9) == auditor.attacks
    selected_ids = list(range(10))
    assert auditor._audit_client_ids_for_attacks(
        ["loss_series", "avg_cosine"], selected_ids
    ) == [0]
    assert auditor._audit_client_ids_for_attacks(
        auditor._attacks_for_round(1), selected_ids
    ) == selected_ids

def test_all_attacks_run_in_toy_clip_mlp_fedavg(tmp_path):
    attacks = [
        "blackbox_loss",
        "loss_series",
        "grad_cosine",
        "avg_cosine",
        "nasr_passive",
        "nasr_active",
        "fedmia_loss",
        "fedmia_cosine",
        "transfer_representation",
        "codepoison",
        "pipra",
        "rmia",
        "imia",
        "quantile_mia",
        "yoqo",
        "canary",
        "promptmia",
        "promptres",
    ]

    def dataset(offset: float) -> TensorDataset:
        images = torch.linspace(
            -1.0 + offset,
            1.0 + offset,
            steps=8 * 6,
        ).reshape(8, 1, 2, 3)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        return TensorDataset(images, labels)

    server = ServerBase(
        train_mode="centralized",
        device=torch.device("cpu"),
        dataset_name="toy",
        train_sets=[dataset(index * 0.02) for index in range(4)],
        test_sets=[dataset(0.4 + index * 0.02) for index in range(4)],
        class_names=["zero", "one"],
        model=CLIPImageMLP(TinyCLIP(), num_classes=2, hidden_dim=5),
        batch_size=4,
        eval_batch_size=8,
        learning_rate=0.05,
        num_glob_iters=2,
        local_epochs=1,
        total_users=4,
        results_dir=str(tmp_path),
        user_per_round=4,
        aggregator=build_aggregator("fedavg"),
        eval_interval=1,
        audit_config={
            "enabled": True,
            "strict": True,
            "target_client_id": 0,
            "ensure_target_participation": True,
            "attacks": attacks,
            "max_samples_per_group": 4,
            "audit_interval": 1,
            "calibration_fraction": 0.5,
            "active_max_samples": 4,
            "active_ascent_steps": 1,
            "active_probe_cycles": 1,
            "auxiliary_fraction": 0.5,
            "qmia_epochs": 3,
            "pipra_shadow_prompts": 2,
            "pipra_shadow_steps": 1,
            "pipra_attack_epochs": 3,
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
        },
        defense_config={"name": "none"},
    )

    summaries = server.train()
    assert server.auditor.errors == {}
    assert {summary["attack"] for summary in summaries} == set(attacks)
    assert all(
        summary["metadata"]["model_type"] == "clip_mlp"
        and summary["metadata"]["trainable_scope"] == "mlp_only"
        for summary in summaries
    )
    signal_spaces = {
        summary["attack"]: summary["metadata"].get("signal_space")
        for summary in summaries
    }
    assert signal_spaces["promptmia"] == "class_decision_vectors"
    assert (tmp_path / "final_mlp.pt").exists()
