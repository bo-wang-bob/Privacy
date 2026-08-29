from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import TensorDataset
import yaml

from aggregator.aggregator_builder import build_aggregator
from context.context import Context
from scripts.run_fedllm_adapter import validate_config
from servers.serverbase import ServerBase
from trainmodel.clip_lora import CLIPLoRA, LoRALinear
from trainmodel.transformer_lora import TransformerLoRAClassifier


_ALL_ATTACKS = {
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
_EXACT_BATCH_ATTACKS = {
    "blackbox_loss",
    "grad_cosine",
    "gradient_diff",
    "projres",
    "score_diff",
    "score_ratio",
}


class _TinyBertSelfAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            self.query(hidden) + self.key(hidden) + self.value(hidden)
        )


class _TinyBertLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = _TinyBertSelfAttention(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.attention.self(hidden)


class _TinyBertBackbone(nn.Module):
    def __init__(self, hidden_size: int = 6, layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            initializer_range=0.02,
        )
        self.embeddings = nn.Embedding(32, hidden_size)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [_TinyBertLayer(hidden_size) for _ in range(layers)]
        )

    def forward(self, *, input_ids, attention_mask, return_dict):
        assert return_dict
        hidden = self.embeddings(input_ids) * attention_mask.unsqueeze(-1)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=hidden[:, 0],
        )


def _model(monkeypatch) -> TransformerLoRAClassifier:
    monkeypatch.setattr(
        "trainmodel.transformer_lora.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: _TinyBertBackbone(),
    )
    return TransformerLoRAClassifier(
        model_path="unused",
        num_classes=3,
        target_modules=["query", "value"],
        layers="all",
        rank=2,
        alpha=4,
        dropout=0.1,
        scaling="rank",
        classifier_dropout=0.0,
        device=torch.device("cpu"),
    )


def _packed_inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [[1, 2, 0, 0], [1, 1, 0, 0]],
            [[3, 4, 5, 0], [1, 1, 1, 0]],
        ]
    )


def test_bert_lora_injects_standard_query_value_targets(monkeypatch):
    model = _model(monkeypatch)

    assert model.model_type == "bert_lora"
    assert model.lora_aggregation == CLIPLoRA.lora_aggregation
    assert model.num_lora_layers == 2
    assert len(model.injected_modules) == 4
    for layer in model.backbone.encoder.layer:
        assert isinstance(layer.attention.self.query, LoRALinear)
        assert isinstance(layer.attention.self.value, LoRALinear)
        assert isinstance(layer.attention.self.key, nn.Linear)

    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert "classifier.weight" in trainable_names
    assert "classifier.bias" in trainable_names
    assert len([name for name in trainable_names if name.endswith(".lora_A")]) == 4
    assert len([name for name in trainable_names if name.endswith(".lora_B")]) == 4
    assert not any(".base." in name for name in trainable_names)
    assert model(_packed_inputs()).shape == (2, 3)


def test_bert_lora_clients_keep_independent_cpu_peft_states(monkeypatch):
    model = _model(monkeypatch)
    client_zero = model.create_client_model(0)
    client_one = model.create_client_model(1)
    parameter_name = next(
        name for name in client_zero.export_trainable_state()
        if name.endswith(".lora_A")
    )
    state_zero = client_zero.export_trainable_state()
    state_zero[parameter_name].fill_(7.0)
    client_zero.load_trainable_state(state_zero)

    assert torch.all(client_zero.export_trainable_state()[parameter_name] == 7)
    assert not torch.equal(
        client_zero.export_trainable_state()[parameter_name],
        client_one.export_trainable_state()[parameter_name],
    )
    assert not torch.equal(
        client_zero.export_trainable_state()[parameter_name],
        model.export_trainable_state()[parameter_name],
    )
    assert client_zero(_packed_inputs()).shape == (2, 3)
    assert model._active_client_model is None
    assert all(parameter.device.type == "cpu" for parameter in client_zero.parameters())


def test_bert_lora_fedsgd_aggregates_every_factor_independently(monkeypatch):
    model = _model(monkeypatch)
    context = Context(2, model, ["0", "1", "2"], learning_rate=0.5)
    base_state = model.export_trainable_state()
    context.set_base_model_state(base_state)
    context.user_selected = [0, 1]
    context.update_sample_counts = {0: 2, 1: 8}
    context.client_gradients = {
        0: {name: torch.ones_like(value) for name, value in base_state.items()},
        1: {name: torch.full_like(value, 3.0) for name, value in base_state.items()},
    }

    build_aggregator("fedsgd", aggregation_weighting="uniform").aggregate(context)

    assert context.aggregation_weights == {0: 0.5, 1: 0.5}
    for name, value in base_state.items():
        assert torch.allclose(context.new_model_state[0][name], value - 1.0)


def test_bert_lora_exposes_lora_a_projres_surface(monkeypatch):
    model = _model(monkeypatch)
    name, layer = model.get_projres_attack_surface()
    representations, active_tokens = model.get_projres_representations(
        _packed_inputs(), parameter_name=name, token_reduction="cls"
    )

    assert name.endswith("query.lora_A")
    assert isinstance(layer, LoRALinear)
    assert representations.shape == (2, 6)
    assert active_tokens == 5


def test_bert_lora_runs_one_real_federated_server_round(monkeypatch, tmp_path):
    model = _model(monkeypatch)
    inputs = torch.stack([_packed_inputs()[0], _packed_inputs()[1]] * 2)
    labels = torch.tensor([0, 1, 2, 1])
    dataset = TensorDataset(inputs, labels)
    server = ServerBase(
        device=torch.device("cpu"),
        dataset_name="toy_text",
        train_sets=[dataset, dataset],
        test_sets=[dataset, dataset],
        class_names=["0", "1", "2"],
        model=model,
        batch_size=2,
        eval_batch_size=4,
        learning_rate=0.1,
        num_glob_iters=1,
        local_epochs=1,
        total_users=2,
        results_dir=str(tmp_path),
        user_per_round=2,
        aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"),
        eval_interval=1,
        audit_config={"enabled": False, "attacks": [], "seed": 7},
        projres_config={"enabled": False},
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

    assert summaries == []
    assert (tmp_path / "final_transformer_lora.pt").exists()
    assert set(server.ctx.protocol_messages) == {0, 1}
    assert all(
        message["kind"] == "gradient"
        for message in server.ctx.protocol_messages.values()
    )


def _text_dataset(samples_per_class: int, offset: int) -> TensorDataset:
    labels = torch.arange(samples_per_class * 3) % 3
    input_ids = torch.stack(
        [
            torch.tensor(
                [1 + offset + int(index), 2 + int(label), 3, 0]
            )
            % 31
            for index, label in enumerate(labels)
        ]
    )
    attention_mask = torch.tensor([[1, 1, 1, 0]]).repeat(labels.numel(), 1)
    return TensorDataset(torch.stack((input_ids, attention_mask), dim=1), labels)


def test_bert_lora_runs_all_eleven_attacks_with_exact_batch_projres(
    monkeypatch, tmp_path
):
    model = _model(monkeypatch)
    audit_config = {
        "enabled": True,
        "strict": True,
        "target_client_id": 0,
        "audit_client_ids": [0],
        "ensure_target_participation": True,
        "attacks": sorted(_ALL_ATTACKS),
        "candidate_sampling": "balanced_global_holdout",
        "require_full_target_train_members": True,
        "nonmember_to_member_ratio": 1,
        "exact_batch_membership_attacks": sorted(_EXACT_BATCH_ATTACKS),
        "exact_batch_nonmember_to_member_ratio": 10,
        "paper_balanced_evaluation_size": 0,
        "low_fpr_min_nonmembers": 2,
        "low_fpr_max_members": 0,
        "low_fpr_max_nonmembers": 0,
        "audit_batch_size": 32,
        "audit_interval": 1,
        "attack_audit_intervals": {attack: 1 for attack in _ALL_ATTACKS},
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
        "seed": 7,
    }
    audit_config["attack_audit_intervals"]["projres"] = 2
    server = ServerBase(
        device=torch.device("cpu"),
        dataset_name="toy_text",
        train_sets=[_text_dataset(3, 0), _text_dataset(3, 2)],
        test_sets=[_text_dataset(24, 4), _text_dataset(24, 6)],
        class_names=["0", "1", "2"],
        model=model,
        batch_size=2,
        eval_batch_size=32,
        learning_rate=0.05,
        num_glob_iters=2,
        local_epochs=1,
        total_users=2,
        results_dir=str(tmp_path),
        user_per_round=2,
        aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"),
        eval_interval=1,
        audit_config=audit_config,
        projres_config={
            "enabled": True,
            "evaluation_interval": 2,
            "decision_mode": "ranking",
            "threshold": None,
            "attacked_parameter": (
                "backbone.encoder.layer.0.attention.self.query.lora_A"
            ),
            "max_candidates": 2,
            "min_nonmembers": 20,
            "max_nonmembers": 20,
            "token_reduction": "auto",
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
    assert {summary["attack"] for summary in summaries} == _ALL_ATTACKS
    assert all(
        summary["metadata"]["model_type"] == "bert_lora"
        for summary in summaries
    )
    assert (
        tmp_path / "privacy_audit" / "exact_batch_candidate_selection.pt"
    ).exists()


def test_bert_lora_default_config_is_valid():
    with open(
        "configs/models/bert_lora.yaml", "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    validate_config(config)

    assert config["model_type"] == "bert_lora"
    assert config["lora"]["target_modules"] == ["query", "value"]
    assert config["lora"]["scaling"] == "rank"
    assert config["aggregation_weighting"] == "uniform"
