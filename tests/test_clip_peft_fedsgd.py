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


@pytest.fixture
def clip_lora_projres_model():
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
    from trainmodel.clip_lora import CLIPLoRA

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    text = CLIPTextConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                          num_hidden_layers=1, num_attention_heads=2,
                          max_position_embeddings=8, eos_token_id=2,
                          bos_token_id=1, pad_token_id=0)
    vision = CLIPVisionConfig(hidden_size=8, intermediate_size=16,
                              num_hidden_layers=2, num_attention_heads=2,
                              image_size=4, patch_size=2)
    backbone = CLIPModel(CLIPConfig(text_config=text.to_dict(),
                                   vision_config=vision.to_dict(), projection_dim=4))
    model = CLIPLoRA(backbone, {"input_ids": torch.tensor([[1, 3, 2], [1, 4, 2], [1, 5, 2]])},
                     ["zero", "one", "two"], encoder="vision",
                     target_modules=["q", "k", "v", "o"], rank=2, dropout=0)
    # Exercise a nonzero A gradient without needing a warmup training round.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_B"):
                parameter.normal_(std=.03)
    yield model
    torch.set_num_threads(previous_threads)


def test_clip_lora_projres_default_uses_last_query_cls(clip_lora_projres_model):
    model = clip_lora_projres_model
    images = torch.randn(3, 3, 4, 4)
    name, layer = model.get_projres_attack_surface()
    assert name == "clip_model.vision_model.encoder.layers.1.self_attn.q_proj.lora_A"
    captured = []
    hook = layer.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
    model.eval()
    with torch.no_grad():
        model.clip_model.get_image_features(pixel_values=images)
    hook.remove()
    hidden = captured[0]
    expected = hidden[:, 0]
    assert not torch.allclose(expected[0], expected[1])
    model.train()
    for kwargs in [{}, {"token_reduction": "auto"}, {"token_reduction": "cls"}]:
        representations, tokens_per_image = model.get_projres_representations(images, **kwargs)
        torch.testing.assert_close(representations, expected)
        assert tokens_per_image == 5
        assert model.training  # Extraction restores training mode.
        assert not representations.requires_grad
    chunked = torch.cat([model.get_projres_representations(image[None])[0] for image in images])
    torch.testing.assert_close(chunked, expected)
    mean, _ = model.get_projres_representations(images, token_reduction="mean")
    torch.testing.assert_close(mean, hidden.mean(dim=1))


def test_clip_lora_projres_preserves_explicit_first_mean(clip_lora_projres_model):
    model = clip_lora_projres_model
    images = torch.randn(3, 3, 4, 4)
    name = "clip_model.vision_model.encoder.layers.0.self_attn.q_proj.lora_A"
    _, module = model.get_projres_attack_surface(name)
    captured = []
    hook = module.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    actual, count = model.get_projres_representations(images, name, "mean")
    hook.remove()
    hidden = captured[0]
    torch.testing.assert_close(actual, hidden.mean(dim=1))
    assert torch.equal(hidden[:, 0], hidden[:1, 0].expand_as(hidden[:, 0]))
    assert not torch.allclose(actual[0], actual[1])
    assert count == 5


@pytest.mark.parametrize("projection", ["q", "k", "v"])
def test_clip_lora_projres_rejects_first_layer_cls(clip_lora_projres_model, projection):
    name = f"clip_model.vision_model.encoder.layers.0.self_attn.{projection}_proj.lora_A"
    with pytest.raises(ValueError, match="constant across images"):
        clip_lora_projres_model.get_projres_representations(
            torch.randn(2, 3, 4, 4), parameter_name=name, token_reduction="cls",
        )


def test_clip_lora_projres_allows_contextualized_cls(clip_lora_projres_model):
    images = torch.randn(2, 3, 4, 4)
    for suffix in ["1.self_attn.q_proj.lora_A", "0.self_attn.out_proj.lora_A"]:
        representations, _ = clip_lora_projres_model.get_projres_representations(
            images, parameter_name="clip_model.vision_model.encoder.layers." + suffix,
            token_reduction="cls",
        )
        assert not torch.allclose(representations[0], representations[1])


@pytest.mark.parametrize("method", ["fedsgd", "fedavg"])
@pytest.mark.parametrize("audit_batch_size", [1, 4])
def test_clip_lora_unified_projres_default_scores_and_token_count(
    clip_lora_projres_model, method, audit_batch_size, tmp_path,
):
    from test_cofedmid import toy_dataset

    audit = _audit_config()
    audit.update(attacks=["blackbox_loss", "projres"],
                 exact_batch_membership_attacks=["blackbox_loss", "projres"],
                 audit_batch_size=audit_batch_size, training_health_check=False,
                 exact_batch_nonmember_to_member_ratio=2)
    server = ServerBase(
        device=torch.device("cpu"), dataset_name="toy", model=clip_lora_projres_model,
        train_sets=[toy_dataset("clip_lora", 2, 20+i) for i in range(2)],
        test_sets=[toy_dataset("clip_lora", 8, 30+i) for i in range(2)],
        class_names=["zero", "one", "two"], batch_size=3, eval_batch_size=8,
        learning_rate=.05, num_glob_iters=1, local_epochs=1, total_users=2,
        results_dir=str(tmp_path), user_per_round=2, eval_interval=1,
        aggregator=build_aggregator(method, aggregation_weighting="uniform"),
        audit_config=audit, projres_config={"enabled": True, "evaluation_interval": 1},
        defense_config={"name": "none"}, method_config={"client_optimizer": "sgd", "seed": 42},
    )
    summaries = server.train()
    assert server.auditor.errors == {}
    result = next(row for row in summaries if row["attack"] == "projres")
    assert not result["score_degenerate"]
    metadata = result["metadata"]
    members = 3 if method == "fedsgd" else 6
    assert result["member_count"] == members
    assert result["nonmember_count"] == 6
    assert metadata["sample_representation"] == "cls_token_input_to_lora_down_projection"
    assert metadata["attacked_parameter"] == "clip_model.vision_model.encoder.layers.1.self_attn.q_proj.lora_A"
    assert metadata["candidate_hidden_vector_count"] == members * 5
    assert metadata["batch_rank_bound"] == (members * 5 if method == "fedsgd" else None)
    assert metadata["paper_fedsgd_exact"] is False
    assert metadata["representation_state"] == "client_post_update_model"


@pytest.mark.parametrize("reduction", [None, "auto"])
def test_clip_lora_independent_projres_resolves_last_query_cls(
    clip_lora_projres_model, reduction, tmp_path,
):
    from privacy_attacks.projres_integrated import run_integrated_projres
    from test_cofedmid import toy_dataset

    model = clip_lora_projres_model
    model.eval()
    images, labels = toy_dataset("clip_lora", 1, 71).tensors
    base = model.lora_state_dict()
    model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(images), labels).backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    post = {name: tensor - .05 * gradients[name] for name, tensor in base.items()}
    user = SimpleNamespace(id=0, last_train_batch=(images, labels), collate_fn=None,
                           test_data=toy_dataset("clip_lora", 2, 72))
    config = {"max_candidates": 3, "min_nonmembers": 6, "max_nonmembers": 6}
    if reduction is not None:
        config["token_reduction"] = reduction
    payload = run_integrated_projres(
        model=model, users=[user], device=torch.device("cpu"),
        base_states={0: base}, updated_states={0: post}, client_gradients={0: gradients},
        learning_rate=.05, batch_size=3, eval_batch_size=2, local_epochs=1,
        round_index=0, seed=42, dataset_name="toy", client_ids=[0], config=config,
        output_path=tmp_path / "projres.json", federated_method="fedsgd",
    )
    result = payload["result"]
    assert result["dimensions"]["sample_representation"] == "cls_token_layer_input"
    assert result["dimensions"]["observed_hidden_vector_count"] == 15
    scores = torch.tensor(result["raw"]["scores"])
    assert torch.isfinite(scores).all() and scores.unique().numel() > 1


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
    # Unseeded tiny ReLU adapters can produce an entirely zero first upload.
    torch.manual_seed(42)
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
    "path, expected_shots",
    [
        ("configs/models/clip_mlp.yaml", 16),
        ("configs/models/clip_adapter.yaml", 100),
        ("configs/models/clip_lora.yaml", 100),
    ],
)
def test_clip_peft_configs_default_to_fedsgd_fewshot_and_bert_candidates(
    path, expected_shots
):
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    validate_config(config)

    assert config["aggregator"] == "fedsgd"
    assert config["aggregation_weighting"] == "uniform"
    assert config["local_epochs"] == 1
    assert config["use_full_dataset"] is False
    assert config["fpl_shots"] == expected_shots
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


@pytest.mark.parametrize("model_type", ["clip_mlp", "clip_adapter"])
@pytest.mark.parametrize("weighting", ["uniform", "sample_count"])
def test_fedavg_multiepoch_all_attacks_use_complete_client_train(
    model_type, weighting, tmp_path
):
    import json
    torch.manual_seed(42)
    train_sets = [_dataset(3, 0), _dataset(5, 0.02)]
    server = ServerBase(
        device=torch.device("cpu"), dataset_name="toy",
        train_sets=train_sets, test_sets=[_dataset(24, 0.4), _dataset(24, 0.5)],
        class_names=["zero", "one"], model=_model(model_type),
        batch_size=4, eval_batch_size=32, learning_rate=0.05,
        num_glob_iters=2, local_epochs=2, total_users=2,
        results_dir=str(tmp_path), user_per_round=2,
        aggregator=build_aggregator("fedavg", aggregation_weighting=weighting),
        eval_interval=1, audit_config=_audit_config(),
        projres_config={"enabled": True, "evaluation_interval": 1},
        defense_config={"name": "none"},
        method_config={"client_optimizer": "sgd", "seed": 7},
    )
    summaries = server.train()
    assert server.auditor.errors == {}
    assert {s["attack"] for s in summaries} == ATTACKS
    assert [u.last_update_sample_count for u in server.ctx.users] == [12, 20]
    assert all(u.last_update_gradients is None for u in server.ctx.users)
    assert all(m["kind"] == "model_update" for m in server.ctx.protocol_messages.values())
    expected_weights = {0: 0.5, 1: 0.5} if weighting == "uniform" else {0: 0.375, 1: 0.625}
    assert server.ctx.aggregation_weights == expected_weights
    for name in server.ctx.trainable_param_names:
        expected = sum(server.ctx.updated_model_state[i][name] * w
                       for i, w in expected_weights.items())
        torch.testing.assert_close(server.ctx.new_model_state[0][name], expected)
    for observation in server.auditor.exact_batch_observations:
        assert int(observation["membership"].sum()) == 6  # Not the final short batch of 2.
        assert observation["membership"].numel() == 12  # Repeated epochs count once.
        assert observation["member_local_indices"].sort().values.tolist() == list(range(6))
        metadata = observation["projres_diagnostics"]["metadata"]
        assert metadata["paper_fedsgd_exact"] is False
        assert metadata["batch_rank_bound"] is None
    saved = json.loads((tmp_path / "privacy_audit/summary.json").read_text())
    assert (tmp_path / "privacy_audit/client_train_update_candidate_selection.pt").exists()
    assert not (tmp_path / "privacy_audit/exact_batch_candidate_selection.pt").exists()
    assert all(s["member_count"] == 6 and s["nonmember_count"] == 6 for s in saved["attacks"])
    from utils.result_formatting import reportable_metric
    assert all(reportable_metric(s, "tpr_at_fpr_0.01") is None for s in summaries)
    assert all("tpr_at_fpr_0.001" in s["reportable_metrics"] for s in summaries)
    assert all(reportable_metric(s, "tpr_at_fpr_0.001") is None for s in summaries)


@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
def test_fedavg_ordinary_training_honors_optimizer_over_complete_epochs(optimizer_name):
    import copy
    from torch.utils.data import DataLoader
    from users.user import UserBase
    from privacy_defenses import DefenseController
    torch.manual_seed(9)
    model = nn.Linear(4, 2)
    reference = copy.deepcopy(model)
    data = _dataset(3, 0.0)
    config = {"client_optimizer": optimizer_name, "momentum": .7 if optimizer_name == "sgd" else 0.,
              "weight_decay": .02, "max_grad_norm": .3, "seed": 31}
    defense = DefenseController(config={"name": "none"}, device=torch.device("cpu"),
                                total_users=2, num_classes=2, total_rounds=1)
    defense.federated_method = "fedavg"
    defense.method_config = config
    user = UserBase(device=torch.device("cpu"), id=0, dataset_name="toy", train_data=data,
                    test_data=data, model=model, batch_size=4, learning_rate=.05,
                    local_epochs=2, federated_method="fedavg", method_config=config,
                    defense_controller=defense)
    optimizer = (torch.optim.SGD(reference.parameters(), lr=.05, momentum=.7, weight_decay=.02)
                 if optimizer_name == "sgd" else torch.optim.AdamW(reference.parameters(), lr=.05, weight_decay=.02))
    loader = DataLoader(data, batch_size=4, shuffle=True,
                        generator=torch.Generator().manual_seed(31))
    for _ in range(2):
        for x, y in loader:
            optimizer.zero_grad()
            nn.functional.cross_entropy(reference(x), y).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), .3)
            optimizer.step()
    user.train()
    assert user.last_update_sample_count == 12
    assert defense.steps[0] == 4
    for actual, expected in zip(user.model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("model", ["clip_adapter", "clip_lora"])
@pytest.mark.parametrize("dataset_name", ["cifar100", "food101"])
@pytest.mark.parametrize("shots", [16, 32, None])
@pytest.mark.parametrize("partition", ["iid", "dirichlet"])
def test_clip_data_regimes_cap_only_global_training(
    model, dataset_name, shots, partition, monkeypatch,
):
    import random
    import numpy as np
    import utils.data_loader as data_loader
    from main import _dataset_split_arguments

    # Exercise the real source loader, few-shot sampler and client splitter.
    # Both source splits exceed the historical per-class limits (200 / 50).
    class SourceDataset(torch.utils.data.Dataset):
        def __init__(self, root, train=None, split=None, **kwargs):
            self.is_train = train if train is not None else split == "train"
            samples_per_class = 256 if self.is_train else 64
            self.targets = [cls for cls in range(3) for _ in range(samples_per_class)]
            self.classes = ["zero", "one", "two"]
            self._labels = self.targets

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            sample = torch.tensor([index + (0 if self.is_train else 10000)])
            return sample, self.targets[index]

    monkeypatch.setattr(data_loader.datasets, "CIFAR100", SourceDataset)
    monkeypatch.setattr(data_loader.datasets, "Food101", SourceDataset)
    arguments = _dataset_split_arguments({
        "model_type": model, "fpl_shots": shots, "use_full_dataset": shots is None,
    })
    splitter = getattr(data_loader, f"generate_{partition}_split")
    if partition == "dirichlet":
        arguments["alpha"] = 0.5

    def sample():
        random.seed(42)
        np.random.seed(42)
        train_sets, test_sets, _ = splitter(dataset_name, num_users=4, **arguments)
        train = [(int(x.item()), label) for client in train_sets for x, label in client]
        test = [(int(x.item()), label) for client in test_sets for x, label in client]
        return train, test

    train, test = sample()
    assert (train, test) == sample()
    expected_per_class = 256 if shots is None else shots
    assert [sum(label == cls for _, label in train) for cls in range(3)] == [expected_per_class] * 3
    assert len({idx for idx, _ in train}) == expected_per_class * 3
    assert {idx for idx, _ in test} == set(range(10000, 10000 + 64 * 3))
    assert len(test) == 64 * 3
    assert not ({idx for idx, _ in train} & {idx for idx, _ in test})
