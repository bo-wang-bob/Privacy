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
from servers.serverbase import ServerBase
from trainmodel.visual_adapter import (
    VisualAdapter,
    VisualCLIPAdapter,
    get_visual_adapter_prompt_template,
)


class TinyCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=4)
        self.encoder = nn.Linear(6, 4)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values.flatten(1))


def _model(classes: int = 3, output_relu: bool = True) -> VisualCLIPAdapter:
    generator = torch.Generator().manual_seed(7)
    text_features = torch.randn(classes, 4, generator=generator)
    return VisualCLIPAdapter(
        TinyCLIP(),
        text_features,
        classnames=[f"class_{index}" for index in range(classes)],
        reduction=2,
        alpha=0.2,
        output_relu=output_relu,
    )


def test_visual_adapter_validation_and_requested_templates():
    with pytest.raises(ValueError, match="divisible"):
        VisualAdapter(feature_dim=5, reduction=2)
    assert (
        get_visual_adapter_prompt_template("CIFAR100")
        == "a photo of a {class}."
    )
    assert (
        get_visual_adapter_prompt_template("caltech101")
        == "a photo of a {class}."
    )
    assert get_visual_adapter_prompt_template("OxfordPets") == (
        "a photo of a {class}, a type of pet."
    )


def test_visual_adapter_freezes_clip_and_trains_only_bottleneck():
    model = _model()
    images = torch.randn(5, 1, 2, 3)
    labels = torch.tensor([0, 1, 2, 1, 0])
    cached_features = model.encode_images(images)
    assert torch.allclose(
        model(images),
        model.forward_from_image_features(cached_features),
        atol=1e-6,
    )
    nn.functional.cross_entropy(model(images), labels).backward()

    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_names == [
        "adapter.net.0.weight",
        "adapter.net.2.weight",
    ]
    assert all(
        parameter.grad is None for parameter in model.clip_model.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in model.adapter.parameters()
    )

    query = torch.randn(1, 1, 2, 3, requires_grad=True)
    model(query).sum().backward()
    assert query.grad is not None and float(query.grad.norm()) > 0


def test_visual_adapter_copy_and_fedavg_cover_only_adapter_parameters():
    model = _model()
    clone = model.clone_adapter_only()
    assert clone.clip_model is model.clip_model
    assert clone.adapter is not model.adapter
    assert torch.equal(clone.text_features, model.text_features)

    ctx = Context(2, model, model.classnames)
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
    assert set(ctx.new_model_state[0]) == {
        "adapter.net.0.weight",
        "adapter.net.2.weight",
    }
    assert all(
        torch.equal(value, torch.full_like(value, 4.0))
        for value in ctx.new_model_state[0].values()
    )


def test_visual_adapter_config_requires_fpl_16shot_fedavg():
    config = default_config()
    config.update(
        {
            "model_type": "visual_adapter",
            "aggregator": "fedavg",
            "train_mode": "centralized",
            "use_full_dataset": False,
            "fpl_shots": 16,
        }
    )
    config["audit"]["enabled"] = False
    config["audit"]["attacks"] = []
    validate_config(config)

    invalid = dict(config)
    invalid["fpl_shots"] = 8
    with pytest.raises(ValueError, match="16-shot"):
        validate_config(invalid)

    invalid = dict(config)
    invalid["aggregator"] = "promptfl"
    with pytest.raises(ValueError, match="aggregator=fedavg"):
        validate_config(invalid)


def test_all_attacks_run_in_toy_visual_adapter_fedavg(tmp_path):
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
        model=_model(classes=2, output_relu=False),
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
        summary["metadata"]["model_type"] == "visual_adapter"
        and summary["metadata"]["trainable_scope"] == "visual_adapter_only"
        for summary in summaries
    )
    signal_spaces = {
        summary["attack"]: summary["metadata"].get("signal_space")
        for summary in summaries
    }
    assert signal_spaces["promptmia"] == "adapter_input_projection_vectors"
    assert (tmp_path / "final_visual_adapter.pt").exists()
