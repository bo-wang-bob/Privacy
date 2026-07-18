from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import CLIPConfig, CLIPModel
import yaml

from aggregator.aggregator_builder import build_aggregator
from context.context import Context
from main import default_config, validate_config
from trainmodel.custom_clip import CustomCLIP, entropic_partial_transport


class PromptState(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_ctx = nn.Parameter(torch.zeros(2, 2))
        self.local_ctx = nn.Parameter(torch.zeros(2, 2))


class ToyProcessor:
    def __call__(self, text=None, **_kwargs):
        count = len(text)
        length = 10
        input_ids = torch.arange(length).unsqueeze(0).repeat(count, 1) % 31
        attention_mask = torch.ones(count, length, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _tiny_clip() -> CLIPModel:
    config = CLIPConfig(
        projection_dim=16,
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "max_position_embeddings": 16,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 1,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 8,
            "patch_size": 4,
            "num_channels": 3,
        },
    )
    return CLIPModel(config)


def test_promptfl_is_a_named_prompt_only_fedavg_entry_point():
    aggregator = build_aggregator("promptfl")
    assert aggregator.name == "promptfl"
    config = default_config()
    config["aggregator"] = "promptfl"
    validate_config(config)

    paper_config = default_config()
    with Path("configs/federated_prompt_paper.yaml").open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    paper_config.update(loaded)
    paper_config["audit"] = default_config()["audit"] | loaded["audit"]
    paper_config["defense"] = default_config()["defense"] | loaded["defense"]
    validate_config(paper_config)


def test_personalized_prompt_aggregation_only_averages_global_context():
    ctx = Context(3, PromptState(), ["a", "b"], mode="local")
    ctx.samples_num = [1, 3, 9]
    ctx.user_selected = [0, 1]
    ctx.base_model_state = {
        user_id: {
            "global_ctx": torch.zeros(2, 2),
            "local_ctx": torch.full((2, 2), float(user_id)),
        }
        for user_id in range(3)
    }
    ctx.updated_model_state = {
        0: {
            "global_ctx": torch.full((2, 2), 2.0),
            "local_ctx": torch.full((2, 2), 10.0),
        },
        1: {
            "global_ctx": torch.full((2, 2), 6.0),
            "local_ctx": torch.full((2, 2), 11.0),
        },
    }

    build_aggregator("fedotp").aggregate(ctx)

    for state in ctx.new_model_state.values():
        assert torch.equal(state["global_ctx"], torch.full((2, 2), 5.0))
    assert torch.equal(ctx.new_model_state[0]["local_ctx"], torch.full((2, 2), 10.0))
    assert torch.equal(ctx.new_model_state[1]["local_ctx"], torch.full((2, 2), 11.0))
    assert torch.equal(ctx.new_model_state[2]["local_ctx"], torch.full((2, 2), 2.0))
    assert all(
        set(message["tensors"]) == {"global_ctx"}
        for message in ctx.protocol_messages.values()
    )


def test_fedotp_partial_transport_is_finite_and_moves_requested_mass():
    similarities = torch.tensor(
        [[[0.9, 0.2], [0.8, 0.1], [0.3, 0.7], [0.2, 0.8]]]
    )
    plan = entropic_partial_transport(
        similarities,
        epsilon=0.1,
        transported_mass=0.8,
        max_iterations=100,
        threshold=1e-6,
    )
    assert torch.isfinite(plan).all()
    assert torch.allclose(plan.sum(), torch.tensor(0.8), atol=1e-4)
    assert bool((plan.sum(dim=-1) <= 0.25 + 1e-5).all())


def test_all_paper_prompt_models_forward_without_data_or_checkpoints():
    images = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 1])
    for method, method_config in (
        ("promptfl", {}),
        (
            "fedotp",
            {
                "epsilon": 0.1,
                "transported_mass": 0.8,
                "max_iterations": 20,
                "threshold": 1e-4,
            },
        ),
        (
            "fedpgp",
            {"rank": 2, "contrastive_weight": 0.5, "temperature": 0.5},
        ),
    ):
        model = CustomCLIP(
            clip_model=_tiny_clip(),
            processor=ToyProcessor(),
            classnames=["class one", "class two"],
            n_ctx=4,
            parameterization=method,
            low_rank=int(method_config.get("rank", 2)),
            method_config=method_config,
        )
        logits = model(images)
        assert logits.shape == (2, 2)
        if method == "fedpgp":
            loss = model.fedpgp_training_loss(images, labels)
        else:
            loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        assert all(
            parameter.grad is not None
            for parameter in model.prompt_learner.parameters()
            if parameter.requires_grad
        )
        assert all(
            parameter.grad is None for parameter in model.clip_model.parameters()
        )
