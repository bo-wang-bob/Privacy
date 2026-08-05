import torch
from torch import nn
from transformers import CLIPConfig, CLIPModel

from aggregator.aggregator_builder import build_aggregator
from main import default_config, validate_config
from trainmodel.custom_clip import CustomCLIP


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

def test_promptfl_model_forwards_and_backpropagates_without_checkpoints():
    model = CustomCLIP(
        clip_model=_tiny_clip(),
        processor=ToyProcessor(),
        classnames=["class one", "class two"],
        n_ctx=4,
        parameterization="promptfl",
    )
    images = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 1])
    logits = model(images)
    assert logits.shape == (2, 2)
    nn.functional.cross_entropy(logits, labels).backward()
    gradients = [
        parameter.grad
        for parameter in model.prompt_learner.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.norm()) > 1e-10 for gradient in gradients)
    assert all(parameter.grad is None for parameter in model.clip_model.parameters())
