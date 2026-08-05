from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from main import default_config
from privacy_attacks.projres_mlp import (
    one_batch_fedsgd_step,
    strict_mlp_projres,
)
from scripts.validate_projres_mlp_real import _validate_strict_config
from trainmodel.clip_mlp import CLIPImageMLP
from trainmodel.visual_adapter import VisualCLIPAdapter


class TinyCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=4)
        self.encoder = nn.Linear(6, 4, bias=False)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values.flatten(1))


class DirectClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(3, 2, bias=False))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


def test_strict_projres_uses_update_row_space_and_raw_l1_rule():
    update = torch.tensor([[2.0, 0.0, 0.0], [0.0, -3.0, 0.0]])
    candidates = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    )

    result = strict_mlp_projres(update, candidates, threshold=0.01)

    assert result.metadata["subspace"]["numerical_rank"] == 2
    assert torch.allclose(result.l1_residuals, torch.tensor([0.0, 0.0, 1.0]))
    assert result.predictions.tolist() == [1, 1, 0]
    assert torch.equal(result.scores, -result.l1_residuals)


def test_strict_projres_is_invariant_to_fedsgd_scale_and_sign():
    torch.manual_seed(4)
    update = torch.randn(3, 7)
    candidates = torch.randn(5, 7)

    first = strict_mlp_projres(update, candidates)
    second = strict_mlp_projres(-0.03 * update, candidates)

    assert torch.allclose(first.l1_residuals, second.l1_residuals, atol=1e-5)


def test_strict_projres_stabilization_preserves_the_same_projection_rule():
    update = torch.tensor([[2.0, 0.0, 0.0], [0.0, -3.0, 0.0]])
    candidates = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 0.0, 4.0], [1.0, 1.0, 1.0]]
    )

    result = strict_mlp_projres(update, candidates, threshold=0.5)
    expected_projection = candidates.clone()
    expected_projection[:, 2] = 0.0
    expected_residual = (candidates - expected_projection).norm(p=1, dim=1)

    assert torch.allclose(result.l1_residuals, expected_residual, atol=1e-7)
    assert result.predictions.tolist() == [1, 0, 0]
    gram = result.basis.t() @ result.basis
    assert torch.allclose(
        gram,
        torch.eye(gram.shape[0], dtype=gram.dtype),
        atol=1e-12,
        rtol=1e-12,
    )
    stabilization = result.metadata["numerical_stabilization"]
    assert stabilization["subspace_dtype"] == "torch.float64"
    assert stabilization["qr_reorthogonalized"] is True


def test_strict_projres_caps_float32_parameter_difference_at_batch_rank():
    torch.manual_seed(42)
    dimension = 64
    batch_size = 8
    learning_rate = 0.01
    before = torch.randn(dimension, dimension) * 0.03
    errors = torch.randn(dimension, batch_size) * 0.03
    members = torch.randn(batch_size, dimension) * 0.03
    exact_gradient = errors @ members / batch_size
    observed_update = before - (before - learning_rate * exact_gradient)
    candidates = torch.cat((members, torch.randn(12, dimension) * 0.03))

    observed = strict_mlp_projres(
        observed_update,
        candidates,
        threshold=0.01,
        max_rank=batch_size,
    )
    ideal = strict_mlp_projres(
        learning_rate * exact_gradient,
        candidates,
        threshold=0.01,
        max_rank=batch_size,
    )

    assert observed.metadata["subspace"]["numerical_rank"] > batch_size
    assert observed.metadata["subspace"]["used_rank"] == batch_size
    assert observed.metadata["numerical_stabilization"]["rank_cap"] == batch_size
    assert torch.equal(observed.predictions, ideal.predictions)
    assert torch.allclose(
        observed.l1_residuals,
        ideal.l1_residuals,
        atol=2e-4,
        rtol=1e-3,
    )


def test_one_batch_step_is_exact_vanilla_fedsgd_on_first_mlp_weight():
    torch.manual_seed(7)
    model = CLIPImageMLP(TinyCLIP(), num_classes=3, hidden_dim=5)
    images = torch.randn(4, 1, 2, 3)
    labels = torch.tensor([0, 1, 2, 1])
    backbone_before = {
        name: value.detach().clone()
        for name, value in model.clip_model.state_dict().items()
    }

    step = one_batch_fedsgd_step(model, images, labels, learning_rate=0.2)

    assert torch.allclose(step.observed_update, 0.2 * step.gradient, atol=1e-7)
    assert step.update_gradient_relative_error < 1e-5
    assert all(
        torch.equal(model.clip_model.state_dict()[name], value)
        for name, value in backbone_before.items()
    )


def test_one_batch_step_and_projres_support_visual_adapter_downsampling_layer():
    torch.manual_seed(5)
    clip = TinyCLIP()
    clip.logit_scale = nn.Parameter(torch.tensor(1.0))
    model = VisualCLIPAdapter(
        clip,
        torch.randn(3, 4),
        reduction=2,
        alpha=0.2,
        output_relu=True,
    )
    with torch.no_grad():
        model.adapter.net[0].weight.fill_(0.2)
        model.adapter.net[2].weight.fill_(0.2)
    member = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    candidates = torch.cat(
        (member, torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, -1.0, 0.0, 0.0]]))
    )

    step = one_batch_fedsgd_step(
        model,
        member,
        torch.tensor([1]),
        learning_rate=0.1,
        parameter_name="adapter.net.0.weight",
    )
    result = strict_mlp_projres(
        step.observed_update,
        candidates,
        threshold=1e-4,
        max_rank=1,
    )

    assert step.parameter_name == "adapter.net.0.weight"
    assert step.update_gradient_relative_error < 1e-5
    assert result.predictions.tolist() == [1, 0, 0]


def test_strict_projres_configuration_accepts_visual_adapter():
    config = default_config()
    config["model_type"] = "visual_adapter"

    _validate_strict_config(config)


def test_observed_one_sample_fedsgd_update_recovers_its_layer_input():
    model = DirectClassifier()
    member = torch.tensor([[1.0, 2.0, 0.0]])
    nonmember = torch.tensor([[0.0, 0.0, 1.0]])
    step = one_batch_fedsgd_step(
        model, member, torch.tensor([1]), learning_rate=0.1
    )

    result = strict_mlp_projres(
        step.observed_update,
        torch.cat((member, nonmember)),
        threshold=1e-5,
    )

    assert result.predictions.tolist() == [1, 0]
    assert result.l1_residuals[0] < 1e-5
    assert result.l1_residuals[1] > 0.99


def test_strict_projres_rejects_wrong_representation_width_and_zero_update():
    with pytest.raises(ValueError, match="width"):
        strict_mlp_projres(torch.ones(2, 3), torch.ones(4, 2))
    with pytest.raises(ValueError, match="zero update"):
        strict_mlp_projres(torch.zeros(2, 3), torch.ones(4, 3))
