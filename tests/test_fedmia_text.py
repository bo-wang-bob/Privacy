from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import default_config, validate_config
from privacy_attacks.fedmia import run_fedmia
from privacy_attacks.fedmia_text import (
    candidate_text_feature_changes,
    direct_text_gradient_changes,
    direct_text_gradient_round_scores,
    frobenius_cosine_scores,
)


class TinyTextFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = nn.Parameter(torch.eye(2, dtype=torch.float32))

    @torch.no_grad()
    def get_text_features(self, normalize: bool = True) -> torch.Tensor:
        if normalize:
            return F.normalize(self.prompt, dim=1)
        return self.prompt.detach().clone()


class TinyBatchedTextFeatureModel(TinyTextFeatureModel):
    def get_text_feature_contexts(self) -> tuple[torch.Tensor, ...]:
        return (self.prompt,)

    @torch.no_grad()
    def get_text_features_for_context_batch(
        self, contexts: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        if normalize:
            return F.normalize(contexts, dim=-1)
        return contexts.detach().clone()


def test_frobenius_cosine_handles_aligned_opposite_and_zero_changes():
    client = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    candidates = torch.stack((client, -client, torch.zeros_like(client)))

    scores = frobenius_cosine_scores(client, candidates)

    assert torch.allclose(scores, torch.tensor([1.0, -1.0, 0.0]))


def test_direct_text_gradient_matches_negative_cross_entropy_gradient():
    text_features = torch.tensor(
        [[0.8, 0.2], [-0.3, 0.7], [0.1, -0.9]], requires_grad=True
    )
    image_features = torch.tensor([[0.6, -0.4]])
    labels = torch.tensor([1])
    scale = 2.5
    loss = F.cross_entropy(
        scale * image_features @ text_features.t(), labels
    )
    expected = -torch.autograd.grad(loss, text_features)[0]

    changes, metadata = direct_text_gradient_changes(
        text_features.detach(), image_features, labels, logit_scale=scale
    )

    assert changes.shape == (1, 3, 2)
    assert torch.allclose(changes[0], expected, atol=1e-7)
    assert metadata["project_tangent"] is False
    assert metadata["zero_candidate_change_count"] == 0


def test_direct_text_gradient_can_project_onto_normalized_feature_tangent():
    text_features = F.normalize(torch.tensor([[1.0, 1.0], [-1.0, 2.0]]), dim=1)
    image_features = F.normalize(torch.tensor([[0.5, -0.7]]), dim=1)
    changes, metadata = direct_text_gradient_changes(
        text_features,
        image_features,
        torch.tensor([0]),
        project_tangent=True,
    )

    radial_components = (changes[0] * text_features).sum(dim=1)
    assert torch.allclose(radial_components, torch.zeros(2), atol=1e-7)
    assert metadata["project_tangent"] is True


def test_direct_text_gradient_round_scores_compare_full_matrices():
    text_features = torch.eye(2)
    image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    candidate_changes, _ = direct_text_gradient_changes(
        text_features, image_features, labels
    )
    clients = torch.stack((candidate_changes[0], -candidate_changes[0]))

    scores, metadata = direct_text_gradient_round_scores(
        text_features, image_features, labels, clients, candidate_batch_size=1
    )

    assert scores.shape == (2, 2)
    assert torch.isclose(scores[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(scores[1, 0], torch.tensor(-1.0), atol=1e-6)
    assert metadata["text_feature_shape"] == [2, 2]


def test_candidate_feature_changes_follow_virtual_descent_and_restore_state():
    model = TinyTextFeatureModel()
    base_state = {"prompt": model.prompt.detach().clone()}
    gradient = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).flatten()
    gradients = torch.stack((gradient, -gradient, torch.zeros_like(gradient)))

    _, changes, metadata = candidate_text_feature_changes(
        model,
        base_state,
        ["prompt"],
        gradients,
        probe_norm=1e-3,
        candidate_batch_size=2,
    )

    assert changes.shape == (3, 2, 2)
    assert changes[0].norm() > 0
    assert changes[1].norm() > 0
    assert torch.equal(changes[2], torch.zeros_like(changes[2]))
    assert metadata["zero_gradient_count"] == 1
    assert metadata["zero_candidate_change_count"] == 1
    assert torch.equal(model.prompt.detach(), base_state["prompt"])
    scores = frobenius_cosine_scores(changes[0], changes)
    assert torch.isclose(scores[0], torch.tensor(1.0), atol=1e-6)
    assert scores[1] < -0.999
    assert scores[2] == 0


def test_candidate_feature_changes_use_batched_context_encoder():
    model = TinyBatchedTextFeatureModel()
    base_state = {"prompt": model.prompt.detach().clone()}
    gradients = torch.tensor(
        [[0.0, 1.0, 1.0, 0.0], [0.0, -1.0, -1.0, 0.0]]
    )

    _, changes, metadata = candidate_text_feature_changes(
        model,
        base_state,
        ["prompt"],
        gradients,
        candidate_batch_size=2,
    )

    assert changes.shape == (2, 2, 2)
    assert metadata["batched_context_encoding"] == 1
    assert torch.equal(model.prompt.detach(), base_state["prompt"])


def test_fedmia_text_reuses_cross_client_null_and_reports_attack_name():
    membership = torch.tensor([1, 1, 0, 0])
    observations = [
        {
            "round": 0,
            "client_ids": torch.tensor([0, 1, 2]),
            "text_feature_cosine": torch.tensor(
                [
                    [0.9, 0.8, 0.2, 0.1],
                    [0.2, 0.2, 0.2, 0.2],
                    [0.3, 0.3, 0.3, 0.3],
                ]
            ),
            "text_feature_shape": [2, 4],
            "text_feature_probe_norm": 1e-3,
            "text_feature_zero_gradient_count": 0,
            "text_feature_zero_candidate_change_count": 0,
            "text_feature_batched_context_encoding": False,
        }
    ]

    result = run_fedmia(
        observations,
        membership,
        target_client_id=0,
        measurement="text_feature_cosine",
    )

    assert result.name == "fedmia_text"
    assert result.metadata["measurement"] == "text_feature_cosine"
    assert result.metadata["text_feature_probe"]["matrix_shapes"] == [[2, 4]]
    assert result.scores[:2].min() > result.scores[2:].max()


def test_fedmia_text_gradient_reuses_null_and_reports_fedmia_four():
    membership = torch.tensor([1, 1, 0, 0])
    observations = [
        {
            "round": 0,
            "client_ids": torch.tensor([0, 1, 2]),
            "text_gradient_cosine": torch.tensor(
                [
                    [0.9, 0.8, 0.2, 0.1],
                    [0.2, 0.2, 0.2, 0.2],
                    [0.3, 0.3, 0.3, 0.3],
                ]
            ),
            "text_gradient_shape": [2, 4],
            "text_gradient_logit_scale": 1.0,
            "text_gradient_project_tangent": False,
            "text_gradient_zero_candidate_change_count": 0,
            "text_gradient_client_change_norms": torch.ones(3),
        }
    ]

    result = run_fedmia(
        observations,
        membership,
        target_client_id=0,
        measurement="text_gradient_cosine",
    )

    assert result.name == "fedmia_text_gradient"
    assert result.metadata["measurement"] == "text_gradient_cosine"
    assert result.metadata["text_matrix_gradient"]["requires_jvp"] is False
    assert result.scores[:2].min() > result.scores[2:].max()


def test_fedmia_text_validation_rejects_hidden_updates_and_invalid_probe():
    config = default_config()
    config["audit"]["attacks"] = ["fedmia_text"]
    config["audit"]["audit_view"] = "released_prompt"
    with pytest.raises(ValueError, match="released_prompt"):
        validate_config(config)

    config["audit"]["audit_view"] = "protocol_plus_released_prompts"
    config["audit"]["fedmia_text_probe_norm"] = 0
    with pytest.raises(ValueError, match="probe_norm"):
        validate_config(config)


def test_fedmia_text_gradient_rejects_fedotp_logits():
    config = default_config()
    config["aggregator"] = "fedotp"
    config["audit"]["attacks"] = ["fedmia_text_gradient"]
    with pytest.raises(ValueError, match="optimal-transport logits"):
        validate_config(config)
