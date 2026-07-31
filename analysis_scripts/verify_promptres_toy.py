#!/usr/bin/env python3
"""Checkpoint-free first-stage feasibility test for PromptRes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from privacy_attacks.features import (
    flatten_state_delta,
    per_sample_prompt_gradients,
    trainable_names,
)
from privacy_attacks.metrics import membership_metrics
from privacy_attacks.promptres import positive_cosine_squared


class FrozenFeaturePromptClassifier(nn.Module):
    """A minimal text-prompt classifier with no trainable feature encoder."""

    def __init__(self, feature_dimension: int, classes: int = 2):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(classes, feature_dimension))

    def forward(self, frozen_features: torch.Tensor) -> torch.Tensor:
        return frozen_features @ self.prompt.T


def run_toy_verification() -> dict[str, float | int | list[str]]:
    """Run one-client, one-batch, one-step SGD membership verification."""
    feature_dimension = 16
    member_count = feature_dimension // 2
    features = torch.eye(feature_dimension)
    labels = torch.tensor([index % 2 for index in range(feature_dimension)])
    membership = torch.cat(
        (
            torch.ones(member_count, dtype=torch.long),
            torch.zeros(member_count, dtype=torch.long),
        )
    )

    model = FrozenFeaturePromptClassifier(feature_dimension)
    names = trainable_names(model)
    base_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    candidate_gradients, _, _ = per_sample_prompt_gradients(
        model, features, labels, names
    )

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.4,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(
        model(features[:member_count]), labels[:member_count]
    )
    loss.backward()
    optimizer.step()
    updated_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    # base - updated is the positive descent-gradient direction -Delta p.
    observed_update = flatten_state_delta(base_state, updated_state, names)
    scores = positive_cosine_squared(observed_update, candidate_gradients)
    metrics = membership_metrics(membership, scores)
    return {
        "auc": metrics["auc"],
        "member_mean_score": float(scores[membership == 1].mean()),
        "nonmember_mean_score": float(scores[membership == 0].mean()),
        "observed_update_norm": float(observed_update.norm()),
        "trainable_parameters": names,
        "member_count": member_count,
        "nonmember_count": member_count,
    }


def main() -> int:
    print(json.dumps(run_toy_verification(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
