from __future__ import annotations

import torch
from analysis_scripts.verify_promptres_toy import run_toy_verification
from main import default_config, validate_config
from privacy_attacks.promptres import (
    positive_cosine_squared,
    promptres_round_scores,
    run_promptres,
)


def test_promptres_toy_prompt_update_separates_members_without_clip_checkpoint():
    result = run_toy_verification()

    assert result["trainable_parameters"] == ["prompt"]
    assert result["observed_update_norm"] > 0
    assert result["auc"] == 1.0
    assert result["member_mean_score"] > result["nonmember_mean_score"]
    assert result["nonmember_mean_score"] == 0.0


def test_promptres_uses_only_positive_candidate_alignment():
    update = torch.tensor([1.0, 0.0])
    candidates = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    scores = positive_cosine_squared(update, candidates)

    assert torch.allclose(scores, torch.tensor([1.0, 0.0, 0.0]))


def test_promptres_background_residual_removes_shared_client_directions():
    target_update = torch.tensor([1.0, 1.0, 1.0])
    references = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )
    candidates = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    direct, _ = promptres_round_scores(target_update, candidates)
    residual, used_rank = promptres_round_scores(
        target_update,
        candidates,
        references,
        background_rank=1,
    )

    assert torch.allclose(direct[0], direct[1])
    assert used_rank == 1
    assert residual[0] > 0.99
    assert residual[1] == 0.0


def test_promptres_aggregates_observed_round_scores():
    membership = torch.tensor([1, 0])
    observations = [
        {
            "round": 1,
            "client_ids": torch.tensor([0, 1]),
            "promptres": torch.tensor([[0.8, 0.2], [0.1, 0.1]]),
            "promptres_effective_ranks": [1, 1],
        },
        {
            "round": 3,
            "client_ids": torch.tensor([1, 0]),
            "promptres": torch.tensor([[0.1, 0.1], [0.6, 0.0]]),
            "promptres_effective_ranks": [1, 1],
        },
    ]

    result = run_promptres(observations, membership, 0, aggregation="mean")

    assert torch.allclose(result.scores, torch.tensor([0.7, 0.1]))
    assert result.metadata["rounds"] == [1, 3]
    assert result.metadata["effective_background_ranks"] == [1, 1]


def test_promptres_is_a_configurable_passive_prompt_attack():
    config = default_config()
    config["audit"]["attacks"] = ["promptres"]
    config["audit"]["promptres_background_rank"] = 2
    config["audit"]["promptres_aggregation"] = "max"

    validate_config(config)


def test_promptres_rejects_background_rank_without_reference_client():
    config = default_config()
    config["sample_users"] = 1
    config["audit"]["attacks"] = ["promptres"]
    config["audit"]["promptres_background_rank"] = 1

    try:
        validate_config(config)
    except ValueError as error:
        assert "two clients" in str(error)
    else:
        raise AssertionError("PromptRes background unexpectedly accepted one client")
