from types import SimpleNamespace

import pytest
import torch

from privacy_defenses.controller import DefenseController
from privacy_defenses.www import encode_training_batches


class _FeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return images * 2.0


def _empty_user(sample_count: int):
    return SimpleNamespace(
        train_samples=sample_count,
        www_feature_seen=None,
        www_class_feature_counts=None,
        www_class_feature_means=None,
        www_within_class_scatter=None,
        www_within_class_covariance=None,
        www_within_class_covariance_dof=0,
    )


def _ranking_user(model: torch.nn.Module):
    user = _empty_user(sample_count=8)
    user.id = 0
    user.model = model
    user.last_train_batch = (
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 1]),
    )
    user.last_train_indices = torch.tensor([2, 6])
    user.get_parameters = lambda: {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    for name in (
        "www_score_count",
        "www_score_sum",
        "www_score_sum_sq",
        "www_score_min",
        "www_score_max",
        "www_score_last",
        "www_score_last_round",
    ):
        setattr(user, name, None)
    return user


def test_encode_training_batches_preserves_alignment_and_mode():
    model = _FeatureModel()
    model.train()
    batches = [
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([0, 1]),
        ),
        (torch.tensor([[5.0, 6.0]]), torch.tensor([1])),
    ]

    features = encode_training_batches(model, batches, torch.device("cpu"))

    assert model.training
    assert torch.equal(
        features,
        torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]),
    )


def test_www_feature_statistics_use_unique_samples_and_pooled_covariance():
    user = _empty_user(sample_count=5)
    update = DefenseController._update_www_feature_statistics

    update(
        user=user,
        encoded_features=torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 4.0]]),
        labels=torch.tensor([0, 1, 1]),
        sample_indices=torch.tensor([0, 2, 3]),
        num_classes=3,
    )
    update(
        user=user,
        encoded_features=torch.tensor(
            [[99.0, 99.0], [3.0, 0.0], [0.0, 6.0]]
        ),
        labels=torch.tensor([0, 0, 1]),
        sample_indices=torch.tensor([0, 1, 4]),
        num_classes=3,
    )

    assert torch.equal(user.www_feature_seen, torch.ones(5, dtype=torch.bool))
    assert torch.equal(
        user.www_class_feature_counts, torch.tensor([2, 3, 0])
    )
    assert torch.allclose(
        user.www_class_feature_means,
        torch.tensor(
            [[2.0, 0.0], [0.0, 4.0], [0.0, 0.0]], dtype=torch.float64
        ),
    )
    assert user.www_within_class_covariance_dof == 3
    assert torch.allclose(
        user.www_within_class_covariance,
        torch.tensor([[2.0 / 3.0, 0.0], [0.0, 8.0 / 3.0]], dtype=torch.float64),
    )


def test_www_feature_statistics_zero_covariance_without_class_dof():
    user = _empty_user(sample_count=2)

    DefenseController._update_www_feature_statistics(
        user=user,
        encoded_features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        labels=torch.tensor([0, 1]),
        sample_indices=torch.tensor([0, 1]),
        num_classes=2,
    )

    assert user.www_within_class_covariance_dof == 0
    assert torch.equal(
        user.www_within_class_covariance,
        torch.zeros((2, 2), dtype=torch.float64),
    )


def test_www_initialization_uses_complete_local_dataset_once():
    user = _empty_user(sample_count=4)
    user.id = 2
    user.model = _FeatureModel()
    batches = [
        (
            torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
        ),
        (
            torch.tensor([[0.0, 2.0], [0.0, 4.0]]),
            torch.tensor([1, 1]),
            torch.tensor([2, 3]),
        ),
    ]
    user.iter_www_statistics_batches = lambda: iter(batches)
    controller = DefenseController(
        config={"name": "www", "www_feature_statistics": True,
                "release_private_diagnostics": True},
        device=torch.device("cpu"),
        total_users=3,
        num_classes=2,
        total_rounds=1,
    )

    controller.initialize_www_feature_statistics([user])

    assert torch.equal(user.www_class_feature_counts, torch.tensor([2, 2]))
    assert torch.allclose(
        user.www_class_feature_means,
        torch.tensor([[4.0, 0.0], [0.0, 6.0]], dtype=torch.float64),
    )
    assert controller.summary()["www"]["feature_statistics"][
        "computation_stage"
    ] == "before_federated_training"


def test_post_round_www_runs_only_at_configured_completed_round(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    user = _ranking_user(model)
    controller = DefenseController(
        config={
            "name": "www",
            "www_analysis_interval": 50,
            "www_analysis_timing": "post_round",
            "www_feature_statistics": False,
            "release_private_diagnostics": True,
        },
        device=torch.device("cpu"),
        total_users=2,
        num_classes=2,
        total_rounds=100,
    )
    controller.federated_method = "fedsgd"
    own_state = {"weight": torch.tensor([[2.0, 0.0], [0.0, 2.0]])}
    other_state = {"weight": torch.tensor([[0.0, 1.0], [1.0, 0.0]])}
    global_state = {
        "weight": (own_state["weight"] + other_state["weight"]) / 2
    }
    arguments = {
        "users": [user],
        "global_state": global_state,
        "updated_states": {0: own_state},
        "aggregation_weights": {0: 0.5},
        "selected_ids": [0],
    }

    controller.analyze_www_completed_round(round_index=48, **arguments)
    assert controller._www_round_metrics == []

    controller.analyze_www_completed_round(round_index=49, **arguments)

    assert len(controller._www_round_metrics) == 1
    assert controller._www_round_metrics[0]["communication_round"] == 50
    assert torch.equal(user.www_local_indices, torch.tensor([2, 6]))
    assert user.www_score_count[2] == 1
    assert user.www_score_count[6] == 1
    summary = controller.save_summary(str(tmp_path))
    assert summary["www"]["completed_rounds"] == [50]
    assert summary["www"]["scheduled_rounds"] == [50, 100]
    assert (tmp_path / "www_round_metrics.csv").is_file()
    assert (tmp_path / "www_round_samples.csv").is_file()
    assert (tmp_path / "www_series.json").is_file()

    nonmember_scores = [0.7] + [0.3] * 10 + [-0.1] * 989
    projres_payload = {
        "communication_round": 50,
        "result": {
            "client_id": 0,
            "dimensions": {"member_candidate_count": 2},
            "candidate_controls": {
                "member_batch_positions": [0, 1],
                "member_local_indices": [2, 6],
                "member_labels": [0, 1],
            },
            "raw": {
                "labels": [1, 1] + [0] * 1000,
                "scores": [0.8, 0.2] + nonmember_scores,
                "l1_residuals": [0.2, 0.8]
                + [-score for score in nonmember_scores],
                "predictions": None,
            },
        },
    }
    relationship_dir = tmp_path / "privacy_audit"
    assert controller.record_www_projres_relationship(
        projres_payload,
        str(relationship_dir),
        round_index=49,
    )
    assert (relationship_dir / "www_projres_samples.csv").is_file()
    assert (relationship_dir / "www_projres_relationship.csv").is_file()
    assert (relationship_dir / "www_projres_relationship.json").is_file()
    relationship = controller._www_projres_metrics[0]
    assert relationship["projres_nonmember_samples"] == 1000
    assert relationship["projres_hit_rate_fpr_0.1"] == pytest.approx(1.0)
    assert relationship["projres_hit_rate_fpr_0.01"] == pytest.approx(0.5)
    assert "projres_hit_rate_fpr_0.001" not in relationship
    assert "projres_hit_rate_www_top_fpr_0.001" not in relationship
    assert "projres_hit_rate_www_bottom_fpr_0.001" not in relationship
    assert "projres_member_hit_rate" not in relationship
    assert "projres_fixed_threshold_prediction" not in (
        controller._www_projres_samples[0]
    )

    mismatched = dict(projres_payload)
    mismatched["communication_round"] = 100
    with pytest.raises(ValueError, match="communication rounds do not match"):
        controller.record_www_projres_relationship(
            mismatched,
            str(relationship_dir),
            round_index=49,
        )
