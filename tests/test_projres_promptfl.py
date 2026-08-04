from __future__ import annotations

import torch
import torch.nn.functional as F

from privacy_attacks.projres_promptfl import (
    dense_text_feature_jacobian,
    numerical_rank,
    principal_angles,
    projection_statistics,
    prompt_gradient_fingerprints,
    prompt_vjp,
    ridge_lift_dense,
    ridge_lift_matrix_free,
    row_subspace_basis,
    text_feature_change_subspace,
    text_feature_gradient,
)
from scripts.validate_projres_promptfl_real import (
    _aggregate_client_results,
    _collect_label_matched_nonmembers,
)


def test_text_feature_gradient_matches_autograd_and_zero_sum_errors():
    torch.manual_seed(3)
    text = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    images = torch.randn(4, 7, dtype=torch.float64)
    labels = torch.tensor([0, 2, 1, 4])
    scale = 2.3
    loss = F.cross_entropy(scale * images @ text.t(), labels)
    expected = torch.autograd.grad(loss, text)[0]

    actual, errors = text_feature_gradient(
        text.detach(), images, labels, logit_scale=scale
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        errors.sum(dim=1), torch.zeros(4, dtype=errors.dtype), atol=1e-12
    )


def test_gradient_row_space_recovers_member_feature_space_below_rank_boundary():
    torch.manual_seed(7)
    batch, classes, dimension = 3, 6, 8
    text = torch.randn(classes, dimension, dtype=torch.float64)
    members = torch.randn(batch, dimension, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2])
    gradient, errors = text_feature_gradient(text, members, labels)

    gradient_basis, gradient_metadata = row_subspace_basis(gradient)
    member_basis, member_metadata = row_subspace_basis(members)
    angles = principal_angles(gradient_basis, member_basis)
    statistics = projection_statistics(members, gradient_basis)

    assert numerical_rank(errors)[0] == batch
    assert gradient_metadata["numerical_rank"] == batch
    assert member_metadata["numerical_rank"] == batch
    assert float(angles.max()) < 1e-7
    assert torch.all(statistics["l2_residual"] < 1e-9)
    assert torch.allclose(
        statistics["projection_energy"], torch.ones(batch, dtype=torch.float64)
    )


def test_full_ambient_row_space_cannot_distinguish_nonmembers():
    dimension = 4
    full_basis = torch.eye(dimension)
    candidates = torch.randn(10, dimension)

    statistics = projection_statistics(candidates, full_basis)

    assert torch.allclose(
        statistics["projection_energy"], torch.ones(10), atol=1e-6
    )
    assert torch.all(statistics["l2_residual"] < 1e-6)


def test_text_feature_change_subspace_centers_endpoint_delta_and_caps_rank():
    torch.manual_seed(9)
    before = torch.randn(6, 8, dtype=torch.float64)
    member_rows = torch.randn(2, 8, dtype=torch.float64)
    coefficients = torch.randn(6, 2, dtype=torch.float64)
    coefficients = coefficients - coefficients.mean(dim=0, keepdim=True)
    common_drift = torch.randn(1, 8, dtype=torch.float64)
    after = before + coefficients @ member_rows
    after = after + common_drift

    change, basis, metadata = text_feature_change_subspace(
        before, after, max_rank=2
    )
    statistics = projection_statistics(member_rows, basis)

    assert torch.allclose(
        change.sum(dim=0), torch.zeros(8, dtype=torch.float64), atol=1e-12
    )
    assert metadata["used_rank"] == 2
    assert metadata["center_rows"] is True
    assert metadata["raw_numerical_rank"] > metadata["numerical_rank"]
    assert metadata["raw_frobenius_norm"] > metadata["frobenius_norm"]
    assert metadata["removed_common_mode_fraction"] > 0
    assert metadata["frobenius_norm"] > 0
    assert metadata["relative_frobenius_change"] > 0
    assert torch.allclose(
        statistics["projection_energy"],
        torch.ones(2, dtype=torch.float64),
        atol=1e-10,
    )


def test_dense_and_matrix_free_ridge_lifts_agree():
    torch.manual_seed(11)
    classes, dimension = 3, 4
    prompt_shape = (2, 3)
    output_width = classes * dimension
    prompt_width = prompt_shape[0] * prompt_shape[1]
    mapping = torch.randn(
        output_width, prompt_width, dtype=torch.float64
    ) / output_width**0.5
    offset = torch.randn(classes, dimension, dtype=torch.float64)

    def feature_function(prompt: torch.Tensor) -> torch.Tensor:
        return offset + (mapping @ prompt.reshape(-1)).reshape(classes, dimension)

    prompt = torch.randn(prompt_shape, dtype=torch.float64)
    true_text_gradient = torch.randn(classes, dimension, dtype=torch.float64)
    measured = prompt_vjp(feature_function, prompt, true_text_gradient)
    jacobian, output_shape = dense_text_feature_jacobian(
        feature_function, prompt, max_elements=1_000
    )

    dense_lift, dense_diagnostics = ridge_lift_dense(
        jacobian, measured, output_shape, ridge=1e-3
    )
    free_lift, free_diagnostics = ridge_lift_matrix_free(
        feature_function,
        prompt,
        measured,
        ridge=1e-3,
        max_iterations=30,
        tolerance=1e-10,
    )

    assert torch.allclose(free_lift, dense_lift, atol=1e-8, rtol=1e-7)
    assert free_diagnostics.converged
    assert free_diagnostics.normal_equation_relative_residual < 1e-8
    assert abs(
        free_diagnostics.measurement_relative_residual
        - dense_diagnostics.measurement_relative_residual
    ) < 1e-8


def test_prompt_fingerprints_match_independent_candidate_gradients():
    torch.manual_seed(17)
    classes, dimension = 4, 5
    prompt = torch.randn(2, 3, dtype=torch.float64)
    mapping = torch.randn(classes * dimension, prompt.numel(), dtype=torch.float64)
    offset = torch.randn(classes, dimension, dtype=torch.float64)

    def feature_function(value: torch.Tensor) -> torch.Tensor:
        return offset + (mapping @ value.flatten()).reshape(classes, dimension)

    images = torch.randn(3, dimension, dtype=torch.float64)
    labels = torch.tensor([0, 2, 1])
    fingerprints, _, losses = prompt_gradient_fingerprints(
        feature_function, prompt, images, labels, logit_scale=1.7
    )

    expected = []
    for image, label in zip(images, labels):
        candidate_prompt = prompt.detach().clone().requires_grad_(True)
        text = feature_function(candidate_prompt)
        loss = F.cross_entropy(
            1.7 * image.unsqueeze(0) @ text.t(), label.view(1)
        )
        expected.append(torch.autograd.grad(loss, candidate_prompt)[0].flatten())

    assert torch.allclose(fingerprints, torch.stack(expected), atol=1e-10)
    assert losses.shape == (3,)


def test_dense_jacobian_guard_rejects_unsafe_materialization():
    prompt = torch.zeros(4)

    def feature_function(value: torch.Tensor) -> torch.Tensor:
        return value.repeat(3).reshape(3, 4)

    try:
        dense_text_feature_jacobian(
            feature_function, prompt, max_elements=10
        )
    except ValueError as error:
        assert "exceeding max_elements" in str(error)
    else:
        raise AssertionError("Dense Jacobian safety guard did not trigger.")


def test_all_client_aggregation_reports_pooled_and_macro_metrics():
    def client_result(client_id: int, member_scores, nonmember_scores):
        scores = member_scores + nonmember_scores
        labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
        metrics = {
            "auc": 1.0,
            "tpr_at_fpr_0.1": 1.0,
            "tpr_at_fpr_0.01": 1.0,
            "tpr_at_fpr_0.001": 1.0,
            "member_mean_score": sum(member_scores) / len(member_scores),
            "nonmember_mean_score": sum(nonmember_scores) / len(nonmember_scores),
            "score_gap": (
                sum(member_scores) / len(member_scores)
                - sum(nonmember_scores) / len(nonmember_scores)
            ),
        }
        return {
            "config": {"target_client": client_id, "dataset_name": "toy"},
            "attacks": {
                "oracle_projres_t": metrics,
                "delta_text_projres": metrics,
                "lifted_projres_p": None,
                "direct_prompt_atom": metrics,
            },
            "raw_scores": {
                "labels": labels,
                "oracle_projres_t": scores,
                "delta_text_projres": scores,
                "lifted_projres_p": None,
                "direct_prompt_atom": scores,
            },
        }

    summary = _aggregate_client_results(
        [
            client_result(0, [0.9, 0.8], [0.2, 0.1]),
            client_result(1, [0.7, 0.6], [0.4, 0.3]),
        ]
    )

    assert summary["client_ids"] == [0, 1]
    assert summary["client_count"] == 2
    assert summary["pooled_attacks"]["oracle_projres_t"]["auc"] == 1.0
    assert summary["pooled_attacks"]["delta_text_projres"]["auc"] == 1.0
    assert summary["client_macro_attacks"]["direct_prompt_atom"]["auc"] == 1.0
    assert summary["pooled_attacks"]["lifted_projres_p"] is None


def test_label_matched_nonmembers_fall_back_to_other_clients():
    independent_test = [
        ("test-class-zero", 0),
        ("test-class-one", 1),
    ]
    other_client_train = [
        ("other-class-zero", 0),
        ("unused-class-two", 2),
    ]

    images, labels, source_counts = _collect_label_matched_nonmembers(
        independent_test,
        other_client_train,
        torch.tensor([0, 0, 1]),
    )

    assert images == ["test-class-zero", "other-class-zero", "test-class-one"]
    assert labels.tolist() == [0, 0, 1]
    assert source_counts == {"independent_test": 2, "other_client_train": 1}
