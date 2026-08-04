"""Strict Projection Residual (ProjRes) attack for a trainable MLP layer.

For a PyTorch ``Linear(input_dim, output_dim)`` layer and one FedSGD batch,
the uploaded weight gradient has shape ``(output_dim, input_dim)`` and equals
``Delta.T @ X``.  ProjRes therefore projects candidate rows of ``X`` onto the
row space of that weight gradient and uses the raw L1 reconstruction residual
as its membership statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from privacy_attacks.projres_promptfl import (
    projection_statistics,
    row_subspace_basis,
)


@dataclass(frozen=True)
class MLPProjResResult:
    """Outputs of the paper-faithful projection-residual decision rule."""

    scores: torch.Tensor
    l1_residuals: torch.Tensor
    predictions: torch.Tensor
    basis: torch.Tensor
    statistics: dict[str, torch.Tensor]
    metadata: dict[str, object]


@dataclass(frozen=True)
class FedSGDStepResult:
    """Observed first-layer update from exactly one vanilla SGD step."""

    parameter_name: str
    loss: float
    gradient: torch.Tensor
    observed_update: torch.Tensor
    update_gradient_relative_error: float


def strict_mlp_projres(
    first_layer_update: torch.Tensor,
    candidate_layer_inputs: torch.Tensor,
    threshold: float = 1e-2,
    *,
    max_rank: int | None = None,
    relative_tolerance: float | None = None,
    absolute_tolerance: float = 0.0,
) -> MLPProjResResult:
    """Run Algorithm 1 of ProjRes on a first-layer MLP weight update.

    Args:
        first_layer_update: Server-observed ``W_before - W_after`` for the
            attacked ``Linear`` layer, shaped ``(output_dim, input_dim)``.
            Multiplying this matrix by a non-zero scalar does not change the
            recovered subspace, so a FedSGD update and its gradient are
            equivalent here.
        candidate_layer_inputs: Candidate representations immediately before
            that layer, shaped ``(candidate_count, input_dim)``.
        threshold: The paper's raw L1 residual threshold. A candidate is
            predicted as a member exactly when ``residual < threshold``.
        max_rank: Optional theoretical upper bound on the observed gradient
            rank. For one-batch FedSGD this is at most the actual batch size.
            Applying this bound removes only finite-precision directions that
            cannot occur in the exact batch gradient.

    Returns:
        Scores use ``-L1 residual`` so larger values consistently mean more
        likely member for ROC/AUC evaluation. The unmodified paper statistic
        is available as ``l1_residuals``.
    """
    if threshold < 0 or not torch.isfinite(torch.tensor(float(threshold))):
        raise ValueError("threshold must be finite and non-negative.")
    if max_rank is not None and max_rank <= 0:
        raise ValueError("max_rank must be positive when provided.")
    if first_layer_update.ndim != 2:
        raise ValueError("first_layer_update must be a matrix.")
    if candidate_layer_inputs.ndim != 2:
        raise ValueError("candidate_layer_inputs must be a matrix.")
    if first_layer_update.shape[1] != candidate_layer_inputs.shape[1]:
        raise ValueError(
            "The update width must match the candidate layer-input width."
        )
    if candidate_layer_inputs.shape[0] == 0:
        raise ValueError("At least one candidate is required.")
    if first_layer_update.device != candidate_layer_inputs.device:
        raise ValueError("The update and candidates must use the same device.")
    if first_layer_update.dtype != candidate_layer_inputs.dtype:
        raise ValueError("The update and candidates must use the same dtype.")
    if not torch.isfinite(first_layer_update).all():
        raise ValueError("first_layer_update must contain only finite values.")
    if not torch.isfinite(candidate_layer_inputs).all():
        raise ValueError(
            "candidate_layer_inputs must contain only finite values."
        )
    if not bool(torch.count_nonzero(first_layer_update)):
        raise ValueError("A zero update has no informative row subspace.")

    # The server-observed float32 parameter difference can contain tiny
    # full-rank cancellation noise even though a batch gradient has rank no
    # larger than the batch size. Compute the same SVD row space in float64 on
    # CPU, retaining the original-dtype rank tolerance, and re-orthogonalize
    # the selected vectors. QR changes only the basis coordinates, not the
    # represented subspace or ProjRes decision rule.
    original_dtype = first_layer_update.dtype
    effective_relative_tolerance = relative_tolerance
    if effective_relative_tolerance is None:
        effective_relative_tolerance = (
            max(first_layer_update.shape) * torch.finfo(original_dtype).eps
        )
    stable_update = first_layer_update.detach().to(
        device="cpu", dtype=torch.float64
    )
    basis, rank_metadata = row_subspace_basis(
        stable_update,
        max_rank=max_rank,
        relative_tolerance=effective_relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if basis.shape[1]:
        pre_qr_gram = basis.t() @ basis
        identity = torch.eye(basis.shape[1], dtype=basis.dtype)
        pre_qr_error = float((pre_qr_gram - identity).abs().max())
        basis = torch.linalg.qr(basis, mode="reduced").Q.contiguous()
        post_qr_error = float((basis.t() @ basis - identity).abs().max())
        qr_reorthogonalized = True
    else:
        pre_qr_error = 0.0
        post_qr_error = 0.0
        qr_reorthogonalized = False
    stable_basis = basis.to(candidate_layer_inputs.device)
    stable_candidates = candidate_layer_inputs.detach().to(torch.float64)
    stable_statistics = projection_statistics(stable_candidates, stable_basis)
    statistics = {
        name: value.to(dtype=original_dtype)
        for name, value in stable_statistics.items()
    }
    residuals = statistics["l1_residual"]
    predictions = (residuals < float(threshold)).to(torch.long)
    metadata: dict[str, object] = {
        "algorithm": "ProjRes Algorithm 1",
        "attacked_parameter": "first_trainable_linear_weight",
        "update_shape": list(first_layer_update.shape),
        "candidate_shape": list(candidate_layer_inputs.shape),
        "input_dimension": int(first_layer_update.shape[1]),
        "output_dimension": int(first_layer_update.shape[0]),
        "threshold": float(threshold),
        "decision_rule": "member iff raw_l1_residual < threshold",
        "roc_score": "negative_raw_l1_residual",
        "subspace": rank_metadata,
        "numerical_stabilization": {
            "input_dtype": str(original_dtype),
            "subspace_device": "cpu",
            "subspace_dtype": "torch.float64",
            "projection_dtype": "torch.float64",
            "rank_cap": None if max_rank is None else int(max_rank),
            "qr_reorthogonalized": qr_reorthogonalized,
            "pre_qr_max_gram_error": pre_qr_error,
            "post_qr_max_gram_error": post_qr_error,
        },
    }
    return MLPProjResResult(
        scores=-residuals,
        l1_residuals=residuals,
        predictions=predictions,
        basis=stable_basis,
        statistics=statistics,
        metadata=metadata,
    )


def one_batch_fedsgd_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
    parameter_name: str = "classifier.0.weight",
) -> FedSGDStepResult:
    """Execute the strict one-batch, one-step FedSGD observation model.

    This deliberately uses vanilla SGD with no momentum, weight decay, local
    epochs, clipping, or noise. Only parameters already marked trainable are
    optimized. The returned update follows the server convention
    ``W_before - W_after``.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    named_parameters = dict(model.named_parameters())
    if parameter_name not in named_parameters:
        raise ValueError(f"Unknown parameter {parameter_name!r}.")
    attacked = named_parameters[parameter_name]
    if not attacked.requires_grad or attacked.ndim != 2:
        raise ValueError("The attacked parameter must be a trainable matrix.")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The model has no trainable parameters.")

    optimizer = torch.optim.SGD(trainable, lr=float(learning_rate))
    before = attacked.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels.to(logits.device, dtype=torch.long))
    loss.backward()
    if attacked.grad is None:
        raise RuntimeError("The attacked parameter did not receive a gradient.")
    gradient = attacked.grad.detach().clone()
    optimizer.step()
    observed_update = before - attacked.detach()
    expected_update = gradient * float(learning_rate)
    relative_error = float(
        (observed_update - expected_update).norm()
        / expected_update.norm().clamp_min(1e-12)
    )
    return FedSGDStepResult(
        parameter_name=parameter_name,
        loss=float(loss.detach()),
        gradient=gradient,
        observed_update=observed_update,
        update_gradient_relative_error=relative_error,
    )
