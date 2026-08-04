from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RidgeLiftDiagnostics:
    iterations: int
    converged: bool
    normal_equation_relative_residual: float
    measurement_relative_residual: float
    prompt_gradient_norm: float
    lifted_text_gradient_norm: float


def _require_finite_matrix(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")


def text_feature_gradient(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``d CE(scale * X @ T.T) / dT`` and its softmax errors.

    ``text_features`` has shape ``(classes, dimension)`` and
    ``image_features`` has shape ``(batch, dimension)``. The returned gradient
    is ``scale * E.T @ X / batch``, with shape ``(classes, dimension)``.
    """
    _require_finite_matrix("text_features", text_features)
    _require_finite_matrix("image_features", image_features)
    if text_features.shape[1] != image_features.shape[1]:
        raise ValueError("Text and image feature dimensions must match.")
    if text_features.device != image_features.device:
        raise ValueError("Text and image features must use the same device.")
    labels = labels.detach().to(
        device=image_features.device, dtype=torch.long
    ).flatten()
    if labels.numel() != image_features.shape[0] or labels.numel() == 0:
        raise ValueError("Every non-empty image-feature row needs one label.")
    if int(labels.min()) < 0 or int(labels.max()) >= text_features.shape[0]:
        raise ValueError("Labels must index rows of the text-feature matrix.")
    if not math.isfinite(float(logit_scale)) or float(logit_scale) <= 0:
        raise ValueError("logit_scale must be positive and finite.")

    logits = float(logit_scale) * image_features @ text_features.t()
    errors = torch.softmax(logits, dim=1)
    errors = errors - F.one_hot(
        labels, num_classes=text_features.shape[0]
    ).to(dtype=errors.dtype)
    gradient = (
        float(logit_scale)
        * errors.t()
        @ image_features
        / image_features.shape[0]
    )
    return gradient, errors


def numerical_rank(
    matrix: torch.Tensor,
    relative_tolerance: float | None = None,
    absolute_tolerance: float = 0.0,
) -> tuple[int, torch.Tensor, float]:
    """Return numerical rank, singular values, and the applied tolerance."""
    _require_finite_matrix("matrix", matrix)
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative.")
    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.numel() == 0 or float(singular_values.max()) == 0.0:
        return 0, singular_values, float(absolute_tolerance)
    if relative_tolerance is None:
        relative_tolerance = (
            max(matrix.shape) * torch.finfo(matrix.dtype).eps
        )
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative.")
    tolerance = max(
        float(absolute_tolerance),
        float(relative_tolerance) * float(singular_values.max()),
    )
    rank = int((singular_values > tolerance).sum().item())
    return rank, singular_values, tolerance


def row_subspace_basis(
    matrix: torch.Tensor,
    max_rank: int | None = None,
    relative_tolerance: float | None = None,
    absolute_tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return an orthonormal column basis for the matrix row space."""
    _require_finite_matrix("matrix", matrix)
    if max_rank is not None and max_rank < 0:
        raise ValueError("max_rank must be non-negative when provided.")
    _, singular_values, right_vectors = torch.linalg.svd(
        matrix, full_matrices=False
    )
    rank, _, tolerance = numerical_rank(
        matrix,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    used_rank = rank if max_rank is None else min(rank, int(max_rank))
    basis = right_vectors[:used_rank].t().contiguous()
    metadata: dict[str, object] = {
        "shape": list(matrix.shape),
        "numerical_rank": rank,
        "used_rank": used_rank,
        "tolerance": tolerance,
        "singular_values": singular_values.detach().cpu().tolist(),
    }
    return basis, metadata


def text_feature_change_subspace(
    before: torch.Tensor,
    after: torch.Tensor,
    max_rank: int,
    center_rows: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Construct a rank-truncated row subspace from an endpoint text change.

    In prompt tuning, ``after - before`` is not the text-feature gradient.  It
    is nevertheless an attacker-observable proxy with the same matrix shape,
    because both endpoint text matrices can be recomputed from the shared
    prompts and frozen text encoder.  When ``center_rows`` is enabled, the raw
    endpoint change is projected onto the zero-row-sum class subspace required
    by cross-entropy text gradients, because each softmax-error row sums to
    zero.  Truncating its row space prevents small nonlinear endpoint changes
    from producing a nearly full-rank projection space that trivially contains
    members and non-members alike.
    """
    _require_finite_matrix("before", before)
    _require_finite_matrix("after", after)
    if before.shape != after.shape:
        raise ValueError("before and after text features must have equal shape.")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive.")
    raw_change = after.detach() - before.detach()
    change = raw_change
    if center_rows:
        change = raw_change - raw_change.mean(dim=0, keepdim=True)
    basis, metadata = row_subspace_basis(change, max_rank=max_rank)
    before_norm = before.detach().norm().clamp_min(1e-12)
    raw_rank, raw_singular_values, raw_tolerance = numerical_rank(raw_change)
    raw_norm = raw_change.norm().clamp_min(1e-12)
    removed_common_mode = raw_change - change
    metadata["raw_numerical_rank"] = raw_rank
    metadata["raw_tolerance"] = raw_tolerance
    metadata["raw_singular_values"] = raw_singular_values.detach().cpu().tolist()
    metadata["raw_frobenius_norm"] = float(raw_change.norm())
    metadata["frobenius_norm"] = float(change.norm())
    metadata["relative_frobenius_change"] = float(change.norm() / before_norm)
    metadata["raw_relative_frobenius_change"] = float(
        raw_change.norm() / before_norm
    )
    metadata["center_rows"] = bool(center_rows)
    metadata["removed_common_mode_fraction"] = float(
        removed_common_mode.norm() / raw_norm
    )
    return change, basis, metadata


def projection_statistics(
    features: torch.Tensor,
    basis: torch.Tensor,
    epsilon: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compute ProjRes residuals and higher-is-member projection scores."""
    _require_finite_matrix("features", features)
    _require_finite_matrix("basis", basis)
    if basis.shape[0] != features.shape[1]:
        raise ValueError("The basis ambient dimension must match feature width.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if basis.shape[1]:
        gram = basis.t() @ basis
        identity = torch.eye(
            basis.shape[1], device=basis.device, dtype=basis.dtype
        )
        if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-4):
            raise ValueError("basis columns must be orthonormal.")
        projection = (features @ basis) @ basis.t()
    else:
        projection = torch.zeros_like(features)
    residual = features - projection
    feature_l1 = features.norm(p=1, dim=1).clamp_min(epsilon)
    feature_l2_squared = features.square().sum(dim=1).clamp_min(epsilon)
    projection_energy = projection.square().sum(dim=1) / feature_l2_squared
    return {
        "l1_residual": residual.norm(p=1, dim=1),
        "relative_l1_residual": residual.norm(p=1, dim=1) / feature_l1,
        "l2_residual": residual.norm(p=2, dim=1),
        "relative_l2_residual": residual.norm(p=2, dim=1)
        / feature_l2_squared.sqrt(),
        "projection_energy": projection_energy.clamp(0.0, 1.0),
    }


def principal_angles(
    first_basis: torch.Tensor,
    second_basis: torch.Tensor,
) -> torch.Tensor:
    """Return principal angles in radians between two orthonormal bases."""
    _require_finite_matrix("first_basis", first_basis)
    _require_finite_matrix("second_basis", second_basis)
    if first_basis.shape[0] != second_basis.shape[0]:
        raise ValueError("Subspaces must have the same ambient dimension.")
    if first_basis.shape[1] == 0 or second_basis.shape[1] == 0:
        return torch.empty(
            0, device=first_basis.device, dtype=first_basis.dtype
        )
    cosines = torch.linalg.svdvals(first_basis.t() @ second_basis)
    return torch.acos(cosines.clamp(-1.0, 1.0))


def dense_text_feature_jacobian(
    text_feature_function: TensorFunction,
    prompt: torch.Tensor,
    max_elements: int = 20_000_000,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Materialize ``d vec(T) / d vec(P)`` for small diagnostics only."""
    if max_elements <= 0:
        raise ValueError("max_elements must be positive.")
    prompt_value = prompt.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        output = text_feature_function(prompt_value)
    if output.ndim != 2:
        raise ValueError("text_feature_function must return a matrix.")
    elements = output.numel() * prompt_value.numel()
    if elements > max_elements:
        raise ValueError(
            "Dense Jacobian would contain "
            f"{elements} elements, exceeding max_elements={max_elements}."
        )
    jacobian = torch.autograd.functional.jacobian(
        lambda value: text_feature_function(value).reshape(-1),
        prompt_value,
        vectorize=True,
    ).reshape(output.numel(), prompt_value.numel())
    return jacobian.detach(), tuple(output.shape)


def ridge_lift_dense(
    jacobian: torch.Tensor,
    prompt_gradient: torch.Tensor,
    output_shape: tuple[int, int],
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, RidgeLiftDiagnostics]:
    """Lift a prompt gradient through a materialized text-feature Jacobian."""
    _require_finite_matrix("jacobian", jacobian)
    if ridge <= 0:
        raise ValueError("ridge must be positive.")
    gradient = prompt_gradient.detach().reshape(-1).to(
        device=jacobian.device, dtype=jacobian.dtype
    )
    if gradient.numel() != jacobian.shape[1]:
        raise ValueError("prompt_gradient width must match Jacobian columns.")
    if math.prod(output_shape) != jacobian.shape[0]:
        raise ValueError("output_shape must match Jacobian rows.")
    normal = jacobian.t() @ jacobian
    normal.diagonal().add_(float(ridge))
    coefficient = torch.linalg.solve(normal, gradient)
    lifted = jacobian @ coefficient
    predicted = jacobian.t() @ lifted
    normal_residual = predicted + float(ridge) * coefficient - gradient
    scale = gradient.norm().clamp_min(1e-12)
    diagnostics = RidgeLiftDiagnostics(
        iterations=1,
        converged=True,
        normal_equation_relative_residual=float(normal_residual.norm() / scale),
        measurement_relative_residual=float((predicted - gradient).norm() / scale),
        prompt_gradient_norm=float(gradient.norm()),
        lifted_text_gradient_norm=float(lifted.norm()),
    )
    return lifted.reshape(output_shape), diagnostics


def _conjugate_gradient(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    right_hand_side: torch.Tensor,
    max_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, int, bool, float]:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_squared = torch.dot(residual.flatten(), residual.flatten())
    initial_norm = residual_squared.sqrt().clamp_min(1e-30)
    if float(initial_norm) == 0.0:
        return solution, 0, True, 0.0
    relative_residual = 1.0
    for iteration in range(1, max_iterations + 1):
        product = matrix_vector_product(direction)
        denominator = torch.dot(direction.flatten(), product.flatten())
        if not torch.isfinite(denominator) or float(denominator) <= 0.0:
            return solution, iteration - 1, False, relative_residual
        step = residual_squared / denominator
        solution = solution + step * direction
        residual = residual - step * product
        next_residual_squared = torch.dot(residual.flatten(), residual.flatten())
        relative_residual = float(next_residual_squared.sqrt() / initial_norm)
        if relative_residual <= tolerance:
            return solution, iteration, True, relative_residual
        direction = residual + (next_residual_squared / residual_squared) * direction
        residual_squared = next_residual_squared
    return solution, max_iterations, False, relative_residual


def ridge_lift_matrix_free(
    text_feature_function: TensorFunction,
    prompt: torch.Tensor,
    prompt_gradient: torch.Tensor,
    ridge: float = 1e-4,
    max_iterations: int = 20,
    tolerance: float = 1e-5,
) -> tuple[torch.Tensor, RidgeLiftDiagnostics]:
    """Lift through a Jacobian without materializing it.

    This solves ``(J.T J + ridge I) v = g`` with conjugate gradients and
    returns ``reshape(J v)``. The implementation uses matrix-free JVP/VJP
    operations from ``torch.func`` and therefore scales to PromptFL Jacobians
    that are too large to store explicitly.
    """
    if ridge <= 0:
        raise ValueError("ridge must be positive.")
    prompt_value = prompt.detach().clone()

    def flattened(value: torch.Tensor) -> torch.Tensor:
        output = text_feature_function(value)
        if output.ndim != 2:
            raise ValueError("text_feature_function must return a matrix.")
        return output.reshape(-1)

    with torch.enable_grad():
        output, vjp_function = torch.func.vjp(flattened, prompt_value)
    output_shape = tuple(text_feature_function(prompt_value).shape)
    gradient = prompt_gradient.detach().reshape_as(prompt_value).to(
        device=prompt_value.device, dtype=prompt_value.dtype
    )

    def jvp_product(vector: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            return torch.func.jvp(
                flattened, (prompt_value,), (vector,)
            )[1]

    def normal_product(vector: torch.Tensor) -> torch.Tensor:
        jvp = jvp_product(vector)
        vjp = vjp_function(jvp)[0]
        return vjp + float(ridge) * vector

    coefficient, iterations, converged, normal_relative = _conjugate_gradient(
        normal_product,
        gradient,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    lifted_flat = jvp_product(coefficient)
    predicted = vjp_function(lifted_flat)[0]
    gradient_norm = gradient.norm().clamp_min(1e-12)
    measurement_relative = float((predicted - gradient).norm() / gradient_norm)
    diagnostics = RidgeLiftDiagnostics(
        iterations=iterations,
        converged=converged,
        normal_equation_relative_residual=normal_relative,
        measurement_relative_residual=measurement_relative,
        prompt_gradient_norm=float(gradient.norm()),
        lifted_text_gradient_norm=float(lifted_flat.norm()),
    )
    return lifted_flat.reshape(output_shape), diagnostics


def prompt_vjp(
    text_feature_function: TensorFunction,
    prompt: torch.Tensor,
    text_gradient: torch.Tensor,
) -> torch.Tensor:
    """Compute ``J.T vec(text_gradient)`` at a fixed prompt."""
    prompt_value = prompt.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        text_features = text_feature_function(prompt_value)
        if text_features.shape != text_gradient.shape:
            raise ValueError("text_gradient must match text-feature output shape.")
        scalar = (text_features * text_gradient.to(text_features)).sum()
        gradient = torch.autograd.grad(scalar, prompt_value)[0]
    return gradient.detach()


def prompt_gradient_fingerprints(
    text_feature_function: TensorFunction,
    prompt: torch.Tensor,
    image_features: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-candidate prompt gradients with one text-encoder forward."""
    _require_finite_matrix("image_features", image_features)
    labels = labels.detach().to(
        device=image_features.device, dtype=torch.long
    ).flatten()
    if labels.numel() != image_features.shape[0]:
        raise ValueError("Every image feature needs one label.")
    prompt_value = prompt.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        text_features = text_feature_function(prompt_value)
        if text_features.ndim != 2:
            raise ValueError("text_feature_function must return a matrix.")
        if text_features.shape[1] != image_features.shape[1]:
            raise ValueError("Text and image feature dimensions must match.")
        logits = float(logit_scale) * image_features @ text_features.t()
        losses = F.cross_entropy(logits, labels, reduction="none")
        rows = []
        for index, loss in enumerate(losses):
            gradient = torch.autograd.grad(
                loss,
                prompt_value,
                retain_graph=index + 1 < losses.numel(),
            )[0]
            rows.append(gradient.detach().reshape(-1))
    return (
        torch.stack(rows),
        text_features.detach(),
        losses.detach(),
    )
