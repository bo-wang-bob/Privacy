from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def text_feature_matrix(
    model: torch.nn.Module,
    normalize: bool = True,
) -> torch.Tensor:
    """Return one detached text-feature row per class."""
    getter = getattr(model, "get_text_features", None)
    if getter is None:
        raise TypeError(
            "Text-matrix FedMIA requires get_text_features(normalize=...)."
        )
    features = getter(normalize=normalize)
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(
            "Text-matrix FedMIA features must have shape "
            "(num_classes, feature_dim)."
        )
    if not torch.isfinite(features).all():
        raise ValueError("Text-matrix FedMIA features contain non-finite values.")
    return features.detach()


def frobenius_cosine_scores(
    client_change: torch.Tensor,
    candidate_changes: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Compare one client matrix change with candidate matrix changes."""
    client_change = client_change.detach()
    candidate_changes = candidate_changes.detach()
    if client_change.ndim != 2:
        raise ValueError("client_change must be a matrix.")
    if candidate_changes.ndim == 2:
        candidate_changes = candidate_changes.unsqueeze(0)
    if candidate_changes.ndim != 3:
        raise ValueError("candidate_changes must be a matrix or a batch of matrices.")
    if tuple(candidate_changes.shape[1:]) != tuple(client_change.shape):
        raise ValueError("Client and candidate text-feature matrices must match.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    client_flat = client_change.flatten()
    candidate_flat = candidate_changes.flatten(1)
    numerator = candidate_flat @ client_flat
    denominator = (
        candidate_flat.norm(dim=1) * client_flat.norm()
    ).clamp_min(epsilon)
    scores = numerator / denominator
    zero = (candidate_flat.norm(dim=1) <= epsilon) | (
        client_flat.norm() <= epsilon
    )
    return scores.masked_fill(zero, 0.0)


@torch.no_grad()
def direct_text_gradient_changes(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: float = 1.0,
    project_tangent: bool = False,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, int | float | bool | list[int]]]:
    """Return the text-matrix descent direction induced by each candidate.

    For ``logits = scale * image_features @ text_features.T`` and cross entropy,
    ``-dL/dT = -scale * (softmax(logits) - one_hot(label)) x.T``.  The optional
    tangent projection removes the radial component of every normalized class
    feature without differentiating through the text encoder.
    """
    text_features = text_features.detach()
    image_features = image_features.detach()
    labels = (
        labels.detach()
        .to(device=image_features.device, dtype=torch.long)
        .flatten()
    )
    if text_features.ndim != 2 or image_features.ndim != 2:
        raise ValueError("text_features and image_features must be matrices.")
    if text_features.shape[1] != image_features.shape[1]:
        raise ValueError("Text and image feature dimensions must match.")
    if text_features.device != image_features.device:
        raise ValueError("Text and image features must use the same device.")
    if image_features.shape[0] != labels.numel():
        raise ValueError("Every image feature needs one label.")
    if labels.numel() and (
        int(labels.min()) < 0 or int(labels.max()) >= text_features.shape[0]
    ):
        raise ValueError("Candidate labels must index the text-feature rows.")
    if not math.isfinite(float(logit_scale)) or float(logit_scale) <= 0:
        raise ValueError("logit_scale must be positive and finite.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not torch.isfinite(text_features).all() or not torch.isfinite(
        image_features
    ).all():
        raise ValueError("Direct text-gradient features must be finite.")

    scale = float(logit_scale)
    logits = scale * (image_features @ text_features.t())
    error = torch.softmax(logits, dim=1)
    if labels.numel():
        error[torch.arange(labels.numel(), device=labels.device), labels] -= 1.0
    directions = image_features.unsqueeze(1).expand(
        -1, text_features.shape[0], -1
    )
    if project_tangent:
        class_cosine = image_features @ text_features.t()
        directions = directions - (
            class_cosine.unsqueeze(-1) * text_features.unsqueeze(0)
        )
    changes = -scale * error.unsqueeze(-1) * directions
    change_norms = changes.flatten(1).norm(dim=1)
    metadata: dict[str, int | float | bool | list[int]] = {
        "text_feature_shape": list(text_features.shape),
        "logit_scale": scale,
        "project_tangent": bool(project_tangent),
        "zero_candidate_change_count": int(
            (change_norms <= epsilon).sum().item()
        ),
    }
    return changes, metadata


@torch.no_grad()
def direct_text_gradient_round_scores(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    labels: torch.Tensor,
    client_changes: torch.Tensor,
    logit_scale: float = 1.0,
    project_tangent: bool = False,
    candidate_batch_size: int = 64,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, int | float | bool | list[int]]]:
    """Compare client text-matrix changes with direct candidate gradients."""
    if client_changes.ndim == 2:
        client_changes = client_changes.unsqueeze(0)
    if client_changes.ndim != 3:
        raise ValueError("client_changes must be a matrix or a batch of matrices.")
    if tuple(client_changes.shape[1:]) != tuple(text_features.shape):
        raise ValueError("Client changes do not match the text-feature shape.")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive.")

    score_parts = []
    zero_candidate_change_count = 0
    metadata: dict[str, int | float | bool | list[int]] | None = None
    for start in range(0, labels.numel(), candidate_batch_size):
        stop = start + candidate_batch_size
        candidate_changes, batch_metadata = direct_text_gradient_changes(
            text_features,
            image_features[start:stop],
            labels[start:stop],
            logit_scale=logit_scale,
            project_tangent=project_tangent,
            epsilon=epsilon,
        )
        zero_candidate_change_count += int(
            batch_metadata["zero_candidate_change_count"]
        )
        metadata = batch_metadata
        score_parts.append(
            torch.stack(
                [
                    frobenius_cosine_scores(change, candidate_changes, epsilon)
                    for change in client_changes
                ]
            )
        )
    if metadata is None:
        metadata = {
            "text_feature_shape": list(text_features.shape),
            "logit_scale": float(logit_scale),
            "project_tangent": bool(project_tangent),
            "zero_candidate_change_count": 0,
        }
    metadata["zero_candidate_change_count"] = zero_candidate_change_count
    scores = (
        torch.cat(score_parts, dim=1)
        if score_parts
        else torch.empty((client_changes.shape[0], 0))
    )
    return scores, metadata


def _parameter_slices(
    model: torch.nn.Module,
    parameter_names: Iterable[str],
) -> tuple[list[tuple[str, torch.nn.Parameter, int, int]], int]:
    named_parameters = dict(model.named_parameters())
    slices = []
    offset = 0
    for name in parameter_names:
        parameter = named_parameters.get(name)
        if parameter is None or not parameter.requires_grad:
            raise ValueError(f"Observable trainable parameter is unavailable: {name}")
        stop = offset + parameter.numel()
        slices.append((name, parameter, offset, stop))
        offset = stop
    if not slices:
        raise ValueError("FedMIA-III needs an observable trainable prompt parameter.")
    return slices, offset


def _feature_contexts(
    model: torch.nn.Module,
    parameter_names: set[str],
) -> tuple[torch.Tensor, ...] | None:
    getter = getattr(model, "get_text_feature_contexts", None)
    if getter is None:
        return None
    contexts = getter()
    if isinstance(contexts, torch.Tensor):
        contexts = (contexts,)
    if not isinstance(contexts, tuple) or not contexts:
        raise ValueError("get_text_feature_contexts() must return prompt contexts.")
    if (
        str(getattr(model, "parameterization", "")).lower() == "fedotp"
        and not any(name.endswith("local_ctx") for name in parameter_names)
    ):
        # In the protocol-visible projection FedOTP's private branch is
        # neutralized by tying it to the communicated global prompt.
        contexts = (contexts[0], contexts[0])
    return tuple(context.detach().clone() for context in contexts)


def _encode_context_batches(
    model: torch.nn.Module,
    context_batches: list[list[torch.Tensor]],
) -> torch.Tensor | None:
    encoder = getattr(model, "get_text_features_for_context_batch", None)
    if encoder is None or not context_batches:
        return None
    encoded_parts = []
    for contexts in context_batches:
        encoded_parts.append(
            encoder(torch.stack(contexts), normalize=True).detach().cpu()
        )
    features = torch.stack(encoded_parts).mean(dim=0)
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _candidate_text_feature_probe(
    model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    parameter_names: list[str],
    candidate_gradients: torch.Tensor,
    client_changes: torch.Tensor | None,
    probe_norm: float = 1e-3,
    candidate_batch_size: int = 8,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float | list[int]]]:
    """Map candidate prompt-gradient directions into text-feature space.

    Each candidate receives a normalized virtual descent step. Models may expose
    batched context encoding to avoid one text-transformer launch per candidate;
    a generic state-based fallback keeps lightweight test models supported.
    """
    if candidate_gradients.ndim != 2:
        raise ValueError("candidate_gradients must have shape (samples, parameters).")
    if probe_norm <= 0 or epsilon <= 0:
        raise ValueError("probe_norm and epsilon must be positive.")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive.")

    slices, width = _parameter_slices(model, parameter_names)
    parameter_name_set = set(parameter_names)
    if candidate_gradients.shape[1] != width:
        raise ValueError(
            "Candidate gradient width does not match observable prompt parameters."
        )
    missing = [name for name, _, _, _ in slices if name not in base_state]
    if missing:
        raise ValueError(f"Base prompt state is missing parameters: {missing}")

    model.load_state_dict(base_state, strict=False)
    model.eval()
    base_features = text_feature_matrix(model, normalize=True).detach().cpu()
    base_values = {
        name: base_state[name].detach().to(parameter.device).clone()
        for name, parameter, _, _ in slices
    }
    outputs = []
    zero_gradient_count = 0
    zero_candidate_change_count = 0
    used_batched_contexts = False
    if client_changes is not None:
        if client_changes.ndim == 2:
            client_changes = client_changes.unsqueeze(0)
        if client_changes.ndim != 3:
            raise ValueError("client_changes must be a matrix or a batch of matrices.")
        if tuple(client_changes.shape[1:]) != tuple(base_features.shape):
            raise ValueError("Client changes do not match the text-feature shape.")
        client_changes = client_changes.detach().cpu()

    try:
        for start in range(0, candidate_gradients.shape[0], candidate_batch_size):
            rows = candidate_gradients[start : start + candidate_batch_size]
            context_batches: list[list[torch.Tensor]] = []
            fallback_features = []
            for row in rows:
                norm = row.norm()
                if float(norm) <= epsilon:
                    zero_gradient_count += 1
                    direction = torch.zeros_like(row)
                else:
                    direction = row / norm
                for name, parameter, begin, stop in slices:
                    update = direction[begin:stop].reshape_as(parameter).to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    parameter.copy_(base_values[name] - probe_norm * update)

                contexts = _feature_contexts(model, parameter_name_set)
                if contexts is None:
                    fallback_features.append(
                        text_feature_matrix(model, normalize=True).detach().cpu()
                    )
                else:
                    if not context_batches:
                        context_batches = [[] for _ in contexts]
                    if len(context_batches) != len(contexts):
                        raise AssertionError("Text-feature context count changed.")
                    for values, context in zip(context_batches, contexts):
                        values.append(context)

            for name, parameter, _, _ in slices:
                parameter.copy_(base_values[name])

            encoded = _encode_context_batches(model, context_batches)
            if encoded is not None:
                used_batched_contexts = True
                virtual_features = encoded
            elif fallback_features:
                virtual_features = torch.stack(fallback_features)
            else:
                raise TypeError(
                    "Model exposes text contexts but cannot batch-encode them."
                )
            batch_changes = virtual_features - base_features.unsqueeze(0)
            zero_candidate_change_count += int(
                (batch_changes.flatten(1).norm(dim=1) <= epsilon).sum().item()
            )
            if client_changes is None:
                outputs.append(batch_changes)
            else:
                outputs.append(
                    torch.stack(
                        [
                            frobenius_cosine_scores(change, batch_changes, epsilon)
                            for change in client_changes
                        ]
                    )
                )
    finally:
        for name, parameter, _, _ in slices:
            parameter.copy_(base_values[name])
        model.load_state_dict(base_state, strict=False)

    if outputs:
        output = torch.cat(outputs, dim=1 if client_changes is not None else 0)
    elif client_changes is None:
        output = torch.empty(
            (0, *base_features.shape), dtype=base_features.dtype
        )
    else:
        output = torch.empty(
            (client_changes.shape[0], 0), dtype=base_features.dtype
        )
    metadata: dict[str, int | float | list[int]] = {
        "probe_norm": float(probe_norm),
        "candidate_batch_size": int(candidate_batch_size),
        "text_feature_shape": list(base_features.shape),
        "zero_gradient_count": int(zero_gradient_count),
        "zero_candidate_change_count": int(zero_candidate_change_count),
        "batched_context_encoding": int(used_batched_contexts),
    }
    return base_features, output, metadata


def candidate_text_feature_changes(
    model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    parameter_names: list[str],
    candidate_gradients: torch.Tensor,
    probe_norm: float = 1e-3,
    candidate_batch_size: int = 8,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float | list[int]]]:
    """Return candidate matrix changes, primarily for focused diagnostics/tests."""
    return _candidate_text_feature_probe(
        model,
        base_state,
        parameter_names,
        candidate_gradients,
        None,
        probe_norm,
        candidate_batch_size,
        epsilon,
    )


def fedmia_text_round_scores(
    model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    parameter_names: list[str],
    candidate_gradients: torch.Tensor,
    client_changes: torch.Tensor,
    probe_norm: float = 1e-3,
    candidate_batch_size: int = 8,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, int | float | list[int]]]:
    """Score candidate batches immediately without retaining all feature matrices."""
    _, scores, metadata = _candidate_text_feature_probe(
        model,
        base_state,
        parameter_names,
        candidate_gradients,
        client_changes,
        probe_norm,
        candidate_batch_size,
        epsilon,
    )
    return scores, metadata
