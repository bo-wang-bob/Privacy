"""Model reconstruction and loss-difference ranking for the WWW defense."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


ModelState = dict[str, torch.Tensor]
TrainingBatch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class WWWRanking:
    """Per-sample scores for one client's actual local-update batch stream."""

    own_losses: torch.Tensor
    other_losses: torch.Tensor
    scores: torch.Tensor
    ranked_positions: torch.Tensor
    labels: torch.Tensor
    sample_indices: torch.Tensor


def encode_training_batches(
    model: torch.nn.Module,
    batches: list[TrainingBatch],
    device: torch.device,
) -> torch.Tensor:
    """Return frozen image-encoder features aligned with a local batch stream.

    Frozen-CLIP runs normally pass precomputed two-dimensional features through
    ``encode_images``.  The same path also supports ordinary image tensors when
    feature precomputation is disabled.  No additional normalization is
    applied, so the statistics use the exact representation returned by the
    model's frozen image-encoding path.
    """
    if not batches:
        raise ValueError("WWW feature statistics require at least one batch.")
    encoder = getattr(model, "encode_images", None)
    if not callable(encoder):
        raise TypeError("WWW feature statistics require model.encode_images().")

    was_training = model.training
    feature_parts = []
    model.eval()
    try:
        with torch.no_grad():
            for images, labels in batches:
                features = encoder(images.to(device))
                if not isinstance(features, torch.Tensor) or features.ndim != 2:
                    raise ValueError(
                        "WWW encoded features must have shape [samples, dimension]."
                    )
                if features.shape[0] != labels.numel():
                    raise ValueError(
                        "WWW encoded features must align one-to-one with labels."
                    )
                feature_parts.append(features.detach().to(device="cpu"))
    finally:
        model.train(was_training)
    return torch.cat(feature_parts, dim=0)


def infer_other_clients_state(
    global_state: ModelState,
    own_state: ModelState,
    own_weight: float,
) -> ModelState:
    """Recover the normalized aggregate of every client except the caller.

    For ``global = own_weight * own + (1 - own_weight) * other``, this returns
    ``other``.  The inputs are expected to contain the exact tensors that took
    part in the server's linear parameter average.
    """
    weight = float(own_weight)
    if not 0.0 < weight < 1.0:
        raise ValueError("WWW reconstruction requires own_weight in (0, 1).")
    if set(global_state) != set(own_state):
        missing_from_own = sorted(set(global_state) - set(own_state))
        missing_from_global = sorted(set(own_state) - set(global_state))
        raise ValueError(
            "WWW model states must contain identical parameter names; "
            f"missing_from_own={missing_from_own}, "
            f"missing_from_global={missing_from_global}."
        )

    denominator = 1.0 - weight
    reconstructed = {}
    for name, global_tensor in global_state.items():
        own_tensor = own_state[name]
        if global_tensor.shape != own_tensor.shape:
            raise ValueError(
                f"WWW parameter {name!r} has incompatible shapes "
                f"{tuple(global_tensor.shape)} and {tuple(own_tensor.shape)}."
            )
        if not (global_tensor.is_floating_point() or global_tensor.is_complex()):
            raise TypeError(
                f"WWW parameter {name!r} must use a floating or complex dtype."
            )
        own_tensor = own_tensor.to(
            device=global_tensor.device,
            dtype=global_tensor.dtype,
        )
        reconstructed[name] = (
            global_tensor.detach() - weight * own_tensor.detach()
        ).div(denominator)
    return reconstructed


def rank_loss_differences(
    model: torch.nn.Module,
    batches: list[TrainingBatch],
    own_state: ModelState,
    other_state: ModelState,
    restore_state: ModelState,
    device: torch.device,
    sample_indices: torch.Tensor | None = None,
) -> WWWRanking:
    """Compute ``L(x; other) - L(x; own)`` and rank it ascending.

    ``batches`` is the exact batch stream already chosen for the upcoming local
    update.  Positions therefore refer to the concatenation of that stream,
    including repeated samples when a protocol deliberately uses multiple
    local epochs.
    """
    if not batches:
        raise ValueError("WWW ranking requires at least one training batch.")

    reference_names = set(restore_state)
    if set(own_state) != reference_names or set(other_state) != reference_names:
        raise ValueError("WWW own, other, and restore states must have equal keys.")

    labels = torch.cat([labels.detach().cpu().long() for _, labels in batches])
    if sample_indices is None:
        sample_indices = torch.arange(labels.numel(), dtype=torch.long)
    else:
        sample_indices = sample_indices.detach().cpu().long().flatten()
    if sample_indices.numel() != labels.numel():
        raise ValueError("WWW sample_indices must align one-to-one with the batch stream.")
    if labels.numel() == 0:
        # Poisson empty draws still execute a noise-only private optimizer step.
        empty = torch.empty(0)
        return WWWRanking(empty, empty.clone(), empty.clone(),
                          torch.empty(0, dtype=torch.long), labels, sample_indices)

    was_training = model.training

    def losses_for(state: ModelState) -> torch.Tensor:
        model.load_state_dict(state, strict=False)
        model.eval()
        losses = []
        with torch.no_grad():
            for images, labels in batches:
                if not labels.numel():
                    continue
                images = images.to(device)
                labels = labels.to(device).long()
                losses.append(
                    F.cross_entropy(model(images), labels, reduction="none")
                    .detach()
                    .cpu()
                )
        return torch.cat(losses)

    try:
        own_losses = losses_for(own_state)
        other_losses = losses_for(other_state)
    finally:
        model.load_state_dict(restore_state, strict=False)
        model.train(was_training)

    scores = other_losses - own_losses
    if not torch.isfinite(scores).all():
        raise ValueError("WWW loss differences must be finite.")
    ranked_positions = torch.argsort(scores, descending=False, stable=True)
    return WWWRanking(
        own_losses=own_losses,
        other_losses=other_losses,
        scores=scores,
        ranked_positions=ranked_positions,
        labels=labels,
        sample_indices=sample_indices,
    )
