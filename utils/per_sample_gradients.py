"""Bounded, exact per-record reverse-mode gradients for existing model graphs.

Batched VJPs keep the real forward graph and parameter objects. This matters
for FedSGD clients whose PEFT parameters are bound into a shared Transformer:
functional_call/vmap over the wrapper cannot safely replace those parameters.
The caller chunks records, then clips/sums all chunks before its one DP step.
"""
from __future__ import annotations

import torch


GRAD_SAMPLE_BACKENDS = {"auto", "loop", "batched", "vmap"}


def resolve_grad_sample_backend(model, backend: str) -> str:
    backend = str(backend).lower()
    if backend not in GRAD_SAMPLE_BACKENDS:
        raise ValueError("grad_sample_backend must be auto, loop, batched, or vmap.")
    if backend != "auto":
        return backend
    underlying = getattr(model, "_shared_model", model)
    if bool(getattr(underlying, "gradient_checkpointing", False)):
        return "loop"
    # Keep the original functional backend for standard ResNet/GroupNorm.
    if str(getattr(model, "model_type", "")).lower() == "resnet18":
        return "vmap"
    return "batched"


def gradients_from_losses(
    losses: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    backend: str = "batched",
    retain_graph: bool = False,
) -> list[torch.Tensor]:
    """Return [records, *parameter.shape] gradients; never write parameter.grad.

    Each VJP selects exactly one loss, including all tokens belonging to that
    record. The norm must be computed jointly across all trainable parameters.
    This helper is intentionally called on small record chunks, not full pools.
    """
    if losses.ndim != 1 or not losses.numel():
        raise ValueError("Per-record gradients require a nonempty loss vector.")
    if backend == "batched":
        basis = torch.eye(losses.numel(), device=losses.device, dtype=losses.dtype)
        values = torch.autograd.grad(
            losses, parameters, grad_outputs=basis, is_grads_batched=True,
            allow_unused=True, retain_graph=retain_graph,
        )
        return [
            value.detach() if value is not None else parameter.new_zeros(
                (losses.numel(), *parameter.shape)
            )
            for parameter, value in zip(parameters, values)
        ]
    if backend != "loop":
        raise ValueError("Loss-graph gradients support loop or batched.")
    rows = [[] for _ in parameters]
    for index, loss in enumerate(losses):
        values = torch.autograd.grad(
            loss, parameters, allow_unused=True,
            retain_graph=retain_graph or index + 1 < losses.numel(),
        )
        for parameter, destination, value in zip(parameters, rows, values):
            destination.append(value.detach() if value is not None else torch.zeros_like(parameter))
    return [torch.stack(values) for values in rows]


def clipped_sum_from_losses(losses, parameters, max_norm, weights=None, *, backend="batched"):
    """Joint L2 clip, optionally apply detached WWW weights, then sum records."""
    gradients = gradients_from_losses(losses, parameters, backend=backend)
    norm_sq = torch.zeros(losses.numel(), device=losses.device, dtype=torch.float32)
    for gradient in gradients:
        norm_sq.add_(gradient.float().reshape(losses.numel(), -1).square().sum(dim=1))
    if not torch.isfinite(norm_sq).all():
        raise ValueError("Non-finite per-record gradient norm.")
    factors = (float(max_norm) / norm_sq.sqrt().clamp_min(1e-12)).clamp(max=1)
    weighted = factors
    if weights is not None:
        if weights.numel() != losses.numel():
            raise ValueError("Per-record weights must align with the loss vector.")
        weighted = factors * weights.detach().to(factors)
    sums = []
    with torch.no_grad():
        for gradient in gradients:
            shape = (losses.numel(),) + (1,) * (gradient.ndim - 1)
            # Some autograd kernels return expanded views; do not mutate them.
            sums.append((gradient * weighted.to(gradient.dtype).view(shape)).sum(dim=0))
    return sums, factors.detach()
