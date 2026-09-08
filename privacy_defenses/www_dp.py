"""WWW's risk-controlled prediction loss, without clipping or noise.

Historical INO/clipping helpers remain available for old standalone benchmarks;
the current WWW training path never calls them. The module/class names are kept
for compatibility with experiment entry points.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import betainc
from utils.per_sample_gradients import (
    clipped_sum_from_losses, gradients_from_losses, resolve_grad_sample_backend,
)
from utils.performance import measure_stage

DEFAULTS = {
    "target_epsilon": None,
    "max_grad_norm": None,
    "delta": None,
    "noise_multiplier": 0.0,
    "www_tail_fraction": 0.8,
    "www_tail_basis": "actual_batch",
    "www_beta_alpha": None,
    "www_beta_beta": None,
    "www_regularization_weight": 1.0,
    "www_analysis_interval": 1,
    "www_analysis_timing": "pre_update",
    "www_feature_statistics": False,
    "www_record_diagnostics": True,
    "www_validation_top_fraction": 0.2,
    "adjacency": None,
    "accountant": None,
    "sampling": "shuffled_batches",
    "reproducible_dp_noise": False,
    "release_private_diagnostics": False,
    "grad_sample_backend": "auto",
    "microbatch_size": 4,
}


def validate_www(config: dict) -> None:
    """Validate risk-loss settings and disable legacy clipping/DP parameters.

    Shared sweep overrides (e.g. target_epsilon for Record-DP) and old WWW
    configurations remain loadable, but cannot enable noise or a DP claim.
    """
    if str(config.get("name", "none")).lower() != "www":
        return
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    for key in ("target_epsilon", "delta", "noise_multiplier", "adjacency", "accountant",
                "max_grad_norm", "www_beta_alpha", "www_beta_beta"):
        config[key] = DEFAULTS[key]
    strength = config["www_regularization_weight"]
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength < 0:
        raise ValueError("defense.www_regularization_weight must be finite and nonnegative.")
    config["www_regularization_weight"] = float(strength)
    for key in ("www_tail_fraction",):
        value = float(config[key])
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"defense.{key} must be in (0, 1).")
        config[key] = value
    if config["sampling"] != "shuffled_batches":
        raise ValueError("WWW requires defense.sampling=shuffled_batches.")
    if config["www_tail_basis"] not in {"actual_batch", "expected_batch"}:
        raise ValueError("WWW www_tail_basis must be actual_batch or expected_batch.")
    if not isinstance(config["www_record_diagnostics"], bool):
        raise ValueError("WWW www_record_diagnostics must be a boolean.")
    if config["www_feature_statistics"] and not config["release_private_diagnostics"]:
        raise ValueError("WWW feature statistics require release_private_diagnostics=true.")
    if str(config["grad_sample_backend"]).lower() not in {"auto", "loop", "batched"}:
        raise ValueError("WWW grad_sample_backend must be auto, loop, or batched.")
    chunk = config["microbatch_size"]
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0:
        raise ValueError("WWW microbatch_size must be a positive integer.")


def risk_regularization_weights(scores, tail_fraction=0.8, *, expected_batch_size,
                                tail_basis="actual_batch"):
    """Freeze stable risk ranks; only the upper tail receives an increasing loss weight.

    Actual-tail rank j=1..m receives (j-.5)/m. The historical expected_batch
    option right-aligns short batches in that fixed width. This is not an INO
    multiplier on CE gradients: CE remains present for every record.
    """
    scores = scores.detach().cpu().double().flatten()
    if not torch.isfinite(scores).all():
        raise ValueError("WWW requires finite sample scores.")
    if not math.isfinite(tail_fraction) or not 0 < tail_fraction < 1:
        raise ValueError("WWW tail_fraction must be in (0, 1).")
    if isinstance(expected_batch_size, bool) or int(expected_batch_size) != expected_batch_size or expected_batch_size <= 0:
        raise ValueError("WWW expected_batch_size must be a positive integer.")
    if tail_basis not in {"actual_batch", "expected_batch"}:
        raise ValueError("WWW tail_basis must be actual_batch or expected_batch.")
    count = scores.numel()
    positions = torch.argsort(scores, stable=True)
    weights, tail = torch.zeros_like(scores), torch.zeros(count, dtype=torch.bool)
    if count:
        width = math.ceil((count if tail_basis == "actual_batch" else expected_batch_size) * tail_fraction)
        selected = min(count, width)
        indices = positions[-selected:]
        weights[indices] = (torch.arange(width-selected, width, dtype=torch.float64) + .5) / width
        tail[indices] = True
    return weights, positions, tail


def risk_controlled_losses(logits, labels, risk_weights, reference_probability, strength):
    """CE + lambda*r*|p_y-q_y|; teacher and ranks never receive gradients.

    exp(-CE) is the true-label softmax probability, from the same student graph.
    The signed CE-gradient factor is diagnostic only, never a gradient rewrite.
    """
    ce = F.cross_entropy(logits, labels, reduction="none")
    weights = risk_weights.detach().to(ce)
    teacher = reference_probability.detach().to(ce)
    if weights.shape != ce.shape or teacher.shape != ce.shape:
        raise ValueError("WWW risk weights and teacher probabilities must align with the batch.")
    if (not torch.isfinite(weights).all() or not torch.isfinite(teacher).all()
            or (weights < 0).any() or (weights > 1).any()
            or (teacher < 0).any() or (teacher > 1).any()):
        raise ValueError("WWW risk weights and teacher probabilities must be finite in [0, 1].")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("WWW regularization strength must be finite and nonnegative.")
    probability = ce.neg().exp()
    gap = probability - teacher
    regularizer = float(strength) * weights * gap.abs()
    total = ce + regularizer
    if not torch.isfinite(total).all():
        raise ValueError("WWW encountered a non-finite training loss.")
    return {
        "ce_loss": ce, "current_probability": probability,
        "reference_probability": teacher, "confidence_gap": gap,
        "cross_difference": gap.abs(), "regularization_loss": regularizer,
        "total_loss": total,
        "ce_gradient_factor": (1 - float(strength) * weights * probability * gap.sign()).detach(),
    }


def _diagnostic_norms(losses, parameters, backend, chunk_size):
    """Bound per-record gradient storage and preserve the graph for training backward."""
    parts = []
    for start in range(0, losses.numel(), chunk_size):
        chunk = losses[start:start + chunk_size]
        gradients = gradients_from_losses(chunk, parameters, backend=backend, retain_graph=True)
        squared = torch.zeros_like(chunk, dtype=torch.float32)
        for gradient in gradients:
            squared.add_(gradient.float().reshape(chunk.numel(), -1).square().sum(dim=1))
        if not torch.isfinite(squared).all():
            raise ValueError("WWW encountered a non-finite diagnostic gradient norm.")
        parts.append(squared.sqrt())
        del gradients
    return torch.cat(parts)


def ino_weights(
    scores: torch.Tensor,
    tail_fraction: float = DEFAULTS["www_tail_fraction"],
    beta_alpha: float = 1.0,
    beta_beta: float = 1.0,
    *,
    expected_batch_size: int,
    tail_basis: str = "actual_batch",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate the flipped Beta CDF on each equal-C gradient interval.

    Return weights in original sample order, ascending positions, and tail mask.
    Equal scores retain batch order. The legacy expected_batch basis retains
    its fixed tail and right-aligns smaller draws (paper C.2.3). Neither basis
    provides a DP sensitivity bound when low-risk gradients are exempt.
    """
    scores = scores.detach().cpu().double().flatten()
    if not torch.isfinite(scores).all():
        raise ValueError("WWW requires finite sample scores.")
    if isinstance(expected_batch_size, bool) or int(expected_batch_size) != expected_batch_size or expected_batch_size <= 0:
        raise ValueError("WWW expected_batch_size must be a positive integer.")
    if not math.isfinite(tail_fraction) or not 0 < tail_fraction < 1:
        raise ValueError("WWW tail_fraction must be in (0, 1).")
    if any(not math.isfinite(x) or x <= 0 for x in (beta_alpha, beta_beta)):
        raise ValueError("WWW Beta shape parameters must be finite and positive.")
    count = scores.numel()
    if tail_basis not in {"actual_batch", "expected_batch"}:
        raise ValueError("WWW tail_basis must be actual_batch or expected_batch.")
    tail_length = math.ceil((count if tail_basis == "actual_batch" else expected_batch_size) * tail_fraction)
    tail_count = min(count, tail_length)
    positions = torch.argsort(scores, stable=True)
    if count == 0:
        return scores.clone(), positions, torch.zeros(0, dtype=torch.bool)
    tail_positions = positions[-tail_count:]
    # H(x) = integral_0^x I_t(alpha, beta) dt; integration by parts.
    x = np.linspace(1.0, 0.0, tail_length + 1)
    primitive = x * betainc(beta_alpha, beta_beta, x) - (
        beta_alpha / (beta_alpha + beta_beta)
    ) * betainc(beta_alpha + 1.0, beta_beta, x)
    tail_weights = tail_length * (primitive[:-1] - primitive[1:])
    weights = torch.ones(count, dtype=torch.float64)
    weights[tail_positions] = torch.from_numpy(tail_weights[-tail_count:].copy()).clamp(0, 1)
    tail = torch.zeros(count, dtype=torch.bool)
    tail[tail_positions] = True
    return weights, positions, tail


def weighted_clipped_sum(model, images, labels, parameters, max_norm, weights,
                         extra_loss=None, *, backend="loop", microbatch_size=4,
                         clip_mask=None, return_diagnostics=False):
    """Clip masked records jointly, apply INO weights, then sum contributions.

    The batched backend bounds gradient storage by microbatch_size; all chunks
    contribute to one sum before the optimizer step. Supported PEFT
    models have no batch-dependent normalization. Loop remains a reference.
    With no mask every record is clipped (warmup/reference calculations).
    return_diagnostics reuses clipping norms without another backward pass.
    """
    if weights.numel() != labels.numel():
        raise ValueError("WWW weights must align with the actual training batch.")
    if clip_mask is not None and (clip_mask.shape != labels.shape or clip_mask.dtype != torch.bool):
        raise ValueError("WWW clipping mask must be boolean and align with the actual batch.")
    backend = resolve_grad_sample_backend(model, backend)
    sums = [torch.zeros_like(p) for p in parameters]
    norm_parts, factor_parts = [], []

    def result():
        if not return_diagnostics:
            return sums
        norms = torch.cat(norm_parts) if norm_parts else images.new_empty(0, dtype=torch.float32)
        factors = torch.cat(factor_parts) if factor_parts else norms.clone()
        clipped = norms * factors
        importance = weights.detach().to(norms)
        return sums, {
            "raw_grad_norm": norms,
            "clip_factor": factors,
            "clipped_grad_norm": clipped,
            "weighted_grad_norm": clipped * importance,
            "effective_factor": factors * importance,
        }

    if backend == "batched" and extra_loss is None:
        for start in range(0, labels.numel(), microbatch_size):
            stop = start + microbatch_size
            losses = F.cross_entropy(model(images[start:stop]), labels[start:stop], reduction="none")
            output = clipped_sum_from_losses(
                losses, parameters, max_norm, weights[start:stop],
                clip_mask=None if clip_mask is None else clip_mask[start:stop],
                return_norms=return_diagnostics,
            )
            partial = output[0]
            if return_diagnostics:
                norm_parts.append(output[2])
                factor_parts.append(output[1])
            with torch.no_grad():
                for destination, value in zip(sums, partial):
                    destination.add_(value)
        return result()
    if backend not in {"loop", "batched"}:
        raise ValueError("WWW gradient backend must resolve to loop or batched.")
    weights = weights.to(device=images.device)
    if clip_mask is not None:
        clip_mask = clip_mask.to(device=images.device)
    for index in range(labels.numel()):
        inputs, targets = images[index:index + 1], labels[index:index + 1]
        loss = F.cross_entropy(model(inputs), targets)
        if extra_loss is not None:
            loss = loss + extra_loss(inputs, targets).mean()
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        norm_sq = sum(g.detach().float().square().sum() for g in gradients if g is not None)
        if not torch.isfinite(norm_sq):
            raise ValueError("WWW encountered a non-finite per-sample gradient.")
        norm = norm_sq.sqrt()
        factor = (max_norm / norm.clamp_min(1e-12)).clamp(max=1)
        if clip_mask is not None:
            factor = torch.where(clip_mask[index], factor, torch.ones_like(factor))
        if return_diagnostics:
            norm_parts.append(norm.detach().reshape(1))
            factor_parts.append(factor.detach().reshape(1))
        factor = factor * weights[index].to(factor)
        with torch.no_grad():
            for destination, gradient in zip(sums, gradients):
                if gradient is not None:
                    destination.add_(gradient * factor.to(gradient))
    return result()


class WWWPrivacy:
    """Optimize risk-controlled loss on shuffled mini-batches, with no DP guarantee.

    The historical class/module names are retained for existing integrations.
    """

    def __init__(self, config, total_rounds, device, seed):
        validate_www(config)
        self.config = config
        self.total_rounds = int(total_rounds)
        self.device = device
        self.seed = seed
        self.planned_steps = {}
        self.batch_sizes = {}
        self.noise_multiplier = 0.0

    def configure(self, users, additional_private_steps=0):
        if additional_private_steps:
            raise ValueError("WWW does not support isolated active client probes.")
        for user in users:
            if user.train_samples <= 0:
                raise ValueError("WWW requires nonempty client training sets.")
            if user.federated_method not in {"fedsgd", "fedavg"}:
                raise ValueError("WWW requires linear FedSGD or FedAvg.")
            if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                   for m in user.model.modules()):
                raise ValueError("WWW does not support batch-dependent BatchNorm.")
            per_round = 1 if user.federated_method == "fedsgd" else user.local_epochs * len(user.www_trainloader)
            self.planned_steps[user.id] = self.total_rounds * per_round
            self.batch_sizes[user.id] = min(int(user.batch_size), user.train_samples)
        if not self.planned_steps or min(self.planned_steps.values()) <= 0:
            raise ValueError("WWW requires a positive training schedule.")

    def step(self, user, model, optimizer, images, labels, weights, steps,
             extra_loss=None, *, reference_probability):
        if user.id not in self.planned_steps:
            raise RuntimeError("WWW must be configured before training.")
        if steps >= self.planned_steps[user.id]:
            raise RuntimeError("WWW cannot exceed the configured training schedule.")
        if not labels.numel():
            raise ValueError("WWW shuffled batches must be nonempty.")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)
        with measure_stage(self, "train.www_loss_forward"):
            terms = risk_controlled_losses(
                model(images), labels, weights, reference_probability,
                self.config["www_regularization_weight"],
            )
            additional = extra_loss(images, labels) if extra_loss is not None else torch.zeros_like(terms["ce_loss"])
            if additional.shape != labels.shape or not torch.isfinite(additional).all():
                raise ValueError("WWW additional loss must be finite and sample-aligned.")
            terms["additional_loss"] = additional
            terms["total_loss"] = terms["total_loss"] + additional
            if not torch.isfinite(terms["total_loss"]).all():
                raise ValueError("WWW encountered a non-finite combined training loss.")
        if self.config["www_record_diagnostics"]:
            with measure_stage(self, "train.www_gradient_diagnostics"):
                backend = resolve_grad_sample_backend(model, self.config["grad_sample_backend"])
                chunk = int(self.config["microbatch_size"])
                raw = _diagnostic_norms(terms["ce_loss"], parameters, backend, chunk)
                factor = terms["ce_gradient_factor"]
                # For ordinary WWW the regularizer gradient is exactly collinear
                # with CE; derive its norm without another set of VJPs. A code-
                # poisoning loss has a different direction and must be measured.
                terms["raw_grad_norm"] = raw
                # Avoid cancellation in (factor - 1) for small penalties.
                regularizer_scale = (self.config["www_regularization_weight"] * weights.to(raw)
                                     * terms["current_probability"].detach()
                                     * terms["confidence_gap"].detach().ne(0))
                terms["regularizer_grad_norm"] = regularizer_scale * raw
                terms["total_grad_norm"] = (factor.abs() * raw if extra_loss is None else
                    _diagnostic_norms(terms["total_loss"], parameters, backend, chunk))
        with measure_stage(self, "train.www_backward_step"):
            terms["total_loss"].mean().backward()
            finite = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
            if not finite or not torch.stack(finite).all():
                raise ValueError("WWW encountered a non-finite batch gradient.")
            optimizer.step()  # Exactly one FedSGD upload, from the actual total loss.
        return {name: value.detach() for name, value in terms.items()}

    def summary(self, steps):
        return {
            "mechanism": "risk_controlled_loss",
            "privacy_unit": None,
            "adjacency": None,
            "accountant": None,
            "sampling": "shuffled_batches",
            "subsampling_amplification": False,
            "target_epsilon": None,
            "epsilon_upper_bound": None,
            "delta": None,
            "max_grad_norm": None,
            "clipping_enabled": False,
            "regularization_weight": self.config["www_regularization_weight"],
            "loss": "mean(CE + lambda * risk_weight * abs(p_y - stopgrad(q_y)))",
            "teacher": "exp(-previous_round_other_client_aggregate_CE)",
            "noise_multiplier": self.noise_multiplier,
            "noise_std_on_sum": 0.0,
            "noise_enabled": False,
            "normalization": "actual_batch_size",
            "grad_sample_backend": self.config["grad_sample_backend"],
            "microbatch_size": int(self.config["microbatch_size"]),
            "per_sample_gradients_for_training": False,
            "per_sample_gradients_for_diagnostics": self.config["www_record_diagnostics"],
            "client_upload_is_private": False,
            "formal_dp_enabled": False,
            "non_dp_reason": "noise_disabled",
            "private_diagnostics_released": bool(self.config["release_private_diagnostics"]
                                                 or self.config["www_record_diagnostics"]),
            "per_client": {str(i): {"actual_steps": int(steps.get(i, 0)),
                                    "planned_steps": self.planned_steps[i],
                                    "batch_size": self.batch_sizes[i],
                                    "tail_length_samples": (math.ceil(self.batch_sizes[i] * self.config["www_tail_fraction"])
                                                            if self.config["www_tail_basis"] == "expected_batch" else None),
                                    "epsilon": None} for i in self.planned_steps},
            "scope": "Risk-controlled prediction regularization only. No clipping or noise "
                     "is applied and no differential privacy guarantee is provided for "
                     "client uploads, models, sampling identities or diagnostics.",
        }
