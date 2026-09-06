"""WWW's INO-SGD weighting and Poisson record-level privacy accounting.

The ranking is ascending loss difference, replacing the paper's descending loss.
The tail has ceil(expected_batch_size * fraction) fixed intervals of length C.
Algorithm 1 clips to C first, then multiplies by the interval's average BIF.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import betainc

from utils.privacy_accounting import (
    calibrate_poisson_sampled_gaussian_noise,
    poisson_sampled_gaussian_epsilon,
    private_generator,
)


DEFAULTS = {
    "target_epsilon": 3.0,
    "max_grad_norm": 8.0,
    "delta": 1e-5,
    "noise_multiplier": "auto",
    "www_tail_fraction": 0.2,
    "www_beta_alpha": 1.0,
    "www_beta_beta": 1.0,
    "www_analysis_interval": 1,
    "www_analysis_timing": "pre_update",
    "www_feature_statistics": False,
    "www_validation_top_fraction": 0.2,
    "adjacency": "add_remove",
    "accountant": "rdp",
    "sampling": "poisson",
    "reproducible_dp_noise": False,
    "release_private_diagnostics": False,
}


def validate_www(config: dict) -> None:
    """Resolve defaults and reject invalid budgets before any training starts."""
    if str(config.get("name", "none")).lower() != "www":
        return
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    for key in ("target_epsilon", "max_grad_norm", "www_beta_alpha", "www_beta_beta"):
        value = float(config[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"defense.{key} must be finite and positive.")
        config[key] = value
    for key in ("delta", "www_tail_fraction"):
        value = float(config[key])
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"defense.{key} must be in (0, 1).")
        config[key] = value
    noise = config["noise_multiplier"]
    if noise != "auto" and (
        not math.isfinite(float(noise)) or float(noise) <= 0
    ):
        raise ValueError("defense.noise_multiplier must be 'auto' or finite and positive.")
    for key in ("adjacency", "accountant", "sampling"):
        if config[key] != DEFAULTS[key]:
            raise ValueError(f"WWW requires defense.{key}={DEFAULTS[key]}.")
    if config["www_feature_statistics"] and not config["release_private_diagnostics"]:
        raise ValueError("WWW feature statistics require release_private_diagnostics=true.")


def ino_weights(
    scores: torch.Tensor,
    tail_fraction: float = 0.2,
    beta_alpha: float = 1.0,
    beta_beta: float = 1.0,
    *,
    expected_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate the flipped Beta CDF on each equal-C gradient interval.

    Return weights in original sample order, ascending positions, and tail mask.
    Equal scores retain batch order. The fixed-length tail is right-aligned with
    the actual batch, including batches shorter than the tail (paper C.2.3).
    Changing its length with the sampled batch would invalidate the C bound.
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
    tail_length = math.ceil(expected_batch_size * tail_fraction)
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
                         extra_loss=None):
    """Algorithm 1: rho_i * g_i / max(1, ||g_i||_2 / C), jointly over parameters.

    One sample graph at a time avoids materializing B full PEFT gradients and
    prevents batch-dependent layers from mixing different records' gradients.
    """
    if weights.numel() != labels.numel():
        raise ValueError("WWW weights must align with the actual training batch.")
    sums = [torch.zeros_like(p) for p in parameters]
    for index in range(labels.numel()):
        inputs, targets = images[index:index + 1], labels[index:index + 1]
        loss = F.cross_entropy(model(inputs), targets)
        if extra_loss is not None:
            loss = loss + extra_loss(inputs, targets).mean()
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        norm_sq = sum(g.detach().float().square().sum() for g in gradients if g is not None)
        if not torch.isfinite(norm_sq):
            raise ValueError("WWW encountered a non-finite per-sample gradient.")
        factor = (max_norm / norm_sq.sqrt().clamp_min(1e-12)).clamp(max=1)
        factor = factor * weights[index].to(factor)
        with torch.no_grad():
            for destination, gradient in zip(sums, gradients):
                if gradient is not None:
                    destination.add_(gradient * factor.to(gradient))
    return sums


class WWWPrivacy:
    """Use the same Poisson sampling schedules and RDP calibration as Record-DP."""

    def __init__(self, config, total_rounds, device, seed):
        validate_www(config)
        self.config = config
        self.total_rounds = int(total_rounds)
        self.device = device
        self.seed = seed
        self.planned_steps = {}
        self.sample_rates = {}
        self.expected_batch_sizes = {}
        self.noise_multiplier = None

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
                raise ValueError("WWW does not support private BatchNorm running statistics.")
            self.planned_steps[user.id] = self.total_rounds * user.record_dp_steps_per_update
            self.sample_rates[user.id] = float(user.record_dp_sample_rate)
            self.expected_batch_sizes[user.id] = int(user.record_dp_expected_batch_size)
        if not self.planned_steps or min(self.planned_steps.values()) <= 0:
            raise ValueError("WWW requires a positive training schedule.")
        schedules = [(self.sample_rates[i], self.planned_steps[i])
                     for i in sorted(self.planned_steps)]
        noise = self.config["noise_multiplier"]
        self.noise_multiplier = (
            calibrate_poisson_sampled_gaussian_noise(
                self.config["target_epsilon"], schedules, self.config["delta"],
            )
            if noise == "auto" else float(noise)
        )
        if any(self.epsilon(self.planned_steps[i], i) > float(self.config["target_epsilon"]) + 1e-10
               for i in self.planned_steps):
            raise ValueError("WWW noise_multiplier is too small for the planned privacy budget.")

    def epsilon(self, steps, client_id=None):
        if self.noise_multiplier is None:
            raise RuntimeError("WWW privacy accounting must be configured first.")
        rates = ([self.sample_rates[client_id]] if client_id is not None
                 else self.sample_rates.values())
        return max(poisson_sampled_gaussian_epsilon(
            self.noise_multiplier, rate, steps, float(self.config["delta"]),
        ) for rate in rates)

    @property
    def reproducible(self):
        return bool(self.config.get("reproducible_noise", self.config["reproducible_dp_noise"]))

    def sampling_generator(self, client_id, round_index):
        return private_generator(
            torch.device("cpu"), self.reproducible,
            self.seed + 1000003 * client_id + 1009 * round_index + 17011,
        )

    def generator(self, client_id, round_index):
        return private_generator(
            self.device, self.reproducible,
            self.seed + 1000003 * client_id + 1009 * round_index + 29009,
        )

    def step(self, user, model, optimizer, images, labels, weights, steps, generator,
             extra_loss=None):
        if self.noise_multiplier is None:
            raise RuntimeError("WWW must be configured before training.")
        if steps >= self.planned_steps[user.id]:
            raise RuntimeError("WWW cannot exceed the calibrated training schedule.")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)
        max_norm = float(self.config["max_grad_norm"])
        sums = weighted_clipped_sum(model, images, labels, parameters, max_norm,
                                    weights, extra_loss)
        # A fixed-length INO tail preserves the add/remove bound C, including
        # changes in other records' weights. Empty draws release pure noise.
        noise_std = self.noise_multiplier * max_norm
        denominator = self.expected_batch_sizes[user.id]
        with torch.no_grad():
            for parameter, gradient_sum in zip(parameters, sums):
                noise = torch.randn(parameter.shape, generator=generator,
                                    device=parameter.device, dtype=parameter.dtype)
                parameter.grad = (gradient_sum + noise * noise_std) / denominator
        optimizer.step()  # FedSGD's pre-hook captures only the privatized gradient.

    def summary(self, steps):
        epsilons = {i: self.epsilon(int(steps.get(i, 0)), i) for i in self.planned_steps}
        return {
            "privacy_unit": "record",
            "adjacency": "add_remove",
            "accountant": "poisson_sampled_gaussian_rdp",
            "sampling": "poisson",
            "subsampling_amplification": True,
            "target_epsilon": float(self.config["target_epsilon"]),
            "epsilon_upper_bound": max(epsilons.values(), default=0.0),
            "delta": float(self.config["delta"]),
            "max_grad_norm": float(self.config["max_grad_norm"]),
            "sum_sensitivity": float(self.config["max_grad_norm"]),
            "noise_multiplier": self.noise_multiplier,
            "noise_std_on_sum": (None if self.noise_multiplier is None else
                                 self.noise_multiplier * float(self.config["max_grad_norm"])),
            "normalization": "fixed_expected_batch_size",
            "client_upload_is_private": True,
            "formal_dp_enabled": not self.reproducible
                                 and not self.config["release_private_diagnostics"],
            "private_diagnostics_released": bool(self.config["release_private_diagnostics"]),
            "per_client": {str(i): {"actual_steps": int(steps.get(i, 0)),
                                    "planned_steps": self.planned_steps[i],
                                    "sample_rate": self.sample_rates[i],
                                    "expected_batch_size": self.expected_batch_sizes[i],
                                    "tail_length_samples": math.ceil(self.expected_batch_sizes[i] * self.config["www_tail_fraction"]),
                                    "epsilon": epsilons[i]} for i in epsilons},
            "scope": "Defended uploads/models; fixed public per-client sampling rates, "
                     "expected batch sizes and tail lengths; disjoint client datasets. "
                     "Sampling identities and realized batch sizes are not DP releases. "
                     "Audit labels, private signals, training data and optional diagnostics "
                     "are local research artifacts and are not DP releases.",
        }
