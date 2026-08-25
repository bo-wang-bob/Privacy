from __future__ import annotations

import math
import secrets

import torch


DEFAULT_RDP_ORDERS = (
    2,
    3,
    4,
    5,
    8,
    16,
    32,
    64,
    128,
    256,
)


def planned_private_probe_steps(audit_config: dict | None) -> int:
    """Conservative count of isolated client-update queries made by active MIAs."""
    config = audit_config or {}
    if not bool(config.get("enabled", True)):
        return 0
    attacks = set(config.get("attacks", []))
    steps = 0
    if "nasr_active" in attacks:
        maximum = int(config.get("active_max_samples", 16))
        cycles = max(1, int(config.get("active_probe_cycles", 3)))
        steps += 2 * max(1, maximum // 2) * cycles
    if "promptmia" in attacks:
        maximum = int(config.get("promptmia_max_samples", 16))
        steps += 2 * max(1, maximum // 2)
    return steps


def gaussian_rdp_epsilon(
    noise_multiplier: float,
    steps: int,
    delta: float,
    mechanisms_per_step: int = 1,
) -> float:
    """Conservative Gaussian RDP bound without subsampling amplification."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if steps < 0 or mechanisms_per_step <= 0:
        raise ValueError("steps must be non-negative and mechanisms_per_step positive.")
    if steps <= 0:
        return 0.0
    if noise_multiplier <= 0:
        return math.inf
    compositions = int(steps) * int(mechanisms_per_step)
    candidates = []
    for order in DEFAULT_RDP_ORDERS:
        rdp = compositions * order / (2.0 * noise_multiplier**2)
        candidates.append(rdp + math.log(1.0 / delta) / (order - 1))
    return min(candidates)


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def poisson_sampled_gaussian_rdp(
    noise_multiplier: float,
    sample_rate: float,
    order: int,
) -> float:
    """RDP of one Poisson-sampled Gaussian mechanism at an integer order.

    The mechanism independently samples each record with probability ``q``,
    clips each contribution to unit norm, sums the clipped contributions, and
    adds Gaussian noise with standard deviation ``noise_multiplier``.  Scaling
    both the clipped sum and noise by the same constant does not change RDP.
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be in [0, 1].")
    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or order < 2
    ):
        raise ValueError("RDP order must be an integer of at least two.")
    if sample_rate == 0.0:
        return 0.0
    if noise_multiplier <= 0:
        return math.inf
    if sample_rate == 1.0:
        return order / (2.0 * noise_multiplier**2)

    log_q = math.log(sample_rate)
    log_one_minus_q = math.log1p(-sample_rate)
    log_terms = []
    for index in range(order + 1):
        log_binomial = (
            math.lgamma(order + 1)
            - math.lgamma(index + 1)
            - math.lgamma(order - index + 1)
        )
        privacy_loss = (index * index - index) / (
            2.0 * noise_multiplier**2
        )
        log_terms.append(
            log_binomial
            + index * log_q
            + (order - index) * log_one_minus_q
            + privacy_loss
        )
    return _logsumexp(log_terms) / (order - 1)


def poisson_sampled_gaussian_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    orders: tuple[int, ...] = DEFAULT_RDP_ORDERS,
) -> float:
    """Compose Poisson-sampled Gaussian DP-SGD steps and return epsilon."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if steps == 0 or sample_rate == 0.0:
        return 0.0
    candidates = []
    for order in orders:
        rdp = steps * poisson_sampled_gaussian_rdp(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            order=order,
        )
        candidates.append(rdp + math.log(1.0 / delta) / (order - 1))
    return min(candidates)


def max_poisson_sampled_gaussian_epsilon(
    noise_multiplier: float,
    schedules: list[tuple[float, int]],
    delta: float,
) -> float:
    """Return the worst per-client epsilon for disjoint client datasets."""
    if not schedules:
        return 0.0
    return max(
        poisson_sampled_gaussian_epsilon(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )
        for sample_rate, steps in schedules
    )


def calibrate_poisson_sampled_gaussian_noise(
    target_epsilon: float,
    schedules: list[tuple[float, int]],
    delta: float,
) -> float:
    """Find one noise multiplier meeting every client's planned budget."""
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive.")
    if not schedules or any(steps <= 0 for _, steps in schedules):
        raise ValueError("At least one positive-step client schedule is required.")
    low, high = 1e-4, 1.0
    while (
        max_poisson_sampled_gaussian_epsilon(high, schedules, delta)
        > target_epsilon
    ):
        high *= 2.0
        if high > 1e6:
            raise ValueError("Could not calibrate a finite noise multiplier.")
    for _ in range(80):
        middle = (low + high) / 2.0
        if (
            max_poisson_sampled_gaussian_epsilon(middle, schedules, delta)
            <= target_epsilon
        ):
            high = middle
        else:
            low = middle
    return high


def calibrate_gaussian_noise(
    target_epsilon: float,
    steps: int,
    delta: float,
    mechanisms_per_step: int = 1,
) -> float:
    """Binary-search a noise multiplier meeting the conservative RDP bound."""
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive when calibrating noise.")
    low, high = 1e-4, 1.0
    while (
        gaussian_rdp_epsilon(high, steps, delta, mechanisms_per_step) > target_epsilon
    ):
        high *= 2.0
        if high > 1e6:
            raise ValueError("Could not calibrate a finite Gaussian noise multiplier.")
    for _ in range(80):
        middle = (low + high) / 2.0
        if (
            gaussian_rdp_epsilon(middle, steps, delta, mechanisms_per_step)
            <= target_epsilon
        ):
            high = middle
        else:
            low = middle
    return high


def private_generator(
    device: torch.device,
    reproducible: bool,
    deterministic_seed: int,
) -> torch.Generator:
    """Use an unrecorded OS-random seed unless reproducibility is explicitly requested."""
    generator_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    seed = deterministic_seed if reproducible else secrets.randbits(63)
    generator.manual_seed(seed)
    return generator
