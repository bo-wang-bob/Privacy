import math
import secrets

import torch


def planned_private_probe_steps(audit_config: dict | None) -> int:
    """Conservative count of isolated client-update queries made by active MIAs."""
    config = audit_config or {}
    if not bool(config.get("enabled", True)):
        return 0
    attacks = set(config.get("attacks", []))
    steps = 0
    if "nasr_active" in attacks:
        maximum = int(config.get("active_max_samples", 16))
        steps += 2 * max(1, maximum // 2)
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
    for order in (2, 3, 4, 5, 8, 16, 32, 64, 128, 256):
        rdp = compositions * order / (2.0 * noise_multiplier**2)
        candidates.append(rdp + math.log(1.0 / delta) / (order - 1))
    return min(candidates)


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
