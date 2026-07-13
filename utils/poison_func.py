from typing import Union

import torch


def get_poisoned_sample_count(
    poison_ratio: Union[int, float],
    batch_size: int,
    fpl: bool = True,
) -> int:
    """Convert the FPL per-batch poison count into a bounded sample count."""
    if not fpl:
        raise ValueError("This branch only supports FPL poisoning semantics.")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative.")
    if isinstance(poison_ratio, float) and not poison_ratio.is_integer():
        raise ValueError(
            "In FPL mode, poisonratio must be an integer count per batch; "
            f"got {poison_ratio}."
        )

    poison_count = int(poison_ratio)
    if poison_count < 0:
        raise ValueError("poisonratio must be non-negative.")
    return min(poison_count, batch_size)


def add_pixel_pattern(
    device: torch.device,
    ori_images: torch.Tensor,
    noise_trigger: torch.Tensor,
    pattern_mask: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    overlay: bool = False,
) -> torch.Tensor:
    """Apply a patch trigger, or add the bounded full-image SABRE perturbation."""
    images = ori_images.clone().to(device)
    noise_trigger = noise_trigger.to(device)

    if overlay:
        images = images + noise_trigger
    else:
        images = images * pattern_mask.to(device)
        images = images + noise_trigger

    for channel in range(images.shape[1]):
        lower_bound = (0.0 - mean[channel]) / std[channel]
        upper_bound = (1.0 - mean[channel]) / std[channel]
        images[:, channel].clamp_(lower_bound, upper_bound)
    return images
