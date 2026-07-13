import logging
from typing import Optional

import torch

from utils.constants import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD, DATASET_MAPPING

logger = logging.getLogger(__name__)

FPL_IMAGE_SIZE = (224, 224)


def create_trigger_pattern(
    size: int,
    max_dim: tuple[int, int] = FPL_IMAGE_SIZE,
    position: str = "bottom_right",
) -> list[list[int]]:
    """Create a square trigger pattern inside a CLIP input image."""
    if position == "bottom_right":
        start_x = max_dim[0] - size
        start_y = max_dim[1] - size
    elif position == "center":
        start_x = (max_dim[0] - size) // 2
        start_y = (max_dim[1] - size) // 2
    elif position == "top_right":
        start_x = 0
        start_y = max_dim[1] - size
    elif position == "bottom_left":
        start_x = max_dim[0] - size
        start_y = 0
    else:
        start_x = 0
        start_y = 0

    return [
        [start_x + row, start_y + col]
        for row in range(size)
        for col in range(size)
    ]


def apply_pattern_to_trigger(
    trigger: torch.Tensor,
    pattern: list[list[int]],
    values: list[float],
) -> torch.Tensor:
    for row, col in pattern:
        for channel, value in enumerate(values):
            trigger[channel, row, col] = value
    return trigger


def initialize_trigger(device: torch.device) -> torch.Tensor:
    """Return an empty trigger in normalized CLIP image space."""
    return torch.zeros((3, *FPL_IMAGE_SIZE), device=device, dtype=torch.float32)


def setup_fpl_triggers(
    dataset_name: str,
    device: torch.device,
    malnum: int,
    total_users: int,
    fpl: bool = True,
    trigger_size: Optional[int] = None,
):
    """Create per-user trigger tensors for the supported FPL attacks."""
    normalized_dataset = dataset_name.lower()
    if normalized_dataset not in DATASET_MAPPING:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if not fpl:
        raise ValueError("This branch only supports FPL triggers.")
    if malnum < 1 or malnum > total_users:
        raise ValueError(
            f"malnum must be in [1, total_users], got {malnum}/{total_users}."
        )

    size = 24 if trigger_size is None else int(trigger_size)
    if size <= 0 or size > min(FPL_IMAGE_SIZE):
        raise ValueError(
            f"trigger_size must be in [1, {min(FPL_IMAGE_SIZE)}], got {size}."
        )

    max_values = [
        (1.0 - mean) / std
        for mean, std in zip(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD)
    ]
    pattern = create_trigger_pattern(size=size)
    trigger_list = []
    trigger_pattern_list = []
    for _ in range(total_users):
        trigger = initialize_trigger(device)
        trigger_list.append(
            apply_pattern_to_trigger(trigger, pattern, max_values)
        )
        trigger_pattern_list.append([point.copy() for point in pattern])

    logger.info(
        "Created FPL triggers: dataset=%s, malicious_users=%s, total_users=%s, size=%s",
        normalized_dataset,
        malnum,
        total_users,
        size,
    )
    return trigger_list, trigger_pattern_list


trigger_create_funcs = {
    "cerberus": setup_fpl_triggers,
    "a3fl": setup_fpl_triggers,
    "sabre": setup_fpl_triggers,
}
