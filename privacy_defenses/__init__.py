"""Independent privacy defenses for federated soft-prompt tuning."""

from privacy_defenses.controller import (
    FEDMIA_BASELINE_DEFENSES,
    SUPPORTED_DEFENSES,
    DefenseController,
    attach_hamp_output_transform,
    attach_output_temperature_transform,
)

__all__ = [
    "SUPPORTED_DEFENSES",
    "FEDMIA_BASELINE_DEFENSES",
    "DefenseController",
    "attach_hamp_output_transform",
    "attach_output_temperature_transform",
]
