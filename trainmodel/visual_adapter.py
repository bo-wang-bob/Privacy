"""Deprecated compatibility imports for the former Visual Adapter name."""

from trainmodel.clip_adapter import (
    CLIPAdapter,
    CLIPBottleneckAdapter,
    DEFAULT_CLIP_ADAPTER_TEMPLATE,
    OXFORDPETS_CLIP_ADAPTER_TEMPLATE,
    build_clip_adapter_text_features,
    get_clip_adapter_prompt_template,
)

DEFAULT_VISUAL_ADAPTER_TEMPLATE = DEFAULT_CLIP_ADAPTER_TEMPLATE
OXFORDPETS_VISUAL_ADAPTER_TEMPLATE = OXFORDPETS_CLIP_ADAPTER_TEMPLATE
get_visual_adapter_prompt_template = get_clip_adapter_prompt_template
build_visual_adapter_text_features = build_clip_adapter_text_features
VisualAdapter = CLIPBottleneckAdapter
VisualCLIPAdapter = CLIPAdapter

__all__ = [
    "CLIPAdapter",
    "CLIPBottleneckAdapter",
    "VisualAdapter",
    "VisualCLIPAdapter",
    "build_clip_adapter_text_features",
    "build_visual_adapter_text_features",
    "get_clip_adapter_prompt_template",
    "get_visual_adapter_prompt_template",
]
