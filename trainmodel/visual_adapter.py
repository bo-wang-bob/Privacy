from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

from trainmodel.custom_clip import format_prompt_template, to_display_name


DEFAULT_VISUAL_ADAPTER_TEMPLATE = "a photo of a {class}."
OXFORDPETS_VISUAL_ADAPTER_TEMPLATE = (
    "a photo of a {class}, a type of pet."
)


def get_visual_adapter_prompt_template(dataset_name: str | None) -> str:
    """Return the fixed zero-shot template used by the visual adapter."""
    if dataset_name and dataset_name.lower() == "oxfordpets":
        return OXFORDPETS_VISUAL_ADAPTER_TEMPLATE
    return DEFAULT_VISUAL_ADAPTER_TEMPLATE


@torch.inference_mode()
def build_visual_adapter_text_features(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    classnames: list[str],
    dataset_name: str | None,
    device: torch.device,
    template: str | None = None,
) -> torch.Tensor:
    """Encode one normalized, fixed CLIP text vector for every class."""
    selected_template = template or get_visual_adapter_prompt_template(dataset_name)
    prompts = [
        format_prompt_template(selected_template, to_display_name(name))
        for name in classnames
    ]
    encoded = processor(
        text=prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {
        name: tensor.to(device)
        for name, tensor in encoded.items()
        if name in {"input_ids", "attention_mask"}
    }
    features = cast(torch.Tensor, clip_model.get_text_features(**text_inputs))
    return F.normalize(features.float(), dim=-1)


class VisualAdapter(nn.Module):
    """CLIP-Adapter bottleneck applied to frozen CLIP features."""

    def __init__(
        self,
        feature_dim: int,
        reduction: int = 4,
        output_relu: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        if feature_dim % reduction != 0:
            raise ValueError("feature_dim must be divisible by reduction")

        hidden_dim = feature_dim // reduction
        layers: list[nn.Module] = [
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim, bias=False),
        ]
        if output_relu:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class VisualCLIPAdapter(nn.Module):
    """Frozen CLIP with trainable residual adapters on both modalities."""

    model_type = "visual_adapter"
    trainable_state_filename = "final_visual_adapter.pt"

    def __init__(
        self,
        clip_model: CLIPModel,
        text_features: torch.Tensor,
        classnames: list[str] | None = None,
        feature_dim: int | None = None,
        reduction: int = 4,
        alpha: float = 0.2,
        output_relu: bool = True,
        text_adapter_enabled: bool = True,
        text_reduction: int | None = None,
        text_alpha: float | None = None,
        text_output_relu: bool | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        resolved_text_alpha = alpha if text_alpha is None else float(text_alpha)
        if not 0.0 <= resolved_text_alpha <= 1.0:
            raise ValueError("text_alpha must be between 0 and 1")
        if text_features.ndim != 2 or text_features.shape[0] <= 0:
            raise ValueError("text_features must have shape (classes, feature_dim)")

        self.clip_model = clip_model.to(device)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        self.clip_model.eval()

        inferred_dim = int(text_features.shape[1])
        configured_dim = inferred_dim if feature_dim is None else int(feature_dim)
        if configured_dim != inferred_dim:
            raise ValueError(
                "feature_dim must match the width of text_features "
                f"({configured_dim} != {inferred_dim})"
            )
        projection_dim = int(self.clip_model.config.projection_dim)
        if configured_dim != projection_dim:
            raise ValueError(
                "Visual adapter feature_dim must match CLIP projection_dim "
                f"({configured_dim} != {projection_dim})"
            )

        self.projection_dim = configured_dim
        self.num_classes = int(text_features.shape[0])
        self.classnames = list(
            classnames
            if classnames is not None
            else [str(index) for index in range(self.num_classes)]
        )
        if len(self.classnames) != self.num_classes:
            raise ValueError("classnames and text_features must have equal length")
        self.reduction = int(reduction)
        self.alpha = float(alpha)
        self.output_relu = bool(output_relu)
        self.text_adapter_enabled = bool(text_adapter_enabled)
        self.text_reduction = int(
            self.reduction if text_reduction is None else text_reduction
        )
        self.text_alpha = resolved_text_alpha
        self.text_output_relu = bool(
            self.output_relu
            if text_output_relu is None
            else text_output_relu
        )
        self.device = device
        self.adapter = VisualAdapter(
            feature_dim=self.projection_dim,
            reduction=self.reduction,
            output_relu=self.output_relu,
        ).to(device)
        self.text_adapter = (
            VisualAdapter(
                feature_dim=self.projection_dim,
                reduction=self.text_reduction,
                output_relu=self.text_output_relu,
            ).to(device)
            if self.text_adapter_enabled
            else None
        )
        self.register_buffer(
            "text_features",
            F.normalize(text_features.detach().float(), dim=-1).to(device),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return frozen CLIP vectors, accepting precomputed vectors as input."""
        if images.ndim == 2:
            if images.shape[1] != self.projection_dim:
                raise ValueError(
                    "Precomputed CLIP features must have width "
                    f"{self.projection_dim}, got {images.shape[1]}."
                )
            return images.to(self.device).float()
        # Ordinary adapter training does not need a CLIP graph. Query-based
        # membership attacks optimize their input and therefore explicitly
        # request that graph through ``requires_grad``.
        gradient_enabled = bool(images.requires_grad)
        with torch.set_grad_enabled(gradient_enabled):
            features = cast(
                torch.Tensor,
                self.clip_model.get_image_features(
                    pixel_values=images.to(self.device)
                ),
            )
        return features.float()

    def mixed_image_features_from_image_features(
        self, image_features: torch.Tensor
    ) -> torch.Tensor:
        image_features = image_features.to(self.device).float()
        adapted_features = self.adapter(image_features)
        mixed_features = (
            self.alpha * adapted_features
            + (1.0 - self.alpha) * image_features
        )
        return F.normalize(mixed_features, dim=-1)

    def mixed_text_features(self) -> torch.Tensor:
        """Return normalized class vectors after the trainable text adapter."""
        if self.text_adapter is None:
            return self.text_features
        adapted_features = self.text_adapter(self.text_features)
        mixed_features = (
            self.text_alpha * adapted_features
            + (1.0 - self.text_alpha) * self.text_features
        )
        return F.normalize(mixed_features, dim=-1)

    def forward_from_image_features(
        self,
        image_features: torch.Tensor,
        return_intermediate: bool = False,
    ):
        mixed_features = self.mixed_image_features_from_image_features(
            image_features
        )
        mixed_text_features = self.mixed_text_features()
        logit_scale = self.clip_model.logit_scale.exp().detach()
        logits = logit_scale * mixed_features @ mixed_text_features.t()
        if return_intermediate:
            return logits, mixed_features, mixed_text_features
        return logits

    def forward(self, images: torch.Tensor, return_intermediate: bool = False):
        return self.forward_from_image_features(
            self.encode_images(images),
            return_intermediate=return_intermediate,
        )

    def get_audit_representation(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_audit_representation_from_image_features(
            self.encode_images(images), labels
        )

    def get_audit_representation_from_image_features(
        self,
        image_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = image_features.to(self.device).float()
        adapted = self.adapter(raw)
        mixed = F.normalize(
            self.alpha * adapted + (1.0 - self.alpha) * raw,
            dim=-1,
        )
        text_features = self.mixed_text_features()
        logits = (
            self.clip_model.logit_scale.exp().detach()
            * mixed
            @ text_features.t()
        )
        class_features = text_features[labels.to(text_features.device)]
        representation = torch.cat(
            (logits, raw, adapted, mixed, mixed * class_features), dim=1
        )
        return logits, representation

    def get_semantic_features(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = self.mixed_image_features_from_image_features(
            self.encode_images(images)
        )
        text_features = self.mixed_text_features()
        return mixed, text_features[labels.to(text_features.device)]

    def get_audit_key_parameter(self) -> torch.nn.Parameter:
        first_layer = cast(nn.Linear, self.adapter.net[0])
        return first_layer.weight

    def clone_adapter_only(self) -> "VisualCLIPAdapter":
        clone = VisualCLIPAdapter(
            clip_model=self.clip_model,
            text_features=self.text_features,
            classnames=self.classnames,
            feature_dim=self.projection_dim,
            reduction=self.reduction,
            alpha=self.alpha,
            output_relu=self.output_relu,
            text_adapter_enabled=self.text_adapter_enabled,
            text_reduction=self.text_reduction,
            text_alpha=self.text_alpha,
            text_output_relu=self.text_output_relu,
            device=self.device,
        )
        clone.adapter.load_state_dict(self.adapter.state_dict(), strict=True)
        if self.text_adapter is not None and clone.text_adapter is not None:
            clone.text_adapter.load_state_dict(
                self.text_adapter.state_dict(), strict=True
            )
        clone.text_features.copy_(self.text_features)
        clone.train(self.training)
        return clone

    def __deepcopy__(self, memo: dict) -> "VisualCLIPAdapter":
        if id(self) in memo:
            return memo[id(self)]
        clone = self.clone_adapter_only()
        memo[id(self)] = clone
        return clone
