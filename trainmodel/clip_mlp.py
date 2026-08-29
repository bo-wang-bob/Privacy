from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIPImageMLP(nn.Module):
    """Frozen CLIP image encoder followed by a two-layer MLP classifier."""

    model_type = "clip_mlp"
    trainable_state_filename = "final_mlp.pt"

    def __init__(
        self,
        clip_model: CLIPModel,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        normalize_features: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if num_classes <= 0 or hidden_dim <= 0:
            raise ValueError("num_classes and hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.clip_model = clip_model.to(device)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        self.clip_model.eval()

        projection_dim = int(self.clip_model.config.projection_dim)
        self.projection_dim = projection_dim
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.normalize_features = bool(normalize_features)
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        ).to(device)

    def train(self, mode: bool = True):
        """Train the MLP while keeping the frozen CLIP encoder in eval mode."""
        super().train(mode)
        self.clip_model.eval()
        return self

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # Federated CLIP-MLP runs may precompute the frozen encoder output once
        # at startup. This fast path makes cached datasets transparent to
        # training, evaluation, and privacy auditing.
        if images.ndim == 2:
            if images.shape[1] != self.projection_dim:
                raise ValueError(
                    "Precomputed CLIP features must have width "
                    f"{self.projection_dim}, got {images.shape[1]}."
                )
            return images.to(self.device)
        images = images.to(self.device)
        features = cast(
            torch.Tensor,
            self.clip_model.get_image_features(pixel_values=images),
        )
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return features

    def hidden_features(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_images(images)
        return self.hidden_features_from_image_features(image_features)

    def hidden_features_from_image_features(
        self, image_features: torch.Tensor
    ) -> torch.Tensor:
        """Apply the first MLP block to already encoded CLIP features."""
        return self.classifier[:3](image_features.to(self.device))

    def forward_from_image_features(
        self,
        image_features: torch.Tensor,
        return_intermediate: bool = False,
    ):
        """Run only the trainable MLP, allowing large audits to cache CLIP."""
        image_features = image_features.to(self.device)
        hidden_features = self.hidden_features_from_image_features(image_features)
        logits = self.classifier[3](hidden_features)
        if return_intermediate:
            return logits, image_features, hidden_features
        return logits

    def forward(self, images: torch.Tensor, return_intermediate: bool = False):
        image_features = self.encode_images(images)
        return self.forward_from_image_features(
            image_features, return_intermediate=return_intermediate
        )

    def get_semantic_features(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned sample-hidden and label-conditioned class features."""
        hidden = self.hidden_features(images)
        final_layer = cast(nn.Linear, self.classifier[3])
        class_features = final_layer.weight[labels.to(final_layer.weight.device)]
        return F.normalize(hidden, dim=1), F.normalize(class_features, dim=1)

    def get_audit_representation(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and an MLP-aware representation for learned attacks."""
        return self.get_audit_representation_from_image_features(
            self.encode_images(images), labels
        )

    def get_audit_representation_from_image_features(
        self,
        image_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the learned-attack representation without re-running CLIP."""
        hidden = self.hidden_features_from_image_features(image_features)
        final_layer = cast(nn.Linear, self.classifier[3])
        logits = final_layer(hidden)
        class_features = final_layer.weight[labels.to(final_layer.weight.device)]
        conditioned = hidden * class_features
        return logits, torch.cat((logits, hidden, conditioned), dim=1)

    def get_audit_key_parameter(self) -> torch.nn.Parameter:
        """Expose class decision vectors for the PromptMIA-style active probe."""
        return cast(nn.Linear, self.classifier[3]).weight

    def get_projres_attack_surface(
        self, parameter_name: str | None = None
    ) -> tuple[str, nn.Linear]:
        """Expose the first MLP projection used by exact-batch ProjRes."""
        name = "classifier.0.weight"
        layer = cast(nn.Linear, self.classifier[0])
        if parameter_name not in {None, name}:
            raise ValueError(
                f"ProjRes parameter {parameter_name!r} must be {name!r}."
            )
        return name, layer

    @torch.no_grad()
    def get_projres_representations(
        self,
        images: torch.Tensor,
        parameter_name: str | None = None,
        token_reduction: str = "none",
    ) -> tuple[torch.Tensor, int]:
        """Return CLIP image vectors entering the attacked MLP projection."""
        del token_reduction
        self.get_projres_attack_surface(parameter_name)
        features = self.encode_images(images)
        return features, int(features.shape[0])

    def clone_head_only(self) -> "CLIPImageMLP":
        """Clone trainable state while sharing the immutable CLIP backbone."""
        clone = CLIPImageMLP(
            clip_model=self.clip_model,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            normalize_features=self.normalize_features,
            device=self.device,
        )
        clone.classifier.load_state_dict(self.classifier.state_dict(), strict=True)
        clone.train(self.training)
        return clone

    def __deepcopy__(self, memo: dict) -> "CLIPImageMLP":
        if id(self) in memo:
            return memo[id(self)]
        clone = self.clone_head_only()
        memo[id(self)] = clone
        return clone
