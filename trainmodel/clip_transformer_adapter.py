"""BERT-style residual adapters inside CLIP's visual Transformer."""
from __future__ import annotations

import torch
from trainmodel.clip_tokens import CLIPTokenInputMixin, ClientCLIPTokenInputMixin
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel

from trainmodel.transformer_adapter import (
    BottleneckAdapter,
    ClientTransformerAdapterClassifier,
    TransformerAdapterClassifier,
    TransformerBlockWithAdapter,
)


class CLIPTransformerAdapter(CLIPTokenInputMixin, TransformerAdapterClassifier):
    """Reuse Transformer Adapter client ownership with an image/text forward.

    Only the visual block adapters are trainable. Class prompts are stored as
    token IDs; both modalities are encoded on demand, without a feature cache.
    The inherited state/session helpers keep a single backbone shared among
    clients, each with independent CPU-resident Adapter parameters.
    """

    model_type = "clip_adapter"
    adapter_variant = "transformer"
    trainable_state_filename = "final_clip_transformer_adapter.pt"
    projres_token_aggregate = True
    text_adapter_enabled = False

    def __init__(
        self,
        clip_model: CLIPModel,
        text_inputs: dict[str, torch.Tensor],
        classnames: list[str],
        reduction: int = 2,
        activation: str = "relu",
        zero_init_up: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # The parent constructor loads a text classifier; only its generic
        # parameter ownership methods are reused here.
        nn.Module.__init__(self)
        if not classnames or "input_ids" not in text_inputs:
            raise ValueError("CLIP Transformer Adapter requires tokenized class prompts.")
        if text_inputs["input_ids"].ndim != 2 or text_inputs["input_ids"].shape[0] != len(classnames):
            raise ValueError("Class prompt input_ids must have shape [classes, tokens].")
        self.clip_model = clip_model
        self.device = device
        self.architecture = "clip"
        self.classnames = list(classnames)
        self.num_classes = len(classnames)
        self.projection_dim = int(clip_model.config.projection_dim)
        self.hidden_size = int(clip_model.config.vision_config.hidden_size)
        self.reduction = int(reduction)
        self.activation_name = str(activation).lower()
        self.zero_init_up = bool(zero_init_up)
        self.gradient_checkpointing = False
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        blocks = self.clip_model.vision_model.encoder.layers
        if any(isinstance(block, TransformerBlockWithAdapter) for block in blocks):
            raise ValueError("CLIP visual blocks already contain Adapters.")
        for index, block in enumerate(list(blocks)):
            blocks[index] = TransformerBlockWithAdapter(
                base_layer=block,
                hidden_size=self.hidden_size,
                reduction=self.reduction,
                activation=self.activation_name,
                initializer_std=0.02,
                zero_init_up=self.zero_init_up,
            )
        self.num_adapter_layers = len(blocks)
        if not self.num_adapter_layers:
            raise ValueError("CLIP must have at least one visual Transformer block.")
        for name in ("input_ids", "attention_mask"):
            if name in text_inputs:
                tensor = text_inputs[name]
                if tensor.shape != text_inputs["input_ids"].shape:
                    raise ValueError("Class prompt masks must match input_ids.")
                self.register_buffer(f"text_{name}", tensor.detach().clone().long())
        self.to(device)
        object.__setattr__(self, "_global_trainable_parameters", self._current_trainable_parameters())
        object.__setattr__(self, "_active_client_model", None)
        self.train()

    def create_client_model(self, client_id: int) -> "ClientCLIPTransformerAdapter":
        if self._active_client_model is not None:
            raise RuntimeError("Create clients only while the backbone is unbound.")
        return ClientCLIPTransformerAdapter(self, client_id)

    @staticmethod
    def _reject_feature_checkpoint(state) -> None:
        if any(name.startswith(("adapter.net.", "text_adapter.net.")) for name in state):
            raise ValueError(
                "A legacy feature Adapter checkpoint cannot initialize the Transformer variant; "
                "select variant=feature with its original configuration."
            )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._reject_feature_checkpoint(state_dict)
        if assign:
            raise ValueError("Shared CLIP Transformer Adapter does not support assign=True.")
        return super().load_state_dict(state_dict, strict=strict)

    def load_trainable_state(self, state, *, strict: bool = True) -> None:
        self._reject_feature_checkpoint(state)
        super().load_trainable_state(state, strict=strict)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        # eval disables frozen dropout, but does not detach the visual graph.
        self.clip_model.eval()
        for module in self.clip_model.modules():
            if isinstance(module, BottleneckAdapter):
                module.train(mode)
        return self

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("CLIP Transformer Adapter requires raw [batch, channels, height, width] images; precomputed features are unsupported.")
        return self.clip_model.get_image_features(pixel_values=images.to(self.device)).float()

    @torch.no_grad()
    def encode_class_texts(self) -> torch.Tensor:
        inputs = {"input_ids": self.text_input_ids}
        if hasattr(self, "text_attention_mask"):
            inputs["attention_mask"] = self.text_attention_mask
        return self.clip_model.get_text_features(**inputs).float()

    def normalized_features(self, images: torch.Tensor):
        return (
            F.normalize(self.encode_images(images), dim=-1),
            F.normalize(self.encode_class_texts(), dim=-1),
        )

    def forward(self, images: torch.Tensor, return_intermediate: bool = False):
        image_features, text_features = self.normalized_features(images)
        logits = self.clip_model.logit_scale.exp().detach() * image_features @ text_features.t()
        return (logits, image_features, text_features) if return_intermediate else logits

    def get_semantic_features(self, images: torch.Tensor, labels: torch.Tensor):
        image_features, text_features = self.normalized_features(images)
        return image_features, text_features[labels.to(text_features.device)]

    def get_audit_representation(self, images: torch.Tensor, labels: torch.Tensor):
        logits, image_features, text_features = self(images, return_intermediate=True)
        class_features = text_features[labels.to(text_features.device)]
        return logits, torch.cat((logits, image_features, image_features * class_features), dim=1)

    def get_audit_key_parameter(self) -> nn.Parameter:
        # Keep other audits' key parameter independent of the ProjRes default.
        return super().get_projres_attack_surface()[1].weight

    def get_projres_attack_surface(
        self, parameter_name: str | None = None
    ) -> tuple[str, nn.Linear]:
        """Default to the final visual Adapter, whose output is CLS-pooled.

        Only its CLS contributes to the classification loss: there is no later
        attention block that mixes its patch outputs into CLS. Explicit down
        weights still use the shared validation for layer ablations.
        """
        if parameter_name is None:
            parameter_name = (
                "clip_model.vision_model.encoder.layers."
                f"{self.num_adapter_layers - 1}.adapter.down.weight"
            )
        return super().get_projres_attack_surface(parameter_name)

    @staticmethod
    def resolve_projres_token_reduction(token_reduction: str = "auto") -> str:
        reduction = str(token_reduction).lower()
        if reduction == "auto":
            reduction = "cls"
        if reduction not in {"mean", "cls"}:
            raise ValueError("CLIP Transformer Adapter token_reduction must be auto, mean, or cls.")
        return reduction

    @torch.no_grad()
    def get_projres_representations(
        self,
        images: torch.Tensor,
        parameter_name: str | None = None,
        token_reduction: str = "auto",
    ) -> tuple[torch.Tensor, int]:
        reduction = self.resolve_projres_token_reduction(token_reduction)
        _, layer = self.get_projres_attack_surface(parameter_name)
        captured = []
        hook = layer.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
        was_training = self.training
        self.eval()
        try:
            self.encode_images(images)
        finally:
            hook.remove()
            self.train(was_training)
        if len(captured) != 1:
            raise RuntimeError("The attacked Adapter must execute exactly once.")
        hidden = captured[0]
        # Inputs are contextualized block outputs; CLS is image-dependent here.
        representations = hidden.mean(dim=1) if reduction == "mean" else hidden[:, 0]
        # Keep the shared API's total input-token count. This is a layout count,
        # not the number of gradient-active tokens (only CLS at the last layer).
        # FedAvg continues to use no batch-rank bound in the shared auditor.
        return representations, int(hidden.shape[0] * hidden.shape[1])


class ClientCLIPTransformerAdapter(ClientCLIPTokenInputMixin, ClientTransformerAdapterClassifier):
    trainable_state_filename = CLIPTransformerAdapter.trainable_state_filename
    adapter_variant = CLIPTransformerAdapter.adapter_variant
    text_adapter_enabled = False

    def load_trainable_state(self, state, *, strict: bool = True) -> None:
        CLIPTransformerAdapter._reject_feature_checkpoint(state)
        super().load_trainable_state(state, strict=strict)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self._call_shared("encode_images", images)
