from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from transformers import AutoModel

from trainmodel.clip_lora import LoRALinear
from trainmodel.transformer_adapter import (
    ClientTransformerAdapterClassifier,
    TransformerAdapterClassifier,
)


def _bert_layer_indices(
    selection: str | Iterable[int], number_of_layers: int
) -> list[int]:
    if isinstance(selection, str):
        normalized = selection.lower()
        if normalized == "all":
            return list(range(number_of_layers))
        if normalized == "last_half":
            return list(range(number_of_layers // 2, number_of_layers))
        raise ValueError("BERT-LoRA layers must be all, last_half, or a list.")
    indices = sorted({int(index) for index in selection})
    if not indices:
        raise ValueError("BERT-LoRA layer selection must not be empty.")
    if indices[0] < 0 or indices[-1] >= number_of_layers:
        raise ValueError("BERT-LoRA layer indices are outside the encoder.")
    return indices


class TransformerLoRAClassifier(TransformerAdapterClassifier):
    """Frozen BERT with client-scoped attention LoRA factors and task head."""

    model_type = "bert_lora"
    trainable_state_filename = "final_transformer_lora.pt"
    lora_aggregation = "factor_wise_linear_aggregation"
    client_scoped_parameters = True

    def __init__(
        self,
        model_path: str,
        num_classes: int = 2,
        target_modules: Iterable[str] = ("query", "value"),
        layers: str | Iterable[int] = "all",
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        scaling: str = "rank",
        classifier_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        nn.Module.__init__(self)
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if not 0.0 <= classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1).")
        aliases = {
            "q": "query",
            "query": "query",
            "k": "key",
            "key": "key",
            "v": "value",
            "value": "value",
        }
        try:
            resolved_targets = tuple(
                dict.fromkeys(aliases[str(name).lower()] for name in target_modules)
            )
        except KeyError as error:
            raise ValueError(
                "BERT-LoRA target_modules must contain query, key, or value."
            ) from error
        if not resolved_targets:
            raise ValueError("BERT-LoRA requires at least one target projection.")

        self.architecture = "bert"
        self.model_path = str(model_path)
        self.num_classes = int(num_classes)
        self.classnames = [str(index) for index in range(num_classes)]
        self.target_modules = resolved_targets
        self.layer_selection = layers
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.lora_dropout = float(dropout)
        self.scaling = str(scaling).lower()
        self.classifier_dropout_probability = float(classifier_dropout)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.device = device

        self.backbone = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.backbone.config.hidden_size)
        initializer_std = float(
            getattr(self.backbone.config, "initializer_range", 0.02)
        )
        self.injected_modules: list[str] = []
        self._inject_lora()
        self.num_lora_layers = len(
            {name.split(".")[3] for name in self.injected_modules}
        )
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=initializer_std)
        nn.init.zeros_(self.classifier.bias)

        if self.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            enable_inputs = getattr(self.backbone, "enable_input_require_grads", None)
            if callable(enable_inputs):
                enable_inputs()

        self.to(device)
        object.__setattr__(
            self,
            "_global_trainable_parameters",
            self._current_trainable_parameters(),
        )
        object.__setattr__(self, "_active_client_model", None)

    def _inject_lora(self) -> None:
        blocks = self.backbone.encoder.layer
        for layer_index in _bert_layer_indices(
            self.layer_selection, len(blocks)
        ):
            attention = blocks[layer_index].attention.self
            for projection_name in self.target_modules:
                projection = getattr(attention, projection_name)
                if not isinstance(projection, nn.Linear):
                    raise TypeError(
                        f"Expected BERT layer {layer_index} {projection_name} "
                        "to be nn.Linear."
                    )
                setattr(
                    attention,
                    projection_name,
                    LoRALinear(
                        projection,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.lora_dropout,
                        scaling=self.scaling,
                    ),
                )
                self.injected_modules.append(
                    f"backbone.encoder.layer.{layer_index}.attention.self."
                    f"{projection_name}"
                )
        if not self.injected_modules:
            raise RuntimeError("BERT-LoRA did not find any attention projections.")

    def create_client_model(
        self, client_id: int
    ) -> "ClientTransformerLoRAClassifier":
        if self._active_client_model is not None:
            raise RuntimeError("Create clients only while BERT-LoRA is unbound.")
        return ClientTransformerLoRAClassifier(self, client_id)

    def train(self, mode: bool = True):
        super().train(mode)
        # TransformerAdapterClassifier disables frozen backbone dropout. LoRA
        # dropout is part of the trainable branch and must retain train/eval mode.
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.dropout.train(mode)
        return self

    def get_projres_attack_surface(
        self, parameter_name: str | None = None
    ) -> tuple[str, LoRALinear]:
        surfaces = [
            (f"{name}.lora_A", module)
            for name, module in self.named_modules()
            if isinstance(module, LoRALinear)
        ]
        if not surfaces:
            raise RuntimeError("BERT has no LoRA attack surface.")
        if parameter_name is None:
            return surfaces[0]
        matches = [surface for surface in surfaces if surface[0] == parameter_name]
        if not matches:
            available = ", ".join(name for name, _ in surfaces)
            raise ValueError(
                f"ProjRes parameter {parameter_name!r} is not a BERT LoRA "
                f"lora_A matrix; available surfaces: {available}."
            )
        return matches[0]

    @staticmethod
    def resolve_projres_token_reduction(token_reduction: str) -> str:
        """Resolve the sample view used for a token-wise LoRA projection.

        Query/Value LoRA is applied to every active BERT token. In
        particular, the CLS input to the first attention layer is constant
        across examples because it has not yet attended to the sentence.
        The mask-weighted token mean remains in the span of the actual layer
        inputs that form the uploaded gradient and yields a sample-specific
        representation, so it is the BERT-LoRA automatic default.
        """
        reduction = str(token_reduction).lower()
        if reduction == "auto":
            reduction = "mean"
        if reduction not in {"cls", "last", "mean"}:
            raise ValueError(
                "BERT-LoRA ProjRes token_reduction must be auto, cls, last, "
                "or mean."
            )
        return reduction

    @torch.no_grad()
    def get_projres_representations(
        self,
        packed_inputs: torch.Tensor,
        parameter_name: str | None = None,
        token_reduction: str = "auto",
    ) -> tuple[torch.Tensor, int]:
        """Capture a sample view of inputs to the attacked LoRA projection.

        ``mean`` is attention-mask weighted and therefore excludes padding.
        It represents all active token inputs that contribute to the
        token-wise Query/Value LoRA gradient. ``auto`` resolves to ``mean``;
        explicit ``cls`` and ``last`` remain available for diagnostics.
        """
        _, attacked_layer = self.get_projres_attack_surface(parameter_name)
        captured: list[torch.Tensor] = []

        def capture_input(_module, args) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise TypeError("BERT LoRA projection received invalid input.")
            captured.append(args[0])

        was_training = self.training
        self.eval()
        handle = attacked_layer.register_forward_pre_hook(capture_input)
        try:
            self.hidden_features(packed_inputs)
        finally:
            handle.remove()
            self.train(was_training)
        if len(captured) != 1:
            raise RuntimeError(
                "ProjRes expected the attacked BERT LoRA layer to execute once."
            )
        hidden = captured[0]
        _, attention_mask = self.unpack_inputs(packed_inputs)
        attention_mask = attention_mask.to(hidden.device)
        reduction = self.resolve_projres_token_reduction(token_reduction)
        if reduction == "cls":
            representations = hidden[:, 0]
        elif reduction == "last":
            positions = attention_mask.sum(dim=1).sub(1).clamp_min(0)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            representations = hidden[rows, positions]
        elif reduction == "mean":
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            representations = (hidden * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1)
        return representations, int(attention_mask.sum().item())


class ClientTransformerLoRAClassifier(ClientTransformerAdapterClassifier):
    """CPU-resident BERT LoRA factors and classifier for one client."""

    model_type = "bert_lora"
    trainable_state_filename = TransformerLoRAClassifier.trainable_state_filename
    lora_aggregation = TransformerLoRAClassifier.lora_aggregation
    client_scoped_parameters = True
    client_scoped_lora = True

    def __init__(
        self,
        shared_model: TransformerLoRAClassifier,
        client_id: int,
    ) -> None:
        super().__init__(shared_model, client_id)
