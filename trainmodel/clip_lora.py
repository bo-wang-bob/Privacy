from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import contextmanager
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

from trainmodel.custom_clip import format_prompt_template, to_display_name
from trainmodel.clip_adapter import get_clip_adapter_prompt_template


class LoRALinear(nn.Module):
    """Frozen linear projection with an unmerged trainable LoRA branch."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        scaling: str = "sqrt_rank",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")
        scaling = str(scaling).lower()
        if scaling not in {"rank", "sqrt_rank"}:
            raise ValueError("LoRA scaling must be rank or sqrt_rank.")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling_mode = scaling
        denominator = self.rank if scaling == "rank" else math.sqrt(self.rank)
        self.scale = self.alpha / denominator
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            base.weight.new_empty((self.rank, base.in_features))
        )
        self.lora_B = nn.Parameter(
            base.weight.new_zeros((base.out_features, self.rank))
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        low_rank = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return base_output + self.scale * low_rank

    def merged_weight(self) -> torch.Tensor:
        """Return the effective dense weight without modifying the base layer."""
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A)

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)


def build_clip_lora_text_inputs(
    processor: CLIPProcessor,
    classnames: list[str],
    dataset_name: str | None,
    template: str | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize the fixed class prompts once; text features remain dynamic."""
    selected_template = template or get_clip_adapter_prompt_template(dataset_name)
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
    return {
        name: tensor.detach().clone()
        for name, tensor in encoded.items()
        if name in {"input_ids", "attention_mask"}
    }


def _layer_indices(selection: str | Iterable[int], count: int) -> list[int]:
    if isinstance(selection, str):
        normalized = selection.lower()
        if normalized == "all":
            return list(range(count))
        if normalized == "last_half":
            return list(range(count // 2, count))
        raise ValueError("CLIP-LoRA layers must be all, last_half, or a list.")
    indices = sorted({int(index) for index in selection})
    if not indices or indices[0] < 0 or indices[-1] >= count:
        raise ValueError("CLIP-LoRA layer indices are outside the encoder.")
    return indices


class CLIPLoRA(nn.Module):
    """One globally shared CLIP backbone with switchable LoRA parameters."""

    model_type = "clip_lora"
    trainable_state_filename = "final_clip_lora.pt"
    lora_aggregation = "factor_wise_linear_aggregation"

    def __init__(
        self,
        clip_model: CLIPModel,
        text_inputs: dict[str, torch.Tensor],
        classnames: list[str],
        encoder: str = "both",
        target_modules: Iterable[str] = ("q", "k", "v"),
        layers: str | Iterable[int] = "all",
        rank: int = 2,
        alpha: float = 1.0,
        dropout: float = 0.25,
        scaling: str = "sqrt_rank",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        encoder = str(encoder).lower()
        if encoder not in {"vision", "text", "both"}:
            raise ValueError("CLIP-LoRA encoder must be vision, text, or both.")
        aliases = {
            "q": "q_proj",
            "k": "k_proj",
            "v": "v_proj",
            "o": "out_proj",
            "q_proj": "q_proj",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "out_proj": "out_proj",
        }
        try:
            resolved_targets = tuple(
                dict.fromkeys(aliases[str(name).lower()] for name in target_modules)
            )
        except KeyError as error:
            raise ValueError("CLIP-LoRA targets must be q, k, v, or o.") from error
        if not resolved_targets:
            raise ValueError("CLIP-LoRA requires at least one target projection.")
        if not text_inputs or "input_ids" not in text_inputs:
            raise ValueError("CLIP-LoRA requires tokenized class text inputs.")
        if not classnames:
            raise ValueError("CLIP-LoRA requires at least one class name.")

        self.clip_model = clip_model.to(device)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        self.encoder = encoder
        self.target_modules = resolved_targets
        self.layer_selection = layers
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.lora_dropout = float(dropout)
        self.scaling = str(scaling).lower()
        self.device = device
        self.classnames = list(classnames)
        self.num_classes = len(self.classnames)
        self.projection_dim = int(self.clip_model.config.projection_dim)

        for name, tensor in text_inputs.items():
            self.register_buffer(
                f"text_{name}", tensor.detach().clone().to(device), persistent=True
            )

        self.injected_modules: list[str] = []
        if encoder in {"vision", "both"}:
            self._inject_encoder(
                "vision",
                self.clip_model.vision_model.encoder.layers,
            )
        if encoder in {"text", "both"}:
            self._inject_encoder(
                "text",
                self.clip_model.text_model.encoder.layers,
            )
        if not self.injected_modules:
            raise RuntimeError("CLIP-LoRA did not find any attention projections.")
        object.__setattr__(
            self,
            "_global_lora_parameters",
            self._current_lora_parameters(),
        )
        object.__setattr__(self, "_active_client_model", None)

    def _inject_encoder(self, prefix: str, blocks: nn.ModuleList) -> None:
        for layer_index in _layer_indices(self.layer_selection, len(blocks)):
            attention = blocks[layer_index].self_attn
            for projection_name in self.target_modules:
                projection = getattr(attention, projection_name)
                if not isinstance(projection, nn.Linear):
                    raise TypeError(
                        f"Expected {prefix} layer {layer_index} "
                        f"{projection_name} to be nn.Linear."
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
                    f"{prefix}.{layer_index}.self_attn.{projection_name}"
                )

    def _lora_parameter_targets(
        self,
    ) -> dict[str, tuple[LoRALinear, str]]:
        targets: dict[str, tuple[LoRALinear, str]] = {}
        for module_name, module in self.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            targets[f"{module_name}.lora_A"] = (module, "lora_A")
            targets[f"{module_name}.lora_B"] = (module, "lora_B")
        return targets

    def _current_lora_parameters(self) -> dict[str, nn.Parameter]:
        return {
            name: getattr(module, attribute)
            for name, (module, attribute) in self._lora_parameter_targets().items()
        }

    def _bind_client_lora(
        self,
        client_model: "ClientCLIPLoRA",
        parameters: dict[str, nn.Parameter],
    ) -> None:
        active = self._active_client_model
        if active is not None and active is not client_model:
            raise RuntimeError(
                "The shared CLIP-LoRA backbone is already bound to another client."
            )
        targets = self._lora_parameter_targets()
        if set(parameters) != set(targets):
            raise ValueError("Client LoRA parameters do not match the shared backbone.")
        for name, parameter in parameters.items():
            expected = self._global_lora_parameters[name]
            if parameter.shape != expected.shape:
                raise ValueError(f"Client LoRA parameter {name!r} has the wrong shape.")
        for name, parameter in parameters.items():
            module, attribute = targets[name]
            setattr(module, attribute, parameter)
        object.__setattr__(self, "_active_client_model", client_model)

    def _release_client_lora(self, client_model: "ClientCLIPLoRA") -> None:
        if self._active_client_model is not client_model:
            raise RuntimeError("Cannot release a LoRA client that is not active.")
        targets = self._lora_parameter_targets()
        for name, parameter in self._global_lora_parameters.items():
            module, attribute = targets[name]
            setattr(module, attribute, parameter)
        object.__setattr__(self, "_active_client_model", None)

    def create_client_model(self, client_id: int) -> "ClientCLIPLoRA":
        """Create an independent LoRA state backed by this shared CLIP."""
        if self._active_client_model is not None:
            raise RuntimeError("Create clients only while the shared CLIP is unbound.")
        return ClientCLIPLoRA(self, client_id=client_id)

    def _text_arguments(self) -> dict[str, torch.Tensor]:
        arguments = {"input_ids": self.text_input_ids}
        attention_mask = getattr(self, "text_attention_mask", None)
        if attention_mask is not None:
            arguments["attention_mask"] = attention_mask
        return arguments

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 2:
            raise ValueError(
                "CLIP-LoRA changes the vision encoder and cannot consume "
                "precomputed CLIP features."
            )
        return cast(
            torch.Tensor,
            self.clip_model.get_image_features(
                pixel_values=images.to(self.device)
            ),
        ).float()

    def encode_class_texts(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.clip_model.get_text_features(**self._text_arguments()),
        ).float()

    def normalized_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = F.normalize(self.encode_images(images), dim=-1)
        text_features = F.normalize(self.encode_class_texts(), dim=-1)
        return image_features, text_features

    def forward(self, images: torch.Tensor, return_intermediate: bool = False):
        image_features, text_features = self.normalized_features(images)
        logits = (
            self.clip_model.logit_scale.exp().detach()
            * image_features
            @ text_features.t()
        )
        if return_intermediate:
            return logits, image_features, text_features
        return logits

    def get_semantic_features(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features, text_features = self.normalized_features(images)
        return image_features, text_features[labels.to(text_features.device)]

    def get_audit_representation(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features, text_features = self.normalized_features(images)
        logits = (
            self.clip_model.logit_scale.exp().detach()
            * image_features
            @ text_features.t()
        )
        class_features = text_features[labels.to(text_features.device)]
        return logits, torch.cat(
            (logits, image_features, class_features, image_features * class_features),
            dim=1,
        )

    def get_audit_key_parameter(self) -> torch.nn.Parameter:
        for name, parameter in self.named_parameters():
            if "text_model" in name and name.endswith("lora_A"):
                return parameter
        for name, parameter in self.named_parameters():
            if name.endswith("lora_A"):
                return parameter
        raise RuntimeError("CLIP-LoRA has no trainable input projection.")

    def get_projres_attack_surface(
        self, parameter_name: str | None = None
    ) -> tuple[str, LoRALinear]:
        """Return the vision LoRA down-projection observed by ProjRes.

        Deng et al. construct the projection subspace from the gradient of a
        trainable linear layer. For LoRA, ``lora_A`` is exactly the
        down-projection whose input is the attention hidden representation.
        """
        modules = dict(self.named_modules())
        if parameter_name is not None:
            normalized = str(parameter_name)
            if not normalized.endswith(".lora_A"):
                raise ValueError("CLIP-LoRA ProjRes must attack a lora_A matrix.")
            module_name = normalized[: -len(".lora_A")]
            module = modules.get(module_name)
            if not isinstance(module, LoRALinear):
                raise ValueError(
                    f"ProjRes parameter {normalized!r} is not a LoRA projection."
                )
            if "vision_model" not in module_name:
                raise ValueError(
                    "CLIP image membership ProjRes requires a vision LoRA layer."
                )
            return normalized, module

        candidates = [
            (f"{name}.lora_A", module)
            for name, module in modules.items()
            if isinstance(module, LoRALinear)
            and "vision_model" in name
            and name.endswith("q_proj")
        ]
        if not candidates:
            candidates = [
                (f"{name}.lora_A", module)
                for name, module in modules.items()
                if isinstance(module, LoRALinear) and "vision_model" in name
            ]
        if not candidates:
            raise ValueError(
                "CLIP-LoRA ProjRes requires LoRA in the vision encoder."
            )
        return candidates[0]

    @torch.no_grad()
    def get_projres_representations(
        self,
        images: torch.Tensor,
        parameter_name: str | None = None,
        token_reduction: str = "cls",
    ) -> tuple[torch.Tensor, int]:
        """Capture sample representations entering the attacked LoRA layer.

        The gradient is formed from every sequence token. For sample-level
        scoring we use either the CLIP class token or the mean token, while
        returning the full token count for the paper's rank-condition audit.
        """
        attacked_parameter, module = self.get_projres_attack_surface(
            parameter_name
        )
        module_name = attacked_parameter[: -len(".lora_A")]
        captured: list[torch.Tensor] = []

        def capture_input(_module, arguments) -> None:
            if not arguments or not isinstance(arguments[0], torch.Tensor):
                raise RuntimeError(
                    f"LoRA module {module_name} did not receive tensor inputs."
                )
            captured.append(arguments[0].detach())

        hook = module.register_forward_pre_hook(capture_input)
        was_training = self.training
        self.eval()
        try:
            self.clip_model.get_image_features(
                pixel_values=images.to(self.device)
            )
        finally:
            hook.remove()
            self.train(was_training)
        if len(captured) != 1 or captured[0].ndim != 3:
            shapes = [tuple(value.shape) for value in captured]
            raise RuntimeError(
                f"Expected one [batch,tokens,hidden] LoRA input; found {shapes}."
            )
        hidden = captured[0]
        token_count = int(hidden.shape[1])
        reduction = str(token_reduction).lower()
        if reduction == "cls":
            representation = hidden[:, 0]
        elif reduction == "mean":
            representation = hidden.mean(dim=1)
        else:
            raise ValueError("ProjRes token_reduction must be cls or mean.")
        return representation.detach().float(), token_count

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        names = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: tensor.detach().clone()
            for name, tensor in self.state_dict().items()
            if name in names
        }


class ClientCLIPLoRA(nn.Module):
    """Per-client LoRA factors executed through one shared frozen CLIP.

    The wrapper registers only this client's A/B factors.  A client session
    temporarily installs those Parameters into the shared LoRA work slots, so
    autograd and the optimizer update the owning client directly.  Releasing
    the session restores the server's global LoRA slots and prevents state
    leakage between sequential clients.
    """

    model_type = "clip_lora"
    trainable_state_filename = CLIPLoRA.trainable_state_filename
    lora_aggregation = CLIPLoRA.lora_aggregation
    client_scoped_lora = True

    def __init__(self, shared_model: CLIPLoRA, client_id: int) -> None:
        super().__init__()
        if shared_model._active_client_model is not None:
            raise RuntimeError("The shared CLIP must be unbound when adding a client.")
        object.__setattr__(self, "_shared_model", shared_model)
        self.client_id = int(client_id)
        initial_state = shared_model.lora_state_dict()
        self._lora_names = tuple(initial_state)
        self._local_lora_parameters = nn.ParameterList(
            [
                nn.Parameter(initial_state[name].detach().clone())
                for name in self._lora_names
            ]
        )
        self._session_depth = 0

    @property
    def shared_model(self) -> CLIPLoRA:
        return self._shared_model

    @property
    def clip_model(self) -> CLIPModel:
        return self._shared_model.clip_model

    @property
    def device(self) -> torch.device:
        return self._shared_model.device

    @property
    def classnames(self) -> list[str]:
        return self._shared_model.classnames

    @property
    def num_classes(self) -> int:
        return self._shared_model.num_classes

    def _named_lora_parameters(self) -> dict[str, nn.Parameter]:
        return dict(zip(self._lora_names, self._local_lora_parameters))

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        del recurse
        seen: set[int] = set()
        for name, parameter in self._named_lora_parameters().items():
            if remove_duplicate and id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield f"{prefix}.{name}" if prefix else name, parameter

    def export_trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self._named_lora_parameters().items()
        }

    def state_dict(
        self,
        destination=None,
        prefix: str = "",
        keep_vars: bool = False,
    ):
        """Expose only canonical LoRA keys, never the shared CLIP backbone."""
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, parameter in self._named_lora_parameters().items():
            destination[prefix + name] = (
                parameter if keep_vars else parameter.detach()
            )
        return destination

    def load_trainable_state(
        self,
        state: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        parameters = self._named_lora_parameters()
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        if strict and (missing or unexpected):
            raise ValueError(
                "Client LoRA state mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        with torch.no_grad():
            for name in set(parameters) & set(state):
                source = state[name]
                target = parameters[name]
                if source.shape != target.shape:
                    raise ValueError(
                        f"Client LoRA parameter {name!r} has shape "
                        f"{tuple(source.shape)}, expected {tuple(target.shape)}."
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        if assign:
            raise ValueError("Client CLIP-LoRA does not support assign=True.")
        state = dict(state_dict)
        names = set(self._lora_names)
        missing = sorted(names - set(state))
        unexpected = sorted(set(state) - names)
        self.load_trainable_state(state, strict=strict)
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)

    @contextmanager
    def use_shared_model(self):
        outermost = self._session_depth == 0
        if outermost:
            self._shared_model._bind_client_lora(self, self._named_lora_parameters())
        self._session_depth += 1
        try:
            self._shared_model.train(self.training)
            yield self._shared_model
        finally:
            self._session_depth -= 1
            if outermost:
                self._shared_model._release_client_lora(self)

    def _call_shared(self, method_name: str, *args, **kwargs):
        with self.use_shared_model() as shared:
            return getattr(shared, method_name)(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._call_shared("forward", *args, **kwargs)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self._call_shared("encode_images", images)

    def get_semantic_features(self, images: torch.Tensor, labels: torch.Tensor):
        return self._call_shared("get_semantic_features", images, labels)

    def get_audit_representation(self, images: torch.Tensor, labels: torch.Tensor):
        return self._call_shared("get_audit_representation", images, labels)

    def get_audit_key_parameter(self) -> torch.nn.Parameter:
        return self._call_shared("get_audit_key_parameter")
