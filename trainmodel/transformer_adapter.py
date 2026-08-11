from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class BottleneckAdapter(nn.Module):
    """Residual Adapter from Deng et al., Eq. (22)-(23)."""

    def __init__(
        self,
        hidden_size: int,
        reduction: int = 2,
        activation: str = "relu",
        initializer_std: float = 0.02,
        zero_init_up: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or reduction <= 0:
            raise ValueError("hidden_size and reduction must be positive.")
        if hidden_size % reduction != 0:
            raise ValueError("hidden_size must be divisible by reduction.")
        if initializer_std <= 0:
            raise ValueError("initializer_std must be positive.")
        activation = str(activation).lower()
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }
        if activation not in activations:
            raise ValueError("activation must be relu or gelu.")

        bottleneck_size = hidden_size // reduction
        self.hidden_size = int(hidden_size)
        self.bottleneck_size = int(bottleneck_size)
        self.reduction = int(reduction)
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = activations[activation]
        self.up = nn.Linear(bottleneck_size, hidden_size)
        nn.init.normal_(self.down.weight, mean=0.0, std=initializer_std)
        nn.init.zeros_(self.down.bias)
        if zero_init_up:
            # Start as an exact identity residual branch.  Randomly
            # initializing both projections perturbs the pretrained
            # representation at every Transformer block before the task head
            # has learned anything, which is especially harmful in one-batch
            # FedSGD.  The down projection still receives gradients as soon as
            # the up projection moves away from zero.
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.normal_(self.up.weight, mean=0.0, std=initializer_std)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(self.activation(self.down(hidden_states)))


class TransformerBlockWithAdapter(nn.Module):
    """Preserve a Hugging Face block API and adapt its hidden-state output."""

    def __init__(
        self,
        base_layer: nn.Module,
        hidden_size: int,
        reduction: int,
        activation: str,
        initializer_std: float,
        zero_init_up: bool,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            reduction=reduction,
            activation=activation,
            initializer_std=initializer_std,
            zero_init_up=zero_init_up,
        )

    def forward(self, *args, **kwargs):
        outputs = self.base_layer(*args, **kwargs)
        if isinstance(outputs, tuple):
            if not outputs or not isinstance(outputs[0], torch.Tensor):
                raise TypeError("Transformer block returned an invalid tuple.")
            return (self.adapter(outputs[0]), *outputs[1:])
        if isinstance(outputs, torch.Tensor):
            return self.adapter(outputs)
        raise TypeError(
            f"Unsupported Transformer block output type: {type(outputs).__name__}."
        )


class TransformerAdapterClassifier(nn.Module):
    """One frozen BERT/GPT2 backbone with switchable client Adapter states."""

    trainable_state_filename = "final_transformer_adapter.pt"
    client_scoped_parameters = True

    def __init__(
        self,
        model_path: str,
        architecture: str,
        num_classes: int = 2,
        reduction: int = 2,
        activation: str = "relu",
        classifier_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        zero_init_up: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        architecture = str(architecture).lower()
        if architecture not in {"bert", "gpt2"}:
            raise ValueError("architecture must be bert or gpt2.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if not 0.0 <= classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1).")

        self.architecture = architecture
        self.model_type = f"{architecture}_adapter"
        self.model_path = str(model_path)
        self.num_classes = int(num_classes)
        self.classnames = [str(index) for index in range(num_classes)]
        self.reduction = int(reduction)
        self.activation_name = str(activation).lower()
        self.classifier_dropout_probability = float(classifier_dropout)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.zero_init_up = bool(zero_init_up)
        self.device = device

        self.backbone = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        hidden_size = int(self.backbone.config.hidden_size)
        initializer_std = float(
            getattr(self.backbone.config, "initializer_range", 0.02)
        )
        self.hidden_size = hidden_size
        self._inject_adapters(initializer_std)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=initializer_std)
        nn.init.zeros_(self.classifier.bias)

        if architecture == "gpt2":
            self.backbone.config.use_cache = False
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

    def _inject_adapters(self, initializer_std: float) -> None:
        if self.architecture == "bert":
            blocks = self.backbone.encoder.layer
        else:
            blocks = self.backbone.h
        for index, block in enumerate(list(blocks)):
            blocks[index] = TransformerBlockWithAdapter(
                base_layer=block,
                hidden_size=self.hidden_size,
                reduction=self.reduction,
                activation=self.activation_name,
                initializer_std=initializer_std,
                zero_init_up=self.zero_init_up,
            )
        self.num_adapter_layers = len(blocks)
        if self.num_adapter_layers <= 0:
            raise RuntimeError("The pretrained model has no Transformer blocks.")

    def _trainable_parameter_targets(
        self,
    ) -> dict[str, tuple[nn.Module, str]]:
        targets: dict[str, tuple[nn.Module, str]] = {}
        for module_name, module in self.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad:
                    continue
                full_name = (
                    f"{module_name}.{parameter_name}"
                    if module_name
                    else parameter_name
                )
                targets[full_name] = (module, parameter_name)
        return targets

    def _current_trainable_parameters(self) -> dict[str, nn.Parameter]:
        return {
            name: getattr(module, attribute)
            for name, (module, attribute) in self._trainable_parameter_targets().items()
        }

    def export_trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self._current_trainable_parameters().items()
        }

    def load_trainable_state(
        self,
        state: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        parameters = self._current_trainable_parameters()
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        if strict and (missing or unexpected):
            raise ValueError(
                "Transformer Adapter state mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        with torch.no_grad():
            for name in set(parameters) & set(state):
                target = parameters[name]
                source = state[name]
                if target.shape != source.shape:
                    raise ValueError(
                        f"Parameter {name!r} has shape {tuple(source.shape)}, "
                        f"expected {tuple(target.shape)}."
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))

    def _bind_client_parameters(
        self,
        client_model: "ClientTransformerAdapterClassifier",
        parameters: dict[str, nn.Parameter],
    ) -> None:
        active = self._active_client_model
        if active is not None and active is not client_model:
            raise RuntimeError("The shared Transformer is bound to another client.")
        targets = self._trainable_parameter_targets()
        if set(parameters) != set(targets):
            raise ValueError("Client parameters do not match the shared model.")
        for name, parameter in parameters.items():
            expected = self._global_trainable_parameters[name]
            if parameter.shape != expected.shape:
                raise ValueError(f"Client parameter {name!r} has the wrong shape.")
            module, attribute = targets[name]
            setattr(module, attribute, parameter)
        object.__setattr__(self, "_active_client_model", client_model)

    def _release_client_parameters(
        self, client_model: "ClientTransformerAdapterClassifier"
    ) -> None:
        if self._active_client_model is not client_model:
            raise RuntimeError("Cannot release a client that is not active.")
        targets = self._trainable_parameter_targets()
        for name, parameter in self._global_trainable_parameters.items():
            module, attribute = targets[name]
            setattr(module, attribute, parameter)
        object.__setattr__(self, "_active_client_model", None)

    def create_client_model(
        self, client_id: int
    ) -> "ClientTransformerAdapterClassifier":
        if self._active_client_model is not None:
            raise RuntimeError("Create clients only while the backbone is unbound.")
        return ClientTransformerAdapterClassifier(self, client_id)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the backbone in training mode so Hugging Face gradient
        # checkpointing remains active, while disabling frozen dropout. Only
        # the explicitly configured task-head dropout remains stochastic.
        self.backbone.train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        for module in self.modules():
            if isinstance(module, BottleneckAdapter):
                module.train(mode)
        self.classifier_dropout.train(mode)
        self.classifier.train(mode)
        return self

    @staticmethod
    def unpack_inputs(
        packed_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if packed_inputs.ndim != 3 or packed_inputs.shape[1] != 2:
            raise ValueError(
                "Text inputs must have shape [batch, 2, sequence_length]."
            )
        return packed_inputs[:, 0].long(), packed_inputs[:, 1].long()

    def hidden_features(self, packed_inputs: torch.Tensor) -> torch.Tensor:
        input_ids, attention_mask = self.unpack_inputs(packed_inputs)
        outputs = self.backbone(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        if self.architecture == "bert":
            # BertForSequenceClassification uses the pretrained pooler rather
            # than the raw final-layer CLS vector.  Retaining that frozen
            # dense+tanh projection gives the randomly initialized task head a
            # substantially better-conditioned input while Adapter gradients
            # still flow through the complete encoder.
            pooler_output = getattr(outputs, "pooler_output", None)
            if isinstance(pooler_output, torch.Tensor):
                return pooler_output
            return hidden_states[:, 0]
        positions = attention_mask.to(hidden_states.device).sum(dim=1) - 1
        positions = positions.clamp_min(0)
        rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[rows, positions]

    def forward(
        self,
        packed_inputs: torch.Tensor,
        return_intermediate: bool = False,
    ):
        hidden = self.hidden_features(packed_inputs)
        logits = self.classifier(self.classifier_dropout(hidden))
        if return_intermediate:
            return logits, hidden
        return logits

    def get_semantic_features(
        self,
        packed_inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden_features(packed_inputs)
        class_features = self.classifier.weight[
            labels.to(self.classifier.weight.device)
        ]
        return hidden, class_features

    def get_audit_representation(
        self,
        packed_inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden_features(packed_inputs)
        logits = self.classifier(hidden)
        class_features = self.classifier.weight[
            labels.to(self.classifier.weight.device)
        ]
        return logits, torch.cat((logits, hidden, hidden * class_features), dim=1)

    def get_audit_key_parameter(self) -> nn.Parameter:
        return self.classifier.weight

    def get_projres_attack_surface(
        self, parameter_name: str | None = None
    ) -> tuple[str, nn.Linear]:
        """Return the Adapter down-projection exposed by a FedSGD upload."""
        surfaces = [
            (f"{name}.down.weight", module.down)
            for name, module in self.named_modules()
            if isinstance(module, BottleneckAdapter)
        ]
        if not surfaces:
            raise RuntimeError("The Transformer has no Adapter attack surface.")
        if parameter_name is None:
            return surfaces[0]
        matches = [surface for surface in surfaces if surface[0] == parameter_name]
        if not matches:
            available = ", ".join(name for name, _ in surfaces)
            raise ValueError(
                f"ProjRes parameter {parameter_name!r} is not an Adapter "
                f"down-projection weight; available surfaces: {available}."
            )
        return matches[0]

    @torch.no_grad()
    def get_projres_representations(
        self,
        packed_inputs: torch.Tensor,
        parameter_name: str | None = None,
        token_reduction: str = "auto",
    ) -> tuple[torch.Tensor, int]:
        """Capture sample embeddings entering the attacked Adapter layer.

        The second return value is the number of padded hidden vectors processed
        in this batch. It supplies ProjRes's theoretical rank cap without
        turning individual tokens into independent membership candidates.
        """
        _, attacked_layer = self.get_projres_attack_surface(parameter_name)
        captured: list[torch.Tensor] = []

        def capture_input(_module, args) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise TypeError("Adapter down-projection received invalid input.")
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
                "ProjRes expected the attacked Adapter to execute exactly once."
            )
        hidden = captured[0]
        _, attention_mask = self.unpack_inputs(packed_inputs)
        attention_mask = attention_mask.to(hidden.device)
        reduction = str(token_reduction).lower()
        if reduction == "auto":
            reduction = "cls" if self.architecture == "bert" else "last"
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
        else:
            raise ValueError("token_reduction must be auto, cls, last, or mean.")
        return representations, int(hidden.shape[0] * hidden.shape[1])


class ClientTransformerAdapterClassifier(nn.Module):
    """CPU-resident per-client PEFT state executed on one shared backbone."""

    trainable_state_filename = TransformerAdapterClassifier.trainable_state_filename
    client_scoped_parameters = True

    def __init__(
        self,
        shared_model: TransformerAdapterClassifier,
        client_id: int,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_shared_model", shared_model)
        self.client_id = int(client_id)
        initial_state = shared_model.export_trainable_state()
        self._parameter_names = tuple(initial_state)
        self._local_parameters = nn.ParameterList(
            [
                nn.Parameter(initial_state[name].detach().cpu().clone())
                for name in self._parameter_names
            ]
        )
        self._session_depth = 0

    @property
    def model_type(self) -> str:
        return self._shared_model.model_type

    @property
    def classnames(self) -> list[str]:
        return self._shared_model.classnames

    @property
    def num_classes(self) -> int:
        return self._shared_model.num_classes

    def _named_local_parameters(self) -> dict[str, nn.Parameter]:
        return dict(zip(self._parameter_names, self._local_parameters))

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        del recurse
        seen: set[int] = set()
        for name, parameter in self._named_local_parameters().items():
            if remove_duplicate and id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield f"{prefix}.{name}" if prefix else name, parameter

    def export_trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self._named_local_parameters().items()
        }

    def state_dict(
        self,
        destination=None,
        prefix: str = "",
        keep_vars: bool = False,
    ):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, parameter in self._named_local_parameters().items():
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
        parameters = self._named_local_parameters()
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        if strict and (missing or unexpected):
            raise ValueError(
                f"Client Adapter state mismatch: missing={missing}, "
                f"unexpected={unexpected}."
            )
        with torch.no_grad():
            for name in set(parameters) & set(state):
                target = parameters[name]
                source = state[name]
                if source.shape != target.shape:
                    raise ValueError(
                        f"Client parameter {name!r} has shape {tuple(source.shape)}, "
                        f"expected {tuple(target.shape)}."
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        if assign:
            raise ValueError("Client Transformer Adapter does not support assign=True.")
        state = dict(state_dict)
        names = set(self._parameter_names)
        missing = sorted(names - set(state))
        unexpected = sorted(set(state) - names)
        self.load_trainable_state(state, strict=strict)
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)

    def _move_local_parameters(self, device: torch.device) -> None:
        for parameter in self._local_parameters:
            parameter.data = parameter.data.to(device)
            if parameter.grad is not None:
                parameter.grad = parameter.grad.to(device)

    def _offload_local_parameters(self) -> None:
        # A new stateless SGD optimizer is constructed next round, so keeping
        # gradients would only double the resident per-client CPU memory.
        for parameter in self._local_parameters:
            parameter.grad = None
            parameter.data = parameter.data.cpu()

    @contextmanager
    def use_shared_model(self):
        outermost = self._session_depth == 0
        if outermost:
            self._move_local_parameters(self._shared_model.device)
            self._shared_model._bind_client_parameters(
                self, self._named_local_parameters()
            )
        self._session_depth += 1
        try:
            self._shared_model.train(self.training)
            yield self._shared_model
        finally:
            self._session_depth -= 1
            if outermost:
                self._shared_model._release_client_parameters(self)
                self._offload_local_parameters()

    def _call_shared(self, method_name: str, *args, **kwargs):
        with self.use_shared_model() as shared:
            return getattr(shared, method_name)(*args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._shared_model._active_client_model is self:
            self._shared_model.train(mode)
        return self

    def forward(self, *args, **kwargs):
        return self._call_shared("forward", *args, **kwargs)

    def get_semantic_features(self, *args, **kwargs):
        return self._call_shared("get_semantic_features", *args, **kwargs)

    def get_audit_representation(self, *args, **kwargs):
        return self._call_shared("get_audit_representation", *args, **kwargs)

    def get_audit_key_parameter(self) -> nn.Parameter:
        return self._call_shared("get_audit_key_parameter")
