import copy
import logging
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from privacy_attacks.code_poison import (
    compromised_prompt_loss,
    generate_membership_encoding_samples,
)
from utils.privacy_accounting import private_generator

logger = logging.getLogger(__name__)


class UserBase:
    """A benign federated prompt-tuning client with optional privacy instrumentation."""

    def __init__(
        self,
        device,
        id,
        dataset_name,
        train_data,
        test_data,
        model,
        batch_size,
        learning_rate,
        local_epochs,
        collate_fn=None,
        eval_batch_size: int = 64,
        code_poison_config: dict | None = None,
        defense_controller=None,
        federated_method: str = "fedavg",
        method_config: dict | None = None,
    ):
        if batch_size <= 0 or eval_batch_size <= 0 or local_epochs <= 0:
            raise ValueError("Batch sizes and local_epochs must be positive.")
        self.device = device
        self.id = id
        self.dataset_name = dataset_name
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.collate_fn = collate_fn
        self.code_poison_config = code_poison_config or {}
        self.defense_controller = defense_controller
        self.federated_method = str(federated_method).lower()
        self.method_config = dict(method_config or {})
        self._method_loss_override = None
        self.trainloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        self.testloader = DataLoader(
            test_data,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        self.model = copy.deepcopy(model)

    def set_parameters(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state, strict=False)

    def get_parameters(self) -> dict[str, torch.Tensor]:
        names = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
            if name in names
        }

    def train_model(
        self,
        model: torch.nn.Module,
        code_poison: bool = False,
        round_index: int = 0,
        privacy_probe: bool = False,
    ) -> None:
        if self.federated_method == "dpfpl":
            self._train_dpfpl(
                model,
                code_poison,
                round_index,
                steps_override=1 if privacy_probe else None,
            )
            if privacy_probe and self.defense_controller is not None:
                self.defense_controller.after_probe_training(
                    model, client_id=self.id, round_index=round_index
                )
            return
        if self.federated_method == "fedask":
            self._train_fedask(
                model,
                code_poison,
                round_index,
                steps_override=1 if privacy_probe else None,
            )
            if privacy_probe and self.defense_controller is not None:
                self.defense_controller.after_probe_training(
                    model, client_id=self.id, round_index=round_index
                )
            return
        if self.defense_controller is not None:
            effective_round = round_index
            if privacy_probe and round_index == 0:
                effective_round = max(
                    0,
                    int(self.defense_controller.total_rounds) - 1,
                )
            self.defense_controller.train_client(
                self,
                model,
                round_index=effective_round,
                code_poison=code_poison,
            )
            if privacy_probe:
                self.defense_controller.after_probe_training(
                    model,
                    client_id=self.id,
                    round_index=effective_round,
                )
            return
        model.to(self.device)
        model.train()
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.SGD(trainable, lr=self.learning_rate)
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                if code_poison:
                    loss = compromised_prompt_loss(
                        model,
                        images,
                        labels,
                        weight=float(self.code_poison_config.get("weight", 1.0)),
                        mean=float(self.code_poison_config.get("synthetic_mean", 0.0)),
                        std=float(self.code_poison_config.get("synthetic_std", 0.1)),
                    )
                else:
                    loss = F.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()

    @staticmethod
    def _clip_tensor(gradient: torch.Tensor, max_norm: float) -> torch.Tensor:
        return gradient * min(1.0, max_norm / max(float(gradient.norm()), 1e-12))

    def _private_hyperparameters(
        self, clip_key: str, noise_key: str
    ) -> tuple[float, float]:
        clip = float(self.method_config.get(clip_key, 1.0))
        noise = float(self.method_config.get(noise_key, 1.0))
        if (
            self.defense_controller is not None
            and self.defense_controller.name == "prompt_dp"
        ):
            clip = min(
                clip,
                float(self.defense_controller.config.get("dp_max_grad_norm", clip)),
            )
            noise = max(
                noise,
                float(self.defense_controller.config.get("dp_noise_multiplier", noise)),
            )
        return clip, noise

    def _method_batches(self, steps: int):
        iterator = iter(self.trainloader)
        for _ in range(max(1, steps)):
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(self.trainloader)
                yield next(iterator)

    def _defended_images(
        self,
        model,
        images: torch.Tensor,
        labels: torch.Tensor,
        round_index: int,
        step_index: int,
    ) -> torch.Tensor:
        controller = self.defense_controller
        if controller is None or controller.name != "soft":
            return images
        strength = max(
            0.0,
            min(1.0, float(controller.config.get("soft_obfuscation_strength", 0.5))),
        )
        noise_std = float(controller.config.get("soft_noise_std", 0.05))
        warmup = round_index == 0
        threshold = (
            math.inf if warmup else controller.validation_loss(model, self.testloader)
        )
        with torch.no_grad():
            losses = F.cross_entropy(model(images), labels, reduction="none")
            influential = losses < threshold
            if warmup:
                influential.zero_()
        transformed = torch.flip(images, dims=(-1,))
        if noise_std > 0:
            transformed = (
                transformed
                + torch.randn(
                    transformed.shape,
                    generator=controller._generator(
                        self.id, round_index, 701 + step_index
                    ),
                    device=images.device,
                    dtype=images.dtype,
                )
                * noise_std
            )
        mask = influential.view(-1, *([1] * (images.ndim - 1)))
        controller._record("soft_selected_fraction", float(influential.float().mean()))
        return torch.where(
            mask, strength * transformed + (1.0 - strength) * images, images
        )

    def _method_losses(
        self,
        model,
        images: torch.Tensor,
        labels: torch.Tensor,
        context: torch.Tensor,
        code_poison: bool,
        round_index: int,
    ) -> torch.Tensor:
        logits = model.forward_with_context(images, context)
        if self._method_loss_override is not None:
            return self._method_loss_override(images, labels, logits)
        controller = self.defense_controller
        if controller is not None and controller.name == "hamp":
            true_probability = float(
                controller.config.get("hamp_true_probability", 0.6)
            )
            classes = logits.shape[1]
            true_probability = max(1.0 / classes, min(0.999, true_probability))
            other = (1.0 - true_probability) / max(1, classes - 1)
            targets = torch.full_like(logits, other)
            targets.scatter_(1, labels.view(-1, 1), true_probability)
            losses = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
            probability = torch.softmax(logits, dim=1)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1)
            losses = (
                losses
                - float(controller.config.get("hamp_entropy_weight", 0.05)) * entropy
            )
            controller._record("hamp_entropy", float(entropy.detach().mean()))
        else:
            losses = F.cross_entropy(logits, labels, reduction="none")

        if code_poison:
            synthetic = generate_membership_encoding_samples(
                images,
                mean=float(self.code_poison_config.get("synthetic_mean", 0.0)),
                std=float(self.code_poison_config.get("synthetic_std", 0.1)),
            )
            secret = F.cross_entropy(
                model.forward_with_context(synthetic, context), labels, reduction="none"
            )
            losses = losses + float(self.code_poison_config.get("weight", 1.0)) * secret

        if controller is not None and controller.name == "cofedmid":
            assigned_classes = controller._cofedmid_classes(self.id, round_index)
            assigned = torch.zeros_like(labels, dtype=torch.bool)
            for class_id in assigned_classes:
                assigned |= labels == class_id
            recycled = torch.zeros_like(assigned)
            excluded = torch.nonzero(~assigned, as_tuple=False).flatten()
            recycle_ratio = float(controller.config.get("cofedmid_recycle_ratio", 0.1))
            if excluded.numel() and recycle_ratio > 0:
                intervals = int(controller.config.get("cofedmid_intervals", 4))
                sorted_indices = excluded[losses.detach()[excluded].argsort()]
                arm = controller.private_cofedmid_arm(self.id, round_index)
                candidates = torch.tensor_split(sorted_indices, intervals)[arm]
                cap = max(1, int(math.floor(recycle_ratio * labels.numel())))
                recycled[candidates[:cap]] = True
            selected = assigned | recycled
            if not bool(selected.any()):
                selected[losses.detach().argmax()] = True
            if bool(recycled.any()):
                probabilities = torch.softmax(logits[recycled], dim=1)
                confidence = (
                    probabilities.gather(1, labels[recycled].view(-1, 1))
                    .detach()
                    .clamp(1.0 / logits.shape[1], 0.999)
                )
                targets = (1.0 - confidence) / max(1, logits.shape[1] - 1)
                targets = targets.expand(-1, logits.shape[1]).clone()
                targets.scatter_(1, labels[recycled].view(-1, 1), confidence)
                confidence_regularizer = -(
                    targets * F.log_softmax(logits[recycled], dim=1)
                ).sum(dim=1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
                    dim=1
                )
                losses = losses.clone()
                losses[recycled] += (
                    confidence_regularizer
                    - float(controller.config.get("cofedmid_entropy_weight", 0.05))
                    * entropy
                )
            controller._record(
                "cofedmid_selected_fraction", float(selected.float().mean())
            )
            losses = losses[selected]
        return losses

    def _train_dpfpl(
        self,
        model: torch.nn.Module,
        code_poison: bool,
        round_index: int,
        steps_override: int | None = None,
    ) -> None:
        """Algorithm 1 adaptation: RGP local prompt plus LDP/GDP global prompt."""
        learner = model.prompt_learner
        global_ctx = learner.global_ctx
        local_ctx = learner.local_ctx
        rank_requested = int(self.method_config.get("rank", 4))
        clip, noise_multiplier = self._private_hyperparameters(
            "local_clip_norm", "local_noise_multiplier"
        )
        global_clip = float(self.method_config.get("global_clip_norm", 1.0))
        generator = private_generator(
            self.device,
            bool(self.method_config.get("reproducible_dp_noise", False))
            or bool(
                self.defense_controller is not None
                and self.defense_controller.config.get("reproducible_dp_noise", False)
            ),
            int(self.method_config.get("seed", 42))
            + 1000003 * self.id
            + 1009 * round_index,
        )
        model.to(self.device)
        model.train()
        cofedmid_before = (
            self.defense_controller.begin_private_cofedmid(self, model, round_index)
            if self.defense_controller is not None
            else None
        )
        steps = int(
            steps_override
            if steps_override is not None
            else self.method_config.get("local_steps", 1)
        )
        for step_index, (images, labels) in enumerate(self._method_batches(steps)):
            images, labels = images.to(self.device), labels.to(self.device)
            images = self._defended_images(
                model, images, labels, round_index, step_index
            )
            local_matrix = local_ctx.detach().reshape(-1, local_ctx.shape[-1])
            with torch.no_grad():
                rank = min(rank_requested, *local_matrix.shape)
                projection = torch.randn(
                    local_matrix.shape[1],
                    rank,
                    generator=generator,
                    device=self.device,
                    dtype=local_matrix.dtype,
                )
                projected = local_matrix @ projection
                projected = local_matrix @ (local_matrix.T @ projected)
                u_value, _ = torch.linalg.qr(projected, mode="reduced")
                v_value = u_value.T @ local_matrix
                residual = local_matrix - u_value @ v_value
            u = u_value.detach().requires_grad_(True)
            v = v_value.detach().requires_grad_(True)
            context = global_ctx + (u @ v + residual).reshape_as(local_ctx)
            losses = self._method_losses(
                model, images, labels, context, code_poison, round_index
            )
            global_rows, u_rows, v_rows = [], [], []
            for index in range(losses.numel()):
                grad_global, grad_u, grad_v = torch.autograd.grad(
                    losses[index],
                    (global_ctx, u, v),
                    retain_graph=index + 1 < losses.numel(),
                )
                global_rows.append(self._clip_tensor(grad_global, global_clip))
                u_rows.append(self._clip_tensor(grad_u, clip))
                v_rows.append(self._clip_tensor(grad_v, clip))
            grad_global = torch.stack(global_rows).mean(dim=0)
            grad_u = torch.stack(u_rows).mean(dim=0)
            grad_v = torch.stack(v_rows).mean(dim=0)
            if noise_multiplier > 0:
                scale = noise_multiplier * clip / max(1, losses.numel())
                grad_u = (
                    grad_u
                    + torch.randn(
                        grad_u.shape,
                        generator=generator,
                        device=self.device,
                        dtype=grad_u.dtype,
                    )
                    * scale
                )
                grad_v = (
                    grad_v
                    + torch.randn(
                        grad_v.shape,
                        generator=generator,
                        device=self.device,
                        dtype=grad_v.dtype,
                    )
                    * scale
                )
            # RGP gradient reconstruction used by DP-FPL.
            local_gradient = (
                grad_u @ v + u @ grad_v - u @ (u.T @ grad_u) @ v
            ).reshape_as(local_ctx)
            with torch.no_grad():
                global_ctx.add_(grad_global, alpha=-self.learning_rate)
                local_ctx.add_(local_gradient, alpha=-self.learning_rate)
            if self.defense_controller is not None:
                self.defense_controller.steps[self.id] += 1
        if self.defense_controller is not None:
            self.defense_controller.finish_private_cofedmid(
                self, model, round_index, cofedmid_before
            )

    def _train_fedask(
        self,
        model: torch.nn.Module,
        code_poison: bool,
        round_index: int,
        steps_override: int | None = None,
    ) -> None:
        """FedASK local step: freeze A and privately update B through full W grads."""
        learner = model.prompt_learner
        a = learner.fedask_A
        b = learner.fedask_B
        clip, noise_multiplier = self._private_hyperparameters(
            "clip_norm", "noise_multiplier"
        )
        generator = private_generator(
            self.device,
            bool(self.method_config.get("reproducible_dp_noise", False))
            or bool(
                self.defense_controller is not None
                and self.defense_controller.config.get("reproducible_dp_noise", False)
            ),
            int(self.method_config.get("seed", 42))
            + 1000003 * self.id
            + 1009 * round_index,
        )
        model.to(self.device)
        model.train()
        cofedmid_before = (
            self.defense_controller.begin_private_cofedmid(self, model, round_index)
            if self.defense_controller is not None
            else None
        )
        steps = int(
            steps_override
            if steps_override is not None
            else self.method_config.get("local_steps", self.local_epochs)
        )
        for step_index, (images, labels) in enumerate(self._method_batches(steps)):
            images, labels = images.to(self.device), labels.to(self.device)
            images = self._defended_images(
                model, images, labels, round_index, step_index
            )
            adapter = (b @ a).detach().requires_grad_(True)
            context = learner.base_ctx + float(
                getattr(learner, "fedask_scaling", 1.0)
            ) * adapter.reshape_as(learner.base_ctx)
            losses = self._method_losses(
                model, images, labels, context, code_poison, round_index
            )
            rows = []
            for index in range(losses.numel()):
                gradient = torch.autograd.grad(
                    losses[index],
                    adapter,
                    retain_graph=index + 1 < losses.numel(),
                )[0]
                rows.append(self._clip_tensor(gradient, clip))
            full_gradient = torch.stack(rows).mean(dim=0)
            if noise_multiplier > 0:
                full_gradient = full_gradient + torch.randn(
                    full_gradient.shape,
                    generator=generator,
                    device=self.device,
                    dtype=full_gradient.dtype,
                ) * (noise_multiplier * clip / max(1, losses.numel()))
            with torch.no_grad():
                b.add_(full_gradient @ a.T, alpha=-self.learning_rate)
            if self.defense_controller is not None:
                self.defense_controller.steps[self.id] += 1
        if self.defense_controller is not None:
            self.defense_controller.finish_private_cofedmid(
                self, model, round_index, cofedmid_before
            )

    def private_mist_refine(
        self,
        peer_models: dict[int, torch.nn.Module],
        round_index: int,
        steps: int,
        weight: float,
    ) -> None:
        if self.federated_method not in {"dpfpl", "fedask"}:
            raise ValueError("Private MIST refinement is method-specific.")

        def mist_losses(images, labels, logits):
            with torch.no_grad():
                peers = []
                for peer_id, peer_model in peer_models.items():
                    if peer_id == self.id:
                        continue
                    probability = torch.softmax(peer_model(images), dim=1)
                    peers.append(probability.gather(1, labels.view(-1, 1)).squeeze(1))
                counterfactual = torch.stack(peers).mean(dim=0)
            own = torch.softmax(logits, dim=1).gather(1, labels.view(-1, 1)).squeeze(1)
            difference = (own - counterfactual).abs()
            if self.defense_controller is not None:
                self.defense_controller._record(
                    "mist_cross_difference", float(difference.detach().mean())
                )
            return weight * difference

        self._method_loss_override = mist_losses
        try:
            if self.federated_method == "dpfpl":
                self._train_dpfpl(self.model, False, round_index, steps_override=steps)
            else:
                self._train_fedask(self.model, False, round_index, steps_override=steps)
        finally:
            self._method_loss_override = None

    def train(self, code_poison: bool = False, round_index: int = 0) -> None:
        self.train_model(
            self.model,
            code_poison=code_poison,
            round_index=round_index,
        )

    @torch.no_grad()
    def evaluate(self) -> tuple[float, int, int]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for images, labels in self.testloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_samples += labels.numel()
        return total_loss, total_correct, total_samples
