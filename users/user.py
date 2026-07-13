import copy
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from privacy_attacks.code_poison import compromised_prompt_loss

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
            name for name, parameter in self.model.named_parameters()
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
        if self.federated_method == "dpfpl" and not privacy_probe:
            self._train_dpfpl(model, code_poison, round_index)
            return
        if self.federated_method == "fedask" and not privacy_probe:
            self._train_fedask(model, code_poison, round_index)
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
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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

    def _method_loss(
        self,
        model,
        images: torch.Tensor,
        labels: torch.Tensor,
        context: torch.Tensor,
        code_poison: bool,
        reduction: str = "mean",
    ) -> torch.Tensor:
        logits = model.forward_with_context(images, context)
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        if not code_poison:
            return loss
        synthetic = torch.randn_like(images) * float(
            self.code_poison_config.get("synthetic_std", 0.1)
        ) + float(self.code_poison_config.get("synthetic_mean", 0.0))
        secret = F.cross_entropy(
            model.forward_with_context(synthetic, context), labels, reduction=reduction
        )
        return loss + float(self.code_poison_config.get("weight", 1.0)) * secret

    def _train_dpfpl(
        self, model: torch.nn.Module, code_poison: bool, round_index: int
    ) -> None:
        """Algorithm 1 adaptation: RGP local prompt plus LDP/GDP global prompt."""
        learner = model.prompt_learner
        global_ctx = learner.global_ctx
        local_ctx = learner.local_ctx
        rank_requested = int(self.method_config.get("rank", 4))
        clip = float(self.method_config.get("local_clip_norm", 1.0))
        global_clip = float(self.method_config.get("global_clip_norm", 1.0))
        noise_multiplier = float(
            self.method_config.get("local_noise_multiplier", 1.0)
        )
        generator = torch.Generator(device=self.device).manual_seed(
            int(self.method_config.get("seed", 42))
            + 1000003 * self.id
            + 1009 * round_index
        )
        model.to(self.device)
        model.train()
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                local_matrix = local_ctx.detach().reshape(-1, local_ctx.shape[-1])
                with torch.no_grad():
                    u0, singular, vh = torch.linalg.svd(
                        local_matrix, full_matrices=False
                    )
                    rank = min(rank_requested, singular.numel())
                    u_value = u0[:, :rank]
                    v_value = singular[:rank].unsqueeze(1) * vh[:rank]
                    residual = local_matrix - u_value @ v_value
                u = u_value.detach().requires_grad_(True)
                v = v_value.detach().requires_grad_(True)
                context = global_ctx + (u @ v + residual).reshape_as(local_ctx)
                loss = self._method_loss(
                    model, images, labels, context, code_poison
                )
                grad_global, grad_u, grad_v = torch.autograd.grad(
                    loss, (global_ctx, u, v)
                )
                grad_global = self._clip_tensor(grad_global, global_clip)
                grad_u = self._clip_tensor(grad_u, clip)
                grad_v = self._clip_tensor(grad_v, clip)
                if noise_multiplier > 0:
                    scale = noise_multiplier * clip / max(1, labels.numel())
                    grad_u = grad_u + torch.randn(
                        grad_u.shape, generator=generator, device=self.device,
                        dtype=grad_u.dtype,
                    ) * scale
                    grad_v = grad_v + torch.randn(
                        grad_v.shape, generator=generator, device=self.device,
                        dtype=grad_v.dtype,
                    ) * scale
                # RGP gradient reconstruction used by DP-FPL.
                local_gradient = (
                    grad_u @ v
                    + u @ grad_v
                    - u @ (u.T @ grad_u) @ v
                ).reshape_as(local_ctx)
                with torch.no_grad():
                    global_ctx.add_(grad_global, alpha=-self.learning_rate)
                    local_ctx.add_(local_gradient, alpha=-self.learning_rate)

        # An independently selected defense remains an explicit refinement phase.
        if self.defense_controller is not None and self.defense_controller.enabled:
            self.defense_controller.train_client(
                self, model, round_index=round_index, code_poison=code_poison
            )

    def _train_fedask(
        self, model: torch.nn.Module, code_poison: bool, round_index: int
    ) -> None:
        """FedASK local step: freeze A and privately update B through full W grads."""
        learner = model.prompt_learner
        a = learner.fedask_A
        b = learner.fedask_B
        clip = float(self.method_config.get("clip_norm", 1.0))
        noise_multiplier = float(self.method_config.get("noise_multiplier", 1.0))
        generator = torch.Generator(device=self.device).manual_seed(
            int(self.method_config.get("seed", 42))
            + 1000003 * self.id
            + 1009 * round_index
        )
        model.to(self.device)
        model.train()
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                adapter = (b @ a).detach().requires_grad_(True)
                context = learner.base_ctx + adapter.reshape_as(learner.base_ctx)
                losses = self._method_loss(
                    model, images, labels, context, code_poison, reduction="none"
                )
                rows = []
                for index in range(losses.numel()):
                    gradient = torch.autograd.grad(
                        losses[index], adapter,
                        retain_graph=index + 1 < losses.numel(),
                    )[0]
                    rows.append(self._clip_tensor(gradient, clip))
                full_gradient = torch.stack(rows).mean(dim=0)
                if noise_multiplier > 0:
                    full_gradient = full_gradient + torch.randn(
                        full_gradient.shape, generator=generator, device=self.device,
                        dtype=full_gradient.dtype,
                    ) * (noise_multiplier * clip / max(1, labels.numel()))
                with torch.no_grad():
                    b.add_(full_gradient @ a.T, alpha=-self.learning_rate)

        if self.defense_controller is not None and self.defense_controller.enabled:
            a_value = a.detach().clone()
            a.requires_grad_(False)
            try:
                self.defense_controller.train_client(
                    self, model, round_index=round_index, code_poison=code_poison
                )
            finally:
                a.requires_grad_(True)
                with torch.no_grad():
                    a.copy_(a_value)

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
