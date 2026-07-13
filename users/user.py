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

    def train_model(self, model: torch.nn.Module, code_poison: bool = False) -> None:
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

    def train(self, code_poison: bool = False) -> None:
        self.train_model(self.model, code_poison=code_poison)

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
