import copy
import logging
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainmodel.base_model import BaseModel
from utils.constants import (
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    SHARED_POISON_TRAIN_ATTACKS,
    SUPPORTED_FPL_ATTACKS,
)
from utils.poison_func import add_pixel_pattern, get_poisoned_sample_count

logger = logging.getLogger(__name__)


class UserBase:
    """FPL client implementation for Cerberus, A3FL, and SABRE."""

    def __init__(
        self,
        device,
        fpl,
        id,
        dataset_name,
        train_data,
        test_data,
        model,
        batch_size,
        learning_rate,
        local_epochs,
        local_poison_epochs,
        defense,
        malicious,
        collate_fn=None,
        eval_batch_size: int = 64,
    ):
        if not fpl:
            raise ValueError("This branch only supports federated prompt learning.")
        if defense.lower() != "seismograph":
            raise ValueError("This branch only supports the seismograph aggregator.")
        if eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be a positive integer.")

        self.device = device
        self.fpl = True
        self.id = id
        self.dataset_name = dataset_name
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.local_poison_epochs = local_poison_epochs
        self.train_data = train_data
        self.test_data = test_data
        self.defense = "seismograph"
        self.malicious = malicious
        self.collate_fn = collate_fn
        self.class_proportions = self._compute_class_proportions(train_data)

        self.trainloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        self.testloader = DataLoader(
            test_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        self.model: BaseModel = copy.deepcopy(model)

        logger.info(
            "Initialized FPL user %s: dataset=%s, train=%s, test=%s, "
            "batch=%s, malicious=%s, class_proportions=%s",
            self.id,
            self.dataset_name,
            self.train_samples,
            self.test_samples,
            self.batch_size,
            self.malicious,
            self.class_proportions,
        )

    @staticmethod
    def _compute_class_proportions(dataset) -> list[float]:
        label_counts: dict[int, int] = {}
        for _, label in dataset:
            label_value = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            label_counts[label_value] = label_counts.get(label_value, 0) + 1

        total_samples = sum(label_counts.values())
        if total_samples == 0:
            return []
        return [
            round(label_counts.get(label, 0) / total_samples, 3)
            for label in range(max(label_counts) + 1)
        ]

    def set_parameters(self, param_dict: dict[str, torch.Tensor]) -> None:
        logger.debug("User %s: loading %s trainable tensors", self.id, len(param_dict))
        self.model.load_state_dict(param_dict, strict=False)

    def get_parameters(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        filtered_state = {
            name: tensor
            for name, tensor in self.model.state_dict().items()
            if name in trainable_names
        }
        logger.debug(
            "User %s: returning %s trainable tensors", self.id, len(filtered_state)
        )
        return filtered_state

    def get_text_features(self, normalize: bool = True) -> torch.Tensor:
        return self.model.get_text_features(normalize=normalize)

    @staticmethod
    def _pattern_mask(trigger: torch.Tensor, pattern) -> torch.Tensor:
        mask = torch.ones_like(trigger, dtype=torch.float32)
        for row, col in pattern:
            mask[:, row, col] = 0
        return mask

    def get_next_poison_train_batch(
        self,
        normal_batch,
        poison_ratio,
        poison_label,
        noise_trigger,
        pattern,
        overlay: bool = False,
    ):
        images, labels = normal_batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        poison_count = get_poisoned_sample_count(
            poison_ratio,
            images.size(0),
            fpl=True,
        )
        poison_images = add_pixel_pattern(
            self.device,
            images[:poison_count],
            noise_trigger,
            pattern_mask=self._pattern_mask(noise_trigger, pattern),
            mean=CLIP_IMAGE_MEAN,
            std=CLIP_IMAGE_STD,
            overlay=overlay,
        )
        poison_labels = torch.full(
            (poison_count,),
            poison_label,
            dtype=labels.dtype,
            device=self.device,
        )
        return (
            torch.cat((poison_images, images[poison_count:]), dim=0),
            torch.cat((poison_labels, labels[poison_count:]), dim=0),
        )

    def normal_train(self, model: nn.Module) -> None:
        model.train()
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                loss = loss_func(model(images), labels)
                loss.backward()
                optimizer.step()

    def train(self) -> None:
        self.model.to(self.device)
        self.normal_train(self.model)

    def evaluate_clean_and_poison(
        self,
        poison_label: int,
        attack_method: str,
        noise_trigger: torch.Tensor,
        pattern,
    ):
        attack_method = attack_method.lower()
        if attack_method not in SUPPORTED_FPL_ATTACKS:
            raise ValueError(f"Unsupported FPL attack: {attack_method}")

        self.model.eval()
        loss_func = nn.CrossEntropyLoss()
        noise_trigger = noise_trigger.to(self.device)
        pattern_mask = self._pattern_mask(noise_trigger, pattern).to(self.device)

        clean_loss = 0.0
        clean_correct = 0
        clean_samples = 0
        poison_loss = 0.0
        poison_correct = 0
        poison_samples = 0

        for images, labels in self.testloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                clean_output = self.model(images)
                clean_batch_loss = loss_func(clean_output, labels)
            batch_size = labels.size(0)
            clean_samples += batch_size
            clean_loss += clean_batch_loss.item() * batch_size
            clean_correct += (clean_output.argmax(1) == labels).sum().item()

            non_target_mask = labels != poison_label
            if not bool(non_target_mask.any().item()):
                continue
            source_images = images[non_target_mask]
            source_labels = labels[non_target_mask]
            poison_samples += source_images.size(0)
            poisoned_images = add_pixel_pattern(
                self.device,
                source_images,
                noise_trigger,
                pattern_mask=pattern_mask,
                mean=CLIP_IMAGE_MEAN,
                std=CLIP_IMAGE_STD,
                overlay=attack_method == "sabre",
            )
            poison_labels = torch.full_like(source_labels, poison_label)
            with torch.no_grad():
                poison_output = self.model(poisoned_images)
                poison_batch_loss = loss_func(poison_output, poison_labels)
            poison_loss += poison_batch_loss.item() * source_images.size(0)
            poison_correct += (
                poison_output.argmax(1) == poison_labels
            ).float().sum().item()

        if clean_samples == 0:
            logger.warning("User %s: empty FPL test set", self.id)
        return (
            clean_loss,
            clean_correct,
            clean_samples,
            poison_loss,
            poison_correct,
            poison_samples,
        )

    def cerberus_train(
        self,
        poison_ratio,
        poison_label,
        trigger,
        pattern,
        poisoned_model_dict: Dict[int, dict[str, torch.Tensor]],
    ) -> None:
        normal_model = deepcopy(self.model)
        self.normal_train(normal_model)
        normal_model_dict = {
            name: parameter.detach()
            for name, parameter in normal_model.named_parameters()
            if parameter.requires_grad
        }

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        loss_func = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.local_poison_epochs):
            for batch in self.trainloader:
                images, labels = self.get_next_poison_train_batch(
                    normal_batch=batch,
                    poison_ratio=poison_ratio,
                    poison_label=poison_label,
                    noise_trigger=trigger,
                    pattern=pattern,
                )
                optimizer.zero_grad()
                class_loss = loss_func(self.model(images), labels)

                normal_distance = sum(
                    torch.norm(parameter - normal_model_dict[name], p=2).square()
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                )

                adversary_similarity = torch.zeros((), device=self.device)
                if poisoned_model_dict:
                    num_layers = len(list(self.model.named_parameters()))
                    for other_model in poisoned_model_dict.values():
                        layer_similarity = torch.zeros((), device=self.device)
                        for name, parameter in self.model.named_parameters():
                            if not parameter.requires_grad:
                                continue
                            other_parameter = other_model[name].to(self.device).view(-1)
                            layer_similarity += torch.abs(
                                nn.functional.cosine_similarity(
                                    parameter.view(-1) + 1e-8,
                                    other_parameter + 1e-8,
                                    dim=0,
                                )
                            )
                        adversary_similarity += layer_similarity / num_layers
                    adversary_similarity /= len(poisoned_model_dict)

                loss = class_loss + 0.01 * normal_distance + 0.1 * adversary_similarity
                loss.backward()
                optimizer.step()

    def poison_train(
        self,
        poison_ratio,
        poison_label,
        trigger,
        pattern,
        attack_method: str,
    ) -> None:
        attack_method = attack_method.lower()
        if attack_method not in SHARED_POISON_TRAIN_ATTACKS:
            raise ValueError(
                "The shared poison trainer only supports A3FL and SABRE; "
                f"got {attack_method!r}."
            )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        loss_func = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.local_poison_epochs):
            for batch in self.trainloader:
                images, labels = self.get_next_poison_train_batch(
                    normal_batch=batch,
                    poison_ratio=poison_ratio,
                    poison_label=poison_label,
                    noise_trigger=trigger,
                    pattern=pattern,
                    overlay=attack_method == "sabre",
                )
                optimizer.zero_grad()
                loss = loss_func(self.model(images), labels)
                loss.backward()
                optimizer.step()
