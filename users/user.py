from __future__ import annotations

import copy
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, default_collate

from privacy_attacks.code_poison import compromised_prompt_loss


class _IndexedDataset(Dataset):
    """Expose stable local positions without changing the wrapped samples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index], int(index)


class UserBase:
    """A benign federated client with optional privacy instrumentation."""

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
        train_generator = None
        if self.federated_method == "fedsgd":
            train_generator = torch.Generator().manual_seed(
                int(self.method_config.get("seed", 42)) + 1000003 * int(self.id)
            )
        self._www_enabled = (
            str(getattr(self.defense_controller, "name", "none")).lower()
            == "www"
        )
        self._record_dp_enabled = (
            str(getattr(self.defense_controller, "name", "none")).lower()
            == "record_dp"
        )
        self.record_dp_expected_batch_size = min(
            int(self.batch_size), int(self.train_samples)
        )
        self.record_dp_sample_rate = (
            self.record_dp_expected_batch_size / self.train_samples
            if self.train_samples > 0
            else 0.0
        )
        self.record_dp_steps_per_update = (
            1
            if self.federated_method == "fedsgd"
            else self.local_epochs
            * math.ceil(
                self.train_samples
                / max(1, self.record_dp_expected_batch_size)
            )
        )
        self._record_dp_empty_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self.trainloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
            generator=train_generator,
        )
        self.www_statistics_loader = (
            DataLoader(
                _IndexedDataset(train_data),
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=self._collate_indexed_batch,
                drop_last=False,
            )
            if self._www_enabled
            else None
        )
        self.testloader = DataLoader(
            test_data,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        client_model_factory = getattr(model, "create_client_model", None)
        if callable(client_model_factory):
            self.model = client_model_factory(client_id=self.id)
        else:
            self.model = copy.deepcopy(model)
        self._train_iterator = None
        self.last_update_sample_count = 0
        self.last_train_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self.last_train_indices: torch.Tensor | None = None
        self.last_train_recycled: torch.Tensor | None = None
        self.last_update_gradients: dict[str, torch.Tensor] | None = None
        self.last_gradient_capture_count = 0
        self.www_source_round: int | None = None
        self.www_aggregation_weight: float | None = None
        self.www_ranking_round: int | None = None
        self.www_own_losses: torch.Tensor | None = None
        self.www_other_losses: torch.Tensor | None = None
        self.www_scores: torch.Tensor | None = None
        self.www_ranked_positions: torch.Tensor | None = None
        self.www_ranked_scores: torch.Tensor | None = None
        self.www_ranked_labels: torch.Tensor | None = None
        self.www_local_indices: torch.Tensor | None = None
        self.www_ranked_local_indices: torch.Tensor | None = None
        self.www_importance_weights: torch.Tensor | None = None
        self.www_tail_mask: torch.Tensor | None = None
        self.www_effective_clip_norms: torch.Tensor | None = None
        self.www_tail_local_indices: torch.Tensor | None = None
        self.www_score_count: torch.Tensor | None = None
        self.www_score_sum: torch.Tensor | None = None
        self.www_score_sum_sq: torch.Tensor | None = None
        self.www_score_min: torch.Tensor | None = None
        self.www_score_max: torch.Tensor | None = None
        self.www_score_last: torch.Tensor | None = None
        self.www_score_last_round: torch.Tensor | None = None
        self.www_feature_seen: torch.Tensor | None = None
        self.www_class_feature_counts: torch.Tensor | None = None
        self.www_class_feature_means: torch.Tensor | None = None
        self.www_within_class_scatter: torch.Tensor | None = None
        self.www_within_class_covariance: torch.Tensor | None = None
        self.www_within_class_covariance_dof: int = 0

    def _collate_indexed_batch(self, batch):
        samples, indices = zip(*batch)
        if self.collate_fn is None:
            images, labels = default_collate(list(samples))
        else:
            images, labels = self.collate_fn(list(samples))
        return images, labels, torch.tensor(indices, dtype=torch.long)

    def begin_local_update(self) -> None:
        self.last_update_sample_count = 0
        self.last_train_batch = None
        self.last_train_indices = None
        self.last_train_recycled = None
        self.last_update_gradients = None
        self.last_gradient_capture_count = 0

    def capture_protocol_gradients(self, model: torch.nn.Module) -> None:
        """Retain the exact gradient transmitted by a one-step FedSGD client."""
        if self.federated_method != "fedsgd":
            return
        gradients = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad
            gradients[name] = (
                torch.zeros_like(parameter, device="cpu")
                if gradient is None
                else gradient.detach().cpu().clone()
            )
        if not gradients:
            raise RuntimeError("FedSGD client produced no trainable gradients.")
        self.last_update_gradients = gradients
        self.last_gradient_capture_count += 1

    def _record_train_batch(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> None:
        self.last_update_sample_count += int(labels.numel())
        self.last_train_batch = (
            images.detach().cpu().clone(),
            labels.detach().cpu().long().clone(),
        )

    def next_train_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next shuffled local mini-batch, cycling across rounds."""
        if self._train_iterator is None:
            self._train_iterator = iter(self.trainloader)
        try:
            batch = next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.trainloader)
            try:
                batch = next(self._train_iterator)
            except StopIteration as error:
                raise ValueError(
                    f"Client {self.id} has no local training batch."
                ) from error
        images, labels = batch
        self._record_train_batch(images, labels)
        return images, labels

    def iter_www_local_batches(self, generator: torch.Generator):
        """Yield the same Poisson draws as Record-DP with stable local indices."""
        if not self._www_enabled:
            raise RuntimeError("Indexed WWW training is not enabled for this client.")
        yield from self.iter_poisson_batches(generator)

    def iter_www_statistics_batches(self):
        """Yield the complete local training set once in stable index order."""
        if self.www_statistics_loader is None:
            raise RuntimeError("WWW statistics are not enabled for this client.")
        yield from self.www_statistics_loader

    def iter_local_batches(self):
        """Yield the batches used by one local update under the active protocol."""
        if self.federated_method == "fedsgd":
            yield self.next_train_batch()
            return
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                self._record_train_batch(images, labels)
                yield images, labels

    def iter_selected_batches(self, indices: torch.Tensor, generator):
        """Train on an indexed defense-selected pool under the active protocol."""
        for _ in range(1 if self.federated_method == "fedsgd" else self.local_epochs):
            shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
            if self.federated_method == "fedsgd":
                shuffled = shuffled[: self.batch_size]
            for start in range(0, shuffled.numel(), self.batch_size):
                batch_indices = shuffled[start : start + self.batch_size]
                inputs, labels = self._collate_train_samples(
                    [self.train_data[index] for index in batch_indices.tolist()]
                )
                self._record_train_batch(inputs, labels)
                self.last_train_indices = batch_indices.clone()
                yield inputs, labels, batch_indices

    def _collate_train_samples(
        self, samples: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.collate_fn is None:
            images, labels = default_collate(samples)
        else:
            images, labels = self.collate_fn(samples)
        return images, labels

    def _empty_record_dp_batch_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._record_dp_empty_batch is None:
            if self.train_samples <= 0:
                raise ValueError(f"Client {self.id} has no local training records.")
            images, labels = self._collate_train_samples([self.train_data[0]])
            self._record_dp_empty_batch = (
                images[:0].detach().cpu(),
                labels[:0].detach().cpu().long(),
            )
        return tuple(tensor.clone() for tensor in self._record_dp_empty_batch)

    def iter_record_dp_batches(self, generator: torch.Generator):
        """Yield Record-DP batches through the common indexed Poisson sampler."""
        if not self._record_dp_enabled:
            raise RuntimeError("Record-DP batches require defense.name=record_dp.")
        for images, labels, _ in self.iter_poisson_batches(generator):
            yield images, labels

    def iter_poisson_batches(self, generator: torch.Generator):
        """Yield fixed-count indexed Poisson draws for one private local update.

        Every local record is sampled independently with probability ``q``.
        Empty draws are retained because conditioning on a non-empty draw would
        not match the sampled-Gaussian privacy accountant.
        """
        if not (self._record_dp_enabled or self._www_enabled):
            raise RuntimeError("Poisson batches require defense.name=record_dp or www.")
        if self.train_samples <= 0 or self.record_dp_sample_rate <= 0:
            raise ValueError(f"Client {self.id} has no local training records.")
        for _ in range(self.record_dp_steps_per_update):
            selected = torch.rand(
                self.train_samples,
                generator=generator,
                device="cpu",
            ) < self.record_dp_sample_rate
            indices = selected.nonzero(as_tuple=False).flatten().tolist()
            if indices:
                images, labels = self._collate_train_samples(
                    [self.train_data[index] for index in indices]
                )
            else:
                images, labels = self._empty_record_dp_batch_tensors()
            self._record_train_batch(images, labels)
            self.last_train_indices = torch.tensor(indices, dtype=torch.long)
            yield images, labels, self.last_train_indices.clone()

    def set_parameters(self, state: dict[str, torch.Tensor]) -> None:
        loader = getattr(self.model, "load_trainable_state", None)
        if callable(loader):
            loader(state, strict=True)
        else:
            self.model.load_state_dict(state, strict=False)

    def get_parameters(self) -> dict[str, torch.Tensor]:
        exporter = getattr(self.model, "export_trainable_state", None)
        if callable(exporter):
            return exporter()
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
        self.begin_local_update()
        shared_session = getattr(model, "use_shared_model", None)
        session = shared_session() if callable(shared_session) else nullcontext()
        with session:
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
                    privacy_probe=privacy_probe,
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
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            if not trainable:
                raise ValueError("The client model has no trainable parameters.")
            optimizer_name = str(
                self.method_config.get("client_optimizer", "sgd")
            ).lower()
            if optimizer_name == "sgd":
                optimizer = torch.optim.SGD(
                    trainable,
                    lr=self.learning_rate,
                    momentum=float(self.method_config.get("momentum", 0.0)),
                    weight_decay=float(self.method_config.get("weight_decay", 0.0)),
                )
            elif optimizer_name == "adamw":
                optimizer = torch.optim.AdamW(
                    trainable,
                    lr=self.learning_rate,
                    weight_decay=float(self.method_config.get("weight_decay", 0.01)),
                )
            else:
                raise ValueError("client_optimizer must be sgd or adamw.")
            for images, labels in self.iter_local_batches():
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
                max_grad_norm = float(
                    self.method_config.get("max_grad_norm", 0.0)
                )
                if max_grad_norm < 0:
                    raise ValueError("max_grad_norm must be non-negative.")
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                self.capture_protocol_gradients(model)
                optimizer.step()

    def train(self, code_poison: bool = False, round_index: int = 0) -> None:
        self.train_model(
            self.model,
            code_poison=code_poison,
            round_index=round_index,
        )

    @torch.no_grad()
    def evaluate_with_confusion(
        self,
    ) -> tuple[float, int, int, torch.Tensor]:
        shared_session = getattr(self.model, "use_shared_model", None)
        session = shared_session() if callable(shared_session) else nullcontext()
        with session:
            self.model.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            confusion: torch.Tensor | None = None
            for images, labels in self.testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                predictions = logits.argmax(dim=1)
                total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
                total_correct += int((predictions == labels).sum())
                total_samples += labels.numel()
                num_classes = int(logits.shape[-1])
                if confusion is None:
                    confusion = torch.zeros(
                        (num_classes, num_classes), dtype=torch.long
                    )
                flat_indices = (
                    labels.detach().cpu().long() * num_classes
                    + predictions.detach().cpu().long()
                )
                confusion += torch.bincount(
                    flat_indices,
                    minlength=num_classes * num_classes,
                ).reshape(num_classes, num_classes)
        if confusion is None:
            confusion = torch.zeros((0, 0), dtype=torch.long)
        return total_loss, total_correct, total_samples, confusion

    @torch.no_grad()
    def evaluate(self) -> tuple[float, int, int]:
        total_loss, total_correct, total_samples, _ = (
            self.evaluate_with_confusion()
        )
        return total_loss, total_correct, total_samples
