from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeaturePrecomputeSummary:
    train_samples: int
    test_samples: int
    feature_dimension: int
    storage_bytes: int


def collate_clip_features(batch):
    """Collate cached ``(CLIP feature, label)`` examples without a processor."""
    features, labels = zip(*batch)
    return torch.stack(tuple(features)), torch.as_tensor(labels, dtype=torch.long)


@torch.inference_mode()
def _encode_dataset(
    model,
    dataset: Dataset,
    collate_fn: Callable,
    batch_size: int,
) -> TensorDataset:
    feature_dimension = int(model.projection_dim)
    if len(dataset) == 0:
        return TensorDataset(
            torch.empty((0, feature_dimension), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )
    feature_parts = []
    label_parts = []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    model.eval()
    for images, labels in loader:
        feature_parts.append(model.encode_images(images).detach().to("cpu"))
        label_parts.append(labels.detach().to(device="cpu", dtype=torch.long))
    return TensorDataset(torch.cat(feature_parts), torch.cat(label_parts))


def precompute_federated_clip_features(
    model,
    train_sets: Sequence[Dataset],
    test_sets: Sequence[Dataset],
    collate_fn: Callable,
    batch_size: int,
) -> tuple[list[TensorDataset], list[TensorDataset], FeaturePrecomputeSummary]:
    """Encode every federated image once and keep the resulting vectors on CPU."""
    if batch_size <= 0:
        raise ValueError("Feature precompute batch size must be positive.")
    if len(train_sets) != len(test_sets):
        raise ValueError("Federated train/test dataset lists must have equal length.")
    logger.info(
        "Precomputing frozen CLIP vectors for %d clients (batch_size=%d)",
        len(train_sets),
        batch_size,
    )
    encoded_train = [
        _encode_dataset(model, dataset, collate_fn, batch_size)
        for dataset in train_sets
    ]
    encoded_test = [
        _encode_dataset(model, dataset, collate_fn, batch_size)
        for dataset in test_sets
    ]
    all_sets = [*encoded_train, *encoded_test]
    storage_bytes = sum(
        tensor.numel() * tensor.element_size()
        for dataset in all_sets
        for tensor in dataset.tensors
    )
    summary = FeaturePrecomputeSummary(
        train_samples=sum(len(dataset) for dataset in encoded_train),
        test_samples=sum(len(dataset) for dataset in encoded_test),
        feature_dimension=int(model.projection_dim),
        storage_bytes=storage_bytes,
    )
    logger.info(
        "CLIP vector precompute complete: train=%d, test=%d, dim=%d, CPU=%.2f MiB",
        summary.train_samples,
        summary.test_samples,
        summary.feature_dimension,
        summary.storage_bytes / (1024**2),
    )
    return encoded_train, encoded_test, summary
