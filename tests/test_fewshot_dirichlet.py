from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from utils.data_loader import (
    dirichlet_partition_indices,
    group_idx_by_class,
    limit_dataset_per_class,
)


class LabeledDataset(Dataset):
    def __init__(self, samples_per_class: int, num_classes: int):
        self.targets = [
            class_id
            for class_id in range(num_classes)
            for _ in range(samples_per_class)
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.tensor([index], dtype=torch.float32), self.targets[index]


def test_fewshot_cap_is_global_before_dirichlet_partition():
    random.seed(42)
    np.random.seed(42)
    dataset = LabeledDataset(samples_per_class=30, num_classes=20)

    fewshot = limit_dataset_per_class(
        dataset,
        num_classes=20,
        max_samples_per_class=16,
    )
    grouped = group_idx_by_class(fewshot, num_classes=20)
    users = dirichlet_partition_indices(grouped, num_users=10, alpha=0.1)

    assert [len(indices) for indices in grouped] == [16] * 20
    assert sum(map(len, users)) == 20 * 16
    assert all(users)
    assert sorted(index for indices in users for index in indices) == list(
        range(20 * 16)
    )


def test_dirichlet_partition_rejects_more_clients_than_samples():
    with pytest.raises(ValueError, match="at least one training sample per client"):
        dirichlet_partition_indices([[0], [1]], num_users=3, alpha=0.1)
