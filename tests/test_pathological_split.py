from __future__ import annotations

import torch
import pytest
from torch.utils.data import Dataset

import utils.data_loader as data_loader
from main import default_config, validate_config


class ToyDataset(Dataset):
    def __init__(self, labels: list[int]):
        self.targets = list(labels)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.tensor([float(index)]), self.targets[index]


def _labels(subset) -> set[int]:
    return {int(subset[index][1]) for index in range(len(subset))}


def test_pathological_assignment_is_disjoint_complete_balanced_and_seeded():
    first = data_loader.pathological_class_assignment(11, 4, seed=42)
    second = data_loader.pathological_class_assignment(11, 4, seed=42)
    different = data_loader.pathological_class_assignment(11, 4, seed=43)

    assert first == second
    assert first != different
    assert sorted(class_id for classes in first for class_id in classes) == list(
        range(11)
    )
    assert max(map(len, first)) - min(map(len, first)) <= 1
    for user_id, classes in enumerate(first):
        for other_id, other_classes in enumerate(first):
            if user_id != other_id:
                assert set(classes).isdisjoint(other_classes)


def test_pathological_split_reuses_class_ownership_for_all_train_and_test_data(
    monkeypatch,
):
    train = ToyDataset([class_id for class_id in range(6) for _ in range(3)])
    test = ToyDataset([class_id for class_id in range(6) for _ in range(2)])
    observed = {}

    def fake_load_dataset(
        dataset_name,
        root_dir,
        fpl,
        fpl_shots,
        use_full_dataset,
    ):
        observed.update(
            dataset_name=dataset_name,
            root_dir=root_dir,
            fpl=fpl,
            fpl_shots=fpl_shots,
            use_full_dataset=use_full_dataset,
        )
        return train, test, [f"class-{index}" for index in range(6)]

    monkeypatch.setattr(data_loader, "_load_dataset", fake_load_dataset)
    train_sets, test_sets, class_names = data_loader.generate_pathological_split(
        "toy",
        num_users=3,
        root_dir="/unused",
        fpl_shots=None,
        use_full_dataset=True,
        seed=7,
    )

    assert observed == {
        "dataset_name": "toy",
        "root_dir": "/unused",
        "fpl": True,
        "fpl_shots": None,
        "use_full_dataset": True,
    }
    assert class_names == [f"class-{index}" for index in range(6)]
    assert sum(map(len, train_sets)) == len(train)
    assert sum(map(len, test_sets)) == len(test)

    repeated_train, repeated_test, _ = data_loader.generate_pathological_split(
        "toy",
        num_users=3,
        root_dir="/unused",
        fpl_shots=None,
        use_full_dataset=True,
        seed=7,
    )
    assert [subset.indices for subset in train_sets] == [
        subset.indices for subset in repeated_train
    ]
    assert [subset.indices for subset in test_sets] == [
        subset.indices for subset in repeated_test
    ]

    for user_id in range(3):
        assert _labels(train_sets[user_id]) == _labels(test_sets[user_id])
        assert len(_labels(train_sets[user_id])) == 2
        for other_id in range(3):
            if user_id != other_id:
                assert _labels(train_sets[user_id]).isdisjoint(
                    _labels(train_sets[other_id])
                )


def test_full_dataset_mode_rejects_a_remaining_few_shot_cap():
    config = default_config()
    config["use_full_dataset"] = True
    config["fpl_shots"] = 16
    with pytest.raises(ValueError, match="use_full_dataset=true requires fpl_shots=null"):
        validate_config(config)
