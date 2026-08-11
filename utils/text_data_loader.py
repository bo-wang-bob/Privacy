from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


SST2_CLASS_NAMES = ["negative", "positive"]


class SST2Subset(Dataset):
    """A stable indexed view over a Hugging Face SST-2 split."""

    def __init__(self, dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[str, int]:
        item = self.dataset[self.indices[index]]
        return str(item["sentence"]), int(item["label"])


@dataclass(frozen=True)
class FederatedSST2:
    train_sets: list[Dataset]
    test_sets: list[Dataset]
    class_names: list[str]
    collate_fn: Callable
    tokenizer: PreTrainedTokenizerBase


def _iid_indices(length: int, num_users: int, seed: int) -> list[list[int]]:
    if num_users <= 1:
        raise ValueError("num_users must be greater than one.")
    if length < num_users:
        raise ValueError(
            f"Cannot split {length} samples across {num_users} non-empty clients."
        )
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(length, generator=generator).tolist()
    base, remainder = divmod(length, num_users)
    result = []
    offset = 0
    for user_id in range(num_users):
        size = base + (1 if user_id < remainder else 0)
        result.append(permutation[offset : offset + size])
        offset += size
    return result


def load_federated_sst2(
    dataset_path: str | Path,
    model_path: str | Path,
    num_users: int,
    seed: int = 42,
    max_length: int = 128,
) -> FederatedSST2:
    """Load the downloaded SST-2 artifact and create an IID client split.

    The official GLUE test labels are hidden, so the validation split is used
    as the independently held-out federated evaluation set.
    """

    from datasets import load_from_disk

    dataset_path = Path(dataset_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SST-2 was not found at {dataset_path}. Run "
            "scripts/download_hf_sst2_models.py first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"The pretrained model was not found at {model_path}. Run "
            "scripts/download_hf_sst2_models.py first."
        )
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    dataset = load_from_disk(str(dataset_path))
    if not {"train", "validation"}.issubset(dataset.keys()):
        raise ValueError("SST-2 must contain train and validation splits.")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer has neither a pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    train_indices = _iid_indices(len(dataset["train"]), num_users, seed)
    validation_indices = _iid_indices(
        len(dataset["validation"]), num_users, seed + 1
    )
    train_sets = [
        SST2Subset(dataset["train"], indices) for indices in train_indices
    ]
    test_sets = [
        SST2Subset(dataset["validation"], indices)
        for indices in validation_indices
    ]

    def collate_fn(batch: list[tuple[str, int]]):
        sentences, labels = zip(*batch)
        encoded = tokenizer(
            list(sentences),
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
            return_attention_mask=True,
        )
        # The existing federated client API accepts one tensor plus labels.
        # Channel 0 stores token IDs and channel 1 stores the attention mask.
        packed_inputs = torch.stack(
            (encoded["input_ids"], encoded["attention_mask"]), dim=1
        )
        return packed_inputs, torch.as_tensor(labels, dtype=torch.long)

    return FederatedSST2(
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=list(SST2_CLASS_NAMES),
        collate_fn=collate_fn,
        tokenizer=tokenizer,
    )
