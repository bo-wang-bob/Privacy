from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


TEXT_DATASET_SPECS = {
    "sst5": {
        "text_column": "text",
        "train_split": "train",
        "evaluation_split": "validation",
        "class_names": [
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ],
    },
    "cola": {
        "text_column": "sentence",
        "train_split": "train",
        "evaluation_split": "validation",
        "class_names": ["unacceptable", "acceptable"],
    },
    "imdb": {
        "text_column": "text",
        "train_split": "train",
        "evaluation_split": "test",
        "class_names": ["negative", "positive"],
    },
}


class TextClassificationSubset(Dataset):
    """A stable indexed view over one Hugging Face text split."""

    def __init__(
        self, dataset, indices: list[int], text_column: str
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.text_column = str(text_column)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[str, int]:
        item = self.dataset[self.indices[index]]
        return str(item[self.text_column]), int(item["label"])


@dataclass(frozen=True)
class FederatedTextClassification:
    dataset_name: str
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


def normalize_text_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip().lower().replace("-", "")
    if normalized not in TEXT_DATASET_SPECS:
        raise ValueError(
            "dataset_name must be sst5, cola, or imdb."
        )
    return normalized


def load_federated_text_classification(
    dataset_name: str,
    dataset_path: str | Path,
    model_path: str | Path,
    num_users: int,
    seed: int = 42,
    max_length: int = 128,
) -> FederatedTextClassification:
    """Load a downloaded text-classification dataset and create IID splits.

    SST-5 and CoLA use validation for evaluation; IMDB uses its official
    labeled test split. Evaluation examples are never assigned to a federated
    client's training data.
    """

    from datasets import load_from_disk

    dataset_name = normalize_text_dataset_name(dataset_name)
    spec = TEXT_DATASET_SPECS[dataset_name]
    dataset_path = Path(dataset_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} was not found at {dataset_path}. Download and "
            "save the Hugging Face dataset first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"The pretrained model was not found at {model_path}. Run "
            "scripts/download_hf_sst5_models.py first."
        )
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    dataset = load_from_disk(str(dataset_path))
    required_splits = {
        str(spec["train_split"]),
        str(spec["evaluation_split"]),
    }
    if not required_splits.issubset(dataset.keys()):
        raise ValueError(
            f"{dataset_name} must contain splits {sorted(required_splits)}."
        )
    text_column = str(spec["text_column"])
    valid_labels = set(range(len(spec["class_names"])))
    for split in required_splits:
        columns = set(dataset[split].column_names)
        if not {text_column, "label"}.issubset(columns):
            raise ValueError(
                f"{dataset_name}/{split} must contain {text_column!r} and "
                "'label' columns."
            )
        labels = set(int(label) for label in dataset[split]["label"])
        if not labels.issubset(valid_labels):
            raise ValueError(
                f"{dataset_name}/{split} contains labels outside "
                f"{sorted(valid_labels)}."
            )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer has neither a pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    train_split = dataset[str(spec["train_split"])]
    evaluation_split = dataset[str(spec["evaluation_split"])]
    train_indices = _iid_indices(len(train_split), num_users, seed)
    evaluation_indices = _iid_indices(
        len(evaluation_split), num_users, seed + 1
    )
    train_sets = [
        TextClassificationSubset(train_split, indices, text_column)
        for indices in train_indices
    ]
    test_sets = [
        TextClassificationSubset(evaluation_split, indices, text_column)
        for indices in evaluation_indices
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

    return FederatedTextClassification(
        dataset_name=dataset_name,
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=list(spec["class_names"]),
        collate_fn=collate_fn,
        tokenizer=tokenizer,
    )


def load_federated_sst5(
    dataset_path: str | Path,
    model_path: str | Path,
    num_users: int,
    seed: int = 42,
    max_length: int = 128,
) -> FederatedTextClassification:
    """Convenience loader for the five-class Stanford Sentiment Treebank."""
    return load_federated_text_classification(
        dataset_name="sst5",
        dataset_path=dataset_path,
        model_path=model_path,
        num_users=num_users,
        seed=seed,
        max_length=max_length,
    )
