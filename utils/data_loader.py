import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import scipy.io as sio
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional
import logging
from utils.constants import (
    AIRCRAFT_DATASET_ALIASES,
    CALTECH256_DATASET_ALIASES,
    DATASET_MAPPING,
    DTD_DATASET_ALIASES,
    NORMALIZE_PARAMS,
    TINYIMAGENET_DATASET_ALIASES,
)
from torchvision.datasets import Caltech101

logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    # def __getitem__(self, idx):
    #     return self.data_list[idx]
    def __getitem__(self, idx):
        img, label = self.data_list[idx]
        # 统一标签为 Tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return img, label


def generate_dirichlet_split(
    dataset_name: str,
    num_users: int,
    alpha: float,
    root_dir: str = "./data",
    fpl: bool = True,
    fpl_shots: Optional[int] = None,
    use_full_dataset: bool = False,
) -> Tuple[List[Subset], List[Subset], List[str]]:
    """
    Generate a Dirichlet (non-IID) split of the dataset.

    Args:
        dataset_name: Name of the dataset
        num_users: Number of users to split data for
        alpha: Dirichlet distribution concentration parameter (smaller = more non-IID)
        root_dir: Root directory for data storage

    Returns:
        Tuple of (train_subsets, test_subsets, class_names)
        - train_subsets: List of Subset for each user's training data
        - test_subsets: List of Subset for each user's test data
        - class_names: List of class names
    """
    if not fpl:
        raise ValueError("This branch only supports FPL dataset loading.")
    logger.info(
        f"Starting Dirichlet split generation for {dataset_name} dataset "
        f"(num_users: {num_users}, alpha: {alpha})"
    )

    # Load raw datasets
    trainset, testset, class_names = _load_dataset(
        dataset_name,
        root_dir,
        fpl=fpl,
        fpl_shots=fpl_shots,
        use_full_dataset=use_full_dataset,
    )
    num_classes = len(class_names)

    # 1. Group training and test indices by class
    train_idx_by_class = group_idx_by_class(trainset, num_classes)
    test_idx_by_class = group_idx_by_class(testset, num_classes)

    # 2. Perform Dirichlet partition on training indices
    train_user_indices = dirichlet_partition_indices(
        train_idx_by_class, num_users, alpha
    )

    # 3. Calculate class distribution for each user based on training data
    user_class_distributions = []
    for user_id in range(num_users):
        user_indices = train_user_indices[user_id]
        if len(user_indices) == 0:
            class_distribution = np.zeros(num_classes)
        else:
            # Count samples per class for this user
            class_counts = np.zeros(num_classes)
            for idx in user_indices:
                _, label = trainset[idx]
                label = int(label) if not isinstance(label, int) else label
                class_counts[label] += 1
            # Normalize to get distribution
            total = class_counts.sum()
            class_distribution = (
                class_counts / total if total > 0 else np.zeros(num_classes)
            )
        user_class_distributions.append(class_distribution)

    # 4. Allocate test data based on training distribution
    test_user_indices: List[List[int]] = [[] for _ in range(num_users)]

    for cls_idx in range(num_classes):
        cls_test_indices = test_idx_by_class[cls_idx]
        cls_size = len(cls_test_indices)
        if cls_size == 0:
            continue

        # Shuffle test indices for this class
        shuffled_indices = cls_test_indices.copy()
        random.shuffle(shuffled_indices)

        # Get class proportions from training distribution
        class_proportions = [dist[cls_idx] for dist in user_class_distributions]
        total = sum(class_proportions)
        if total > 0:
            class_proportions = [p / total for p in class_proportions]
        else:
            # If no user has this class in training, distribute evenly
            class_proportions = [1.0 / num_users for _ in range(num_users)]

        # Allocate test samples based on proportions
        sample_counts = np.random.multinomial(cls_size, class_proportions)

        start_idx = 0
        for user_id in range(num_users):
            end_idx = start_idx + sample_counts[user_id]
            if end_idx > start_idx:
                test_user_indices[user_id].extend(shuffled_indices[start_idx:end_idx])
            start_idx = end_idx

    # 5. Create Subset objects for each user
    train_subsets = [Subset(trainset, indices) for indices in train_user_indices]
    test_subsets = [Subset(testset, indices) for indices in test_user_indices]

    logger.info(
        f"Dirichlet split completed! Generated training and test data for {num_users} users"
    )

    # Log statistics
    for user_id in range(min(5, num_users)):  # Log first 5 users
        logger.debug(
            f"User {user_id}: train_samples={len(train_user_indices[user_id])}, "
            f"test_samples={len(test_user_indices[user_id])}"
        )

    return train_subsets, test_subsets, class_names


def dirichlet_partition_indices(
    idx_by_class: List[List[int]],
    num_users: int,
    alpha: float,
    max_attempts: int = 100,
) -> List[List[int]]:
    """
    Allocate data indices to different users based on Dirichlet distribution.

    Args:
        idx_by_class: List of indices grouped by class
        num_users: Number of users
        alpha: Dirichlet distribution concentration parameter

    Returns:
        List of index lists for each user
    """
    if num_users <= 0:
        raise ValueError("num_users must be positive.")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    total_samples = sum(len(indices) for indices in idx_by_class)
    if total_samples < num_users:
        raise ValueError(
            "Dirichlet partition needs at least one training sample per client: "
            f"samples={total_samples}, num_users={num_users}."
        )

    # Small alpha values and many clients can occasionally produce an empty
    # client. A shuffled DataLoader cannot train on such a subset, so redraw
    # the complete Dirichlet allocation while preserving the requested model.
    for attempt in range(1, max_attempts + 1):
        user_indices: List[List[int]] = [[] for _ in range(num_users)]
        for cls_idx, cls_indices in enumerate(idx_by_class):
            cls_size = len(cls_indices)
            if cls_size == 0:
                continue

            shuffled_indices = cls_indices.copy()
            random.shuffle(shuffled_indices)
            proportions = np.random.dirichlet([alpha] * num_users)
            sample_counts = np.random.multinomial(cls_size, proportions)

            start_idx = 0
            for user_id in range(num_users):
                end_idx = start_idx + sample_counts[user_id]
                if end_idx > start_idx:
                    user_indices[user_id].extend(
                        shuffled_indices[start_idx:end_idx]
                    )
                start_idx = end_idx

            assert start_idx == cls_size, (
                f"Class {cls_idx} allocation incomplete: "
                f"should have {cls_size} samples, actually allocated {start_idx}"
            )

        if all(user_indices):
            if attempt > 1:
                logger.info(
                    "Accepted non-empty Dirichlet allocation after %d attempts.",
                    attempt,
                )
            return user_indices

    raise ValueError(
        "Dirichlet partition repeatedly produced empty clients after "
        f"{max_attempts} attempts (samples={total_samples}, users={num_users}, "
        f"alpha={alpha}). Increase fpl_shots or dirichlet_alpha, or reduce "
        "total_users."
    )


def pathological_class_assignment(
    num_classes: int,
    num_users: int,
    seed: int,
) -> List[List[int]]:
    """Assign every class to exactly one client, balancing class counts.

    Class identifiers are shuffled deterministically from ``seed`` and then
    assigned round-robin.  Consequently, client class sets are pairwise
    disjoint, their union contains every class, and class counts differ by at
    most one.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if num_users <= 0:
        raise ValueError("num_users must be positive.")
    if num_classes < num_users:
        raise ValueError(
            "Pathological label split requires at least one class per client: "
            f"num_classes={num_classes}, num_users={num_users}."
        )

    class_ids = list(range(num_classes))
    random.Random(int(seed)).shuffle(class_ids)
    assignment: List[List[int]] = [[] for _ in range(num_users)]
    for position, class_id in enumerate(class_ids):
        assignment[position % num_users].append(class_id)
    for classes in assignment:
        classes.sort()
    return assignment


def pathological_partition_indices(
    idx_by_class: List[List[int]],
    class_assignment: List[List[int]],
) -> List[List[int]]:
    """Allocate all examples of a class to its sole assigned client."""
    num_classes = len(idx_by_class)
    owners: Dict[int, int] = {}
    for user_id, class_ids in enumerate(class_assignment):
        for class_id in class_ids:
            if not 0 <= int(class_id) < num_classes:
                raise ValueError(f"Invalid class id in assignment: {class_id}")
            if int(class_id) in owners:
                raise ValueError(f"Class {class_id} is assigned to multiple clients.")
            owners[int(class_id)] = user_id
    if set(owners) != set(range(num_classes)):
        missing = sorted(set(range(num_classes)) - set(owners))
        raise ValueError(f"Pathological assignment is missing classes: {missing}")

    user_indices: List[List[int]] = [[] for _ in class_assignment]
    for class_id, indices in enumerate(idx_by_class):
        user_indices[owners[class_id]].extend(indices)
    return user_indices


def generate_pathological_split(
    dataset_name: str,
    num_users: int,
    root_dir: str = "./data",
    fpl: bool = True,
    fpl_shots: Optional[int] = None,
    use_full_dataset: bool = False,
    seed: int = 42,
) -> Tuple[List[Subset], List[Subset], List[str]]:
    """Generate a label-exclusive pathological federated split.

    The same deterministic class ownership is applied to the train and test
    splits.  Every sample is assigned exactly once, every class belongs to one
    client, and clients only train and evaluate on their owned classes.
    """
    if not fpl:
        raise ValueError("This branch only supports FPL dataset loading.")
    logger.info(
        "Starting pathological label split for %s (num_users=%d, seed=%d)",
        dataset_name,
        num_users,
        seed,
    )
    trainset, testset, class_names = _load_dataset(
        dataset_name,
        root_dir,
        fpl=fpl,
        fpl_shots=fpl_shots,
        use_full_dataset=use_full_dataset,
    )
    num_classes = len(class_names)
    assignment = pathological_class_assignment(num_classes, num_users, seed)
    train_by_class = group_idx_by_class(trainset, num_classes)
    test_by_class = group_idx_by_class(testset, num_classes)
    train_user_indices = pathological_partition_indices(train_by_class, assignment)
    test_user_indices = pathological_partition_indices(test_by_class, assignment)
    # Avoid class-block ordering in each client subset. Besides giving the
    # local loader a neutral base ordering, this keeps bounded privacy-audit
    # candidate collection from taking every example from the first owned
    # class. Separate deterministic streams preserve reproducibility.
    for user_id, indices in enumerate(train_user_indices):
        random.Random(int(seed) + 1009 * (user_id + 1)).shuffle(indices)
    for user_id, indices in enumerate(test_user_indices):
        random.Random(int(seed) + 2003 * (user_id + 1)).shuffle(indices)

    empty_train_users = [
        user_id for user_id, indices in enumerate(train_user_indices) if not indices
    ]
    if empty_train_users:
        raise ValueError(
            "Pathological split produced clients without training samples: "
            f"{empty_train_users}."
        )
    if sum(map(len, train_user_indices)) != len(trainset):
        raise AssertionError("Pathological training allocation dropped samples.")
    if sum(map(len, test_user_indices)) != len(testset):
        raise AssertionError("Pathological test allocation dropped samples.")

    train_subsets = [Subset(trainset, indices) for indices in train_user_indices]
    test_subsets = [Subset(testset, indices) for indices in test_user_indices]
    logger.info(
        "Pathological split completed: train=%d, test=%d, classes=%d, users=%d",
        len(trainset),
        len(testset),
        num_classes,
        num_users,
    )
    for user_id, class_ids in enumerate(assignment):
        logger.info(
            "Pathological user %d owns classes=%s train_samples=%d test_samples=%d",
            user_id,
            class_ids,
            len(train_user_indices[user_id]),
            len(test_user_indices[user_id]),
        )
    return train_subsets, test_subsets, class_names


def generate_iid_split(
    dataset_name: str,
    num_users: int,
    root_dir: str = "./data",
    fpl: bool = True,
    fpl_shots: Optional[int] = None,
    use_full_dataset: bool = False,
) -> Tuple[List[Subset], List[Subset], List[str]]:
    """
    Generate an IID split: for each class, distribute samples as evenly as possible
    across all users for both train and test sets.

    Args:
        dataset_name: Name of the dataset
        num_users: Number of users to split data for
        root_dir: Root directory for data storage

    Returns:
        Tuple of (train_subsets, test_subsets, class_names)
        - train_subsets: List of Subset for each user's training data
        - test_subsets: List of Subset for each user's test data
        - class_names: List of class names
    """
    if not fpl:
        raise ValueError("This branch only supports FPL dataset loading.")
    logger.info(
        f"Starting IID split generation for {dataset_name} dataset "
        f"(num_users: {num_users})"
    )

    # Load raw datasets
    trainset, testset, class_names = _load_dataset(
        dataset_name,
        root_dir,
        fpl=fpl,
        fpl_shots=fpl_shots,
        use_full_dataset=use_full_dataset,
    )
    num_classes = len(class_names)

    # Group indices by class
    train_idx_by_class = group_idx_by_class(trainset, num_classes)
    test_idx_by_class = group_idx_by_class(testset, num_classes)

    # Check if any class has fewer samples than num_users
    for cls_idx, cls_indices in enumerate(train_idx_by_class):
        if len(cls_indices) < num_users:
            raise ValueError(
                f"IID split error: Class {cls_idx} ('{class_names[cls_idx]}') has only "
                f"{len(cls_indices)} training samples, but num_users={num_users}. "
                f"For IID distribution, each class must have at least {num_users} samples. "
                f"Please reduce num_users to at most {len(cls_indices)} or use a larger dataset."
            )

    for cls_idx, cls_indices in enumerate(test_idx_by_class):
        if len(cls_indices) < num_users:
            logger.warning(
                f"IID split warning: Class {cls_idx} ('{class_names[cls_idx]}') has only "
                f"{len(cls_indices)} test samples, but num_users={num_users}. "
                f"Will distribute one sample per user to the first {len(cls_indices)} users."
            )

    # Prepare per-user index containers
    train_user_indices: List[List[int]] = [[] for _ in range(num_users)]
    test_user_indices: List[List[int]] = [[] for _ in range(num_users)]

    # Helper to evenly split class indices across users
    def _even_split_indices(
        cls_indices: List[int],
        dest_indices: List[List[int]],
        allow_partial: bool = False,
    ):
        cls_size = len(cls_indices)
        if cls_size == 0:
            return

        # Shuffle indices to avoid ordering bias
        shuffled_indices = cls_indices.copy()
        random.shuffle(shuffled_indices)

        # If allow_partial and fewer samples than users, give one to each of the first users
        if allow_partial and cls_size < num_users:
            for user_id in range(cls_size):
                dest_indices[user_id].append(shuffled_indices[user_id])
            return

        base = cls_size // num_users
        remainder = cls_size % num_users

        start = 0
        for user_id in range(num_users):
            add_count = base + (1 if user_id < remainder else 0)
            if add_count <= 0:
                continue
            end = start + add_count
            dest_indices[user_id].extend(shuffled_indices[start:end])
            start = end

        # Sanity check
        assert start == cls_size, f"Allocation mismatch: {start}!={cls_size}"

    # Split training indices evenly per class
    for cls_indices in train_idx_by_class:
        _even_split_indices(cls_indices, train_user_indices, allow_partial=False)

    # Split test indices evenly per class (allow partial distribution if samples < users)
    for cls_indices in test_idx_by_class:
        _even_split_indices(cls_indices, test_user_indices, allow_partial=True)

    # Create Subset objects for each user
    train_subsets = [Subset(trainset, indices) for indices in train_user_indices]
    test_subsets = [Subset(testset, indices) for indices in test_user_indices]

    logger.info(f"IID split completed for {num_users} users (dataset={dataset_name})")

    # Log statistics
    for user_id in range(min(5, num_users)):  # Log first 5 users
        logger.debug(
            f"User {user_id}: train_samples={len(train_user_indices[user_id])}, "
            f"test_samples={len(test_user_indices[user_id])}"
        )

    return train_subsets, test_subsets, class_names


def _load_dataset(
    dataset_name: str,
    root_dir: str = "./data",
    fpl: bool = True,
    fpl_shots: Optional[int] = None,
    use_full_dataset: bool = False,
):
    """
    Load dataset with optional transforms applied depending on ``fpl``.

    Args:
        dataset_name: Name of the dataset (mnist, fashionmnist, cifar10, cifar100, svhn, tinyimagenet, tiny-imagenet-200, caltech101, oxfordpets, flowers, food101, caltech256, fgvc-aircraft-2013b)
        root_dir: Root directory for data storage
        fpl: If True, load samples without applying data transforms (typically returning raw ``PIL.Image`` objects);
            if False, use the dataset-specific transform pipeline (e.g., via ``get_data_transform(...)``), which
            usually returns transformed tensors.

    Returns:
        Tuple of (trainset, testset, class_names)
        - trainset: Training dataset; individual samples are either raw ``PIL.Image`` objects (when ``fpl=True``)
          or transformed samples (e.g., tensors) when ``fpl=False``, depending on the underlying dataset loader.
        - testset: Test dataset with the same sample type behavior as ``trainset``
        - class_names: List of class names
    """
    if not fpl:
        raise ValueError("This branch only supports FPL dataset loading.")
    dataset = dataset_name.lower()
    if dataset not in DATASET_MAPPING:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}, available options: {list(DATASET_MAPPING.keys())}"
        )

    # Build dataset path
    dataset_full_name = DATASET_MAPPING[dataset]
    # root = (
    #     os.path.join(root_dir, dataset_full_name, "data")
    #     if dataset != "svhn"
    #     else os.path.join(root_dir, dataset_full_name)
    # )

    if dataset == "svhn":
        root = os.path.join(root_dir, dataset_full_name)
    elif dataset == "tinyimagenet" or dataset in TINYIMAGENET_DATASET_ALIASES:
        root = root_dir
    elif dataset in ["caltech101", "food101", "food-101"]:
        root = root_dir
    elif dataset in CALTECH256_DATASET_ALIASES:
        root = os.path.join(root_dir, dataset_full_name)
    elif dataset in AIRCRAFT_DATASET_ALIASES:
        root = os.path.join(root_dir, dataset_full_name)
    elif dataset in DTD_DATASET_ALIASES:
        root = os.path.join(root_dir, dataset_full_name)
    elif dataset == "oxfordpets":
        root = os.path.join(root_dir, dataset_full_name)
    elif dataset == "flowers":
        root = os.path.join(root_dir, dataset_full_name)
    else:
        root = os.path.join(root_dir, dataset_full_name, "data")

    logger.info(
        f"Loading dataset {dataset_name}, path: {root}, fpl: {fpl}"
    )

    if dataset in ["mnist", "fashionmnist", "cifar10", "cifar100"]:
        trainset, testset, class_names = _load_standard_dataset(
            dataset,
            root,
            fpl=fpl,
            use_full_dataset=use_full_dataset,
        )

    elif dataset == "svhn":
        trainset, testset, class_names = _load_svhn_dataset(
            root,
            fpl=fpl,
        )
    elif dataset == "tinyimagenet" or dataset in TINYIMAGENET_DATASET_ALIASES:
        trainset, testset, class_names = _load_tinyimagenet_dataset(
            root_dir,
            fpl=fpl,
        )
    elif dataset == "caltech101":
        trainset, testset, class_names = _load_caltech101(
            root,
            fpl=fpl,
        )
    elif dataset == "oxfordpets":
        if not fpl:
            raise ValueError(
                "Dataset 'oxfordpets' is currently only supported with fpl=True. "
                "Please enable FPL or choose a different dataset."
            )
        trainset, testset, class_names = _load_oxfordpets(
            root,
            fpl=fpl,
        )
    elif dataset == "flowers":
        trainset, testset, class_names = _load_flowers(
            root,
            fpl=fpl,
        )
    elif dataset in ["food101", "food-101"]:
        trainset, testset, class_names = _load_food101(
            root,
            fpl=fpl,
            use_full_dataset=use_full_dataset,
        )
    elif dataset in CALTECH256_DATASET_ALIASES:
        trainset, testset, class_names = _load_caltech256_object_categories(
            root,
            fpl=fpl,
        )
    elif dataset in AIRCRAFT_DATASET_ALIASES:
        trainset, testset, class_names = _load_fgvc_aircraft(
            root,
            fpl=fpl,
        )
    elif dataset in DTD_DATASET_ALIASES:
        trainset, testset, class_names = _load_dtd(
            root,
            fpl=fpl,
        )
    else:
        raise ValueError(f"Unimplemented dataset loading logic: {dataset_name}")

    if fpl and fpl_shots is not None:
        trainset = limit_dataset_per_class(
            trainset,
            num_classes=len(class_names),
            max_samples_per_class=fpl_shots,
        )
        logger.info(
            "Applied FPL few-shot training subset: shots=%d, train_samples=%d",
            fpl_shots,
            len(trainset),
        )

    return trainset, testset, class_names


def limit_dataset_per_class(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    max_samples_per_class: int,
) -> Subset:
    if max_samples_per_class <= 0:
        raise ValueError(
            f"max_samples_per_class must be positive, got {max_samples_per_class}"
        )

    idx_by_class = group_idx_by_class(dataset, num_classes)
    selected_indices: List[int] = []
    for cls_idx, cls_indices in enumerate(idx_by_class):
        if not cls_indices:
            logger.warning(
                "Few-shot subset: class %d has no training samples and is skipped.",
                cls_idx,
            )
            continue
        shuffled_indices = cls_indices.copy()
        random.shuffle(shuffled_indices)
        selected_indices.extend(shuffled_indices[:max_samples_per_class])

    selected_indices.sort()
    return Subset(dataset, selected_indices)


def _load_standard_dataset(
    dataset: str,
    root: str,
    fpl: bool = False,
    seed: int = 42,
    cifar100_train_per_class: int = 200,
    cifar100_test_per_class: int = 50,
    use_full_dataset: bool = False,
):
    """Load PyTorch officially supported standard datasets (MNIST/FashionMNIST/CIFAR series)."""
    # Dataset class mapping
    dataset_cls = {
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }[dataset]

    train_transform = None
    test_transform = None
    if not fpl:
        train_transform = get_data_transform(dataset, is_train=True)
        test_transform = get_data_transform(dataset, is_train=False)

    # Load training and test sets
    trainset = dataset_cls(
        root=root, train=True, download=False, transform=train_transform
    )
    testset = dataset_cls(
        root=root, train=False, download=False, transform=test_transform
    )
    # class_names = trainset.classes
    if dataset == "cifar10" or dataset == "cifar100":
        class_names = list(trainset.classes)

    elif dataset == "mnist":
        class_names = ["handwritten digit {}".format(i) for i in range(10)]

    elif dataset == "fashionmnist":
        class_names = [
            "T-shirt",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    else:
        raise ValueError(f"Unsupported dataset for class names: {dataset}")

    if dataset == "cifar100" and not use_full_dataset:
        train_indices = _sample_indices_per_class(
            trainset,
            cifar100_train_per_class,
            seed,
        )
        test_indices = _sample_indices_per_class(
            testset,
            cifar100_test_per_class,
            seed,
        )
        logger.info(
            "Loaded CIFAR100 subset: classes=%d, train=%d, test=%d",
            len(class_names),
            len(train_indices),
            len(test_indices),
        )
        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)
    elif dataset == "cifar100":
        logger.info(
            "Loaded full CIFAR100: classes=%d, train=%d, test=%d",
            len(class_names),
            len(trainset),
            len(testset),
        )

    return trainset, testset, class_names


class Caltech101Subset(Dataset):
    """Caltech101 subset with optional contiguous-label remapping."""

    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        transform=None,
        label_mapping: Optional[Dict[int, int]] = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.label_mapping = dict(label_mapping or {})
        # Build targets list for this subset
        self.targets = []
        for idx in indices:
            _, target = dataset[idx]
            target = int(target)
            self.targets.append(self.label_mapping.get(target, target))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, target = self.dataset[original_idx]
        target = self.label_mapping.get(int(target), int(target))
        if self.transform:
            img = self.transform(img)
        return img, target


class OxfordPetsDataset(Dataset):
    """Oxford-IIIT Pet dataset from annotation split files."""

    def __init__(self, root: str, split_file: str, transform=None):
        self.root = root
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.targets: List[int] = []

        anno_path = os.path.join(root, "annotations", split_file)
        image_dir = os.path.join(root, "images")
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"OxfordPets split file does not exist: {anno_path}")

        missing_images = 0
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                image_stem = parts[0]
                class_id = int(parts[1]) - 1  # official annotations use 1-based labels
                image_path = os.path.join(image_dir, f"{image_stem}.jpg")
                if os.path.exists(image_path):
                    self.samples.append((image_path, class_id))
                    self.targets.append(class_id)
                else:
                    missing_images += 1

        if missing_images > 0:
            logger.warning(
                "OxfordPetsDataset: %d images listed in '%s' were not found under '%s' and were skipped.",
                missing_images,
                anno_path,
                image_dir,
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class OxfordFlowersDataset(Dataset):
    """Oxford Flowers 102 dataset driven by index splits and MAT labels."""

    def __init__(self, root: str, indices: List[int], labels: np.ndarray, transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, "jpg")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Oxford Flowers image directory does not exist: {self.image_dir}")

        # MAT split indices are 1-based image ids; labels are converted to 0-based.
        self.indices = [int(i) for i in indices]
        self.labels = labels
        self.targets: List[int] = []
        self.samples: List[Tuple[str, int]] = []

        for image_id in self.indices:
            if image_id < 1 or image_id > len(self.labels):
                continue
            image_path = os.path.join(self.image_dir, f"image_{image_id:05d}.jpg")
            if not os.path.exists(image_path):
                continue
            label = int(self.labels[image_id - 1])
            self.samples.append((image_path, label))
            self.targets.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


OXFORD_FLOWERS_102_CLASS_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


def _load_oxfordpets(
    root: str,
    fpl: bool = False,
):
    """Load Oxford-IIIT Pet dataset with official trainval/test split."""

    list_file = os.path.join(root, "annotations", "list.txt")
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"OxfordPets class list does not exist: {list_file}")

    id_to_class: Dict[int, str] = {}
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            image_stem = parts[0]
            class_id = int(parts[1]) - 1
            class_name = image_stem.rsplit("_", 1)[0]
            if class_id not in id_to_class:
                id_to_class[class_id] = class_name

    class_names = [id_to_class[idx] for idx in sorted(id_to_class.keys())]

    train_transform = None
    test_transform = None
    if not fpl:
        train_transform = get_data_transform("oxfordpets", is_train=True)
        test_transform = get_data_transform("oxfordpets", is_train=False)

    trainset = OxfordPetsDataset(root, split_file="trainval.txt", transform=train_transform)
    testset = OxfordPetsDataset(root, split_file="test.txt", transform=test_transform)

    return trainset, testset, class_names


def _load_flowers(
    root: str,
    fpl: bool = False,
):
    """Load Oxford Flowers 102 with official setid split (train+valid vs test)."""

    labels_path = os.path.join(root, "imagelabels.mat")
    split_path = os.path.join(root, "setid.mat")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Oxford Flowers labels file does not exist: {labels_path}")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Oxford Flowers split file does not exist: {split_path}")

    labels_mat = sio.loadmat(labels_path)
    split_mat = sio.loadmat(split_path)

    if "labels" not in labels_mat:
        raise KeyError(f"Key 'labels' not found in Oxford Flowers labels file: {labels_path}")
    for required_key in ["trnid", "valid", "tstid"]:
        if required_key not in split_mat:
            raise KeyError(f"Key '{required_key}' not found in Oxford Flowers split file: {split_path}")

    labels = np.asarray(labels_mat["labels"]).reshape(-1).astype(np.int64) - 1
    train_indices = np.concatenate(
        [
            np.asarray(split_mat["trnid"]).reshape(-1),
            np.asarray(split_mat["valid"]).reshape(-1),
        ]
    ).astype(np.int64)
    test_indices = np.asarray(split_mat["tstid"]).reshape(-1).astype(np.int64)

    train_transform = None
    test_transform = None
    if not fpl:
        train_transform = get_data_transform("flowers", is_train=True)
        test_transform = get_data_transform("flowers", is_train=False)

    trainset = OxfordFlowersDataset(root, train_indices.tolist(), labels, transform=train_transform)
    testset = OxfordFlowersDataset(root, test_indices.tolist(), labels, transform=test_transform)

    class_names = list(OXFORD_FLOWERS_102_CLASS_NAMES)
    return trainset, testset, class_names


def _load_food101(
    root: str,
    fpl: bool = False,
    seed: int = 42,
    train_per_class: int = 200,
    test_per_class: int = 50,
    use_full_dataset: bool = False,
):
    """Load Food-101 with the official train/test split.

    torchvision.datasets.Food101 expects root to be the parent directory of
    food-101, so a local dataset at data/food-101 is loaded with root="./data".
    By default this loader retains the historical deterministic subset of up
    to 200 train images and 50 test images per class.  ``use_full_dataset``
    returns the complete official train and test splits.
    """

    train_transform = None
    test_transform = None
    if not fpl:
        train_transform = get_data_transform("food101", is_train=True)
        test_transform = get_data_transform("food101", is_train=False)

    trainset = datasets.Food101(
        root=root,
        split="train",
        download=False,
        transform=train_transform,
    )
    testset = datasets.Food101(
        root=root,
        split="test",
        download=False,
        transform=test_transform,
    )

    class_names = list(trainset.classes)
    if use_full_dataset:
        logger.info(
            "Loaded full Food101: classes=%d, train=%d, test=%d",
            len(class_names),
            len(trainset),
            len(testset),
        )
        return trainset, testset, class_names

    def sample_per_class(dataset, per_class: int) -> list[int]:
        rng = random.Random(seed)
        labels = getattr(dataset, "_labels")
        idx_by_class: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            idx_by_class.setdefault(int(label), []).append(idx)

        sampled_indices: list[int] = []
        for label in sorted(idx_by_class):
            indices = idx_by_class[label].copy()
            rng.shuffle(indices)
            sampled_indices.extend(indices[:per_class])
        rng.shuffle(sampled_indices)
        return sampled_indices

    train_indices = sample_per_class(trainset, train_per_class)
    test_indices = sample_per_class(testset, test_per_class)
    return Subset(trainset, train_indices), Subset(testset, test_indices), class_names


class FGVCAircraftDataset(Dataset):
    """FGVC Aircraft dataset backed by the official variant annotation files."""

    def __init__(
        self,
        data_root: str,
        split_file: str,
        class_to_idx: Dict[str, int],
        transform=None,
    ):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "images")
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.targets: List[int] = []

        annotation_path = os.path.join(data_root, split_file)
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(
                f"FGVC Aircraft split file does not exist: {annotation_path}"
            )
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"FGVC Aircraft image directory does not exist: {self.image_dir}"
            )

        missing_images = 0
        unknown_classes = set()
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                image_id, class_name = parts
                if class_name not in class_to_idx:
                    unknown_classes.add(class_name)
                    continue

                image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
                if os.path.exists(image_path):
                    label = class_to_idx[class_name]
                    self.samples.append((image_path, label))
                    self.targets.append(label)
                else:
                    missing_images += 1

        if unknown_classes:
            logger.warning(
                "FGVCAircraftDataset: %d unknown classes in '%s' were skipped.",
                len(unknown_classes),
                annotation_path,
            )
        if missing_images > 0:
            logger.warning(
                "FGVCAircraftDataset: %d images listed in '%s' were not found under '%s' and were skipped.",
                missing_images,
                annotation_path,
                self.image_dir,
            )
        if not self.samples:
            raise ValueError(f"No FGVC Aircraft samples found from: {annotation_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _resolve_fgvc_aircraft_data_root(root: str) -> str:
    candidates = [
        root,
        os.path.join(root, "data"),
        os.path.join(root, "fgvc-aircraft-2013b", "data"),
    ]
    for candidate in candidates:
        if (
            os.path.isdir(os.path.join(candidate, "images"))
            and os.path.exists(os.path.join(candidate, "variants.txt"))
            and os.path.exists(os.path.join(candidate, "images_variant_trainval.txt"))
            and os.path.exists(os.path.join(candidate, "images_variant_test.txt"))
        ):
            return candidate
    raise FileNotFoundError(
        "FGVC Aircraft 2013b path does not exist. Checked: "
        + ", ".join(candidates)
    )


def _load_fgvc_aircraft(
    root: str,
    fpl: bool = False,
    seed: int = 42,
    fpl_train_per_class: int = 16,
    fpl_test_per_class: int = 50,
):
    """Load FGVC Aircraft 2013b at variant level using trainval/test splits.

    In FPL mode, keep a deterministic per-class subset for prompt learning and
    evaluation: up to 16 trainval images and up to 50 test images per class.
    """

    data_root = _resolve_fgvc_aircraft_data_root(root)
    variants_path = os.path.join(data_root, "variants.txt")
    with open(variants_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    if not class_names:
        raise ValueError(f"No FGVC Aircraft classes found in: {variants_path}")

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    train_transform = None
    test_transform = None
    if not fpl:
        train_transform = get_data_transform("fgvc-aircraft-2013b", is_train=True)
        test_transform = get_data_transform("fgvc-aircraft-2013b", is_train=False)

    trainset = FGVCAircraftDataset(
        data_root,
        split_file="images_variant_trainval.txt",
        class_to_idx=class_to_idx,
        transform=train_transform,
    )
    testset = FGVCAircraftDataset(
        data_root,
        split_file="images_variant_test.txt",
        class_to_idx=class_to_idx,
        transform=test_transform,
    )

    if fpl:
        train_indices = _sample_indices_per_class(trainset, fpl_train_per_class, seed)
        test_indices = _sample_indices_per_class(testset, fpl_test_per_class, seed)
        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)
        logger.info(
            "Applied FPL FGVC Aircraft subset: train_per_class=%d, test_per_class=%d, train=%d, test=%d",
            fpl_train_per_class,
            fpl_test_per_class,
            len(trainset),
            len(testset),
        )

    logger.info(
        "Loaded FGVC Aircraft 2013b from %s: classes=%d, train=%d, test=%d",
        data_root,
        len(class_names),
        len(trainset),
        len(testset),
    )
    return trainset, testset, class_names


class Caltech256ObjectCategoriesDataset(Dataset):
    """Dataset backed by Caltech-256 object category image paths."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        class_names: List[str],
        transform=None,
    ):
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.classes = class_names
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _clean_caltech256_class_name(folder_name: str) -> str:
    class_name = folder_name.split(".", 1)[1] if "." in folder_name else folder_name
    if class_name.endswith("-101"):
        class_name = class_name[:-4]
    return class_name.replace("-", " ").replace("_", " ").strip()


def _load_caltech256_object_categories(
    root: str,
    seed: int = 42,
    train_per_class: int = 60,
    fpl_train_per_class: int = 16,
    fpl_test_per_class: int = 50,
    remove_clutter: bool = True,
    fpl: bool = False,
):
    """Load Caltech-256 from data/256_ObjectCategories.

    Caltech-256 does not provide a canonical split file in the raw directory.
    This loader uses a deterministic per-class split. In normal mode it keeps
    up to 60 training images per class and uses the remaining images for
    testing. In FPL mode it keeps up to 16 training images and up to 50 testing
    images per class. The optional 257.clutter directory is excluded by default
    so the dataset has 256 object classes.
    """

    data_root = root
    if not os.path.isdir(data_root):
        nested_root = os.path.join(root, "256_ObjectCategories")
        if os.path.isdir(nested_root):
            data_root = nested_root
        else:
            raise FileNotFoundError(
                f"Caltech-256 object categories directory does not exist: {root}"
            )

    train_transform = None
    test_transform = None
    if not fpl:
        train_transform = get_data_transform("caltech256", is_train=True)
        test_transform = get_data_transform("caltech256", is_train=False)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    class_dirs = [
        name
        for name in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, name))
    ]
    if remove_clutter:
        class_dirs = [
            name
            for name in class_dirs
            if not (
                name.lower() == "clutter"
                or name.lower().startswith("257.")
                or name.lower().endswith(".clutter")
            )
        ]

    rng = random.Random(seed)
    class_names: List[str] = []
    train_samples: List[Tuple[str, int]] = []
    test_samples: List[Tuple[str, int]] = []

    for class_dir in class_dirs:
        class_path = os.path.join(data_root, class_dir)
        image_paths = [
            os.path.join(class_path, filename)
            for filename in sorted(os.listdir(class_path))
            if filename.lower().endswith(image_extensions)
        ]
        if not image_paths:
            logger.warning("Skipping empty Caltech-256 class directory: %s", class_path)
            continue

        class_idx = len(class_names)
        class_names.append(_clean_caltech256_class_name(class_dir))

        shuffled_paths = image_paths.copy()
        rng.shuffle(shuffled_paths)
        effective_train_per_class = fpl_train_per_class if fpl else train_per_class
        if len(shuffled_paths) >= 2:
            n_train = min(effective_train_per_class, max(1, len(shuffled_paths) - 1))
        else:
            n_train = 1

        train_samples.extend((path, class_idx) for path in shuffled_paths[:n_train])
        test_paths = shuffled_paths[n_train:]
        if fpl:
            test_paths = test_paths[:fpl_test_per_class]
        test_samples.extend((path, class_idx) for path in test_paths)

    if not class_names:
        raise ValueError(f"No Caltech-256 classes found under: {data_root}")

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    trainset = Caltech256ObjectCategoriesDataset(
        train_samples,
        class_names,
        transform=train_transform,
    )
    testset = Caltech256ObjectCategoriesDataset(
        test_samples,
        class_names,
        transform=test_transform,
    )

    logger.info(
        "Loaded Caltech-256 object categories from %s: classes=%d, train=%d, test=%d",
        data_root,
        len(class_names),
        len(trainset),
        len(testset),
    )
    return trainset, testset, class_names


class DTDDataset(Dataset):
    """Describable Textures Dataset (DTD) backed by official split files."""

    def __init__(self, root: str, split_files: List[str], transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"DTD image directory does not exist: {self.image_dir}")

        # Derive sorted class names from image subdirectories
        class_names = sorted(
            name
            for name in os.listdir(self.image_dir)
            if os.path.isdir(os.path.join(self.image_dir, name))
        )
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(class_names)
        }

        self.samples: List[Tuple[str, int]] = []
        self.targets: List[int] = []

        missing = 0
        for split_file in split_files:
            split_path = os.path.join(self.labels_dir, split_file)
            if not os.path.exists(split_path):
                raise FileNotFoundError(f"DTD split file does not exist: {split_path}")
            with open(split_path, "r", encoding="utf-8") as f:
                for line in f:
                    rel_path = line.strip()
                    if not rel_path:
                        continue
                    class_name = rel_path.split("/")[0]
                    if class_name not in self.class_to_idx:
                        continue
                    image_path = os.path.join(self.image_dir, rel_path)
                    if os.path.exists(image_path):
                        label = self.class_to_idx[class_name]
                        self.samples.append((image_path, label))
                        self.targets.append(label)
                    else:
                        missing += 1

        if missing > 0:
            logger.warning(
                "DTDDataset: %d images listed in split files were not found and were skipped.",
                missing,
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _load_dtd(
    root: str,
    fpl: bool = False,
    fold: int = 1,
):
    """Load the Describable Textures Dataset (DTD) using official split files.

    Uses fold ``fold`` (default 1). Training images come from train<fold>.txt
    and val<fold>.txt combined; test images come from test<fold>.txt.
    """
    image_dir = os.path.join(root, "images")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"DTD image directory does not exist: {image_dir}")

    class_names = sorted(
        name
        for name in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, name))
    )
    if not class_names:
        raise ValueError(f"No DTD classes found under: {image_dir}")

    train_transform = None
    test_transform = None
    if not fpl:
        train_transform = get_data_transform("dtd", is_train=True)
        test_transform = get_data_transform("dtd", is_train=False)

    trainset = DTDDataset(
        root,
        split_files=[f"train{fold}.txt", f"val{fold}.txt"],
        transform=train_transform,
    )
    testset = DTDDataset(
        root,
        split_files=[f"test{fold}.txt"],
        transform=test_transform,
    )

    logger.info(
        "Loaded DTD (fold %d) from %s: classes=%d, train=%d, test=%d",
        fold,
        root,
        len(class_names),
        len(trainset),
        len(testset),
    )
    return trainset, testset, class_names


def _load_caltech101(
    root: str,
    seed: int = 42,
    protocol: str = "per_class",
    train_per_class: int = 30,
    train_ratio: float = 0.5,
    remove_background: bool = True,
    fpl: bool = False,
):
    """Load Caltech101 dataset

    Returns:
        trainset: Training dataset with PIL.Image samples
        testset: Test dataset with PIL.Image samples
        class_names: List of class names
    """
    raw_candidates = [
        os.path.join(root, "caltech-101", "101_ObjectCategories"),
        os.path.join(root, "101_ObjectCategories"),
        os.path.join(root, "caltech101", "101_ObjectCategories"),
    ]
    raw_directory = next(
        (candidate for candidate in raw_candidates if os.path.isdir(candidate)),
        None,
    )
    if raw_directory is not None:
        ds: Dataset = ImageFolder(raw_directory, transform=None)
        source_class_names = list(ds.classes)  # type: ignore[attr-defined]
        raw_targets = list(ds.targets)  # type: ignore[attr-defined]
    else:
        torchvision_dataset = Caltech101(
            root=root,
            download=False,
            target_type="category",
            transform=None,
        )
        ds = torchvision_dataset
        source_class_names = list(torchvision_dataset.categories)
        raw_targets = list(
            torchvision_dataset.y
            if hasattr(torchvision_dataset, "y")
            else torchvision_dataset.targets  # type: ignore[attr-defined]
        )
    class_names = list(source_class_names)
    name2idx = {name: idx for idx, name in enumerate(source_class_names)}

    def to_int_target(t):
        if isinstance(t, int):
            return t
        if isinstance(t, str) and t in name2idx:
            return name2idx[t]
        return int(t)

    targets = [to_int_target(target) for target in raw_targets]

    valid_indices = list(range(len(ds)))
    label_mapping = {index: index for index in range(len(source_class_names))}
    if remove_background and "BACKGROUND_Google" in name2idx:
        bg_idx = name2idx["BACKGROUND_Google"]
        valid_indices = [i for i in valid_indices if targets[i] != bg_idx]
        kept_labels = [
            old_label
            for old_label, name in enumerate(source_class_names)
            if name != "BACKGROUND_Google"
        ]
        label_mapping = {
            old_label: new_label for new_label, old_label in enumerate(kept_labels)
        }
        class_names = [source_class_names[old_label] for old_label in kept_labels]

    class2indices = {}
    for i in valid_indices:
        y = targets[i]
        class2indices.setdefault(y, []).append(i)

    rng = random.Random(seed)
    train_idx, test_idx = [], []
    protocol = protocol.lower()

    for y, idxs in class2indices.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)

        if protocol == "per_class":
            n_train = (
                min(train_per_class, max(1, len(idxs) - 1)) if len(idxs) >= 2 else 1
            )
        else:
            n_train = max(1, int(round(len(idxs) * train_ratio)))
            n_train = min(n_train, len(idxs) - 1) if len(idxs) >= 2 else 1

        train_idx.extend(idxs[:n_train])
        test_idx.extend(idxs[n_train:])

    train_transform = None
    test_transform = None
    if not fpl:
        train_transform = get_data_transform("caltech101", is_train=True)
        test_transform = get_data_transform("caltech101", is_train=False)

    # Create subset datasets with PIL.Image format (if fpl then no transform)
    trainset = Caltech101Subset(
        ds,
        train_idx,
        transform=train_transform,
        label_mapping=label_mapping,
    )
    testset = Caltech101Subset(
        ds,
        test_idx,
        transform=test_transform,
        label_mapping=label_mapping,
    )

    return trainset, testset, class_names


def _load_svhn_dataset(root: str, fpl: bool = False):
    """Load SVHN dataset using torchvision.datasets.SVHN

    Returns:
        trainset: Training dataset with PIL.Image samples
        testset: Test dataset with PIL.Image samples
        class_names: List of class names (digits 0-9)
    """
    # Load training and test sets without transform
    train_tranform = None
    test_transform = None
    if not fpl:
        train_tranform = get_data_transform("svhn", is_train=True)
        test_transform = get_data_transform("svhn", is_train=False)
    trainset = datasets.SVHN(
        root=root, split="train", download=False, transform=train_tranform
    )
    testset = datasets.SVHN(
        root=root, split="test", download=False, transform=test_transform
    )

    # SVHN class names (digits 0-9)
    class_names = ["digit {}".format(i) for i in range(10)]

    return trainset, testset, class_names


class TinyImageNetValDataset(Dataset):
    """Custom Dataset for TinyImageNet validation set with PIL.Image samples"""

    def __init__(
        self, val_dir: str, anno_path: str, class_to_idx: Dict[str, int], transform=None
    ):
        self.val_img_dir = os.path.join(val_dir, "images")
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Read annotation file (format: image_name class_name ...)
        self.samples = []
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                class_name = parts[1]
                if class_name in class_to_idx:
                    img_path = os.path.join(self.val_img_dir, img_name)
                    if os.path.exists(img_path):
                        self.samples.append((img_path, class_to_idx[class_name]))

        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _sample_indices_per_class(dataset, per_class: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    labels = getattr(dataset, "targets")
    idx_by_class: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        idx_by_class.setdefault(int(label), []).append(idx)

    sampled_indices: List[int] = []
    for label in sorted(idx_by_class):
        indices = idx_by_class[label].copy()
        rng.shuffle(indices)
        sampled_indices.extend(indices[:per_class])
    rng.shuffle(sampled_indices)
    return sampled_indices


def _resolve_tinyimagenet_root(save_dir: str) -> str:
    candidates = [
        os.path.join(save_dir, "tiny-imagenet-200"),
        os.path.join(save_dir, "TinyImageNet", "data", "tiny-imagenet-200"),
        os.path.join(save_dir, "TinyImageNet", "tiny-imagenet-200"),
    ]
    for candidate in candidates:
        train_dir = os.path.join(candidate, "train")
        val_anno = os.path.join(candidate, "val", "val_annotations.txt")
        if os.path.isdir(train_dir) and os.path.exists(val_anno):
            return candidate
    raise FileNotFoundError(
        "TinyImageNet path does not exist. Checked: "
        + ", ".join(candidates)
    )


def _load_tinyimagenet_dataset(
    save_dir: str,
    fpl: bool = False,
    seed: int = 42,
    train_per_class: int = 200,
    test_per_class: int = 25,
):
    """Load TinyImageNet-200 dataset (custom directory structure).

    The loader supports data/tiny-imagenet-200 and legacy TinyImageNet paths,
    then samples a deterministic subset of up to 200 train images and 25
    validation images per class.

    Returns:
        trainset: Training dataset with PIL.Image samples
        testset: Test/validation dataset with PIL.Image samples
        class_names: List of class names (200 classes)
    """
    data_root = _resolve_tinyimagenet_root(save_dir)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    val_anno = os.path.join(val_dir, "val_annotations.txt")

    # Check path validity
    for path in [train_dir, val_anno]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"TinyImageNet path does not exist: {path}")

    train_transform = None
    test_transform = None
    if not fpl:
        train_transform = get_data_transform("tinyimagenet", is_train=True)
        test_transform = get_data_transform("tinyimagenet", is_train=False)

    # Load training set without transform (PIL.Image format)
    trainset = ImageFolder(train_dir, transform=train_transform)
    # Get class names from training set folder structure
    class_names = trainset.classes  # List of class folder names (wnids)

    # Load validation set without transform (PIL.Image format)
    testset = TinyImageNetValDataset(
        val_dir,
        val_anno,
        trainset.class_to_idx,
        transform=test_transform,
    )

    train_indices = _sample_indices_per_class(trainset, train_per_class, seed)
    test_indices = _sample_indices_per_class(testset, test_per_class, seed)

    logger.info(
        "Loaded TinyImageNet-200 from %s: classes=%d, train=%d, test=%d",
        data_root,
        len(class_names),
        len(train_indices),
        len(test_indices),
    )
    return Subset(trainset, train_indices), Subset(testset, test_indices), class_names


def group_idx_by_class(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
) -> List[List[int]]:
    """
    按类别分组图像数据

    Args:
        dataset: 图像数据集 (PIL.Image format)
        num_classes: 类别数量``
    Returns:
        按类别分组的图像数据索引列表，格式为 [list_class0, list_class1, ..., list_classN]
    """
    # 初始化每个类别的索引列表
    class_indices: List[List[int]] = [[] for _ in range(num_classes)]

    if isinstance(dataset, Subset):
        subset_indices = list(dataset.indices)
        base_dataset = dataset.dataset
        for attr_name in ("targets", "labels", "_labels", "y"):
            candidate = getattr(base_dataset, attr_name, None)
            if candidate is not None and len(candidate) == len(base_dataset):  # type: ignore
                for subset_pos, base_idx in enumerate(subset_indices):
                    label = candidate[base_idx]
                    if isinstance(label, torch.Tensor):
                        label = int(label.item())
                    class_indices[int(label)].append(subset_pos)
                return class_indices

    labels = None
    for attr_name in ("targets", "labels", "_labels", "y"):
        candidate = getattr(dataset, attr_name, None)
        if candidate is not None and len(candidate) == len(dataset):  # type: ignore
            labels = candidate
            break

    if labels is not None:
        for idx, label in enumerate(labels):
            if isinstance(label, torch.Tensor):
                label = int(label.item())
            class_indices[int(label)].append(idx)
        return class_indices

    # 遍历数据集，按类别分组索引
    for idx in range(len(dataset)):  # type: ignore
        _, label = dataset[idx]
        if isinstance(label, int):
            class_indices[label].append(idx)
        else:
            class_indices[int(label)].append(idx)
    return class_indices


def get_data_transform(dataset_name: str, is_train: bool = True) -> transforms.Compose:
    dataset = dataset_name.lower()
    if dataset not in NORMALIZE_PARAMS:
        raise ValueError(f"Unsupported dataset transform: {dataset_name}")

    mean, std = NORMALIZE_PARAMS[dataset]

    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    # if dataset == "caltech101":
    #     base_transforms = [
    #         transforms.ToTensor(),
    #         transforms.Grayscale(num_output_channels=3),
    #         transforms.Normalize(mean, std),
    #     ]
    # else:

    # 训练集数据增强（按数据集特性定制）
    if is_train:
        augment_transforms = []
        if dataset in ["mnist", "fashionmnist"]:
            augment_transforms.append(transforms.RandomRotation(10))
            # Grayscale images only rotation
            if dataset == "fashionmnist":
                augment_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        elif dataset in ["cifar10", "cifar100"]:
            augment_transforms.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                ]
            )
            if dataset == "cifar100":
                augment_transforms.append(transforms.RandomRotation(15))
                # CIFAR100 larger rotation angle
        elif dataset == "svhn":
            augment_transforms.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                ]
            )
        elif dataset == "tinyimagenet" or dataset in TINYIMAGENET_DATASET_ALIASES:
            augment_transforms.extend(
                [
                    transforms.Resize((64, 64)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                ]
            )
        elif dataset == "caltech101":
            augment_transforms.extend(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ]
            )
        elif dataset == "oxfordpets":
            augment_transforms.extend(
                [
                    transforms.Resize((64, 64)),
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                ]
            )
        elif dataset == "flowers":
            augment_transforms.extend(
                [
                    transforms.Resize((64, 64)),
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                ]
            )
        elif dataset in ["food101", "food-101"]:
            augment_transforms.extend(
                [
                    transforms.Resize((72, 72)),
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                    ),
                ]
            )
        elif dataset in AIRCRAFT_DATASET_ALIASES:
            augment_transforms.extend(
                [
                    transforms.Resize((72, 72)),
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                    ),
                ]
            )
        elif dataset in CALTECH256_DATASET_ALIASES:
            augment_transforms.extend(
                [
                    transforms.Resize((72, 72)),
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                ]
            )
        elif dataset in DTD_DATASET_ALIASES:
            augment_transforms.extend(
                [
                    transforms.Resize((72, 72)),
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                ]
            )
        else:
            raise ValueError(f"Unsupported dataset transform: {dataset_name}")
        # Combine augmentation transforms with base transforms
        return transforms.Compose(augment_transforms + base_transforms)

    # Test set only basic transforms
    if dataset == "tinyimagenet" or dataset in TINYIMAGENET_DATASET_ALIASES:
        return transforms.Compose([transforms.Resize((64, 64))] + base_transforms)
    elif dataset == "caltech101":
        return transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize((64, 64))] + base_transforms)
    elif dataset == "oxfordpets":
        return transforms.Compose([transforms.Resize((64, 64))] + base_transforms)
    elif dataset == "flowers":
        return transforms.Compose([transforms.Resize((64, 64))] + base_transforms)
    elif dataset in ["food101", "food-101"]:
        return transforms.Compose([transforms.Resize((64, 64))] + base_transforms)
    elif dataset in AIRCRAFT_DATASET_ALIASES:
        return transforms.Compose([transforms.Resize((64, 64))] + base_transforms)
    elif dataset in CALTECH256_DATASET_ALIASES:
        return transforms.Compose([transforms.Resize((64, 64))] + base_transforms)
    elif dataset in DTD_DATASET_ALIASES:
        return transforms.Compose([transforms.Resize((64, 64))] + base_transforms)
    else:
        return transforms.Compose(base_transforms)

# class TransformWrapper(Dataset):
#     def __init__(self, base_ds, transform=None):
#         self.base_ds = base_ds
#         self.transform = transform

#     def __len__(self):
#         return len(self.base_ds)

#     def __getitem__(self, idx):
#         img, label = self.base_ds[idx]  # img通常是PIL
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, int(label)
