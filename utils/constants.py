from typing import Dict, Tuple

DTD_DATASET_ALIASES: Tuple[str, ...] = (
    "dtd",
    "describable-textures",
    "describabletextures",
)

CALTECH256_DATASET_ALIASES: Tuple[str, ...] = (
    "caltech256",
    "caltech-256",
    "256_objectcategories",
    "256-objectcategories",
    "256objectcategories",
)

TINYIMAGENET_DATASET_ALIASES: Tuple[str, ...] = (
    "tiny-imagenet",
    "tinyimagenet200",
    "tiny-imagenet-200",
)

AIRCRAFT_DATASET_ALIASES: Tuple[str, ...] = (
    "aircraft",
    "fgvcaircraft",
    "fgvc-aircraft",
    "fgvc_aircraft",
    "fgvc-aircraft-2013b",
    "fgvc_aircraft_2013b",
)

SUPPORTED_FPL_ATTACKS = ("cerberus", "a3fl", "sabre")

# A3FL and SABRE use the shared poisoned local-training routine. Cerberus has
# its own collusion-aware objective.
SHARED_POISON_TRAIN_ATTACKS = ("a3fl", "sabre")

DATASET_MAPPING: Dict[str, str] = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "svhn": "SVHN",
    "tinyimagenet": "TinyImageNet",
    **{alias: "tiny-imagenet-200" for alias in TINYIMAGENET_DATASET_ALIASES},
    "caltech101": "caltech101",
    "oxfordpets": "OxfordPets",
    "flowers": "Flowers",
    "food101": "Food101",
    "food-101": "Food101",
    **{alias: "256_ObjectCategories" for alias in CALTECH256_DATASET_ALIASES},
    **{alias: "fgvc-aircraft-2013b" for alias in AIRCRAFT_DATASET_ALIASES},
    **{alias: "dtd" for alias in DTD_DATASET_ALIASES},
}

NORMALIZE_PARAMS: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashionmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    **{
        alias: ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        for alias in TINYIMAGENET_DATASET_ALIASES
    },
    "caltech101": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "oxfordpets": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "flowers": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "food101": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "food-101": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    **{
        alias: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        for alias in CALTECH256_DATASET_ALIASES
    },
    **{
        alias: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        for alias in AIRCRAFT_DATASET_ALIASES
    },
    **{
        alias: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        for alias in DTD_DATASET_ALIASES
    },
}


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
