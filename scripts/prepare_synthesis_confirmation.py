#!/usr/bin/env python
"""Reserve unused CIFAR100 source records without training or scoring them.

This writes a partition manifest only. It does not change the current loader
or any existing experiment, and is not itself a training entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def reserve_indices(labels, excluded, *, holdout_per_class=100, train_shots=100, seed=20260912):
    labels = np.asarray(labels)
    excluded = np.asarray(excluded, dtype=int)
    if (len(np.unique(excluded)) != len(excluded) or len(excluded) == 0
            or excluded.min() < 0 or excluded.max() >= len(labels)):
        raise ValueError("Excluded source identities must be unique and in range.")
    if holdout_per_class < 1 or train_shots < 1:
        raise ValueError("Holdout and training quotas must be positive.")
    allowed = np.ones(len(labels), dtype=bool)
    allowed[excluded] = False
    rng = np.random.default_rng(seed)
    train, evaluation = [], []
    for label in np.unique(labels):
        candidates = np.flatnonzero(allowed & (labels == label))
        if len(candidates) < holdout_per_class + train_shots:
            raise ValueError(f"Class {label} has too few unused records for both partitions.")
        order = rng.permutation(candidates)
        evaluation.extend(order[:holdout_per_class].tolist())
        train.extend(order[holdout_per_class:].tolist())
    return dict(train_pool_indices=sorted(train), evaluation_indices=sorted(evaluation),
                excluded_exploration_indices=sorted(excluded.tolist()))


def original_indices(dataset):
    from torch.utils.data import Subset
    indices = np.arange(len(dataset))
    while isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices)[indices]
        dataset = dataset.dataset
    return dataset, indices.astype(int)


def prepare(reference, replay_path, output, seed):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import yaml
    from main import _dataset_split_arguments
    from scripts.analyze_risk_synthesis import digest
    from utils.data_loader import generate_iid_split

    reference = reference.resolve()
    config_path = reference/"run_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    summary = json.loads((reference/"risk_synthesis/synthesis_summary.json").read_text())
    if (config["dataset_name"].lower() != "cifar100" or config["partition_mode"] != "iid"
            or config["model_type"] not in {"clip_adapter", "clip_lora"}
            or config["fpl_shots"] != 100 or config["total_users"] != 10
            or config["seed"] != 42
            or summary["status"] != "completed"):
        raise ValueError("Expected a completed seed-42 CIFAR100 100-per-class/10-client IID synthesis reference.")
    replay = json.loads(replay_path.read_text())
    replay_hashes = {r["client"]: r["source_sha256"] for r in replay["clients"]}
    if len(replay_hashes) != 10 or not all(r["requests_verified"] for r in replay["clients"]):
        raise ValueError("The source-code replay certificate is incomplete.")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    clients, _, names = generate_iid_split(config["dataset_name"], config["total_users"],
                                          **_dataset_split_arguments(config))
    mappings, roots, source_hashes = [], [], []
    for client, dataset in enumerate(clients):
        root, indices = original_indices(dataset)
        state = torch.load(reference/"risk_synthesis"/f"client_{client}_distribution.pt",
                           map_location="cpu", weights_only=True, mmap=True)
        if (not np.array_equal(np.asarray(root.targets)[indices], state["labels"].numpy())
                or state["source_sha256"] != replay_hashes[client]):
            raise ValueError("Reconstructed source labels or prior exact-code identity certificate differ.")
        mappings.append(indices.tolist())
        roots.append(root)
        source_hashes.append(dict(client=client, source_sha256=state["source_sha256"]))
    source = roots[0]
    if (len(source) != 50000 or not source.train or len(names) != 100
            or not all(root is source for root in roots) or not source._check_integrity()):
        raise ValueError("Expected the unchanged original CIFAR100 training source.")
    labels = np.asarray(source.targets, dtype=np.int64)
    used = np.concatenate([np.asarray(m) for m in mappings])
    if not np.array_equal(np.bincount(labels[used], minlength=100), np.full(100, 100)):
        raise ValueError("The exploration identities are not exactly 100 per class.")
    partitions = reserve_indices(labels, used, seed=seed)
    fingerprint = hashlib.sha256()
    for start in range(0, len(source), 1000):
        fingerprint.update(source.data[start:start+1000].tobytes())
    counts = {key: np.bincount(labels[indices], minlength=100).tolist()
              for key, indices in partitions.items()}
    if counts["train_pool_indices"] != [300]*100 or counts["evaluation_indices"] != [100]*100:
        raise ValueError("Unexpected reserved class counts.")
    manifest = dict(schema_version=1, status="reserved_not_integrated_or_evaluated",
                    dataset="cifar100", source_partition="original_train", source_samples=50000,
                    source_image_shape=list(source.data.shape), source_image_dtype=str(source.data.dtype),
                    source_images_sha256=fingerprint.hexdigest(),
                    source_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                    partition_seed=seed, fpl_shots_after_train_pool_selection=100,
                    candidate_confirmation_seeds=[43, 44, 45],
                    **partitions, class_histograms=counts, exploratory_client_source_indices=mappings,
                    exploratory_source_code_hashes=source_hashes,
                    provenance={str(config_path): digest(config_path), str(replay_path): digest(replay_path)},
                    interpretation="Exclude all 10,000 seed-42 exploration training records. Reserve 10,000 previously unused original-train records for confirmation evaluation/nonmembers, and 30,000 other unused records as the future training sampling pool. No classifier was trained or evaluated by this script. Current experiments and the original official test split are unchanged.")
    output.mkdir(parents=True, exist_ok=False)
    (output/"split.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(dict(status=manifest["status"], sizes={k: len(v) for k, v in partitions.items()},
                          class_counts={k: sorted(set(v)) for k, v in counts.items()},
                          source_images_sha256=fingerprint.hexdigest()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--replay-certificate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    prepare(args.reference, args.replay_certificate, args.output, args.seed)
