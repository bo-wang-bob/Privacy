"""Opt-in CIFAR100 confirmation partitions with original-record provenance.

The manifest reserves records before any confirmation model is trained. Both
roles come from the original training source; membership is determined by the
disjoint record identities, not torchvision's ``train`` flag.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

PARTITIONS = ("train_pool_indices", "evaluation_indices", "excluded_exploration_indices")


def array_digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    for start in range(0, len(array), 1000):
        digest.update(array[start:start + 1000].tobytes())
    return digest.hexdigest()


def checked_indices(values, size, name):
    if not isinstance(values, list) or not values or any(type(i) is not int for i in values):
        raise ValueError(f"{name} must contain integer source identities.")
    indices = np.asarray(values, dtype=np.int64)
    if indices.min() < 0 or indices.max() >= size or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} source identities must be unique and in range.")
    return indices


def validate_partitions(manifest, labels=None):
    size = manifest["source_samples"]
    partitions = {key: checked_indices(manifest[key], size, key) for key in PARTITIONS}
    combined = np.concatenate(list(partitions.values()))
    if len(combined) != size or len(np.unique(combined)) != size:
        raise ValueError("Confirmation partitions must be disjoint and cover the original source.")
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)
        if len(labels) != size or labels.min() < 0:
            raise ValueError("Source labels are not aligned with the confirmation manifest.")
        for key, indices in partitions.items():
            counts = np.bincount(labels[indices], minlength=100).tolist()
            if counts != manifest["class_histograms"][key]:
                raise ValueError(f"Confirmation class histogram differs: {key}.")
    return partitions


def read_manifest(path, expected_sha256=None):
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError("Confirmation manifest changed after configuration was resolved.")
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != 1 or manifest.get("dataset") != "cifar100"
            or manifest.get("source_partition") != "original_train"
            or manifest.get("source_samples") != 50000
            or manifest.get("fpl_shots_after_train_pool_selection") != 100):
        raise ValueError("Expected the reserved CIFAR100 100-per-class confirmation manifest.")
    validate_partitions(manifest)
    expected = dict(zip(PARTITIONS, (300, 100, 100)))
    if any(manifest["class_histograms"][key] != [count] * 100 for key, count in expected.items()):
        raise ValueError("Confirmation requires 300/100/100 source records per class.")
    return manifest, sha256


def validate_confirmation_config(config):
    path = config.get("confirmation_split_manifest")
    if path is None:
        if config.get("confirmation_split_sha256") is not None:
            raise ValueError("confirmation_split_sha256 requires confirmation_split_manifest.")
        return
    model = str(config.get("model_type", "")).lower()
    defense = config.get("defense", {})
    if (model not in {"clip_adapter", "clip_lora"}
            or (model == "clip_adapter" and config.get("clip_adapter", {}).get("variant") != "transformer")
            or str(config.get("dataset_name", "")).lower() != "cifar100"
            or config.get("partition_mode") != "iid" or config.get("aggregator") != "fedavg"
            or config.get("total_users") != 10 or config.get("fpl_shots") != 100
            or bool(config.get("use_full_dataset", False))
            or defense.get("name", "none") not in {"none", "www", "risk_synthesis"}
            or float(defense.get("cofedmid_validation_fraction", 0) or 0) != 0):
        raise ValueError("confirmation_split_manifest requires CLIP transformer Adapter/LoRA, "
                         "CIFAR100, IID, FedAvg, 10 clients, 100-per-class training, "
                         "and none/www/risk_synthesis without an additional validation reservation.")
    path = Path(path).resolve()
    manifest, sha256 = read_manifest(path, config.get("confirmation_split_sha256"))
    if config.get("seed") not in manifest["candidate_confirmation_seeds"]:
        raise ValueError("Choose a seed from the confirmation manifest's reserved seed list.")
    config["confirmation_split_manifest"] = str(path)
    config["confirmation_split_sha256"] = sha256


def load_confirmation_pools(source, path, expected_sha256):
    manifest, _ = read_manifest(path, expected_sha256)
    if (isinstance(source, Subset) or len(source) != manifest["source_samples"]
            or getattr(source, "train", None) is not True):
        raise ValueError("Confirmation must be selected from the complete original training source.")
    data = np.asarray(source.data)
    labels = np.asarray(source.targets, dtype=np.int64)
    if (list(data.shape) != manifest["source_image_shape"]
            or str(data.dtype) != manifest["source_image_dtype"]
            or array_digest(data) != manifest["source_images_sha256"]
            or array_digest(labels) != manifest["source_labels_sha256"]):
        raise ValueError("Confirmation source image or label fingerprint differs.")
    indices = validate_partitions(manifest, labels)
    return (Subset(source, indices["train_pool_indices"].tolist()),
            Subset(source, indices["evaluation_indices"].tolist()))


def source_indices(dataset):
    indices = np.arange(len(dataset))
    while isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices, dtype=np.int64)[indices]
        dataset = dataset.dataset
    return dataset, indices.astype(np.int64)


def validate_mapping(mapping, manifest):
    partitions = validate_partitions(manifest)
    clients = mapping["clients"]
    if [client["client_id"] for client in clients] != list(range(10)):
        raise ValueError("Confirmation provenance requires ten clients in client-id order.")
    output = {}
    for role, pool in (("train", "train_pool_indices"), ("evaluation", "evaluation_indices")):
        indices, labels = [], []
        for client in clients:
            ids = checked_indices(client[f"{role}_source_indices"], manifest["source_samples"], role)
            ys = np.asarray(client[f"{role}_labels"], dtype=np.int64)
            if (len(ids) != 1000 or len(ys) != len(ids) or ys.min() < 0
                    or not np.array_equal(np.bincount(ys, minlength=100), np.full(100, 10))):
                raise ValueError("Confirmation must retain ten original records per client/class/role.")
            indices.extend(ids.tolist())
            labels.extend(ys.tolist())
        if (len(set(indices)) != len(indices)
                or not set(indices).issubset(set(partitions[pool].tolist()))):
            raise ValueError("Confirmation client records overlap or use an unreserved source role.")
        if role == "evaluation" and set(indices) != set(partitions[pool].tolist()):
            raise ValueError("Confirmation evaluation must use its entire reserved partition.")
        output[role] = (np.asarray(indices), np.asarray(labels))
    return output


def map_confirmation_candidates(mapping, manifest, client_id, selection):
    pools = validate_mapping(mapping, manifest)
    client = mapping["clients"][client_id]
    member = checked_indices(np.asarray(selection["member_pool_indices"]).tolist(), 1000, "member")
    nonmember = checked_indices(np.asarray(selection["nonmember_pool_indices"]).tolist(),
                                len(pools["evaluation"][0]), "nonmember")
    if len(member) != 1000:
        raise ValueError("Confirmation audit must include the full original target training set.")
    return dict(member_source_indices=np.asarray(client["train_source_indices"])[member],
                nonmember_source_indices=pools["evaluation"][0][nonmember],
                member_labels=np.asarray(client["train_labels"])[member],
                nonmember_labels=pools["evaluation"][1][nonmember])


def write_confirmation_provenance(train_sets, test_sets, config, result_dir):
    path, expected = config["confirmation_split_manifest"], config["confirmation_split_sha256"]
    manifest, actual = read_manifest(path, expected)
    if len(train_sets) != 10 or len(test_sets) != 10:
        raise ValueError("Confirmation provenance requires ten clients.")
    roots, clients = [], []
    for client_id, (train, evaluation) in enumerate(zip(train_sets, test_sets)):
        row = dict(client_id=client_id)
        for role, dataset in (("train", train), ("evaluation", evaluation)):
            root, ids = source_indices(dataset)
            roots.append(root)
            row[f"{role}_source_indices"] = ids.tolist()
            row[f"{role}_labels"] = np.asarray(root.targets, dtype=np.int64)[ids].tolist()
        clients.append(row)
    if not all(root is roots[0] for root in roots):
        raise ValueError("Confirmation roles must map to one verified original source.")
    mapping = dict(schema_version=1, dataset="cifar100", source_partition="original_train",
                   manifest_sha256=actual, seed=config["seed"], clients=clients,
                   index_semantics="Member pool indices address the target client's train list. "
                   "Nonmember pool indices address the client-id ordered concatenation of evaluation lists.",
                   source_images_sha256=manifest["source_images_sha256"],
                   source_labels_sha256=manifest["source_labels_sha256"])
    validate_mapping(mapping, manifest)
    result_dir = Path(result_dir)
    # Fresh task artifacts only; existing results must not be replaced.
    with (result_dir / "confirmation_split.json").open("xb") as handle:
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Confirmation manifest changed during provenance capture.")
        handle.write(raw)
    with (result_dir / "data_partition.json").open("x") as handle:
        json.dump(mapping, handle, indent=2, allow_nan=False)
    return mapping
