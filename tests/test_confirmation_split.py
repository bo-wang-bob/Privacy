import copy
import json
import random

import numpy as np
import pytest
from torch.utils.data import Dataset

from scripts.prepare_synthesis_confirmation import reserve_indices
from utils.confirmation_split import (
    array_digest, load_confirmation_pools, map_confirmation_candidates, read_manifest, source_indices,
    validate_confirmation_config, validate_mapping, write_confirmation_provenance,
)


class Source(Dataset):
    """Small pixel payload, but the real 500/100 source class cardinalities."""
    def __init__(self, root=None, train=True, **kwargs):
        self.train = train
        self.targets = np.repeat(np.arange(100), 500 if train else 100).tolist()
        self.data = np.arange(len(self.targets), dtype=np.int32).reshape(-1, 1)
        self.classes = [str(i) for i in range(100)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        assert self.train, "The official test source must not be used by confirmation."
        return int(self.data[index, 0]), self.targets[index]


@pytest.fixture
def reservation(tmp_path):
    source = Source()
    excluded = [500 * label + i for label in range(100) for i in range(100)]
    partitions = reserve_indices(source.targets, excluded)
    labels = np.asarray(source.targets, dtype=np.int64)
    manifest = dict(schema_version=1, dataset="cifar100", source_partition="original_train",
                    source_samples=len(source), source_image_shape=list(source.data.shape),
                    source_image_dtype=str(source.data.dtype),
                    source_images_sha256=array_digest(source.data),
                    source_labels_sha256=array_digest(labels),
                    fpl_shots_after_train_pool_selection=100,
                    candidate_confirmation_seeds=[43, 44, 45], **partitions,
                    class_histograms={key: np.bincount(labels[ids], minlength=100).tolist()
                                      for key, ids in partitions.items()})
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    _, fingerprint = read_manifest(path)
    config = dict(model_type="clip_adapter", clip_adapter={"variant": "transformer"},
                  dataset_name="cifar100", partition_mode="iid", aggregator="fedavg",
                  total_users=10, fpl_shots=100, use_full_dataset=False, seed=43,
                  confirmation_split_manifest=str(path), defense={"name": "none"})
    return source, path, fingerprint, manifest, config


def test_confirmation_loader_caps_after_reservation_and_saves_original_identities(reservation, monkeypatch, tmp_path):
    import utils.data_loader as loader
    from main import _dataset_split_arguments
    source, path, fingerprint, manifest, config = reservation
    monkeypatch.setattr(loader.datasets, "CIFAR100", Source)
    validate_confirmation_config(config)
    assert config["confirmation_split_sha256"] == fingerprint

    def split(seed):
        random.seed(seed)
        return loader.generate_iid_split("cifar100", 10, **_dataset_split_arguments(config))

    train, evaluation, names = split(43)
    again_train, again_eval, _ = split(43)
    assert len(names) == 100
    first = write_confirmation_provenance(train, evaluation, config, tmp_path)
    pools = validate_mapping(first, manifest)
    for left, right in zip(train + evaluation, again_train + again_eval):
        np.testing.assert_array_equal(source_indices(left)[1], source_indices(right)[1])
    assert len(pools["train"][0]) == len(pools["evaluation"][0]) == 10000
    assert not set(pools["train"][0]) & set(pools["evaluation"][0])
    assert not set(pools["train"][0]) & set(manifest["excluded_exploration_indices"])
    assert not set(pools["evaluation"][0]) & set(manifest["excluded_exploration_indices"])
    # Values returned by the actual nested Subsets recover exactly these IDs.
    assert [sample for client in train for sample, _ in client] == pools["train"][0].tolist()
    assert [sample for client in evaluation for sample, _ in client] == pools["evaluation"][0].tolist()
    other_train, _, _ = split(44)
    assert set(np.concatenate([source_indices(c)[1] for c in other_train])) != set(pools["train"][0])
    assert (tmp_path / "confirmation_split.json").read_bytes() == path.read_bytes()
    assert json.loads((tmp_path / "data_partition.json").read_text()) == first
    candidate_positions = dict(member_pool_indices=np.arange(999, -1, -1),
                               nonmember_pool_indices=np.arange(0, 10000, 10))
    mapped = map_confirmation_candidates(first, manifest, 0, candidate_positions)
    np.testing.assert_array_equal(mapped["member_source_indices"], first["clients"][0]["train_source_indices"][::-1])
    np.testing.assert_array_equal(mapped["nonmember_source_indices"], pools["evaluation"][0][::10])
    assert not set(mapped["member_source_indices"]) & set(mapped["nonmember_source_indices"])
    with pytest.raises(ValueError, match="full original"):
        map_confirmation_candidates(first, manifest, 0, {**candidate_positions, "member_pool_indices": [0]})
    # Exercise saved-audit verification, including indices into the global
    # evaluation concatenation and labels after a nontrivial member reordering.
    import torch
    from scripts.analyze_risk_synthesis import verify_confirmation_sources
    audit = tmp_path / "privacy_audit"
    audit.mkdir()
    saved_positions = {key: torch.as_tensor(value) for key, value in candidate_positions.items()}
    torch.save(saved_positions, audit / "candidate_selection.pt")
    torch.save({"rounds": [saved_positions]}, audit / "client_train_update_candidate_selection.pt")
    signals = dict(membership=np.repeat([1, 0], 1000),
                   candidate_labels=np.concatenate([mapped["member_labels"], mapped["nonmember_labels"]]))
    config["audit"] = {"audit_client_ids": [0]}
    verified = verify_confirmation_sources(tmp_path, config, signals, {})
    assert verified["roles_disjoint"] and verified["exploration_records_excluded"]
    signals["candidate_labels"][0] = (signals["candidate_labels"][0] + 1) % 100
    with pytest.raises(ValueError, match="audit labels"):
        verify_confirmation_sources(tmp_path, config, signals, {})
    with pytest.raises(FileExistsError):
        write_confirmation_provenance(train, evaluation, config, tmp_path)

    corrupted = copy.deepcopy(first)
    corrupted["clients"][0]["evaluation_source_indices"][0] = first["clients"][0]["train_source_indices"][0]
    with pytest.raises(ValueError, match="overlap|unreserved"):
        validate_mapping(corrupted, manifest)


@pytest.mark.parametrize("which", ["data", "targets"])
def test_confirmation_rejects_changed_source_records(reservation, which):
    source, path, fingerprint, _, _ = reservation
    if which == "data":
        source.data[0, 0] += 1
    else:
        source.targets[0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        load_confirmation_pools(source, path, fingerprint)


@pytest.mark.parametrize("change, message", [
    ({"seed": 42}, "reserved seed"),
    ({"partition_mode": "dirichlet"}, "requires"),
    ({"model_type": "clip_mlp"}, "requires"),
    ({"fpl_shots": 16}, "requires"),
    ({"defense": {"name": "none", "cofedmid_validation_fraction": .1}}, "requires"),
])
def test_confirmation_rejects_incompatible_training_protocol(reservation, change, message):
    *_, config = reservation
    config.update(change)
    with pytest.raises(ValueError, match=message):
        validate_confirmation_config(config)


@pytest.mark.parametrize("change, message", [("overlap", "disjoint"), ("float", "integer"), ("histogram", "300/100/100")])
def test_confirmation_rejects_modified_or_invalid_manifest(reservation, change, message):
    _, path, fingerprint, manifest, _ = reservation
    if change == "overlap":
        manifest["evaluation_indices"][0] = manifest["train_pool_indices"][0]
    elif change == "float":
        manifest["train_pool_indices"][0] = float(manifest["train_pool_indices"][0])
    else:
        manifest["class_histograms"]["evaluation_indices"][0] -= 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed"):
        read_manifest(path, fingerprint)
    with pytest.raises(ValueError, match=message):
        read_manifest(path)


def test_text_entry_rejects_confirmation_flag_instead_of_silently_ignoring_it():
    from scripts.run_fedllm_adapter import validate_config
    with pytest.raises(ValueError, match="CLIP vision"):
        validate_config({"confirmation_split_manifest": "unused.json"})
