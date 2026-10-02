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


@pytest.mark.parametrize('method', ['fedavg', 'fedsgd'])
def test_confirmation_loader_caps_after_reservation_and_saves_original_identities(reservation, monkeypatch, tmp_path, method):
    import utils.data_loader as loader
    from main import _dataset_split_arguments
    source, path, fingerprint, manifest, config = reservation
    config['aggregator'] = method
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
    if method == 'fedsgd':
        batch = dict(communication_round=1, member_local_indices=torch.tensor([1, 3, 5]),
                     nonmember_pool_indices=torch.tensor([2, 4, 6]))
        torch.save({'rounds': [batch]}, audit / 'exact_batch_candidate_selection.pt')
    else:
        torch.save({"rounds": [saved_positions]}, audit / "client_train_update_candidate_selection.pt")
    signals = dict(membership=np.repeat([1, 0], 1000),
                   candidate_labels=np.concatenate([mapped["member_labels"], mapped["nonmember_labels"]]))
    if method == 'fedsgd':
        signals['exact_batch_observations'] = [dict(round=0, membership=np.repeat([1, 0], 3),
            member_local_indices=batch['member_local_indices'],
            nonmember_pool_indices=batch['nonmember_pool_indices'],
            candidate_labels=np.concatenate([np.asarray(first['clients'][0]['train_labels'])[[1, 3, 5]],
                                             pools['evaluation'][1][[2, 4, 6]]]))]
    config["audit"] = {"audit_client_ids": [0]}
    verified = verify_confirmation_sources(tmp_path, config, signals, {})
    assert verified["roles_disjoint"] and verified["exploration_records_excluded"]
    if method == 'fedsgd':
        assert verified['original_source_batches']['1']['member_source_indices'] == [
            first['clients'][0]['train_source_indices'][i] for i in (1, 3, 5)]
        bad = copy.deepcopy(signals)
        bad['exact_batch_observations'][0]['candidate_labels'][0] += 1
        with pytest.raises(ValueError, match='audit labels'):
            verify_confirmation_sources(tmp_path, config, bad, {})
        bad = copy.deepcopy(signals)
        bad['exact_batch_observations'][0]['member_local_indices'][0] = 7
        with pytest.raises(ValueError, match='original identities'):
            verify_confirmation_sources(tmp_path, config, bad, {})
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


def test_actual_auditor_preserves_disjoint_shared_source_identities(reservation, monkeypatch, tmp_path):
    """Exercise both candidate paths through actual nested subsets, without scoring."""
    from types import SimpleNamespace
    import torch
    import utils.data_loader as loader
    from main import _dataset_split_arguments
    from privacy_attacks.auditor import MembershipAuditor

    *_, manifest, config = reservation
    monkeypatch.setattr(loader.datasets, "CIFAR100", Source)
    validate_confirmation_config(config)
    random.seed(43)
    train, evaluation, _ = loader.generate_iid_split(
        "cifar100", 10, **_dataset_split_arguments(config))
    mapping = write_confirmation_provenance(train, evaluation, config, tmp_path)
    users = [SimpleNamespace(id=i, train_data=a, test_data=b)
             for i, (a, b) in enumerate(zip(train, evaluation))]
    model = torch.nn.Linear(1, 100)
    model.model_type = "clip_adapter"
    # Candidate construction must not infer membership from root identity or
    # call a model on any confirmation record.
    model.forward = lambda *args, **kwargs: pytest.fail("Unexpected model scoring")
    auditor = MembershipAuditor(
        model=model, users=users, target_client_id=0, device=torch.device("cpu"),
        results_dir=str(tmp_path), num_classes=100, federated_method="fedavg",
        config=dict(enabled=True, attacks=["blackbox_loss", "loss_series"],
                    candidate_sampling="balanced_global_holdout",
                    require_full_target_train_members=True,
                    low_fpr_max_members=0, low_fpr_max_nonmembers=0,
                    low_fpr_min_nonmembers=1000, nonmember_to_member_ratio=1,
                    client_train_membership_attacks=["blackbox_loss"],
                    paper_balanced_evaluation_size=100, seed=43))
    selection = auditor.low_fpr_candidate_selection
    mapped = map_confirmation_candidates(mapping, manifest, 0, selection)
    expected_inputs = np.concatenate([mapped["member_source_indices"], mapped["nonmember_source_indices"]])
    np.testing.assert_array_equal(auditor.images.numpy().reshape(-1), expected_inputs)
    np.testing.assert_array_equal(auditor.labels.numpy(), np.concatenate(
        [mapped["member_labels"], mapped["nonmember_labels"]]))
    assert not set(mapped["member_source_indices"]) & set(mapped["nonmember_source_indices"])
    for round_index in (49, 99):
        update = auditor._build_exact_batch_candidates(round_index)
        assert update["selection"]["membership_definition"] == "target_client_original_training_set"
        assert update["selection"]["member_count"] == update["selection"]["nonmember_count"] == 1000
        torch.testing.assert_close(update["inputs"], auditor.images, rtol=0, atol=0)
        actual = map_confirmation_candidates(mapping, manifest, 0, update["selection"])
        for key in mapped:
            np.testing.assert_array_equal(actual[key], mapped[key])
