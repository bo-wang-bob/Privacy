import numpy as np
import pytest

from scripts.confirmation_duplicate_sensitivity import retained_indices, reportable_tpr


def test_cross_role_exclusion_restores_classes_by_identity_without_scores():
    ids = np.array([60, 20, 40, 10, 90, 30, 50, 70, 80, 100, 110, 120])
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    membership = np.array([1]*6 + [0]*6)
    # One member in class0 and one nonmember in class1 are exact duplicates.
    keep, removed = retained_indices(ids, membership, labels, {40, 110})
    assert {row["source_index"] for row in removed if row["reason"] == "exact_cross_role_pixel_duplicate"} == {40, 110}
    # Restore balance by removing the smallest remaining source identity in
    # the larger opposing group, regardless of current array order.
    assert {row["source_index"] for row in removed if row["reason"] == "restore_exact_class_ratio"} == {50, 10}
    for role in (0, 1):
        assert np.bincount(labels[keep][membership[keep] == role]).tolist() == [2, 2]
    permutation = np.array([11, 9, 5, 3, 8, 0, 7, 4, 1, 6, 2, 10])
    again, _ = retained_indices(ids[permutation], membership[permutation], labels[permutation], {40, 110})
    assert set(ids[permutation][again]) == set(ids[keep])


def test_no_duplicate_keeps_full_balanced_candidate_pool():
    ids = np.arange(8)
    membership = np.repeat([1, 0], 4)
    labels = np.tile([0, 0, 1, 1], 2)
    keep, removed = retained_indices(ids, membership, labels, set())
    np.testing.assert_array_equal(keep, ids)
    assert not removed


def test_loss_of_one_nonmember_makes_point_one_percent_unreportable():
    membership = np.repeat([1, 0], 1000)
    scores = np.arange(2000, dtype=float)
    assert reportable_tpr(membership, scores, .001) == 0
    assert reportable_tpr(membership[:-1], scores[:-1], .001) is None
    assert reportable_tpr(membership[:-1], scores[:-1], .01) == 0


def test_duplicate_original_identities_and_empty_group_are_rejected():
    with pytest.raises(ValueError, match="unique original"):
        retained_indices([1, 1, 2, 3], [1, 1, 0, 0], [0, 0, 0, 0], set())
    with pytest.raises(ValueError, match="Insufficient"):
        retained_indices([1, 2, 3, 4], [1, 1, 0, 0], [0, 0, 0, 0], {1, 2})
