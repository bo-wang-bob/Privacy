#!/usr/bin/env python
"""Paired, class/membership-stratified candidate resampling for a verified pair.

These intervals condition on two already trained models. They do not cover
training randomness, hyperparameter selection, or population generalization.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def score_metrics(scores, membership, fpr=.01):
    """Return average-rank AUC and attainable threshold TPR for each score row."""
    scores = np.asarray(scores, dtype=float)
    membership = np.asarray(membership, dtype=int)
    if (scores.ndim < 2 or scores.shape[-1] != len(membership)
            or set(membership.tolist()) != {0, 1} or not np.isfinite(scores).all()):
        raise ValueError("Expected finite, aligned scores and binary membership.")
    auc, tpr = [], []
    for row in scores.reshape(-1, scores.shape[-1]):
        positive = row[membership == 1]
        negative = np.sort(row[membership == 0])
        lower = np.searchsorted(negative, positive, side="left")
        upper = np.searchsorted(negative, positive, side="right")
        auc.append(float((lower+upper).mean()/(2*len(negative))))
        threshold = np.nextafter(negative[-1-int(np.floor(fpr*len(negative)))], np.inf)
        tpr.append(float((positive >= threshold).mean()))
    shape = scores.shape[:-1]
    return np.array(auc).reshape(shape), np.array(tpr).reshape(shape)


def paired_resampling(scores, membership, classes, *, replicates=2000, seed=20260912):
    """Use the same resampled identities for both models and every attack."""
    scores = np.asarray(scores, dtype=float)
    membership, classes = np.asarray(membership, dtype=int), np.asarray(classes)
    if scores.ndim != 3 or scores.shape[0] != 2 or len(classes) != len(membership):
        raise ValueError("Expected two models, a shared attack axis, and aligned classes.")
    if replicates < 2:
        raise ValueError("At least two resampling replicates are required.")
    score_metrics(scores, membership)
    strata = [np.flatnonzero((membership == m) & (classes == c))
              for m in (0, 1) for c in np.unique(classes)]
    strata = [group for group in strata if len(group)]
    rng = np.random.default_rng(seed)
    aucs = np.empty((replicates, 2, scores.shape[1]))
    tprs = np.empty_like(aucs)
    for repeat in range(replicates):
        indices = np.arange(len(membership))
        for group in strata:
            indices[group] = rng.choice(group, size=len(group), replace=True)
        aucs[repeat], tprs[repeat] = score_metrics(scores[:, :, indices], membership)
    return aucs, tprs


def select_pair(verified, treatment=None, control=None):
    if (treatment is None) != (control is None):
        raise ValueError("Specify both treatment and control, or neither.")
    if treatment is None:
        if len(verified["matched_comparisons"]) != 1:
            raise ValueError("Provide one verified pair or explicitly name its two runs.")
        pair = verified["matched_comparisons"][0]
        treatment, control = pair["run"], pair["control"]
    if treatment == control:
        raise ValueError("Choose two distinct runs.")
    records = {r["run"]: r for r in verified["runs"]}
    if treatment not in records or control not in records:
        raise ValueError("Requested runs are not in the verified report.")
    a, b = records[control], records[treatment]
    if (not a["complete"] or not b["complete"]
            or a["comparison_key"] != b["comparison_key"]
            or a["candidate_metadata"] != b["candidate_metadata"]
            or a["candidate_selection_digests"] != b["candidate_selection_digests"]):
        raise ValueError("The selected pair is incomplete or has mismatched protocol/candidates.")
    return a, b


def run(verified_path, output, replicates, seed, treatment=None, control_name=None):
    import torch
    # Support direct execution as well as imports by the repository tests.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_risk_synthesis import digest

    verified = json.loads(verified_path.read_text())
    control, defense = select_pair(verified, treatment, control_name)
    comparison = dict(run=defense["run"], control=control["run"],
                      accuracy_delta=defense["accuracy"]-control["accuracy"])
    names = [a["attack"] for a in control["attacks"]]
    if set(names) != {a["attack"] for a in defense["attacks"]}:
        raise ValueError("Attack sets differ.")
    all_scores, common_membership, common_classes = [], None, None
    sources = {str(verified_path): digest(verified_path)}
    for record in (control, defense):
        audit = Path(record["path"])/"privacy_audit"
        for name in ("predictions.csv", "signals.pt", "candidate_selection.pt",
                     "client_train_update_candidate_selection.pt"):
            path = audit/name
            actual = digest(path)
            if actual != record["sources"][str(path)]:
                raise ValueError(f"Verified input changed: {path}")
            sources[str(path)] = actual
        signals = torch.load(audit/"signals.pt", map_location="cpu", weights_only=True, mmap=True)
        selection = torch.load(audit/"candidate_selection.pt", map_location="cpu", weights_only=True)
        update = torch.load(audit/"client_train_update_candidate_selection.pt", map_location="cpu", weights_only=True)
        for entry in update["rounds"]:
            for key in ("member_pool_indices", "nonmember_pool_indices"):
                if not torch.equal(selection[key], entry[key]):
                    raise ValueError("Joint resampling requires shared identities across all attacks.")
        by_attack = defaultdict(list)
        with (audit/"predictions.csv").open() as handle:
            for row in csv.DictReader(handle):
                by_attack[row["attack"]].append(row)
        rows_of_scores = []
        for name in names:
            rows = sorted(by_attack[name], key=lambda r: int(r["sample_index"]))
            if [int(r["sample_index"]) for r in rows] != list(range(len(rows))):
                raise ValueError("Unexpected candidate indices or multiple target clients.")
            membership = np.array([int(r["membership"]) for r in rows])
            label_source = next((s for s in reversed(signals.get("client_train_update_observations", []))
                                 if name in s.get("attacks", [])), signals)
            classes = np.asarray(label_source["candidate_labels"])
            if not np.array_equal(membership, np.asarray(label_source["membership"])):
                raise ValueError("Saved prediction membership is misaligned.")
            if common_membership is None:
                common_membership, common_classes = membership, classes
            elif (not np.array_equal(common_membership, membership)
                    or not np.array_equal(common_classes, classes)):
                raise ValueError("Candidate ordering/classes differ between attacks or models.")
            rows_of_scores.append([float(r["score"]) for r in rows])
        all_scores.append(rows_of_scores)
    scores = np.asarray(all_scores)
    point_auc, point_tpr = score_metrics(scores, common_membership)
    for index, record in enumerate((control, defense)):
        official = {a["attack"]: a for a in record["attacks"]}
        for j, name in enumerate(names):
            if (not np.isclose(point_auc[index, j], official[name]["auc"], atol=1e-12, rtol=0)
                    or official[name]["tpr_at_1pct"] is None
                    or not np.isclose(point_tpr[index, j], official[name]["tpr_at_1pct"], atol=1e-12, rtol=0)):
                raise ValueError("Point metrics do not reproduce the verified report.")
    aucs, tprs = paired_resampling(scores, common_membership, common_classes,
                                   replicates=replicates, seed=seed)
    rows = []

    def interval(name, metric, point, draws):
        low, high = np.quantile(draws, [.025, .975])
        rows.append(dict(attack=name, metric=metric, defense_minus_control=float(point),
                         candidate_resampling_025=float(low), candidate_resampling_975=float(high)))

    for j, name in enumerate(names):
        interval(name, "auc", point_auc[1, j]-point_auc[0, j], aucs[:, 1, j]-aucs[:, 0, j])
        interval(name, "tpr_at_1pct", point_tpr[1, j]-point_tpr[0, j], tprs[:, 1, j]-tprs[:, 0, j])
    interval("maximum_across_registered_attacks", "auc", point_auc[1].max()-point_auc[0].max(),
             aucs[:, 1].max(axis=1)-aucs[:, 0].max(axis=1))
    interval("maximum_across_registered_attacks", "tpr_at_1pct", point_tpr[1].max()-point_tpr[0].max(),
             tprs[:, 1].max(axis=1)-tprs[:, 0].max(axis=1))
    output.mkdir(parents=True, exist_ok=False)
    with (output/"paired_candidate_intervals.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(comparison=comparison, replicates=replicates, resampling_seed=seed,
                  stratification="within class and membership; same original identities for both runs/all attacks",
                  intervals="pointwise empirical 2.5%/97.5% bootstrap quantiles",
                  limitations="Conditions on fixed trained models and class counts. Does not account for training-seed variation, selecting this configuration, or multiple per-attack comparisons. Not confirmation or a privacy guarantee.",
                  sources=sources, results=rows)
    (output/"resampling_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps([r for r in rows if r["attack"] in ("projres", "maximum_across_registered_attacks")], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verified", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--treatment", help="Exact run name in the verified report")
    parser.add_argument("--control", help="Exact comparator run name in the verified report")
    args = parser.parse_args()
    run(args.verified, args.output, args.replicates, args.seed, args.treatment, args.control)
