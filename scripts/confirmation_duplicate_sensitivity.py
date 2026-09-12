#!/usr/bin/env python
"""Supplementary score-only sensitivity to exact cross-role pixel duplicates.

Retains original training, frozen primary criteria and existing artifacts.
Exclusion and label balancing use source identities, labels and roles only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import class_diagnostics, digest, recompute_auc, recompute_tpr


def retained_indices(source_ids, membership, labels, flagged):
    source_ids, membership, labels = map(np.asarray, (source_ids, membership, labels))
    if (source_ids.ndim != 1 or membership.shape != source_ids.shape or labels.shape != source_ids.shape
            or len(set(source_ids.tolist())) != len(source_ids) or set(membership.tolist()) != {0, 1}):
        raise ValueError("Expected unique original identities with aligned labels and membership.")
    keep = ~np.isin(source_ids, list(flagged))
    reasons = {i:"exact_cross_role_pixel_duplicate" for i in np.flatnonzero(~keep)}
    for label in np.unique(labels):
        groups = [np.flatnonzero(keep & (labels == label) & (membership == role)) for role in (0, 1)]
        quota = min(map(len, groups))
        for group in groups:
            # Remove surplus candidates in ascending source-id order, never by score.
            surplus = len(group)-quota
            removed = group[np.argsort(source_ids[group], kind="stable")[:surplus]]
            keep[removed] = False
            reasons.update({int(i):"restore_exact_class_ratio" for i in removed})
    if sum(keep & (membership == 0)) < 2 or sum(keep & (membership == 1)) < 2:
        raise ValueError("Insufficient candidates after exact duplicate exclusion and class matching.")
    return np.flatnonzero(keep), [dict(sample_index=int(i), source_index=int(source_ids[i]),
        membership=int(membership[i]), label=int(labels[i]), reason=reason) for i, reason in sorted(reasons.items())]


def reportable_tpr(membership, scores, fpr):
    count = int(sum(np.asarray(membership) == 0))
    return recompute_tpr(membership, scores, fpr) if count >= int(np.ceil(1/fpr)) else None


def run(verified_path, duplicate_path, output):
    import torch
    verified = json.loads(verified_path.read_text())
    duplicates = json.loads(duplicate_path.read_text())
    duplicate_seeds = {row["seed"]:row for row in duplicates["seeds"]}
    if len(duplicate_seeds) != len(duplicates["seeds"]):
        raise ValueError("Duplicate source audit repeats a seed.")
    sources = {str(path.resolve()):digest(path) for path in (verified_path, duplicate_path, Path(__file__))}
    results, rows = [], []
    for record in verified["runs"]:
        source = record.get("confirmation_source", {})
        if not record["complete"] or source.get("manifest_sha256") != duplicates["manifest_sha256"]:
            raise ValueError("Sensitivity requires complete, source-verified confirmation records.")
        seed = record["protocol"]["seed"]
        flagged = set()
        for field in ("train_vs_evaluation", "evaluation_vs_exploration"):
            for match in duplicate_seeds[seed][field]["matches"]:
                flagged.update(match["left_source_indices"])
                flagged.update(match["right_source_indices"])
        audit = Path(record["path"])/"privacy_audit"
        for name in ("signals.pt", "predictions.csv", "candidate_selection.pt",
                     "client_train_update_candidate_selection.pt"):
            path = audit/name
            if digest(path) != record["sources"][str(path)]:
                raise ValueError(f"Verified input changed: {path}")
            sources[str(path)] = digest(path)
        signals = torch.load(audit/"signals.pt", map_location="cpu", weights_only=True, mmap=True)
        membership = np.asarray(signals["membership"])
        labels = np.asarray(signals["candidate_labels"])
        ids = np.empty(len(membership), dtype=np.int64)
        ids[membership == 1] = source["member_source_indices"]
        ids[membership == 0] = source["nonmember_source_indices"]
        keep, removed = retained_indices(ids, membership, labels, flagged)
        filtered_m, filtered_y = membership[keep], labels[keep]
        candidate_hash = hashlib.sha256(np.column_stack((ids[keep], filtered_m, filtered_y)).tobytes()).hexdigest()
        scores = {}
        for row in csv.DictReader((audit/"predictions.csv").open()):
            name = row["attack"]
            values = scores.setdefault(name, {})
            i = int(row["sample_index"])
            if i in values or not 0 <= i < len(membership) or int(row["membership"]) != membership[i]:
                raise ValueError("Prediction identity mismatch or duplicate.")
            values[i] = float(row["score"])
        if set(scores) != {attack["attack"] for attack in record["attacks"]}:
            raise ValueError("Sensitivity attack set differs from verified primary results.")
        metrics = []
        for attack in record["attacks"]:
            name = attack["attack"]
            if set(scores[name]) != set(range(len(membership))):
                raise ValueError("An attack does not cover the complete original candidate pool.")
            values = np.asarray([scores[name][i] for i in keep])
            auc = recompute_auc(filtered_m, values)
            geometry = class_diagnostics(filtered_m, values, filtered_y,
                                         report_low_fpr=sum(filtered_m == 0) >= 100)
            metric = dict(attack=name, auc=auc, original_auc=attack["auc"],
                direction_symmetric_auc=max(auc, 1-auc), class_conditional_auc=geometry["class_conditional_auc"],
                tpr_at_10pct=reportable_tpr(filtered_m, values, .1),
                tpr_at_1pct=reportable_tpr(filtered_m, values, .01),
                tpr_at_01pct=reportable_tpr(filtered_m, values, .001))
            metrics.append(metric)
            rows.append(dict(run=record["run"], seed=seed, **metric))
        arm = record["synthesis_options"]["mode"] if record["defense"] == "risk_synthesis" else record["defense"]
        results.append(dict(run=record["run"], seed=seed, arm=arm, comparison_key=record["comparison_key"],
            candidate_hash=candidate_hash, members=int(sum(filtered_m == 1)), nonmembers=int(sum(filtered_m == 0)),
            fpr_resolution=1/int(sum(filtered_m == 0)), removed=removed, attacks=metrics,
            strongest_auc=max(metric["auc"] for metric in metrics),
            strongest_tpr_at_1pct=max((metric["tpr_at_1pct"] for metric in metrics
                                      if metric["tpr_at_1pct"] is not None), default=None)))
    comparisons = []
    for treatment in results:
        for control in results:
            if (treatment["comparison_key"] != control["comparison_key"] or treatment["run"] == control["run"]
                    or (treatment["arm"], control["arm"]) not in {
                        ("risk", "none"), ("risk", "shuffled_risk"), ("risk", "www"),
                        ("shuffled_risk", "none"), ("www", "none")}):
                continue
            if treatment["candidate_hash"] != control["candidate_hash"]:
                raise ValueError("Filtered paired arms retained different original candidates.")
            base = {metric["attack"]:metric for metric in control["attacks"]}
            comparisons.append(dict(run=treatment["run"], control=control["run"], seed=treatment["seed"],
                maximum_auc_delta=treatment["strongest_auc"]-control["strongest_auc"],
                maximum_tpr_at_1pct_delta=(treatment["strongest_tpr_at_1pct"]-control["strongest_tpr_at_1pct"]
                    if treatment["strongest_tpr_at_1pct"] is not None and control["strongest_tpr_at_1pct"] is not None else None),
                attack_auc_deltas={metric["attack"]:metric["auc"]-base[metric["attack"]]["auc"] for metric in treatment["attacks"]}))
    report = dict(status="supplementary_exact_duplicate_sensitivity", sources=sources, results=results,
        comparisons=comparisons, rule="Exclude original identities belonging to exact pixel groups crossing "
            "train/evaluation or evaluation/exploration roles. Restore each class to a 1:1 candidate ratio by "
            "removing surplus candidates in ascending source-id order. Same identities for every paired model and attack.",
        interpretation="Supplementary analysis specified after the first baseline, before paired confirmation outcomes. "
            "Does not replace frozen primary criteria or alter training. Source-record roles remain disjoint; exact "
            "pixel overlap is a distinct limitation. Near duplicates and pretraining exposure are not evaluated. "
            "Removing candidates can reduce FPR resolution; unavailable values stay null. No additional classifier scoring.")
    output.mkdir(parents=True, exist_ok=False)
    (output/"sensitivity_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    with (output/"attack_metrics.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps([dict(run=r["run"], members=r["members"], nonmembers=r["nonmembers"],
                           removed=r["removed"], strongest_auc=r["strongest_auc"],
                           strongest_tpr_at_1pct=r["strongest_tpr_at_1pct"]) for r in results], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", required=True, type=Path)
    parser.add_argument("--duplicate-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.verified, args.duplicate_report, args.output)
