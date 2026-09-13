"""Independently replay v5 local history, assignments and candidate decisions."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def batch_key(row):
    return tuple(int(row[k]) for k in ("round", "client", "step"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    directory = Path(directory)
    summary_path = directory / "synthesis_summary.json"
    summary = json.loads(summary_path.read_text())
    options = summary["options"]
    class_only = summary.get("geometry_source") == "local_class_only"
    if class_only:
        require(not ({"pooled_rank", "shrinkage"} & set(options)), "Class-only options retain removed pooled parameters.")
    multiview = options.get("views_per_record", 1) > 1
    multiview_evidence = None
    if multiview:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.verify_synthesis_multiview import verify as verify_views
        multiview_evidence = verify_views(directory)
    history_enabled = options.get("risk_history", "none") == "zero_risk_frequency"
    choices_enabled = not multiview and options.get("candidate_selection", "first_semantic") == "least_local_similarity"
    require(history_enabled or choices_enabled or multiview or class_only, "No v5 history or selection to verify.")
    require(summary["status"] == "completed", "Cannot verify incomplete synthesis.")
    sources = [summary_path, directory / "synthetic_exposure.csv"]
    sizes = {}
    for path in directory.glob("client_*_distribution.pt"):
        client = int(path.stem.split("_")[1])
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if class_only:
            require(state.get("geometry_source") == "local_class_only"
                    and not ({"pooled_factor", "pooled_metadata"} & set(state)), "Unexpected pooled geometry in class-only run.")
        sizes[client] = len(state["labels"])
        if choices_enabled:
            z = state["semantic_source_features"]
            require(len(z) == sizes[client] and torch.isfinite(z).all(), "Missing original teacher features.")
        sources.append(path)
    histories, generators = {}, {}
    counters = dict(batches_verified=0, history_visits_verified=0, selected_visits_verified=0,
                    candidates_verified=0)
    choice_handle = None
    if choices_enabled:
        path = directory / "candidate_choices.csv"
        sources.append(path)
        choice_handle = path.open()
        choice_groups = itertools.groupby(csv.DictReader(choice_handle), batch_key)
    try:
        with (directory / "synthetic_exposure.csv").open() as handle:
            for key, group in itertools.groupby(csv.DictReader(handle), batch_key):
                records = list(group)
                round_number, client, _ = key
                n = len(records)
                ids = np.array([int(row["sample_id"]) for row in records])
                require(client in sizes and len(np.unique(ids)) == n and
                        np.all((ids >= 0) & (ids < sizes[client])), "Invalid original IDs within batch.")
                used = np.array([float(row["used_risk"]) for row in records])
                raw = np.array([float(row["risk"]) for row in records])
                require(np.all(np.isfinite(used)) and np.all((used >= 0) & (used <= 1)), "Invalid used risk.")
                require(all(row["requested"] == row["accepted"] == "1" and float(row["original_distance"]) > 0
                            for row in records), "Every original position must use changed virtual input.")
                if history_enabled:
                    if client not in histories:
                        histories[client] = dict(round=round_number, total=np.zeros(sizes[client]),
                            rounds=np.zeros(sizes[client], dtype=np.int64),
                            zeros=np.zeros(sizes[client], dtype=np.int64),
                            visits=np.zeros(sizes[client], dtype=np.int64))
                    h = histories[client]
                    require(round_number >= h["round"], "History rounds moved backwards.")
                    if round_number > h["round"]:
                        seen = h["visits"] > 0
                        h["total"][seen] += h["zeros"][seen] / h["visits"][seen]
                        h["rounds"][seen] += 1
                        h["zeros"].fill(0)
                        h["visits"].fill(0)
                        h["round"] = round_number
                    values = h["total"][ids] / np.maximum(h["rounds"][ids], 1)
                    require(np.allclose(values, [float(r["history_exposure"]) for r in records], rtol=0, atol=1e-12)
                            and np.array_equal(h["rounds"][ids], [int(r["history_rounds"]) for r in records]),
                            "Logged history disagrees with past successful visits.")
                    available = [int(row["source_round"]) >= 0 for row in records]
                    require(len(set(available)) == 1, "Mixed reference availability in batch.")
                    if available[0]:
                        loss = np.array([float(row["loss_gap"]) for row in records])
                        require(np.isfinite(loss).all(), "Nonfinite original loss gaps.")
                        rank_sum = (rankdata(loss, method="average") + rankdata(values, method="average") - 1) / n
                        require(np.allclose(rank_sum, [float(r["joint_rank_score"]) for r in records], rtol=0, atol=1e-12),
                                "Joint rank score disagrees with loss and history.")
                        width = math.ceil(.8 * n)
                        weights = np.zeros(n)
                        weights[-width:] = (np.arange(width) + .5) / width
                        base = np.zeros(n)
                        base[np.argsort(loss, kind="stable")] = weights
                        require(np.allclose(raw, base, rtol=0, atol=3e-8), "Original WWW risk weights changed.")
                        assigned = np.zeros(n)
                        assigned[np.lexsort((np.arange(n), loss, rank_sum))] = weights
                    else:
                        require(np.all(raw == 0) and all(r["loss_gap"] == r["joint_rank_score"] == "" for r in records),
                                "Bootstrap must retain unavailable risk and no invented loss gap.")
                        assigned = np.zeros(n)
                    require(np.allclose(assigned, [float(r["assigned_risk"]) for r in records], rtol=0, atol=3e-8),
                            "Risk assignment disagrees with joint history ordering.")
                    if options["mode"] == "shuffled_risk":
                        if client not in generators:
                            generators[client] = torch.Generator().manual_seed(summary["seed"] + 1000003*client + 31415)
                        assigned = assigned[torch.randperm(n, generator=generators[client]).numpy()]
                    require(np.allclose(assigned, used, rtol=0, atol=3e-8), "Whole joint assignment was not correctly scrambled.")
                    np.add.at(h["zeros"], ids, used == 0)
                    np.add.at(h["visits"], ids, 1)
                    counters["history_visits_verified"] += n
                if choices_enabled:
                    candidate_key, candidate_group = next(choice_groups, (None, []))
                    require(candidate_key == key, "Candidate stream is not aligned with training batches.")
                    by_source = {int(sid): [] for sid in ids}
                    for row in candidate_group:
                        sid = int(row["sample_id"])
                        require(sid in by_source, "Candidate refers to another source.")
                        margin, cosine = float(row["teacher_margin_delta"]), float(row["nearest_teacher_cosine"])
                        require(math.isfinite(margin) and math.isfinite(cosine) and -1.00001 <= cosine <= 1.00001,
                                "Nonfinite candidate semantics or similarity.")
                        passed = margin >= -options["margin_tolerance"]
                        require(int(row["quality_passed"]) == passed and
                                options["norm_ratio_min"] <= float(row["norm_ratio"]) <= options["norm_ratio_max"] and
                                0 <= int(row["nearest_teacher_source_id"]) < sizes[client], "Invalid candidate constraints.")
                        by_source[sid].append(row)
                    for record in records:
                        candidates = by_source[int(record["sample_id"])]
                        require(candidates and int(record["attempts"]) == options["attempts"],
                                "Multi-candidate selection stopped before its fixed budget.")
                        attempts = [int(c["attempt"]) for c in candidates]
                        require(len(set(attempts)) == len(attempts) and min(attempts) >= 1 and max(attempts) <= options["attempts"],
                                "Candidate attempts are duplicated or outside budget.")
                        feasible = [c for c in candidates if int(c["quality_passed"])]
                        if feasible:
                            selected = min(feasible, key=lambda c: (float(c["nearest_teacher_cosine"]),
                                -float(c["teacher_margin_delta"]), int(c["attempt"])))
                        else:
                            selected = max(candidates, key=lambda c: (float(c["teacher_margin_delta"]), -int(c["attempt"])))
                        require(int(record["selected_attempt"]) == int(selected["attempt"]), "Candidate selection rule disagrees.")
                        for field in ("norm_ratio", "teacher_margin_delta", "quality_passed", "nearest_teacher_cosine",
                                      "nearest_teacher_source_id"):
                            require(record[field] == selected[field], f"Selected candidate {field} disagrees.")
                    counters["selected_visits_verified"] += n
                    counters["candidates_verified"] += sum(len(v) for v in by_source.values())
                counters["batches_verified"] += 1
        if choices_enabled:
            require(next(choice_groups, None) is None, "Unused candidate batches remain.")
    finally:
        if choice_handle is not None:
            choice_handle.close()
    if history_enabled:
        path = directory / "history_state.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        sources.append(path)
        require(set(saved) == set(histories), "Saved history client identities differ.")
        for client, h in histories.items():
            require(saved[client]["round"] == h["round"] - 1, "Saved history round differs.")
            for field in ("total", "rounds", "zeros", "visits"):
                require(np.allclose(saved[client][field].numpy(), h[field], rtol=0, atol=1e-12),
                        f"Saved history {field} differs from replay.")
        require(counters["history_visits_verified"] == summary["counts"]["visits"], "History visit total differs.")
    if choices_enabled:
        require(counters["selected_visits_verified"] == summary["counts"]["visits"], "Selection visit total differs.")
    hashes = {str(p): fingerprint(p) for p in sources}
    if multiview_evidence:
        hashes.update(multiview_evidence["source_hashes"])
    return dict(status="verified", **counters, source_hashes=hashes,
                **({"multiview": {k:v for k,v in multiview_evidence.items() if k!='source_hashes'}} if multiview else {}),
                scope="Exact stream replay; candidate distances are logged teacher proxies, not MIA guarantees.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    if args.output:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "source_hashes"}, indent=2))
