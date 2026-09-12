"""Post-hoc, member-aligned diagnosis of risk ordering and token geometry.

Does not retrain, alter formal attacks, or fit a risk factor to audit outcomes.
All predictive associations are descriptive, conditional on this existing seed.
"""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import digest

VERIFIED = ROOT / "analysis_scripts/risk_synthesis_all_study_verified_20260913/verified_results.json"
OUTPUT = ROOT / "analysis_scripts/risk_synthesis_failure_diagnosis_20260913"


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), min=float(a.min()),
                p10=float(np.quantile(a, .1)), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), max=float(a.max()))


def rho(a, b):
    return float(spearmanr(a, b).statistic) if np.std(a) > 0 and np.std(b) > 0 else None


def orthogonal_basis(matrix):
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    return u[:, s > s.max()*1e-8]


def main():
    torch.set_num_threads(1)
    report = json.loads(VERIFIED.read_text())
    names = ("none", "legacy_partial_risk", "risk", "shuffled_risk")
    records = dict(zip(names, report["runs"]))
    assert len(records) == 4 and all(r["complete"] for r in records.values())
    OUTPUT.mkdir(exist_ok=False)
    sources = {str(VERIFIED): digest(VERIFIED), str(Path(__file__)): digest(Path(__file__))}

    def source(path):
        sources[str(path)] = digest(path)
        return path

    base = Path(records["risk"]["path"])
    distribution = torch.load(source(base / "risk_synthesis/client_0_distribution.pt"), weights_only=True,
                              map_location="cpu", mmap=True)
    codes = torch.load(source(base / "risk_synthesis/client_0_source_codes.pt"), weights_only=True,
                       map_location="cpu", mmap=True)
    labels = distribution["labels"]
    assert hashlib.sha256(codes.numpy().tobytes()+labels.numpy().tobytes()).hexdigest() == distribution["source_sha256"]
    assert codes.shape == (1000, 37632)
    geometry = [None]*len(codes)
    for label, group in distribution["classes"].items():
        indices = group["indices"]
        x = codes[indices].double()
        residual = x-x.mean(0)
        factor = torch.cat((group["factor"], distribution["pooled_factor"]), dim=1).double()
        basis = orthogonal_basis(factor)
        projected = residual @ basis
        energies = residual.square().sum(1)
        outside = (energies-projected.square().sum(1)).clamp_min(0)/energies
        expected_noise_energy = .01*.5*factor.square().sum()
        distances = torch.cdist(x, x)
        distances.fill_diagonal_(float("inf"))
        for offset, sid in enumerate(indices.tolist()):
            donors = x[torch.arange(len(x)) != offset]
            anchor = donors.mean(0)
            d = x[offset]-anchor
            centered = donors-anchor
            # Leave this source out of the class tangent estimate entirely.
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            tangent = vh[:5]
            novelty = (d.square().sum()-(tangent@d).square().sum()).clamp_min(0)/d.square().sum()
            geometry[sid] = dict(sample_id=sid, label=int(label), class_count=len(indices),
                noise_rank=basis.shape[1], class_retained_variance=group["retained_variance"],
                residual_uncovered_energy_fraction=float(outside[offset]),
                leave_self_out_residual_energy_fraction=float(novelty),
                leave_self_out_distance=float(d.norm()), nearest_same_class_distance=float(distances[offset].min()),
                rms_noise_to_source_residual=float(torch.sqrt(expected_noise_energy/energies[offset])))
    print("Client-0 geometry measured", flush=True)

    histories, exposure_summaries = {}, {}
    for arm in ("risk", "shuffled_risk"):
        path = source(Path(records[arm]["path"]) / "risk_synthesis/synthetic_exposure.csv")
        matrices = {k: np.full((100, 1000), np.nan) for k in
                    ("risk", "used_risk", "quality_passed", "original_distance", "norm_ratio", "teacher_margin_delta")}
        seen = np.zeros((100, 1000), dtype=bool)
        all_counts = defaultdict(lambda: dict(visits=0, failures=0, two_attempts=0))
        for row in csv.DictReader(path.open()):
            r, client, sid = int(row["round"])-1, int(row["client"]), int(row["sample_id"])
            assert int(row["requested"]) == int(row["accepted"]) == 1
            used = float(row["used_risk"])
            key = str(min(4, int(5*used)))
            all_counts[key]["visits"] += 1
            all_counts[key]["failures"] += 1-int(row["quality_passed"])
            all_counts[key]["two_attempts"] += int(row["attempts"]) == 2
            if client != 0:
                continue
            assert not seen[r, sid] and int(labels[sid]) == int(row["label"])
            seen[r, sid] = True
            for key in matrices:
                matrices[key][r, sid] = float(row[key])
        assert seen.all() and all(np.isfinite(a).all() for a in matrices.values())
        assert sum(c["visits"] for c in all_counts.values()) == 1000000
        used = matrices["used_risk"]
        unmodified_coefficient = used == 0
        unique_distances = [len(np.unique(matrices["original_distance"][:, i])) for i in range(1000)]
        adjacent = [rho(matrices["risk"][r], matrices["risk"][r+1]) for r in range(1, 99)]
        top_overlap = [len(set(np.argsort(matrices["risk"][r])[-200:]) &
                           set(np.argsort(matrices["risk"][r+1])[-200:]))/200 for r in range(1, 99)]
        exposure_summaries[arm] = dict(visits=100000, source_count=1000, visits_per_source=100,
            mean_retained_original_fraction=float((1-used).mean()),
            zero_used_risk_fraction_after_first=float(unmodified_coefficient[1:].mean()),
            zero_used_risk_visits_per_source=stats(unmodified_coefficient[1:].sum(0)),
            unique_logged_original_distances_per_source=stats(unique_distances),
            adjacent_round_assigned_risk_spearman=stats(adjacent),
            adjacent_top20pct_overlap=stats(top_overlap), global_used_risk_bins=dict(all_counts))
        histories[arm] = matrices
    print("Two complete exposure streams checked", flush=True)

    attack_arrays = {}
    for arm in ("none", "risk", "shuffled_risk"):
        directory = Path(records[arm]["path"])
        selection = torch.load(source(directory / "privacy_audit/candidate_selection.pt"),
                               weights_only=True, map_location="cpu")
        member_ids = selection["member_pool_indices"].numpy()
        assert sorted(member_ids.tolist()) == list(range(1000))
        by_attack = defaultdict(list)
        for row in csv.DictReader(source(directory / "privacy_audit/predictions.csv").open()):
            by_attack[row["attack"]].append(row)
        attack_arrays[arm] = {}
        official = {a["attack"]:a for a in records[arm]["attacks"]}
        for name, rows in by_attack.items():
            negative = np.array([float(r["score"]) for r in rows if int(r["membership"]) == 0])
            threshold = np.nextafter(np.sort(negative)[::-1][10], np.inf)
            scores = np.full(1000, np.nan)
            for row in rows:
                if int(row["membership"]):
                    scores[member_ids[int(row["sample_index"])]] = float(row["score"])
            assert np.isfinite(scores).all()
            hits = scores >= threshold
            assert abs(float(hits.mean())-official[name]["tpr_at_1pct"]) < 1e-12
            attack_arrays[arm][name] = dict(score=scores, hits=hits, threshold=threshold)

    h = histories["risk"]
    features = {"early_assigned_risk_r2_r10": h["risk"][1:10].mean(0),
                "mean_assigned_risk_r2_r100":h["risk"][1:].mean(0),
                "last_assigned_risk":h["risk"][-1],
                "zero_risk_visit_fraction":(h["used_risk"][1:] == 0).mean(0),
                "quality_failure_fraction":1-h["quality_passed"].mean(0),
                **{k:np.array([g[k] for g in geometry]) for k in
                   ("residual_uncovered_energy_fraction", "leave_self_out_residual_energy_fraction",
                    "leave_self_out_distance", "nearest_same_class_distance")}}
    correlations = []
    for arm, attacks in attack_arrays.items():
        for attack, values in attacks.items():
            for name, feature in features.items():
                hits = values["hits"]
                top = np.argsort(feature)[-200:]
                correlations.append(dict(outcome_arm=arm, attack=attack, feature=name,
                    member_score_spearman=rho(feature, values["score"]),
                    detected_member_count=int(hits.sum()),
                    top20pct_factor_hit_rate=float(hits[top].mean()),
                    fraction_of_detected_members_in_top20pct=(float(hits[top].sum()/hits.sum()) if hits.sum() else None),
                    mean_factor_detected=float(feature[hits].mean()) if hits.any() else None,
                    mean_factor_undetected=float(feature[~hits].mean())))
    for name, rows in (("geometry.csv", geometry), ("risk_attack_associations.csv", correlations)):
        with (OUTPUT/name).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    per_source = []
    for sid in range(1000):
        per_source.append(dict(sample_id=sid,label=int(labels[sid]), **{k:float(v[sid]) for k,v in features.items()},
            **{f"{arm}_{attack}_hit1pct":int(v["hits"][sid]) for arm, attacks in attack_arrays.items()
               for attack,v in attacks.items()}))
    with (OUTPUT/"member_diagnostics.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(per_source[0]));w.writeheader();w.writerows(per_source)
    outcome = dict(scope="Post-hoc diagnosis of existing seed43, client0; no training or new attack protocol.",
        source_count=1000, embedding_dimension=37632,
        geometry={k:stats([g[k] for g in geometry]) for k in geometry[0] if k not in ("sample_id", "label")},
        exposure=exposure_summaries,
        interpretation="Member-only correlations and hit-rate enrichment, not MIA AUC or causal factor validation. "
            "Treatment-derived risk can be endogenous. Geometry is input-token geometry, not the attacked last-Adapter "
            "CLS representation. Low-dimensional noise leaves orthogonal input directions unperturbed by that noise; "
            "this is not proof those directions survive the nonlinear model or are visible to the server. "
            "Geometric proposals are measured without tuning to final attack labels; efficacy needs fresh experiments.",
        sources=sources)
    (OUTPUT/"diagnosis.json").write_text(json.dumps(outcome,indent=2,allow_nan=False))
    print(json.dumps(outcome,indent=2),flush=True)


if __name__ == "__main__":
    main()
