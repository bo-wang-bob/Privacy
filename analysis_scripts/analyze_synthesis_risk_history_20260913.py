"""Check temporal feedback and class-conditioned associations without retraining."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import digest

OUTPUT = ROOT / "analysis_scripts/risk_synthesis_failure_diagnosis_20260913"


def ranks_in_class(a, labels):
    result = np.zeros(len(a))
    for label in np.unique(labels):
        ids = labels == label
        result[ids] = rankdata(a[ids])-((ids.sum()+1)/2)
    return result


def main():
    verified_path = ROOT / "analysis_scripts/risk_synthesis_all_study_verified_20260913/verified_results.json"
    runs = dict(zip(("none", "legacy", "risk", "shuffle"), json.loads(verified_path.read_text())["runs"]))
    source_hashes = {str(verified_path): digest(verified_path), str(Path(__file__)):digest(Path(__file__))}
    histories = {}
    labels = np.array([int(r["label"]) for r in csv.DictReader((OUTPUT/"member_diagnostics.csv").open())])
    source_hashes[str(OUTPUT/"member_diagnostics.csv")] = digest(OUTPUT/"member_diagnostics.csv")
    for arm in ("risk", "shuffle"):
        p = Path(runs[arm]["path"])/"risk_synthesis/synthetic_exposure.csv"
        source_hashes[str(p)] = digest(p)
        risk = np.full((100,1000), np.nan)
        for row in csv.DictReader(p.open()):
            if int(row["client"]) == 0:
                risk[int(row["round"])-1,int(row["sample_id"])] = float(row["risk"])
        assert np.isfinite(risk).all()
        histories[arm] = risk
    history = histories["risk"]
    features = {}
    for end in (10,50,100):
        retained = 1-history[1:end]
        features[f"mean_risk_r2_r{end}"] = 1-retained.mean(0)
        features[f"zero_risk_fraction_r2_r{end}"] = (retained == 1).mean(0)
        for power in (2,4):
            features[f"mean_retention_power{power}_r2_r{end}"] = (retained**power).mean(0)
    geometry = list(csv.DictReader((OUTPUT/"geometry.csv").open()))
    for key in ("residual_uncovered_energy_fraction","leave_self_out_residual_energy_fraction",
                "leave_self_out_distance","nearest_same_class_distance"):
        features[key] = np.array([float(g[key]) for g in geometry])
    rows = []
    for arm in ("none", "risk", "shuffle"):
        p = Path(runs[arm]["path"])/"privacy_audit"
        source_hashes[str(p/"candidate_selection.pt")] = digest(p/"candidate_selection.pt")
        source_hashes[str(p/"predictions.csv")] = digest(p/"predictions.csv")
        ids = torch.load(p/"candidate_selection.pt",weights_only=True,map_location="cpu")["member_pool_indices"].numpy()
        scores = {}
        for row in csv.DictReader((p/"predictions.csv").open()):
            if int(row["membership"]) == 1:
                a=scores.setdefault(row["attack"],np.full(1000,np.nan))
                a[ids[int(row["sample_index"])]] = float(row["score"])
        for attack, outcome in scores.items():
            assert np.isfinite(outcome).all()
            for feature, value in features.items():
                local_x,local_y = ranks_in_class(value,labels),ranks_in_class(outcome,labels)
                rows.append(dict(outcome_arm=arm,attack=attack,feature=feature,
                    spearman=float(spearmanr(value,outcome).statistic),
                    correlation_of_centered_within_class_ranks=float(np.corrcoef(local_x,local_y)[0,1])))
    with (OUTPUT/"conditional_associations.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lags = {}
    for arm, risk in histories.items():
        lags[arm] = {str(lag):float(np.mean([spearmanr(risk[r],risk[r+lag]).statistic
                    for r in range(1,100-lag)])) for lag in range(1,6)}
    example_masks = dict(high_then_low=(history[1:-1]>=.8)&(history[2:]<=.2),
                         low_then_high=(history[1:-1]<=.2)&(history[2:]>=.8))
    p_high_low=float(example_masks["high_then_low"].sum()/(history[1:-1]>=.8).sum())
    p_low_high=float(example_masks["low_then_high"].sum()/(history[1:-1]<=.2).sum())
    report = dict(status="completed_posthoc_diagnosis", lag_spearman=lags,
        transition=dict(probability_next_risk_at_most_point2_given_current_at_least_point8=p_high_low,
                        probability_next_risk_at_least_point8_given_current_at_most_point2=p_low_high),
        sources=source_hashes,
        interpretation="Descriptive within-member associations. Factors use defended trajectories; even early windows "
                       "are not independent causal validation. No parameter was fitted to scores or hit labels. "
                       "Lag correlation describes policy feedback, not proof of a particular causal mechanism.")
    (OUTPUT/"history_diagnosis.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
