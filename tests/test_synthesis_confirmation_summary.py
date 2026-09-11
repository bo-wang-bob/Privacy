import copy
import csv
import json

import pytest

from scripts.analyze_risk_synthesis import digest, read_run
from scripts.summarize_synthesis_confirmation import (
    ARMS, run, summarize_records, validate_design,
)


@pytest.fixture
def study():
    jobs, runs = [], []
    names = [f"attack_{i}" for i in range(11)]
    for seed in (43, 44, 45):
        protocol = dict(seed=seed, audit={"attacks": names})
        for arm in sorted(ARMS):
            job = dict(name=f"{seed}_{arm}", seed=seed, arm=arm, protocol=protocol)
            attacks = [dict(attack=name, auc=.5, tpr_at_1pct=.01,
                            class_conditional_auc=.55) for name in names]
            # The strongest attack moves to a different method under defense.
            if arm == "none":
                attacks[0].update(auc=.7, tpr_at_1pct=.3, class_conditional_auc=.75)
            else:
                attacks[0].update(auc=.55, tpr_at_1pct=.02)
                attacks[1].update(auc=.67, tpr_at_1pct=.2, class_conditional_auc=.72)
            if arm == "shuffled_risk":
                attacks[1].update(auc=.65, tpr_at_1pct=.15)
            jobs.append(job)
            runs.append(dict(path=f"/{seed}/{arm}", run=job["name"], complete=True,
                protocol=protocol, comparison_key=f"seed{seed}", attacks=attacks, accuracy=.8,
                strongest_auc=.999,  # Do not trust a cached maximum.
                candidate_selection_digests={"original_ids": str(seed)},
                candidate_metadata={"members": 1000, "nonmembers": 1000},
                confirmation_source=dict(roles_disjoint=True, exploration_records_excluded=True, ids=[seed])))
    acceptance = dict(mean_auc_reduction_at_least=.02, each_seed_auc_must_decrease=True,
                      mean_max_tpr_at_1pct_must_decrease=True, each_seed_accuracy_drop_at_most=.02)
    return jobs, runs, acceptance


def test_maximum_is_recomputed_per_model_and_risk_contribution_is_separate(study):
    report = summarize_records(*study)
    assert report["preregistered_criteria_met"]
    primary = [r for r in report["paired_seed_effects"] if r["treatment"] == "risk" and r["control"] == "none"]
    assert len(primary) == 3
    assert all(r["maximum_auc_delta"] == pytest.approx(-.03) for r in primary)
    assert all(r["maximum_tpr_at_1pct_delta"] == pytest.approx(-.1) for r in primary)
    risk_effect = [r for r in report["paired_seed_effects"] if r["treatment"] == "risk" and r["control"] == "shuffled_risk"]
    assert all(r["maximum_auc_delta"] == pytest.approx(.02) for r in risk_effect)
    assert len(report["paired_attack_effects"]) == 3 * 5 * 11


def test_mean_success_does_not_hide_one_seed_privacy_or_utility_failure(study):
    jobs, runs, acceptance = study
    for job, record in zip(jobs, runs):
        if job["arm"] == "risk":
            record["attacks"][1]["auc"] = .6 if job["seed"] != 43 else .71
            if job["seed"] == 43:
                record["accuracy"] = .77
    report = summarize_records(jobs, runs, acceptance)
    assert report["preregistered_checks"]["mean_maximum_auc_reduction"]
    assert not report["preregistered_checks"]["every_seed_maximum_auc_decreases"]
    assert not report["preregistered_checks"]["every_seed_accuracy_within_tolerance"]
    assert not report["preregistered_criteria_met"]


def test_unreportable_tpr_and_candidate_changes_are_rejected(study):
    jobs, runs, acceptance = study
    original = copy.deepcopy(runs)
    runs[1]["attacks"][2]["tpr_at_1pct"] = None
    with pytest.raises(ValueError, match="reportable"):
        summarize_records(jobs, runs, acceptance)
    original[1]["candidate_selection_digests"]["original_ids"] = "different"
    with pytest.raises(ValueError, match="candidate identities"):
        summarize_records(jobs, original, acceptance)


def test_incomplete_jobs_missing_seeds_and_missing_attacks_are_rejected(study):
    jobs, runs, acceptance = study
    with pytest.raises(ValueError, match="four-arm"):
        validate_design(jobs[:-1])
    runs[0]["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        summarize_records(jobs, runs, acceptance)
    runs[0]["complete"] = True
    runs[0]["attacks"].pop()
    with pytest.raises(ValueError, match="Every planned attack"):
        summarize_records(jobs, runs, acceptance)


def test_score_orientation_is_reported_without_changing_formal_attacks(study):
    jobs, runs, acceptance = study
    for job, record in zip(jobs, runs):
        if job["arm"] == "risk":
            record["attacks"][3]["auc"] = .1
    report = summarize_records(jobs, runs, acceptance)
    primary = [r for r in report["paired_seed_effects"] if r["treatment"] == "risk" and r["control"] == "none"]
    assert all(r["maximum_auc_delta"] == pytest.approx(-.03) for r in primary)
    assert all(r["maximum_direction_symmetric_auc_delta"] == pytest.approx(.2) for r in primary)
    assert report["orientation_diagnostic_worsens_in_any_seed"]


def test_running_lane_cannot_create_confirmation_report(study, tmp_path):
    jobs, _, acceptance = study
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(schema_version=1, jobs=jobs, acceptance=acceptance)))
    state = tmp_path / "state.json"
    state.write_text(json.dumps(dict(plan=str(plan), plan_sha256=digest(plan), status="running")))
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="not complete"):
        run(plan, [state], output)
    assert not output.exists()


def test_all_reportable_fpr_levels_are_verified_without_raw_fallback(tmp_path):
    """A known 2/10 score ordering has AUC .65 and attainable 10%-FPR TPR .5."""
    (tmp_path / "run_config.yaml").write_text("num_global_iters: 100\naudit:\n  attacks: [blackbox_loss]\n")
    (tmp_path / "training_metrics.csv").write_text("round,accuracy\n100,0.8\n")
    (tmp_path / "performance_summary.json").write_text('{"status":"completed"}')
    audit = tmp_path / "privacy_audit"
    audit.mkdir()
    summary = dict(attacks=[dict(attack="blackbox_loss", auc=.65, member_count=2, nonmember_count=10,
        reportable_metrics={"tpr_at_fpr_0.1": .5, "tpr_at_fpr_0.01": None, "tpr_at_fpr_0.001": None})])
    summary["attacks"][0]["tpr_at_fpr_0.01"] = .9
    path = audit / "summary.json"
    path.write_text(json.dumps(summary))
    scores = [.9, .4, .95, .85, .75, .65, .55, .45, .35, .25, .15, .05]
    with (audit / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attack", "audit_client_id", "sample_index", "membership", "score"])
        writer.writeheader()
        writer.writerows(dict(attack="blackbox_loss", audit_client_id=0, sample_index=i,
                              membership=int(i < 2), score=value) for i, value in enumerate(scores))
    record = read_run(tmp_path)
    assert record["attacks"][0]["tpr_at_10pct"] == .5
    assert record["attacks"][0]["tpr_at_1pct"] is None
    summary["attacks"][0]["reportable_metrics"]["tpr_at_fpr_0.1"] = .123
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Independent TPR"):
        read_run(tmp_path)
