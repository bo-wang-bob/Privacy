import json

import pytest

from scripts import analyze_synthesis_history_study as analysis


def record(tmp_path, name, auc, tpr, accuracy, identities="same"):
    path = tmp_path / name
    path.mkdir()
    (path / "performance_summary.json").write_text(json.dumps({"stages": {"run": {"wall_seconds": 60}}}))
    return dict(path=str(path), run=name, accuracy=accuracy, strongest_auc=auc,
        strongest_direction_symmetric_auc=auc, strongest_class_conditional_auc=auc,
        attacks=[dict(attack="one", auc=auc, tpr_at_1pct=tpr)], comparison_key="matched",
        candidate_selection_digests={"original_identity": identities})


def test_comparison_checks_privacy_and_accuracy_jointly(tmp_path):
    records = {"none_43": record(tmp_path, "none", .67, .37, .81),
               "combined_43": record(tmp_path, "combined", .65, .36, .79)}
    result = analysis.compare("combined_43", "none_43", records)
    assert result["overall_vs_none_passed"] and result["added_value_passed"]
    records["combined_43"]["accuracy"] = .789
    assert not analysis.compare("combined_43", "none_43", records)["overall_vs_none_passed"]


def test_comparison_rejects_unmatched_original_identities(tmp_path):
    records = {"none_43": record(tmp_path, "none", .67, .37, .81),
               "combined_43": record(tmp_path, "combined", .60, .18, .81, identities="different")}
    with pytest.raises(ValueError, match="original candidate identities"):
        analysis.compare("combined_43", "none_43", records)


def test_incomplete_study_cannot_create_effect_report(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    (study / "plan.json").write_text(json.dumps(dict(controls={}, jobs=[dict(id="combined_43")])) )
    monkeypatch.setattr(analysis, "check_sources", lambda plan: None)
    output = tmp_path / "analysis"
    with pytest.raises(RuntimeError, match="not started"):
        analysis.run(study, output)
    assert not output.exists()
