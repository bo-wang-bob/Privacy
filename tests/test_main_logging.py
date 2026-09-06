import logging

from main import (
    LAUNCHER_LOG_CAPTURE_ENV,
    _build_logging_handlers,
)
from utils.result_formatting import format_attack_table


def test_privacy_audit_log_is_compact_and_multiline():
    summaries = [
        {
            "attack": "fedmia_loss",
            "primary_metric": "tpr_at_fpr_0.01",
            "primary_score": 0.0197368421,
            "auc": 0.5827602904,
            "reportable_metrics": {
                "auc": 0.5827602904,
                "tpr_at_fpr_0.01": 0.0197368421,
            },
            "num_samples": 2432,
            "member_count": 1216,
            "nonmember_count": 1216,
            "metadata": {"large_internal_payload": list(range(100))},
        },
        {
            "attack": "fedmia_cosine",
            "primary_metric": "tpr_at_fpr_0.01",
            "primary_score": None,
            "auc": 0.5621490056,
            "reportable_metrics": {
                "auc": 0.5621490056,
                "tpr_at_fpr_0.01": None,
            },
            "num_samples": 80,
            "member_count": 40,
            "nonmember_count": 40,
        },
    ]

    message = format_attack_table(summaries)

    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in message.splitlines()
        if line.startswith("|")
    ]
    assert rows == [
        ["Attack", "AUC", "TPR@1%FPR", "TPR@0.1%FPR", "TPR@10%FPR", "Members", "Nonmembers"],
        ["fedmia_loss", "0.5828", "1.97%", "N/A", "N/A", "1216", "1216"],
        ["fedmia_cosine", "0.5621", "N/A", "N/A", "N/A", "40", "40"],
    ]
    assert "metadata" not in message
    assert "large_internal_payload" not in message


def test_privacy_audit_log_handles_no_results():
    assert (
        format_attack_table([])
        == "No reported attack results."
    )


def test_direct_run_keeps_run_log(tmp_path, monkeypatch):
    monkeypatch.delenv(LAUNCHER_LOG_CAPTURE_ENV, raising=False)

    handlers = _build_logging_handlers(str(tmp_path))
    try:
        assert any(isinstance(handler, logging.FileHandler) for handler in handlers)
        assert (tmp_path / "run.log").is_file()
    finally:
        for handler in handlers:
            handler.close()


def test_launcher_capture_avoids_duplicate_run_log(tmp_path, monkeypatch):
    monkeypatch.setenv(LAUNCHER_LOG_CAPTURE_ENV, "1")

    handlers = _build_logging_handlers(str(tmp_path))
    try:
        assert not any(
            isinstance(handler, logging.FileHandler) for handler in handlers
        )
        assert not (tmp_path / "run.log").exists()
    finally:
        for handler in handlers:
            handler.close()
