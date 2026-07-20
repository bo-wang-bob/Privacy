from main import _format_privacy_audit_summary


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

    message = _format_privacy_audit_summary(summaries)

    assert message.splitlines() == [
        "Privacy audit completed:",
        "  fedmia_loss | AUC=0.5828 | TPR@1%FPR=1.97% | "
        "samples=2432 (members=1216, non-members=1216)",
        "  fedmia_cosine | AUC=0.5621 | TPR@1%FPR=n/a | "
        "samples=80 (members=40, non-members=40)",
    ]
    assert "metadata" not in message
    assert "large_internal_payload" not in message


def test_privacy_audit_log_handles_no_results():
    assert (
        _format_privacy_audit_summary([])
        == "Privacy audit completed: no attack results."
    )
