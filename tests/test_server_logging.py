from servers.serverbase import _format_round_progress


def test_round_progress_compacts_full_client_selection():
    message = _format_round_progress(
        round_index=4,
        total_rounds=50,
        loss=1.23456,
        accuracy=0.4321,
        selected_ids=list(range(10)),
        total_users=10,
        audit_snapshots=5,
    )

    assert message == (
        "Progress | round=5/50 | loss=1.2346 | accuracy=43.21% | "
        "selected=all(10) | audit_snapshots=5"
    )


def test_round_progress_keeps_partial_client_ids():
    message = _format_round_progress(
        round_index=9,
        total_rounds=50,
        loss=0.5,
        accuracy=0.75,
        selected_ids=[0, 2, 7],
        total_users=10,
        audit_snapshots=10,
    )

    assert "selected=[0,2,7]" in message
