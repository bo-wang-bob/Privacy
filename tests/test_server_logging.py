import pytest

from servers.serverbase import (
    _format_round_progress,
    _is_evaluation_round,
    _scheduled_learning_rate,
)


def test_evaluation_interval_uses_completed_round_numbers():
    evaluated_rounds = [
        round_index + 1
        for round_index in range(16)
        if _is_evaluation_round(round_index, total_rounds=16, eval_interval=5)
    ]

    assert evaluated_rounds == [5, 10, 15, 16]


def test_learning_rate_decay_uses_initial_rate_for_first_round():
    assert _scheduled_learning_rate(0.1, 0.99, 0) == pytest.approx(0.1)
    assert _scheduled_learning_rate(0.1, 0.99, 1) == pytest.approx(0.099)
    assert _scheduled_learning_rate(0.1, 0.99, 299) == pytest.approx(
        0.1 * 0.99**299
    )


def test_learning_rate_decay_can_step_once_every_five_rounds():
    rates = [
        _scheduled_learning_rate(0.1, 0.99, round_index, decay_interval=5)
        for round_index in range(11)
    ]

    assert rates[:5] == pytest.approx([0.1] * 5)
    assert rates[5:10] == pytest.approx([0.099] * 5)
    assert rates[10] == pytest.approx(0.1 * 0.99**2)


@pytest.mark.parametrize("decay", [0.0, -0.1, 1.01])
def test_learning_rate_decay_rejects_invalid_factors(decay):
    with pytest.raises(ValueError, match="learning_rate_decay"):
        _scheduled_learning_rate(0.1, decay, 0)


def test_learning_rate_decay_rejects_invalid_interval():
    with pytest.raises(ValueError, match="learning_rate_decay_interval"):
        _scheduled_learning_rate(0.1, 0.99, 0, decay_interval=0)


def test_round_zero_progress_is_rendered_as_pretraining_evaluation():
    message = _format_round_progress(
        round_index=-1,
        total_rounds=50,
        loss=1.0,
        accuracy=0.5,
        selected_ids=list(range(2)),
        total_users=2,
        audit_snapshots=0,
        learning_rate=0.001,
    )

    assert "round=0/50" in message
    assert "lr=0.001" in message


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
