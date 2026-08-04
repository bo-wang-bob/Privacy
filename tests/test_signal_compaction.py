from __future__ import annotations

import torch

from analysis_scripts.compact_fedmia_signals import compact_payload


def test_compact_payload_keeps_only_fields_required_by_recorded_attacks():
    payload = {
        "candidate_labels": torch.tensor([0, 1]),
        "membership": torch.tensor([1, 0]),
        "observations": [
            {
                "round": 0,
                "client_ids": torch.tensor([0, 1]),
                "confidence": torch.ones(2, 2),
                "cosine": torch.ones(2, 2),
                "probabilities": torch.ones(2, 2, 5),
                "representations": torch.ones(2, 2, 8),
                "client_states": {0: {"prompt": torch.ones(2)}},
                "protocol_messages": {0: {}},
            }
        ],
    }
    compact = compact_payload(
        payload,
        {
            "fedmia_loss",
            "fedmia_cosine",
        },
    )
    assert compact["storage_mode"] == "compact"
    assert set(compact["observations"][0]) == {
        "round",
        "client_ids",
        "confidence",
        "cosine",
    }
    assert torch.equal(compact["candidate_labels"], payload["candidate_labels"])
    assert torch.equal(compact["membership"], payload["membership"])
