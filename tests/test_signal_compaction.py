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
                "text_feature_cosine": torch.ones(2, 2),
                "text_feature_client_change_norms": torch.ones(2),
                "text_feature_shape": [5, 8],
                "text_feature_probe_norm": 1e-3,
                "text_feature_zero_gradient_count": 0,
                "text_feature_zero_candidate_change_count": 0,
                "text_feature_batched_context_encoding": True,
                "text_gradient_cosine": torch.ones(2, 2),
                "text_gradient_client_change_norms": torch.ones(2),
                "text_gradient_shape": [5, 8],
                "text_gradient_logit_scale": 100.0,
                "text_gradient_project_tangent": False,
                "text_gradient_zero_candidate_change_count": 0,
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
            "fedmia_text",
            "fedmia_text_gradient",
        },
    )
    assert compact["storage_mode"] == "compact"
    assert set(compact["observations"][0]) == {
        "round",
        "client_ids",
        "confidence",
        "cosine",
        "text_feature_cosine",
        "text_feature_client_change_norms",
        "text_feature_shape",
        "text_feature_probe_norm",
        "text_feature_zero_gradient_count",
        "text_feature_zero_candidate_change_count",
        "text_feature_batched_context_encoding",
        "text_gradient_cosine",
        "text_gradient_client_change_norms",
        "text_gradient_shape",
        "text_gradient_logit_scale",
        "text_gradient_project_tangent",
        "text_gradient_zero_candidate_change_count",
    }
    assert torch.equal(compact["candidate_labels"], payload["candidate_labels"])
    assert torch.equal(compact["membership"], payload["membership"])
