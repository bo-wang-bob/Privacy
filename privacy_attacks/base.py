from dataclasses import dataclass, field

import torch

from privacy_attacks.metrics import PRIMARY_METRIC, membership_metrics


@dataclass
class AttackResult:
    name: str
    scores: torch.Tensor
    labels: torch.Tensor
    sample_indices: torch.Tensor
    metadata: dict = field(default_factory=dict)

    def to_summary(self) -> dict:
        metrics = membership_metrics(self.labels, self.scores)
        member_count = int((self.labels == 1).sum())
        nonmember_count = int((self.labels == 0).sum())
        return {
            "attack": self.name,
            "primary_metric": PRIMARY_METRIC,
            "primary_score": metrics[PRIMARY_METRIC],
            **metrics,
            "num_samples": int(self.labels.numel()),
            "member_count": member_count,
            "nonmember_count": nonmember_count,
            "fpr_resolution": 1.0 / nonmember_count,
            "metadata": self.metadata,
        }
