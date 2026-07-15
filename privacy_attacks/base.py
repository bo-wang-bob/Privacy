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
        return {
            "attack": self.name,
            "primary_metric": PRIMARY_METRIC,
            "primary_score": metrics[PRIMARY_METRIC],
            **metrics,
            "num_samples": int(self.labels.numel()),
            "metadata": self.metadata,
        }
