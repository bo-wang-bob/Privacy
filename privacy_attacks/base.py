import math
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

    def to_summary(
        self,
        fpr_targets: tuple[float, ...] = (0.1, 0.01, 0.001),
    ) -> dict:
        if PRIMARY_METRIC not in {
            f"tpr_at_fpr_{target:g}" for target in fpr_targets
        }:
            raise ValueError(
                f"fpr_targets must include the primary metric {PRIMARY_METRIC}."
            )
        metrics = membership_metrics(
            self.labels,
            self.scores,
            target_fprs=fpr_targets,
        )
        member_count = int((self.labels == 1).sum())
        nonmember_count = int((self.labels == 0).sum())
        availability = {}
        reportable_metrics = {"auc": metrics["auc"]}
        for target in fpr_targets:
            key = f"tpr_at_fpr_{target:g}"
            minimum_nonmembers = math.ceil(1.0 / target)
            resolvable = nonmember_count >= minimum_nonmembers
            availability[key] = {
                "resolvable": resolvable,
                "minimum_nonmembers": minimum_nonmembers,
                "actual_nonmembers": nonmember_count,
            }
            reportable_metrics[key] = metrics[key] if resolvable else None
        unique_scores = int(torch.unique(self.scores.detach().cpu()).numel())
        score_std = float(
            self.scores.detach().to(torch.float64).std(unbiased=False).cpu()
        )
        return {
            "attack": self.name,
            "primary_metric": PRIMARY_METRIC,
            "primary_score": reportable_metrics[PRIMARY_METRIC],
            **metrics,
            "reportable_metrics": reportable_metrics,
            "metric_availability": availability,
            "num_samples": int(self.labels.numel()),
            "member_count": member_count,
            "nonmember_count": nonmember_count,
            "fpr_resolution": 1.0 / nonmember_count,
            "score_unique_count": unique_scores,
            "score_std": score_std,
            "score_degenerate": unique_scores <= 1 or score_std == 0.0,
            "metadata": self.metadata,
        }
