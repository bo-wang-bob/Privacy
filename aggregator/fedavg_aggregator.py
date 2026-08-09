from __future__ import annotations

import torch

from aggregator.base_aggregator import BaseAggregator
from context.context import Context


def aggregate_fedavg_model_states(
    ctx: Context,
    aggregation_ids: list[int],
    sample_counts: dict[int, int] | None = None,
) -> None:
    """Average trainable client states using the supplied client counts."""
    if not aggregation_ids:
        raise ValueError("FedAvg requires at least one client update.")

    counts = (
        {user_id: int(sample_counts[user_id]) for user_id in aggregation_ids}
        if sample_counts is not None
        else {user_id: int(ctx.samples_num[user_id]) for user_id in aggregation_ids}
    )
    total_samples = sum(counts.values())
    if total_samples <= 0:
        raise ValueError("FedAvg cannot aggregate zero training samples.")
    if any(count < 0 for count in counts.values()):
        raise ValueError("FedAvg client sample counts must be non-negative.")

    ctx.aggregation_weights = {
        user_id: counts[user_id] / total_samples for user_id in aggregation_ids
    }

    first_state = ctx.updated_model_state[aggregation_ids[0]]
    aggregated = {
        name: torch.zeros_like(first_state[name]) for name in ctx.trainable_param_names
    }
    for user_id in aggregation_ids:
        weight = ctx.aggregation_weights[user_id]
        state = ctx.updated_model_state[user_id]
        for name in ctx.trainable_param_names:
            aggregated[name].add_(state[name], alpha=weight)
    ctx.new_model_state[0] = aggregated


class FedAvgAggregator(BaseAggregator):
    def __init__(
        self,
        device=torch.device("cpu"),
        aggregation_weighting: str = "sample_count",
        **_: object,
    ):
        super().__init__(device=device)
        self.name = "fedavg"
        self.aggregation_weighting = str(aggregation_weighting).lower()
        if self.aggregation_weighting not in {"sample_count", "uniform"}:
            raise ValueError(
                "aggregation_weighting must be sample_count or uniform."
            )

    def _in_aggregation(self, ctx: Context):
        selected_ids = list(ctx.user_selected)
        ctx.protocol_messages = {
            user_id: {
                "kind": "model_update",
                "tensors": {
                    name: (
                        ctx.updated_model_state[user_id][name]
                        - ctx.get_base_model_state()[name]
                    )
                    .detach()
                    .clone()
                    for name in ctx.trainable_param_names
                },
            }
            for user_id in selected_ids
        }
        sample_counts = (
            {user_id: 1 for user_id in selected_ids}
            if self.aggregation_weighting == "uniform"
            else None
        )
        aggregate_fedavg_model_states(
            ctx,
            selected_ids,
            sample_counts=sample_counts,
        )


class FedSGDAggregator(FedAvgAggregator):
    """Aggregate one-mini-batch client SGD steps by their actual batch sizes."""

    def __init__(self, device=torch.device("cpu"), **kwargs: object):
        super().__init__(device=device, **kwargs)
        self.name = "fedsgd"

    def _in_aggregation(self, ctx: Context):
        selected_ids = list(ctx.user_selected)
        missing = sorted(set(selected_ids) - set(ctx.update_sample_counts))
        if missing:
            raise ValueError(
                "FedSGD requires an observed mini-batch size for every client; "
                f"missing clients={missing}."
            )
        ctx.protocol_messages = {
            user_id: {
                "kind": "model_update",
                "tensors": {
                    name: (
                        ctx.updated_model_state[user_id][name]
                        - ctx.get_base_model_state()[name]
                    )
                    .detach()
                    .clone()
                    for name in ctx.trainable_param_names
                },
            }
            for user_id in selected_ids
        }
        sample_counts = (
            {user_id: 1 for user_id in selected_ids}
            if self.aggregation_weighting == "uniform"
            else ctx.update_sample_counts
        )
        aggregate_fedavg_model_states(ctx, selected_ids, sample_counts=sample_counts)


class PromptFLAggregator(FedAvgAggregator):
    """Paper-named PromptFL entry point using standard prompt-only FedAvg."""

    def __init__(self, device=torch.device("cpu"), **kwargs: object):
        super().__init__(device=device, **kwargs)
        self.name = "promptfl"
