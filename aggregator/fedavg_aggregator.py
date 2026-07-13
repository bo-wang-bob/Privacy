import torch

from aggregator.base_aggregator import BaseAggregator
from context.context import Context


def aggregate_fedavg_model_states(
    ctx: Context, aggregation_ids: list[int]
) -> None:
    """Sample-weighted FedAvg over prompt parameters only."""
    if not aggregation_ids:
        raise ValueError("FedAvg requires at least one client update.")

    total_samples = sum(ctx.samples_num[user_id] for user_id in aggregation_ids)
    if total_samples <= 0:
        raise ValueError("FedAvg cannot aggregate zero training samples.")

    first_state = ctx.updated_model_state[aggregation_ids[0]]
    aggregated = {
        name: torch.zeros_like(first_state[name])
        for name in ctx.trainable_param_names
    }
    for user_id in aggregation_ids:
        weight = ctx.samples_num[user_id] / total_samples
        state = ctx.updated_model_state[user_id]
        for name in ctx.trainable_param_names:
            aggregated[name].add_(state[name], alpha=weight)
    ctx.new_model_state[0] = aggregated


class FedAvgAggregator(BaseAggregator):
    def __init__(self, device=torch.device("cpu"), **_: object):
        super().__init__(device=device)
        self.name = "fedavg"

    def _in_aggregation(self, ctx: Context):
        selected_ids = list(ctx.user_selected)
        if ctx.mode == "local":
            ctx.new_model_state = {
                user_id: ctx.updated_model_state.get(
                    user_id, ctx.base_model_state[user_id]
                )
                for user_id in range(ctx.users_num)
            }
            return
        aggregate_fedavg_model_states(ctx, selected_ids)
