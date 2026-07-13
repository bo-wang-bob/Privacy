import logging
from typing import Optional

import torch

from .base_aggregator import BaseAggregator
from context.context import Context
from utils.seismograph_text_feature_analysis import (
    collect_text_feature_raw_top1_singular_values,
    filter_users_by_raw_top1_svd_history,
)
from utils.seismograph_update_analysis import save_pairwise_update_metric_matrices

logger = logging.getLogger(__name__)


def aggregate_seismograph_model_states(
    ctx: Context, aggregation_ids: list[int]
) -> None:
    if not aggregation_ids:
        raise ValueError("SeismographAggregator requires at least one client update.")
    aggregated_state_dict = {}
    total_samples = sum([ctx.samples_num[user_id] for user_id in aggregation_ids])
    if total_samples <= 0:
        raise ValueError("SeismographAggregator cannot aggregate zero training samples.")
    for key in ctx.trainable_param_names:
        aggregated_state_dict[key] = torch.zeros_like(
            ctx.updated_model_state[aggregation_ids[0]][key]
        )
    for user_id in aggregation_ids:
        state_dict = ctx.updated_model_state[user_id]
        num_samples = ctx.samples_num[user_id]
        weight_factor = num_samples / total_samples
        for key in ctx.trainable_param_names:
            aggregated_state_dict[key] += state_dict[key] * weight_factor

    ctx.new_model_state[0] = aggregated_state_dict


class SeismographAggregator(BaseAggregator):
    def __init__(self, device=torch.device("cpu"), **kwargs):
        super().__init__(device=device)
        self.name = "seismograph"
        self.raw_top1_log_history: dict[int, list[float]] = {}
        self.raw_top1_seismograph_state: dict[int, float] = {}
        self.seismograph_k: float = float(kwargs.get("seismograph_k", 1.0))
        self.seismograph_h: float = float(kwargs.get("seismograph_h", 5.0))
        if self.seismograph_k < 0 or self.seismograph_h <= 0:
            raise ValueError(
                "seismograph_k must be non-negative and seismograph_h must be positive."
            )

    def _run_round_analyses(
        self,
        ctx: Context,
        user_ids: list[int],
    ) -> Optional[tuple[list[int], torch.Tensor]]:
        try:
            save_pairwise_update_metric_matrices(ctx, user_ids, self.device)
        except Exception:
            logger.exception(
                "SeismographAggregator: failed to save pairwise update metric matrices for round %s.",
                ctx.glob_iter,
            )

        try:
            return collect_text_feature_raw_top1_singular_values(
                ctx,
                user_ids,
                self.device,
            )
        except Exception:
            logger.exception(
                "SeismographAggregator: failed to collect text-feature raw top-1 singular values for round %s.",
                ctx.glob_iter,
            )
            return None

    def _in_aggregation(self, ctx: Context):
        if not ctx.fpl:
            raise NotImplementedError("SeismographAggregator only supports FPL.")
        if ctx.mode != "centralized" and ctx.mode != "local":
            raise NotImplementedError(
                "SeismographAggregator only supports centralized and local mode."
            )

        if ctx.mode == "local":
            user_ids = ctx.user_selected or sorted(ctx.updated_model_state.keys())
            self._run_round_analyses(ctx, user_ids)
            ctx.new_model_state = ctx.updated_model_state
            return

        selected_ids = list(ctx.user_selected)
        raw_top1_values = self._run_round_analyses(ctx, selected_ids)

        aggregation_ids = selected_ids
        if raw_top1_values is not None:
            valid_user_ids, raw_values = raw_top1_values
            aggregation_ids = filter_users_by_raw_top1_svd_history(
                ctx=ctx,
                selected_ids=selected_ids,
                valid_user_ids=valid_user_ids,
                raw_top1_values=raw_values,
                device=self.device,
                raw_top1_log_history=self.raw_top1_log_history,
                raw_top1_seismograph_state=self.raw_top1_seismograph_state,
                seismograph_k=self.seismograph_k,
                seismograph_h=self.seismograph_h,
            )

        aggregate_seismograph_model_states(ctx, aggregation_ids)
