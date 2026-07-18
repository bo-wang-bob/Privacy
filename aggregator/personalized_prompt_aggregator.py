from __future__ import annotations

import torch

from aggregator.base_aggregator import BaseAggregator
from context.context import Context


class PersonalizedPromptAggregator(BaseAggregator):
    """Aggregate a global prompt while retaining personalization on clients."""

    def __init__(self, method_name: str, device=torch.device("cpu"), **_: object):
        super().__init__(device=device)
        self.name = str(method_name).lower()

    @staticmethod
    def _global_name(ctx: Context) -> str:
        names = [
            name for name in ctx.trainable_param_names if name.endswith("global_ctx")
        ]
        if len(names) != 1:
            raise ValueError(
                "Personalized prompt aggregation requires exactly one global_ctx."
            )
        return names[0]

    def _in_aggregation(self, ctx: Context) -> None:
        selected = list(ctx.user_selected)
        if not selected:
            raise ValueError(f"{self.name} requires at least one client update.")
        global_name = self._global_name(ctx)
        total = sum(ctx.samples_num[user_id] for user_id in selected)
        if total <= 0:
            raise ValueError(f"{self.name} cannot aggregate zero training samples.")

        first = ctx.updated_model_state[selected[0]][global_name]
        global_state = torch.zeros_like(first)
        ctx.protocol_messages = {}
        for user_id in selected:
            weight = ctx.samples_num[user_id] / total
            updated = ctx.updated_model_state[user_id][global_name]
            base = ctx.get_base_model_state(user_id)[global_name]
            global_state.add_(updated, alpha=weight)
            ctx.protocol_messages[user_id] = {
                "kind": "global_prompt_update",
                "tensors": {
                    global_name: (updated - base).detach().clone(),
                },
            }

        # Every client receives the new global prompt. Selected clients retain
        # their newly trained personalized parameters; non-participants retain
        # detached copies from their previous local state.
        ctx.new_model_state = {}
        for user_id in range(ctx.users_num):
            previous = ctx.get_base_model_state(user_id)
            source = ctx.updated_model_state.get(user_id, previous)
            state = {
                name: source[name].detach().clone()
                for name in ctx.trainable_param_names
            }
            state[global_name] = global_state.detach().clone()
            ctx.new_model_state[user_id] = state


class FedOTPAggregator(PersonalizedPromptAggregator):
    def __init__(self, **kwargs: object):
        super().__init__(method_name="fedotp", **kwargs)


class FedPGPAggregator(PersonalizedPromptAggregator):
    def __init__(self, **kwargs: object):
        super().__init__(method_name="fedpgp", **kwargs)
