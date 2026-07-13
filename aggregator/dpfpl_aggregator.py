import torch

from aggregator.base_aggregator import BaseAggregator
from context.context import Context
from utils.privacy_accounting import private_generator


def _clip_delta(delta: torch.Tensor, max_norm: float) -> tuple[torch.Tensor, float]:
    norm = float(delta.norm())
    factor = min(1.0, max_norm / max(norm, 1e-12))
    return delta * factor, factor


class DPFPLAggregator(BaseAggregator):
    """DP-FPL global-gradient aggregation with persistent local prompts."""

    def __init__(
        self,
        device=torch.device("cpu"),
        global_clip_norm: float = 1.0,
        global_noise_multiplier: float = 1.0,
        seed: int = 42,
        reproducible_dp_noise: bool = False,
        **_: object,
    ):
        super().__init__(device=device)
        self.name = "dpfpl"
        self.global_clip_norm = float(global_clip_norm)
        self.global_noise_multiplier = float(global_noise_multiplier)
        self.seed = int(seed)
        self.reproducible_dp_noise = bool(reproducible_dp_noise)
        self.last_clip_fraction = 0.0

    @staticmethod
    def _names(ctx: Context) -> tuple[str, str]:
        global_names = [
            name for name in ctx.trainable_param_names if name.endswith("global_ctx")
        ]
        local_names = [
            name for name in ctx.trainable_param_names if name.endswith("local_ctx")
        ]
        if len(global_names) != 1 or len(local_names) != 1:
            raise ValueError(
                "DP-FPL requires exactly one global_ctx and one local_ctx tensor."
            )
        return global_names[0], local_names[0]

    def _in_aggregation(self, ctx: Context):
        selected = list(ctx.user_selected)
        if not selected:
            raise ValueError("DP-FPL requires at least one client update.")
        global_name, local_name = self._names(ctx)
        total = sum(ctx.samples_num[user_id] for user_id in selected)
        if total <= 0:
            raise ValueError("DP-FPL cannot aggregate zero training samples.")

        reference = ctx.get_base_model_state(selected[0])[global_name]
        mean_gradient = torch.zeros_like(reference)
        clipped = 0
        weights = {user_id: ctx.samples_num[user_id] / total for user_id in selected}
        ctx.protocol_messages = {}
        for user_id in selected:
            base = ctx.get_base_model_state(user_id)[global_name]
            gradient = -(ctx.updated_model_state[user_id][global_name] - base) / max(
                ctx.learning_rate, 1e-12
            )
            gradient, factor = _clip_delta(gradient, self.global_clip_norm)
            clipped += int(factor < 1.0)
            mean_gradient.add_(gradient, alpha=weights[user_id])
            ctx.protocol_messages[user_id] = {
                "kind": "global_prompt_gradient",
                "tensors": {global_name: gradient.detach().clone()},
            }

        if self.global_noise_multiplier > 0:
            generator = private_generator(
                reference.device,
                self.reproducible_dp_noise,
                self.seed + 104729 * int(ctx.glob_iter),
            )
            noise = torch.randn(
                reference.shape,
                generator=generator,
                device=reference.device,
                dtype=reference.dtype,
            )
            mean_gradient.add_(
                noise,
                alpha=(
                    self.global_noise_multiplier
                    * self.global_clip_norm
                    * max(weights.values())
                ),
            )
        global_state = reference - ctx.learning_rate * mean_gradient
        self.last_clip_fraction = clipped / len(selected)

        ctx.new_model_state = {}
        for user_id in range(ctx.users_num):
            previous = ctx.get_base_model_state(user_id)
            local = ctx.updated_model_state.get(user_id, previous)[local_name]
            ctx.new_model_state[user_id] = {
                global_name: global_state.detach().clone(),
                local_name: local.detach().clone(),
            }
