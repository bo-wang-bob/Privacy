import torch

from aggregator.base_aggregator import BaseAggregator
from context.context import Context


class FedASKAggregator(BaseAggregator):
    """Two-stage asymmetric-sketch aggregation from the FedASK paper/code."""

    def __init__(
        self,
        device=torch.device("cpu"),
        rank: int = 4,
        oversampling: int = 2,
        seed: int = 42,
        **_: object,
    ):
        super().__init__(device=device)
        self.name = "fedask"
        self.rank = int(rank)
        self.oversampling = int(oversampling)
        self.seed = int(seed)
        self.omega: torch.Tensor | None = None
        self.last_reconstruction_error = 0.0

    @staticmethod
    def _names(ctx: Context) -> tuple[str, str]:
        a_names = [name for name in ctx.trainable_param_names if name.endswith("fedask_A")]
        b_names = [name for name in ctx.trainable_param_names if name.endswith("fedask_B")]
        if len(a_names) != 1 or len(b_names) != 1:
            raise ValueError("FedASK requires exactly one fedask_A/fedask_B pair.")
        return a_names[0], b_names[0]

    def _in_aggregation(self, ctx: Context):
        selected = list(ctx.user_selected)
        if not selected:
            raise ValueError("FedASK requires at least one client update.")
        a_name, b_name = self._names(ctx)
        first_a = ctx.updated_model_state[selected[0]][a_name]
        sketch_dim = min(
            int(first_a.shape[1]), self.rank + max(0, self.oversampling)
        )
        if self.omega is None or self.omega.shape != (first_a.shape[1], sketch_dim):
            generator = torch.Generator(device=first_a.device).manual_seed(self.seed)
            self.omega = torch.randn(
                first_a.shape[1], sketch_dim,
                generator=generator, device=first_a.device, dtype=first_a.dtype,
            )

        total = sum(ctx.samples_num[user_id] for user_id in selected)
        if total <= 0:
            raise ValueError("FedASK cannot aggregate zero training samples.")
        weights = {user_id: ctx.samples_num[user_id] / total for user_id in selected}

        # Stage 1: clients send Y_i = B_i(A_i Omega); server forms Q.
        y = None
        products = []
        for user_id in selected:
            state = ctx.updated_model_state[user_id]
            product = state[b_name] @ state[a_name]
            products.append((weights[user_id], product))
            sketch = state[b_name] @ (state[a_name] @ self.omega)
            y = sketch * weights[user_id] if y is None else y + sketch * weights[user_id]
        assert y is not None
        q, _ = torch.linalg.qr(y, mode="reduced")

        # Stage 2: clients send P_i = A_i^T(B_i^T Q); server reconstructs BA.
        p = None
        for user_id in selected:
            state = ctx.updated_model_state[user_id]
            sketch = state[a_name].T @ (state[b_name].T @ q)
            p = sketch * weights[user_id] if p is None else p + sketch * weights[user_id]
        assert p is not None
        u, singular, vh = torch.linalg.svd(p.T, full_matrices=False)
        rank = min(self.rank, singular.numel())
        root = singular[:rank].clamp_min(0).sqrt()
        new_b = q @ (u[:, :rank] * root)
        new_a = root.unsqueeze(1) * vh[:rank]
        ctx.new_model_state[0] = {
            a_name: new_a.detach().clone(),
            b_name: new_b.detach().clone(),
        }

        exact = sum(weight * product for weight, product in products)
        reconstructed = new_b @ new_a
        self.last_reconstruction_error = float(
            (reconstructed - exact).norm() / exact.norm().clamp_min(1e-12)
        )

