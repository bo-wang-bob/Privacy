import torch
from context.context import Context


class BaseAggregator:
    def __init__(self, device=torch.device("cpu")):
        self.name = "BaseAggregator"
        self.device = device

    def _before_aggregation(self, ctx: Context):
        pass

    def _in_aggregation(self, ctx: Context):
        pass

    def _after_aggregation(self, ctx: Context):
        pass

    def aggregate(self, ctx: Context):
        self._before_aggregation(ctx)
        self._in_aggregation(ctx)
        self._after_aggregation(ctx)