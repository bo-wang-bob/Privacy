from aggregator.fedavg_aggregator import FedAvgAggregator
from aggregator.dpfpl_aggregator import DPFPLAggregator
from aggregator.fedask_aggregator import FedASKAggregator


def build_aggregator(aggregator_name: str, **kwargs):
    normalized = aggregator_name.lower()
    builders = {
        "fedavg": FedAvgAggregator,
        "dpfpl": DPFPLAggregator,
        "fedask": FedASKAggregator,
    }
    if normalized not in builders:
        raise ValueError(
            f"Unknown federated method {aggregator_name!r}; choose fedavg, dpfpl, or fedask."
        )
    return builders[normalized](**kwargs)
