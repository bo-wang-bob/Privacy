from aggregator.fedavg_aggregator import (
    FedAvgAggregator,
    FedSGDAggregator,
    PromptFLAggregator,
)


def build_aggregator(aggregator_name: str, **kwargs):
    normalized = aggregator_name.lower()
    builders = {
        "fedavg": FedAvgAggregator,
        "fedsgd": FedSGDAggregator,
        "promptfl": PromptFLAggregator,
    }
    if normalized not in builders:
        raise ValueError(
            f"Unknown federated method {aggregator_name!r}; choose fedavg, "
            "fedsgd, or promptfl."
        )
    return builders[normalized](**kwargs)
