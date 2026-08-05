from aggregator.fedavg_aggregator import FedAvgAggregator, PromptFLAggregator


def build_aggregator(aggregator_name: str, **kwargs):
    normalized = aggregator_name.lower()
    builders = {
        "fedavg": FedAvgAggregator,
        "promptfl": PromptFLAggregator,
    }
    if normalized not in builders:
        raise ValueError(
            f"Unknown federated method {aggregator_name!r}; choose fedavg or promptfl."
        )
    return builders[normalized](**kwargs)
