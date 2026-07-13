from aggregator.fedavg_aggregator import FedAvgAggregator


def build_aggregator(aggregator_name: str, **kwargs) -> FedAvgAggregator:
    normalized = aggregator_name.lower()
    if normalized != "fedavg":
        raise ValueError(f"Only plain FedAvg is supported; got {aggregator_name!r}.")
    return FedAvgAggregator(**kwargs)
