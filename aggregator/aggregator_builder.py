from aggregator.seismograph_aggregator import SeismographAggregator


def build_aggregator(aggregator_name: str, **kwargs) -> SeismographAggregator:
    """Build the only aggregator supported by this minimal branch."""
    normalized_name = aggregator_name.lower()
    if normalized_name != "seismograph":
        raise ValueError(
            "This branch only supports the 'seismograph' aggregator; "
            f"got {aggregator_name!r}."
        )
    return SeismographAggregator(**kwargs)
