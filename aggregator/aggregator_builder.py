from aggregator.fedavg_aggregator import FedAvgAggregator, PromptFLAggregator
from aggregator.dpfpl_aggregator import DPFPLAggregator
from aggregator.fedask_aggregator import FedASKAggregator
from aggregator.personalized_prompt_aggregator import (
    FedOTPAggregator,
    FedPGPAggregator,
)


def build_aggregator(aggregator_name: str, **kwargs):
    normalized = aggregator_name.lower()
    builders = {
        "fedavg": FedAvgAggregator,
        "promptfl": PromptFLAggregator,
        "fedotp": FedOTPAggregator,
        "fedpgp": FedPGPAggregator,
        "dpfpl": DPFPLAggregator,
        "fedask": FedASKAggregator,
    }
    if normalized not in builders:
        raise ValueError(
            f"Unknown federated method {aggregator_name!r}; choose promptfl, "
            "fedotp, fedpgp, fedavg, dpfpl, or fedask."
        )
    return builders[normalized](**kwargs)
