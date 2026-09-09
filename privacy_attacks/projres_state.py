"""Server-observable post-update model used for ProjRes representations."""

from contextlib import contextmanager
import math


@contextmanager
def projres_post_model(
    model, *, base_state, updated_state, protocol_message, learning_rate,
    federated_method,
):
    """Load one client's public endpoint, then restore the shared model.

    FedSGD exposes gradients, so its endpoint is the observable SGD step,
    not a simulator-only local optimizer state. FedAvg exposes the full delta.
    """
    if protocol_message is None and federated_method == "fedavg":
        # Legacy integrated callers supply the uploaded final parameters.
        post_state = {**base_state, **updated_state}
        source = "uploaded_client_post_state"
    else:
        kind = "gradient" if federated_method == "fedsgd" else "model_update"
        if protocol_message is None or protocol_message.get("kind") != kind:
            raise ValueError(f"ProjRes post-update representations require a {kind} upload.")
        tensors = protocol_message.get("tensors", {})
        required = {name for name, p in model.named_parameters() if p.requires_grad}
        if not required.issubset(tensors) or not set(tensors).issubset(base_state):
            raise ValueError("ProjRes post-update upload must cover all trainable base parameters.")
        if federated_method == "fedsgd":
            if learning_rate is None or not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
                raise ValueError("ProjRes post-step reconstruction requires a finite positive learning rate.")
            scale = -float(learning_rate)
            source = "base_minus_learning_rate_times_uploaded_gradient"
        else:
            scale = 1.0
            source = "base_plus_uploaded_model_delta"
        post_state = dict(base_state)
        post_state.update({
            name: base_state[name].detach() + scale * value.detach().to(base_state[name])
            for name, value in tensors.items()
        })
    current = model.state_dict()
    saved = {name: current[name].detach().clone() for name in post_state if name in current}
    try:
        model.load_state_dict(post_state, strict=False)
        yield source
    finally:
        model.load_state_dict(saved, strict=False)


def projres_uses_frozen_features(model):
    """Only these attack inputs are invariant to every local PEFT update."""
    return model.model_type == "clip_mlp" or (
        model.model_type in {"clip_adapter", "visual_adapter"}
        and getattr(model, "adapter_variant", "feature") == "feature"
    )
