"""Resolve membership semantics consistently across experiment entry points."""

import math

PEFT_MODELS = {
    "clip_mlp", "clip_adapter", "visual_adapter", "clip_lora",
    "bert_adapter", "bert_lora", "gpt2_adapter",
}


def update_membership_attacks(audit: dict) -> list[str]:
    return list(dict.fromkeys(
        audit.get("exact_batch_membership_attacks", [])
        + audit.get("client_train_membership_attacks", [])
    ))


def resolve_membership_protocol(audit: dict, method: str) -> None:
    """Route the six upload attacks; their score definitions stay unchanged."""
    for key in ("exact_batch_membership_attacks", "client_train_membership_attacks"):
        if not isinstance(audit.get(key, []), list):
            raise ValueError(f"audit.{key} must be a list.")
    attacks = update_membership_attacks(audit)
    if method not in {"fedsgd", "fedavg"} or not attacks:
        return
    fedavg = method == "fedavg"
    audit["exact_batch_membership_attacks"] = [] if fedavg else attacks
    audit["client_train_membership_attacks"] = attacks if fedavg else []
    audit["membership_protocol"] = "client_train" if fedavg else "exact_batch"
    if fedavg:
        if audit.get("candidate_sampling") != "balanced_global_holdout":
            raise ValueError("FedAvg upload attacks require balanced_global_holdout.")
        if any(int(audit.get(key, 0)) != 0 for key in (
            "low_fpr_max_members", "low_fpr_max_nonmembers"
        )):
            raise ValueError("FedAvg uses the complete client training candidate pool; remove candidate caps.")
        audit["require_full_target_train_members"] = True


def resolve_federated_protocol(config: dict) -> None:
    model = str(config.get("model_type", "")).lower()
    if model not in PEFT_MODELS:
        return
    method = str(config.setdefault("aggregator", "fedsgd")).lower()
    if method not in {"fedsgd", "fedavg"}:
        raise ValueError("PEFT aggregator must be fedsgd or fedavg.")
    config["aggregator"] = method
    epochs = config.setdefault("local_epochs", 1)
    if isinstance(epochs, bool) or int(epochs) != float(epochs) or int(epochs) < 1:
        raise ValueError("local_epochs must be a positive integer.")
    if method == "fedsgd" and int(epochs) != 1:
        raise ValueError("FedSGD requires local_epochs=1 (one mini-batch per round).")
    config["local_epochs"] = int(epochs)
    validate_local_optimizer(config.get("optimization", {}) if "architecture" in config
                             else config.get(method, {}), method,
                             config.get("defense", {}).get("name", "none"))
    audit = config.setdefault("audit", {})
    resolve_membership_protocol(audit, method)
    if method == "fedavg":
        defense = config.get("defense", {})
        if defense.get("name") == "cofedmid" and defense.get("cofedmid_noise_space") == "gradient":
            raise ValueError("FedAvg CoFedMID requires parameter-space upload noise.")
        if defense.get("name") == "local_client_dp":
            raise ValueError("local_client_dp currently requires one-batch FedSGD.")
        if (defense.get("name") == "www"
                and defense.get("release_private_diagnostics", False)
                and defense.get("www_analysis_timing", "post_round") == "post_round"):
            raise ValueError("WWW post_round legacy diagnostics require FedSGD; disable release_private_diagnostics. Per-batch www_record_diagnostics supports FedAvg.")
    if "projres" in audit.get("client_train_membership_attacks", []):
        # Batch-size bounds have no meaning for complete-client candidates.
        projres = config.setdefault("projres", {})
        projres.update(max_candidates=0, min_nonmembers=0, max_nonmembers=0)
        projres["candidate_scope"] = "full_client_train"
    elif "projres" in audit.get("exact_batch_membership_attacks", []):
        projres = config.setdefault("projres", {})
        if projres.get("candidate_scope") == "full_client_train":
            members = (0 if config.get("defense", {}).get("name") == "record_dp"
                       else int(config["batch_size"]))
            nonmembers = members * int(audit.get("exact_batch_nonmember_to_member_ratio", 10))
            projres.update(max_candidates=members, min_nonmembers=nonmembers,
                           max_nonmembers=nonmembers)
        projres["candidate_scope"] = "exact_batch"


def validate_local_optimizer(optimization: dict, method: str, defense: str) -> None:
    optimizer = str(optimization.get("client_optimizer", "sgd")).lower()
    if optimizer not in {"sgd", "adamw"}:
        raise ValueError("client_optimizer must be sgd or adamw.")
    values = {}
    for key in ("momentum", "weight_decay", "max_grad_norm"):
        value = float(optimization.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
        values[key] = value
    if optimizer == "adamw" and values["momentum"]:
        raise ValueError("AdamW does not use the SGD momentum option.")
    if method == "fedsgd" and (optimizer != "sgd" or any(values.values())):
        raise ValueError("FedSGD requires vanilla SGD: momentum=weight_decay=max_grad_norm=0.")
    if values["max_grad_norm"] and defense not in {"none", "cofedmid"}:
        raise ValueError("Optimizer max_grad_norm is supported for none/cofedmid; use the selected defense's own gradient rule.")
