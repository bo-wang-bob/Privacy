"""In-process ProjRes on real client uploads and CLIP representations."""

from __future__ import annotations

import gc
import json
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from privacy_attacks.metrics import membership_metrics
from privacy_attacks.projres_mlp import strict_mlp_projres


logger = logging.getLogger(__name__)


def _observed_attack_tensor(
    *,
    parameter_name: str,
    federated_method: str,
    base_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
    client_gradient: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, str]:
    """Return the parameter signal actually visible to the server."""
    if federated_method == "fedsgd":
        if client_gradient is None or parameter_name not in client_gradient:
            raise ValueError(
                "FedSGD ProjRes requires the client's uploaded gradient for "
                f"{parameter_name}."
            )
        return (
            client_gradient[parameter_name].detach().cpu().float(),
            "uploaded_client_gradient",
        )
    if parameter_name not in base_state or parameter_name not in updated_state:
        raise ValueError(
            f"Observed client update does not contain {parameter_name}."
        )
    return (
        base_state[parameter_name].detach().cpu().float()
        - updated_state[parameter_name].detach().cpu().float(),
        "base_minus_client_post_state",
    )


def _metric_payload(
    labels: torch.Tensor,
    scores: torch.Tensor,
    residuals: torch.Tensor,
) -> dict[str, object]:
    labels = labels.detach().cpu().long()
    scores = scores.detach().cpu()
    residuals = residuals.detach().cpu()
    member_residuals = residuals[labels == 1]
    nonmember_residuals = residuals[labels == 0]
    metrics = membership_metrics(labels, scores)
    nonmember_count = int((labels == 0).sum())
    availability = {}
    reportable_metrics = {"auc": metrics["auc"]}
    for target in (0.1, 0.01, 0.001):
        key = f"tpr_at_fpr_{target:g}"
        minimum = math.ceil(1.0 / target)
        resolvable = nonmember_count >= minimum
        availability[key] = {
            "resolvable": resolvable,
            "minimum_nonmembers": minimum,
            "actual_nonmembers": nonmember_count,
        }
        reportable_metrics[key] = metrics[key] if resolvable else None
    return {
        **metrics,
        "reportable_metrics": reportable_metrics,
        "metric_availability": availability,
        "fpr_resolution": 1.0 / nonmember_count,
        "member_mean_l1_residual": float(member_residuals.mean()),
        "nonmember_mean_l1_residual": float(nonmember_residuals.mean()),
        "l1_residual_gap": float(
            nonmember_residuals.mean() - member_residuals.mean()
        ),
    }


def _attacked_layer(
    model: torch.nn.Module,
) -> tuple[str, torch.nn.Linear]:
    model_type = str(getattr(model, "model_type", ""))
    if model_type == "clip_mlp":
        return "classifier.0.weight", model.classifier[0]
    if model_type == "visual_adapter":
        return "adapter.net.0.weight", model.adapter.net[0]
    raise ValueError(f"Unsupported ProjRes model type {model_type!r}.")


def _collect_cached_features(
    datasets: list,
    source_names: list[str],
    collate_fn,
    batch_size: int,
    maximum: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if len(datasets) != len(source_names) or not datasets:
        raise ValueError("ProjRes candidate sources must be non-empty and named.")
    remaining = maximum
    feature_parts = []
    label_parts = []
    source_counts = {}
    for dataset, source in zip(datasets, source_names):
        count = 0
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )
        for features, labels in loader:
            if features.ndim != 2:
                raise ValueError(
                    "Integrated ProjRes requires precomputed CLIP feature datasets."
                )
            if remaining is not None:
                take = min(remaining, int(labels.numel()))
                features = features[:take]
                labels = labels[:take]
            feature_parts.append(features.detach().cpu().float())
            label_parts.append(labels.detach().cpu().long())
            count += int(labels.numel())
            if remaining is not None:
                remaining -= int(labels.numel())
                if remaining == 0:
                    break
        source_counts[source] = count
        if remaining == 0:
            break
    if not label_parts:
        raise ValueError("The integrated ProjRes candidate pool is empty.")
    return torch.cat(feature_parts), torch.cat(label_parts), source_counts


def _collect_lora_representations(
    datasets: list,
    source_names: list[str],
    model: torch.nn.Module,
    collate_fn,
    batch_size: int,
    maximum: int | None,
    attacked_parameter: str,
    token_reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], int]:
    """Encode raw candidates at the input of the attacked vision LoRA layer."""
    if len(datasets) != len(source_names) or not datasets:
        raise ValueError("LoRA ProjRes candidate sources must be non-empty and named.")
    extractor = getattr(model, "get_projres_representations", None)
    if not callable(extractor):
        raise TypeError("CLIP-LoRA must expose get_projres_representations().")
    remaining = maximum
    representation_parts = []
    label_parts = []
    source_counts = {}
    tokens_per_sample = None
    for dataset, source in zip(datasets, source_names):
        count = 0
        loader = DataLoader(
            dataset,
            batch_size=min(int(batch_size), 64),
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )
        for images, labels in loader:
            if images.ndim != 4:
                raise ValueError(
                    "CLIP-LoRA ProjRes requires raw preprocessed image tensors."
                )
            take = int(labels.numel())
            if remaining is not None:
                take = min(remaining, take)
                images = images[:take]
                labels = labels[:take]
            representations, observed_tokens = extractor(
                images,
                parameter_name=attacked_parameter,
                token_reduction=token_reduction,
            )
            if representations.ndim != 2 or representations.shape[0] != take:
                raise ValueError(
                    "LoRA ProjRes representations must be [samples, hidden]."
                )
            if tokens_per_sample is None:
                tokens_per_sample = int(observed_tokens)
            elif tokens_per_sample != int(observed_tokens):
                raise ValueError("LoRA ProjRes token counts changed across batches.")
            representation_parts.append(representations.detach().cpu().float())
            label_parts.append(labels.detach().cpu().long())
            count += take
            if remaining is not None:
                remaining -= take
                if remaining == 0:
                    break
        source_counts[source] = count
        if remaining == 0:
            break
    if not label_parts or tokens_per_sample is None:
        raise ValueError("The CLIP-LoRA ProjRes candidate pool is empty.")
    return (
        torch.cat(representation_parts),
        torch.cat(label_parts),
        source_counts,
        tokens_per_sample,
    )


def _run_lora_client(
    client_id: int,
    users: list,
    global_model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
    client_gradient: dict[str, torch.Tensor] | None,
    nonmember_representations: torch.Tensor,
    nonmember_labels: torch.Tensor,
    nonmember_source_counts: dict[str, int],
    nonmember_tokens_per_sample: int,
    learning_rate: float,
    round_index: int,
    threshold: float | None,
    max_candidates: int,
    attacked_parameter: str,
    token_reduction: str,
) -> dict[str, object]:
    """Run Deng et al.'s ProjRes on a real one-batch LoRA-A upload."""
    target = users[client_id]
    if target.last_train_batch is None:
        raise ValueError(
            f"FedSGD client {client_id} did not retain its observed batch."
        )
    member_images, member_labels = target.last_train_batch
    if member_images.ndim != 4:
        raise ValueError("CLIP-LoRA ProjRes members must be raw image tensors.")
    extractor = getattr(global_model, "get_projres_representations")
    member_representations, member_tokens_per_sample = extractor(
        member_images,
        parameter_name=attacked_parameter,
        token_reduction=token_reduction,
    )
    member_representations = member_representations.detach().cpu().float()
    member_labels = member_labels.detach().cpu().long()
    if int(member_tokens_per_sample) != int(nonmember_tokens_per_sample):
        raise ValueError("Member and non-member CLIP token counts must match.")
    actual_batch_size = int(member_labels.numel())
    member_count = min(actual_batch_size, int(max_candidates))

    observed_update, update_source = _observed_attack_tensor(
        parameter_name=attacked_parameter,
        federated_method="fedsgd",
        base_state=base_state,
        updated_state=updated_state,
        client_gradient=client_gradient,
    )
    candidate_representations = torch.cat(
        (member_representations[:member_count], nonmember_representations)
    )
    hidden_vector_count = actual_batch_size * int(member_tokens_per_sample)
    attack = strict_mlp_projres(
        observed_update,
        candidate_representations,
        threshold=threshold,
        max_rank=hidden_vector_count,
    )
    attack.metadata["attacked_parameter"] = attacked_parameter
    attack.metadata["sample_representation"] = (
        f"{token_reduction}_token_input_to_attacked_vision_lora"
    )
    attack.metadata["lora_factor"] = "A_down_projection"
    labels = torch.cat(
        (
            torch.ones(member_count, dtype=torch.long),
            torch.zeros(nonmember_labels.numel(), dtype=torch.long),
        )
    )
    metrics = _metric_payload(labels, attack.scores, attack.l1_residuals)
    surface_getter = getattr(global_model, "get_projres_attack_surface")
    _, attacked_module = surface_getter(attacked_parameter)
    input_dimension = int(attacked_module.in_features)
    output_dimension = int(attacked_module.rank)
    logger.info(
        "Integrated LoRA ProjRes | round=%d | client=%d | hidden_vectors=%d | "
        "rank=%d | auc=%.4f",
        round_index + 1,
        client_id,
        hidden_vector_count,
        int(attack.metadata["subspace"]["numerical_rank"]),
        float(metrics["auc"]),
    )
    return {
        "client_id": client_id,
        "model_type": "clip_lora",
        "threat_model": {
            "server": "honest-but-curious",
            "communication_round": round_index + 1,
            "rounds_observed": 1,
            "local_batches": 1,
            "local_epochs": 1,
            "federated_method": "fedsgd",
            "optimizer": "vanilla_sgd",
            "attacked_parameter": attacked_parameter,
            "attacked_lora_factor": "A_down_projection",
            "member_definition": (
                "present_in_the_observed_target_client_fedsgd_batch"
            ),
            "execution": "integrated_from_uploaded_client_gradient",
            "paper_fedsgd_exact": True,
            "paper_reference": (
                "Deng et al. (2026), Toward Efficient Membership Inference "
                "Attacks against Federated Large Language Models"
            ),
        },
        "dimensions": {
            "candidate_sampling_batch_size": actual_batch_size,
            "member_candidate_count": member_count,
            "nonmember_candidate_count": int(nonmember_labels.numel()),
            "observed_local_batches": 1,
            "input_dimension": input_dimension,
            "first_layer_output_dimension": output_dimension,
            "tokens_per_sample": int(member_tokens_per_sample),
            "observed_hidden_vector_count": hidden_vector_count,
            "sample_representation": f"{token_reduction}_token_layer_input",
            "paper_favorable_rank_condition": bool(
                hidden_vector_count <= output_dimension
                and hidden_vector_count < input_dimension
            ),
        },
        "optimization": {
            "learning_rate": learning_rate,
            "observed_signal_norm": float(observed_update.norm()),
            "update_source": update_source,
        },
        "attack": {"metrics": metrics, "metadata": attack.metadata},
        "raw": {
            "labels": labels.tolist(),
            "scores": attack.scores.detach().cpu().tolist(),
            "l1_residuals": attack.l1_residuals.detach().cpu().tolist(),
            "predictions": None,
        },
        "candidate_controls": {
            "label_matched_nonmembers": False,
            "member_labels": member_labels[:member_count].tolist(),
            "nonmember_labels": nonmember_labels.tolist(),
            "nonmember_source_counts": nonmember_source_counts,
            "nonmember_training_exposure": "never_trained",
        },
    }


def _collect_text_representations(
    datasets: list,
    source_names: list[str],
    model: torch.nn.Module,
    collate_fn,
    batch_size: int,
    maximum: int | None,
    attacked_parameter: str,
    token_reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Collect never-trained sample embeddings at an Adapter input."""
    if len(datasets) != len(source_names) or not datasets:
        raise ValueError("Text ProjRes candidate sources must be non-empty and named.")
    extractor = getattr(model, "get_projres_representations", None)
    if not callable(extractor):
        raise TypeError(
            "Transformer Adapter models must expose get_projres_representations()."
        )
    remaining = maximum
    representation_parts = []
    label_parts = []
    source_counts = {}
    for dataset, source in zip(datasets, source_names):
        count = 0
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )
        for packed_inputs, labels in loader:
            if packed_inputs.ndim != 3 or packed_inputs.shape[1] != 2:
                raise ValueError(
                    "Text ProjRes requires packed [batch, 2, length] inputs."
                )
            take = int(labels.numel())
            if remaining is not None:
                take = min(remaining, take)
                packed_inputs = packed_inputs[:take]
                labels = labels[:take]
            representations, _ = extractor(
                packed_inputs,
                parameter_name=attacked_parameter,
                token_reduction=token_reduction,
            )
            if representations.ndim != 2 or representations.shape[0] != take:
                raise ValueError(
                    "Text ProjRes representations must be [samples, hidden]."
                )
            representation_parts.append(representations.detach().cpu().float())
            label_parts.append(labels.detach().cpu().long())
            count += take
            if remaining is not None:
                remaining -= take
                if remaining == 0:
                    break
        source_counts[source] = count
        if remaining == 0:
            break
    if not label_parts:
        raise ValueError("The text ProjRes non-member pool is empty.")
    return (
        torch.cat(representation_parts),
        torch.cat(label_parts),
        source_counts,
    )


def _run_text_adapter_client(
    client_id: int,
    users: list,
    global_model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
    client_gradient: dict[str, torch.Tensor] | None,
    nonmember_representations: torch.Tensor,
    nonmember_labels: torch.Tensor,
    nonmember_source_counts: dict[str, int],
    learning_rate: float,
    round_index: int,
    threshold: float | None,
    max_candidates: int,
    attacked_parameter: str,
    token_reduction: str,
) -> dict[str, object]:
    """Run paper-faithful ProjRes on one real text Adapter FedSGD upload."""
    target = users[client_id]
    if target.last_train_batch is None:
        raise ValueError(
            f"FedSGD client {client_id} did not retain its observed batch."
        )
    member_inputs, member_labels = target.last_train_batch
    if member_inputs.ndim != 3 or member_inputs.shape[1] != 2:
        raise ValueError("Text ProjRes members must be packed token tensors.")
    extractor = getattr(global_model, "get_projres_representations")
    member_representations, hidden_vector_count = extractor(
        member_inputs,
        parameter_name=attacked_parameter,
        token_reduction=token_reduction,
    )
    member_representations = member_representations.detach().cpu().float()
    member_labels = member_labels.detach().cpu().long()
    actual_batch_size = int(member_labels.numel())
    member_count = min(actual_batch_size, int(max_candidates))
    member_batch_positions = list(range(member_count))
    retained_local_indices = getattr(target, "last_train_indices", None)
    member_local_indices = (
        retained_local_indices.detach().cpu().long()[:member_count].tolist()
        if retained_local_indices is not None
        else None
    )
    if (
        member_local_indices is not None
        and len(member_local_indices) != member_count
    ):
        raise ValueError(
            "Retained text ProjRes member indices do not match the batch."
        )
    observed_update, update_source = _observed_attack_tensor(
        parameter_name=attacked_parameter,
        federated_method="fedsgd",
        base_state=base_state,
        updated_state=updated_state,
        client_gradient=client_gradient,
    )
    candidates = torch.cat(
        (member_representations[:member_count], nonmember_representations)
    )
    attack = strict_mlp_projres(
        observed_update,
        candidates,
        threshold=threshold,
        max_rank=int(hidden_vector_count),
    )
    attack.metadata["attacked_parameter"] = attacked_parameter
    attack.metadata["sample_representation"] = (
        f"{token_reduction}_sample_embedding_input_to_adapter_down_projection"
    )
    labels = torch.cat(
        (
            torch.ones(member_count, dtype=torch.long),
            torch.zeros(nonmember_labels.numel(), dtype=torch.long),
        )
    )
    metrics = _metric_payload(labels, attack.scores, attack.l1_residuals)
    _, attacked_layer = global_model.get_projres_attack_surface(
        attacked_parameter
    )
    input_dimension = int(attacked_layer.in_features)
    output_dimension = int(attacked_layer.out_features)
    model_type = str(getattr(global_model, "model_type", ""))
    logger.info(
        "Integrated text ProjRes | round=%d | client=%d | hidden_vectors=%d | "
        "rank=%d | auc=%.4f",
        round_index + 1,
        client_id,
        hidden_vector_count,
        int(attack.metadata["subspace"]["numerical_rank"]),
        float(metrics["auc"]),
    )
    return {
        "client_id": client_id,
        "model_type": model_type,
        "threat_model": {
            "server": "honest-but-curious",
            "communication_round": round_index + 1,
            "rounds_observed": 1,
            "local_batches": 1,
            "local_epochs": 1,
            "federated_method": "fedsgd",
            "optimizer": "vanilla_sgd",
            "attacked_parameter": attacked_parameter,
            "member_definition": (
                "present_in_the_observed_target_client_fedsgd_batch"
            ),
            "execution": "integrated_from_uploaded_client_gradient",
            # The paper training protocol uses one batch of 16 examples. A
            # larger one-batch run remains a valid observed-update ProjRes
            # experiment, but must not be labeled as the exact paper setup.
            "paper_fedsgd_exact": actual_batch_size == 16,
            "paper_reference": (
                "Deng et al. (2026), Toward Efficient Membership Inference "
                "Attacks against Federated Large Language Models"
            ),
        },
        "dimensions": {
            "candidate_sampling_batch_size": actual_batch_size,
            "member_candidate_count": member_count,
            "nonmember_candidate_count": int(nonmember_labels.numel()),
            "observed_local_batches": 1,
            "input_dimension": input_dimension,
            "first_layer_output_dimension": output_dimension,
            "observed_hidden_vector_count": int(hidden_vector_count),
            "sample_representation": f"{token_reduction}_sample_embedding",
            "paper_favorable_rank_condition": bool(
                hidden_vector_count <= output_dimension
                and hidden_vector_count < input_dimension
            ),
        },
        "optimization": {
            "learning_rate": learning_rate,
            "observed_signal_norm": float(observed_update.norm()),
            "update_source": update_source,
        },
        "attack": {"metrics": metrics, "metadata": attack.metadata},
        "raw": {
            "labels": labels.tolist(),
            "scores": attack.scores.detach().cpu().tolist(),
            "l1_residuals": attack.l1_residuals.detach().cpu().tolist(),
            "predictions": None,
        },
        "candidate_controls": {
            "label_matched_nonmembers": False,
            "member_labels": member_labels[:member_count].tolist(),
            "member_batch_positions": member_batch_positions,
            "member_local_indices": member_local_indices,
            "nonmember_labels": nonmember_labels.tolist(),
            "nonmember_source_counts": nonmember_source_counts,
            "nonmember_training_exposure": "never_trained",
        },
    }


def _run_client(
    client_id: int,
    users: list,
    global_model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
    client_gradient: dict[str, torch.Tensor] | None,
    learning_rate: float,
    batch_size: int,
    eval_batch_size: int,
    local_epochs: int,
    federated_method: str,
    round_index: int,
    seed: int,
    threshold: float | None,
    max_candidates: int,
    min_nonmembers: int,
    max_nonmembers: int | None,
) -> dict[str, object]:
    target = users[client_id]
    if federated_method == "fedsgd":
        if target.last_train_batch is None:
            raise ValueError(
                f"FedSGD client {client_id} did not retain its observed batch."
            )
        member_features, member_labels = target.last_train_batch
    else:
        loader = DataLoader(
            target.train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=target.collate_fn,
            generator=torch.Generator().manual_seed(seed + client_id),
            drop_last=False,
        )
        try:
            member_features, member_labels = next(iter(loader))
        except StopIteration as error:
            raise ValueError(f"Client {client_id} has no training batch.") from error
    if member_features.ndim != 2:
        raise ValueError(
            "Integrated ProjRes requires precomputed CLIP feature datasets."
        )
    member_features = member_features.detach().cpu().float()
    member_labels = member_labels.detach().cpu().long()
    member_count = min(int(member_labels.numel()), max_candidates)

    nonmember_datasets = [
        user.test_data for user in users if len(user.test_data)
    ]
    nonmember_source_names = [
        f"independent_test:{user.id}"
        for user in users
        if len(user.test_data)
    ]
    nonmember_datasets.extend(
        user.train_data
        for user in users
        if user.id != client_id and len(user.train_data)
    )
    nonmember_source_names.extend(
        f"other_client_train:{user.id}"
        for user in users
        if user.id != client_id and len(user.train_data)
    )
    nonmember_features, nonmember_labels, source_counts = (
        _collect_cached_features(
            nonmember_datasets,
            nonmember_source_names,
            target.collate_fn,
            eval_batch_size,
            max_nonmembers,
        )
    )
    if nonmember_labels.numel() < min_nonmembers:
        raise ValueError(
            "Strict ProjRes needs at least "
            f"{min_nonmembers} non-members to resolve 0.1% FPR; found "
            f"{nonmember_labels.numel()}."
        )

    attacked_parameter, first_layer = _attacked_layer(global_model)
    observed_update, update_source = _observed_attack_tensor(
        parameter_name=attacked_parameter,
        federated_method=federated_method,
        base_state=base_state,
        updated_state=updated_state,
        client_gradient=client_gradient,
    )
    candidate_features = torch.cat(
        (member_features[:member_count], nonmember_features)
    )
    attack = strict_mlp_projres(
        observed_update,
        candidate_features,
        threshold=threshold,
    )
    labels = torch.cat(
        (
            torch.ones(member_count, dtype=torch.long),
            torch.zeros(nonmember_labels.numel(), dtype=torch.long),
        )
    )
    metrics = _metric_payload(labels, attack.scores, attack.l1_residuals)
    actual_batch_size = int(member_labels.numel())
    if federated_method == "fedsgd":
        local_batches = 1
    else:
        batches_per_epoch = (
            len(target.trainloader)
            if hasattr(target, "trainloader")
            else math.ceil(len(target.train_data) / batch_size)
        )
        local_batches = int(local_epochs * batches_per_epoch)
    logger.info(
        "Integrated ProjRes | round=%d | client=%d | local_batches=%d | "
        "rank=%d | auc=%.4f",
        round_index + 1,
        client_id,
        local_batches,
        int(attack.metadata["subspace"]["numerical_rank"]),
        float(metrics["auc"]),
    )
    return {
        "client_id": client_id,
        "model_type": str(getattr(global_model, "model_type", "")),
        "threat_model": {
            "server": "honest-but-curious",
            "communication_round": round_index + 1,
            "rounds_observed": 1,
            "local_batches": local_batches,
            "local_epochs": int(local_epochs),
            "federated_method": federated_method,
            "optimizer": "actual_client_optimizer",
            "attacked_parameter": attacked_parameter,
            "member_definition": (
                "present_in_the_observed_target_client_fedsgd_batch"
                if federated_method == "fedsgd"
                else "present_in_the_target_client_local_training_data_for_the_"
                "observed_round"
            ),
            "execution": (
                "integrated_from_uploaded_client_gradient"
                if federated_method == "fedsgd"
                else "integrated_from_client_post_state"
            ),
            "paper_fedsgd_exact": local_batches == 1,
        },
        "dimensions": {
            "candidate_sampling_batch_size": actual_batch_size,
            "member_candidate_count": member_count,
            "nonmember_candidate_count": int(nonmember_labels.numel()),
            "observed_local_batches": local_batches,
            "input_dimension": int(first_layer.in_features),
            "first_layer_output_dimension": int(first_layer.out_features),
            "paper_favorable_rank_condition": bool(
                local_batches == 1
                and actual_batch_size <= int(first_layer.out_features)
                and actual_batch_size < int(first_layer.in_features)
            ),
        },
        "optimization": {
            "learning_rate": learning_rate,
            "observed_signal_norm": float(observed_update.norm()),
            "update_source": update_source,
        },
        "attack": {"metrics": metrics, "metadata": attack.metadata},
        "raw": {
            "labels": labels.tolist(),
            "scores": attack.scores.detach().cpu().tolist(),
            "l1_residuals": attack.l1_residuals.detach().cpu().tolist(),
            "predictions": None,
        },
        "candidate_controls": {
            "label_matched_nonmembers": False,
            "member_labels": member_labels[:member_count].tolist(),
            "nonmember_labels": nonmember_labels.tolist(),
            "nonmember_source_counts": source_counts,
        },
    }


def _aggregate(
    results: list[dict[str, object]], model_type: str
) -> dict[str, object]:
    labels = torch.cat(
        [torch.tensor(row["raw"]["labels"], dtype=torch.long) for row in results]
    )
    scores = torch.cat(
        [torch.tensor(row["raw"]["scores"], dtype=torch.float64) for row in results]
    )
    residuals = torch.cat(
        [
            torch.tensor(row["raw"]["l1_residuals"], dtype=torch.float64)
            for row in results
        ]
    )
    pooled = _metric_payload(labels, scores, residuals)
    metric_rows = [row["attack"]["metrics"] for row in results]
    numeric_keys = [
        key
        for key, value in metric_rows[0].items()
        if isinstance(value, (int, float))
    ]
    macro = {
        key: sum(float(row[key]) for row in metric_rows) / len(metric_rows)
        for key in numeric_keys
    }
    return {
        "experiment": f"observed_update_{model_type}_projres_all_clients",
        "model_type": model_type,
        "pooled_metrics": pooled,
        "client_macro_metrics": macro,
        "per_client": results,
    }


def run_integrated_projres(
    *,
    model: torch.nn.Module,
    users: list,
    device: torch.device,
    base_states: dict[int, dict[str, torch.Tensor]],
    updated_states: dict[int, dict[str, torch.Tensor]],
    client_gradients: dict[int, dict[str, torch.Tensor]] | None = None,
    learning_rate: float,
    batch_size: int,
    eval_batch_size: int,
    local_epochs: int,
    round_index: int,
    seed: int,
    dataset_name: str,
    client_ids: list[int],
    config: dict,
    output_path: str | Path,
    federated_method: str = "fedavg",
) -> dict[str, object]:
    """Run ProjRes on client updates observed in one real training round."""
    if not client_ids:
        raise ValueError("Integrated ProjRes requires at least one client.")
    if str(getattr(model, "model_type", "")) not in {
        "clip_mlp",
        "visual_adapter",
        "clip_lora",
        "bert_adapter",
        "gpt2_adapter",
    }:
        raise ValueError(
            "Integrated ProjRes requires a supported CLIP or Transformer "
            "parameter-efficient model."
        )
    threshold = config.get("threshold")
    if threshold is not None:
        raise ValueError("ProjRes is ranking-only and requires threshold=null.")
    if str(config.get("decision_mode", "ranking")).lower() != "ranking":
        raise ValueError("ProjRes decision_mode must be ranking.")
    max_candidates = int(config.get("max_candidates", 32))
    min_nonmembers = int(config.get("min_nonmembers", 1000))
    configured_max_nonmembers = int(config.get("max_nonmembers", 20000))
    results = []
    model_type = str(getattr(model, "model_type", ""))
    if federated_method == "fedsgd":
        missing_gradients = sorted(
            set(client_ids) - set(client_gradients or {})
        )
        if missing_gradients:
            raise ValueError(
                "FedSGD ProjRes requires uploaded gradients for all audited "
                f"clients; missing={missing_gradients}."
            )
    if model_type in {"bert_adapter", "gpt2_adapter"}:
        if federated_method != "fedsgd" or int(local_epochs) != 1:
            raise ValueError(
                "Paper-faithful text Adapter ProjRes requires one-batch FedSGD."
            )
        attacked_parameter, _ = model.get_projres_attack_surface(
            config.get("attacked_parameter")
        )
        token_reduction = str(config.get("token_reduction", "auto")).lower()
        if token_reduction == "auto":
            token_reduction = (
                "cls" if str(getattr(model, "architecture", "")) == "bert" else "last"
            )
        model.load_state_dict(base_states[client_ids[0]], strict=False)
        nonmember_datasets = [
            user.test_data for user in users if len(user.test_data)
        ]
        nonmember_source_names = [
            f"independent_test:{user.id}" for user in users if len(user.test_data)
        ]
        (
            nonmember_representations,
            nonmember_labels,
            nonmember_source_counts,
        ) = _collect_text_representations(
            nonmember_datasets,
            nonmember_source_names,
            model,
            users[client_ids[0]].collate_fn,
            eval_batch_size,
            None if configured_max_nonmembers == 0 else configured_max_nonmembers,
            attacked_parameter,
            token_reduction,
        )
        if nonmember_labels.numel() < min_nonmembers:
            raise ValueError(
                "Strict text Adapter ProjRes needs at least "
                f"{min_nonmembers} never-trained non-members; found "
                f"{nonmember_labels.numel()}."
            )
        for client_id in client_ids:
            results.append(
                _run_text_adapter_client(
                    client_id,
                    users,
                    model,
                    base_states[client_id],
                    updated_states[client_id],
                    client_gradients[client_id],
                    nonmember_representations,
                    nonmember_labels,
                    nonmember_source_counts,
                    learning_rate,
                    round_index,
                    threshold,
                    max_candidates,
                    attacked_parameter,
                    token_reduction,
                )
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    elif model_type == "clip_lora":
        if federated_method != "fedsgd" or int(local_epochs) != 1:
            raise ValueError(
                "Paper-faithful CLIP-LoRA ProjRes requires one-batch FedSGD."
            )
        attacked_parameter, _ = model.get_projres_attack_surface(
            config.get("attacked_parameter")
        )
        token_reduction = str(config.get("token_reduction", "cls")).lower()
        # Candidate representations must be computed under the released global
        # state that preceded the observed client update. All clients share it.
        model.load_state_dict(base_states[client_ids[0]], strict=False)
        nonmember_datasets = [
            user.test_data for user in users if len(user.test_data)
        ]
        nonmember_source_names = [
            f"independent_test:{user.id}" for user in users if len(user.test_data)
        ]
        (
            nonmember_representations,
            nonmember_labels,
            nonmember_source_counts,
            nonmember_tokens_per_sample,
        ) = _collect_lora_representations(
            nonmember_datasets,
            nonmember_source_names,
            model,
            users[client_ids[0]].collate_fn,
            eval_batch_size,
            (
                None
                if configured_max_nonmembers == 0
                else configured_max_nonmembers
            ),
            attacked_parameter,
            token_reduction,
        )
        if nonmember_labels.numel() < min_nonmembers:
            raise ValueError(
                "Strict CLIP-LoRA ProjRes needs at least "
                f"{min_nonmembers} never-trained non-members; found "
                f"{nonmember_labels.numel()}."
            )
        for client_id in client_ids:
            results.append(
                _run_lora_client(
                    client_id,
                    users,
                    model,
                    base_states[client_id],
                    updated_states[client_id],
                    client_gradients[client_id],
                    nonmember_representations,
                    nonmember_labels,
                    nonmember_source_counts,
                    nonmember_tokens_per_sample,
                    learning_rate,
                    round_index,
                    threshold,
                    max_candidates,
                    attacked_parameter,
                    token_reduction,
                )
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        for client_id in client_ids:
            results.append(
                _run_client(
                    client_id,
                    users,
                    model,
                    base_states[client_id],
                    updated_states[client_id],
                    (
                        client_gradients[client_id]
                        if client_gradients is not None
                        else None
                    ),
                    learning_rate,
                    batch_size,
                    eval_batch_size,
                    local_epochs,
                    federated_method,
                    round_index,
                    seed,
                    threshold,
                    max_candidates,
                    min_nonmembers,
                    (
                        None
                        if configured_max_nonmembers == 0
                        else configured_max_nonmembers
                    ),
                )
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(results) == 1:
        payload = {
            "experiment": f"observed_update_{model_type}_projres_single_client",
            "model_type": model_type,
            "dataset_name": dataset_name,
            "seed": seed,
            "device": str(device),
            "communication_round": round_index + 1,
            "execution": (
                "integrated_from_uploaded_client_gradient"
                if federated_method == "fedsgd"
                else "integrated_from_client_post_state"
            ),
            "result": results[0],
        }
    else:
        payload = _aggregate(results, model_type)
        payload.update(
            {
                "dataset_name": dataset_name,
                "seed": seed,
                "device": str(device),
                "communication_round": round_index + 1,
                "execution": (
                    "integrated_from_uploaded_client_gradient"
                    if federated_method == "fedsgd"
                    else "integrated_from_client_post_state"
                ),
            }
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved integrated observed-update ProjRes results to %s", path)
    return payload
