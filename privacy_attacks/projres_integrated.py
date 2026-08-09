"""In-process ProjRes on real client uploads and cached CLIP features."""

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


def _metric_payload(
    labels: torch.Tensor,
    scores: torch.Tensor,
    residuals: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, object]:
    labels = labels.detach().cpu().long()
    scores = scores.detach().cpu()
    residuals = residuals.detach().cpu()
    predictions = predictions.detach().cpu().long()
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
        "threshold_accuracy": float((predictions == labels).float().mean()),
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


def _run_client(
    client_id: int,
    users: list,
    global_model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
    learning_rate: float,
    batch_size: int,
    eval_batch_size: int,
    local_epochs: int,
    federated_method: str,
    round_index: int,
    seed: int,
    threshold: float,
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
    if attacked_parameter not in base_state or attacked_parameter not in updated_state:
        raise ValueError(
            f"Observed client update does not contain {attacked_parameter}."
        )
    observed_update = (
        base_state[attacked_parameter].detach().cpu()
        - updated_state[attacked_parameter].detach().cpu()
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
    metrics = _metric_payload(
        labels, attack.scores, attack.l1_residuals, attack.predictions
    )
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
            "execution": "integrated_from_observed_client_update",
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
            "observed_update_norm": float(observed_update.norm()),
            "update_source": "base_state_minus_uploaded_client_state",
        },
        "attack": {"metrics": metrics, "metadata": attack.metadata},
        "raw": {
            "labels": labels.tolist(),
            "scores": attack.scores.detach().cpu().tolist(),
            "l1_residuals": attack.l1_residuals.detach().cpu().tolist(),
            "predictions": attack.predictions.detach().cpu().tolist(),
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
    predictions = torch.cat(
        [
            torch.tensor(row["raw"]["predictions"], dtype=torch.long)
            for row in results
        ]
    )
    pooled = _metric_payload(labels, scores, residuals, predictions)
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
    }:
        raise ValueError("Integrated ProjRes requires CLIP-MLP or Visual Adapter.")
    threshold = float(config.get("threshold", 0.01))
    max_candidates = int(config.get("max_candidates", 32))
    min_nonmembers = int(config.get("min_nonmembers", 1000))
    configured_max_nonmembers = int(config.get("max_nonmembers", 20000))
    results = []
    for client_id in client_ids:
        results.append(
            _run_client(
                client_id,
                users,
                model,
                base_states[client_id],
                updated_states[client_id],
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
    model_type = str(getattr(model, "model_type", ""))
    if len(results) == 1:
        payload = {
            "experiment": f"observed_update_{model_type}_projres_single_client",
            "model_type": model_type,
            "dataset_name": dataset_name,
            "seed": seed,
            "device": str(device),
            "communication_round": round_index + 1,
            "execution": "integrated_from_observed_client_update",
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
                "execution": "integrated_from_observed_client_update",
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
