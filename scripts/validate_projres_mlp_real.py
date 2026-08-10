#!/usr/bin/env python3
"""Run strict one-round ProjRes on a frozen-CLIP MLP or visual adapter."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from main import _load_local_clip, default_config, validate_config
from privacy_attacks.metrics import membership_metrics
from privacy_attacks.projres_mlp import (
    one_batch_fedsgd_step,
    strict_mlp_projres,
)
from trainmodel.clip_mlp import CLIPImageMLP
from trainmodel.visual_adapter import (
    VisualCLIPAdapter,
    build_visual_adapter_text_features,
)
from utils.data_loader import generate_dirichlet_split, generate_iid_split


logger = logging.getLogger("projres_validation")


def _load_config(path: Path) -> dict:
    config = default_config()
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    for nested in ("clip_mlp", "visual_adapter", "audit", "defense"):
        config[nested] = config.get(nested, {}) | loaded.pop(nested, {})
    config.update(loaded)
    return config


def _resolved_config(args: argparse.Namespace) -> dict:
    config = _load_config(args.config)
    for name in (
        "dataset_name",
        "data_root",
        "cache_dir",
        "batch_size",
        "learning_rate",
        "dirichlet_alpha",
        "gpu",
        "seed",
    ):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
    return config


def _validate_strict_config(config: dict) -> None:
    validate_config(config)
    model_type = str(config.get("model_type", "")).lower()
    if model_type not in {"clip_mlp", "visual_adapter"}:
        raise ValueError(
            "Strict ProjRes requires model_type=clip_mlp or visual_adapter."
        )
    if str(config.get("aggregator", "")).lower() != "fedavg":
        raise ValueError("Strict ProjRes requires aggregator=fedavg.")
    if (
        model_type == "clip_mlp"
        and float(config.get("clip_mlp", {}).get("dropout", 0.0)) != 0.0
    ):
        raise ValueError("Strict MLP ProjRes requires clip_mlp.dropout=0.")
    if config.get("model_load_path"):
        raise ValueError(
            "This strict entry targets round 1 and requires model_load_path=null."
        )


def _split_data(config: dict):
    arguments = {
        "root_dir": config["data_root"],
        "fpl": True,
        "fpl_shots": config.get("fpl_shots"),
        "use_full_dataset": bool(config.get("use_full_dataset", False)),
    }
    mode = str(config.get("partition_mode", "iid")).lower()
    if mode == "iid" or (
        mode == "auto" and float(config.get("dirichlet_alpha", 0.1)) >= 10
    ):
        return generate_iid_split(
            config["dataset_name"], int(config["total_users"]), **arguments
        )
    if mode not in {"auto", "dirichlet"}:
        raise ValueError("Strict ProjRes supports iid or dirichlet splits.")
    return generate_dirichlet_split(
        config["dataset_name"],
        int(config["total_users"]),
        float(config["dirichlet_alpha"]),
        **arguments,
    )


def _metric_payload(
    labels: torch.Tensor,
    scores: torch.Tensor,
    residuals: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, float]:
    labels = labels.detach().cpu()
    scores = scores.detach().cpu()
    residuals = residuals.detach().cpu()
    predictions = predictions.detach().cpu()
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


@torch.no_grad()
def _candidate_features(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    parts = [
        model.encode_images(pixels[start : start + batch_size]).detach()
        for start in range(0, pixels.shape[0], batch_size)
    ]
    return torch.cat(parts)


@torch.no_grad()
def _encode_nonmember_pool(
    model: torch.nn.Module,
    processor,
    datasets: list,
    source_names: list[str],
    batch_size: int,
    max_nonmembers: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Stream a disjoint non-member pool through frozen CLIP exactly once."""
    if len(datasets) != len(source_names) or not datasets:
        raise ValueError("Non-member datasets must be non-empty and named.")

    def collate(batch):
        images, labels = zip(*batch)
        pixels = processor(images=list(images), return_tensors="pt")["pixel_values"]
        return pixels, torch.as_tensor(labels, dtype=torch.long)

    remaining = max_nonmembers
    feature_parts = []
    label_parts = []
    source_counts: dict[str, int] = {}
    model.eval()
    for dataset, source in zip(datasets, source_names):
        count = 0
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        for pixels, labels in loader:
            if remaining is not None:
                take = min(remaining, int(labels.numel()))
                pixels = pixels[:take]
                labels = labels[:take]
            features = model.encode_images(pixels.to(model.device))
            feature_parts.append(features.detach().cpu())
            label_parts.append(labels.detach().cpu())
            count += int(labels.numel())
            if remaining is not None:
                remaining -= int(labels.numel())
                if remaining == 0:
                    break
        source_counts[source] = count
        if remaining == 0:
            break
    if not label_parts:
        raise ValueError("The strict ProjRes non-member pool is empty.")
    return torch.cat(feature_parts), torch.cat(label_parts), source_counts


def _clone_trainable_model(model: torch.nn.Module) -> torch.nn.Module:
    model_type = str(getattr(model, "model_type", ""))
    if model_type == "clip_mlp":
        return model.clone_head_only()
    if model_type == "visual_adapter":
        return model.clone_adapter_only()
    raise ValueError(f"Unsupported ProjRes model type {model_type!r}.")


def _attacked_layer(
    model: torch.nn.Module,
) -> tuple[str, torch.nn.Linear]:
    model_type = str(getattr(model, "model_type", ""))
    if model_type == "clip_mlp":
        return "classifier.0.weight", model.classifier[0]
    if model_type == "visual_adapter":
        # This is the Adapter downsampling layer used in the paper's
        # derivation: dW has the input hidden representations in its row span.
        return "adapter.net.0.weight", model.adapter.net[0]
    raise ValueError(f"Unsupported ProjRes model type {model_type!r}.")


def _run_client(
    client_id: int,
    config: dict,
    train_sets,
    test_sets,
    processor,
    global_model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    max_candidates: int,
    min_nonmembers: int,
    max_nonmembers: int | None,
) -> dict[str, object]:
    seed = int(config["seed"])

    def collate(batch):
        images, labels = zip(*batch)
        processed = processor(images=list(images), return_tensors="pt")
        return processed["pixel_values"], torch.as_tensor(labels, dtype=torch.long)

    loader = DataLoader(
        train_sets[client_id],
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed + client_id),
        drop_last=False,
    )
    try:
        member_pixels, member_labels = next(iter(loader))
    except StopIteration as error:
        raise ValueError(f"Client {client_id} has no training batch.") from error
    member_count = min(int(member_labels.numel()), max_candidates)
    selected_member_pixels = member_pixels[:member_count]
    selected_member_labels = member_labels[:member_count]

    local_model = _clone_trainable_model(global_model).to(device)
    attacked_parameter, first_layer = _attacked_layer(local_model)
    member_features = _candidate_features(
        local_model,
        selected_member_pixels.to(device),
        int(config.get("eval_batch_size", 64)),
    )
    nonmember_datasets = [dataset for dataset in test_sets if len(dataset)]
    nonmember_source_names = [
        f"independent_test:{index}"
        for index, dataset in enumerate(test_sets)
        if len(dataset)
    ]
    nonmember_datasets.extend(
        dataset
        for index, dataset in enumerate(train_sets)
        if index != client_id and len(dataset)
    )
    nonmember_source_names.extend(
        f"other_client_train:{index}"
        for index, dataset in enumerate(train_sets)
        if index != client_id and len(dataset)
    )
    nonmember_features, nonmember_labels, source_counts = _encode_nonmember_pool(
        local_model,
        processor,
        nonmember_datasets,
        nonmember_source_names,
        int(config.get("eval_batch_size", 64)),
        max_nonmembers,
    )
    if nonmember_labels.numel() < min_nonmembers:
        raise ValueError(
            "Strict ProjRes needs at least "
            f"{min_nonmembers} non-members to resolve 0.1% FPR; found "
            f"{nonmember_labels.numel()}."
        )
    candidate_features = torch.cat(
        (member_features, nonmember_features.to(device))
    )
    # The update must include the entire actual batch, even if reporting caps
    # the number of member candidates.
    local_model.train()
    step = one_batch_fedsgd_step(
        local_model,
        member_pixels.to(device),
        member_labels.to(device),
        float(config["learning_rate"]),
        parameter_name=attacked_parameter,
    )
    attack = strict_mlp_projres(
        step.observed_update,
        candidate_features,
        threshold=threshold,
        # For dW = E.T @ X / batch, rank(dW) cannot exceed the actual
        # FedSGD batch size. This removes only float32 subtraction directions
        # that are impossible in the exact one-batch gradient.
        max_rank=int(member_labels.numel()),
    )
    labels = torch.cat(
        (
            torch.ones(member_count, dtype=torch.long, device=device),
            torch.zeros(
                nonmember_labels.numel(), dtype=torch.long, device=device
            ),
        )
    )
    metrics = _metric_payload(
        labels, attack.scores, attack.l1_residuals, attack.predictions
    )
    batch_size = int(member_labels.numel())
    input_dimension = int(first_layer.in_features)
    output_dimension = int(first_layer.out_features)
    logger.info(
        "client=%d | batch=%d | rank=%d | auc=%.4f | threshold_acc=%.4f",
        client_id,
        batch_size,
        int(attack.metadata["subspace"]["numerical_rank"]),
        metrics["auc"],
        metrics["threshold_accuracy"],
    )
    return {
        "client_id": client_id,
        "model_type": str(config["model_type"]),
        "threat_model": {
            "server": "honest-but-curious",
            "rounds_observed": 1,
            "local_batches": 1,
            "optimizer": "vanilla_sgd",
            "attacked_parameter": step.parameter_name,
            "member_definition": "present_in_the_observed_fedsgd_batch",
        },
        "dimensions": {
            "actual_batch_size": batch_size,
            "member_candidate_count": member_count,
            "nonmember_candidate_count": int(nonmember_labels.numel()),
            "input_dimension": input_dimension,
            "first_layer_output_dimension": output_dimension,
            "paper_favorable_rank_condition": bool(
                batch_size <= output_dimension and batch_size < input_dimension
            ),
        },
        "optimization": {
            "loss": step.loss,
            "learning_rate": float(config["learning_rate"]),
            "observed_update_norm": float(step.observed_update.norm()),
            "update_over_lr_vs_gradient_relative_error": (
                step.update_gradient_relative_error
            ),
        },
        "attack": {"metrics": metrics, "metadata": attack.metadata},
        "raw": {
            "labels": labels.detach().cpu().tolist(),
            "scores": attack.scores.detach().cpu().tolist(),
            "l1_residuals": attack.l1_residuals.detach().cpu().tolist(),
            "predictions": attack.predictions.detach().cpu().tolist(),
        },
        "candidate_controls": {
            "label_matched_nonmembers": False,
            "member_labels": selected_member_labels.tolist(),
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
        "experiment": f"strict_{model_type}_projres_all_clients",
        "model_type": model_type,
        "pooled_metrics": pooled,
        "client_macro_metrics": macro,
        "per_client": results,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    config = _resolved_config(args)
    _validate_strict_config(config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        args.device
        if args.device is not None
        else f"cuda:{int(config.get('gpu', 0))}"
        if torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")

    train_sets, test_sets, class_names = _split_data(config)
    processor, clip_model = _load_local_clip(config["cache_dir"], device)
    model_type = str(config["model_type"]).lower()
    if model_type == "clip_mlp":
        mlp_config = config["clip_mlp"]
        global_model = CLIPImageMLP(
            clip_model=clip_model,
            num_classes=len(class_names),
            hidden_dim=int(mlp_config["hidden_dim"]),
            dropout=float(mlp_config["dropout"]),
            normalize_features=bool(mlp_config.get("normalize_features", False)),
            device=device,
        ).to(device)
    else:
        adapter_config = config["visual_adapter"]
        text_features = build_visual_adapter_text_features(
            clip_model=clip_model,
            processor=processor,
            classnames=class_names,
            dataset_name=config["dataset_name"],
            device=device,
            template=adapter_config.get("template"),
        )
        global_model = VisualCLIPAdapter(
            clip_model=clip_model,
            text_features=text_features,
            classnames=class_names,
            feature_dim=int(adapter_config.get("feature_dim", 512)),
            reduction=int(adapter_config.get("reduction", 4)),
            alpha=float(adapter_config.get("alpha", 0.2)),
            output_relu=bool(adapter_config.get("output_relu", True)),
            text_adapter_enabled=bool(
                adapter_config.get("text_adapter_enabled", False)
            ),
            text_reduction=int(
                adapter_config.get(
                    "text_reduction", adapter_config.get("reduction", 4)
                )
            ),
            text_alpha=float(
                adapter_config.get(
                    "text_alpha", adapter_config.get("alpha", 0.2)
                )
            ),
            text_output_relu=bool(
                adapter_config.get(
                    "text_output_relu",
                    adapter_config.get("output_relu", True),
                )
            ),
            device=device,
        ).to(device)
    selected = (
        list(range(len(train_sets)))
        if str(args.target_client).lower() == "all"
        else [int(args.target_client)]
    )
    if any(client_id < 0 or client_id >= len(train_sets) for client_id in selected):
        raise ValueError("target-client is outside the configured client range.")
    results = []
    for client_id in selected:
        results.append(
            _run_client(
                client_id,
                config,
                train_sets,
                test_sets,
                processor,
                global_model,
                device,
                args.threshold,
                args.max_candidates,
                args.min_nonmembers,
                None if args.max_nonmembers == 0 else args.max_nonmembers,
            )
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(results) == 1:
        return {
            "experiment": f"strict_{model_type}_projres_single_client",
            "model_type": model_type,
            "dataset_name": config["dataset_name"],
            "seed": seed,
            "device": str(device),
            "result": results[0],
        }
    output = _aggregate(results, model_type)
    output.update(
        {
            "dataset_name": config["dataset_name"],
            "seed": seed,
            "device": str(device),
        }
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/clip_mlp_projres.yaml")
    )
    parser.add_argument("--dataset-name")
    parser.add_argument("--data-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--target-client", default="all")
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--min-nonmembers", type=int, default=1000)
    parser.add_argument(
        "--max-nonmembers",
        type=int,
        default=20000,
        help="Maximum non-members, or 0 to use the complete disjoint pool.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.threshold < 0
        or args.max_candidates <= 0
        or args.min_nonmembers < 1000
        or args.max_nonmembers < 0
        or (args.max_nonmembers and args.max_nonmembers < args.min_nonmembers)
    ):
        parser.error(
            "threshold must be non-negative; max-candidates must be positive; "
            "min-nonmembers must be at least 1000; max-nonmembers must be 0 "
            "or no smaller than min-nonmembers"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        logger.info("Saved strict ProjRes results to %s", args.output)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
