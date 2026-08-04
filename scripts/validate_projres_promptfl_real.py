#!/usr/bin/env python3
"""Validate Oracle and prompt-observable ProjRes variants on local data."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import gc
import json
import logging
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from main import _load_local_clip, default_config
from privacy_attacks.metrics import membership_metrics
from privacy_attacks.projres_promptfl import (
    numerical_rank,
    principal_angles,
    projection_statistics,
    prompt_gradient_fingerprints,
    prompt_vjp,
    ridge_lift_matrix_free,
    row_subspace_basis,
    text_feature_change_subspace,
)
from privacy_attacks.promptres import positive_cosine_squared
from trainmodel.custom_clip import CustomCLIP, get_default_prompt_template
from utils.data_loader import generate_dirichlet_split, generate_iid_split


logger = logging.getLogger("projres_promptfl_validation")


def _load_config(path: Path) -> dict:
    config = default_config()
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    audit = config["audit"] | loaded.pop("audit", {})
    defense = config["defense"] | loaded.pop("defense", {})
    config.update(loaded)
    config["audit"] = audit
    config["defense"] = defense
    config["aggregator"] = "promptfl"
    return config


def _resolved_config(args: argparse.Namespace) -> dict:
    config = _load_config(args.config)
    for name in (
        "dataset_name",
        "data_root",
        "cache_dir",
        "batch_size",
        "learning_rate",
        "fpl_shots",
        "dirichlet_alpha",
        "n_ctx",
        "seed",
    ):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
    return config


def _split_data(config: dict):
    arguments = {
        "root_dir": config["data_root"],
        "fpl": True,
        "fpl_shots": config.get("fpl_shots"),
        "use_full_dataset": bool(config.get("use_full_dataset", False)),
    }
    mode = str(config.get("partition_mode", "auto")).lower()
    if mode == "iid" or (
        mode == "auto" and float(config.get("dirichlet_alpha", 0.1)) >= 10
    ):
        return generate_iid_split(
            config["dataset_name"], int(config["total_users"]), **arguments
        )
    if mode not in {"auto", "dirichlet"}:
        raise ValueError(
            "This validator supports iid or dirichlet partitions; "
            f"received {mode!r}."
        )
    return generate_dirichlet_split(
        config["dataset_name"],
        int(config["total_users"]),
        float(config["dirichlet_alpha"]),
        **arguments,
    )


def _collect_label_matched_nonmembers(
    independent_test_dataset,
    other_client_train_dataset,
    member_labels: torch.Tensor,
) -> tuple[list, torch.Tensor, dict[str, int]]:
    needed = Counter(int(value) for value in member_labels.tolist())
    collected: dict[int, list] = {label: [] for label in needed}
    source_counts = {"independent_test": 0, "other_client_train": 0}

    def collect_from(dataset, source: str) -> None:
        for image, label in dataset:
            value = int(label)
            if value in needed and len(collected[value]) < needed[value]:
                collected[value].append(image)
                source_counts[source] += 1
            if all(
                len(collected[label]) >= count
                for label, count in needed.items()
            ):
                break

    collect_from(independent_test_dataset, "independent_test")
    if any(
        len(collected[label]) < count for label, count in needed.items()
    ):
        collect_from(other_client_train_dataset, "other_client_train")
    missing = {
        label: needed[label] - len(collected[label])
        for label in needed
        if len(collected[label]) < needed[label]
    }
    if missing:
        raise ValueError(
            "Not enough label-matched nonmembers across independent test and "
            f"other-client training pools: {missing}"
        )
    offsets = Counter()
    images = []
    labels = []
    for label in member_labels.tolist():
        value = int(label)
        images.append(collected[value][offsets[value]])
        labels.append(value)
        offsets[value] += 1
    return images, torch.tensor(labels, dtype=torch.long), source_counts


@torch.no_grad()
def _encode_images(
    model: CustomCLIP,
    pixel_values: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, pixel_values.shape[0], batch_size):
        features = model.clip_model.get_image_features(
            pixel_values=pixel_values[start : start + batch_size]
        )
        parts.append(F.normalize(features, dim=1).detach())
    return torch.cat(parts)


def _metric_payload(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    metrics = membership_metrics(labels, scores)
    members = scores[labels == 1]
    nonmembers = scores[labels == 0]
    return {
        **metrics,
        "member_mean_score": float(members.mean()),
        "nonmember_mean_score": float(nonmembers.mean()),
        "score_gap": float(members.mean() - nonmembers.mean()),
    }


def run_validation(args: argparse.Namespace) -> dict[str, object]:
    config = _resolved_config(args)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(
            f"cuda:{int(config.get('gpu', 0))}"
            if torch.cuda.is_available()
            else "cpu"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")

    train_sets, test_sets, class_names = _split_data(config)
    target_client = int(args.target_client)
    if not 0 <= target_client < len(train_sets):
        raise ValueError("target-client is outside the configured client range.")
    logger.info(
        "client=%d | train_samples=%d | loading local CLIP with eager attention",
        target_client,
        len(train_sets[target_client]),
    )
    # torch.func.jvp requires forward-mode AD. PyTorch's efficient SDPA kernel
    # does not implement it, while Transformers' mathematically equivalent
    # eager attention path is composed of forward-AD-compatible operations.
    processor, clip_model = _load_local_clip(
        config["cache_dir"], device, attn_implementation="eager"
    )
    model = CustomCLIP(
        clip_model=clip_model,
        processor=processor,
        classnames=class_names,
        n_ctx=int(config["n_ctx"]),
        template=get_default_prompt_template(config["dataset_name"]),
        class_specific_ctx=bool(config.get("class_specific_ctx", False)),
        parameterization="promptfl",
        device=device,
    ).to(device)
    model.eval()

    def collate(batch):
        images, labels = zip(*batch)
        processed = processor(images=list(images), return_tensors="pt")
        return processed["pixel_values"], torch.as_tensor(labels, dtype=torch.long)

    loader_generator = torch.Generator().manual_seed(seed + target_client)
    train_loader = DataLoader(
        train_sets[target_client],
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        generator=loader_generator,
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise ValueError("The target client has no training batches.")

    prompt_parameter = model.prompt_learner.ctx
    base_prompt = prompt_parameter.detach().clone()
    learning_rate = float(config["learning_rate"])
    accumulated_text_gradient = None
    member_pixels = []
    member_labels = []
    member_features = []
    step_losses = []
    error_ranks = []
    iterator = iter(train_loader)
    for step_index in range(args.local_steps):
        try:
            pixels, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            pixels, labels = next(iterator)
        pixels = pixels.to(device)
        labels = labels.to(device)
        logits, image_features, text_features = model(
            pixels, return_intermediate=True
        )
        loss = F.cross_entropy(logits, labels)
        text_gradient, prompt_gradient = torch.autograd.grad(
            loss, (text_features, prompt_parameter)
        )
        errors = torch.softmax(logits.detach(), dim=1) - F.one_hot(
            labels, num_classes=len(class_names)
        ).to(dtype=logits.dtype)
        error_ranks.append(numerical_rank(errors)[0])
        accumulated_text_gradient = (
            text_gradient.detach().clone()
            if accumulated_text_gradient is None
            else accumulated_text_gradient + text_gradient.detach()
        )
        member_pixels.append(pixels.detach().cpu())
        member_labels.append(labels.detach().cpu())
        member_features.append(F.normalize(image_features.detach(), dim=1))
        step_losses.append(float(loss.detach()))
        logger.info(
            "client=%d | local_step=%d/%d | batch=%d | loss=%.6f | error_rank=%d",
            target_client,
            step_index + 1,
            args.local_steps,
            labels.numel(),
            float(loss.detach()),
            error_ranks[-1],
        )
        with torch.no_grad():
            prompt_parameter.add_(prompt_gradient, alpha=-learning_rate)

    if accumulated_text_gradient is None:
        raise AssertionError("Local validation did not produce a text gradient.")
    trained_prompt = prompt_parameter.detach().clone()
    observed_prompt_gradient = (base_prompt - trained_prompt) / learning_rate
    all_member_pixels = torch.cat(member_pixels)
    all_member_labels = torch.cat(member_labels)
    all_member_features = torch.cat(member_features)
    local_training_examples = int(all_member_labels.numel())
    candidate_count = min(args.max_candidates, all_member_labels.numel())
    all_member_pixels = all_member_pixels[:candidate_count]
    all_member_labels = all_member_labels[:candidate_count]
    all_member_features = all_member_features[:candidate_count]

    other_client_training = ConcatDataset(
        [
            dataset
            for client_id, dataset in enumerate(train_sets)
            if client_id != target_client
        ]
    )
    (
        raw_nonmembers,
        nonmember_labels,
        nonmember_source_counts,
    ) = _collect_label_matched_nonmembers(
        ConcatDataset(test_sets),
        other_client_training,
        all_member_labels,
    )
    nonmember_pixels = processor(
        images=raw_nonmembers, return_tensors="pt"
    )["pixel_values"]
    candidate_pixels = torch.cat((all_member_pixels, nonmember_pixels)).to(device)
    candidate_labels = torch.cat((all_member_labels, nonmember_labels)).to(device)
    candidate_features = _encode_images(
        model, candidate_pixels, int(config.get("eval_batch_size", 64))
    )
    membership = torch.cat(
        (
            torch.ones(candidate_count, dtype=torch.long),
            torch.zeros(candidate_count, dtype=torch.long),
        )
    )

    with torch.no_grad():
        prompt_parameter.copy_(base_prompt)

    def text_feature_function(context: torch.Tensor) -> torch.Tensor:
        return model.get_text_features_from_context(context, normalize=True)

    oracle_basis, oracle_metadata = row_subspace_basis(
        accumulated_text_gradient
    )
    candidate_stats = projection_statistics(candidate_features, oracle_basis)
    oracle_metrics = _metric_payload(
        membership, candidate_stats["projection_energy"].detach().cpu()
    )
    member_basis, member_metadata = row_subspace_basis(all_member_features)
    oracle_angles = principal_angles(oracle_basis, member_basis)

    base_vjp = prompt_vjp(
        text_feature_function, base_prompt, accumulated_text_gradient
    )
    local_drift_relative_error = float(
        (base_vjp - observed_prompt_gradient).norm()
        / observed_prompt_gradient.norm().clamp_min(1e-12)
    )

    lifted_metrics = None
    lifted_metadata = None
    lift_diagnostics = None
    lifted_oracle_angles = None
    if not args.skip_lift:
        logger.info(
            "client=%d | starting matrix-free Jacobian lift "
            "(ridge=%g, max_iterations=%d, tolerance=%g)",
            target_client,
            args.ridge,
            args.lift_iterations,
            args.lift_tolerance,
        )
        lifted_gradient, diagnostics = ridge_lift_matrix_free(
            text_feature_function,
            base_prompt,
            observed_prompt_gradient,
            ridge=args.ridge,
            max_iterations=args.lift_iterations,
            tolerance=args.lift_tolerance,
        )
        lifted_basis, lifted_metadata = row_subspace_basis(lifted_gradient)
        lifted_stats = projection_statistics(candidate_features, lifted_basis)
        lifted_metrics = _metric_payload(
            membership, lifted_stats["projection_energy"].detach().cpu()
        )
        lifted_oracle_angles = principal_angles(lifted_basis, oracle_basis)
        lift_diagnostics = asdict(diagnostics)
        logger.info(
            "client=%d | lift finished | iterations=%d | converged=%s | "
            "measurement_residual=%.6g",
            target_client,
            diagnostics.iterations,
            diagnostics.converged,
            diagnostics.measurement_relative_residual,
        )

    fingerprints, base_text_features, candidate_losses = (
        prompt_gradient_fingerprints(
            text_feature_function,
            base_prompt,
            candidate_features,
            candidate_labels,
            logit_scale=float(model.clip_model.logit_scale.detach().exp()),
        )
    )
    direct_scores = positive_cosine_squared(
        observed_prompt_gradient.flatten(), fingerprints
    )
    direct_metrics = _metric_payload(
        membership, direct_scores.detach().cpu()
    )

    # Both prompt endpoints are visible to an honest-but-curious server.  The
    # server can therefore recompute their text matrices locally without
    # observing the private batches.  We remove the class-common component
    # before SVD because valid cross-entropy text gradients have zero row sum.
    with torch.no_grad():
        trained_text_features = text_feature_function(trained_prompt)
    delta_rank_cap = min(
        local_training_examples,
        len(class_names) - 1,
        int(base_text_features.shape[1]) - 1,
    )
    text_feature_delta, delta_basis, delta_metadata = (
        text_feature_change_subspace(
            base_text_features,
            trained_text_features,
            max_rank=delta_rank_cap,
        )
    )
    delta_stats = projection_statistics(candidate_features, delta_basis)
    delta_metrics = _metric_payload(
        membership, delta_stats["projection_energy"].detach().cpu()
    )
    delta_oracle_angles = principal_angles(delta_basis, oracle_basis)
    delta_member_angles = principal_angles(delta_basis, member_basis)
    logger.info(
        "client=%d | text-delta proxy | raw_rank=%d | centered_rank=%d | "
        "used_rank=%d | rank_cap=%d | removed_common=%.4f | auc=%.4f | "
        "oracle_max_angle_deg=%.4f",
        target_client,
        int(delta_metadata["raw_numerical_rank"]),
        int(delta_metadata["numerical_rank"]),
        int(delta_metadata["used_rank"]),
        delta_rank_cap,
        float(delta_metadata["removed_common_mode_fraction"]),
        float(delta_metrics["auc"]),
        (
            float(torch.rad2deg(delta_oracle_angles).max())
            if delta_oracle_angles.numel()
            else float("nan")
        ),
    )

    return {
        "experiment": "real_promptfl_projres_validation",
        "config": {
            "dataset_name": config["dataset_name"],
            "target_client": target_client,
            "clients": len(train_sets),
            "classes": len(class_names),
            "feature_dimension": int(base_text_features.shape[1]),
            "prompt_shape": list(base_prompt.shape),
            "batch_size": int(config["batch_size"]),
            "local_steps": args.local_steps,
            "learning_rate": learning_rate,
            "candidate_count_per_group": candidate_count,
            "label_matched_nonmembers": True,
            "device": str(device),
            "seed": seed,
        },
        "rank_boundary": {
            "theoretical_max_batch": min(
                len(class_names) - 1, int(base_text_features.shape[1]) - 1
            ),
            "step_error_ranks": error_ranks,
            "distinct_training_rows": int(member_metadata["numerical_rank"]),
            "oracle_gradient": oracle_metadata,
            "lifted_gradient": lifted_metadata,
            "text_feature_delta": delta_metadata,
            "text_feature_delta_rank_cap": delta_rank_cap,
        },
        "optimization": {
            "step_losses": step_losses,
            "observed_prompt_gradient_norm": float(observed_prompt_gradient.norm()),
            "fixed_base_jacobian_relative_error": local_drift_relative_error,
        },
        "subspace": {
            "oracle_to_member_max_angle_degrees": (
                float(torch.rad2deg(oracle_angles).max())
                if oracle_angles.numel()
                else None
            ),
            "lifted_to_oracle_max_angle_degrees": (
                float(torch.rad2deg(lifted_oracle_angles).max())
                if lifted_oracle_angles is not None
                and lifted_oracle_angles.numel()
                else None
            ),
            "text_delta_to_oracle_max_angle_degrees": (
                float(torch.rad2deg(delta_oracle_angles).max())
                if delta_oracle_angles.numel()
                else None
            ),
            "text_delta_to_member_max_angle_degrees": (
                float(torch.rad2deg(delta_member_angles).max())
                if delta_member_angles.numel()
                else None
            ),
        },
        "attacks": {
            "oracle_projres_t": oracle_metrics,
            "delta_text_projres": delta_metrics,
            "lifted_projres_p": lifted_metrics,
            "direct_prompt_atom": direct_metrics,
        },
        "raw_scores": {
            "labels": membership.tolist(),
            "oracle_projres_t": candidate_stats["projection_energy"]
            .detach()
            .cpu()
            .tolist(),
            "delta_text_projres": delta_stats["projection_energy"]
            .detach()
            .cpu()
            .tolist(),
            "lifted_projres_p": (
                None
                if lifted_metrics is None
                else lifted_stats["projection_energy"].detach().cpu().tolist()
            ),
            "direct_prompt_atom": direct_scores.detach().cpu().tolist(),
        },
        "text_feature_delta": {
            "shape": list(text_feature_delta.shape),
            "definition": "row_centered_T_after_minus_T_before",
            "normalized_text_features": True,
            "zero_row_sum_projected": True,
        },
        "lift": lift_diagnostics,
        "candidate_controls": {
            "member_mean_loss_at_base": float(
                candidate_losses[:candidate_count].mean()
            ),
            "nonmember_mean_loss_at_base": float(
                candidate_losses[candidate_count:].mean()
            ),
            "member_labels": all_member_labels.tolist(),
            "nonmember_labels": nonmember_labels.tolist(),
            "nonmember_source_counts": nonmember_source_counts,
        },
    }


def _aggregate_client_results(
    client_results: list[dict[str, object]],
) -> dict[str, object]:
    if not client_results:
        raise ValueError("At least one client result is required.")
    attack_names = (
        "oracle_projres_t",
        "delta_text_projres",
        "lifted_projres_p",
        "direct_prompt_atom",
    )
    pooled = {}
    macro = {}
    for attack in attack_names:
        metric_rows = [
            result["attacks"][attack]  # type: ignore[index]
            for result in client_results
            if result["attacks"][attack] is not None  # type: ignore[index]
        ]
        score_rows = [
            result["raw_scores"][attack]  # type: ignore[index]
            for result in client_results
            if result["raw_scores"][attack] is not None  # type: ignore[index]
        ]
        if not metric_rows or len(metric_rows) != len(client_results):
            pooled[attack] = None
            macro[attack] = None
            continue
        labels = torch.cat(
            [
                torch.tensor(
                    result["raw_scores"]["labels"], dtype=torch.long  # type: ignore[index]
                )
                for result in client_results
            ]
        )
        scores = torch.cat(
            [torch.tensor(row, dtype=torch.float64) for row in score_rows]
        )
        pooled[attack] = _metric_payload(labels, scores)
        numeric_keys = metric_rows[0].keys()
        macro[attack] = {
            key: sum(float(row[key]) for row in metric_rows) / len(metric_rows)
            for key in numeric_keys
        }
    client_ids = [
        int(result["config"]["target_client"])  # type: ignore[index]
        for result in client_results
    ]
    return {
        "experiment": "real_promptfl_projres_all_clients_validation",
        "dataset_name": client_results[0]["config"]["dataset_name"],  # type: ignore[index]
        "client_ids": client_ids,
        "client_count": len(client_ids),
        "pooled_attacks": pooled,
        "client_macro_attacks": macro,
        "per_client": client_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/federated_prompt_paper.yaml"),
    )
    parser.add_argument("--dataset-name")
    parser.add_argument("--data-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--fpl-shots", type=int)
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--n-ctx", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--target-client",
        default="all",
        help="Client id for a focused diagnostic, or 'all' for pooled auditing.",
    )
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--lift-iterations", type=int, default=20)
    parser.add_argument("--lift-tolerance", type=float, default=1e-5)
    parser.add_argument("--skip-lift", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.local_steps <= 0 or args.max_candidates <= 0:
        parser.error("local-steps and max-candidates must be positive")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    resolved = _resolved_config(args)
    selected_clients = (
        list(range(int(resolved["total_users"])))
        if str(args.target_client).lower() == "all"
        else [args.target_client]
    )
    logger.info("Starting PromptFL ProjRes validation")
    logger.info(
        "config=%s | dataset=%s | data_root=%s | cache_dir=%s",
        args.config,
        resolved["dataset_name"],
        resolved["data_root"],
        resolved["cache_dir"],
    )
    logger.info(
        "partition=%s | full_dataset=%s | clients=%s | batch_size=%d | "
        "learning_rate=%g | n_ctx=%d | seed=%d",
        resolved.get("partition_mode", "auto"),
        bool(resolved.get("use_full_dataset", False)),
        selected_clients,
        int(resolved["batch_size"]),
        float(resolved["learning_rate"]),
        int(resolved["n_ctx"]),
        int(resolved["seed"]),
    )
    logger.info(
        "local_steps=%d | max_candidates_per_group=%d | ridge=%g | "
        "lift_iterations=%d | lift_tolerance=%g | skip_lift=%s",
        args.local_steps,
        args.max_candidates,
        args.ridge,
        args.lift_iterations,
        args.lift_tolerance,
        args.skip_lift,
    )
    if str(args.target_client).lower() == "all":
        client_results = []
        total_clients = int(resolved["total_users"])
        for client_id in range(total_clients):
            logger.info(
                "client=%d | starting audit (%d/%d)",
                client_id,
                client_id + 1,
                total_clients,
            )
            client_args = copy.copy(args)
            client_args.target_client = client_id
            client_result = run_validation(client_args)
            client_results.append(client_result)
            attacks = client_result["attacks"]
            logger.info(
                "client=%d | completed | oracle_auc=%.4f | delta_auc=%.4f | "
                "lifted_auc=%s | direct_auc=%.4f",
                client_id,
                float(attacks["oracle_projres_t"]["auc"]),
                float(attacks["delta_text_projres"]["auc"]),
                (
                    "skipped"
                    if attacks["lifted_projres_p"] is None
                    else f"{float(attacks['lifted_projres_p']['auc']):.4f}"
                ),
                float(attacks["direct_prompt_atom"]["auc"]),
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = _aggregate_client_results(client_results)
    else:
        try:
            args.target_client = int(args.target_client)
        except ValueError:
            parser.error("target-client must be a non-negative integer or 'all'")
        if args.target_client < 0:
            parser.error("target-client must be non-negative")
        result = run_validation(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        logger.info("Saved validation results to %s", args.output)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
