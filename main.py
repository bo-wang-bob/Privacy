from __future__ import annotations

import argparse
import copy
import datetime
import logging
import math
import os
import random

import numpy as np
import torch
import yaml

from aggregator.aggregator_builder import build_aggregator
from privacy_attacks.auditor import POOLED_CLIENT_ATTACKS, SUPPORTED_ATTACKS
from privacy_defenses import FEDMIA_BASELINE_DEFENSES, SUPPORTED_DEFENSES
from servers.serverbase import ServerBase
from utils.data_loader import (
    generate_dirichlet_split,
    generate_iid_split,
    generate_pathological_split,
)

logger = logging.getLogger(__name__)

LAUNCHER_LOG_CAPTURE_ENV = "FEDMIA_LAUNCHER_LOG_CAPTURE"


def _build_logging_handlers(result_dir: str) -> list[logging.Handler]:
    """Use one canonical log file while keeping direct runs self-contained."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if os.environ.get(LAUNCHER_LOG_CAPTURE_ENV) != "1":
        handlers.insert(
            0,
            logging.FileHandler(
                os.path.join(result_dir, "run.log"), encoding="utf-8"
            ),
        )
    return handlers


def _format_privacy_audit_summary(summaries: list[dict]) -> str:
    """Format the final audit log without dumping full result metadata."""
    if not summaries:
        return "Privacy audit completed: no attack results."

    lines = ["Privacy audit completed:"]
    for summary in summaries:
        reportable = summary.get("reportable_metrics", {})
        auc = reportable.get("auc", summary.get("auc"))
        primary_metric = summary.get("primary_metric", "tpr_at_fpr_0.01")
        primary_score = summary.get("primary_score")
        if primary_score is None:
            primary_score = reportable.get(primary_metric)

        auc_text = "n/a" if auc is None else f"{float(auc):.4f}"
        primary_text = (
            "n/a"
            if primary_score is None
            else f"{100.0 * float(primary_score):.2f}%"
        )
        primary_label = {
            "tpr_at_fpr_0.1": "TPR@10%FPR",
            "tpr_at_fpr_0.01": "TPR@1%FPR",
            "tpr_at_fpr_0.001": "TPR@0.1%FPR",
        }.get(primary_metric, primary_metric)
        lines.append(
            "  %s | AUC=%s | %s=%s | samples=%s (members=%s, non-members=%s)"
            % (
                summary.get("attack", "unknown"),
                auc_text,
                primary_label,
                primary_text,
                summary.get("num_samples", "n/a"),
                summary.get("member_count", "n/a"),
                summary.get("nonmember_count", "n/a"),
            )
        )
    return "\n".join(lines)


def _result_run_id(
    now: datetime.datetime | None = None, process_id: int | None = None
) -> str:
    now = datetime.datetime.now() if now is None else now
    process_id = os.getpid() if process_id is None else int(process_id)
    return now.strftime("%Y%m%d_%H%M%S_%f") + f"_{process_id}"


def _result_directory(config: dict) -> str:
    """Return the canonical output directory for a direct or sweep run."""
    if bool(config.get("results_dir_is_run_dir", False)):
        return str(config["results_dir"])
    timestamp = _result_run_id()
    configured_attacks = list(config.get("audit", {}).get("attacks", []))
    if (
        not bool(config.get("audit", {}).get("enabled", True))
        or not configured_attacks
    ):
        attack_label = "no_attack"
    elif len(configured_attacks) == 1:
        attack_label = configured_attacks[0]
    else:
        attack_label = "multi_attack"
    defense_label = str(config.get("defense", {}).get("name", "none")).lower()
    return os.path.join(
        config["results_dir"],
        f"{config['dataset_name']}_{config['aggregator']}_{attack_label}_{defense_label}_{timestamp}",
    )


def _load_local_clip(
    cache_dir: str,
    device: torch.device,
    attn_implementation: str | None = None,
):
    from transformers import CLIPModel, CLIPProcessor

    try:
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            use_fast=True,
            local_files_only=True,
        )
        model_arguments = {
            "cache_dir": cache_dir,
            "local_files_only": True,
        }
        if attn_implementation is not None:
            model_arguments["attn_implementation"] = attn_implementation
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            **model_arguments,
        ).to(device)
    except OSError as error:
        raise RuntimeError(
            "Federated prompt tuning requires a local openai/clip-vit-base-patch32 "
            f"cache under {cache_dir!r}."
        ) from error
    return processor, clip_model


def validate_config(config: dict) -> None:
    model_type = str(config.get("model_type", "prompt")).lower()
    if model_type not in {
        "prompt",
        "clip_mlp",
        "visual_adapter",
        "clip_lora",
    }:
        raise ValueError(
            "model_type must be prompt, clip_mlp, visual_adapter, or clip_lora."
        )
    method = config["aggregator"].lower()
    supported_methods = {"fedavg", "fedsgd", "promptfl"}
    if method not in supported_methods:
        raise ValueError("aggregator must be fedavg, fedsgd, or promptfl.")
    if config["total_users"] <= 1:
        raise ValueError("total_users must be greater than one.")
    if not 1 <= config["sample_users"] <= config["total_users"]:
        raise ValueError("sample_users must be in [1, total_users].")
    if float(config.get("learning_rate", 0.0)) <= 0:
        raise ValueError("learning_rate must be positive.")
    learning_rate_decay = float(config.get("learning_rate_decay", 1.0))
    if not 0 < learning_rate_decay <= 1:
        raise ValueError("learning_rate_decay must be in (0, 1].")
    learning_rate_decay_interval = int(
        config.get("learning_rate_decay_interval", 1)
    )
    if learning_rate_decay_interval <= 0:
        raise ValueError("learning_rate_decay_interval must be positive.")
    aggregation_weighting = str(
        config.get("aggregation_weighting", "sample_count")
    ).lower()
    if aggregation_weighting not in {"sample_count", "uniform"}:
        raise ValueError(
            "aggregation_weighting must be sample_count or uniform."
        )
    partition_mode = str(config.get("partition_mode", "auto")).lower()
    if partition_mode not in {"auto", "dirichlet", "iid", "pathological"}:
        raise ValueError(
            "partition_mode must be auto, dirichlet, iid, or pathological."
        )
    fpl_shots = config.get("fpl_shots")
    if fpl_shots is not None and int(fpl_shots) <= 0:
        raise ValueError("fpl_shots must be positive when set.")
    if bool(config.get("use_full_dataset", False)) and fpl_shots is not None:
        raise ValueError(
            "use_full_dataset=true requires fpl_shots=null so training is not capped."
        )
    if model_type == "clip_mlp":
        mlp_config = dict(config.get("clip_mlp", {}))
        if method != "fedavg":
            raise ValueError("clip_mlp requires aggregator=fedavg.")
        if not bool(config.get("use_full_dataset", False)) or fpl_shots is not None:
            raise ValueError(
                "clip_mlp uses the full dataset and requires "
                "use_full_dataset=true and fpl_shots=null."
            )
        if int(mlp_config.get("hidden_dim", 512)) <= 0:
            raise ValueError("clip_mlp.hidden_dim must be positive.")
        if not 0 <= float(mlp_config.get("dropout", 0.0)) < 1:
            raise ValueError("clip_mlp.dropout must be in [0, 1).")
        if int(mlp_config.get("precompute_batch_size", 64)) <= 0:
            raise ValueError("clip_mlp.precompute_batch_size must be positive.")
    if model_type == "visual_adapter":
        adapter_config = dict(config.get("visual_adapter", {}))
        if method not in {"fedavg", "fedsgd"}:
            raise ValueError(
                "visual_adapter requires aggregator=fedavg or fedsgd."
            )
        if method == "fedsgd" and int(config.get("local_epochs", 1)) != 1:
            raise ValueError(
                "visual_adapter FedSGD requires local_epochs=1 because each "
                "client performs exactly one mini-batch step per round."
            )
        if bool(config.get("use_full_dataset", False)) or fpl_shots != 16:
            raise ValueError(
                "visual_adapter requires the FPL-style 16-shot setting: "
                "use_full_dataset=false and fpl_shots=16."
            )
        reduction = int(adapter_config.get("reduction", 4))
        if reduction <= 0:
            raise ValueError("visual_adapter.reduction must be positive.")
        feature_dim = int(adapter_config.get("feature_dim", 512))
        if feature_dim <= 0 or feature_dim % reduction != 0:
            raise ValueError(
                "visual_adapter.feature_dim must be positive and divisible "
                "by visual_adapter.reduction."
            )
        alpha = float(adapter_config.get("alpha", 0.2))
        if not 0 <= alpha <= 1:
            raise ValueError("visual_adapter.alpha must be in [0, 1].")
        text_reduction = int(adapter_config.get("text_reduction", reduction))
        if text_reduction <= 0 or feature_dim % text_reduction != 0:
            raise ValueError(
                "visual_adapter.feature_dim must be divisible by "
                "visual_adapter.text_reduction."
            )
        text_alpha = float(adapter_config.get("text_alpha", alpha))
        if not 0 <= text_alpha <= 1:
            raise ValueError("visual_adapter.text_alpha must be in [0, 1].")
        if int(adapter_config.get("precompute_batch_size", 64)) <= 0:
            raise ValueError(
                "visual_adapter.precompute_batch_size must be positive."
            )
    if model_type == "clip_lora":
        lora_config = dict(config.get("clip_lora", {}))
        if method not in {"fedavg", "fedsgd"}:
            raise ValueError("clip_lora requires factor-wise FedAvg or FedSGD.")
        if method == "fedsgd" and int(config["local_epochs"]) != 1:
            raise ValueError("clip_lora FedSGD requires local_epochs=1.")
        if bool(config.get("use_full_dataset", False)) or fpl_shots != 16:
            raise ValueError(
                "clip_lora requires the 16-shot setting: "
                "use_full_dataset=false and fpl_shots=16."
            )
        if str(lora_config.get("encoder", "both")).lower() not in {
            "vision",
            "text",
            "both",
        }:
            raise ValueError("clip_lora.encoder must be vision, text, or both.")
        targets = list(lora_config.get("target_modules", ["q", "k", "v"]))
        if not targets or set(targets) - {
            "q",
            "k",
            "v",
            "o",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        }:
            raise ValueError(
                "clip_lora.target_modules must contain q, k, v, or o."
            )
        if int(lora_config.get("rank", 2)) <= 0:
            raise ValueError("clip_lora.rank must be positive.")
        if float(lora_config.get("alpha", 1.0)) <= 0:
            raise ValueError("clip_lora.alpha must be positive.")
        if not 0 <= float(lora_config.get("dropout", 0.25)) < 1:
            raise ValueError("clip_lora.dropout must be in [0, 1).")
        if str(lora_config.get("scaling", "sqrt_rank")).lower() not in {
            "rank",
            "sqrt_rank",
        }:
            raise ValueError("clip_lora.scaling must be rank or sqrt_rank.")
        layers = lora_config.get("layers", "all")
        if isinstance(layers, str) and layers.lower() not in {
            "all",
            "last_half",
        }:
            raise ValueError("clip_lora.layers must be all, last_half, or a list.")
    if float(config.get("dirichlet_alpha", 0.1)) <= 0:
        raise ValueError("dirichlet_alpha must be positive.")
    audit = config.get("audit", {})
    if not 0 <= int(audit.get("target_client_id", 0)) < config["total_users"]:
        raise ValueError("audit.target_client_id must identify an existing client.")
    audit_client_ids = audit.get("audit_client_ids")
    if audit_client_ids is not None:
        if isinstance(audit_client_ids, str):
            if audit_client_ids.lower() != "all":
                raise ValueError("audit.audit_client_ids must be 'all' or a list.")
        elif isinstance(audit_client_ids, list):
            normalized_ids = [int(value) for value in audit_client_ids]
            if (
                not normalized_ids
                or len(set(normalized_ids)) != len(normalized_ids)
                or min(normalized_ids) < 0
                or max(normalized_ids) >= config["total_users"]
            ):
                raise ValueError(
                    "audit.audit_client_ids must contain unique existing clients."
                )
        else:
            raise ValueError("audit.audit_client_ids must be 'all' or a list.")
    if str(audit.get("audit_view", "protocol_plus_released_prompts")).lower() not in {
        "protocol_plus_released_prompts",
        "protocol_plus_queries",
        "full_whitebox",
        "released_prompt",
    }:
        raise ValueError("Unknown audit.audit_view.")
    attacks = set(audit.get("attacks", []))
    unknown = sorted(attacks - SUPPORTED_ATTACKS)
    if unknown:
        raise ValueError(f"Unknown membership attacks: {', '.join(unknown)}")
    if audit.get("enabled", True) and not attacks:
        raise ValueError("audit.enabled=true requires at least one membership attack.")
    pooled_audit = audit_client_ids == "all" or (
        isinstance(audit_client_ids, list) and len(audit_client_ids) > 1
    )
    if bool(audit.get("ensure_target_participation", True)):
        required_audit_clients = (
            config["total_users"]
            if audit_client_ids == "all"
            else len(audit_client_ids)
            if isinstance(audit_client_ids, list)
            else 1
        )
        if config["sample_users"] < required_audit_clients:
            raise ValueError(
                "audit.ensure_target_participation requires sample_users to be "
                "at least the number of audited clients."
            )
    candidate_sampling = str(audit.get("candidate_sampling", "legacy")).lower()
    if candidate_sampling not in {
        "legacy",
        "fedmia_mix",
        "low_fpr_full",
        "balanced_holdout",
    }:
        raise ValueError(
            "audit.candidate_sampling must be legacy, fedmia_mix, "
            "low_fpr_full, or balanced_holdout."
        )
    nonmember_ratio = float(audit.get("nonmember_to_member_ratio", 1.0))
    if nonmember_ratio <= 0:
        raise ValueError(
            "audit.nonmember_to_member_ratio must be positive."
        )
    if candidate_sampling == "fedmia_mix" and bool(
        audit.get("match_candidate_labels", False)
    ):
        raise ValueError(
            "audit.candidate_sampling=fedmia_mix requires "
            "match_candidate_labels=false to reproduce the reference protocol."
        )
    if model_type == "clip_lora" and candidate_sampling == "low_fpr_full":
        raise ValueError(
            "clip_lora cannot use low_fpr_full because vision LoRA makes "
            "precomputed image features stale; use balanced_holdout."
        )
    if candidate_sampling in {"low_fpr_full", "balanced_holdout"}:
        low_fpr_attacks = {
            "blackbox_loss",
            "loss_series",
            "grad_cosine",
            "avg_cosine",
            "fedmia_loss",
            "fedmia_cosine",
        }
        if model_type not in {"clip_mlp", "visual_adapter", "clip_lora"}:
            raise ValueError(
                f"audit.candidate_sampling={candidate_sampling} requires a "
                "supported CLIP parameter-efficient model."
            )
        unsupported_low_fpr = sorted(attacks - low_fpr_attacks)
        if unsupported_low_fpr:
            raise ValueError(
                f"{candidate_sampling} does not support: "
                + ", ".join(unsupported_low_fpr)
            )
    if candidate_sampling == "low_fpr_full":
        if int(audit.get("low_fpr_min_nonmembers", 1000)) < 1000:
            raise ValueError(
                "audit.low_fpr_min_nonmembers must be at least 1000."
            )
        low_fpr_min_nonmembers = int(
            audit.get("low_fpr_min_nonmembers", 1000)
        )
        low_fpr_max_members = int(audit.get("low_fpr_max_members", 0))
        low_fpr_max_nonmembers = int(
            audit.get("low_fpr_max_nonmembers", 0)
        )
        if low_fpr_max_members < 0 or low_fpr_max_members == 1:
            raise ValueError(
                "audit.low_fpr_max_members must be 0 (unlimited) or at least 2."
            )
        if (
            low_fpr_max_nonmembers < 0
            or 0 < low_fpr_max_nonmembers < low_fpr_min_nonmembers
        ):
            raise ValueError(
                "audit.low_fpr_max_nonmembers must be 0 (unlimited) or at "
                "least audit.low_fpr_min_nonmembers."
            )
    if candidate_sampling == "balanced_holdout":
        low_fpr_max_members = int(audit.get("low_fpr_max_members", 0))
        low_fpr_max_nonmembers = int(audit.get("low_fpr_max_nonmembers", 0))
        if low_fpr_max_members < 0 or low_fpr_max_members == 1:
            raise ValueError(
                "audit.low_fpr_max_members must be 0 (unlimited) or at least 2."
            )
        if low_fpr_max_nonmembers < 0 or low_fpr_max_nonmembers == 1:
            raise ValueError(
                "audit.low_fpr_max_nonmembers must be 0 (unlimited) or at least 2."
            )
    if pooled_audit and attacks - POOLED_CLIENT_ATTACKS:
        raise ValueError(
            "Multi-client pooled auditing currently supports only: "
            + ", ".join(sorted(POOLED_CLIENT_ATTACKS))
        )
    if (
        pooled_audit
        and candidate_sampling == "legacy"
        and not bool(audit.get("match_candidate_labels", False))
    ):
        raise ValueError(
            "Multi-client pooled auditing requires match_candidate_labels=true."
        )
    if "nasr_active" in attacks and int(audit.get("active_max_samples", 16)) < 2:
        raise ValueError("audit.active_max_samples must be at least 2.")
    if "nasr_active" in attacks and int(audit.get("active_probe_cycles", 3)) <= 0:
        raise ValueError("audit.active_probe_cycles must be positive.")
    if "promptmia" in attacks and int(audit.get("promptmia_max_samples", 16)) < 2:
        raise ValueError("audit.promptmia_max_samples must be at least 2.")
    legacy_candidates = int(audit.get("max_samples_per_group", 32))
    if int(audit.get("max_member_samples", legacy_candidates)) < 2:
        raise ValueError("audit.max_member_samples must be at least 2.")
    if int(audit.get("max_nonmember_samples", legacy_candidates)) < 2:
        raise ValueError("audit.max_nonmember_samples must be at least 2.")
    if candidate_sampling == "fedmia_mix":
        required_nonmembers = math.ceil(
            int(audit.get("max_member_samples", legacy_candidates))
            * nonmember_ratio
        )
        if int(audit.get("max_nonmember_samples", legacy_candidates)) < required_nonmembers:
            raise ValueError(
                "FedMIA mix sampling requires max_nonmember_samples >= "
                "ceil(max_member_samples * nonmember_to_member_ratio), got "
                f"{audit.get('max_nonmember_samples', legacy_candidates)} < "
                f"{required_nonmembers}."
            )
    if str(audit.get("signal_storage", "compact")).lower() not in {
        "none",
        "compact",
        "full",
    }:
        raise ValueError("audit.signal_storage must be none, compact, or full.")
    for key in (
        "fedmia_tail",
        "fedmia_loss_tail",
        "fedmia_cosine_tail",
    ):
        if key in audit and str(audit[key]).lower() not in {
            "upper",
            "lower",
            "calibrated",
        }:
            raise ValueError(f"audit.{key} must be upper, lower, or calibrated.")
    if not 0 < float(audit.get("fedmia_tail_calibration_fraction", 0.25)) < 1:
        raise ValueError(
            "audit.fedmia_tail_calibration_fraction must be between 0 and 1."
        )
    if int(audit.get("promptres_background_rank", 0)) < 0:
        raise ValueError("audit.promptres_background_rank must be non-negative.")
    if str(audit.get("promptres_aggregation", "mean")).lower() not in {
        "mean",
        "max",
        "last",
    }:
        raise ValueError("audit.promptres_aggregation must be mean, max, or last.")
    if (
        "promptres" in attacks
        and int(audit.get("promptres_background_rank", 0)) > 0
        and config["sample_users"] < 2
    ):
        raise ValueError(
            "PromptRes background residualization requires two clients per round."
        )
    for key in (
        "min_trainable_update_norm",
        "max_stagnant_loss_range",
        "uniform_loss_tolerance",
    ):
        if float(audit.get(key, 0.0)) < 0:
            raise ValueError(f"audit.{key} must be non-negative.")
    if str(
        audit.get("audit_view", "protocol_plus_released_prompts")
    ).lower() == "released_prompt" and attacks & {
        "nasr_passive",
        "fedmia_cosine",
        "grad_cosine",
        "avg_cosine",
        "promptres",
    }:
        raise ValueError(
            "released_prompt audit view cannot run update-dependent attacks "
            "nasr_passive, fedmia_cosine, grad_cosine, avg_cosine, or promptres."
        )
    projres = config.get("projres", {})
    if bool(projres.get("enabled", False)):
        if model_type not in {"clip_mlp", "visual_adapter", "clip_lora"}:
            raise ValueError(
                "Integrated ProjRes requires CLIP-MLP, Visual Adapter, or "
                "CLIP-LoRA."
            )
        if method not in {"fedavg", "fedsgd"}:
            raise ValueError(
                "Integrated ProjRes requires FedAvg or FedSGD training."
            )
        if model_type != "clip_lora" and not bool(
            config.get(model_type, {}).get("precompute_features", True)
        ):
            raise ValueError(
                "Integrated ProjRes requires precompute_features=true."
            )
        if model_type == "clip_lora":
            if method != "fedsgd":
                raise ValueError(
                    "Paper-faithful CLIP-LoRA ProjRes requires one-batch FedSGD."
                )
            optimizer_config = dict(config.get("fedsgd", {}))
            if str(optimizer_config.get("client_optimizer", "sgd")).lower() != "sgd":
                raise ValueError("CLIP-LoRA ProjRes requires vanilla client SGD.")
            if float(optimizer_config.get("momentum", 0.0)) != 0.0 or float(
                optimizer_config.get("weight_decay", 0.0)
            ) != 0.0:
                raise ValueError(
                    "CLIP-LoRA ProjRes requires zero momentum and weight decay."
                )
            if float(config.get("clip_lora", {}).get("dropout", 0.25)) != 0.0:
                raise ValueError(
                    "CLIP-LoRA ProjRes requires clip_lora.dropout=0 so the "
                    "observed LoRA input matches the projected representation."
                )
            if str(config.get("clip_lora", {}).get("encoder", "both")).lower() not in {
                "vision",
                "both",
            }:
                raise ValueError(
                    "CLIP-LoRA ProjRes requires LoRA in the vision encoder."
                )
            attacked_parameter = projres.get("attacked_parameter")
            if attacked_parameter is not None and (
                "vision_model" not in str(attacked_parameter)
                or not str(attacked_parameter).endswith(".lora_A")
            ):
                raise ValueError(
                    "projres.attacked_parameter must name a vision lora_A matrix."
                )
            if str(projres.get("token_reduction", "cls")).lower() not in {
                "cls",
                "mean",
            }:
                raise ValueError("projres.token_reduction must be cls or mean.")
        threshold = float(projres.get("threshold", 0.01))
        max_candidates = int(projres.get("max_candidates", 32))
        min_nonmembers = int(projres.get("min_nonmembers", 1000))
        max_nonmembers = int(projres.get("max_nonmembers", 20000))
        if threshold < 0 or max_candidates <= 0 or min_nonmembers < 1000:
            raise ValueError(
                "ProjRes threshold must be non-negative, max_candidates must "
                "be positive, and min_nonmembers must be at least 1000."
            )
        if max_nonmembers < 0 or (
            max_nonmembers and max_nonmembers < min_nonmembers
        ):
            raise ValueError(
                "ProjRes max_nonmembers must be 0 or at least min_nonmembers."
            )
        evaluation_round = projres.get("evaluation_round", "last")
        if str(evaluation_round).lower() != "last":
            try:
                evaluation_round = int(evaluation_round)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "projres.evaluation_round must be 'last' or an integer."
                ) from error
            if not 1 <= evaluation_round <= config["num_global_iters"]:
                raise ValueError(
                    "projres.evaluation_round must be in "
                    "[1, num_global_iters]."
                )
    defense = config.get("defense", {})
    defense_name = str(defense.get("name", "none")).lower()
    if defense_name not in SUPPORTED_DEFENSES:
        raise ValueError(f"Unknown privacy defense: {defense_name}")
    if model_type in {
        "clip_mlp",
        "visual_adapter",
        "clip_lora",
    } and defense_name not in {
        "none",
        "iclr",
    }:
        raise ValueError(
            f"{model_type} attack experiments currently require defense.name "
            "to be none or iclr."
        )
    if defense_name == "iclr":
        if model_type not in {"clip_mlp", "visual_adapter", "clip_lora"}:
            raise ValueError(
                "ICLR requires CLIP-MLP, Visual Adapter, or CLIP-LoRA."
            )
        if method not in {"fedavg", "fedsgd"}:
            raise ValueError("ICLR requires linear FedAvg or FedSGD aggregation.")
        if config["sample_users"] < 2:
            raise ValueError("ICLR requires at least two selected clients per round.")
        model_config = dict(config.get(model_type, {}))
        if model_type != "clip_lora" and not bool(
            model_config.get("precompute_features", True)
        ):
            raise ValueError(
                "ICLR currently requires precomputed CLIP features so the exact "
                "local-update batch stream can be ranked without retaining raw images."
            )
        top_fraction = float(defense.get("iclr_validation_top_fraction", 0.2))
        if not 0.0 < top_fraction <= 0.5:
            raise ValueError(
                "defense.iclr_validation_top_fraction must be in (0, 0.5]."
            )
    if (
        defense_name in FEDMIA_BASELINE_DEFENSES
        and method not in {"fedavg", "promptfl"}
    ):
        raise ValueError(
            "FedMIA baseline defenses require a shared global FedAvg or PromptFL "
            "model and must be evaluated as standalone comparisons."
        )
    if defense_name == "mist" and config["sample_users"] < 2:
        raise ValueError("MIST requires at least two selected client submodels.")
    if not 0 < float(defense.get("dp_delta", 1e-5)) < 1:
        raise ValueError("defense.dp_delta must be in (0, 1).")
    if defense_name == "prompt_dp":
        if float(defense.get("dp_max_grad_norm", 1.0)) <= 0:
            raise ValueError("defense.dp_max_grad_norm must be positive.")
        if float(defense.get("dp_noise_multiplier", 1.0)) <= 0:
            raise ValueError("defense.dp_noise_multiplier must be positive.")
    if float(defense.get("perturb_clip_norm", 1.0)) <= 0:
        raise ValueError("defense.perturb_clip_norm must be positive.")
    if float(defense.get("perturb_noise_std", 0.05)) < 0:
        raise ValueError("defense.perturb_noise_std must be non-negative.")
    if not 0 <= float(defense.get("sparse_ratio", 0.9)) < 1:
        raise ValueError("defense.sparse_ratio must be in [0, 1).")
    if float(defense.get("mixup_alpha", 1.0)) <= 0:
        raise ValueError("defense.mixup_alpha must be positive.")
    if not 0 < float(defense.get("sampling_ratio", 0.5)) <= 1:
        raise ValueError("defense.sampling_ratio must be in (0, 1].")
    if not 0 <= float(defense.get("data_aug_strength", 0.1)) <= 1:
        raise ValueError("defense.data_aug_strength must be in [0, 1].")
    if not 0 <= float(defense.get("data_aug_flip_probability", 0.5)) <= 1:
        raise ValueError("defense.data_aug_flip_probability must be in [0, 1].")
    if not 0 <= float(defense.get("data_aug_color_jitter", 0.1)) <= 1:
        raise ValueError("defense.data_aug_color_jitter must be in [0, 1].")
    if not 0 <= float(defense.get("cofedmid_recycle_ratio", 0.1)) <= 1:
        raise ValueError("defense.cofedmid_recycle_ratio must be in [0, 1].")
    if not 0 <= float(defense.get("cofedmid_perturb_ratio", 0.1)) <= 1:
        raise ValueError("defense.cofedmid_perturb_ratio must be in [0, 1].")
    if int(defense.get("cofedmid_intervals", 4)) <= 0:
        raise ValueError("defense.cofedmid_intervals must be positive.")
    if not 0 <= float(defense.get("cofedmid_exp3_gamma", 0.2)) <= 1:
        raise ValueError("defense.cofedmid_exp3_gamma must be in [0, 1].")
    if not 0 <= float(defense.get("soft_obfuscation_strength", 0.5)) <= 1:
        raise ValueError("defense.soft_obfuscation_strength must be in [0, 1].")
    if float(defense.get("hamp_output_temperature", 4.0)) < 1:
        raise ValueError("defense.hamp_output_temperature must be at least one.")
    if not 0 < float(defense.get("hamp_true_probability", 0.6)) < 1:
        raise ValueError("defense.hamp_true_probability must be in (0, 1).")
    if int(defense.get("local_ggeur_augments", 2)) < 0:
        raise ValueError("defense.local_ggeur_augments must be non-negative.")
    if float(defense.get("local_ggeur_geometry_scale", 0.45)) < 0:
        raise ValueError("defense.local_ggeur_geometry_scale must be non-negative.")
    if float(defense.get("local_ggeur_original_noise", 0.03)) < 0:
        raise ValueError("defense.local_ggeur_original_noise must be non-negative.")
    if float(defense.get("local_ggeur_mean_noise_std", 0.0)) < 0:
        raise ValueError("defense.local_ggeur_mean_noise_std must be non-negative.")
    if float(defense.get("local_ggeur_output_temperature", 4.0)) < 1:
        raise ValueError(
            "defense.local_ggeur_output_temperature must be at least one."
        )
    output_margin = defense.get("local_ggeur_output_margin")
    if output_margin is not None and float(output_margin) < 0:
        raise ValueError("defense.local_ggeur_output_margin must be non-negative.")
    entropy_rounds = defense.get("local_ggeur_entropy_rounds")
    if entropy_rounds is not None and int(entropy_rounds) < 0:
        raise ValueError("defense.local_ggeur_entropy_rounds must be non-negative.")
    late_start = defense.get("local_ggeur_late_start_round")
    if late_start is not None and int(late_start) < 0:
        raise ValueError("defense.local_ggeur_late_start_round must be non-negative.")
    late_augments = defense.get("local_ggeur_late_augments")
    if late_augments is not None and int(late_augments) < 0:
        raise ValueError("defense.local_ggeur_late_augments must be non-negative.")
    upload_clip = defense.get("local_ggeur_upload_clip_norm")
    if upload_clip is not None and float(upload_clip) < 0:
        raise ValueError("defense.local_ggeur_upload_clip_norm must be non-negative.")
    if float(defense.get("local_ggeur_upload_noise_std", 0.0)) < 0:
        raise ValueError("defense.local_ggeur_upload_noise_std must be non-negative.")
    if not 0 <= float(defense.get("local_ggeur_mean_mix", 0.8)) <= 1:
        raise ValueError("defense.local_ggeur_mean_mix must be in [0, 1].")
    if str(defense.get("local_ggeur_anchor_mode", "class_mean")).lower() not in {
        "class_mean",
        "sample",
    }:
        raise ValueError(
            "defense.local_ggeur_anchor_mode must be 'class_mean' or 'sample'."
        )
    if str(
        defense.get("local_ggeur_original_mode", "class_mean_noise")
    ).lower() not in {
        "drop",
        "class_mean",
        "class_mean_noise",
        "mean_mix",
        "blur",
        "noise",
    }:
        raise ValueError("defense.local_ggeur_original_mode is invalid.")
    spatial = {
        "fedmia_loss",
        "fedmia_cosine",
        "transfer_representation",
        "rmia",
        "yoqo",
        "canary",
    }
    if audit.get("enabled", True) and attacks & spatial and config["sample_users"] < 2:
        raise ValueError(
            "Cross-client membership attacks require at least two clients per round."
        )
def run(config: dict) -> list[dict]:
    from trainmodel.custom_clip import CustomCLIP, get_default_prompt_template
    from trainmodel.clip_mlp import CLIPImageMLP
    from trainmodel.clip_lora import (
        CLIPLoRA,
        build_clip_lora_text_inputs,
    )
    from trainmodel.visual_adapter import (
        VisualCLIPAdapter,
        build_visual_adapter_text_features,
    )
    from trainmodel.clip_feature_cache import (
        collate_clip_features,
        precompute_federated_clip_features,
    )

    config = copy.deepcopy(config)
    if str(config.get("model_type", "prompt")).lower() in {
        "clip_mlp",
        "visual_adapter",
        "clip_lora",
    }:
        config["aggregation_weighting"] = "uniform"
    validate_config(config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    if bool(config.get("require_cuda", False)) and not cuda_available:
        raise RuntimeError(
            "This experiment requires CUDA, but torch.cuda.is_available() is false."
        )
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        f"cuda:{config['gpu']}" if cuda_available else "cpu"
    )
    method = config["aggregator"].lower()
    method_config = dict(config.get(method, {}))
    method_config.setdefault("seed", seed)
    config[method] = method_config

    # Sweep jobs already have a stable directory; direct runs receive a
    # collision-free timestamped child from this helper.
    result_dir = _result_directory(config)
    os.makedirs(result_dir, exist_ok=True)
    with open(
        os.path.join(result_dir, "run_config.yaml"), "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=_build_logging_handlers(result_dir),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger(__name__).info(
        "Using device %s (require_cuda=%s)",
        device,
        bool(config.get("require_cuda", False)),
    )

    partition_mode = str(config.get("partition_mode", "auto")).lower()
    split_arguments = {
        "root_dir": config.get("data_root", "./data"),
        "fpl": True,
        "fpl_shots": config.get("fpl_shots"),
        "use_full_dataset": bool(config.get("use_full_dataset", False)),
    }
    if partition_mode == "pathological":
        train_sets, test_sets, class_names = generate_pathological_split(
            config["dataset_name"],
            num_users=config["total_users"],
            seed=seed,
            **split_arguments,
        )
    elif partition_mode == "iid" or (
        partition_mode == "auto" and float(config["dirichlet_alpha"]) >= 10
    ):
        train_sets, test_sets, class_names = generate_iid_split(
            config["dataset_name"],
            num_users=config["total_users"],
            **split_arguments,
        )
    else:
        train_sets, test_sets, class_names = generate_dirichlet_split(
            config["dataset_name"],
            config["total_users"],
            config["dirichlet_alpha"],
            **split_arguments,
        )

    processor, clip_model = _load_local_clip(config["cache_dir"], device)
    model_type = str(config.get("model_type", "prompt")).lower()
    if model_type == "clip_mlp":
        mlp_config = dict(config.get("clip_mlp", {}))
        model = CLIPImageMLP(
            clip_model=clip_model,
            num_classes=len(class_names),
            hidden_dim=int(mlp_config.get("hidden_dim", 512)),
            dropout=float(mlp_config.get("dropout", 0.0)),
            normalize_features=bool(mlp_config.get("normalize_features", False)),
            device=device,
        )
    elif model_type == "visual_adapter":
        adapter_config = dict(config.get("visual_adapter", {}))
        text_features = build_visual_adapter_text_features(
            clip_model=clip_model,
            processor=processor,
            classnames=class_names,
            dataset_name=config["dataset_name"],
            device=device,
            template=adapter_config.get("template"),
        )
        model = VisualCLIPAdapter(
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
        )
    elif model_type == "clip_lora":
        lora_config = dict(config.get("clip_lora", {}))
        model = CLIPLoRA(
            clip_model=clip_model,
            text_inputs=build_clip_lora_text_inputs(
                processor=processor,
                classnames=class_names,
                dataset_name=config["dataset_name"],
                template=lora_config.get("template"),
            ),
            classnames=class_names,
            encoder=str(lora_config.get("encoder", "both")),
            target_modules=lora_config.get(
                "target_modules", ["q", "k", "v"]
            ),
            layers=lora_config.get("layers", "all"),
            rank=int(lora_config.get("rank", 2)),
            alpha=float(lora_config.get("alpha", 1.0)),
            dropout=float(lora_config.get("dropout", 0.25)),
            scaling=str(lora_config.get("scaling", "sqrt_rank")),
            device=device,
        )
    else:
        parameterization = {
            "fedavg": "full",
            "promptfl": "promptfl",
        }[method]
        model = CustomCLIP(
            clip_model=clip_model,
            processor=processor,
            classnames=class_names,
            device=device,
            n_ctx=config["n_ctx"],
            template=get_default_prompt_template(config["dataset_name"]),
            class_specific_ctx=config["class_specific_ctx"],
            parameterization=parameterization,
            method_config=method_config,
        )

    def collate_fn(batch):
        images, labels = zip(*batch)
        processed = processor(images=list(images), return_tensors="pt")
        return processed["pixel_values"], torch.as_tensor(labels, dtype=torch.long)

    if (
        model_type in {"clip_mlp", "visual_adapter"}
        and bool(config.get(model_type, {}).get("precompute_features", True))
    ):
        train_sets, test_sets, _feature_summary = (
            precompute_federated_clip_features(
                model,
                train_sets,
                test_sets,
                collate_fn,
                int(
                    config.get(model_type, {}).get(
                        "precompute_batch_size", 64
                    )
                ),
            )
        )
        collate_fn = collate_clip_features

    audit_config = dict(config.get("audit", {}))
    audit_config.setdefault("seed", seed)
    audit_config.setdefault(
        "few_shot",
        config.get("fpl_shots") is not None
        and not bool(config.get("use_full_dataset", False)),
    )
    audit_config.setdefault("fpl_shots", config.get("fpl_shots"))
    server = ServerBase(
        device=device,
        dataset_name=config["dataset_name"],
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=class_names,
        model=model,
        batch_size=config["batch_size"],
        eval_batch_size=config["eval_batch_size"],
        learning_rate=config["learning_rate"],
        learning_rate_decay=config.get("learning_rate_decay", 1.0),
        learning_rate_decay_interval=config.get(
            "learning_rate_decay_interval", 1
        ),
        num_glob_iters=config["num_global_iters"],
        local_epochs=config["local_epochs"],
        total_users=config["total_users"],
        results_dir=result_dir,
        user_per_round=config["sample_users"],
        aggregator=build_aggregator(
            method,
            device=device,
            aggregation_weighting=config.get(
                "aggregation_weighting", "sample_count"
            ),
            seed=seed,
            rank=int(method_config.get("rank", 4)),
            oversampling=int(method_config.get("oversampling", 2)),
            global_clip_norm=float(method_config.get("global_clip_norm", 1.0)),
            global_noise_multiplier=float(
                method_config.get("global_noise_multiplier", 1.0)
            ),
            reproducible_dp_noise=bool(
                method_config.get("reproducible_dp_noise", False)
            ),
        ),
        model_load_path=config.get("model_load_path"),
        save_models=config["save_models"],
        collate_fn=collate_fn,
        eval_interval=config["eval_interval"],
        audit_config=audit_config,
        projres_config=config.get("projres", {"enabled": False}),
        defense_config=config.get("defense", {"name": "none"}),
        method_config=method_config,
    )
    summaries = server.train()
    logger.info("%s", _format_privacy_audit_summary(summaries))
    return summaries


def default_config() -> dict:
    return {
        "model_type": "prompt",
        "dataset_name": "caltech101",
        "data_root": "./data",
        "batch_size": 16,
        "eval_batch_size": 64,
        "learning_rate": 0.001,
        "learning_rate_decay": 1.0,
        "learning_rate_decay_interval": 1,
        "num_global_iters": 20,
        "local_epochs": 2,
        "total_users": 10,
        "sample_users": 10,
        "aggregator": "fedavg",
        "aggregation_weighting": "sample_count",
        "clip_mlp": {
            "hidden_dim": 512,
            "dropout": 0.0,
            "normalize_features": False,
            "precompute_features": True,
            "precompute_batch_size": 64,
        },
        "visual_adapter": {
            "feature_dim": 512,
            "reduction": 4,
            "alpha": 0.2,
            "output_relu": True,
            "text_adapter_enabled": True,
            "text_reduction": 4,
            "text_alpha": 0.2,
            "text_output_relu": True,
            "precompute_features": True,
            "precompute_batch_size": 64,
            "template": None,
        },
        "clip_lora": {
            "encoder": "both",
            "target_modules": ["q", "k", "v"],
            "layers": "all",
            "rank": 2,
            "alpha": 1.0,
            "dropout": 0.25,
            "scaling": "sqrt_rank",
            "template": None,
        },
        "promptfl": {},
        "dirichlet_alpha": 0.1,
        "partition_mode": "auto",
        "use_full_dataset": False,
        "gpu": 0,
        "require_cuda": False,
        "seed": 42,
        "fpl_shots": 16,
        "cache_dir": "./checkpoints/clip-vit-base-patch32",
        "n_ctx": 32,
        "class_specific_ctx": False,
        "model_load_path": None,
        "save_models": False,
        "eval_interval": 5,
        "results_dir": "./results",
        "results_dir_is_run_dir": False,
        "audit": {
            "enabled": True,
            "audit_view": "protocol_plus_released_prompts",
            "strict": True,
            "target_client_id": 0,
            "ensure_target_participation": True,
            "attacks": [
                "blackbox_loss",
                "loss_series",
                "grad_cosine",
                "avg_cosine",
                "nasr_passive",
                "nasr_active",
                "fedmia_loss",
                "fedmia_cosine",
                "transfer_representation",
                "codepoison",
                "pipra",
                "rmia",
                "imia",
                "quantile_mia",
                "yoqo",
                "canary",
                "promptmia",
            ],
            "max_samples_per_group": 32,
            "calibration_fraction": 0.5,
            "candidate_sampling": "legacy",
            "nonmember_to_member_ratio": 1.0,
            "match_candidate_labels": False,
            "low_fpr_min_nonmembers": 1000,
            "low_fpr_max_members": 0,
            "low_fpr_max_nonmembers": 0,
            "signal_storage": "compact",
            "fedmia_tail": "upper",
            "fedmia_tail_calibration_fraction": 0.25,
            "training_health_check": True,
            "fedmia_signal_health_check": False,
            "min_trainable_update_norm": 1e-12,
            "max_stagnant_loss_range": 1e-8,
            "uniform_loss_tolerance": 1e-4,
            "active_max_samples": 16,
            "active_ascent_steps": 1,
            "active_ascent_lr": 0.01,
            "active_probe_cycles": 3,
            "codepoison_weight": 1.0,
            "synthetic_mean": 0.0,
            "synthetic_std": 0.1,
            "auxiliary_fraction": 0.5,
            "rmia_offline_a": 0.3,
            "rmia_gamma": 2.0,
            "qmia_quantile": 0.9,
            "qmia_epochs": 200,
            "qmia_learning_rate": 0.01,
            "pipra_shadow_prompts": 4,
            "pipra_shadow_steps": 20,
            "pipra_shadow_learning_rate": 0.02,
            "pipra_attack_epochs": 200,
            "pipra_attack_learning_rate": 0.01,
            "pipra_temperature": 0.1,
            "imia_models": 4,
            "imia_warmup_steps": 10,
            "imia_imitation_steps": 20,
            "imia_pivot_steps": 20,
            "imia_learning_rate": 0.02,
            "imia_pivots_per_class": 4,
            "query_max_samples": 32,
            "query_reference_models": 2,
            "query_epsilon": 0.1,
            "yoqo_steps": 50,
            "yoqo_learning_rate": 0.001,
            "yoqo_distortion_weight": 5.0,
            "yoqo_loss_threshold": 0.4,
            "canary_num_queries": 2,
            "canary_steps": 20,
            "canary_shadow_steps": 3,
            "canary_learning_rate": 0.01,
            "canary_shadow_learning_rate": 0.02,
            "promptmia_max_samples": 32,
            "promptmia_keys": 4,
            "promptmia_delta_min": 0.02,
            "promptmia_similarity_span": 0.05,
            "promptres_background_rank": 0,
            "promptres_aggregation": "mean",
        },
        "projres": {
            "enabled": False,
            "evaluation_round": "last",
            "threshold": 0.01,
            "max_candidates": 32,
            "min_nonmembers": 1000,
            "max_nonmembers": 20000,
            "attacked_parameter": None,
            "token_reduction": "cls",
        },
        "defense": {
            "name": "none",
            "perturb_clip_norm": 1.0,
            "perturb_noise_std": 0.05,
            "sparse_ratio": 0.9,
            "mixup_alpha": 1.0,
            "sampling_ratio": 0.5,
            "data_aug_strength": 0.1,
            "data_aug_flip_probability": 0.5,
            "data_aug_color_jitter": 0.1,
            "dp_max_grad_norm": 1.0,
            "dp_noise_multiplier": 1.0,
            "dp_delta": 1e-5,
            "reproducible_dp_noise": False,
            "cofedmid_intervals": 4,
            "cofedmid_recycle_ratio": 0.1,
            "cofedmid_entropy_weight": 0.05,
            "cofedmid_exp3_gamma": 0.2,
            "cofedmid_noise_std": 0.05,
            "cofedmid_perturb_ratio": 0.1,
            "mist_cross_steps": 1,
            "mist_cross_weight": 1.0,
            "soft_obfuscation_strength": 0.5,
            "soft_noise_std": 0.05,
            "hamp_true_probability": 0.6,
            "hamp_entropy_weight": 0.05,
            "hamp_output_temperature": 4.0,
            "local_ggeur_augments": 3,
            "local_ggeur_geometry_scale": 0.6,
            "local_ggeur_anchor_mode": "class_mean",
            "local_ggeur_original_mode": "class_mean_noise",
            "local_ggeur_original_noise": 0.08,
            "local_ggeur_mean_noise_std": 0.0,
            "local_ggeur_mean_mix": 0.8,
            "local_ggeur_fallback_std": 0.02,
            "local_ggeur_entropy_weight": 0.0,
            "local_ggeur_entropy_rounds": None,
            "local_ggeur_late_start_round": None,
            "local_ggeur_late_augments": None,
            "local_ggeur_output_temperature": 4.0,
            "local_ggeur_output_margin": None,
            "local_ggeur_calibrate_observations": False,
            "local_ggeur_class_balanced": False,
            "local_ggeur_upload_clip_norm": 0.5,
            "local_ggeur_upload_noise_std": 0.07,
            "iclr_validation_top_fraction": 0.2,
        },
    }


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Federated prompt tuning membership-privacy benchmark"
    )
    parser.add_argument("--config", default="configs/clip_mlp_privacy.yaml")
    parser.add_argument("--dataset_name")
    parser.add_argument("--data_root")
    parser.add_argument("--cache_dir")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_global_iters", type=int)
    parser.add_argument("--total_users", type=int)
    parser.add_argument("--sample_users", type=int)
    parser.add_argument("--local_epochs", type=int)
    parser.add_argument("--fpl_shots", "--fpl-shots", "--shots", type=int)
    parser.add_argument(
        "--dirichlet_alpha", "--dirichlet-alpha", type=float
    )
    parser.add_argument(
        "--partition_mode",
        choices=["auto", "dirichlet", "iid", "pathological"],
    )
    parser.add_argument(
        "--use_full_dataset",
        action="store_true",
        default=None,
        help="Use complete official train/test splits; requires fpl_shots=null.",
    )
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument(
        "--learning_rate_decay",
        "--learning-rate-decay",
        type=float,
        help="Multiply the learning rate by this factor after each communication round.",
    )
    parser.add_argument(
        "--learning_rate_decay_interval",
        "--learning-rate-decay-interval",
        type=int,
        help="Apply one learning-rate decay after this many communication rounds.",
    )
    parser.add_argument(
        "--model_type",
        choices=["prompt", "clip_mlp", "visual_adapter", "clip_lora"],
    )
    parser.add_argument("--mlp_hidden_dim", type=int)
    parser.add_argument("--mlp_dropout", type=float)
    parser.add_argument("--adapter_reduction", type=int)
    parser.add_argument("--adapter_alpha", type=float)
    parser.add_argument(
        "--aggregator",
        choices=["fedavg", "fedsgd", "promptfl"],
    )
    parser.add_argument(
        "--aggregation_weighting",
        "--aggregation-weighting",
        choices=["sample_count", "uniform"],
    )
    parser.add_argument("--target_client_id", type=int)
    parser.add_argument("--audit_attacks", help="Comma-separated attack names")
    parser.add_argument(
        "--attack",
        choices=["none", *sorted(SUPPORTED_ATTACKS)],
        help="Run one attack; use 'none' for defense-only or plain training.",
    )
    parser.add_argument(
        "--defense",
        choices=sorted(SUPPORTED_DEFENSES),
        help="Run one independent defense; use 'none' for attack-only or plain training.",
    )
    parser.add_argument("--perturb_clip_norm", type=float)
    parser.add_argument("--perturb_noise_std", type=float)
    parser.add_argument("--sparse_ratio", type=float)
    parser.add_argument("--mixup_alpha", type=float)
    parser.add_argument("--sampling_ratio", type=float)
    parser.add_argument("--data_aug_strength", type=float)
    parser.add_argument("--data_aug_flip_probability", type=float)
    parser.add_argument("--data_aug_color_jitter", type=float)
    parser.add_argument("--local_ggeur_augments", type=int)
    parser.add_argument("--local_ggeur_geometry_scale", type=float)
    parser.add_argument("--local_ggeur_anchor_mode")
    parser.add_argument("--local_ggeur_original_mode")
    parser.add_argument("--local_ggeur_original_noise", type=float)
    parser.add_argument("--local_ggeur_mean_noise_std", type=float)
    parser.add_argument("--local_ggeur_mean_mix", type=float)
    parser.add_argument("--local_ggeur_fallback_std", type=float)
    parser.add_argument("--local_ggeur_entropy_weight", type=float)
    parser.add_argument("--local_ggeur_entropy_rounds", type=int)
    parser.add_argument("--local_ggeur_late_start_round", type=int)
    parser.add_argument("--local_ggeur_late_augments", type=int)
    parser.add_argument("--local_ggeur_output_temperature", type=float)
    parser.add_argument("--local_ggeur_output_margin", type=float)
    parser.add_argument("--local_ggeur_upload_clip_norm", type=float)
    parser.add_argument("--local_ggeur_upload_noise_std", type=float)
    parser.add_argument(
        "--local_ggeur_calibrate_observations",
        action="store_true",
        help="Apply Local-GGEUR output calibration to round-level audit observations.",
    )
    parser.add_argument(
        "--local_ggeur_class_balanced",
        action="store_true",
        help="Uniformly sample local classes for Local-GGEUR feature training.",
    )
    args = parser.parse_args()
    config = default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        audit = config["audit"] | loaded.pop("audit", {})
        defense = config["defense"] | loaded.pop("defense", {})
        config.update(loaded)
        config["audit"] = audit
        config["defense"] = defense
    for key in (
        "dataset_name",
        "data_root",
        "cache_dir",
        "gpu",
        "seed",
        "num_global_iters",
        "total_users",
        "sample_users",
        "local_epochs",
        "fpl_shots",
        "dirichlet_alpha",
        "partition_mode",
        "use_full_dataset",
        "learning_rate",
        "learning_rate_decay",
        "learning_rate_decay_interval",
        "model_type",
        "aggregator",
        "aggregation_weighting",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.mlp_hidden_dim is not None:
        config["clip_mlp"]["hidden_dim"] = args.mlp_hidden_dim
    if args.mlp_dropout is not None:
        config["clip_mlp"]["dropout"] = args.mlp_dropout
    if args.adapter_reduction is not None:
        config["visual_adapter"]["reduction"] = args.adapter_reduction
    if args.adapter_alpha is not None:
        config["visual_adapter"]["alpha"] = args.adapter_alpha
    if args.model_type == "clip_mlp":
        config["aggregator"] = "fedavg"
        config["aggregation_weighting"] = "uniform"
        config["fpl_shots"] = None
        config["use_full_dataset"] = True
        if args.attack is None and args.audit_attacks is None:
            config["audit"]["enabled"] = False
            config["audit"]["attacks"] = []
    elif args.model_type == "visual_adapter":
        config["aggregator"] = "fedsgd"
        config["aggregation_weighting"] = "uniform"
        config["local_epochs"] = 1
        config["fpl_shots"] = 16
        config["use_full_dataset"] = False
    elif args.model_type == "clip_lora":
        config["aggregator"] = "fedavg"
        config["aggregation_weighting"] = "uniform"
        config["fpl_shots"] = 16
        config["use_full_dataset"] = False
    # A direct few-shot/alpha override denotes the Dirichlet experiment
    # family unless the caller explicitly selected another partition mode.
    if args.fpl_shots is not None and args.use_full_dataset is None:
        config["use_full_dataset"] = False
    if (
        args.partition_mode is None
        and (args.fpl_shots is not None or args.dirichlet_alpha is not None)
    ):
        config["partition_mode"] = "dirichlet"
    if args.target_client_id is not None:
        config["audit"]["target_client_id"] = args.target_client_id
    if args.audit_attacks:
        config["audit"]["attacks"] = [
            item.strip() for item in args.audit_attacks.split(",") if item.strip()
        ]
        config["audit"]["enabled"] = bool(config["audit"]["attacks"])
    if args.attack is not None:
        config["audit"]["enabled"] = args.attack != "none"
        config["audit"]["attacks"] = [] if args.attack == "none" else [args.attack]
    if args.defense is not None:
        config["defense"]["name"] = args.defense
    for key in (
        "perturb_clip_norm",
        "perturb_noise_std",
        "sparse_ratio",
        "mixup_alpha",
        "sampling_ratio",
        "data_aug_strength",
        "data_aug_flip_probability",
        "data_aug_color_jitter",
        "local_ggeur_augments",
        "local_ggeur_geometry_scale",
        "local_ggeur_anchor_mode",
        "local_ggeur_original_mode",
        "local_ggeur_original_noise",
        "local_ggeur_mean_noise_std",
        "local_ggeur_mean_mix",
        "local_ggeur_fallback_std",
        "local_ggeur_entropy_weight",
        "local_ggeur_entropy_rounds",
        "local_ggeur_late_start_round",
        "local_ggeur_late_augments",
        "local_ggeur_output_temperature",
        "local_ggeur_output_margin",
        "local_ggeur_upload_clip_norm",
        "local_ggeur_upload_noise_std",
    ):
        value = getattr(args, key)
        if value is not None:
            config["defense"][key] = value
    if args.local_ggeur_calibrate_observations:
        config["defense"]["local_ggeur_calibrate_observations"] = True
    if args.local_ggeur_class_balanced:
        config["defense"]["local_ggeur_class_balanced"] = True
    return config


if __name__ == "__main__":
    run(parse_args())
