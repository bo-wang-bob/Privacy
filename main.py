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
from utils.privacy_accounting import (
    calibrate_gaussian_noise,
    planned_private_probe_steps,
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


def _load_local_clip(cache_dir: str, device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    try:
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            use_fast=True,
            local_files_only=True,
        )
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            local_files_only=True,
        ).to(device)
    except OSError as error:
        raise RuntimeError(
            "Federated prompt tuning requires a local openai/clip-vit-base-patch32 "
            f"cache under {cache_dir!r}."
        ) from error
    return processor, clip_model


def validate_config(config: dict) -> None:
    if config["train_mode"] not in {"centralized", "local"}:
        raise ValueError("train_mode must be 'centralized' or 'local'.")
    method = config["aggregator"].lower()
    supported_methods = {
        "promptfl",
        "fedotp",
        "fedpgp",
        "fedavg",
        "dpfpl",
        "fedask",
    }
    if method not in supported_methods:
        raise ValueError(
            "aggregator must be promptfl, fedotp, fedpgp, fedavg, dpfpl, or fedask."
        )
    method_config = config.get(method, {})
    if method in {"dpfpl", "fedask"} and int(method_config.get("rank", 4)) <= 0:
        raise ValueError(f"{method}.rank must be positive.")
    if method in {"dpfpl", "fedask"} and int(method_config.get("local_steps", 1)) <= 0:
        raise ValueError(f"{method}.local_steps must be positive.")
    if method == "dpfpl":
        for key in ("local_clip_norm", "global_clip_norm"):
            if float(method_config.get(key, 1.0)) <= 0:
                raise ValueError(f"dpfpl.{key} must be positive.")
        for key in ("local_noise_multiplier", "global_noise_multiplier"):
            if float(method_config.get(key, 1.0)) < 0:
                raise ValueError(f"dpfpl.{key} must be non-negative.")
        for key in ("local_target_epsilon", "global_target_epsilon"):
            value = method_config.get(key)
            if value is not None and float(value) <= 0:
                raise ValueError(f"dpfpl.{key} must be positive when set.")
    if method == "fedask":
        if float(method_config.get("scaling", 1.0)) <= 0:
            raise ValueError("fedask.scaling must be positive.")
        if float(method_config.get("clip_norm", 1.0)) <= 0:
            raise ValueError("fedask.clip_norm must be positive.")
        if float(method_config.get("noise_multiplier", 1.0)) < 0:
            raise ValueError("fedask.noise_multiplier must be non-negative.")
        if int(method_config.get("oversampling", 2)) < 0:
            raise ValueError("fedask.oversampling must be non-negative.")
        target = method_config.get("target_epsilon")
        if target is not None and float(target) <= 0:
            raise ValueError("fedask.target_epsilon must be positive when set.")
    if method == "fedotp":
        if float(method_config.get("epsilon", 0.01)) <= 0:
            raise ValueError("fedotp.epsilon must be positive.")
        if not 0 < float(method_config.get("transported_mass", 0.8)) <= 1:
            raise ValueError("fedotp.transported_mass must be in (0, 1].")
        if int(method_config.get("max_iterations", 100)) <= 0:
            raise ValueError("fedotp.max_iterations must be positive.")
        if float(method_config.get("threshold", 1e-3)) <= 0:
            raise ValueError("fedotp.threshold must be positive.")
    if method == "fedpgp":
        if int(method_config.get("rank", 8)) <= 0:
            raise ValueError("fedpgp.rank must be positive.")
        if float(method_config.get("contrastive_weight", 0.5)) < 0:
            raise ValueError("fedpgp.contrastive_weight must be non-negative.")
        if float(method_config.get("temperature", 0.5)) <= 0:
            raise ValueError("fedpgp.temperature must be positive.")
    if (
        method in {"dpfpl", "fedask"}
        and not 0 < float(method_config.get("delta", 1e-5)) < 1
    ):
        raise ValueError(f"{method}.delta must be in (0, 1).")
    if config["total_users"] <= 1:
        raise ValueError("total_users must be greater than one.")
    if not 1 <= config["sample_users"] <= config["total_users"]:
        raise ValueError("sample_users must be in [1, total_users].")
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
    candidate_sampling = str(audit.get("candidate_sampling", "legacy")).lower()
    if candidate_sampling not in {"legacy", "fedmia_mix"}:
        raise ValueError(
            "audit.candidate_sampling must be legacy or fedmia_mix."
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
    for key in ("fedmia_tail", "fedmia_loss_tail", "fedmia_cosine_tail"):
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
    if (
        "promptres" in attacks
        and method == "fedask"
        and str(audit.get("audit_view", "protocol_plus_released_prompts")).lower()
        != "full_whitebox"
    ):
        raise ValueError(
            "PromptRes needs a prompt update; FedASK exposes only sketches outside "
            "the full_whitebox audit view."
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
    defense = config.get("defense", {})
    defense_name = str(defense.get("name", "none")).lower()
    if defense_name not in SUPPORTED_DEFENSES:
        raise ValueError(f"Unknown privacy defense: {defense_name}")
    if defense_name in FEDMIA_BASELINE_DEFENSES and (
        method not in {"fedavg", "promptfl"}
        or config["train_mode"] != "centralized"
    ):
        raise ValueError(
            "FedMIA baseline defenses require centralized FedAvg and must be "
            "evaluated as standalone comparisons, not stacked with personalized "
            "or private prompt algorithms."
        )
    if method in {"fedotp", "fedpgp"} and defense_name != "none":
        raise ValueError(
            f"{method} paper training currently requires defense.name=none; "
            "stacking a defense would change its published objective."
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
    if (
        audit.get("enabled", True)
        and attacks & spatial
        and config["train_mode"] != "centralized"
        and method != "dpfpl"
    ):
        raise ValueError(
            "Cross-client membership attacks require a shared centralized FedAvg model."
        )


def run(config: dict) -> list[dict]:
    from trainmodel.custom_clip import CustomCLIP, get_default_prompt_template

    validate_config(config)
    config = copy.deepcopy(config)
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
    extra_steps = (
        int(config.get("defense", {}).get("mist_cross_steps", 1))
        if str(config.get("defense", {}).get("name", "none")).lower() == "mist"
        else 0
    )
    audit_probe_steps = planned_private_probe_steps(config.get("audit"))
    delta = float(method_config.get("delta", 1e-5))
    if method == "dpfpl":
        local_steps = (
            config["num_global_iters"]
            * (int(method_config.get("local_steps", 1)) + extra_steps)
            + audit_probe_steps
        )
        if method_config.get("local_target_epsilon") is not None:
            method_config["local_noise_multiplier"] = calibrate_gaussian_noise(
                float(method_config["local_target_epsilon"]),
                local_steps,
                delta,
                mechanisms_per_step=2,
            )
        if method_config.get("global_target_epsilon") is not None:
            method_config["global_noise_multiplier"] = calibrate_gaussian_noise(
                float(method_config["global_target_epsilon"]),
                config["num_global_iters"],
                delta,
            )
    elif method == "fedask" and method_config.get("target_epsilon") is not None:
        local_steps = (
            config["num_global_iters"]
            * (
                int(method_config.get("local_steps", config["local_epochs"]))
                + extra_steps
            )
            + audit_probe_steps
        )
        method_config["noise_multiplier"] = calibrate_gaussian_noise(
            float(method_config["target_epsilon"]),
            local_steps,
            delta,
        )
    config[method] = method_config
    effective_train_mode = (
        "local"
        if method in {"dpfpl", "fedotp", "fedpgp"}
        else "centralized"
        if method in {"fedask", "promptfl"}
        else config["train_mode"]
    )
    config["effective_train_mode"] = effective_train_mode

    # Include microseconds and the process id so concurrent sweeps of the same
    # method/defense cannot write into one directory and corrupt each other.
    timestamp = _result_run_id()
    configured_attacks = list(config.get("audit", {}).get("attacks", []))
    if not bool(config.get("audit", {}).get("enabled", True)) or not configured_attacks:
        attack_label = "no_attack"
    elif len(configured_attacks) == 1:
        attack_label = configured_attacks[0]
    else:
        attack_label = "multi_attack"
    defense_label = str(config.get("defense", {}).get("name", "none")).lower()
    result_dir = os.path.join(
        config["results_dir"],
        f"{config['dataset_name']}_{config['aggregator']}_{attack_label}_{defense_label}_{timestamp}",
    )
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
    parameterization = {
        "fedavg": "full",
        "promptfl": "promptfl",
        "fedotp": "fedotp",
        "fedpgp": "fedpgp",
        "dpfpl": "dpfpl",
        "fedask": "fedask",
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
        low_rank=int(method_config.get("rank", 4)),
        low_rank_scaling=float(method_config.get("scaling", 1.0)),
        method_config=method_config,
    )

    def collate_fn(batch):
        images, labels = zip(*batch)
        processed = processor(images=list(images), return_tensors="pt")
        return processed["pixel_values"], torch.as_tensor(labels, dtype=torch.long)

    audit_config = dict(config.get("audit", {}))
    audit_config.setdefault("seed", seed)
    audit_config.setdefault(
        "few_shot",
        config.get("fpl_shots") is not None
        and not bool(config.get("use_full_dataset", False)),
    )
    audit_config.setdefault("fpl_shots", config.get("fpl_shots"))
    server = ServerBase(
        train_mode=effective_train_mode,
        device=device,
        dataset_name=config["dataset_name"],
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=class_names,
        model=model,
        batch_size=config["batch_size"],
        eval_batch_size=config["eval_batch_size"],
        learning_rate=config["learning_rate"],
        num_glob_iters=config["num_global_iters"],
        local_epochs=config["local_epochs"],
        total_users=config["total_users"],
        results_dir=result_dir,
        user_per_round=config["sample_users"],
        aggregator=build_aggregator(
            method,
            device=device,
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
        defense_config=config.get("defense", {"name": "none"}),
        method_config=method_config,
    )
    summaries = server.train()
    logger.info("%s", _format_privacy_audit_summary(summaries))
    return summaries


def default_config() -> dict:
    return {
        "train_mode": "centralized",
        "dataset_name": "caltech101",
        "data_root": "./data",
        "batch_size": 16,
        "eval_batch_size": 64,
        "learning_rate": 0.001,
        "num_global_iters": 20,
        "local_epochs": 2,
        "total_users": 10,
        "sample_users": 10,
        "aggregator": "fedavg",
        "promptfl": {},
        "fedotp": {
            "epsilon": 0.01,
            "transported_mass": 0.8,
            "max_iterations": 100,
            "threshold": 0.001,
        },
        "fedpgp": {
            "rank": 8,
            "contrastive_weight": 0.5,
            "temperature": 0.5,
        },
        "dpfpl": {
            "rank": 4,
            "local_steps": 1,
            "local_clip_norm": 1.0,
            "local_noise_multiplier": 1.0,
            "local_target_epsilon": None,
            "global_clip_norm": 1.0,
            "global_noise_multiplier": 1.0,
            "global_target_epsilon": None,
            "delta": 1e-5,
            "reproducible_dp_noise": False,
        },
        "fedask": {
            "rank": 4,
            "oversampling": 2,
            "scaling": 1.0,
            "local_steps": 2,
            "clip_norm": 1.0,
            "noise_multiplier": 1.0,
            "target_epsilon": None,
            "delta": 1e-5,
            "reproducible_dp_noise": False,
        },
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
            "audit_interval": 2,
            "calibration_fraction": 0.5,
            "candidate_sampling": "legacy",
            "nonmember_to_member_ratio": 1.0,
            "match_candidate_labels": False,
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
        },
    }


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Federated prompt tuning membership-privacy benchmark"
    )
    parser.add_argument("--config", default="configs/fedprompt_privacy.yaml")
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
        "--aggregator",
        choices=["promptfl", "fedotp", "fedpgp", "fedavg", "dpfpl", "fedask"],
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
        "aggregator",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
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
