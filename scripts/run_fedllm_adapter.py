#!/usr/bin/env python3
"""Federated text Adapter fine-tuning for BERT-Base or GPT2-Large."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import logging
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aggregator.aggregator_builder import build_aggregator
from servers.serverbase import ServerBase
from trainmodel.transformer_adapter import TransformerAdapterClassifier
from utils.text_data_loader import (
    load_federated_text_classification,
    normalize_text_dataset_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune BERT-Base or GPT2-Large on federated SST-5, CoLA, "
            "or IMDB using a ratio-2 Adapter after every Transformer layer."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/bert_base_sst5_adapter.yaml",
        help="YAML configuration for this single training task.",
    )
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-dir")
    parser.add_argument(
        "--defense",
        choices=("none", "iclr", "record_dp", "local_client_dp"),
        help="Override defense.name from the YAML configuration.",
    )
    parser.add_argument(
        "--target-epsilon",
        type=float,
        help=(
            "Override the selected DP defense target_epsilon and automatically "
            "calibrate its noise multiplier."
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        help="Override the Record-DP per-sequence clipping threshold C.",
    )
    parser.add_argument(
        "--max-client-update-norm",
        type=float,
        help=(
            "Override the local client-DP joint client-gradient clipping "
            "threshold S."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("sst5", "cola", "imdb"),
        help=(
            "Dataset preset. The local path is resolved as a sibling of the "
            "configured dataset_path."
        ),
    )
    parser.add_argument(
        "--attacks",
        help=(
            "Comma-separated attacks. Defaults to the attacks in the YAML "
            "configuration."
        ),
    )
    parser.add_argument("--target-client-id", type=int)
    parser.add_argument(
        "--projres",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable strict observed-batch ProjRes.",
    )
    parser.add_argument(
        "--skip-projres",
        action="store_false",
        dest="projres",
        default=None,
        help="Alias matching the CLIP sweep: disable strict ProjRes.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config_path = (REPOSITORY_ROOT / args.config).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config["config_path"] = str(config_path)
    if args.gpu is not None:
        config["gpu"] = args.gpu
    if args.rounds is not None:
        config["num_global_iters"] = args.rounds
    if args.seed is not None:
        config["seed"] = args.seed
    if args.results_dir is not None:
        config["results_dir"] = args.results_dir
    if args.defense is not None:
        config.setdefault("defense", {})["name"] = args.defense
    if args.target_epsilon is not None:
        defense = config.setdefault("defense", {})
        if str(defense.get("name", "none")).lower() not in {
            "record_dp",
            "local_client_dp",
        }:
            raise ValueError(
                "--target-epsilon requires a DP defense."
            )
        defense["target_epsilon"] = float(args.target_epsilon)
        defense["noise_multiplier"] = "auto"
    max_grad_norm = getattr(args, "max_grad_norm", None)
    if max_grad_norm is not None:
        defense = config.setdefault("defense", {})
        if str(defense.get("name", "none")).lower() != "record_dp":
            raise ValueError(
                "--max-grad-norm requires defense.name=record_dp."
            )
        defense["max_grad_norm"] = float(max_grad_norm)
    max_client_update_norm = getattr(args, "max_client_update_norm", None)
    if max_client_update_norm is not None:
        defense = config.setdefault("defense", {})
        if str(defense.get("name", "none")).lower() != "local_client_dp":
            raise ValueError(
                "--max-client-update-norm requires "
                "defense.name=local_client_dp."
            )
        defense["max_update_norm"] = float(max_client_update_norm)
    if args.dataset is not None:
        configured_path = Path(str(config["dataset_path"]))
        config["dataset_name"] = args.dataset
        config["dataset_path"] = str(configured_path.parent / args.dataset)
    if args.attacks is not None:
        audit = config.setdefault("audit", {})
        audit["attacks"] = [
            attack.strip() for attack in args.attacks.split(",") if attack.strip()
        ]
        if "exact_batch_membership_attacks" in audit:
            audit["exact_batch_membership_attacks"] = [
                attack
                for attack in audit["exact_batch_membership_attacks"]
                if attack in audit["attacks"]
            ]
    if args.target_client_id is not None:
        config.setdefault("audit", {})["target_client_id"] = (
            args.target_client_id
        )
        config["audit"]["audit_client_ids"] = [args.target_client_id]
    if args.projres is not None:
        config.setdefault("projres", {})["enabled"] = args.projres
        if args.projres is False:
            audit = config.setdefault("audit", {})
            audit["attacks"] = [
                attack
                for attack in audit.get("attacks", [])
                if attack != "projres"
            ]
            audit["exact_batch_membership_attacks"] = [
                attack
                for attack in audit.get(
                    "exact_batch_membership_attacks", []
                )
                if attack != "projres"
            ]
    if args.require_cuda is not None:
        config["require_cuda"] = args.require_cuda
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    architecture = str(config.get("architecture", "")).lower()
    expected_type = f"{architecture}_adapter"
    if architecture not in {"bert", "gpt2"}:
        raise ValueError("architecture must be bert or gpt2.")
    if str(config.get("model_type", "")).lower() != expected_type:
        raise ValueError(f"model_type must be {expected_type}.")
    dataset_name = normalize_text_dataset_name(config.get("dataset_name", ""))
    config["dataset_name"] = dataset_name
    config["primary_metric"] = "mcc" if dataset_name == "cola" else "accuracy"
    if int(config.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be positive.")
    if int(config.get("total_users", 0)) <= 1:
        raise ValueError("total_users must be greater than one.")
    if int(config.get("sample_users", 0)) != int(config["total_users"]):
        raise ValueError("Synchronous paper mode requires all clients each round.")
    for key in ("num_global_iters", "eval_batch_size", "eval_interval"):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive.")
    if int(config.get("learning_rate_decay_interval", 1)) <= 0:
        raise ValueError("learning_rate_decay_interval must be positive.")
    if float(config.get("learning_rate", 0.0)) <= 0:
        raise ValueError("learning_rate must be positive.")
    if not 0 < float(config.get("learning_rate_decay", 1.0)) <= 1:
        raise ValueError("learning_rate_decay must be in (0, 1].")
    adapter = dict(config.get("adapter", {}))
    if int(adapter.get("reduction", 0)) != 2:
        raise ValueError("The paper Adapter requires reduction=2.")
    if str(config.get("aggregation_weighting", "uniform")) != "uniform":
        raise ValueError("Paper FedSGD aggregates client updates uniformly.")
    optimization = dict(config.get("optimization", {}))
    if str(optimization.get("client_optimizer", "sgd")).lower() != "sgd":
        raise ValueError("FedLLM privacy runs require one-batch SGD uploads.")
    if float(optimization.get("max_grad_norm", 0.0)) < 0:
        raise ValueError("max_grad_norm must be non-negative.")
    supported_attacks = {
        "blackbox_loss",
        "loss_series",
        "grad_cosine",
        "avg_cosine",
        "fedmia_loss",
        "fedmia_cosine",
        "gradient_diff",
        "score_diff",
        "score_ratio",
        "fta",
        "projres",
    }
    audit = dict(config.get("audit", {}))
    unknown_attacks = sorted(set(audit.get("attacks", [])) - supported_attacks)
    if unknown_attacks:
        raise ValueError(
            "Text Adapter experiments currently support these common attacks: "
            + ", ".join(sorted(supported_attacks))
            + ". Unsupported: "
            + ", ".join(unknown_attacks)
        )
    exact_batch_attacks = audit.get("exact_batch_membership_attacks", [])
    if not isinstance(exact_batch_attacks, list):
        raise ValueError(
            "audit.exact_batch_membership_attacks must be a list."
        )
    supported_exact_batch_attacks = {
        "blackbox_loss",
        "grad_cosine",
        "gradient_diff",
        "projres",
        "score_diff",
        "score_ratio",
    }
    unknown_exact_batch_attacks = sorted(
        set(exact_batch_attacks) - supported_exact_batch_attacks
    )
    if unknown_exact_batch_attacks:
        raise ValueError(
            "Exact-batch membership supports only: "
            + ", ".join(sorted(supported_exact_batch_attacks))
        )
    missing_exact_batch_attacks = sorted(
        set(exact_batch_attacks) - set(audit.get("attacks", []))
    )
    if missing_exact_batch_attacks:
        raise ValueError(
            "Exact-batch attacks must also appear in audit.attacks: "
            + ", ".join(missing_exact_batch_attacks)
        )
    if "projres" in audit.get("attacks", []) and "projres" not in set(
        exact_batch_attacks
    ):
        raise ValueError(
            "Text Adapter ProjRes must use the shared exact-batch membership "
            "protocol."
        )
    exact_batch_ratio = audit.get(
        "exact_batch_nonmember_to_member_ratio",
        audit.get("nonmember_to_member_ratio", 1),
    )
    if (
        isinstance(exact_batch_ratio, bool)
        or int(exact_batch_ratio) < 1
        or float(exact_batch_ratio) != int(exact_batch_ratio)
    ):
        raise ValueError(
            "audit.exact_batch_nonmember_to_member_ratio must be a positive "
            "integer."
        )
    if float(audit.get("score_ratio_damping", 1e-6)) <= 0:
        raise ValueError("audit.score_ratio_damping must be positive.")
    if str(audit.get("fta_measurement", "confidence")).lower() not in {
        "confidence",
        "loss",
    }:
        raise ValueError("audit.fta_measurement must be confidence or loss.")
    candidate_sampling = str(
        audit.get("candidate_sampling", "legacy")
    ).lower()
    if candidate_sampling not in {
        "legacy",
        "balanced_global_holdout",
    }:
        raise ValueError(
            "Text Adapter experiments support candidate_sampling=legacy or "
            "balanced_global_holdout."
        )
    if exact_batch_attacks and candidate_sampling != "balanced_global_holdout":
        raise ValueError(
            "Exact-batch membership requires "
            "candidate_sampling=balanced_global_holdout."
        )
    if bool(audit.get("require_full_target_train_members", False)) and (
        candidate_sampling != "balanced_global_holdout"
    ):
        raise ValueError(
            "require_full_target_train_members requires "
            "candidate_sampling=balanced_global_holdout."
        )
    if candidate_sampling == "balanced_global_holdout":
        ratio = float(audit.get("nonmember_to_member_ratio", 1.0))
        minimum_nonmembers = int(
            audit.get("low_fpr_min_nonmembers", 1000)
        )
        maximum_members = int(audit.get("low_fpr_max_members", 0))
        maximum_nonmembers = int(
            audit.get("low_fpr_max_nonmembers", 0)
        )
        if ratio < 1 or not ratio.is_integer():
            raise ValueError(
                "balanced_global_holdout requires a positive integer "
                "nonmember_to_member_ratio."
            )
        if maximum_members < 0 or maximum_members == 1:
            raise ValueError(
                "balanced_global_holdout requires low_fpr_max_members to be "
                "0 (unlimited) or at least 2."
            )
        if minimum_nonmembers < 2:
            raise ValueError(
                "balanced_global_holdout requires low_fpr_min_nonmembers >= 2."
            )
        if maximum_nonmembers < 0 or maximum_nonmembers == 1:
            raise ValueError(
                "balanced_global_holdout requires low_fpr_max_nonmembers to "
                "be 0 (unlimited) or at least 2."
            )
        if 0 < maximum_nonmembers < minimum_nonmembers:
            raise ValueError(
                "balanced_global_holdout requires low_fpr_max_nonmembers >= "
                "low_fpr_min_nonmembers."
            )
        if (
            maximum_members > 0
            and maximum_nonmembers > 0
            and maximum_nonmembers < maximum_members * ratio
        ):
            raise ValueError(
                "balanced_global_holdout requires enough non-member capacity "
                "for the configured member cap and ratio."
            )
    defense = dict(config.get("defense", {"name": "none"}))
    defense_name = str(defense.get("name", "none")).lower()
    if defense_name not in {"none", "iclr", "record_dp", "local_client_dp"}:
        raise ValueError(
            "Text Adapter experiments support defense=none, iclr, record_dp, "
            "or local_client_dp."
        )
    if defense_name == "iclr":
        if architecture != "bert":
            raise ValueError("Text Adapter ICLR currently supports BERT only.")
        interval = int(defense.get("iclr_analysis_interval", 50))
        if interval <= 0:
            raise ValueError("defense.iclr_analysis_interval must be positive.")
        if interval > int(config["num_global_iters"]):
            raise ValueError(
                "defense.iclr_analysis_interval cannot exceed num_global_iters."
            )
        if str(
            defense.get("iclr_analysis_timing", "post_round")
        ).lower() != "post_round":
            raise ValueError(
                "BERT ICLR requires iclr_analysis_timing=post_round."
            )
        top_fraction = float(
            defense.get("iclr_validation_top_fraction", 0.2)
        )
        if not 0.0 < top_fraction <= 0.5:
            raise ValueError(
                "defense.iclr_validation_top_fraction must be in (0, 0.5]."
            )
    if defense_name == "record_dp":
        max_norm = float(
            defense.get("max_grad_norm", defense.get("dp_max_grad_norm", 1.0))
        )
        delta = float(defense.get("delta", defense.get("dp_delta", 1e-5)))
        if max_norm <= 0:
            raise ValueError("defense.max_grad_norm must be positive.")
        if not 0 < delta < 1:
            raise ValueError("Record-DP delta must be in (0, 1).")
        if str(defense.get("adjacency", "add_remove")).lower() != "add_remove":
            raise ValueError("Record-DP currently requires add_remove adjacency.")
        if str(defense.get("sampling", "poisson")).lower() != "poisson":
            raise ValueError("Record-DP currently requires Poisson sampling.")
        if str(defense.get("accountant", "rdp")).lower() != "rdp":
            raise ValueError("Record-DP currently requires accountant=rdp.")
        backend = str(defense.get("grad_sample_backend", "auto")).lower()
        if backend not in {"auto", "loop", "vmap"}:
            raise ValueError(
                "Record-DP grad_sample_backend must be auto, loop, or vmap."
            )
        if int(defense.get("microbatch_size", 1)) <= 0:
            raise ValueError("Record-DP microbatch_size must be positive.")
        target_epsilon = defense.get("target_epsilon")
        noise = defense.get(
            "noise_multiplier", defense.get("dp_noise_multiplier")
        )
        if target_epsilon is None and noise in {None, "auto"}:
            raise ValueError(
                "Record-DP requires target_epsilon or a numeric noise_multiplier."
            )
        if target_epsilon is not None and float(target_epsilon) <= 0:
            raise ValueError("Record-DP target_epsilon must be positive.")
        if target_epsilon is not None and noise not in {None, "auto"}:
            raise ValueError(
                "Record-DP must configure target_epsilon or noise_multiplier, "
                "not both."
            )
        if target_epsilon is None and float(noise) <= 0:
            raise ValueError("Record-DP noise_multiplier must be positive.")
    if defense_name == "local_client_dp":
        max_norm = float(defense.get("max_update_norm", 1.0))
        delta = float(defense.get("delta", defense.get("dp_delta", 1e-5)))
        if str(defense.get("privacy_unit", "client")).lower() != "client":
            raise ValueError("Local client-DP requires privacy_unit=client.")
        if max_norm <= 0:
            raise ValueError("defense.max_update_norm must be positive.")
        if not 0 < delta < 1:
            raise ValueError("Local client-DP delta must be in (0, 1).")
        if str(defense.get("adjacency", "add_remove")).lower() != "add_remove":
            raise ValueError(
                "Local client-DP currently requires add_remove adjacency."
            )
        if str(
            defense.get("sampling", "full_participation")
        ).lower() != "full_participation":
            raise ValueError(
                "Local client-DP currently requires full_participation."
            )
        if str(defense.get("accountant", "rdp")).lower() != "rdp":
            raise ValueError("Local client-DP currently requires accountant=rdp.")
        target_epsilon = defense.get("target_epsilon")
        noise = defense.get(
            "noise_multiplier", defense.get("dp_noise_multiplier")
        )
        if target_epsilon is None and noise in {None, "auto"}:
            raise ValueError(
                "Local client-DP requires target_epsilon or a numeric "
                "noise_multiplier."
            )
        if target_epsilon is not None and float(target_epsilon) <= 0:
            raise ValueError("Local client-DP target_epsilon must be positive.")
        if target_epsilon is not None and noise not in {None, "auto"}:
            raise ValueError(
                "Local client-DP must configure target_epsilon or "
                "noise_multiplier, not both."
            )
        if target_epsilon is None and float(noise) <= 0:
            raise ValueError(
                "Local client-DP noise_multiplier must be positive."
            )
    target_client_id = int(audit.get("target_client_id", 0))
    if not 0 <= target_client_id < int(config["total_users"]):
        raise ValueError("audit.target_client_id is outside the client range.")
    projres = dict(config.get("projres", {}))
    if projres.get("threshold") is not None:
        raise ValueError("ProjRes is ranking-only; threshold must be null.")
    if str(projres.get("decision_mode", "ranking")).lower() != "ranking":
        raise ValueError("ProjRes decision_mode must be ranking.")
    unified_projres = "projres" in set(exact_batch_attacks)
    if unified_projres:
        if not bool(projres.get("enabled", True)):
            raise ValueError(
                "audit exact-batch ProjRes requires projres.enabled=true."
            )
        if defense_name == "record_dp":
            if any(
                int(projres.get(key, 0)) != 0
                for key in (
                    "max_candidates",
                    "min_nonmembers",
                    "max_nonmembers",
                )
            ):
                raise ValueError(
                    "Record-DP Poisson batches require dynamic unified ProjRes "
                    "candidate bounds (all three bounds must be zero)."
                )
        else:
            expected_nonmembers = int(config["batch_size"]) * int(
                exact_batch_ratio
            )
            if int(projres.get("max_candidates", 0)) != int(
                config["batch_size"]
            ):
                raise ValueError(
                    "Unified ProjRes must audit the complete real training batch."
                )
            if int(projres.get("min_nonmembers", 0)) != expected_nonmembers or int(
                projres.get("max_nonmembers", 0)
            ) != expected_nonmembers:
                raise ValueError(
                    "Unified ProjRes min_nonmembers and max_nonmembers must equal "
                    "batch_size * exact_batch_nonmember_to_member_ratio."
                )
    if "evaluation_interval" in projres and "evaluation_round" in projres:
        raise ValueError(
            "Configure only one of projres.evaluation_interval and "
            "projres.evaluation_round."
        )
    if unified_projres and "evaluation_round" in projres:
        raise ValueError(
            "Unified ProjRes is scheduled by audit.attack_audit_intervals; "
            "use projres.evaluation_interval only as the matching protocol "
            "declaration."
        )
    if "evaluation_interval" in projres:
        configured_interval = projres["evaluation_interval"]
        if isinstance(configured_interval, bool):
            raise ValueError(
                "projres.evaluation_interval must be a positive integer."
            )
        try:
            evaluation_interval = int(configured_interval)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "projres.evaluation_interval must be a positive integer."
            ) from error
        if (
            evaluation_interval <= 0
            or str(configured_interval).strip() != str(evaluation_interval)
        ):
            raise ValueError(
                "projres.evaluation_interval must be a positive integer."
            )
        if unified_projres:
            audit_interval = int(
                audit.get("attack_audit_intervals", {}).get(
                    "projres", audit.get("audit_interval", 1)
                )
            )
            if evaluation_interval != audit_interval:
                raise ValueError(
                    "Unified ProjRes evaluation_interval must match its "
                    "shared audit interval."
                )
    elif str(projres.get("evaluation_round", "last")).lower() != "last":
        evaluation_round = int(projres["evaluation_round"])
        if not 1 <= evaluation_round <= int(config["num_global_iters"]):
            raise ValueError(
                "projres.evaluation_round is outside the communication rounds."
            )


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def make_result_dir(config: dict) -> Path:
    root = resolve_path(str(config.get("results_dir", "./results")))
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    defense = str(config.get("defense", {}).get("name", "none")).lower()
    target_client = int(config.get("audit", {}).get("target_client_id", 0))
    epsilon_suffix = ""
    clipping_suffix = ""
    if defense in {"record_dp", "local_client_dp"}:
        defense_config = config.get("defense", {})
        target_epsilon = defense_config.get("target_epsilon")
        if target_epsilon is not None:
            epsilon_suffix = f"_eps{float(target_epsilon):g}"
        if defense == "record_dp":
            max_grad_norm = defense_config.get(
                "max_grad_norm", defense_config.get("dp_max_grad_norm")
            )
            if max_grad_norm is not None:
                clipping_suffix = f"_c{float(max_grad_norm):g}"
        else:
            max_update_norm = defense_config.get("max_update_norm")
            if max_update_norm is not None:
                clipping_suffix = f"_s{float(max_update_norm):g}"
    run_name = (
        f"{timestamp}_{config['model_type']}_{config['dataset_name']}_fedsgd_"
        f"{defense}{epsilon_suffix}_seed{int(config['seed'])}_"
        f"target{target_client}{clipping_suffix}"
    )
    result_dir = root / run_name
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def configure_logging(result_dir: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(result_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream, file_handler],
        force=True,
    )


def log_task_configuration(logger: logging.Logger, config: dict) -> None:
    """Emit one task's resolved settings only after that task has started."""
    audit = dict(config.get("audit", {}))
    projres = dict(config.get("projres", {}))
    defense = dict(config.get("defense", {"name": "none"}))
    rows = (
        ("model_type", config["model_type"]),
        ("dataset", config["dataset_name"]),
        ("evaluation.primary_metric", config["primary_metric"]),
        ("device", config["device"]),
        ("model_path", config["model_path"]),
        ("dataset_path", config["dataset_path"]),
        ("federated.aggregator", "fedsgd"),
        ("federated.aggregation_weighting", config["aggregation_weighting"]),
        ("federated.total_users", config["total_users"]),
        ("federated.sample_users", config["sample_users"]),
        ("federated.num_global_iters", config["num_global_iters"]),
        ("optimization.batch_size", config["batch_size"]),
        ("optimization.learning_rate", config["learning_rate"]),
        (
            "optimization.learning_rate_decay",
            config.get("learning_rate_decay", 1.0),
        ),
        (
            "optimization.learning_rate_decay_interval",
            config.get("learning_rate_decay_interval", 1),
        ),
        ("adapter.reduction", config["adapter"]["reduction"]),
        (
            "adapter.zero_init_up",
            config["adapter"].get("zero_init_up", True),
        ),
        (
            "optimization.max_grad_norm",
            config.get("optimization", {}).get("max_grad_norm", 0.0),
        ),
        ("privacy_audit.attacks", ", ".join(audit.get("attacks", []))),
        (
            "privacy_audit.exact_batch_membership_attacks",
            ", ".join(audit.get("exact_batch_membership_attacks", [])),
        ),
        (
            "privacy_audit.exact_batch_nonmember_ratio",
            audit.get("exact_batch_nonmember_to_member_ratio"),
        ),
        ("privacy_audit.target_client_id", audit.get("target_client_id", 0)),
        ("privacy_audit.candidate_sampling", audit.get("candidate_sampling")),
        ("iclr.enabled", defense.get("name", "none") == "iclr"),
        (
            "iclr.analysis_interval",
            defense.get("iclr_analysis_interval"),
        ),
        ("iclr.analysis_timing", defense.get("iclr_analysis_timing")),
        ("record_dp.enabled", defense.get("name", "none") == "record_dp"),
        ("record_dp.target_epsilon", defense.get("target_epsilon")),
        (
            "record_dp.noise_multiplier",
            defense.get("noise_multiplier", defense.get("dp_noise_multiplier")),
        ),
        (
            "record_dp.max_grad_norm",
            defense.get("max_grad_norm", defense.get("dp_max_grad_norm")),
        ),
        ("record_dp.delta", defense.get("delta", defense.get("dp_delta"))),
        ("record_dp.sampling", defense.get("sampling")),
        (
            "record_dp.grad_sample_backend",
            defense.get("grad_sample_backend"),
        ),
        (
            "local_client_dp.enabled",
            defense.get("name", "none") == "local_client_dp",
        ),
        ("local_client_dp.target_epsilon", defense.get("target_epsilon")),
        (
            "local_client_dp.noise_multiplier",
            defense.get("noise_multiplier", defense.get("dp_noise_multiplier")),
        ),
        ("local_client_dp.max_update_norm", defense.get("max_update_norm")),
        (
            "local_client_dp.delta",
            defense.get("delta", defense.get("dp_delta")),
        ),
        ("local_client_dp.sampling", defense.get("sampling")),
        ("projres.enabled", projres.get("enabled", True)),
        (
            "projres.evaluation_interval",
            projres.get("evaluation_interval"),
        ),
        (
            "projres.evaluation_round",
            projres.get(
                "evaluation_round",
                None if "evaluation_interval" in projres else "last",
            ),
        ),
    )
    logger.info("Resolved FedLLM task configuration")
    for key, value in rows:
        logger.info("  %-42s: %s", key, value)


def main() -> None:
    args = parse_args()
    config = load_config(args)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(config.get("require_cuda", False)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the selected configuration.")
    device = torch.device(
        f"cuda:{int(config.get('gpu', 0))}"
        if torch.cuda.is_available()
        else "cpu"
    )

    result_dir = make_result_dir(config)
    configure_logging(result_dir)
    logger = logging.getLogger(__name__)
    resolved_config = copy.deepcopy(config)
    resolved_config["model_path"] = str(resolve_path(config["model_path"]))
    resolved_config["dataset_path"] = str(resolve_path(config["dataset_path"]))
    resolved_config["device"] = str(device)
    with (result_dir / "run_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_config, file, sort_keys=False, allow_unicode=True)
    log_task_configuration(logger, resolved_config)

    logger.info(
        "Loading %s/%s on %s | clients=%d | batch=%d | rounds=%d",
        config["model_type"],
        config["dataset_name"],
        device,
        int(config["total_users"]),
        int(config["batch_size"]),
        int(config["num_global_iters"]),
    )
    data = load_federated_text_classification(
        dataset_name=str(config["dataset_name"]),
        dataset_path=resolved_config["dataset_path"],
        model_path=resolved_config["model_path"],
        num_users=int(config["total_users"]),
        seed=seed,
        max_length=int(config.get("max_length", 128)),
    )
    adapter = dict(config.get("adapter", {}))
    model = TransformerAdapterClassifier(
        model_path=resolved_config["model_path"],
        architecture=str(config["architecture"]),
        num_classes=len(data.class_names),
        reduction=int(adapter.get("reduction", 2)),
        activation=str(adapter.get("activation", "relu")),
        classifier_dropout=float(adapter.get("classifier_dropout", 0.0)),
        gradient_checkpointing=bool(
            adapter.get("gradient_checkpointing", False)
        ),
        zero_init_up=bool(adapter.get("zero_init_up", True)),
        device=device,
    )
    model.classnames = list(data.class_names)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    logger.info(
        "Adapter ready | layers=%d | reduction=%d | trainable=%d | frozen=%d",
        model.num_adapter_layers,
        model.reduction,
        trainable_parameters,
        frozen_parameters,
    )

    optimization = dict(config.get("optimization", {}))
    method_config = {
        "client_optimizer": str(
            optimization.get("client_optimizer", "sgd")
        ).lower(),
        "momentum": float(optimization.get("momentum", 0.0)),
        "weight_decay": float(optimization.get("weight_decay", 0.0)),
        "max_grad_norm": float(optimization.get("max_grad_norm", 0.0)),
        "seed": seed,
    }
    audit_config = copy.deepcopy(config.get("audit", {}))
    audit_config.setdefault("enabled", True)
    audit_config.setdefault("target_client_id", 0)
    audit_config.setdefault("audit_client_ids", [audit_config["target_client_id"]])
    audit_config["seed"] = seed
    audit_config["training_health_check"] = True
    defense_config = copy.deepcopy(config.get("defense", {"name": "none"}))
    defense_config.setdefault("name", "none")
    defense_config.setdefault("seed", seed)
    projres_config = copy.deepcopy(config.get("projres", {}))
    projres_config.setdefault("enabled", True)
    server = ServerBase(
        device=device,
        dataset_name=str(config["dataset_name"]),
        train_sets=data.train_sets,
        test_sets=data.test_sets,
        class_names=data.class_names,
        model=model,
        batch_size=int(config["batch_size"]),
        eval_batch_size=int(config["eval_batch_size"]),
        learning_rate=float(config["learning_rate"]),
        learning_rate_decay=float(config.get("learning_rate_decay", 1.0)),
        learning_rate_decay_interval=int(
            config.get("learning_rate_decay_interval", 1)
        ),
        num_glob_iters=int(config["num_global_iters"]),
        local_epochs=1,
        total_users=int(config["total_users"]),
        results_dir=str(result_dir),
        user_per_round=int(config["sample_users"]),
        aggregator=build_aggregator(
            "fedsgd",
            device=device,
            aggregation_weighting="uniform",
        ),
        save_models=bool(config.get("save_models", False)),
        collate_fn=data.collate_fn,
        eval_interval=int(config["eval_interval"]),
        audit_config=audit_config,
        projres_config=projres_config,
        defense_config=defense_config,
        method_config=method_config,
    )
    server.train()
    logger.info("Training complete: %s", result_dir)


if __name__ == "__main__":
    main()
