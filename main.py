import argparse
import datetime
import logging
import os
import random

import numpy as np
import torch
import yaml

from aggregator.aggregator_builder import build_aggregator
from privacy_attacks.auditor import SUPPORTED_ATTACKS
from servers.serverbase import ServerBase
from utils.data_loader import generate_dirichlet_split, generate_iid_split

logger = logging.getLogger(__name__)


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
    if config["aggregator"].lower() != "fedavg":
        raise ValueError("This privacy branch intentionally supports plain FedAvg only.")
    if config["total_users"] <= 1:
        raise ValueError("total_users must be greater than one.")
    if not 1 <= config["sample_users"] <= config["total_users"]:
        raise ValueError("sample_users must be in [1, total_users].")
    audit = config.get("audit", {})
    attacks = set(audit.get("attacks", []))
    unknown = sorted(attacks - SUPPORTED_ATTACKS)
    if unknown:
        raise ValueError(f"Unknown membership attacks: {', '.join(unknown)}")
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
    if audit.get("enabled", True) and attacks & spatial and config["train_mode"] != "centralized":
        raise ValueError(
            "Cross-client membership attacks require a shared centralized FedAvg model."
        )


def run(config: dict) -> list[dict]:
    from trainmodel.custom_clip import CustomCLIP, get_default_prompt_template

    validate_config(config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu"
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(
        config["results_dir"],
        f"{config['dataset_name']}_fedprompt_privacy_{timestamp}",
    )
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "run_config.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(result_dir, "run.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if float(config["dirichlet_alpha"]) >= 10:
        train_sets, test_sets, class_names = generate_iid_split(
            config["dataset_name"],
            num_users=config["total_users"],
            fpl=True,
            fpl_shots=config["fpl_shots"],
        )
    else:
        train_sets, test_sets, class_names = generate_dirichlet_split(
            config["dataset_name"],
            config["total_users"],
            config["dirichlet_alpha"],
            fpl=True,
            fpl_shots=config["fpl_shots"],
        )

    processor, clip_model = _load_local_clip(config["cache_dir"], device)
    model = CustomCLIP(
        clip_model=clip_model,
        processor=processor,
        classnames=class_names,
        device=device,
        n_ctx=config["n_ctx"],
        template=get_default_prompt_template(config["dataset_name"]),
        class_specific_ctx=config["class_specific_ctx"],
    )

    def collate_fn(batch):
        images, labels = zip(*batch)
        processed = processor(images=list(images), return_tensors="pt")
        return processed["pixel_values"], torch.as_tensor(labels, dtype=torch.long)

    audit_config = dict(config.get("audit", {}))
    audit_config.setdefault("seed", seed)
    server = ServerBase(
        train_mode=config["train_mode"],
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
        aggregator=build_aggregator("fedavg", device=device),
        model_load_path=config.get("model_load_path"),
        save_models=config["save_models"],
        collate_fn=collate_fn,
        eval_interval=config["eval_interval"],
        audit_config=audit_config,
    )
    summaries = server.train()
    logger.info("Privacy audit completed: %s", summaries)
    return summaries


def default_config() -> dict:
    return {
        "train_mode": "centralized",
        "dataset_name": "caltech101",
        "batch_size": 16,
        "eval_batch_size": 64,
        "learning_rate": 0.001,
        "num_global_iters": 20,
        "local_epochs": 2,
        "total_users": 10,
        "sample_users": 10,
        "aggregator": "fedavg",
        "dirichlet_alpha": 0.1,
        "gpu": 0,
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
            "strict": True,
            "target_client_id": 0,
            "ensure_target_participation": True,
            "attacks": [
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
            "active_max_samples": 16,
            "active_ascent_steps": 1,
            "active_ascent_lr": 0.01,
            "codepoison_weight": 1.0,
            "synthetic_mean": 0.0,
            "synthetic_std": 0.1,
            "auxiliary_fraction": 0.5,
            "rmia_offline_a": 0.3,
            "rmia_gamma": 1.0,
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
            "query_max_samples": 16,
            "query_reference_models": 2,
            "query_epsilon": 0.1,
            "yoqo_steps": 20,
            "yoqo_learning_rate": 0.01,
            "yoqo_distortion_weight": 1.0,
            "canary_num_queries": 2,
            "canary_steps": 20,
            "canary_shadow_steps": 3,
            "canary_learning_rate": 0.01,
            "canary_shadow_learning_rate": 0.02,
            "promptmia_max_samples": 16,
            "promptmia_keys": 4,
            "promptmia_delta_min": 0.02,
            "promptmia_similarity_span": 0.05,
        },
    }


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Federated prompt tuning membership-privacy benchmark"
    )
    parser.add_argument("--config", default="configs/fedprompt_privacy.yaml")
    parser.add_argument("--dataset_name")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_global_iters", type=int)
    parser.add_argument("--sample_users", type=int)
    parser.add_argument("--target_client_id", type=int)
    parser.add_argument("--audit_attacks", help="Comma-separated attack names")
    args = parser.parse_args()
    config = default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        audit = config["audit"] | loaded.pop("audit", {})
        config.update(loaded)
        config["audit"] = audit
    for key in ("dataset_name", "gpu", "seed", "num_global_iters", "sample_users"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.target_client_id is not None:
        config["audit"]["target_client_id"] = args.target_client_id
    if args.audit_attacks:
        config["audit"]["attacks"] = [
            item.strip() for item in args.audit_attacks.split(",") if item.strip()
        ]
    return config


if __name__ == "__main__":
    run(parse_args())
