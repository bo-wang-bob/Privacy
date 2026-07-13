import argparse
import datetime
import json
import logging
import os
import random
from typing import Optional, Union

import numpy as np
import torch
import yaml
from transformers import CLIPModel, CLIPProcessor

from aggregator.aggregator_builder import build_aggregator
from servers.serverbase import ServerBase
from trainmodel.custom_clip import CustomCLIP, get_default_prompt_template
from utils.constants import SUPPORTED_FPL_ATTACKS
from utils.data_loader import generate_dirichlet_split, generate_iid_split
from utils.trigger import trigger_create_funcs

logger = logging.getLogger(__name__)


def parse_poisonratio_arg(value: str) -> Union[int, float]:
    value = value.strip()
    if value.lstrip("+-").isdigit():
        return int(value)
    return float(value)


def normalize_poisonratio_for_mode(
    poisonratio: Union[int, float],
    batch_size: int,
    fpl: bool,
) -> int:
    if not fpl:
        raise ValueError("This branch only supports federated prompt learning.")
    if isinstance(poisonratio, float) and not poisonratio.is_integer():
        raise ValueError(
            "In FPL mode, poisonratio must be an integer count per batch; "
            f"got {poisonratio}."
        )
    poison_count = int(poisonratio)
    if poison_count < 0:
        raise ValueError("poisonratio must be non-negative.")
    if poison_count > batch_size:
        raise ValueError(
            f"poisonratio cannot exceed batch_size ({poison_count} > {batch_size})."
        )
    return poison_count


def validate_minimal_scope(
    *,
    attack_method: str,
    defense: str,
    fpl: bool,
    train_mode: str,
) -> None:
    attack_method = attack_method.lower()
    if not fpl:
        raise ValueError("This branch only supports FPL; set fpl=true.")
    if attack_method not in SUPPORTED_FPL_ATTACKS:
        raise ValueError(
            f"Unsupported attack {attack_method!r}; this branch only supports "
            f"{', '.join(SUPPORTED_FPL_ATTACKS)}."
        )
    if defense.lower() != "seismograph":
        raise ValueError(
            f"Unsupported defense {defense!r}; this branch only supports 'seismograph'."
        )
    if train_mode.lower() not in {"centralized", "local"}:
        raise ValueError("train_mode must be 'centralized' or 'local'.")


def _build_output_tag(
    *,
    dataset_name: str,
    train_mode: str,
    attack_method: str,
    attack_start: int,
    attack_interval: int,
    attack_end: int,
    total_users: int,
    malnum: int,
    user_per_round: int,
    batch_size: int,
    learning_rate: float,
    local_epochs: int,
    local_poison_epochs: int,
    dirichlet_alpha: float,
    fpl_shots: Optional[int],
    trigger_optimization_interval: int,
) -> str:
    shot_tag = f"_shot{fpl_shots}" if fpl_shots is not None else ""
    trigger_interval_tag = (
        f"_toi{trigger_optimization_interval}"
        if trigger_optimization_interval != 1
        else ""
    )
    return (
        f"{dataset_name}_{train_mode}"
        f"_atk{attack_method}_as{attack_start}_ai{attack_interval}_ae{attack_end}"
        f"_defseismograph_u{total_users}_m{malnum}_s{user_per_round}"
        f"_bs{batch_size}_lr{learning_rate}"
        f"_le{local_epochs}_lpe{local_poison_epochs}"
        f"_dir{dirichlet_alpha}_fpl1{shot_tag}{trigger_interval_tag}"
    )


def _load_local_clip(cache_dir: str, device: torch.device):
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
            "FPL requires a local CLIP cache. Expected "
            f"'openai/clip-vit-base-patch32' under {cache_dir!r}; "
            "pre-download it before running this offline branch."
        ) from error
    return processor, clip_model


def main(
    train_mode,
    dataset_name,
    batch_size,
    learning_rate,
    num_glob_iters,
    local_epochs,
    local_poison_epochs,
    total_users,
    user_per_round,
    malnum,
    poisonratio,
    attack_method,
    attack_start,
    attack_interval,
    attack_end,
    defense,
    dirichlet_alpha,
    gpu,
    model_load_path=None,
    fpl=True,
    fpl_shots: Optional[int] = 16,
    cache_dir="./checkpoints/clip-vit-base-patch32",
    defense_params=None,
    n_ctx: int = 32,
    class_specific_ctx: bool = False,
    save_models: bool = False,
    trigger_size: Optional[int] = None,
    poison_label: int = 7,
    eval_interval: int = 5,
    eval_batch_size: int = 64,
    trigger_optimization_interval: int = 1,
):
    attack_method = attack_method.lower()
    defense = defense.lower()
    train_mode = train_mode.lower()
    validate_minimal_scope(
        attack_method=attack_method,
        defense=defense,
        fpl=fpl,
        train_mode=train_mode,
    )
    poisonratio = normalize_poisonratio_for_mode(poisonratio, batch_size, fpl=True)
    if attack_start >= attack_end:
        raise ValueError("attack_start must be less than attack_end.")
    if attack_interval <= 0 or trigger_optimization_interval <= 0:
        raise ValueError("Attack and trigger-optimization intervals must be positive.")
    if not 1 <= user_per_round <= total_users:
        raise ValueError("sample_users must be in [1, total_users].")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    output_tag = _build_output_tag(
        dataset_name=dataset_name,
        train_mode=train_mode,
        attack_method=attack_method,
        attack_start=attack_start,
        attack_interval=attack_interval,
        attack_end=attack_end,
        total_users=total_users,
        malnum=malnum,
        user_per_round=user_per_round,
        batch_size=batch_size,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        local_poison_epochs=local_poison_epochs,
        dirichlet_alpha=dirichlet_alpha,
        fpl_shots=fpl_shots,
        trigger_optimization_interval=trigger_optimization_interval,
    )
    logger_path = os.path.join("logs", f"{output_tag}_{current_time}.log")
    os.makedirs(os.path.dirname(logger_path), exist_ok=True)
    logging.basicConfig(
        filename=logger_path,
        level=logging.INFO,
        format=(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    trigger_list, trigger_pattern_list = trigger_create_funcs[attack_method](
        dataset_name,
        device,
        malnum,
        total_users=total_users,
        fpl=True,
        trigger_size=trigger_size,
    )
    if attack_method == "sabre":
        trigger_list = [torch.zeros_like(trigger) for trigger in trigger_list]
        logger.info("SABRE: initialized the bounded perturbation to zero.")

    result_dir = os.path.join("results", f"{output_tag}_{current_time}")
    os.makedirs(result_dir, exist_ok=True)
    run_config = {
        "train_mode": train_mode,
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "num_global_iters": num_glob_iters,
        "local_epochs": local_epochs,
        "local_poison_epochs": local_poison_epochs,
        "total_users": total_users,
        "sample_users": user_per_round,
        "malnum": malnum,
        "poisonratio": poisonratio,
        "poison_label": poison_label,
        "attack_method": attack_method,
        "attack_start": attack_start,
        "attack_interval": attack_interval,
        "attack_end": attack_end,
        "trigger_optimization_interval": trigger_optimization_interval,
        "defense": "seismograph",
        "defense_params": defense_params or {},
        "dirichlet_alpha": dirichlet_alpha,
        "gpu": gpu,
        "model_load_path": model_load_path,
        "fpl": True,
        "fpl_shots": fpl_shots,
        "cache_dir": cache_dir,
        "n_ctx": n_ctx,
        "class_specific_ctx": class_specific_ctx,
        "save_models": save_models,
        "trigger_size": trigger_size,
        "eval_interval": eval_interval,
    }
    with open(
        os.path.join(result_dir, "run_config.yaml"),
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(run_config, file, sort_keys=True, allow_unicode=False)

    attack_rounds = list(range(attack_start, attack_end, attack_interval))
    if dirichlet_alpha >= 10:
        train_sets, test_sets, class_names = generate_iid_split(
            dataset_name,
            num_users=total_users,
            fpl=True,
            fpl_shots=fpl_shots,
        )
    else:
        train_sets, test_sets, class_names = generate_dirichlet_split(
            dataset_name,
            total_users,
            dirichlet_alpha,
            fpl=True,
            fpl_shots=fpl_shots,
        )
    if not 0 <= poison_label < len(class_names):
        raise ValueError(
            f"poison_label must be in [0, {len(class_names) - 1}] for "
            f"{dataset_name}; got {poison_label}."
        )

    processor, clip_model = _load_local_clip(cache_dir, device)
    model = CustomCLIP(
        clip_model=clip_model,
        processor=processor,
        classnames=class_names,
        device=device,
        n_ctx=n_ctx,
        template=get_default_prompt_template(dataset_name),
        class_specific_ctx=class_specific_ctx,
    )

    def collate_fn(batch):
        images, labels = zip(*batch)
        processed = processor(images=list(images), return_tensors="pt")
        return processed["pixel_values"], torch.as_tensor(labels, dtype=torch.long)

    aggregator = build_aggregator(
        "seismograph",
        device=device,
        **(defense_params or {}),
    )
    server = ServerBase(
        train_mode=train_mode,
        fpl=True,
        device=device,
        dataset_name=dataset_name,
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=class_names,
        model=model,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        num_glob_iters=num_glob_iters,
        local_epochs=local_epochs,
        local_poison_epochs=local_poison_epochs,
        total_users=total_users,
        malnum=malnum,
        malclient_ids=list(range(malnum)),
        poisonratio=poisonratio,
        poison_label=poison_label,
        attack_method=attack_method,
        defense="seismograph",
        results_dir=result_dir,
        user_per_round=user_per_round,
        aggregator=aggregator,
        model_load_path=model_load_path,
        save_models=save_models,
        collate_fn=collate_fn,
        eval_interval=eval_interval,
        trigger_optimization_interval=trigger_optimization_interval,
    )
    server.train(
        pattern_list=trigger_pattern_list,
        trigger_list=trigger_list,
        attack_rounds=attack_rounds,
    )
    logger.info("Training completed.")


def load_config_from_yaml(yaml_path: Optional[str]) -> dict:
    if not yaml_path:
        return {}
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _parse_params(value: str) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        if not os.path.exists(value):
            raise ValueError(
                "defense_params must be a JSON object or an existing YAML path."
            )
        with open(value, "r", encoding="utf-8") as file:
            parsed = yaml.safe_load(file) or {}
    if not isinstance(parsed, dict):
        raise ValueError("defense_params must resolve to a mapping.")
    return parsed


def parse_args():
    defaults = {
        "train_mode": "centralized",
        "dataset_name": "caltech101",
        "batch_size": 16,
        "eval_batch_size": 64,
        "learning_rate": 0.001,
        "num_global_iters": 60,
        "local_epochs": 2,
        "local_poison_epochs": 2,
        "total_users": 10,
        "sample_users": 10,
        "malnum": 2,
        "poisonratio": 4,
        "poison_label": 7,
        "attack_method": "a3fl",
        "attack_start": 20,
        "attack_interval": 1,
        "attack_end": 60,
        "trigger_optimization_interval": 2,
        "defense": "seismograph",
        "defense_params": {},
        "gpu": 0,
        "dirichlet_alpha": 0.1,
        "model_load_path": None,
        "seed": 42,
        "fpl": True,
        "fpl_shots": 16,
        "cache_dir": "./checkpoints/clip-vit-base-patch32",
        "n_ctx": 32,
        "class_specific_ctx": False,
        "save_models": False,
        "trigger_size": None,
        "eval_interval": 5,
    }

    parser = argparse.ArgumentParser(
        description="SEISMOGRAPH FPL Cerberus/A3FL/SABRE experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_mode", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_global_iters", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--local_poison_epochs", type=int, default=None)
    parser.add_argument("--total_users", type=int, default=None)
    parser.add_argument("--sample_users", type=int, default=None)
    parser.add_argument("--malnum", type=int, default=None)
    parser.add_argument("--poisonratio", type=parse_poisonratio_arg, default=None)
    parser.add_argument("--poison_label", type=int, default=None)
    parser.add_argument(
        "--attack_method",
        type=str,
        choices=SUPPORTED_FPL_ATTACKS,
        default=None,
    )
    parser.add_argument("--attack_start", type=int, default=None)
    parser.add_argument("--attack_interval", type=int, default=None)
    parser.add_argument("--attack_end", type=int, default=None)
    parser.add_argument("--trigger_optimization_interval", type=int, default=None)
    parser.add_argument(
        "--defense", type=str, choices=("seismograph",), default=None
    )
    parser.add_argument("--defense_params", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--dirichlet_alpha", type=float, default=None)
    parser.add_argument("--model_load_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--fpl",
        type=lambda value: value.lower() == "true",
        default=None,
    )
    parser.add_argument("--fpl_shots", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--n_ctx", type=int, default=None)
    parser.add_argument(
        "--class_specific_ctx",
        type=lambda value: value.lower() == "true",
        default=None,
    )
    parser.add_argument(
        "--save_models",
        type=lambda value: value.lower() == "true",
        default=None,
    )
    parser.add_argument("--trigger_size", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)

    preliminary_args, _ = parser.parse_known_args()
    yaml_config = load_config_from_yaml(preliminary_args.config)
    unknown_keys = sorted(set(yaml_config) - set(defaults))
    if unknown_keys:
        raise ValueError(
            "Unsupported configuration keys in this minimal branch: "
            + ", ".join(unknown_keys)
        )

    final_config = defaults.copy()
    for key, value in yaml_config.items():
        final_config[key] = value

    args = parser.parse_args()
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        final_config[key] = (
            _parse_params(value)
            if key == "defense_params" and isinstance(value, str)
            else value
        )
    if isinstance(final_config["defense_params"], str):
        final_config["defense_params"] = _parse_params(
            final_config["defense_params"]
        )
    if final_config["fpl_shots"] is not None and final_config["fpl_shots"] <= 0:
        final_config["fpl_shots"] = None
    validate_minimal_scope(
        attack_method=final_config["attack_method"],
        defense=final_config["defense"],
        fpl=final_config["fpl"],
        train_mode=final_config["train_mode"],
    )

    final_args = argparse.Namespace(**final_config)
    final_args.config = args.config
    return final_args


if __name__ == "__main__":
    arguments = parse_args()
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)

    print("=" * 80)
    print("SEISMOGRAPH experiment")
    for label, value in (
        ("Training mode", arguments.train_mode),
        ("Attack", arguments.attack_method),
        ("Defense", arguments.defense),
        ("Dataset", arguments.dataset_name),
        ("Users", f"{arguments.sample_users}/{arguments.total_users}"),
        ("Malicious users", arguments.malnum),
        ("Poison samples/batch", arguments.poisonratio),
        ("GPU", arguments.gpu),
        ("Seed", arguments.seed),
    ):
        print(f"{label:<30}: {value}")
    print("=" * 80)

    try:
        main(
            train_mode=arguments.train_mode,
            dataset_name=arguments.dataset_name.lower(),
            batch_size=arguments.batch_size,
            eval_batch_size=arguments.eval_batch_size,
            learning_rate=arguments.learning_rate,
            num_glob_iters=arguments.num_global_iters,
            local_epochs=arguments.local_epochs,
            local_poison_epochs=arguments.local_poison_epochs,
            total_users=arguments.total_users,
            user_per_round=arguments.sample_users,
            malnum=arguments.malnum,
            poisonratio=arguments.poisonratio,
            poison_label=arguments.poison_label,
            attack_method=arguments.attack_method,
            attack_start=arguments.attack_start,
            attack_interval=arguments.attack_interval,
            attack_end=arguments.attack_end,
            defense=arguments.defense,
            dirichlet_alpha=arguments.dirichlet_alpha,
            gpu=arguments.gpu,
            model_load_path=arguments.model_load_path,
            fpl=arguments.fpl,
            fpl_shots=arguments.fpl_shots,
            cache_dir=arguments.cache_dir,
            defense_params=arguments.defense_params,
            n_ctx=arguments.n_ctx,
            class_specific_ctx=arguments.class_specific_ctx,
            save_models=arguments.save_models,
            trigger_size=arguments.trigger_size,
            eval_interval=arguments.eval_interval,
            trigger_optimization_interval=arguments.trigger_optimization_interval,
        )
    except Exception:
        logger.exception("Unhandled error during training run.")
        raise
