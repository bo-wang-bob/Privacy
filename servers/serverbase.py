import copy
import csv
import logging
import os
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from aggregator.base_aggregator import BaseAggregator
from context.context import Context
from users.user import UserBase
from utils.constants import (
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    SHARED_POISON_TRAIN_ATTACKS,
    SUPPORTED_FPL_ATTACKS,
)
from utils.data_loader import SimpleDataset

logger = logging.getLogger(__name__)


class ServerBase:
    """Federated prompt-learning server with the seismograph aggregator only."""

    def __init__(
        self,
        train_mode,
        fpl,
        device,
        dataset_name,
        train_sets,
        test_sets,
        class_names,
        model,
        batch_size,
        learning_rate,
        num_glob_iters,
        local_epochs,
        local_poison_epochs,
        total_users,
        malnum,
        malclient_ids,
        poisonratio,
        poison_label,
        attack_method,
        defense,
        results_dir: str,
        user_per_round: int,
        aggregator: BaseAggregator,
        model_load_path=None,
        save_models: bool = False,
        collate_fn=None,
        eval_interval: int = 5,
        eval_batch_size: int = 64,
        trigger_optimization_interval: int = 1,
    ):
        attack_method = attack_method.lower()
        if not fpl:
            raise ValueError("This branch only supports federated prompt learning.")
        if defense.lower() != "seismograph":
            raise ValueError("This branch only supports the seismograph aggregator.")
        if attack_method not in SUPPORTED_FPL_ATTACKS:
            raise ValueError(
                f"Unsupported attack {attack_method!r}; expected one of "
                f"{', '.join(SUPPORTED_FPL_ATTACKS)}."
            )
        if train_mode not in {"centralized", "local"}:
            raise ValueError("train_mode must be 'centralized' or 'local'.")
        if total_users <= 1:
            raise ValueError("total_users must be greater than one.")
        if not 1 <= malnum < total_users:
            raise ValueError("malnum must be in [1, total_users).")
        if not 1 <= user_per_round <= total_users:
            raise ValueError("user_per_round must be in [1, total_users].")
        if batch_size <= 0 or eval_batch_size <= 0:
            raise ValueError("Training and evaluation batch sizes must be positive.")
        if eval_interval <= 0 or trigger_optimization_interval <= 0:
            raise ValueError(
                "Evaluation and trigger-optimization intervals must be positive."
            )

        self.train_mode = train_mode
        self.fpl = True
        self.device = device
        self.dataset_name = dataset_name
        self.train_sets = train_sets
        self.test_sets = test_sets
        self.class_names = class_names
        self.collate_fn = collate_fn
        self.model = model
        self.total_users_num = total_users
        self.defense = "seismograph"
        self.num_glob_iters = num_glob_iters
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.local_poison_epochs = local_poison_epochs
        self.total_train_samples = 0
        self.user_per_round = user_per_round
        self.aggregator = aggregator
        self.save_models_enabled = save_models
        self.eval_interval = eval_interval
        self.trigger_optimization_interval = trigger_optimization_interval
        self.final_eval_rounds = 5

        self.malnum = malnum
        self.malclient_ids = list(malclient_ids)
        self.poisonratio = [poisonratio] * malnum
        self.poisonlabel = poison_label
        self.attack_method = attack_method

        self.results_dir = results_dir
        self.model_save_path = os.path.join(self.results_dir, "saved_models")
        os.makedirs(self.model_save_path, exist_ok=True)
        self.detailed_csv_path = os.path.join(
            self.results_dir,
            "detailed_metrics.csv",
        )
        self.summary_csv_path = os.path.join(
            self.results_dir,
            "summary_metrics.csv",
        )
        self._initialize_metric_files()

        self.ctx = Context(
            users_num=total_users,
            model=model,
            class_names=class_names,
            mode=train_mode,
            learning_rate=learning_rate,
            fpl=True,
            results_dir=results_dir,
        )
        self.ctx.poison_label = poison_label
        self.start_round = 0

        for user_id in range(total_users):
            user = UserBase(
                device=device,
                fpl=True,
                id=user_id,
                dataset_name=dataset_name,
                train_data=train_sets[user_id],
                test_data=test_sets[user_id],
                model=model,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                local_poison_epochs=local_poison_epochs,
                defense="seismograph",
                malicious=user_id < malnum,
                collate_fn=collate_fn,
            )
            self.ctx.users.append(user)
            self.ctx.samples_num.append(user.train_samples)
            self.total_train_samples += user.train_samples

        if model_load_path is not None:
            self._load_user_models(model_load_path)
        logger.info("Finished creating the FPL seismograph server.")

    def _initialize_metric_files(self) -> None:
        with open(self.detailed_csv_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [
                    "round",
                    "user_id",
                    "is_malicious",
                    "loss",
                    "accuracy",
                    "poison_loss",
                    "poison_accuracy",
                ]
            )
        with open(self.summary_csv_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [
                    "round",
                    "all_avg_loss",
                    "all_avg_acc",
                    "all_avg_poison_loss",
                    "all_avg_poison_acc",
                    "benign_count",
                    "benign_avg_loss",
                    "benign_avg_acc",
                    "benign_avg_poison_loss",
                    "benign_avg_poison_acc",
                    "malicious_count",
                    "malicious_avg_loss",
                    "malicious_avg_acc",
                    "malicious_avg_poison_loss",
                    "malicious_avg_poison_acc",
                ]
            )

    def _load_user_models(self, model_load_path: str) -> None:
        logger.info("Loading user model states from %s", model_load_path)
        self.start_round = int(os.path.basename(os.path.normpath(model_load_path)))
        for user in self.ctx.users:
            model_path = os.path.join(model_load_path, f"{user.id}_model.pth")
            state_dict = torch.load(model_path, map_location=self.device)
            user.model.load_state_dict(state_dict, strict=False)

    def _should_evaluate_round(self, glob_iter: int) -> bool:
        final_eval_start = max(self.num_glob_iters - self.final_eval_rounds, 0)
        return glob_iter % self.eval_interval == 0 or glob_iter >= final_eval_start

    @staticmethod
    def _clone_state_dict(
        state_dict: dict[str, torch.Tensor],
        *,
        cpu: bool = False,
    ) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone() if cpu else tensor.detach().clone()
            for name, tensor in state_dict.items()
        }

    def _sample_users(self, should_attack: bool) -> list[int]:
        if not should_attack:
            return sorted(
                random.sample(range(self.total_users_num), self.user_per_round)
            )

        proportional_malicious = int(
            self.malnum / self.total_users_num * self.user_per_round
        )
        malicious_count = min(
            self.malnum,
            self.user_per_round,
            max(1, proportional_malicious),
        )
        benign_count = self.user_per_round - malicious_count
        available_benign = self.total_users_num - self.malnum
        if benign_count > available_benign:
            shift = benign_count - available_benign
            malicious_count += shift
            benign_count -= shift

        malicious_ids = random.sample(range(self.malnum), malicious_count)
        benign_ids = random.sample(
            range(self.malnum, self.total_users_num),
            benign_count,
        )
        return sorted(malicious_ids + benign_ids)

    def train(
        self,
        trigger_list,
        attack_rounds,
        pattern_list,
    ):
        if trigger_list is None or pattern_list is None:
            raise ValueError("The supported attacks require trigger and pattern lists.")
        optimized_trigger_list = copy.deepcopy(trigger_list)
        self.ctx.attack_rounds = list(attack_rounds)
        logger.info("Starting FPL training for %s global rounds", self.num_glob_iters)

        if self.save_models_enabled and self.start_round == 0:
            self.save_models(0)

        for glob_iter in range(self.start_round, self.num_glob_iters):
            if glob_iter == self.start_round:
                for user in self.ctx.users:
                    self.ctx.set_base_model_state(
                        user.id,
                        self._clone_state_dict(user.get_parameters()),
                    )
            else:
                self.ctx.continue_to_next_round()
            self.ctx.glob_iter = glob_iter
            logger.info("--------- Global Round : %s ---------", glob_iter)

            for user in self.ctx.users:
                user.set_parameters(self.ctx.get_base_model_state(user.id))

            if self._should_evaluate_round(glob_iter):
                self.test_and_log_user_metrics(
                    glob_iter,
                    optimized_trigger_list,
                    pattern_list,
                )
            else:
                logger.info(
                    "Round %s: skipped full evaluation (interval=%s).",
                    glob_iter,
                    self.eval_interval,
                )

            should_attack = glob_iter in attack_rounds
            self.ctx.user_selected = self._sample_users(should_attack)
            should_optimize_trigger = (
                should_attack
                and bool(attack_rounds)
                and (glob_iter - attack_rounds[0])
                % self.trigger_optimization_interval
                == 0
            )
            if should_optimize_trigger:
                malicious_user = next(
                    self.ctx.users[user_id]
                    for user_id in self.ctx.user_selected
                    if self.ctx.users[user_id].malicious
                )
                logger.info("%s: optimizing the trigger", self.attack_method)
                if self.attack_method == "a3fl":
                    optimized_trigger_list = self.search_trigger(
                        model=malicious_user.model,
                        trigger_list=optimized_trigger_list,
                        pattern_list=pattern_list,
                    )
                else:
                    optimized_trigger_list = self.trigger_evasion(
                        model=malicious_user.model,
                        trigger_list=optimized_trigger_list,
                        pattern_list=pattern_list,
                    )
            elif should_attack:
                logger.info(
                    "%s: skipped trigger optimization at round %s (interval=%s)",
                    self.attack_method,
                    glob_iter,
                    self.trigger_optimization_interval,
                )

            logger.info("Selected users: %s", self.ctx.user_selected)
            poisoned_model_dict: dict[int, dict[str, torch.Tensor]] = {}
            for user_id in tqdm(
                self.ctx.user_selected,
                desc=f"Round {glob_iter} - Training Users",
                unit="user",
            ):
                user = self.ctx.users[user_id]
                if user.malicious and should_attack:
                    if self.attack_method == "cerberus":
                        user.cerberus_train(
                            poison_ratio=self.poisonratio[user.id],
                            poison_label=self.poisonlabel,
                            trigger=optimized_trigger_list[user.id],
                            pattern=pattern_list[user.id],
                            poisoned_model_dict=poisoned_model_dict,
                        )
                        poisoned_model_dict[user.id] = {
                            name: parameter.detach().clone()
                            for name, parameter in user.model.named_parameters()
                            if parameter.requires_grad
                        }
                    elif self.attack_method in SHARED_POISON_TRAIN_ATTACKS:
                        user.poison_train(
                            poison_ratio=self.poisonratio[user.id],
                            poison_label=self.poisonlabel,
                            trigger=optimized_trigger_list[user.id],
                            pattern=pattern_list[user.id],
                            attack_method=self.attack_method,
                        )
                else:
                    user.train()

            for user in self.ctx.users:
                self.ctx.set_updated_model_state(
                    user.id,
                    self._clone_state_dict(user.get_parameters()),
                )
            self.ctx.text_feature_dict = {}
            self.aggregator.aggregate(self.ctx)

            if self.save_models_enabled:
                self.save_models(glob_iter + 1)

        self.save_final_analysis_artifacts(
            trigger_list=optimized_trigger_list,
            pattern_list=pattern_list,
        )
        logger.info("FPL seismograph training finished")
        return optimized_trigger_list

    @staticmethod
    def _clone_artifact(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {
                key: ServerBase._clone_artifact(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [ServerBase._clone_artifact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(ServerBase._clone_artifact(item) for item in value)
        return copy.deepcopy(value)

    def _get_latest_global_model_state(
        self,
    ) -> Optional[dict[str, torch.Tensor]]:
        if self.ctx.mode != "centralized":
            return None
        if 0 in self.ctx.new_model_state:
            return self.ctx.new_model_state[0]
        return self.ctx.base_model_state.get(0)

    def save_final_analysis_artifacts(self, trigger_list, pattern_list) -> None:
        final_save_path = os.path.join(self.model_save_path, "final_analysis")
        os.makedirs(final_save_path, exist_ok=True)
        global_state = self._get_latest_global_model_state()
        saved_trainable_names = []
        if global_state is not None:
            global_trainable_params = {}
            for name in self.ctx.trainable_param_names:
                if name in global_state:
                    global_trainable_params[name] = global_state[name].detach().cpu().clone()
                    saved_trainable_names.append(name)
            torch.save(
                global_trainable_params,
                os.path.join(final_save_path, "global_trainable_params.pth"),
            )

        trigger_by_user = {
            user_id: self._clone_artifact(trigger)
            for user_id, trigger in enumerate(trigger_list)
        }
        trigger_bundle = {
            "attack_method": self.attack_method,
            "poison_label": self.poisonlabel,
            "malicious_user_ids": self.malclient_ids,
            "num_global_rounds": self.num_glob_iters,
            "trainable_param_names": saved_trainable_names,
            "trigger_by_user": trigger_by_user,
            "pattern_by_user": self._clone_artifact(pattern_list),
        }
        torch.save(
            trigger_bundle,
            os.path.join(final_save_path, "trigger_bundle.pth"),
        )
        trigger_dir = os.path.join(final_save_path, "triggers")
        os.makedirs(trigger_dir, exist_ok=True)
        if 0 in trigger_by_user:
            torch.save(
                trigger_by_user[0],
                os.path.join(trigger_dir, "trigger_user_0.pth"),
            )

    def save_models(self, glob_iter: int) -> None:
        round_save_path = os.path.join(self.model_save_path, str(glob_iter))
        os.makedirs(round_save_path, exist_ok=True)
        for user in self.ctx.users:
            torch.save(
                self._clone_state_dict(user.get_parameters(), cpu=True),
                os.path.join(round_save_path, f"{user.id}_model.pth"),
            )
        if glob_iter != 0 and self.ctx.mode == "centralized":
            torch.save(
                self._clone_state_dict(self.ctx.get_base_model_state(0), cpu=True),
                os.path.join(round_save_path, "global_model.pth"),
            )

    def _evaluate_user_metrics(self, user, trigger_list, pattern_list) -> dict:
        (
            total_loss,
            total_correct,
            total_samples,
            poison_loss,
            poison_correct,
            poison_samples,
        ) = user.evaluate_clean_and_poison(
            poison_label=self.poisonlabel,
            attack_method=self.attack_method,
            noise_trigger=trigger_list[user.id],
            pattern=pattern_list[user.id],
        )

        def safe_average(total, count):
            return total / count if count > 0 else 0.0

        return {
            "loss": safe_average(total_loss, total_samples),
            "accuracy": safe_average(total_correct, total_samples),
            "poison_loss": safe_average(poison_loss, poison_samples),
            "poison_accuracy": safe_average(poison_correct, poison_samples),
            "total_loss": total_loss,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "total_poison_loss": poison_loss,
            "total_poison_correct": poison_correct,
            "total_poison_samples": poison_samples,
        }

    @staticmethod
    def _summarize_metrics(entries: list[tuple[UserBase, dict]]) -> tuple:
        sample_count = sum(item[1]["total_samples"] for item in entries)
        poison_sample_count = sum(
            item[1]["total_poison_samples"] for item in entries
        )

        def safe_average(total, count):
            return total / count if count > 0 else 0.0

        return (
            len(entries),
            safe_average(sum(item[1]["total_loss"] for item in entries), sample_count),
            safe_average(
                sum(item[1]["total_correct"] for item in entries), sample_count
            ),
            safe_average(
                sum(item[1]["total_poison_loss"] for item in entries),
                poison_sample_count,
            ),
            safe_average(
                sum(item[1]["total_poison_correct"] for item in entries),
                poison_sample_count,
            ),
        )

    def test_and_log_user_metrics(
        self,
        glob_iter: int,
        trigger_list,
        pattern_list,
    ) -> None:
        entries = [
            (user, self._evaluate_user_metrics(user, trigger_list, pattern_list))
            for user in self.ctx.users
        ]
        with open(self.detailed_csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for user, metrics in entries:
                writer.writerow(
                    [
                        glob_iter,
                        user.id,
                        user.malicious,
                        f"{metrics['loss']:.6f}",
                        f"{metrics['accuracy']:.6f}",
                        f"{metrics['poison_loss']:.6f}",
                        f"{metrics['poison_accuracy']:.6f}",
                    ]
                )

        all_summary = self._summarize_metrics(entries)
        benign_summary = self._summarize_metrics(
            [entry for entry in entries if not entry[0].malicious]
        )
        malicious_summary = self._summarize_metrics(
            [entry for entry in entries if entry[0].malicious]
        )
        with open(self.summary_csv_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [
                    glob_iter,
                    *[f"{value:.6f}" for value in all_summary[1:]],
                    benign_summary[0],
                    *[f"{value:.6f}" for value in benign_summary[1:]],
                    malicious_summary[0],
                    *[f"{value:.6f}" for value in malicious_summary[1:]],
                ]
            )
        logger.info(
            "Round %s summary: all_acc=%.4f, benign_acc=%.4f, malicious_acc=%.4f, "
            "all_asr=%.4f",
            glob_iter,
            all_summary[2],
            benign_summary[2],
            malicious_summary[2],
            all_summary[4],
        )

    def _collect_malicious_balanced_dataset(
        self,
        per_class: int = 4,
    ) -> list[tuple[torch.Tensor, int]]:
        dataset = []
        for label in range(self.ctx.num_classes):
            count = 0
            for user in self.ctx.users:
                if not user.malicious:
                    continue
                for images, labels in user.trainloader:
                    indices = torch.nonzero(labels == label, as_tuple=False).flatten()
                    remaining = per_class - count
                    if remaining <= 0:
                        break
                    for index in indices[:remaining].tolist():
                        dataset.append((images[index].detach().cpu().clone(), label))
                    count += min(len(indices), remaining)
                    if count >= per_class:
                        break
                if count >= per_class:
                    break
        random.shuffle(dataset)
        return dataset

    def _trigger_optimization_loader(self) -> Optional[DataLoader]:
        dataset = self._collect_malicious_balanced_dataset(per_class=4)
        if len(dataset) < self.batch_size:
            logger.warning(
                "Trigger optimization skipped: only %s balanced samples for batch size %s.",
                len(dataset),
                self.batch_size,
            )
            return None
        return DataLoader(
            SimpleDataset(dataset),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def trigger_evasion(
        self,
        model: nn.Module,
        trigger_list: list[torch.Tensor],
        pattern_list,
    ) -> list[torch.Tensor]:
        if self.attack_method not in {"cerberus", "sabre"}:
            raise ValueError("trigger_evasion only supports Cerberus and SABRE.")
        data_loader = self._trigger_optimization_loader()
        if data_loader is None:
            return [trigger.detach().clone() for trigger in trigger_list]

        attack_model = copy.deepcopy(model).to(self.device)
        attack_model.train()
        trigger = trigger_list[0].detach().clone().to(self.device).requires_grad_(True)
        is_sabre = self.attack_method == "sabre"
        optimizer = torch.optim.SGD([trigger], lr=0.01 if is_sabre else 0.1)
        channels, height, width = trigger.shape
        pattern_mask = torch.ones_like(trigger)
        for row, col in pattern_list[0]:
            pattern_mask[:, row, col] = 0

        optimize_steps = 2 if is_sabre else 10
        for _ in range(optimize_steps):
            for images, _ in data_loader:
                images = images.to(self.device)
                if is_sabre:
                    triggered_images = images + trigger.unsqueeze(0)
                else:
                    batch_mask = pattern_mask.unsqueeze(0).expand(
                        images.size(0), channels, height, width
                    )
                    batch_trigger = trigger.unsqueeze(0).expand_as(batch_mask)
                    triggered_images = images * batch_mask + batch_trigger
                target_labels = torch.full(
                    (images.size(0),),
                    self.poisonlabel,
                    dtype=torch.long,
                    device=self.device,
                )
                optimizer.zero_grad()
                F.cross_entropy(attack_model(triggered_images), target_labels).backward()
                optimizer.step()
                with torch.no_grad():
                    if is_sabre:
                        trigger.clamp_(-0.05, 0.05)
                    else:
                        for channel in range(channels):
                            lower = (0.0 - CLIP_IMAGE_MEAN[channel]) / CLIP_IMAGE_STD[channel]
                            upper = (1.0 - CLIP_IMAGE_MEAN[channel]) / CLIP_IMAGE_STD[channel]
                            trigger[channel].clamp_(lower, upper)
                        trigger.mul_(1 - pattern_mask)
        return [trigger.detach().clone() for _ in trigger_list]

    def get_adv_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        trigger: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[nn.Module, float]:
        adversarial_model = copy.deepcopy(model).to(self.device)
        adversarial_model.train()
        trainable_params = [
            parameter
            for parameter in adversarial_model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.SGD(trainable_params, lr=0.001)
        channels, height, width = trigger.shape
        for _ in range(5):
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_mask = mask.unsqueeze(0).expand(
                    images.size(0), channels, height, width
                )
                batch_trigger = trigger.unsqueeze(0).expand_as(batch_mask)
                optimizer.zero_grad()
                loss = F.cross_entropy(
                    adversarial_model(images * batch_mask + batch_trigger),
                    labels,
                )
                loss.backward()
                optimizer.step()

        similarities = [
            F.cosine_similarity(first.flatten(), second.flatten(), dim=0).item()
            for first, second in zip(adversarial_model.parameters(), model.parameters())
            if first.requires_grad and second.requires_grad
        ]
        mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return adversarial_model, mean_similarity

    def search_trigger(
        self,
        model: nn.Module,
        trigger_list: list[torch.Tensor],
        pattern_list,
    ) -> list[torch.Tensor]:
        if self.attack_method != "a3fl":
            raise ValueError("search_trigger only supports A3FL.")
        data_loader = self._trigger_optimization_loader()
        if data_loader is None:
            return [trigger.detach().clone() for trigger in trigger_list]

        attack_model = copy.deepcopy(model).to(self.device)
        attack_model.train()
        trigger = trigger_list[0].detach().clone().to(self.device).requires_grad_(True)
        optimizer = torch.optim.SGD([trigger], lr=0.1)
        channels, height, width = trigger.shape
        pattern_mask = torch.ones_like(trigger)
        for row, col in pattern_list[0]:
            pattern_mask[:, row, col] = 0

        adversarial_model = None
        adversarial_similarity = None
        for step in range(10):
            for images, _ in data_loader:
                images = images.to(self.device)
                batch_mask = pattern_mask.unsqueeze(0).expand(
                    images.size(0), channels, height, width
                )
                batch_trigger = trigger.unsqueeze(0).expand_as(batch_mask)
                triggered_images = images * batch_mask + batch_trigger
                target_labels = torch.full(
                    (images.size(0),),
                    self.poisonlabel,
                    dtype=torch.long,
                    device=self.device,
                )
                loss = F.cross_entropy(attack_model(triggered_images), target_labels)
                if adversarial_model is not None and adversarial_similarity is not None:
                    loss += 0.01 * adversarial_similarity * F.cross_entropy(
                        adversarial_model(triggered_images),
                        target_labels,
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for channel in range(channels):
                        lower = (0.0 - CLIP_IMAGE_MEAN[channel]) / CLIP_IMAGE_STD[channel]
                        upper = (1.0 - CLIP_IMAGE_MEAN[channel]) / CLIP_IMAGE_STD[channel]
                        trigger[channel].clamp_(lower, upper)
                    trigger.mul_(1 - pattern_mask)

            if (step + 1) % 4 == 0:
                adversarial_model, adversarial_similarity = self.get_adv_model(
                    model=attack_model,
                    data_loader=data_loader,
                    trigger=trigger,
                    mask=pattern_mask,
                )
                logger.info(
                    "A3FL trigger search step %s/10: adversarial similarity=%.4f",
                    step + 1,
                    adversarial_similarity,
                )
        return [trigger.detach().clone() for _ in trigger_list]
