import csv
import logging
import os
import random

import torch

from aggregator.base_aggregator import BaseAggregator
from context.context import Context
from privacy_attacks.auditor import MembershipAuditor
from users.user import UserBase

logger = logging.getLogger(__name__)


class ServerBase:
    """Federated prompt-tuning server with FedAvg and privacy audit hooks."""

    def __init__(
        self,
        train_mode,
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
        total_users,
        results_dir: str,
        user_per_round: int,
        aggregator: BaseAggregator,
        model_load_path=None,
        save_models: bool = False,
        collate_fn=None,
        eval_interval: int = 5,
        eval_batch_size: int = 64,
        audit_config: dict | None = None,
    ):
        if train_mode not in {"centralized", "local"}:
            raise ValueError("train_mode must be 'centralized' or 'local'.")
        if total_users <= 1:
            raise ValueError("total_users must be greater than one.")
        if not 1 <= user_per_round <= total_users:
            raise ValueError("user_per_round must be in [1, total_users].")
        if num_glob_iters <= 0 or eval_interval <= 0:
            raise ValueError("Training rounds and eval_interval must be positive.")
        self.train_mode = train_mode
        self.device = device
        self.dataset_name = dataset_name
        self.model = model
        self.total_users_num = total_users
        self.num_glob_iters = num_glob_iters
        self.user_per_round = user_per_round
        self.aggregator = aggregator
        self.results_dir = results_dir
        self.save_models_enabled = save_models
        self.eval_interval = eval_interval
        self.audit_config = audit_config or {"enabled": True}
        self.target_client_id = int(self.audit_config.get("target_client_id", 0))
        self.ensure_target = bool(
            self.audit_config.get("ensure_target_participation", True)
        )
        os.makedirs(results_dir, exist_ok=True)
        self.metrics_path = os.path.join(results_dir, "training_metrics.csv")
        with open(self.metrics_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(("round", "loss", "accuracy", "samples"))

        self.ctx = Context(
            users_num=total_users,
            model=model,
            class_names=class_names,
            mode=train_mode,
            learning_rate=learning_rate,
            results_dir=results_dir,
        )
        code_poison_config = {
            "weight": float(self.audit_config.get("codepoison_weight", 1.0)),
            "synthetic_mean": float(self.audit_config.get("synthetic_mean", 0.0)),
            "synthetic_std": float(self.audit_config.get("synthetic_std", 0.1)),
        }
        for user_id in range(total_users):
            user = UserBase(
                device=device,
                id=user_id,
                dataset_name=dataset_name,
                train_data=train_sets[user_id],
                test_data=test_sets[user_id],
                model=model,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                collate_fn=collate_fn,
                code_poison_config=code_poison_config,
            )
            self.ctx.users.append(user)
            self.ctx.samples_num.append(user.train_samples)

        if model_load_path:
            state = torch.load(model_load_path, map_location=device)
            for user in self.ctx.users:
                user.set_parameters(state)

        self.auditor = MembershipAuditor(
            model=model,
            users=self.ctx.users,
            target_client_id=self.target_client_id,
            device=device,
            results_dir=results_dir,
            collate_fn=collate_fn,
            config=self.audit_config,
        )
        self.code_poison_enabled = "codepoison" in self.auditor.attacks

    @staticmethod
    def _clone_state(state: dict[str, torch.Tensor], cpu: bool = False):
        return {
            name: tensor.detach().cpu().clone() if cpu else tensor.detach().clone()
            for name, tensor in state.items()
        }

    def _sample_users(self) -> list[int]:
        if not self.ensure_target or self.user_per_round == self.total_users_num:
            return sorted(random.sample(range(self.total_users_num), self.user_per_round))
        others = [
            user_id for user_id in range(self.total_users_num)
            if user_id != self.target_client_id
        ]
        selected = random.sample(others, self.user_per_round - 1)
        return sorted([self.target_client_id, *selected])

    def _evaluate(self, round_index: int) -> None:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for user in self.ctx.users:
            if self.train_mode == "centralized":
                evaluation_state = self.ctx.new_model_state[0]
            else:
                evaluation_state = self.ctx.new_model_state.get(
                    user.id, self.ctx.base_model_state[user.id]
                )
            user.set_parameters(evaluation_state)
            loss, correct, samples = user.evaluate()
            total_loss += loss
            total_correct += correct
            total_samples += samples
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                (
                    round_index,
                    total_loss / max(total_samples, 1),
                    total_correct / max(total_samples, 1),
                    total_samples,
                )
            )

    def _save_round(self, round_index: int) -> None:
        if not self.save_models_enabled:
            return
        path = os.path.join(self.results_dir, "saved_models")
        os.makedirs(path, exist_ok=True)
        if self.train_mode == "centralized":
            state = self.ctx.new_model_state.get(0, self.ctx.base_model_state[0])
            torch.save(
                self._clone_state(state, cpu=True),
                os.path.join(path, f"global_round_{round_index}.pt"),
            )

    def train(self) -> list[dict]:
        for user in self.ctx.users:
            self.ctx.set_base_model_state(user.id, self._clone_state(user.get_parameters()))

        for round_index in range(self.num_glob_iters):
            if round_index > 0:
                self.ctx.continue_to_next_round()
            self.ctx.glob_iter = round_index
            selected_ids = self._sample_users()
            self.ctx.user_selected = selected_ids
            logger.info("Round %s selected clients: %s", round_index, selected_ids)

            base_state = self.ctx.get_base_model_state(self.target_client_id)
            for user_id in selected_ids:
                user = self.ctx.users[user_id]
                user.set_parameters(self.ctx.get_base_model_state(user_id))
                use_code_poison = (
                    self.code_poison_enabled and user_id == self.target_client_id
                )
                user.train(code_poison=use_code_poison)
                self.ctx.set_updated_model_state(
                    user_id, self._clone_state(user.get_parameters())
                )

            self.auditor.observe_round(
                round_index=round_index,
                base_state=base_state,
                updated_states=self.ctx.updated_model_state,
                selected_ids=selected_ids,
            )
            self.aggregator.aggregate(self.ctx)
            self._save_round(round_index + 1)
            if round_index % self.eval_interval == 0 or round_index == self.num_glob_iters - 1:
                self._evaluate(round_index)

        if self.train_mode == "centralized":
            final_state = self.ctx.new_model_state[0]
        else:
            final_state = self.ctx.new_model_state[self.target_client_id]
        self.model.load_state_dict(final_state, strict=False)
        torch.save(
            self._clone_state(final_state, cpu=True),
            os.path.join(self.results_dir, "final_prompt.pt"),
        )
        return self.auditor.finalize(self.model, final_state)
