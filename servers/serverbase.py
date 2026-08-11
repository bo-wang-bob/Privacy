from __future__ import annotations

import csv
import logging
import os
import random
import json
import math
import statistics

import torch

from aggregator.base_aggregator import BaseAggregator
from context.context import Context
from privacy_attacks.auditor import MembershipAuditor
from privacy_attacks.projres_integrated import run_integrated_projres
from privacy_defenses import (
    FEDMIA_BASELINE_DEFENSES,
    DefenseController,
    attach_hamp_output_transform,
)
from utils.privacy_accounting import planned_private_probe_steps
from users.user import UserBase

logger = logging.getLogger(__name__)


def _is_evaluation_round(
    round_index: int, total_rounds: int, eval_interval: int
) -> bool:
    """Schedule evaluation by completed, one-based communication rounds."""
    completed_round = round_index + 1
    return completed_round % eval_interval == 0 or completed_round == total_rounds


def _scheduled_learning_rate(
    initial_learning_rate: float,
    decay: float,
    round_index: int,
    decay_interval: int = 1,
) -> float:
    """Return the LR for a zero-based communication round.

    Round zero (the first training round) uses the configured initial learning
    rate.  The rate is reduced only after each complete ``decay_interval``
    block; a decay of one disables scheduling.
    """
    if initial_learning_rate <= 0:
        raise ValueError("initial_learning_rate must be positive.")
    if not 0 < decay <= 1:
        raise ValueError("learning_rate_decay must be in (0, 1].")
    if round_index < 0:
        raise ValueError("round_index must be non-negative.")
    if decay_interval <= 0:
        raise ValueError("learning_rate_decay_interval must be positive.")
    decay_steps = round_index // decay_interval
    return initial_learning_rate * decay**decay_steps


def _format_round_progress(
    round_index: int,
    total_rounds: int,
    loss: float,
    accuracy: float,
    selected_ids: list[int],
    total_users: int,
    audit_snapshots: int,
    learning_rate: float | None = None,
) -> str:
    if sorted(selected_ids) == list(range(total_users)):
        selected = f"all({total_users})"
    else:
        selected = "[" + ",".join(str(user_id) for user_id in selected_ids) + "]"
    learning_rate_text = (
        "" if learning_rate is None else f" | lr={learning_rate:.6g}"
    )
    return (
        f"Progress | round={round_index + 1}/{total_rounds} | loss={loss:.4f} | "
        f"accuracy={100.0 * accuracy:.2f}%{learning_rate_text} | selected={selected} | "
        f"audit_snapshots={audit_snapshots}"
    )


class ServerBase:
    """Federated prompt-tuning server with pluggable methods and privacy audits."""

    def __init__(
        self,
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
        learning_rate_decay: float = 1.0,
        learning_rate_decay_interval: int = 1,
        model_load_path=None,
        save_models: bool = False,
        collate_fn=None,
        eval_interval: int = 5,
        eval_batch_size: int = 64,
        audit_config: dict | None = None,
        projres_config: dict | None = None,
        defense_config: dict | None = None,
        method_config: dict | None = None,
    ):
        if total_users <= 1:
            raise ValueError("total_users must be greater than one.")
        if not 1 <= user_per_round <= total_users:
            raise ValueError("user_per_round must be in [1, total_users].")
        if num_glob_iters <= 0 or eval_interval <= 0:
            raise ValueError("Training rounds and eval_interval must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 < learning_rate_decay <= 1:
            raise ValueError("learning_rate_decay must be in (0, 1].")
        if learning_rate_decay_interval <= 0:
            raise ValueError("learning_rate_decay_interval must be positive.")
        self.device = device
        self.dataset_name = dataset_name
        self.num_classes = len(class_names)
        self.model = model
        self.total_users_num = total_users
        self.num_glob_iters = num_glob_iters
        self.user_per_round = user_per_round
        self.aggregator = aggregator
        if str(getattr(model, "model_type", "")).lower() in {
            "clip_mlp",
            "visual_adapter",
            "clip_lora",
        } and hasattr(self.aggregator, "aggregation_weighting"):
            self.aggregator.aggregation_weighting = "uniform"
        self.federated_method = aggregator.name
        self.method_config = dict(method_config or {})
        self.results_dir = results_dir
        self.save_models_enabled = save_models
        self.eval_interval = eval_interval
        self.initial_learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.learning_rate_decay_interval = int(learning_rate_decay_interval)
        self.current_learning_rate = float(learning_rate)
        self.audit_config = dict(audit_config or {"enabled": True})
        self.audit_config.setdefault("total_rounds", num_glob_iters)
        self.projres_config = dict(projres_config or {"enabled": False})
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.local_epochs = int(local_epochs)
        if self.federated_method == "fedsgd" and self.local_epochs != 1:
            raise ValueError(
                "FedSGD uses exactly one mini-batch per client and requires "
                "local_epochs=1."
            )
        configured_projres_round = self.projres_config.get(
            "evaluation_round", "last"
        )
        if str(configured_projres_round).lower() == "last":
            self.projres_evaluation_round = self.num_glob_iters
        else:
            self.projres_evaluation_round = int(configured_projres_round)
        if not 1 <= self.projres_evaluation_round <= self.num_glob_iters:
            raise ValueError(
                "projres.evaluation_round must be 'last' or a communication "
                "round in [1, num_global_iters]."
            )
        self.defense_config = defense_config or {"name": "none"}
        self.target_client_id = int(self.audit_config.get("target_client_id", 0))
        self.ensure_target = bool(self.audit_config.get("enabled", True)) and bool(
            self.audit_config.get("ensure_target_participation", True)
        )
        os.makedirs(results_dir, exist_ok=True)
        self.metrics_path = os.path.join(results_dir, "training_metrics.csv")
        with open(self.metrics_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                ("round", "loss", "accuracy", "samples", "learning_rate")
            )
        self.training_metrics: list[dict[str, float | int]] = []

        self.ctx = Context(
            users_num=total_users,
            model=model,
            class_names=class_names,
            learning_rate=learning_rate,
            results_dir=results_dir,
        )
        code_poison_config = {
            "weight": float(self.audit_config.get("codepoison_weight", 1.0)),
            "synthetic_mean": float(self.audit_config.get("synthetic_mean", 0.0)),
            "synthetic_std": float(self.audit_config.get("synthetic_std", 0.1)),
        }
        defense_seeded_config = dict(self.defense_config)
        defense_seeded_config.setdefault("seed", int(self.audit_config.get("seed", 42)))
        self.defense = DefenseController(
            config=defense_seeded_config,
            device=device,
            total_users=total_users,
            num_classes=len(class_names),
            total_rounds=num_glob_iters,
        )
        self.defense.federated_method = self.federated_method
        self.defense.method_config = self.method_config
        if (
            self.defense.name in FEDMIA_BASELINE_DEFENSES
            and self.federated_method not in {"fedavg", "promptfl"}
        ):
            raise ValueError(
                "FedMIA baseline defenses require a shared global FedAvg or "
                "PromptFL model and are standalone comparisons."
            )
        if self.defense.name == "hamp":
            attach_hamp_output_transform(
                model,
                float(self.defense_config.get("hamp_output_temperature", 4.0)),
            )
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
                defense_controller=self.defense,
                federated_method=self.federated_method,
                method_config=self.method_config,
            )
            self.ctx.users.append(user)
            if self.defense.name == "hamp":
                attach_hamp_output_transform(
                    user.model,
                    float(self.defense_config.get("hamp_output_temperature", 4.0)),
                )
            self.ctx.samples_num.append(user.train_samples)
        self.defense.samples_num = list(self.ctx.samples_num)

        if model_load_path:
            state = torch.load(model_load_path, map_location=device)
            self.model.load_state_dict(state, strict=False)
            for user in self.ctx.users:
                user.set_parameters(state)

        self.defense.initialize_iclr_feature_statistics(self.ctx.users)

        self.auditor = MembershipAuditor(
            model=model,
            users=self.ctx.users,
            target_client_id=self.target_client_id,
            device=device,
            results_dir=results_dir,
            collate_fn=collate_fn,
            config=self.audit_config,
            defense_config=self.defense_config,
            federated_method=self.federated_method,
            num_classes=self.num_classes,
        )
        self.required_audit_client_ids = (
            list(self.auditor.audit_client_ids) if self.ensure_target else []
        )
        if len(self.required_audit_client_ids) > self.user_per_round:
            raise ValueError(
                "ensure_target_participation requires sample_users to be at "
                "least the number of audited clients."
            )
        self.code_poison_enabled = (
            self.auditor.enabled and "codepoison" in self.auditor.attacks
        )
        self.private_probe_steps = planned_private_probe_steps(self.audit_config)
        target_user = self.ctx.users[self.target_client_id]
        local_steps_per_probe = (
            1
            if self.federated_method == "fedsgd"
            else target_user.local_epochs * len(target_user.trainloader)
        )
        self.defense.additional_private_steps = (
            self.private_probe_steps * local_steps_per_probe
        )

    @staticmethod
    def _clone_state(state: dict[str, torch.Tensor], cpu: bool = False):
        return {
            name: tensor.detach().cpu().clone() if cpu else tensor.detach().clone()
            for name, tensor in state.items()
        }

    def _sample_users(self) -> list[int]:
        if not self.ensure_target or self.user_per_round == self.total_users_num:
            return sorted(
                random.sample(range(self.total_users_num), self.user_per_round)
            )
        required = set(self.required_audit_client_ids)
        others = [
            user_id
            for user_id in range(self.total_users_num)
            if user_id not in required
        ]
        selected = random.sample(
            others, self.user_per_round - len(self.required_audit_client_ids)
        )
        return sorted([*self.required_audit_client_ids, *selected])

    def _evaluate(
        self,
        round_index: int,
        selected_ids: list[int],
        learning_rate: float,
    ) -> None:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for user in self.ctx.users:
            local_state = (
                self._clone_state(user.get_parameters())
                if self.defense.name == "iclr"
                or bool(getattr(user.model, "client_scoped_lora", False))
                else None
            )
            evaluation_state = self.ctx.new_model_state.get(
                0, self.ctx.base_model_state[0]
            )
            try:
                user.set_parameters(evaluation_state)
                loss, correct, samples = user.evaluate()
            finally:
                if local_state is not None:
                    user.set_parameters(local_state)
            total_loss += loss
            total_correct += correct
            total_samples += samples
        metrics = {
            "round": int(round_index + 1),
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "samples": int(total_samples),
            "learning_rate": float(learning_rate),
        }
        self.training_metrics.append(metrics)
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                (
                    metrics["round"],
                    metrics["loss"],
                    metrics["accuracy"],
                    metrics["samples"],
                    metrics["learning_rate"],
                )
            )
        logger.info(
            "%s",
            _format_round_progress(
                round_index=round_index,
                total_rounds=self.num_glob_iters,
                loss=float(metrics["loss"]),
                accuracy=float(metrics["accuracy"]),
                selected_ids=selected_ids,
                total_users=self.total_users_num,
                audit_snapshots=len(self.auditor.observations),
                learning_rate=float(metrics["learning_rate"]),
            ),
        )

    def _validate_training_health(
        self,
        initial_state: dict[str, torch.Tensor],
        final_state: dict[str, torch.Tensor],
    ) -> dict:
        """Persist and enforce inexpensive training-degeneracy checks."""
        enabled = bool(self.audit_config.get("training_health_check", True))
        minimum_update = float(
            self.audit_config.get("min_trainable_update_norm", 1e-12)
        )
        stagnant_tolerance = float(
            self.audit_config.get("max_stagnant_loss_range", 1e-8)
        )
        uniform_tolerance = float(
            self.audit_config.get("uniform_loss_tolerance", 1e-4)
        )
        if minimum_update < 0 or stagnant_tolerance < 0 or uniform_tolerance < 0:
            raise ValueError("Training health tolerances must be non-negative.")

        squared_delta = 0.0
        finite_state = True
        compared = 0
        for name, initial in initial_state.items():
            if name not in final_state:
                continue
            final = final_state[name]
            finite_state = finite_state and bool(torch.isfinite(final).all())
            difference = (
                final.detach().to(torch.float64).cpu()
                - initial.to(torch.float64).cpu()
            )
            squared_delta += float(
                difference.square().sum()
            )
            compared += final.numel()
        update_norm = math.sqrt(squared_delta)
        losses = [float(item["loss"]) for item in self.training_metrics]
        finite_metrics = bool(losses) and all(
            math.isfinite(value) for value in losses
        )
        loss_range = max(losses) - min(losses) if losses else None
        uniform_loss = math.log(max(self.num_classes, 1))
        uniform_stagnation = bool(
            len(losses) >= 2
            and loss_range is not None
            and loss_range <= stagnant_tolerance
            and abs(statistics.fmean(losses) - uniform_loss) <= uniform_tolerance
        )
        reasons = []
        if not finite_state or not finite_metrics:
            reasons.append("non_finite_state_or_metrics")
        if compared == 0 or update_norm <= minimum_update:
            reasons.append("trainable_parameters_did_not_move")
        if uniform_stagnation:
            reasons.append("loss_stagnated_at_uniform_classifier_baseline")
        health = {
            "enabled": enabled,
            "passed": not reasons,
            "reasons": reasons,
            "trainable_update_norm": update_norm,
            "minimum_trainable_update_norm": minimum_update,
            "finite_state": finite_state,
            "finite_metrics": finite_metrics,
            "evaluation_count": len(losses),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "loss_range": loss_range,
            "uniform_classifier_loss": uniform_loss,
        }
        with open(
            os.path.join(self.results_dir, "training_health.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(health, file, indent=2, allow_nan=False)
        if enabled and reasons:
            raise RuntimeError(
                "Training health check failed: " + ", ".join(reasons)
            )
        return health

    def _save_round(self, round_index: int) -> None:
        if not self.save_models_enabled:
            return
        path = os.path.join(self.results_dir, "saved_models")
        os.makedirs(path, exist_ok=True)
        state = self.ctx.new_model_state.get(0, self.ctx.base_model_state[0])
        torch.save(
            self._clone_state(state, cpu=True),
            os.path.join(path, f"global_round_{round_index}.pt"),
        )

    def train(self) -> list[dict]:
        self.ctx.set_base_model_state(
            self._clone_state(self.ctx.users[0].get_parameters())
        )
        initial_target_state = self._clone_state(
            self.ctx.get_base_model_state(), cpu=True
        )

        # Record the true pretrained/random-head baseline before any client
        # performs a local update.  Auditing still starts only after training
        # rounds, so this does not add a round-zero attack observation.
        self._evaluate(
            round_index=-1,
            selected_ids=list(range(self.total_users_num)),
            learning_rate=self.initial_learning_rate,
        )

        previous_selected_ids: set[int] = set()
        previous_aggregation_weights: dict[int, float] = {}
        for round_index in range(self.num_glob_iters):
            if round_index > 0:
                self.ctx.continue_to_next_round()
            self.ctx.glob_iter = round_index
            self.current_learning_rate = _scheduled_learning_rate(
                self.initial_learning_rate,
                self.learning_rate_decay,
                round_index,
                self.learning_rate_decay_interval,
            )
            self.ctx.learning_rate = self.current_learning_rate
            for user in self.ctx.users:
                user.learning_rate = self.current_learning_rate
            selected_ids = self._sample_users()
            self.ctx.user_selected = selected_ids
            self.defense.prepare_round(selected_ids, round_index)
            logger.debug("Round %s selected clients: %s", round_index, selected_ids)

            base_states = {
                user_id: self._clone_state(self.ctx.get_base_model_state())
                for user_id in selected_ids
            }
            base_state = base_states.get(
                self.target_client_id,
                self._clone_state(self.ctx.get_base_model_state()),
            )
            for user_id in selected_ids:
                user = self.ctx.users[user_id]
                if user_id in previous_selected_ids:
                    self.defense.prepare_client_training(
                        user=user,
                        global_state=self.ctx.get_base_model_state(),
                        own_weight=previous_aggregation_weights[user_id],
                        source_round=round_index - 1,
                    )
                user.set_parameters(self.ctx.get_base_model_state())
                use_code_poison = (
                    self.code_poison_enabled and user_id == self.target_client_id
                )
                user.train(
                    code_poison=use_code_poison,
                    round_index=round_index,
                )
                if self.federated_method == "fedsgd":
                    if user.last_update_sample_count <= 0:
                        raise RuntimeError(
                            f"FedSGD client {user_id} did not consume a mini-batch."
                        )
                    self.ctx.update_sample_counts[user_id] = (
                        user.last_update_sample_count
                    )
                self.ctx.set_updated_model_state(
                    user_id, self._clone_state(user.get_parameters())
                )

            self.defense.after_local_training(
                users=self.ctx.users,
                base_state=base_state,
                updated_states=self.ctx.updated_model_state,
                selected_ids=selected_ids,
                round_index=round_index,
            )

            if str(getattr(self.model, "model_type", "")).lower() == "clip_lora":
                for user_id in selected_ids:
                    self.ctx.users[user_id].set_parameters(
                        self.ctx.updated_model_state[user_id]
                    )

            if (
                bool(self.projres_config.get("enabled", False))
                and round_index + 1 == self.projres_evaluation_round
            ):
                projres_client_ids = list(self.auditor.audit_client_ids)
                missing_clients = sorted(
                    set(projres_client_ids) - set(selected_ids)
                )
                if missing_clients:
                    raise ValueError(
                        "ProjRes evaluation round did not include audited clients: "
                        f"{missing_clients}. Enable ensure_target_participation or "
                        "increase sample_users."
                    )
                run_integrated_projres(
                    model=self.model,
                    users=self.ctx.users,
                    device=self.device,
                    base_states=base_states,
                    updated_states=self.ctx.updated_model_state,
                    learning_rate=self.current_learning_rate,
                    batch_size=self.batch_size,
                    eval_batch_size=self.eval_batch_size,
                    local_epochs=self.local_epochs,
                    federated_method=self.federated_method,
                    round_index=round_index,
                    seed=int(self.audit_config.get("seed", 42)),
                    dataset_name=self.dataset_name,
                    client_ids=projres_client_ids,
                    config=self.projres_config,
                    output_path=os.path.join(
                        self.results_dir,
                        "privacy_audit",
                        "projres_strict.json",
                    ),
                )

            self.aggregator.aggregate(self.ctx)
            previous_selected_ids = set(selected_ids)
            previous_aggregation_weights = dict(self.ctx.aggregation_weights)
            self.auditor.observe_round(
                round_index=round_index,
                base_state=base_state,
                updated_states=self.ctx.updated_model_state,
                selected_ids=selected_ids,
                base_states=base_states,
                protocol_messages=self.ctx.protocol_messages,
                released_states=self.ctx.new_model_state,
            )
            self._save_round(round_index + 1)
            if _is_evaluation_round(
                round_index, self.num_glob_iters, self.eval_interval
            ):
                self._evaluate(
                    round_index,
                    selected_ids,
                    learning_rate=self.current_learning_rate,
                )

        final_state = self.ctx.new_model_state[0]
        training_health = self._validate_training_health(
            initial_target_state, final_state
        )
        self.model.load_state_dict(final_state, strict=False)
        torch.save(
            self._clone_state(final_state, cpu=True),
            os.path.join(
                self.results_dir,
                str(
                    getattr(
                        self.model,
                        "trainable_state_filename",
                        "final_prompt.pt",
                    )
                ),
            ),
        )
        defense_summary = self.defense.save_summary(self.results_dir)
        logger.info("Privacy defense completed: %s", defense_summary)
        method_summary = {
            "federated_method": self.federated_method,
            "model_type": str(getattr(self.model, "model_type", "prompt")),
            "configuration": self.method_config,
            "training_health": training_health,
            "state_scope": "shared_global",
            "optimization": {
                "initial_learning_rate": self.initial_learning_rate,
                "learning_rate_decay": self.learning_rate_decay,
                "learning_rate_decay_interval": self.learning_rate_decay_interval,
                "final_round_learning_rate": self.current_learning_rate,
            },
        }
        if self.federated_method == "promptfl":
            method_summary["paper_alignment"] = (
                "CoOp-style shared soft text prompt with sample-weighted FedAvg"
            )
        if str(getattr(self.model, "model_type", "")).lower() == "clip_lora":
            method_summary["lora"] = {
                "trainable_scope": "attention_lora_A_and_lora_B_only",
                "client_storage": "independent_lora_factors_only",
                "shared_clip_backbone": True,
                "aggregation": str(
                    f"factor_wise_{self.federated_method}"
                ),
                "aggregation_weighting": str(
                    getattr(self.aggregator, "aggregation_weighting", "uniform")
                ),
                "frozen_backbone": True,
            }
        with open(
            os.path.join(self.results_dir, "federated_method_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(method_summary, file, indent=2, allow_nan=False)
        return self.auditor.finalize(self.model, final_state)
