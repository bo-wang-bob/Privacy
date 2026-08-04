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
from privacy_defenses import (
    FEDMIA_BASELINE_DEFENSES,
    DefenseController,
    attach_hamp_output_transform,
)
from utils.privacy_accounting import (
    gaussian_rdp_epsilon,
    planned_private_probe_steps,
)
from users.user import UserBase

logger = logging.getLogger(__name__)


def _format_round_progress(
    round_index: int,
    total_rounds: int,
    loss: float,
    accuracy: float,
    selected_ids: list[int],
    total_users: int,
    audit_snapshots: int,
) -> str:
    if sorted(selected_ids) == list(range(total_users)):
        selected = f"all({total_users})"
    else:
        selected = "[" + ",".join(str(user_id) for user_id in selected_ids) + "]"
    return (
        f"Progress | round={round_index + 1}/{total_rounds} | loss={loss:.4f} | "
        f"accuracy={100.0 * accuracy:.2f}% | selected={selected} | "
        f"audit_snapshots={audit_snapshots}"
    )


class ServerBase:
    """Federated prompt-tuning server with pluggable methods and privacy audits."""

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
        defense_config: dict | None = None,
        method_config: dict | None = None,
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
        self.num_classes = len(class_names)
        self.model = model
        self.total_users_num = total_users
        self.num_glob_iters = num_glob_iters
        self.user_per_round = user_per_round
        self.aggregator = aggregator
        self.federated_method = aggregator.name
        self.method_config = dict(method_config or {})
        self.results_dir = results_dir
        self.save_models_enabled = save_models
        self.eval_interval = eval_interval
        self.audit_config = audit_config or {"enabled": True}
        self.defense_config = defense_config or {"name": "none"}
        self.target_client_id = int(self.audit_config.get("target_client_id", 0))
        self.ensure_target = bool(self.audit_config.get("enabled", True)) and bool(
            self.audit_config.get("ensure_target_participation", True)
        )
        os.makedirs(results_dir, exist_ok=True)
        self.metrics_path = os.path.join(results_dir, "training_metrics.csv")
        with open(self.metrics_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(("round", "loss", "accuracy", "samples"))
        self.training_metrics: list[dict[str, float | int]] = []

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
            and (
                self.federated_method not in {"fedavg", "promptfl"}
                or self.train_mode != "centralized"
            )
        ):
            raise ValueError(
                "FedMIA baseline defenses require centralized FedAvg and are "
                "standalone comparisons; stacking them with personalized or "
                "private prompt methods would change both mechanisms."
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
            defense_config=self.defense_config,
            federated_method=self.federated_method,
            num_classes=self.num_classes,
        )
        self.code_poison_enabled = (
            self.auditor.enabled and "codepoison" in self.auditor.attacks
        )
        self.private_probe_steps = planned_private_probe_steps(self.audit_config)
        if self.federated_method in {"dpfpl", "fedask"}:
            self.defense.additional_private_steps = self.private_probe_steps
        else:
            target_user = self.ctx.users[self.target_client_id]
            self.defense.additional_private_steps = self.private_probe_steps * (
                target_user.local_epochs * len(target_user.trainloader)
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
        others = [
            user_id
            for user_id in range(self.total_users_num)
            if user_id != self.target_client_id
        ]
        selected = random.sample(others, self.user_per_round - 1)
        return sorted([self.target_client_id, *selected])

    def _evaluate(self, round_index: int, selected_ids: list[int]) -> None:
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
        metrics = {
            "round": int(round_index),
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "samples": int(total_samples),
        }
        self.training_metrics.append(metrics)
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                (
                    metrics["round"],
                    metrics["loss"],
                    metrics["accuracy"],
                    metrics["samples"],
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
        if self.train_mode == "centralized":
            state = self.ctx.new_model_state.get(0, self.ctx.base_model_state[0])
            torch.save(
                self._clone_state(state, cpu=True),
                os.path.join(path, f"global_round_{round_index}.pt"),
            )
        else:
            for user_id in range(self.total_users_num):
                state = self.ctx.new_model_state.get(
                    user_id, self.ctx.base_model_state[user_id]
                )
                torch.save(
                    self._clone_state(state, cpu=True),
                    os.path.join(path, f"client_{user_id}_round_{round_index}.pt"),
                )

    def train(self) -> list[dict]:
        for user in self.ctx.users:
            self.ctx.set_base_model_state(
                user.id, self._clone_state(user.get_parameters())
            )
        initial_target_state = self._clone_state(
            self.ctx.get_base_model_state(self.target_client_id), cpu=True
        )

        for round_index in range(self.num_glob_iters):
            if round_index > 0:
                self.ctx.continue_to_next_round()
            self.ctx.glob_iter = round_index
            selected_ids = self._sample_users()
            self.ctx.user_selected = selected_ids
            self.defense.prepare_round(selected_ids, round_index)
            logger.debug("Round %s selected clients: %s", round_index, selected_ids)

            base_states = {
                user_id: self._clone_state(self.ctx.get_base_model_state(user_id))
                for user_id in selected_ids
            }
            base_state = base_states.get(
                self.target_client_id,
                self._clone_state(self.ctx.get_base_model_state(self.target_client_id)),
            )
            for user_id in selected_ids:
                user = self.ctx.users[user_id]
                user.set_parameters(self.ctx.get_base_model_state(user_id))
                use_code_poison = (
                    self.code_poison_enabled and user_id == self.target_client_id
                )
                user.train(
                    code_poison=use_code_poison,
                    round_index=round_index,
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

            self.aggregator.aggregate(self.ctx)
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
            if (
                round_index % self.eval_interval == 0
                or round_index == self.num_glob_iters - 1
            ):
                self._evaluate(round_index, selected_ids)

        if self.train_mode == "centralized":
            final_state = self.ctx.new_model_state[0]
        else:
            final_state = self.ctx.new_model_state[self.target_client_id]
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
            "state_scope": (
                "global_plus_persistent_per_client_local"
                if self.federated_method in {"dpfpl", "fedotp", "fedpgp"}
                else "shared_global"
            ),
        }
        if self.federated_method == "promptfl":
            method_summary["paper_alignment"] = (
                "CoOp-style shared soft text prompt with sample-weighted FedAvg"
            )
        elif self.federated_method == "fedotp":
            method_summary["paper_alignment"] = (
                "global/local full-rank prompts with fixed-plan entropic "
                "unbalanced optimal transport; only global_ctx is communicated"
            )
        elif self.federated_method == "fedpgp":
            method_summary["paper_alignment"] = (
                "aggregated global prompt plus persistent low-rank local "
                "adaptation and CLIP-guided prompt-wise contrastive loss"
            )
        if self.federated_method == "dpfpl":
            method_summary["privacy_mechanisms"] = {
                "local": "RGP low-rank gradient clipping and Gaussian perturbation",
                "global": "client update clipping and server Gaussian perturbation",
                "delta": float(self.method_config.get("delta", 1e-5)),
            }
            local_steps = self.num_glob_iters * int(
                self.method_config.get("local_steps", 1)
            )
            if self.defense.name == "mist":
                local_steps += self.num_glob_iters * int(
                    self.defense_config.get("mist_cross_steps", 1)
                )
            local_steps += self.private_probe_steps
            delta = float(self.method_config.get("delta", 1e-5))
            local_epsilon = gaussian_rdp_epsilon(
                float(self.method_config.get("local_noise_multiplier", 1.0)),
                local_steps,
                delta,
                mechanisms_per_step=2,
            )
            global_epsilon = gaussian_rdp_epsilon(
                float(self.method_config.get("global_noise_multiplier", 1.0)),
                self.num_glob_iters,
                delta,
            )
            method_summary["privacy_accounting"] = {
                "local_epsilon_upper_bound": (
                    local_epsilon if math.isfinite(local_epsilon) else None
                ),
                "global_epsilon_upper_bound": (
                    global_epsilon if math.isfinite(global_epsilon) else None
                ),
                "delta": delta,
                "formal_dp_enabled": bool(
                    float(self.method_config.get("local_noise_multiplier", 1.0)) > 0
                    and float(self.method_config.get("global_noise_multiplier", 1.0))
                    > 0
                    and not self.method_config.get("reproducible_dp_noise", False)
                    and not self.defense_config.get("reproducible_dp_noise", False)
                    and self.defense.name not in {"cofedmid", "soft"}
                ),
                "accountant": "conservative Gaussian RDP; no subsampling amplification",
                "active_audit_probe_steps": self.private_probe_steps,
                "formal_dp_caveat": (
                    "SOFT/CoFedMID use data-dependent selection not covered by this accountant."
                    if self.defense.name in {"cofedmid", "soft"}
                    else None
                ),
            }
        elif self.federated_method == "fedask":
            method_summary["privacy_mechanisms"] = {
                "local": "full-prompt per-sample clipping/noise followed by asymmetric B update",
                "aggregation": "two-stage randomized sketch and SVD reconstruction",
                "delta": float(self.method_config.get("delta", 1e-5)),
            }
            local_steps = self.num_glob_iters * int(
                self.method_config.get("local_steps", self.ctx.users[0].local_epochs)
            )
            if self.defense.name == "mist":
                local_steps += self.num_glob_iters * int(
                    self.defense_config.get("mist_cross_steps", 1)
                )
            local_steps += self.private_probe_steps
            delta = float(self.method_config.get("delta", 1e-5))
            epsilon = gaussian_rdp_epsilon(
                float(self.method_config.get("noise_multiplier", 1.0)),
                local_steps,
                delta,
            )
            method_summary["privacy_accounting"] = {
                "epsilon_upper_bound": epsilon if math.isfinite(epsilon) else None,
                "delta": delta,
                "formal_dp_enabled": bool(
                    float(self.method_config.get("noise_multiplier", 1.0)) > 0
                    and not self.method_config.get("reproducible_dp_noise", False)
                    and not self.defense_config.get("reproducible_dp_noise", False)
                    and self.defense.name not in {"cofedmid", "soft"}
                ),
                "accountant": "conservative Gaussian RDP; no client/data subsampling amplification",
                "active_audit_probe_steps": self.private_probe_steps,
                "formal_dp_caveat": (
                    "SOFT/CoFedMID use data-dependent selection not covered by this accountant."
                    if self.defense.name in {"cofedmid", "soft"}
                    else None
                ),
            }
        if hasattr(self.aggregator, "last_reconstruction_error"):
            method_summary["last_reconstruction_error"] = float(
                self.aggregator.last_reconstruction_error
            )
            method_summary["last_pretruncation_error"] = float(
                self.aggregator.last_pretruncation_error
            )
            method_summary["last_subspace_rank"] = int(
                self.aggregator.last_subspace_rank
            )
            method_summary["rank_based_minimum_oversampling"] = int(
                self.aggregator.last_required_oversampling
            )
            method_summary["rank_based_width_condition_met"] = bool(
                int(self.method_config.get("oversampling", 2))
                >= self.aggregator.last_required_oversampling
            )
        if hasattr(self.aggregator, "last_clip_fraction"):
            method_summary["last_global_clip_fraction"] = float(
                self.aggregator.last_clip_fraction
            )
        with open(
            os.path.join(self.results_dir, "federated_method_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(method_summary, file, indent=2, allow_nan=False)
        return self.auditor.finalize(self.model, final_state)
