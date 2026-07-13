import copy
import csv
import json
import logging
import os

import torch
from torch.utils.data import DataLoader

from privacy_attacks.code_poison import run_code_poison_attack
from privacy_attacks.features import (
    flatten_state_delta,
    logits_and_representation,
    per_sample_prompt_gradients,
    trainable_names,
)
from privacy_attacks.fedmia import run_fedmia
from privacy_attacks.imia import run_imia
from privacy_attacks.model_utils import last_client_states
from privacy_attacks.pipra import run_pipra
from privacy_attacks.promptmia import run_promptmia
from privacy_attacks.quantile import run_quantile_mia
from privacy_attacks.query_attacks import run_canary, run_yoqo
from privacy_attacks.rmia import run_rmia
from privacy_attacks.transfer import run_transfer_representation_attack
from privacy_attacks.whitebox import run_active_whitebox, run_passive_whitebox
from privacy_defenses import attach_hamp_output_transform

logger = logging.getLogger(__name__)

SUPPORTED_ATTACKS = {
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
}


class MembershipAuditor:
    """Collect once, then run all configured prompt-tuning membership attacks."""

    def __init__(
        self,
        model: torch.nn.Module,
        users: list,
        target_client_id: int,
        device: torch.device,
        results_dir: str,
        collate_fn=None,
        config: dict | None = None,
        defense_config: dict | None = None,
    ):
        self.config = config or {}
        self.defense_config = dict(defense_config or {"name": "none"})
        self.defense_name = str(self.defense_config.get("name", "none")).lower()
        self.enabled = bool(self.config.get("enabled", True))
        self.attacks = list(
            self.config.get(
                "attacks",
                [
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
            )
        )
        unknown = sorted(set(self.attacks) - SUPPORTED_ATTACKS)
        if unknown:
            raise ValueError(f"Unsupported membership attacks: {', '.join(unknown)}")
        if not 0 <= target_client_id < len(users):
            raise ValueError("target_client_id is outside the client range.")
        self.model = copy.deepcopy(model).to(device)
        if self.defense_name == "hamp":
            attach_hamp_output_transform(
                self.model,
                float(self.defense_config.get("hamp_output_temperature", 4.0)),
            )
        self.users = users
        self.target_client_id = target_client_id
        self.device = device
        self.results_dir = os.path.join(results_dir, "privacy_audit")
        self.collate_fn = collate_fn
        self.seed = int(self.config.get("seed", 42))
        self.audit_interval = int(self.config.get("audit_interval", 1))
        self.calibration_fraction = float(
            self.config.get("calibration_fraction", 0.5)
        )
        if self.audit_interval <= 0:
            raise ValueError("audit_interval must be positive.")
        self.observations: list[dict] = []
        self.results = []
        self.errors: dict[str, str] = {}

        if self.enabled:
            max_samples = int(self.config.get("max_samples_per_group", 32))
            if max_samples < 2:
                raise ValueError("max_samples_per_group must be at least 2.")
            target = users[target_client_id]
            member_images, member_labels = self._collect_many(
                [target.train_data], max_samples
            )
            nonmember_datasets = [target.test_data] + [
                user.test_data for user in users if user.id != target_client_id
            ]
            nonmember_images, nonmember_labels = self._collect_many(
                nonmember_datasets, max_samples
            )
            self.images = torch.cat((member_images, nonmember_images)).to(device)
            self.labels = torch.cat((member_labels, nonmember_labels)).to(device)
            self.membership = torch.cat(
                (
                    torch.ones(member_labels.numel(), dtype=torch.long),
                    torch.zeros(nonmember_labels.numel(), dtype=torch.long),
                )
            )

    def _collect_many(
        self, datasets: list, limit: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_parts = []
        label_parts = []
        remaining = limit
        for dataset in datasets:
            if remaining == 0:
                break
            loader = DataLoader(
                dataset,
                batch_size=min(remaining, 64),
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            for images, labels in loader:
                take = min(remaining, labels.numel())
                image_parts.append(images[:take].detach().cpu())
                label_parts.append(labels[:take].detach().cpu().long())
                remaining -= take
                if remaining == 0:
                    break
        if remaining == limit:
            raise ValueError("Cannot audit an empty candidate dataset.")
        if limit - remaining < 2:
            raise ValueError("Each membership group needs at least two candidates.")
        return torch.cat(image_parts), torch.cat(label_parts)

    def should_observe(self, round_index: int) -> bool:
        return self.enabled and round_index % self.audit_interval == 0

    def observe_round(
        self,
        round_index: int,
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        base_states: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> None:
        if not self.should_observe(round_index):
            return
        names = trainable_names(self.model)
        self.model.load_state_dict(base_state, strict=False)
        self.model.eval()
        sample_gradients, signatures, _ = per_sample_prompt_gradients(
            self.model, self.images, self.labels
        )

        confidence = []
        cosine = []
        gradient_difference = []
        probabilities = []
        representations = []
        for user_id in selected_ids:
            state = updated_states[user_id]
            client_base = base_state if base_states is None else base_states[user_id]
            update = flatten_state_delta(client_base, state, names)
            update_norm = update.norm().clamp_min(1e-12)
            sample_norm = sample_gradients.norm(dim=1).clamp_min(1e-12)
            cosine.append((sample_gradients @ update) / (sample_norm * update_norm))
            update_sq = update.square().sum()
            gradient_difference.append(
                update_sq - (update.unsqueeze(0) - sample_gradients).square().sum(dim=1)
            )

            self.model.load_state_dict(state, strict=False)
            logits, representation, losses = logits_and_representation(
                self.model, self.images, self.labels
            )
            confidence.append(-losses)
            probabilities.append(torch.softmax(logits, dim=1))
            representations.append(representation)

        observation = {
            "round": int(round_index),
            "client_ids": torch.tensor(selected_ids, dtype=torch.long),
            "confidence": torch.stack(confidence),
            "cosine": torch.stack(cosine),
            "gradient_difference": torch.stack(gradient_difference),
            "gradient_signature": signatures,
            "probabilities": torch.stack(probabilities),
            "representations": torch.stack(representations),
            "client_states": {
                user_id: {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in updated_states[user_id].items()
                }
                for user_id in selected_ids
            },
        }
        self.observations.append(observation)
        logger.info("Collected privacy signals for round %s", round_index)

    def _run(self, attack: str, final_model, final_state):
        if attack == "fedmia_loss":
            return run_fedmia(
                self.observations, self.membership, self.target_client_id, "confidence"
            )
        if attack == "fedmia_cosine":
            return run_fedmia(
                self.observations, self.membership, self.target_client_id, "cosine"
            )
        if attack == "nasr_passive":
            return run_passive_whitebox(
                self.observations,
                self.membership,
                self.target_client_id,
                self.calibration_fraction,
                self.seed,
            )
        if attack == "nasr_active":
            return run_active_whitebox(
                base_model=final_model,
                final_state=final_state,
                target_user=self.users[self.target_client_id],
                images=self.images,
                labels=self.labels,
                membership=self.membership,
                max_samples=int(self.config.get("active_max_samples", 16)),
                ascent_steps=int(self.config.get("active_ascent_steps", 1)),
                ascent_lr=float(self.config.get("active_ascent_lr", 0.01)),
            )
        if attack == "transfer_representation":
            return run_transfer_representation_attack(
                self.observations,
                self.membership,
                self.target_client_id,
                self.calibration_fraction,
                self.seed,
            )
        if attack == "codepoison":
            return run_code_poison_attack(
                final_model,
                self.images,
                self.labels,
                self.membership,
                mean=float(self.config.get("synthetic_mean", 0.0)),
                std=float(self.config.get("synthetic_std", 0.1)),
            )
        if attack == "rmia":
            return run_rmia(
                self.observations,
                self.membership,
                self.labels,
                self.target_client_id,
                float(self.config.get("auxiliary_fraction", 0.5)),
                self.seed,
                offline_a=float(self.config.get("rmia_offline_a", 0.3)),
                gamma=float(self.config.get("rmia_gamma", 1.0)),
            )
        if attack == "quantile_mia":
            return run_quantile_mia(
                self.observations,
                self.membership,
                self.labels,
                self.target_client_id,
                float(self.config.get("auxiliary_fraction", 0.5)),
                self.seed,
                quantile=float(self.config.get("qmia_quantile", 0.9)),
                epochs=int(self.config.get("qmia_epochs", 200)),
                learning_rate=float(self.config.get("qmia_learning_rate", 0.01)),
            )
        if attack in {"pipra", "imia"}:
            target_model = copy.deepcopy(final_model)
            try:
                target_state, _, _ = last_client_states(
                    self.observations, self.target_client_id
                )
                target_model.load_state_dict(target_state, strict=False)
            except ValueError:
                target_model.load_state_dict(final_state, strict=False)
            if attack == "pipra":
                return run_pipra(
                    target_model,
                    self.images,
                    self.labels,
                    self.membership,
                    float(self.config.get("auxiliary_fraction", 0.5)),
                    self.seed,
                    shadow_prompts=int(self.config.get("pipra_shadow_prompts", 4)),
                    shadow_steps=int(self.config.get("pipra_shadow_steps", 20)),
                    shadow_learning_rate=float(
                        self.config.get("pipra_shadow_learning_rate", 0.02)
                    ),
                    attack_epochs=int(self.config.get("pipra_attack_epochs", 200)),
                    attack_learning_rate=float(
                        self.config.get("pipra_attack_learning_rate", 0.01)
                    ),
                    temperature=float(self.config.get("pipra_temperature", 0.1)),
                )
            return run_imia(
                target_model,
                self.images,
                self.labels,
                self.membership,
                float(self.config.get("auxiliary_fraction", 0.5)),
                self.seed,
                imitative_models=int(self.config.get("imia_models", 4)),
                warmup_steps=int(self.config.get("imia_warmup_steps", 10)),
                imitation_steps=int(self.config.get("imia_imitation_steps", 20)),
                pivot_steps=int(self.config.get("imia_pivot_steps", 20)),
                learning_rate=float(self.config.get("imia_learning_rate", 0.02)),
                pivots_per_class=int(self.config.get("imia_pivots_per_class", 4)),
            )
        if attack == "yoqo":
            return run_yoqo(
                final_model,
                self.observations,
                self.target_client_id,
                self.images,
                self.labels,
                self.membership,
                max_samples=int(self.config.get("query_max_samples", 16)),
                steps=int(self.config.get("yoqo_steps", 20)),
                learning_rate=float(self.config.get("yoqo_learning_rate", 0.01)),
                epsilon=float(self.config.get("query_epsilon", 0.1)),
                distortion_weight=float(
                    self.config.get("yoqo_distortion_weight", 1.0)
                ),
                reference_models=int(self.config.get("query_reference_models", 2)),
            )
        if attack == "canary":
            return run_canary(
                final_model,
                self.observations,
                self.target_client_id,
                self.images,
                self.labels,
                self.membership,
                max_samples=int(self.config.get("query_max_samples", 16)),
                num_canaries=int(self.config.get("canary_num_queries", 2)),
                optimization_steps=int(self.config.get("canary_steps", 20)),
                shadow_steps=int(self.config.get("canary_shadow_steps", 3)),
                learning_rate=float(self.config.get("canary_learning_rate", 0.01)),
                shadow_learning_rate=float(
                    self.config.get("canary_shadow_learning_rate", 0.02)
                ),
                epsilon=float(self.config.get("query_epsilon", 0.1)),
                reference_models=int(self.config.get("query_reference_models", 2)),
            )
        if attack == "promptmia":
            return run_promptmia(
                final_model,
                final_state,
                self.users[self.target_client_id],
                self.images,
                self.labels,
                self.membership,
                max_samples=int(self.config.get("promptmia_max_samples", 16)),
                adversarial_keys=int(self.config.get("promptmia_keys", 4)),
                delta_min=float(self.config.get("promptmia_delta_min", 0.02)),
                similarity_span=float(
                    self.config.get("promptmia_similarity_span", 0.05)
                ),
                seed=self.seed,
            )
        raise AssertionError(f"Unhandled attack {attack}")

    def finalize(
        self,
        final_model: torch.nn.Module,
        final_state: dict[str, torch.Tensor],
    ) -> list[dict]:
        if not self.enabled:
            return []
        os.makedirs(self.results_dir, exist_ok=True)
        final_model = copy.deepcopy(final_model).to(self.device)
        final_model.load_state_dict(final_state, strict=False)
        if self.defense_name == "hamp":
            attach_hamp_output_transform(
                final_model,
                float(self.defense_config.get("hamp_output_temperature", 4.0)),
            )
        for attack in self.attacks:
            try:
                self.results.append(self._run(attack, final_model, final_state))
            except Exception as error:
                logger.exception("Membership attack %s failed", attack)
                self.errors[attack] = f"{type(error).__name__}: {error}"
        summaries = [result.to_summary() for result in self.results]
        with open(
            os.path.join(self.results_dir, "summary.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(
                {
                    "target_client_id": self.target_client_id,
                    "defense": self.defense_name,
                    "attacks": summaries,
                    "errors": self.errors,
                },
                file,
                indent=2,
            )
        with open(
            os.path.join(self.results_dir, "predictions.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(("attack", "sample_index", "membership", "score"))
            for result in self.results:
                for index, label, score in zip(
                    result.sample_indices.tolist(),
                    result.labels.tolist(),
                    result.scores.tolist(),
                ):
                    writer.writerow((result.name, index, label, score))
        torch.save(
            {
                "candidate_labels": self.labels.detach().cpu(),
                "membership": self.membership,
                "observations": self.observations,
            },
            os.path.join(self.results_dir, "signals.pt"),
        )
        if self.errors and bool(self.config.get("strict", True)):
            failed = ", ".join(sorted(self.errors))
            raise RuntimeError(
                f"Configured membership attacks failed: {failed}. "
                "Partial diagnostics were saved to privacy_audit/summary.json."
            )
        return summaries
