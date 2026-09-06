from __future__ import annotations

import copy
import csv
import json
import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from privacy_attacks.code_poison import run_code_poison_attack
from privacy_attacks.base import AttackResult
from privacy_attacks.features import (
    flatten_state_delta,
    logits_and_representation,
    per_sample_prompt_gradients,
    trainable_names,
)
from privacy_attacks.fedmia import FEDMIA_MEASUREMENT_NAMES, run_fedmia
from privacy_attacks.fedmia_baselines import run_fedmia_baseline
from privacy_attacks.imia import run_imia
from privacy_attacks.model_utils import last_client_states, trainable_scope_name
from privacy_attacks.pipra import run_pipra
from privacy_attacks.promptmia import run_promptmia
from privacy_attacks.promptres import promptres_round_scores, run_promptres
from privacy_attacks.projres_mlp import strict_mlp_projres
from privacy_attacks.quantile import run_quantile_mia
from privacy_attacks.query_attacks import run_canary, run_yoqo
from privacy_attacks.rmia import run_rmia
from privacy_attacks.transfer import run_transfer_representation_attack
from privacy_attacks.update_attacks import run_update_attack
from privacy_attacks.whitebox import run_active_whitebox, run_passive_whitebox
from privacy_defenses import attach_hamp_output_transform
from privacy_defenses.www import infer_other_clients_state
from privacy_defenses.www_validation import (
    _pearson,
    _spearman,
    validate_www_attack_relationships,
)
from utils.data_loader import group_idx_by_class

logger = logging.getLogger(__name__)

SUPPORTED_ATTACKS = {
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

# These attacks can be evaluated client by client from one shared observation
# trajectory, then combined without treating client identity as membership
# evidence. Active/query attacks remain single-client because their probes must
# be scheduled and accounted for separately for every target user.
POOLED_CLIENT_ATTACKS = {
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
}

_CLIENT_CANDIDATE_FIELDS = {
    "confidence",
    "pre_confidence",
    "true_label_confidence",
    "pre_true_label_confidence",
    "cosine",
    "gradient_difference",
    "gradient_diff_score",
    "probabilities",
    "representations",
    "promptres",
}
_CANDIDATE_FIELDS = {
    "gradient_signature",
    "candidate_labels",
}

_FEDMIA_BASELINE_ATTACKS = {
    "blackbox_loss",
    "loss_series",
    "grad_cosine",
    "avg_cosine",
}
_UPDATE_ATTACKS = {"gradient_diff", "score_diff", "score_ratio", "fta"}
_EXACT_BATCH_MEMBERSHIP_ATTACKS = {
    "blackbox_loss",
    "grad_cosine",
    "gradient_diff",
    "projres",
    "score_diff",
    "score_ratio",
}
_TARGET_ONLY_SIGNAL_ATTACKS = _FEDMIA_BASELINE_ATTACKS | _UPDATE_ATTACKS
_SINGLE_ROUND_ATTACKS = {"blackbox_loss", "grad_cosine", "projres"}
_PERIODIC_METRIC_ATTACKS = _FEDMIA_BASELINE_ATTACKS | {
    "fedmia_loss",
    "fedmia_cosine",
} | _UPDATE_ATTACKS | {"projres"}
_EXACT_BATCH_REPORTED_FPR_TARGETS = (0.1, 0.01)


def _signal_needs(attacks: set[str], full_signals: bool = False) -> dict[str, bool]:
    """Return only the signal families needed by the attacks due this round."""
    return {
        "confidence": full_signals
        or bool(
            attacks
            & {
                "blackbox_loss",
                "loss_series",
                "nasr_passive",
                "fedmia_loss",
                "score_diff",
                "score_ratio",
                "fta",
            }
        ),
        "pre_confidence": full_signals
        or bool(attacks & {"score_diff", "score_ratio", "fta"}),
        "true_label_confidence": full_signals or "fta" in attacks,
        "pre_true_label_confidence": full_signals or "fta" in attacks,
        "cosine": full_signals
        or bool(
            attacks
            & {"grad_cosine", "avg_cosine", "nasr_passive", "fedmia_cosine"}
        ),
        "promptres": full_signals or "promptres" in attacks,
        "gradient_diff_score": full_signals or "gradient_diff" in attacks,
        "whitebox_features": full_signals or "nasr_passive" in attacks,
        "probabilities": full_signals
        or bool(attacks & {"nasr_passive", "rmia", "quantile_mia"}),
        "representations": full_signals
        or bool(
            attacks
            & {"nasr_passive", "transfer_representation", "quantile_mia"}
        ),
        "client_states": full_signals
        or bool(attacks & {"pipra", "imia", "yoqo", "canary", "promptmia"}),
    }


class MembershipAuditor:
    """Collect once, then run configured parameter-efficient membership attacks."""

    exact_batch_membership_attacks: set[str] = set()

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
        federated_method: str = "fedavg",
        num_classes: int | None = None,
    ):
        self.config = config or {}
        self.defense_config = dict(defense_config or {"name": "none"})
        self.defense_name = str(self.defense_config.get("name", "none")).lower()
        self.federated_method = str(federated_method).lower()
        self.model_type = str(getattr(model, "model_type", "prompt"))
        self.audit_view = str(
            self.config.get("audit_view", "protocol_plus_released_prompts")
        ).lower()
        if self.audit_view == "protocol_plus_queries":
            self.audit_view = "protocol_plus_released_prompts"
        if self.defense_name == "cofedmid" and self.audit_view == "full_whitebox":
            raise ValueError(
                "CoFedMID auditing must observe defended protocol uploads, "
                "not private pre-noise client states."
            )
        if self.audit_view not in {
            "protocol_plus_released_prompts",
            "full_whitebox",
            "released_prompt",
        }:
            raise ValueError(
                "audit.audit_view must be protocol_plus_released_prompts, full_whitebox, or released_prompt."
            )
        self.enabled = bool(self.config.get("enabled", True))
        self.attacks = list(
            self.config.get(
                "attacks",
                [
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
                ],
            )
        )
        unknown = sorted(set(self.attacks) - SUPPORTED_ATTACKS)
        if unknown:
            raise ValueError(f"Unsupported membership attacks: {', '.join(unknown)}")
        self.signal_storage = str(
            self.config.get("signal_storage", "compact")
        ).lower()
        if self.signal_storage not in {"none", "compact", "full"}:
            raise ValueError("audit.signal_storage must be none, compact, or full.")
        full_signals = self.signal_storage == "full"
        requested_attacks = set(self.attacks)
        configured_needs = _signal_needs(requested_attacks, full_signals)
        self._needs_confidence = configured_needs["confidence"]
        self._needs_pre_confidence = configured_needs["pre_confidence"]
        self._needs_true_label_confidence = configured_needs[
            "true_label_confidence"
        ]
        self._needs_pre_true_label_confidence = configured_needs[
            "pre_true_label_confidence"
        ]
        self._needs_cosine = configured_needs["cosine"]
        self._needs_gradient_diff_score = configured_needs[
            "gradient_diff_score"
        ]
        self._needs_promptres = configured_needs["promptres"]
        self._needs_whitebox_features = configured_needs["whitebox_features"]
        self._needs_probabilities = configured_needs["probabilities"]
        self._needs_representations = configured_needs["representations"]
        self._needs_client_states = configured_needs["client_states"]
        if not 0 <= target_client_id < len(users):
            raise ValueError("target_client_id is outside the client range.")
        configured_client_ids = self.config.get("audit_client_ids")
        if configured_client_ids is None:
            self.audit_client_ids = [int(target_client_id)]
        elif isinstance(configured_client_ids, str):
            if configured_client_ids.lower() != "all":
                raise ValueError("audit_client_ids must be 'all' or a list.")
            self.audit_client_ids = list(range(len(users)))
        elif isinstance(configured_client_ids, list):
            self.audit_client_ids = [int(value) for value in configured_client_ids]
        else:
            raise ValueError("audit_client_ids must be 'all' or a list.")
        if (
            not self.audit_client_ids
            or len(set(self.audit_client_ids)) != len(self.audit_client_ids)
            or min(self.audit_client_ids) < 0
            or max(self.audit_client_ids) >= len(users)
        ):
            raise ValueError("audit_client_ids must contain unique existing clients.")
        self.pooled_client_audit = len(self.audit_client_ids) > 1
        if self.pooled_client_audit and set(self.attacks) - POOLED_CLIENT_ATTACKS:
            raise ValueError(
                "Multi-client pooled auditing supports only: "
                + ", ".join(sorted(POOLED_CLIENT_ATTACKS))
            )
        self.num_classes = int(
            num_classes
            if num_classes is not None
            else len(getattr(model, "classnames", ()))
        )
        if self.num_classes <= 0:
            raise ValueError("Membership auditing requires a positive class count.")
        # Plain training runs do not need an auditor-side model clone. This is
        # especially important for multi-billion-byte frozen language-model
        # backbones used with parameter-efficient fine-tuning.
        # Client-scoped PEFT models already share one immutable multi-GB
        # backbone. Auditing is serialized between local-training rounds, so
        # reuse that model and only load released trainable states into it.
        self.model = (
            model
            if getattr(model, "client_scoped_parameters", False)
            else copy.deepcopy(model).to(device)
            if self.enabled
            else model
        )
        self.initial_prompt_state = (
            {}
            if getattr(model, "client_scoped_parameters", False)
            else {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
        )
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
        self.few_shot = bool(self.config.get("few_shot", False))
        self.fpl_shots = self.config.get("fpl_shots")
        self.allow_partial_client_audit = bool(
            self.config.get("allow_partial_client_audit", self.few_shot)
        )
        self.requested_audit_client_ids = list(self.audit_client_ids)
        self.skipped_audit_clients: dict[str, str] = {}
        self.exact_batch_skipped_rounds: list[dict] = []
        self.audit_interval = int(self.config.get("audit_interval", 1))
        configured_attack_intervals = self.config.get(
            "attack_audit_intervals", {}
        )
        if not isinstance(configured_attack_intervals, dict):
            raise ValueError("audit.attack_audit_intervals must be a mapping.")
        unknown_interval_attacks = sorted(
            set(configured_attack_intervals) - SUPPORTED_ATTACKS
        )
        if unknown_interval_attacks:
            raise ValueError(
                "Unsupported attacks in audit.attack_audit_intervals: "
                + ", ".join(unknown_interval_attacks)
            )
        self.attack_audit_intervals = {}
        for attack, configured_interval in configured_attack_intervals.items():
            if isinstance(configured_interval, bool):
                raise ValueError(
                    f"Audit interval for {attack} must be a positive integer."
                )
            try:
                interval = int(configured_interval)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Audit interval for {attack} must be a positive integer."
                ) from error
            if interval <= 0 or str(configured_interval).strip() != str(interval):
                raise ValueError(
                    f"Audit interval for {attack} must be a positive integer."
                )
            self.attack_audit_intervals[str(attack)] = interval
        self.audit_batch_size = int(self.config.get("audit_batch_size", 64))
        self.total_rounds = int(self.config.get("total_rounds", 0))
        self.calibration_fraction = float(self.config.get("calibration_fraction", 0.5))
        self.match_candidate_labels = bool(
            self.config.get("match_candidate_labels", False)
        )
        self.candidate_sampling_mode = str(
            self.config.get("candidate_sampling", "legacy")
        ).lower()
        if self.candidate_sampling_mode not in {
            "legacy",
            "fedmia_mix",
            "low_fpr_full",
            "balanced_holdout",
            "balanced_global_holdout",
        }:
            raise ValueError(
                "audit.candidate_sampling must be legacy, fedmia_mix, "
                "low_fpr_full, balanced_holdout, or "
                "balanced_global_holdout."
            )
        if self.candidate_sampling_mode in {
            "balanced_holdout",
            "balanced_global_holdout",
        }:
            self.match_candidate_labels = True
        self.require_full_target_train_members = bool(
            self.config.get("require_full_target_train_members", False)
        )
        if (
            self.require_full_target_train_members
            and self.candidate_sampling_mode != "balanced_global_holdout"
        ):
            raise ValueError(
                "require_full_target_train_members requires "
                "candidate_sampling=balanced_global_holdout."
            )
        self.nonmember_to_member_ratio = float(
            self.config.get("nonmember_to_member_ratio", 1.0)
        )
        if self.nonmember_to_member_ratio <= 0:
            raise ValueError(
                "audit.nonmember_to_member_ratio must be positive."
            )
        configured_batch_attacks = self.config.get(
            "exact_batch_membership_attacks", []
        )
        if not isinstance(configured_batch_attacks, list):
            raise ValueError(
                "audit.exact_batch_membership_attacks must be a list."
            )
        self.exact_batch_membership_attacks = {
            str(attack).lower() for attack in configured_batch_attacks
        }
        unsupported_batch_attacks = sorted(
            self.exact_batch_membership_attacks
            - _EXACT_BATCH_MEMBERSHIP_ATTACKS
        )
        if unsupported_batch_attacks:
            raise ValueError(
                "Exact-batch membership is supported only for: "
                + ", ".join(sorted(_EXACT_BATCH_MEMBERSHIP_ATTACKS))
                + ". Unsupported: "
                + ", ".join(unsupported_batch_attacks)
            )
        unconfigured_batch_attacks = sorted(
            self.exact_batch_membership_attacks - set(self.attacks)
        )
        if unconfigured_batch_attacks:
            raise ValueError(
                "Exact-batch attacks must also appear in audit.attacks: "
                + ", ".join(unconfigured_batch_attacks)
            )
        if "projres" in self.attacks and "projres" not in (
            self.exact_batch_membership_attacks
        ):
            raise ValueError(
                "Unified ProjRes must use the shared exact-batch "
                "membership protocol."
            )
        if self.exact_batch_membership_attacks and self.pooled_client_audit:
            raise ValueError(
                "Exact-batch membership currently requires one audit client."
            )
        if self.exact_batch_membership_attacks and (
            self.candidate_sampling_mode != "balanced_global_holdout"
        ):
            raise ValueError(
                "Exact-batch membership requires "
                "candidate_sampling=balanced_global_holdout."
            )
        configured_exact_batch_ratio = self.config.get(
            "exact_batch_nonmember_to_member_ratio",
            self.nonmember_to_member_ratio,
        )
        self.exact_batch_nonmember_ratio = int(configured_exact_batch_ratio)
        if (
            isinstance(configured_exact_batch_ratio, bool)
            or self.exact_batch_nonmember_ratio < 1
            or float(configured_exact_batch_ratio)
            != self.exact_batch_nonmember_ratio
        ):
            raise ValueError(
                "audit.exact_batch_nonmember_to_member_ratio must be a "
                "positive integer."
            )
        self.exact_batch_projres_config = dict(
            self.config.get("exact_batch_projres", {})
        )
        if "projres" in self.exact_batch_membership_attacks:
            if self.model_type not in {
                "clip_mlp",
                "bert_adapter",
                "bert_lora",
                "gpt2_adapter",
                "clip_adapter",
                "visual_adapter",
                "clip_lora",
            }:
                raise ValueError(
                    "Unified exact-batch ProjRes requires a Transformer "
                    "PEFT, CLIP-MLP, CLIP-Adapter, or CLIP-LoRA model."
                )
            if self.exact_batch_projres_config.get("threshold") is not None:
                raise ValueError(
                    "Unified ProjRes is ranking-only and requires threshold=null."
                )
            if (
                str(
                    self.exact_batch_projres_config.get(
                        "decision_mode", "ranking"
                    )
                ).lower()
                != "ranking"
            ):
                raise ValueError("Unified ProjRes decision_mode must be ranking.")
        if self.candidate_sampling_mode == "balanced_global_holdout" and (
            not self.nonmember_to_member_ratio.is_integer()
            or self.nonmember_to_member_ratio < 1
        ):
            raise ValueError(
                "balanced_global_holdout requires a positive integer "
                "nonmember_to_member_ratio."
            )
        self.paper_balanced_evaluation_size = int(
            self.config.get("paper_balanced_evaluation_size", 0)
        )
        if self.paper_balanced_evaluation_size not in {0, 100}:
            raise ValueError(
                "audit.paper_balanced_evaluation_size must be 0 or 100."
            )
        if self.paper_balanced_evaluation_size and (
            self.candidate_sampling_mode != "balanced_global_holdout"
        ):
            raise ValueError(
                "paper_balanced_evaluation_size requires "
                "candidate_sampling=balanced_global_holdout."
            )
        if self.candidate_sampling_mode == "fedmia_mix" and self.match_candidate_labels:
            raise ValueError(
                "FedMIA mix sampling reproduces the reference evaluation and "
                "therefore requires match_candidate_labels=false."
            )
        self.low_fpr_min_nonmembers = int(
            self.config.get("low_fpr_min_nonmembers", 1000)
        )
        self.low_fpr_max_members = int(
            self.config.get("low_fpr_max_members", 0)
        )
        self.low_fpr_max_nonmembers = int(
            self.config.get("low_fpr_max_nonmembers", 0)
        )
        if self.candidate_sampling_mode in {
            "low_fpr_full",
            "balanced_holdout",
            "balanced_global_holdout",
        }:
            if self.low_fpr_max_members < 0 or self.low_fpr_max_members == 1:
                raise ValueError(
                    "low_fpr_max_members must be 0 (unlimited) or at least 2."
                )
            if (
                self.low_fpr_max_nonmembers < 0
                or self.low_fpr_max_nonmembers == 1
            ):
                raise ValueError(
                    "low_fpr_max_nonmembers must be 0 (unlimited) or at least 2."
                )
        if self.candidate_sampling_mode == "low_fpr_full":
            if self.low_fpr_min_nonmembers < 1000:
                raise ValueError(
                    "low_fpr_min_nonmembers must be at least 1000 to resolve "
                    "0.1% FPR."
                )
            if (
                0
                < self.low_fpr_max_nonmembers
                < self.low_fpr_min_nonmembers
            ):
                raise ValueError(
                    "low_fpr_max_nonmembers must be 0 (unlimited) or at least "
                    "low_fpr_min_nonmembers."
                )
        if self.audit_interval <= 0 or self.audit_batch_size <= 0:
            raise ValueError("audit_interval and audit_batch_size must be positive.")
        self.observations: list[dict] = []
        self.exact_batch_observations: list[dict] = []
        self.exact_batch_candidate_selections: list[dict] = []
        self._exact_batch_nonmember_dataset = None
        self._exact_batch_nonmember_groups = None
        self.results = []
        self.errors: dict[str, str] = {}
        self.candidate_inputs_are_features = False
        self.low_fpr_candidate_selection: dict | None = None
        self.candidate_local_indices: torch.Tensor | None = None
        self.paper_balanced_candidate_indices: torch.Tensor | None = None
        self.paper_balanced_evaluation: dict | None = None
        self.www_candidate_scoring = bool(
            self.config.get("www_candidate_scoring", False)
        )
        self.candidate_source_names: list[str] | None = None
        if self.www_candidate_scoring:
            if self.federated_method != "fedavg":
                raise ValueError(
                    "WWW candidate scoring requires linear FedAvg aggregation."
                )
            if self.candidate_sampling_mode != "fedmia_mix" or set(
                self.attacks
            ) != {"fedmia_loss"}:
                raise ValueError(
                    "WWW candidate scoring requires the fixed fedmia_mix view "
                    "with FedMIA-Loss."
                )
            if self.pooled_client_audit:
                raise ValueError(
                    "WWW candidate scoring currently supports one target client."
                )
            if self.defense_name != "none":
                raise ValueError(
                    "WWW candidate scoring is observational and requires "
                    "defense.name=none."
                )

        if self.enabled:
            if self.candidate_sampling_mode == "balanced_global_holdout":
                self._initialize_balanced_global_holdout_candidates(
                    users, target_client_id
                )
                return
            if self.candidate_sampling_mode == "balanced_holdout":
                self._initialize_balanced_holdout_candidates(
                    users, target_client_id
                )
                return
            if self.candidate_sampling_mode == "low_fpr_full":
                self._initialize_low_fpr_full_candidates(users, target_client_id)
                return
            legacy_max = int(self.config.get("max_samples_per_group", 32))
            max_members = int(self.config.get("max_member_samples", legacy_max))
            max_nonmembers = int(
                self.config.get("max_nonmember_samples", legacy_max)
            )
            if max_members < 2 or max_nonmembers < 2:
                raise ValueError(
                    "max_member_samples and max_nonmember_samples must be at least 2."
                )
            if self.pooled_client_audit:
                if (
                    self.candidate_sampling_mode == "legacy"
                    and not self.match_candidate_labels
                ):
                    raise ValueError(
                        "Multi-client pooled auditing requires exact label matching."
                    )
                image_parts = []
                label_parts = []
                membership_parts = []
                client_parts = []
                self.candidate_sampling_by_client = {}

                per_client_overlap = {}
                eligible_client_ids = []
                for client_id in self.audit_client_ids:
                    target = users[client_id]
                    try:
                        if self.candidate_sampling_mode == "fedmia_mix":
                            (
                                member_images,
                                member_labels,
                                nonmember_images,
                                nonmember_labels,
                                sampling,
                            ) = self._collect_fedmia_mix_candidates(
                                users,
                                client_id,
                                max_members,
                                max_nonmembers,
                            )
                        else:
                            (
                                member_images,
                                member_labels,
                                nonmember_images,
                                nonmember_labels,
                                sampling,
                            ) = self._collect_exact_paired_candidates(
                                target.train_data,
                                target.test_data,
                                max_members,
                                max_nonmembers,
                                client_id,
                            )
                    except ValueError as error:
                        if not self.allow_partial_client_audit:
                            raise
                        self.skipped_audit_clients[str(client_id)] = str(error)
                        logger.warning(
                            "Skipping client %d in few-shot privacy audit: %s",
                            client_id,
                            error,
                        )
                        continue
                    eligible_client_ids.append(client_id)
                    images = torch.cat((member_images, nonmember_images))
                    labels = torch.cat((member_labels, nonmember_labels))
                    memberships = torch.cat(
                        (
                            torch.ones(member_labels.numel(), dtype=torch.long),
                            torch.zeros(nonmember_labels.numel(), dtype=torch.long),
                        )
                    )
                    image_parts.append(images)
                    label_parts.append(labels)
                    membership_parts.append(memberships)
                    client_parts.append(
                        torch.full((labels.numel(),), client_id, dtype=torch.long)
                    )
                    self.candidate_sampling_by_client[str(client_id)] = sampling
                    support = set(member_labels.unique().tolist())
                    other_support = set()
                    for user in users:
                        if user.id == client_id:
                            continue
                        grouped = group_idx_by_class(
                            user.train_data, self.num_classes
                        )
                        other_support.update(
                            class_id
                            for class_id, indices in enumerate(grouped)
                            if indices
                        )
                    per_client_overlap[str(client_id)] = sorted(
                        support & other_support
                    )
                if not eligible_client_ids:
                    if self.candidate_sampling_mode == "fedmia_mix":
                        raise ValueError(
                            "FedMIA mix audit found no client with enough member and "
                            "mixed non-member candidates. Increase fpl_shots, reduce "
                            "the requested ratio, or provide a larger test split."
                        )
                    raise ValueError(
                        "Few-shot privacy audit found no client with at least two "
                        "same-label member/non-member pairs. Increase fpl_shots or "
                        "dirichlet_alpha, reduce total_users, or provide a larger "
                        "non-member test split."
                    )
                self.audit_client_ids = eligible_client_ids
                # Pooled candidates can exceed a gigabyte at CLIP resolution.
                # Keep them on CPU and transfer only the active audit batch.
                self.images = torch.cat(image_parts)
                self.labels = torch.cat(label_parts)
                self.membership = torch.cat(membership_parts)
                self.candidate_client_ids = torch.cat(client_parts)
                self.candidate_label_support = sorted(
                    self.labels.detach().cpu().unique().tolist()
                )
                self.null_client_candidate_label_overlap_by_client = (
                    per_client_overlap
                )
                self.null_client_candidate_label_overlap = sorted(
                    {
                        class_id
                        for overlap in per_client_overlap.values()
                        for class_id in overlap
                    }
                )
                self.nonmember_source_priority = (
                    ["independent_test", "other_client_train"]
                    if self.candidate_sampling_mode == "fedmia_mix"
                    else ["same_client_test"]
                )
            else:
                target = users[target_client_id]
                if self.candidate_sampling_mode == "fedmia_mix":
                    (
                        member_images,
                        member_labels,
                        nonmember_images,
                        nonmember_labels,
                        sampling,
                    ) = self._collect_fedmia_mix_candidates(
                        users,
                        target_client_id,
                        max_members,
                        max_nonmembers,
                    )
                    self.images = torch.cat(
                        (member_images, nonmember_images)
                    ).to(device)
                    self.labels = torch.cat(
                        (member_labels, nonmember_labels)
                    ).to(device)
                    self.membership = torch.cat(
                        (
                            torch.ones(member_labels.numel(), dtype=torch.long),
                            torch.zeros(nonmember_labels.numel(), dtype=torch.long),
                        )
                    )
                    self.candidate_client_ids = torch.full(
                        (self.labels.numel(),), target_client_id, dtype=torch.long
                    )
                    self.candidate_sampling_by_client = {
                        str(target_client_id): sampling
                    }
                    source_names = ["target_client_train"] * int(
                        member_labels.numel()
                    )
                    for source_name, source_count in sampling[
                        "nonmember_source_counts"
                    ].items():
                        source_names.extend([source_name] * int(source_count))
                    if len(source_names) != int(self.labels.numel()):
                        raise AssertionError(
                            "FedMIA candidate-source metadata is not sample-aligned."
                        )
                    self.candidate_source_names = source_names
                    candidate_support = set(member_labels.unique().tolist())
                    other_training_support = set()
                    for user in users:
                        if user.id == target_client_id:
                            continue
                        grouped = group_idx_by_class(user.train_data, self.num_classes)
                        other_training_support.update(
                            class_id
                            for class_id, indices in enumerate(grouped)
                            if indices
                        )
                    self.null_client_candidate_label_overlap = sorted(
                        candidate_support & other_training_support
                    )
                    self.null_client_candidate_label_overlap_by_client = {
                        str(target_client_id): self.null_client_candidate_label_overlap
                    }
                    self.candidate_label_support = sorted(
                        set(self.labels.detach().cpu().unique().tolist())
                    )
                    self.nonmember_source_priority = [
                        "independent_test",
                        "other_client_train",
                    ]
                    return
                member_images, member_labels = self._collect_many(
                    [target.train_data], max_members
                )
                candidate_support = set(member_labels.unique().tolist())
                other_training_support = set()
                class_count = max(
                    self.num_classes,
                    int(member_labels.max().item()) + 1,
                )
                for user in users:
                    if user.id == target_client_id:
                        continue
                    grouped = group_idx_by_class(user.train_data, class_count)
                    other_training_support.update(
                        class_id
                        for class_id, indices in enumerate(grouped)
                        if indices
                    )
                self.null_client_candidate_label_overlap = sorted(
                    candidate_support & other_training_support
                )
                self.null_client_candidate_label_overlap_by_client = {
                    str(target_client_id): self.null_client_candidate_label_overlap
                }
                self.candidate_label_support = sorted(candidate_support)
                unused_training_pool = self._unused_few_shot_training_pool(users)
                nonmember_datasets = [target.test_data]
                self.nonmember_source_priority = ["target_test"]
                if unused_training_pool is not None:
                    nonmember_datasets.append(unused_training_pool)
                    self.nonmember_source_priority.append("unused_training_pool")
                nonmember_datasets += [
                    user.test_data for user in users if user.id != target_client_id
                ] + [
                    user.train_data for user in users if user.id != target_client_id
                ]
                self.nonmember_source_priority += [
                    "other_client_test",
                    "other_client_train",
                ]
                if self.match_candidate_labels:
                    nonmember_images, nonmember_labels = self._collect_label_matched(
                        nonmember_datasets, member_labels, max_nonmembers
                    )
                else:
                    nonmember_images, nonmember_labels = self._collect_many(
                        nonmember_datasets, max_nonmembers
                    )
                self.images = torch.cat((member_images, nonmember_images)).to(device)
                self.labels = torch.cat((member_labels, nonmember_labels)).to(device)
                self.membership = torch.cat(
                    (
                        torch.ones(member_labels.numel(), dtype=torch.long),
                        torch.zeros(nonmember_labels.numel(), dtype=torch.long),
                    )
                )
                self.candidate_client_ids = torch.full(
                    (self.labels.numel(),), target_client_id, dtype=torch.long
                )
                self.candidate_sampling_by_client = {}

    @torch.no_grad()
    def _collect_encoded_candidates(
        self,
        datasets: list,
        source_names: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """Encode complete datasets once for a scalable MLP-only audit."""
        if len(datasets) != len(source_names) or not datasets:
            raise ValueError("Low-FPR candidate sources must be non-empty and named.")
        feature_parts = []
        label_parts = []
        source_counts: dict[str, int] = {}
        self.model.eval()
        for dataset, source in zip(datasets, source_names):
            if len(dataset) == 0:
                source_counts[source] = 0
                continue
            # The main CLIP-MLP/Adapter path has already encoded these images
            # into CPU-resident TensorDatasets. Reuse their backing tensors
            # directly instead of copying every vector CPU -> GPU -> CPU.
            if isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 2:
                cached_features, cached_labels = dataset.tensors[:2]
                if (
                    cached_features.ndim == 2
                    and cached_features.shape[1]
                    == int(getattr(self.model, "projection_dim", -1))
                    and cached_labels.ndim == 1
                ):
                    feature_parts.append(cached_features.detach().cpu())
                    labels = cached_labels.detach().cpu().long()
                    label_parts.append(labels)
                    source_counts[source] = int(labels.numel())
                    continue
            count = 0
            loader = DataLoader(
                dataset,
                batch_size=self.audit_batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            for images, labels in loader:
                features = self.model.encode_images(images.to(self.device))
                feature_parts.append(features.detach().cpu())
                labels = labels.detach().cpu().long()
                label_parts.append(labels)
                count += int(labels.numel())
            source_counts[source] = count
        if not label_parts or sum(source_counts.values()) < 2:
            raise ValueError("Each low-FPR membership group needs two candidates.")
        return torch.cat(feature_parts), torch.cat(label_parts), source_counts

    @staticmethod
    def _stratified_subsample_indices(
        labels: torch.Tensor,
        maximum: int,
        seed: int,
    ) -> torch.Tensor:
        """Select an exact-size proportional, deterministic stratified sample."""
        labels = labels.detach().cpu().long().reshape(-1)
        sample_count = int(labels.numel())
        if maximum <= 0 or sample_count <= maximum:
            return torch.arange(sample_count, dtype=torch.long)

        classes, counts = torch.unique(labels, sorted=True, return_counts=True)
        ideal = counts.to(torch.float64) * (float(maximum) / sample_count)
        allocation = torch.floor(ideal).to(torch.long)
        remaining = maximum - int(allocation.sum())
        if remaining:
            fractions = (ideal - allocation).tolist()
            order = sorted(
                range(len(classes)),
                key=lambda index: (-fractions[index], int(classes[index])),
            )
            for index in order[:remaining]:
                allocation[index] += 1

        generator = torch.Generator().manual_seed(int(seed))
        selected_parts = []
        for class_id, class_sample_count in zip(classes, allocation):
            take = int(class_sample_count)
            if take == 0:
                continue
            class_indices = torch.nonzero(
                labels == class_id, as_tuple=False
            ).flatten()
            order = torch.randperm(class_indices.numel(), generator=generator)
            selected_parts.append(class_indices[order[:take]])
        selected = torch.cat(selected_parts)
        order = torch.randperm(selected.numel(), generator=generator)
        return selected[order]

    @staticmethod
    def _selected_source_counts(
        selected_indices: torch.Tensor,
        source_counts: dict[str, int],
    ) -> dict[str, int]:
        """Count selected pool positions using source-order boundaries."""
        selected_indices = selected_indices.detach().cpu().long()
        selected_counts = {}
        start = 0
        for source, count in source_counts.items():
            stop = start + int(count)
            selected_counts[source] = int(
                ((selected_indices >= start) & (selected_indices < stop)).sum()
            )
            start = stop
        return selected_counts

    def _initialize_low_fpr_full_candidates(
        self, users: list, target_client_id: int
    ) -> None:
        """Build reproducible low-FPR pools for one or more audit clients."""
        supported = {
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
        unsupported = sorted(set(self.attacks) - supported)
        if self.model_type not in {
            "clip_mlp",
            "clip_adapter",
            "visual_adapter",
        }:
            raise ValueError(
                "low_fpr_full currently requires a frozen-CLIP feature model."
            )
        if unsupported:
            raise ValueError(
                "low_fpr_full does not support: " + ", ".join(unsupported)
            )
        image_parts = []
        label_parts = []
        membership_parts = []
        client_parts = []
        local_index_parts = []
        candidate_sampling_by_client = {}
        selection_by_client = {}
        overlap_by_client = {}
        source_priority = []
        eligible_client_ids = []

        for client_id in self.audit_client_ids:
            try:
                target = users[client_id]
                member_features, member_labels, member_sources = (
                    self._collect_encoded_candidates(
                        [target.train_data], [f"target_train:{client_id}"]
                    )
                )
                nonmember_datasets = []
                nonmember_source_names = []
                for user in users:
                    if len(user.test_data):
                        nonmember_datasets.append(user.test_data)
                        nonmember_source_names.append(
                            f"independent_test:{user.id}"
                        )
                for user in users:
                    if user.id != client_id and len(user.train_data):
                        nonmember_datasets.append(user.train_data)
                        nonmember_source_names.append(
                            f"other_client_train:{user.id}"
                        )
                nonmember_features, nonmember_labels, nonmember_sources = (
                    self._collect_encoded_candidates(
                        nonmember_datasets, nonmember_source_names
                    )
                )
                minimum_nonmembers = self.low_fpr_min_nonmembers
                if nonmember_labels.numel() < minimum_nonmembers:
                    raise ValueError(
                        "low_fpr_full needs at least "
                        f"{minimum_nonmembers} non-members, but only found "
                        f"{nonmember_labels.numel()}."
                    )
            except ValueError as error:
                if not self.allow_partial_client_audit:
                    raise
                self.skipped_audit_clients[str(client_id)] = str(error)
                logger.warning(
                    "Skipping client %d in low-FPR audit: %s",
                    client_id,
                    error,
                )
                continue

            member_pool_count = int(member_labels.numel())
            nonmember_pool_count = int(nonmember_labels.numel())
            client_seed_offset = 1000003 * int(client_id)
            member_sampling_seed = self.seed + 104729 + client_seed_offset
            nonmember_sampling_seed = self.seed + 130363 + client_seed_offset
            member_pool_indices = self._stratified_subsample_indices(
                member_labels,
                self.low_fpr_max_members,
                member_sampling_seed,
            )
            nonmember_pool_indices = self._stratified_subsample_indices(
                nonmember_labels,
                self.low_fpr_max_nonmembers,
                nonmember_sampling_seed,
            )
            selected_member_sources = self._selected_source_counts(
                member_pool_indices, member_sources
            )
            selected_nonmember_sources = self._selected_source_counts(
                nonmember_pool_indices, nonmember_sources
            )
            member_features = member_features.index_select(
                0, member_pool_indices
            )
            member_labels = member_labels.index_select(0, member_pool_indices)
            nonmember_features = nonmember_features.index_select(
                0, nonmember_pool_indices
            )
            nonmember_labels = nonmember_labels.index_select(
                0, nonmember_pool_indices
            )
            selection = {
                "seed": self.seed,
                "member_sampling_seed": member_sampling_seed,
                "nonmember_sampling_seed": nonmember_sampling_seed,
                "index_convention": (
                    "zero-based positions in this client's complete concatenated "
                    "candidate pool; source boundaries follow the ordered "
                    "pool_source_counts"
                ),
                "member_pool_indices": member_pool_indices,
                "nonmember_pool_indices": nonmember_pool_indices,
                "member_pool_source_counts": member_sources,
                "nonmember_pool_source_counts": nonmember_sources,
            }
            selection_by_client[str(client_id)] = selection
            logger.info(
                "Low-FPR client %d candidates selected: members=%d/%d, "
                "non-members=%d/%d, seed=%d",
                client_id,
                int(member_labels.numel()),
                member_pool_count,
                int(nonmember_labels.numel()),
                nonmember_pool_count,
                self.seed,
            )
            images = torch.cat((member_features, nonmember_features))
            labels = torch.cat((member_labels, nonmember_labels))
            memberships = torch.cat(
                (
                    torch.ones(member_labels.numel(), dtype=torch.long),
                    torch.zeros(nonmember_labels.numel(), dtype=torch.long),
                )
            )
            image_parts.append(images)
            label_parts.append(labels)
            membership_parts.append(memberships)
            client_parts.append(
                torch.full((labels.numel(),), client_id, dtype=torch.long)
            )
            local_index_parts.append(
                torch.cat(
                    (
                        member_pool_indices.detach().cpu().long(),
                        torch.full(
                            (nonmember_labels.numel(),), -1, dtype=torch.long
                        ),
                    )
                )
            )
            eligible_client_ids.append(client_id)
            candidate_sampling_by_client[str(client_id)] = {
                "mode": "low_fpr_full",
                "member_count": int(member_labels.numel()),
                "nonmember_count": int(nonmember_labels.numel()),
                "member_pool_count": member_pool_count,
                "nonmember_pool_count": nonmember_pool_count,
                "member_source_counts": selected_member_sources,
                "nonmember_source_counts": selected_nonmember_sources,
                "member_pool_source_counts": member_sources,
                "nonmember_pool_source_counts": nonmember_sources,
                "minimum_nonmembers": minimum_nonmembers,
                "maximum_members": self.low_fpr_max_members,
                "maximum_nonmembers": self.low_fpr_max_nonmembers,
                "selection_seed": self.seed,
                "selection_method": (
                    "proportional_class_stratified_without_replacement"
                ),
                "selection_artifact": "candidate_selection.pt",
                "fpr_resolution": 1.0 / int(nonmember_labels.numel()),
            }
            overlap_by_client[str(client_id)] = sorted(
                set(member_labels.unique().tolist())
                & set(nonmember_labels.unique().tolist())
            )
            source_priority.extend(nonmember_source_names)

        if not eligible_client_ids:
            raise ValueError(
                "Low-FPR audit found no client with enough member and "
                "non-member candidates."
            )
        self.audit_client_ids = eligible_client_ids
        self.pooled_client_audit = len(eligible_client_ids) > 1
        if not self.pooled_client_audit:
            self.target_client_id = eligible_client_ids[0]
        self.low_fpr_candidate_selection = (
            {
                "scope": "pooled_clients",
                "seed": self.seed,
                "audit_client_ids": eligible_client_ids,
                "per_client": selection_by_client,
            }
            if self.pooled_client_audit
            else selection_by_client[str(eligible_client_ids[0])]
        )
        self.images = torch.cat(image_parts)
        self.labels = torch.cat(label_parts)
        self.membership = torch.cat(membership_parts)
        self.candidate_client_ids = torch.cat(client_parts)
        self.candidate_local_indices = torch.cat(local_index_parts)
        self.candidate_inputs_are_features = True
        self.candidate_sampling_by_client = candidate_sampling_by_client
        self.candidate_label_support = sorted(
            set(self.labels.detach().cpu().unique().tolist())
        )
        self.null_client_candidate_label_overlap = sorted(
            {
                class_id
                for overlap in overlap_by_client.values()
                for class_id in overlap
            }
        )
        self.null_client_candidate_label_overlap_by_client = overlap_by_client
        self.nonmember_source_priority = list(dict.fromkeys(source_priority))

    @staticmethod
    def _allocate_proportional_with_capacities(
        weights: list[int], capacities: list[int], total: int
    ) -> list[int]:
        """Allocate an exact proportional budget without exceeding capacities."""
        if len(weights) != len(capacities) or not weights:
            raise ValueError("Weights and capacities must be non-empty and aligned.")
        if total < 0 or sum(capacities) < total:
            raise ValueError("Candidate capacities cannot satisfy the requested total.")
        if total == 0:
            return [0] * len(weights)
        weight_total = sum(weights)
        if weight_total <= 0:
            raise ValueError("At least one candidate class must have positive weight.")

        ideals = [total * weight / weight_total for weight in weights]
        allocation = [0] * len(weights)
        for _ in range(total):
            available = [
                class_id
                for class_id, capacity in enumerate(capacities)
                if allocation[class_id] < capacity
            ]
            if not available:
                raise AssertionError("Proportional candidate allocation stalled.")
            selected = max(
                available,
                key=lambda class_id: (
                    ideals[class_id] - allocation[class_id],
                    weights[class_id],
                    -class_id,
                ),
            )
            allocation[selected] += 1
        return allocation

    def _collect_global_proportional_candidates(
        self,
        member_dataset,
        nonmember_datasets: list,
        nonmember_source_names: list[str],
        member_limit: int,
        nonmember_limit: int,
        client_id: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """Sample fixed-ratio train/global-test candidates.

        The default path preserves exact class proportions. When the complete
        target training set is required, every member is retained and class
        matching becomes best effort: per-class shortages are filled from
        other evaluation classes without replacement.
        """
        if len(nonmember_datasets) != len(nonmember_source_names):
            raise ValueError("Global holdout datasets and sources must be aligned.")
        if not nonmember_datasets:
            raise ValueError("Global holdout sampling requires evaluation data.")
        ratio = int(self.nonmember_to_member_ratio)
        nonmember_dataset = ConcatDataset(nonmember_datasets)
        member_groups = group_idx_by_class(member_dataset, self.num_classes)
        nonmember_groups = group_idx_by_class(
            nonmember_dataset, self.num_classes
        )
        member_pool_histogram = [len(group) for group in member_groups]
        nonmember_pool_histogram = [len(group) for group in nonmember_groups]
        member_capacities = [
            min(
                member_pool_histogram[class_id],
                nonmember_pool_histogram[class_id] // ratio,
            )
            for class_id in range(self.num_classes)
        ]
        maximum_members = min(
            len(member_dataset),
            len(member_dataset) if member_limit <= 0 else member_limit,
        )
        maximum_nonmembers = min(
            len(nonmember_dataset),
            len(nonmember_dataset)
            if nonmember_limit <= 0
            else nonmember_limit,
        )
        if self.require_full_target_train_members:
            member_budget = len(member_dataset)
            if maximum_members < member_budget:
                raise ValueError(
                    f"Client {client_id} requires all {member_budget} training "
                    f"members, but low_fpr_max_members={member_limit} truncates "
                    "that pool. Set low_fpr_max_members=0."
                )
            nonmember_budget = member_budget * ratio
            if maximum_nonmembers < nonmember_budget:
                raise ValueError(
                    f"Client {client_id} requires {nonmember_budget} independent "
                    "nonmembers for the configured ratio, but only "
                    f"{maximum_nonmembers} are available under the current cap."
                )
            member_quotas = list(member_pool_histogram)
            desired_nonmember_quotas = [
                quota * ratio for quota in member_quotas
            ]
            nonmember_quotas = [
                min(desired, available)
                for desired, available in zip(
                    desired_nonmember_quotas, nonmember_pool_histogram
                )
            ]
            shortage = nonmember_budget - sum(nonmember_quotas)
            if shortage:
                spare_capacities = [
                    available - selected
                    for available, selected in zip(
                        nonmember_pool_histogram, nonmember_quotas
                    )
                ]
                redistribution = self._allocate_proportional_with_capacities(
                    desired_nonmember_quotas,
                    spare_capacities,
                    shortage,
                )
                nonmember_quotas = [
                    selected + extra
                    for selected, extra in zip(
                        nonmember_quotas, redistribution
                    )
                ]
        else:
            member_budget = min(
                maximum_members,
                maximum_nonmembers // ratio,
                sum(member_capacities),
            )
            if member_budget < 2:
                raise ValueError(
                    f"Client {client_id} cannot construct two exact-ratio "
                    "label-matched member candidates."
                )
            member_quotas = self._allocate_proportional_with_capacities(
                member_pool_histogram,
                member_capacities,
                member_budget,
            )
            desired_nonmember_quotas = [
                quota * ratio for quota in member_quotas
            ]
            nonmember_quotas = list(desired_nonmember_quotas)

        def select_indices(
            groups: list[list[int]], quotas: list[int], salt: int
        ) -> list[int]:
            selected = []
            for class_id, count in enumerate(quotas):
                if count == 0:
                    continue
                generator = torch.Generator().manual_seed(
                    self.seed
                    + 104729 * (client_id + 1)
                    + salt
                    + 2 * class_id
                )
                order = torch.randperm(
                    len(groups[class_id]), generator=generator
                )[:count]
                selected.extend(
                    groups[class_id][position] for position in order.tolist()
                )
            shuffle_generator = torch.Generator().manual_seed(
                self.seed + 999983 * (client_id + 1) + salt
            )
            order = torch.randperm(len(selected), generator=shuffle_generator)
            return [selected[position] for position in order.tolist()]

        member_indices = select_indices(member_groups, member_quotas, 11)
        nonmember_indices = select_indices(
            nonmember_groups, nonmember_quotas, 17
        )
        member_images, member_labels = self._collect_many(
            [Subset(member_dataset, member_indices)], len(member_indices)
        )
        nonmember_images, nonmember_labels = self._collect_many(
            [Subset(nonmember_dataset, nonmember_indices)],
            len(nonmember_indices),
        )
        member_histogram = torch.bincount(
            member_labels, minlength=self.num_classes
        )
        nonmember_histogram = torch.bincount(
            nonmember_labels, minlength=self.num_classes
        )
        label_histograms_matched = torch.equal(
            nonmember_histogram, member_histogram * ratio
        )
        if (
            not self.require_full_target_train_members
            and not label_histograms_matched
        ):
            raise AssertionError(
                "Global holdout candidate label distributions drifted."
            )
        label_tv_distance = 0.5 * sum(
            abs(
                int(member_histogram[class_id])
                / int(member_histogram.sum())
                - int(nonmember_histogram[class_id])
                / int(nonmember_histogram.sum())
            )
            for class_id in range(self.num_classes)
        )
        nonmember_pool_source_counts = {
            source: len(dataset)
            for source, dataset in zip(
                nonmember_source_names, nonmember_datasets
            )
        }
        selected_nonmember_source_counts = self._selected_source_counts(
            torch.as_tensor(nonmember_indices, dtype=torch.long),
            nonmember_pool_source_counts,
        )
        return (
            member_images,
            member_labels,
            nonmember_images,
            nonmember_labels,
            {
                "client_id": int(client_id),
                "member_count": int(member_labels.numel()),
                "nonmember_count": int(nonmember_labels.numel()),
                "member_label_histogram": member_histogram.tolist(),
                "nonmember_label_histogram": nonmember_histogram.tolist(),
                "desired_nonmember_label_histogram": (
                    desired_nonmember_quotas
                ),
                "label_histograms_matched": label_histograms_matched,
                "label_matching_mode": (
                    "best_effort_full_target_train_global_test"
                    if self.require_full_target_train_members
                    else "exact_proportional_target_train_global_test"
                ),
                "label_total_variation_distance": label_tv_distance,
                "member_pool_label_histogram": member_pool_histogram,
                "nonmember_pool_label_histogram": nonmember_pool_histogram,
                "member_capacity_by_label": member_capacities,
                "sampling_seed": int(self.seed),
                "member_indices": member_indices,
                "nonmember_indices": nonmember_indices,
                "member_pool_source_counts": {
                    f"target_train:{client_id}": len(member_dataset)
                },
                "nonmember_pool_source_counts": nonmember_pool_source_counts,
                "member_source_counts": {
                    f"target_train:{client_id}": int(member_labels.numel())
                },
                "nonmember_source_counts": selected_nonmember_source_counts,
                "index_convention": (
                    "member indices are zero-based positions in the target "
                    "client train dataset; non-member indices are zero-based "
                    "positions in the client-id ordered concatenation of all "
                    "independent evaluation datasets"
                ),
            },
        )

    def _initialize_balanced_global_holdout_candidates(
        self, users: list, target_client_id: int
    ) -> None:
        """Build exact-ratio target-train/global-test candidate pools."""
        del target_client_id
        supported = {
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
        unsupported = sorted(set(self.attacks) - supported)
        if self.model_type not in {
            "clip_mlp",
            "clip_adapter",
            "visual_adapter",
            "clip_lora",
            "bert_adapter",
            "bert_lora",
            "gpt2_adapter",
        }:
            raise ValueError(
                "balanced_global_holdout requires a supported "
                "parameter-efficient model."
            )
        if unsupported:
            raise ValueError(
                "balanced_global_holdout does not support: "
                + ", ".join(unsupported)
            )
        nonmember_datasets = [
            user.test_data for user in users if len(user.test_data)
        ]
        nonmember_source_names = [
            f"independent_test:{user.id}"
            for user in users
            if len(user.test_data)
        ]
        if self.exact_batch_membership_attacks:
            self._exact_batch_nonmember_dataset = ConcatDataset(
                nonmember_datasets
            )
            self._exact_batch_nonmember_groups = group_idx_by_class(
                self._exact_batch_nonmember_dataset, self.num_classes
            )

        image_parts = []
        label_parts = []
        membership_parts = []
        client_parts = []
        local_index_parts = []
        candidate_sampling_by_client = {}
        selection_by_client = {}
        overlap_by_client = {}
        eligible_client_ids = []

        for client_id in self.audit_client_ids:
            target = users[client_id]
            try:
                (
                    member_images,
                    member_labels,
                    nonmember_images,
                    nonmember_labels,
                    sampling,
                ) = self._collect_global_proportional_candidates(
                    target.train_data,
                    nonmember_datasets,
                    nonmember_source_names,
                    self.low_fpr_max_members,
                    self.low_fpr_max_nonmembers,
                    client_id,
                )
                if nonmember_labels.numel() < self.low_fpr_min_nonmembers:
                    raise ValueError(
                        "balanced_global_holdout needs at least "
                        f"{self.low_fpr_min_nonmembers} non-members, but "
                        f"only constructed {nonmember_labels.numel()}."
                    )
                if (
                    self.require_full_target_train_members
                    and member_labels.numel() != len(target.train_data)
                ):
                    raise ValueError(
                        f"Client {client_id} requested its complete training "
                        f"set as members, but candidate sampling retained "
                        f"only {member_labels.numel()}/"
                        f"{len(target.train_data)} samples."
                    )
            except ValueError as error:
                if not self.allow_partial_client_audit:
                    raise
                self.skipped_audit_clients[str(client_id)] = str(error)
                logger.warning(
                    "Skipping client %d in balanced global holdout audit: %s",
                    client_id,
                    error,
                )
                continue

            member_indices = torch.as_tensor(
                sampling.pop("member_indices"), dtype=torch.long
            )
            nonmember_indices = torch.as_tensor(
                sampling.pop("nonmember_indices"), dtype=torch.long
            )
            member_count = int(member_labels.numel())
            nonmember_count = int(nonmember_labels.numel())
            actual_ratio = nonmember_count / member_count
            if actual_ratio != self.nonmember_to_member_ratio:
                raise AssertionError(
                    "Balanced global holdout candidate ratio drifted."
                )

            images = self._cat_candidate_inputs(
                (member_images, nonmember_images)
            )
            labels = torch.cat((member_labels, nonmember_labels))
            memberships = torch.cat(
                (
                    torch.ones(member_count, dtype=torch.long),
                    torch.zeros(nonmember_count, dtype=torch.long),
                )
            )
            image_parts.append(images)
            label_parts.append(labels)
            membership_parts.append(memberships)
            client_parts.append(
                torch.full((labels.numel(),), client_id, dtype=torch.long)
            )
            local_index_parts.append(
                torch.cat(
                    (
                        member_indices,
                        torch.full((nonmember_count,), -1, dtype=torch.long),
                    )
                )
            )
            eligible_client_ids.append(client_id)
            selection_by_client[str(client_id)] = {
                "seed": self.seed,
                "member_pool_indices": member_indices,
                "nonmember_pool_indices": nonmember_indices,
                "member_pool_source_counts": sampling[
                    "member_pool_source_counts"
                ],
                "nonmember_pool_source_counts": sampling[
                    "nonmember_pool_source_counts"
                ],
                "member_label_histogram": sampling[
                    "member_label_histogram"
                ],
                "nonmember_label_histogram": sampling[
                    "nonmember_label_histogram"
                ],
                "desired_nonmember_label_histogram": sampling[
                    "desired_nonmember_label_histogram"
                ],
                "label_histograms_matched": sampling[
                    "label_histograms_matched"
                ],
                "label_matching_mode": sampling["label_matching_mode"],
                "label_total_variation_distance": sampling[
                    "label_total_variation_distance"
                ],
                "index_convention": sampling["index_convention"],
            }
            candidate_sampling_by_client[str(client_id)] = {
                **sampling,
                "mode": "balanced_global_holdout",
                "member_count": member_count,
                "nonmember_count": nonmember_count,
                "member_pool_count": len(target.train_data),
                "nonmember_pool_count": sum(
                    len(dataset) for dataset in nonmember_datasets
                ),
                "minimum_nonmembers": self.low_fpr_min_nonmembers,
                "maximum_members": self.low_fpr_max_members,
                "maximum_nonmembers": self.low_fpr_max_nonmembers,
                "selection_method": (
                    "best_effort_class_stratified_without_replacement"
                    if self.require_full_target_train_members
                    else "exact_proportional_class_stratified_without_replacement"
                ),
                "selection_artifact": "candidate_selection.pt",
                "membership_definition": "global_model_record_membership",
                "full_target_train_members_required": (
                    self.require_full_target_train_members
                ),
                "member_pool_fully_selected": (
                    member_count == len(target.train_data)
                ),
                "nonmember_training_exposure": "never_trained",
                "requested_nonmember_to_member_ratio": (
                    self.nonmember_to_member_ratio
                ),
                "actual_nonmember_to_member_ratio": actual_ratio,
                "fpr_resolution": 1.0 / nonmember_count,
            }
            member_histogram = torch.bincount(
                member_labels, minlength=self.num_classes
            )
            overlap_by_client[str(client_id)] = [
                class_id
                for class_id, count in enumerate(member_histogram.tolist())
                if count > 0
            ]
            logger.info(
                "Balanced global holdout client %d candidates selected: "
                "members=%d/%d, non-members=%d/%d, ratio=1:%d, "
                "label_tv=%.6f",
                client_id,
                member_count,
                len(target.train_data),
                nonmember_count,
                sum(len(dataset) for dataset in nonmember_datasets),
                int(self.nonmember_to_member_ratio),
                sampling["label_total_variation_distance"],
            )

        if not eligible_client_ids:
            raise ValueError(
                "Balanced global holdout audit found no client with enough "
                "train/evaluation candidates."
            )
        self.audit_client_ids = eligible_client_ids
        self.pooled_client_audit = len(eligible_client_ids) > 1
        if not self.pooled_client_audit:
            self.target_client_id = eligible_client_ids[0]
        self.low_fpr_candidate_selection = (
            {
                "scope": "pooled_clients",
                "seed": self.seed,
                "audit_client_ids": eligible_client_ids,
                "per_client": selection_by_client,
            }
            if self.pooled_client_audit
            else selection_by_client[str(eligible_client_ids[0])]
        )
        self.images = self._cat_candidate_inputs(image_parts)
        self.labels = torch.cat(label_parts)
        self.membership = torch.cat(membership_parts)
        self.candidate_client_ids = torch.cat(client_parts)
        self.candidate_local_indices = torch.cat(local_index_parts)
        self.candidate_inputs_are_features = bool(
            self.images.ndim == 2
            and self.images.shape[1]
            == int(getattr(self.model, "projection_dim", -1))
        )
        self.candidate_sampling_by_client = candidate_sampling_by_client
        self.candidate_label_support = sorted(
            set(self.labels.detach().cpu().unique().tolist())
        )
        self.null_client_candidate_label_overlap = sorted(
            {
                class_id
                for overlap in overlap_by_client.values()
                for class_id in overlap
            }
        )
        self.null_client_candidate_label_overlap_by_client = overlap_by_client
        self.nonmember_source_priority = nonmember_source_names
        self._initialize_paper_balanced_evaluation_view()

    def _initialize_paper_balanced_evaluation_view(self) -> None:
        """Select one fixed balanced view with best-effort class matching."""
        size = self.paper_balanced_evaluation_size
        if size == 0:
            return
        if len(self.audit_client_ids) != 1:
            raise ValueError(
                "Paper-balanced evaluation currently requires one audit client."
            )
        client_id = int(self.audit_client_ids[0])
        client_mask = self.candidate_client_ids == client_id
        member_positions = torch.nonzero(
            client_mask & (self.membership == 1), as_tuple=False
        ).flatten()
        nonmember_positions = torch.nonzero(
            client_mask & (self.membership == 0), as_tuple=False
        ).flatten()
        if member_positions.numel() < size:
            raise ValueError(
                "Paper-balanced evaluation requires at least "
                f"{size} selected members, but found "
                f"{member_positions.numel()}."
            )

        def select_positions(
            positions: torch.Tensor,
            quotas: list[int],
            salt: int,
        ) -> torch.Tensor:
            selected = []
            for class_id, count in enumerate(quotas):
                if count == 0:
                    continue
                class_positions = positions[
                    self.labels[positions] == class_id
                ]
                generator = torch.Generator().manual_seed(
                    self.seed
                    + 15485863 * (client_id + 1)
                    + salt
                    + 2 * class_id
                )
                order = torch.randperm(
                    class_positions.numel(), generator=generator
                )[:count]
                selected.append(class_positions[order])
            return torch.cat(selected)

        member_capacities = torch.bincount(
            self.labels[member_positions], minlength=self.num_classes
        ).tolist()
        member_quotas = self._allocate_proportional_with_capacities(
            member_capacities, member_capacities, size
        )
        selected_members = select_positions(member_positions, member_quotas, 31)
        member_histogram = torch.bincount(
            self.labels[selected_members], minlength=self.num_classes
        )
        nonmember_capacities = torch.bincount(
            self.labels[nonmember_positions], minlength=self.num_classes
        ).tolist()
        nonmember_quotas = [
            min(required, available)
            for required, available in zip(
                member_histogram.tolist(), nonmember_capacities
            )
        ]
        shortage = size - sum(nonmember_quotas)
        if shortage:
            spare_capacities = [
                available - selected
                for available, selected in zip(
                    nonmember_capacities, nonmember_quotas
                )
            ]
            redistribution = self._allocate_proportional_with_capacities(
                member_histogram.tolist(), spare_capacities, shortage
            )
            nonmember_quotas = [
                selected + extra
                for selected, extra in zip(
                    nonmember_quotas, redistribution
                )
            ]
        selected_nonmembers = select_positions(
            nonmember_positions, nonmember_quotas, 47
        )
        selected_histogram = torch.bincount(
            self.labels[selected_nonmembers], minlength=self.num_classes
        )
        label_histograms_matched = torch.equal(
            selected_histogram, member_histogram
        )
        label_tv_distance = 0.5 * sum(
            abs(
                int(member_histogram[class_id]) / size
                - int(selected_histogram[class_id]) / size
            )
            for class_id in range(self.num_classes)
        )
        self.paper_balanced_candidate_indices = torch.cat(
            (selected_members, selected_nonmembers)
        )
        self.paper_balanced_evaluation = {
            "name": f"paper_{size}_{size}",
            "client_id": client_id,
            "seed": self.seed,
            "member_count": size,
            "nonmember_count": size,
            "candidate_indices": self.paper_balanced_candidate_indices,
            "member_label_histogram": member_histogram.tolist(),
            "nonmember_label_histogram": selected_histogram.tolist(),
            "label_histograms_matched": label_histograms_matched,
            "label_total_variation_distance": label_tv_distance,
            "selection_method": (
                "fixed_best_effort_class_matched_subset_of_candidates"
            ),
            "shared_across_attacks_and_rounds": True,
        }
        if self.low_fpr_candidate_selection is None:
            raise AssertionError("Candidate selection metadata was not initialized.")
        self.low_fpr_candidate_selection["paper_balanced_evaluation"] = (
            self.paper_balanced_evaluation
        )

    def _initialize_balanced_holdout_candidates(
        self, users: list, target_client_id: int
    ) -> None:
        """Build per-client 1:1 train/test pools with identical label counts.

        Members come only from the target client's local training data.  Every
        non-member comes from that client's independently partitioned test data,
        which is never used by any federated client for training.  Exact
        per-class pairing prevents client label skew from becoming a membership
        shortcut.
        """
        supported = {
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
        unsupported = sorted(set(self.attacks) - supported)
        if self.model_type not in {
            "clip_mlp",
            "clip_adapter",
            "visual_adapter",
            "clip_lora",
            "bert_adapter",
            "bert_lora",
            "gpt2_adapter",
        }:
            raise ValueError(
                "balanced_holdout requires a supported parameter-efficient "
                "model."
            )
        if unsupported:
            raise ValueError(
                "balanced_holdout does not support: " + ", ".join(unsupported)
            )

        image_parts = []
        label_parts = []
        membership_parts = []
        client_parts = []
        local_index_parts = []
        candidate_sampling_by_client = {}
        selection_by_client = {}
        overlap_by_client = {}
        eligible_client_ids = []

        for client_id in self.audit_client_ids:
            target = users[client_id]
            member_limit = (
                self.low_fpr_max_members
                if self.low_fpr_max_members > 0
                else len(target.train_data)
            )
            nonmember_limit = (
                self.low_fpr_max_nonmembers
                if self.low_fpr_max_nonmembers > 0
                else len(target.test_data)
            )
            try:
                (
                    member_images,
                    member_labels,
                    nonmember_images,
                    nonmember_labels,
                    sampling,
                ) = self._collect_exact_paired_candidates(
                    target.train_data,
                    target.test_data,
                    member_limit,
                    nonmember_limit,
                    client_id,
                )
            except ValueError as error:
                if not self.allow_partial_client_audit:
                    raise
                self.skipped_audit_clients[str(client_id)] = str(error)
                logger.warning(
                    "Skipping client %d in balanced holdout audit: %s",
                    client_id,
                    error,
                )
                continue

            member_indices = torch.as_tensor(
                sampling.pop("member_indices"), dtype=torch.long
            )
            nonmember_indices = torch.as_tensor(
                sampling.pop("nonmember_indices"), dtype=torch.long
            )
            member_count = int(member_labels.numel())
            nonmember_count = int(nonmember_labels.numel())
            if member_count != nonmember_count:
                raise AssertionError(
                    "Balanced holdout membership groups must have equal size."
                )

            images = self._cat_candidate_inputs(
                (member_images, nonmember_images)
            )
            labels = torch.cat((member_labels, nonmember_labels))
            memberships = torch.cat(
                (
                    torch.ones(member_count, dtype=torch.long),
                    torch.zeros(nonmember_count, dtype=torch.long),
                )
            )
            image_parts.append(images)
            label_parts.append(labels)
            membership_parts.append(memberships)
            client_parts.append(
                torch.full((labels.numel(),), client_id, dtype=torch.long)
            )
            local_index_parts.append(
                torch.cat(
                    (
                        member_indices,
                        torch.full((nonmember_count,), -1, dtype=torch.long),
                    )
                )
            )
            eligible_client_ids.append(client_id)
            paired_histogram = torch.bincount(
                member_labels, minlength=self.num_classes
            ).tolist()
            selection_by_client[str(client_id)] = {
                "seed": self.seed,
                "member_pool_indices": member_indices,
                "nonmember_pool_indices": nonmember_indices,
                "member_pool_source_counts": {
                    f"target_train:{client_id}": len(target.train_data)
                },
                "nonmember_pool_source_counts": {
                    f"target_independent_test:{client_id}": len(target.test_data)
                },
                "index_convention": (
                    "zero-based local positions in the target client's train "
                    "or independent test dataset"
                ),
            }
            candidate_sampling_by_client[str(client_id)] = {
                **sampling,
                "mode": "balanced_holdout",
                "member_count": member_count,
                "nonmember_count": nonmember_count,
                "member_pool_count": len(target.train_data),
                "nonmember_pool_count": len(target.test_data),
                "member_source_counts": {
                    f"target_train:{client_id}": member_count
                },
                "nonmember_source_counts": {
                    f"target_independent_test:{client_id}": nonmember_count
                },
                "maximum_members": self.low_fpr_max_members,
                "maximum_nonmembers": self.low_fpr_max_nonmembers,
                "selection_method": "exact_per_class_paired_without_replacement",
                "selection_artifact": "candidate_selection.pt",
                "membership_definition": "global_model_record_membership",
                "nonmember_training_exposure": "never_trained",
                "actual_nonmember_to_member_ratio": 1.0,
                "paired_label_histogram": paired_histogram,
                "fpr_resolution": 1.0 / nonmember_count,
            }
            overlap_by_client[str(client_id)] = [
                class_id
                for class_id, count in enumerate(paired_histogram)
                if count > 0
            ]
            logger.info(
                "Balanced holdout client %d candidates selected: "
                "members=%d/%d, non-members=%d/%d, ratio=1:1",
                client_id,
                member_count,
                len(target.train_data),
                nonmember_count,
                len(target.test_data),
            )

        if not eligible_client_ids:
            raise ValueError(
                "Balanced holdout audit found no client with at least two "
                "same-label train/test pairs."
            )
        self.audit_client_ids = eligible_client_ids
        self.pooled_client_audit = len(eligible_client_ids) > 1
        if not self.pooled_client_audit:
            self.target_client_id = eligible_client_ids[0]
        self.low_fpr_candidate_selection = (
            {
                "scope": "pooled_clients",
                "seed": self.seed,
                "audit_client_ids": eligible_client_ids,
                "per_client": selection_by_client,
            }
            if self.pooled_client_audit
            else selection_by_client[str(eligible_client_ids[0])]
        )
        self.images = self._cat_candidate_inputs(image_parts)
        self.labels = torch.cat(label_parts)
        self.membership = torch.cat(membership_parts)
        self.candidate_client_ids = torch.cat(client_parts)
        self.candidate_local_indices = torch.cat(local_index_parts)
        self.candidate_inputs_are_features = bool(
            self.images.ndim == 2
            and self.images.shape[1]
            == int(getattr(self.model, "projection_dim", -1))
        )
        self.candidate_sampling_by_client = candidate_sampling_by_client
        self.candidate_label_support = sorted(
            set(self.labels.detach().cpu().unique().tolist())
        )
        self.null_client_candidate_label_overlap = sorted(
            {
                class_id
                for overlap in overlap_by_client.values()
                for class_id in overlap
            }
        )
        self.null_client_candidate_label_overlap_by_client = overlap_by_client
        self.nonmember_source_priority = ["target_client_independent_test"]

    @staticmethod
    def _unused_few_shot_training_pool(users) -> Subset | None:
        """Recover examples excluded before the federated few-shot split.

        Federated user datasets are Subsets of one shared few-shot Subset. Its
        parent contains same-domain training examples that were never assigned
        to any client, making them valid non-member audit candidates.
        """
        if not users:
            return None
        outer_datasets = [user.train_data for user in users]
        if not all(isinstance(dataset, Subset) for dataset in outer_datasets):
            return None
        few_shot_dataset = outer_datasets[0].dataset
        if not all(dataset.dataset is few_shot_dataset for dataset in outer_datasets):
            return None
        if not isinstance(few_shot_dataset, Subset):
            return None
        parent = few_shot_dataset.dataset
        used = {int(index) for index in few_shot_dataset.indices}
        unused = [index for index in range(len(parent)) if index not in used]
        return Subset(parent, unused) if unused else None

    def _collect_many(
        self, datasets: list, limit: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_parts = []
        label_parts = []
        remaining = limit
        for dataset_index, dataset in enumerate(datasets):
            loader_generator = torch.Generator().manual_seed(
                self.seed + 104729 * dataset_index
            )
            if remaining == 0:
                break
            loader = DataLoader(
                dataset,
                batch_size=min(remaining, 64),
                shuffle=False,
                collate_fn=self.collate_fn,
                generator=loader_generator,
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
        return self._cat_candidate_inputs(image_parts), torch.cat(label_parts)

    @staticmethod
    def _cat_candidate_inputs(parts) -> torch.Tensor:
        """Concatenate images/features or dynamically padded text batches."""
        tensors = list(parts)
        if not tensors:
            raise ValueError("Cannot concatenate an empty candidate pool.")
        if all(tensor.ndim == 3 and tensor.shape[1] == 2 for tensor in tensors):
            maximum_length = max(int(tensor.shape[-1]) for tensor in tensors)
            tensors = [
                F.pad(tensor, (0, maximum_length - int(tensor.shape[-1])))
                if tensor.shape[-1] != maximum_length
                else tensor
                for tensor in tensors
            ]
        return torch.cat(tensors)

    def _exact_batch_membership_definition(self) -> str:
        if getattr(self, "federated_method", "fedsgd") == "fedsgd":
            return "current_round_exact_upload_batch"
        return "current_round_last_local_training_batch"

    def _cofedmid_metadata(self) -> dict | None:
        if getattr(self, "defense_name", "none") != "cofedmid":
            return None
        from privacy_defenses.cofedmid import DEFAULTS, coalition_ids
        cfg = {**DEFAULTS, **self.defense_config}
        clients = coalition_ids(cfg, len(self.users))
        protected = self.target_client_id in clients
        return {
            "coalition_clients": clients,
            "target_defended": protected,
            "exact_batch": self.federated_method == "fedsgd",
            "upload_perturbed": (
                protected and cfg["cofedmid_perturbation"]
                and cfg["cofedmid_noise_std"] > 0 and cfg["cofedmid_perturb_ratio"] > 0
            ),
            "custom_training_loss": protected and cfg["cofedmid_compensation"],
            "candidate_gradient_loss": "original_attack_objective_without_recycled_regularizer",
            "training_exposure_file": "../cofedmid_sample_exposure.pt",
        }

    def _build_exact_batch_candidates(self, round_index: int) -> dict:
        """Pair the target client's real upload batch with matched holdouts."""
        target = self.users[self.target_client_id]
        if target.last_train_batch is None:
            raise ValueError(
                "Exact-batch membership requires the target client's retained "
                "local-update batch."
            )
        if (
            self._exact_batch_nonmember_dataset is None
            or self._exact_batch_nonmember_groups is None
        ):
            raise ValueError(
                "Exact-batch membership requires the never-trained global "
                "evaluation pool from candidate_sampling=balanced_global_holdout."
            )
        member_inputs, member_labels = target.last_train_batch
        member_inputs = member_inputs.detach().cpu()
        member_labels = member_labels.detach().cpu().long().reshape(-1)
        if member_labels.numel() < 1:
            raise ValueError(
                "Exact-batch membership needs a non-empty training batch."
            )
        member_histogram = torch.bincount(
            member_labels, minlength=self.num_classes
        )
        ratio = self.exact_batch_nonmember_ratio
        sampling_seed = (
            self.seed
            + 1000003 * (int(round_index) + 1)
            + 104729 * (int(self.target_client_id) + 1)
        )
        selected_nonmembers = []
        for class_id, member_count in enumerate(member_histogram.tolist()):
            if member_count == 0:
                continue
            candidates = self._exact_batch_nonmember_groups[class_id]
            required = int(member_count) * ratio
            if len(candidates) < required:
                raise ValueError(
                    "Exact-batch label matching lacks never-trained "
                    f"nonmembers for class {class_id}: required={required}, "
                    f"available={len(candidates)}."
                )
            generator = torch.Generator().manual_seed(
                sampling_seed + 2 * class_id
            )
            order = torch.randperm(len(candidates), generator=generator)
            selected_nonmembers.extend(
                candidates[position] for position in order[:required].tolist()
            )
        shuffle_generator = torch.Generator().manual_seed(
            sampling_seed + 999983
        )
        order = torch.randperm(
            len(selected_nonmembers), generator=shuffle_generator
        )
        nonmember_indices = torch.tensor(
            [selected_nonmembers[position] for position in order.tolist()],
            dtype=torch.long,
        )
        nonmember_inputs, nonmember_labels = self._collect_many(
            [Subset(self._exact_batch_nonmember_dataset, nonmember_indices)],
            int(nonmember_indices.numel()),
        )
        nonmember_histogram = torch.bincount(
            nonmember_labels, minlength=self.num_classes
        )
        if not torch.equal(
            nonmember_histogram, member_histogram * ratio
        ):
            raise AssertionError(
                "Exact-batch nonmember label matching drifted."
            )
        inputs = self._cat_candidate_inputs(
            (member_inputs, nonmember_inputs)
        )
        labels = torch.cat((member_labels, nonmember_labels))
        membership = torch.cat(
            (
                torch.ones(member_labels.numel(), dtype=torch.long),
                torch.zeros(nonmember_labels.numel(), dtype=torch.long),
            )
        )
        local_indices = getattr(target, "last_train_indices", None)
        if local_indices is None:
            local_indices = torch.full(
                (member_labels.numel(),), -1, dtype=torch.long
            )
        else:
            local_indices = local_indices.detach().cpu().long().reshape(-1)
        if local_indices.numel() != member_labels.numel():
            raise ValueError(
                "Retained target-client batch indices do not match the batch size."
            )
        recycled = getattr(target, "last_train_recycled", None)
        if recycled is None:
            recycled = torch.zeros(member_labels.numel(), dtype=torch.bool)
        return {
            "inputs": inputs,
            "member_inputs": member_inputs,
            "nonmember_inputs": nonmember_inputs,
            "labels": labels,
            "membership": membership,
            "member_local_indices": local_indices,
            "member_recycled": recycled,
            "nonmember_pool_indices": nonmember_indices,
            "selection": {
                "communication_round": int(round_index) + 1,
                "target_client_id": int(self.target_client_id),
                "membership_definition": (
                    self._exact_batch_membership_definition()
                ),
                "nonmember_training_exposure": "never_trained",
                "label_matching_mode": "exact_batch_histogram_ratio",
                "sampling_seed": int(sampling_seed),
                "nonmember_to_member_ratio": int(ratio),
                "member_count": int(member_labels.numel()),
                "nonmember_count": int(nonmember_labels.numel()),
                "member_label_histogram": member_histogram.tolist(),
                "nonmember_label_histogram": nonmember_histogram.tolist(),
                "member_local_indices": local_indices,
                "member_recycled": recycled,
                "nonmember_pool_indices": nonmember_indices,
                "nonmember_index_convention": (
                    "zero-based positions in the client-id ordered "
                    "concatenation of all independent evaluation datasets"
                ),
            },
        }

    def _score_exact_batch_projres(
        self,
        *,
        round_index: int,
        member_inputs: torch.Tensor,
        nonmember_inputs: torch.Tensor,
        labels: torch.Tensor,
        membership: torch.Tensor,
        member_local_indices: torch.Tensor,
        nonmember_pool_indices: torch.Tensor,
        base_state: dict[str, torch.Tensor],
        updated_state: dict[str, torch.Tensor],
        protocol_message: dict | None,
        learning_rate: float | None,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Run ProjRes on the shared exact-batch candidate view."""
        member_count = int((membership == 1).sum())
        nonmember_count = int((membership == 0).sum())
        if member_count + nonmember_count != labels.numel():
            raise ValueError("ProjRes candidates and membership are misaligned.")
        parameter_name, attacked_layer = self.model.get_projres_attack_surface(
            self.exact_batch_projres_config.get("attacked_parameter")
        )
        token_reduction = str(
            self.exact_batch_projres_config.get("token_reduction", "auto")
        ).lower()
        if token_reduction == "auto":
            resolver = getattr(
                self.model, "resolve_projres_token_reduction", None
            )
            if callable(resolver):
                token_reduction = resolver(token_reduction)
            elif self.model_type in {
                "clip_mlp",
                "clip_adapter",
                "visual_adapter",
            }:
                token_reduction = "none"
            else:
                token_reduction = (
                    "cls"
                    if str(getattr(self.model, "architecture", "")) == "bert"
                    else "last"
                )
        self.model.load_state_dict(base_state, strict=False)
        extractor = self.model.get_projres_representations
        member_representations, hidden_vector_count = extractor(
            member_inputs,
            parameter_name=parameter_name,
            token_reduction=token_reduction,
        )
        representation_parts = [member_representations.detach().cpu().float()]
        for start in range(0, nonmember_count, self.audit_batch_size):
            stop = start + self.audit_batch_size
            representations, _ = extractor(
                nonmember_inputs[start:stop],
                parameter_name=parameter_name,
                token_reduction=token_reduction,
            )
            representation_parts.append(
                representations.detach().cpu().float()
            )
        candidate_representations = torch.cat(representation_parts)
        if parameter_name not in base_state or parameter_name not in updated_state:
            raise ValueError(
                f"Observed target update does not contain {parameter_name}."
            )
        if getattr(self, "federated_method", "fedsgd") == "fedsgd":
            if protocol_message is None or protocol_message.get("kind") != "gradient":
                raise ValueError(
                    "FedSGD ProjRes requires the target client's uploaded gradient."
                )
            tensors = protocol_message.get("tensors", {})
            if parameter_name not in tensors:
                raise ValueError(
                    f"FedSGD gradient does not contain {parameter_name}."
                )
            observed_update = tensors[parameter_name].detach().cpu().float()
            update_source = "uploaded_client_gradient"
        else:
            observed_update = (
                base_state[parameter_name].detach().cpu().float()
                - updated_state[parameter_name].detach().cpu().float()
            )
            update_source = "base_minus_client_post_state"
        cofedmid = self._cofedmid_metadata()
        parameter_perturbed = getattr(self, "defense_name", "none") == "www"
        if cofedmid and cofedmid["upload_perturbed"]:
            # The unperturbed batch-rank bound need not hold after upload noise.
            # Reconstruct the public mask from the shared trainable parameter order.
            widths = [
                (name, parameter.numel())
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            total_width = sum(width for _, width in widths)
            tail_start = total_width - math.floor(
                total_width * self.defense_config.get("cofedmid_perturb_ratio", 0.2)
            )
            offset = 0
            for name, width in widths:
                if name == parameter_name:
                    parameter_perturbed = offset + width > tail_start
                    break
                offset += width
        rank_bound = None if parameter_perturbed else int(hidden_vector_count)
        attack = strict_mlp_projres(
            observed_update,
            candidate_representations,
            threshold=None,
            max_rank=rank_bound,
        )
        attack_result = AttackResult(
            name="projres",
            scores=attack.scores.detach().cpu(),
            labels=membership.detach().cpu(),
            sample_indices=torch.arange(labels.numel()),
        )
        summary = attack_result.to_summary(
            fpr_targets=_EXACT_BATCH_REPORTED_FPR_TARGETS
        )
        paper_fedsgd_exact = (
            getattr(self, "federated_method", "fedsgd") == "fedsgd"
            and getattr(self, "defense_name", "none") != "www"
        )
        if cofedmid and (cofedmid["upload_perturbed"] or cofedmid["custom_training_loss"]):
            paper_fedsgd_exact = False
        if self.model_type == "clip_mlp":
            sample_representation = "clip_image_feature_input_to_first_mlp_projection"
        elif self.model_type in {"clip_adapter", "visual_adapter"}:
            sample_representation = (
                "clip_image_feature_input_to_adapter_down_projection"
            )
        elif self.model_type in {"clip_lora", "bert_lora"}:
            sample_representation = (
                f"{token_reduction}_token_input_to_lora_down_projection"
            )
        else:
            sample_representation = (
                f"{token_reduction}_sample_embedding_input_to_"
                "adapter_down_projection"
            )
        metadata = {
            **attack.metadata,
            "attacked_parameter": parameter_name,
            "sample_representation": sample_representation,
            "membership_definition": self._exact_batch_membership_definition(),
            "nonmember_training_exposure": "never_trained",
            "label_matching_mode": "exact_batch_histogram_ratio",
            "nonmember_to_member_ratio": self.exact_batch_nonmember_ratio,
            "communication_round": int(round_index) + 1,
            "observed_hidden_vector_count": int(hidden_vector_count),
            "observed_update_norm": float(observed_update.norm()),
            "update_source": update_source,
            "learning_rate": (
                None if learning_rate is None else float(learning_rate)
            ),
            "paper_fedsgd_exact": paper_fedsgd_exact,
            "cofedmid": cofedmid,
            "attacked_parameter_perturbed": parameter_perturbed,
            "batch_rank_bound": rank_bound,
            "reported_fpr_targets": list(
                _EXACT_BATCH_REPORTED_FPR_TARGETS
            ),
        }
        batch_positions = list(range(member_count))
        result_payload = {
            "client_id": int(self.target_client_id),
            "model_type": self.model_type,
            "threat_model": {
                "communication_round": int(round_index) + 1,
                "member_definition": (
                    "present_in_the_observed_target_client_fedsgd_batch"
                ),
                "execution": "unified_exact_batch_auditor",
                "paper_fedsgd_exact": paper_fedsgd_exact,
            },
            "dimensions": {
                "candidate_sampling_batch_size": member_count,
                "member_candidate_count": member_count,
                "nonmember_candidate_count": nonmember_count,
                "input_dimension": int(attacked_layer.in_features),
                "first_layer_output_dimension": int(attacked_layer.out_features),
                "observed_hidden_vector_count": int(hidden_vector_count),
            },
            "optimization": {
                "learning_rate": (
                    None if learning_rate is None else float(learning_rate)
                ),
                "observed_update_norm": float(observed_update.norm()),
                "update_source": update_source,
            },
            "attack": {"metrics": summary, "metadata": metadata},
            "raw": {
                "labels": membership.tolist(),
                "scores": attack.scores.detach().cpu().tolist(),
                "l1_residuals": attack.l1_residuals.detach().cpu().tolist(),
                "predictions": None,
            },
            "candidate_controls": {
                "label_matched_nonmembers": True,
                "label_matching_mode": "exact_batch_histogram_ratio",
                "member_labels": labels[:member_count].tolist(),
                "member_batch_positions": batch_positions,
                "member_local_indices": member_local_indices.tolist(),
                "nonmember_labels": labels[member_count:].tolist(),
                "nonmember_pool_indices": nonmember_pool_indices.tolist(),
                "nonmember_training_exposure": "never_trained",
            },
        }
        payload = {
            "communication_round": int(round_index) + 1,
            "result": result_payload,
        }
        diagnostics = {
            "l1_residuals": attack.l1_residuals.detach().cpu(),
            "predictions": None,
            "metadata": metadata,
        }
        return attack.scores.detach().cpu(), diagnostics, payload

    def _collect_source_balanced(
        self,
        datasets: list,
        source_names: list[str],
        limit: int,
        client_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """Collect a deterministic, source-balanced FedMIA non-member pool.

        The reference FedMIA CIFAR-100 evaluation uses one independent test
        source and one training source for every non-target client. With ten
        clients it samples the same number from each of those ten sources.
        This helper preserves that construction for arbitrary client counts
        and redistributes a short source's quota without duplicating samples.
        """
        if len(datasets) != len(source_names) or not datasets:
            raise ValueError("FedMIA mix sampling requires named data sources.")
        capacities = [len(dataset) for dataset in datasets]
        if sum(capacities) < limit:
            raise ValueError(
                "FedMIA mix sampling cannot reach the requested non-member "
                f"count {limit}; only {sum(capacities)} candidates are available."
            )

        counts = [0] * len(datasets)
        remaining = int(limit)
        while remaining:
            allocated = False
            for source_id, capacity in enumerate(capacities):
                if counts[source_id] >= capacity:
                    continue
                counts[source_id] += 1
                remaining -= 1
                allocated = True
                if remaining == 0:
                    break
            if not allocated:
                raise AssertionError("FedMIA source-balanced allocation stalled.")

        selected = []
        for source_id, (dataset, count) in enumerate(zip(datasets, counts)):
            if count == 0:
                continue
            generator = torch.Generator().manual_seed(
                self.seed
                + 999983 * (client_id + 1)
                + 104729 * (source_id + 1)
            )
            indices = torch.randperm(len(dataset), generator=generator)[:count]
            selected.append(Subset(dataset, indices.tolist()))
        images, labels = self._collect_many(selected, limit)
        return images, labels, {
            name: int(count) for name, count in zip(source_names, counts)
        }

    def _collect_fedmia_mix_candidates(
        self,
        users: list,
        client_id: int,
        member_limit: int,
        nonmember_limit: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """Reproduce FedMIA's target-train versus mixed non-member protocol."""
        target = users[client_id]
        member_images, member_labels = self._collect_many(
            [target.train_data], member_limit
        )
        requested_nonmembers = int(
            math.ceil(member_labels.numel() * self.nonmember_to_member_ratio)
        )
        if requested_nonmembers > nonmember_limit:
            raise ValueError(
                "FedMIA mix sampling needs max_nonmember_samples >= "
                "ceil(actual_members * nonmember_to_member_ratio); requested "
                f"{requested_nonmembers}, configured {nonmember_limit}."
            )

        test_datasets = [user.test_data for user in users if len(user.test_data)]
        if not test_datasets:
            raise ValueError("FedMIA mix sampling requires an independent test pool.")
        independent_test = ConcatDataset(test_datasets)
        nonmember_datasets = [independent_test]
        source_names = ["independent_test"]
        for user in users:
            if user.id == client_id:
                continue
            nonmember_datasets.append(user.train_data)
            source_names.append(f"other_client_train:{user.id}")
        nonmember_images, nonmember_labels, source_counts = (
            self._collect_source_balanced(
                nonmember_datasets,
                source_names,
                requested_nonmembers,
                client_id,
            )
        )
        return (
            member_images,
            member_labels,
            nonmember_images,
            nonmember_labels,
            {
                "client_id": int(client_id),
                "mode": "fedmia_mix",
                "member_count": int(member_labels.numel()),
                "nonmember_count": int(nonmember_labels.numel()),
                "requested_nonmember_to_member_ratio": float(
                    self.nonmember_to_member_ratio
                ),
                "actual_nonmember_to_member_ratio": float(
                    nonmember_labels.numel() / member_labels.numel()
                ),
                "nonmember_source_counts": source_counts,
                "sampling_seed": int(self.seed),
            },
        )

    def _collect_exact_paired_candidates(
        self,
        member_dataset,
        nonmember_dataset,
        member_limit: int,
        nonmember_limit: int,
        client_id: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """Jointly sample equal per-class train/test candidates for one client."""
        member_groups = group_idx_by_class(member_dataset, self.num_classes)
        nonmember_groups = group_idx_by_class(nonmember_dataset, self.num_classes)
        capacities = [
            min(len(member_groups[class_id]), len(nonmember_groups[class_id]))
            for class_id in range(self.num_classes)
        ]
        budget = min(member_limit, nonmember_limit, sum(capacities))
        if budget < 2:
            raise ValueError(
                f"Client {client_id} has fewer than two same-label train/test pairs."
            )

        active_classes = [
            class_id for class_id, capacity in enumerate(capacities) if capacity > 0
        ]
        order_generator = torch.Generator().manual_seed(
            self.seed + 65537 * (client_id + 1)
        )
        order = [
            active_classes[index]
            for index in torch.randperm(
                len(active_classes), generator=order_generator
            ).tolist()
        ]
        quotas = [0] * self.num_classes
        remaining = budget
        while remaining:
            allocated = False
            for class_id in order:
                if quotas[class_id] >= capacities[class_id]:
                    continue
                quotas[class_id] += 1
                remaining -= 1
                allocated = True
                if remaining == 0:
                    break
            if not allocated:
                raise AssertionError("Paired candidate allocation stalled.")

        def select_indices(indices: list[int], count: int, salt: int) -> list[int]:
            if count == 0:
                return []
            generator = torch.Generator().manual_seed(
                self.seed + 104729 * (client_id + 1) + salt
            )
            permutation = torch.randperm(len(indices), generator=generator)[:count]
            return [indices[position] for position in permutation.tolist()]

        member_indices = []
        nonmember_indices = []
        for class_id, count in enumerate(quotas):
            member_indices.extend(
                select_indices(member_groups[class_id], count, 2 * class_id + 1)
            )
            nonmember_indices.extend(
                select_indices(
                    nonmember_groups[class_id], count, 2 * class_id + 2
                )
            )

        def shuffle_indices(indices: list[int], salt: int) -> list[int]:
            generator = torch.Generator().manual_seed(
                self.seed + 999983 * (client_id + 1) + salt
            )
            permutation = torch.randperm(len(indices), generator=generator)
            return [indices[position] for position in permutation.tolist()]

        member_subset = Subset(member_dataset, shuffle_indices(member_indices, 11))
        nonmember_subset = Subset(
            nonmember_dataset, shuffle_indices(nonmember_indices, 17)
        )
        member_images, member_labels = self._collect_many(
            [member_subset], budget
        )
        nonmember_images, nonmember_labels = self._collect_many(
            [nonmember_subset], budget
        )
        member_histogram = torch.bincount(
            member_labels, minlength=self.num_classes
        )
        nonmember_histogram = torch.bincount(
            nonmember_labels, minlength=self.num_classes
        )
        if not torch.equal(member_histogram, nonmember_histogram):
            raise AssertionError("Paired candidate label histograms drifted.")
        if member_histogram.tolist() != quotas:
            raise AssertionError("Paired candidate allocation did not match quotas.")
        sampling = {
            "client_id": int(client_id),
            "member_count": int(budget),
            "nonmember_count": int(budget),
            "paired_label_histogram": quotas,
            "pair_capacity_by_label": capacities,
            "excluded_member_only_labels": [
                class_id
                for class_id in range(self.num_classes)
                if member_groups[class_id] and not nonmember_groups[class_id]
            ],
            "sampling_seed": int(self.seed + 65537 * (client_id + 1)),
            "member_indices": member_indices,
            "nonmember_indices": nonmember_indices,
        }
        return (
            member_images,
            member_labels,
            nonmember_images,
            nonmember_labels,
            sampling,
        )

    def _collect_label_matched(
        self,
        datasets: list,
        target_labels: torch.Tensor,
        limit: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect non-members with the member-label distribution.

        Class identity can otherwise dominate a non-IID membership audit even
        when a method has not memorized an individual.  The non-member
        histogram is therefore an integer multiple of the member histogram;
        if that is impossible under the available data, fail instead of
        silently changing the requested label distribution.
        """
        target_labels = target_labels.detach().cpu().long().flatten()
        classes = len(getattr(self.model, "classnames", ()))
        classes = max(classes, int(target_labels.max().item()) + 1)
        member_counts = torch.bincount(target_labels, minlength=classes).long()
        grouped_by_dataset = [
            group_idx_by_class(dataset, classes) for dataset in datasets
        ]
        availability = torch.tensor(
            [
                sum(len(grouped[class_id]) for grouped in grouped_by_dataset)
                for class_id in range(classes)
            ],
            dtype=torch.long,
        )
        positive = member_counts > 0
        if not bool(torch.all(availability[positive] >= member_counts[positive])):
            missing = {
                class_id: {
                    "members": int(member_counts[class_id]),
                    "available_nonmembers": int(availability[class_id]),
                }
                for class_id in torch.nonzero(positive, as_tuple=False).flatten().tolist()
                if availability[class_id] < member_counts[class_id]
            }
            raise ValueError(
                "Exact label-distribution matching is impossible; reduce member "
                f"sampling or add non-member data for classes {missing}."
            )
        requested = target_labels.numel() if limit is None else int(limit)
        maximum_by_limit = requested // target_labels.numel()
        maximum_by_availability = min(
            int(availability[class_id] // member_counts[class_id])
            for class_id in torch.nonzero(positive, as_tuple=False).flatten().tolist()
        )
        multiplier = min(maximum_by_limit, maximum_by_availability)
        if multiplier < 1:
            raise ValueError(
                "max_nonmember_samples is too small for exact label matching."
            )
        desired = member_counts * multiplier

        remaining = desired.tolist()
        selected = []
        for dataset, grouped in zip(datasets, grouped_by_dataset):
            indices = []
            for class_id, needed in enumerate(remaining):
                take = min(int(needed), len(grouped[class_id]))
                if take:
                    indices.extend(grouped[class_id][:take])
                    remaining[class_id] -= take
            if indices:
                selected.append(Subset(dataset, indices))
            if not any(remaining):
                break
        if any(remaining):
            missing = {
                class_id: count
                for class_id, count in enumerate(remaining)
                if count
            }
            raise ValueError(
                "Cannot construct label-matched non-members; missing counts "
                f"{missing}."
            )
        images, labels = self._collect_many(selected, int(desired.sum().item()))
        if not torch.equal(torch.bincount(labels, minlength=classes), desired):
            raise AssertionError("Label-matched membership collection drifted.")
        return images, labels

    def _single_round_index_for_attack(self, attack: str) -> int | None:
        if attack not in _SINGLE_ROUND_ATTACKS:
            return None
        # An explicit per-attack interval intentionally promotes the otherwise
        # single-round baseline to a periodic diagnostic.  Each stored point is
        # still evaluated with the original single-round attack definition.
        if attack in self.attack_audit_intervals:
            return None
        selector = self.config.get("fedmia_baseline_single_round", "last")
        if isinstance(selector, str):
            normalized = selector.lower()
            if normalized == "last":
                return self.total_rounds - 1 if self.total_rounds > 0 else None
            if normalized == "first":
                return 0
            try:
                selector = int(normalized)
            except ValueError as error:
                raise ValueError(
                    "FedMIA single-round baselines require 'first', 'last', or "
                    "an observed integer round."
                ) from error
        return int(selector)

    def _attack_is_due(self, attack: str, round_index: int) -> bool:
        single_round = self._single_round_index_for_attack(attack)
        if attack in _SINGLE_ROUND_ATTACKS and single_round is not None:
            return round_index == single_round
        is_final_round = (
            self.total_rounds > 0 and round_index == self.total_rounds - 1
        )
        # ``round_index`` is zero-based but an interval describes completed
        # communication rounds. For example, interval=10 observes rounds
        # 10, 20, ... at indices 9, 19, ... rather than indices 0, 10, ....
        return is_final_round or (
            (round_index + 1) % self._audit_interval_for_attack(attack) == 0
        )

    def _attacks_for_round(self, round_index: int) -> list[str]:
        return [
            attack
            for attack in self.attacks
            if self._attack_is_due(attack, round_index)
        ]

    def should_observe(self, round_index: int) -> bool:
        return self.enabled and bool(self._attacks_for_round(round_index))

    def _audit_client_ids_for_attacks(
        self, active_attacks: list[str], selected_ids: list[int]
    ) -> list[int]:
        if (
            self.signal_storage != "full"
            and not self.pooled_client_audit
            and set(active_attacks).issubset(_TARGET_ONLY_SIGNAL_ATTACKS)
        ):
            return [
                user_id
                for user_id in selected_ids
                if user_id == self.target_client_id
            ]
        return list(selected_ids)

    def _audit_interval_for_attack(self, attack: str) -> int:
        return self.attack_audit_intervals.get(attack, self.audit_interval)

    def _attack_schedule_metadata(self, attack: str) -> dict:
        single_round = self._single_round_index_for_attack(attack)
        if attack in _SINGLE_ROUND_ATTACKS and single_round is not None:
            return {"mode": "single_round", "round": single_round}
        return {
            "mode": "interval",
            "interval": self._audit_interval_for_attack(attack),
            "interval_basis": "completed_rounds",
            "include_final_round": self.total_rounds > 0,
        }

    def _observations_for_attack(self, attack: str) -> list[dict]:
        if attack in self.exact_batch_membership_attacks:
            return [
                observation
                for observation in self.exact_batch_observations
                if attack in observation["attacks"]
                and self._attack_is_due(attack, int(observation["round"]))
            ]
        return [
            observation
            for observation in self.observations
            if self._attack_is_due(attack, int(observation["round"]))
        ]

    @torch.no_grad()
    def _candidate_outputs(
        self,
        model: torch.nn.Module,
        require_representation: bool = True,
        candidate_inputs: torch.Tensor | None = None,
        candidate_labels: torch.Tensor | None = None,
        candidate_inputs_are_features: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Evaluate large audit candidate pools without one oversized CLIP batch."""
        inputs = self.images if candidate_inputs is None else candidate_inputs
        labels_cpu = self.labels if candidate_labels is None else candidate_labels
        inputs_are_features = (
            getattr(self, "candidate_inputs_are_features", False)
            if candidate_inputs_are_features is None
            else bool(candidate_inputs_are_features)
        )
        if inputs.shape[0] != labels_cpu.numel():
            raise ValueError("Candidate inputs and labels must have equal length.")
        logits_parts = []
        representation_parts = []
        loss_parts = []
        feature_forward = getattr(model, "forward_from_image_features", None)
        feature_representation = getattr(
            model, "get_audit_representation_from_image_features", None
        )
        for start in range(0, labels_cpu.numel(), self.audit_batch_size):
            stop = start + self.audit_batch_size
            images = inputs[start:stop].to(self.device)
            labels = labels_cpu[start:stop].to(self.device)
            if inputs_are_features:
                if feature_forward is None:
                    raise TypeError(
                        "Cached audit features require forward_from_image_features."
                    )
                if require_representation:
                    if feature_representation is None:
                        raise TypeError(
                            "Cached audit features require an MLP representation getter."
                        )
                    logits, representation = feature_representation(images, labels)
                    losses = F.cross_entropy(logits, labels, reduction="none")
                    compact = F.adaptive_avg_pool1d(
                        representation.unsqueeze(1),
                        min(64, representation.shape[1]),
                    ).squeeze(1)
                    representation_parts.append(compact.detach().cpu())
                else:
                    logits = feature_forward(images)
                    losses = F.cross_entropy(logits, labels, reduction="none")
            elif require_representation:
                logits, representation, losses = logits_and_representation(
                    model, images, labels
                )
                representation_parts.append(representation)
            else:
                logits = model(images)
                losses = F.cross_entropy(
                    logits, labels, reduction="none"
                )
            logits_parts.append(logits.detach().cpu())
            loss_parts.append(losses.detach().cpu())
        return (
            torch.cat(logits_parts),
            torch.cat(representation_parts) if representation_parts else None,
            torch.cat(loss_parts),
        )

    def _candidate_gradients(
        self,
        model: torch.nn.Module,
        parameter_names: list[str],
        gradient_loss: str = "true_label",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample gradients while bounding resident GPU images."""
        gradient_parts = []
        signature_parts = []
        loss_parts = []
        feature_forward = (
            getattr(model, "forward_from_image_features", None)
            if getattr(self, "candidate_inputs_are_features", False)
            else None
        )
        if (
            getattr(self, "candidate_inputs_are_features", False)
            and feature_forward is None
        ):
            raise TypeError(
                "Cached audit features require forward_from_image_features."
            )
        for start in range(0, self.labels.numel(), self.audit_batch_size):
            stop = start + self.audit_batch_size
            gradients, signatures, losses = per_sample_prompt_gradients(
                model,
                self.images[start:stop].to(self.device),
                self.labels[start:stop].to(self.device),
                parameter_names,
                forward=feature_forward,
                gradient_loss=gradient_loss,
            )
            gradient_parts.append(gradients)
            signature_parts.append(signatures)
            loss_parts.append(losses)
        return (
            torch.cat(gradient_parts),
            torch.cat(signature_parts),
            torch.cat(loss_parts),
        )

    def _cached_feature_gradient_cosines(
        self,
        model: torch.nn.Module,
        parameter_names: list[str],
        updates: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute exact MLP gradient cosines without retaining all gradients."""
        if not updates:
            raise ValueError("At least one client update is required.")
        feature_forward = getattr(model, "forward_from_image_features", None)
        if feature_forward is None:
            raise TypeError("Cached features require forward_from_image_features.")
        update_matrix = torch.stack([update.detach().cpu() for update in updates])
        update_norms = update_matrix.norm(dim=1).clamp_min(1e-12)
        score_parts = []
        gradient_batch_size = int(
            self.config.get("low_fpr_gradient_batch_size", 8)
        )
        if gradient_batch_size <= 0:
            raise ValueError("low_fpr_gradient_batch_size must be positive.")
        for start in range(0, self.labels.numel(), gradient_batch_size):
            stop = start + gradient_batch_size
            gradients, _, _ = per_sample_prompt_gradients(
                model,
                self.images[start:stop].to(self.device),
                self.labels[start:stop].to(self.device),
                parameter_names,
                forward=feature_forward,
            )
            compared = gradients
            compared_updates = update_matrix
            if update_matrix.shape[1] != gradients.shape[1]:
                width = min(64, update_matrix.shape[1], gradients.shape[1])
                compared_updates = F.adaptive_avg_pool1d(
                    update_matrix.unsqueeze(1), width
                ).squeeze(1)
                compared = F.adaptive_avg_pool1d(
                    gradients.unsqueeze(1), width
                ).squeeze(1)
                batch_update_norms = compared_updates.norm(dim=1).clamp_min(1e-12)
            else:
                batch_update_norms = update_norms
            gradient_norms = compared.norm(dim=1).clamp_min(1e-12)
            score_parts.append(
                (compared @ compared_updates.t())
                / (gradient_norms[:, None] * batch_update_norms[None, :])
            )
        return torch.cat(score_parts).t().contiguous()

    def _cached_feature_gradient_differences(
        self,
        model: torch.nn.Module,
        parameter_names: list[str],
        updates: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the exact Gradient-Diff statistic for cached CLIP features."""
        if not updates:
            raise ValueError("At least one client gradient is required.")
        feature_forward = getattr(model, "forward_from_image_features", None)
        if feature_forward is None:
            raise TypeError("Cached features require forward_from_image_features.")
        update_matrix = torch.stack([update.detach().cpu() for update in updates])
        score_parts = []
        gradient_batch_size = int(
            self.config.get("low_fpr_gradient_batch_size", 8)
        )
        if gradient_batch_size <= 0:
            raise ValueError("low_fpr_gradient_batch_size must be positive.")
        for start in range(0, self.labels.numel(), gradient_batch_size):
            stop = start + gradient_batch_size
            gradients, _, _ = per_sample_prompt_gradients(
                model,
                self.images[start:stop].to(self.device),
                self.labels[start:stop].to(self.device),
                parameter_names,
                forward=feature_forward,
                gradient_loss="sum_over_labels",
            )
            gradients = gradients.to(torch.float64)
            gradient_sq = gradients.square().sum(dim=1)
            dots = gradients @ update_matrix.to(torch.float64).t()
            score_parts.append(
                (2.0 * dots - gradient_sq[:, None]).to(torch.float32)
            )
        return torch.cat(score_parts).t().contiguous()

    def _raw_input_gradient_measurements(
        self,
        model: torch.nn.Module,
        parameter_names: list[str],
        updates: list[tuple[dict[str, torch.Tensor], float, float]],
        *,
        need_cosine: bool,
        need_gradient_difference: bool,
        gradient_difference_update_indices: list[int] | None = None,
        candidate_inputs: torch.Tensor | None = None,
        candidate_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stream exact text-sample gradients without retaining the full pool."""
        if not updates:
            raise ValueError("At least one client update is required.")
        allowed = set(parameter_names)
        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name in allowed
        ]
        parameters = [parameter for _, parameter in named_parameters]
        if not parameters or any(
            set(update) != allowed for update, _, _ in updates
        ):
            raise ValueError(
                "Text gradient observations and uploaded PEFT updates must "
                "have identical parameter scopes."
            )
        update_norms = []
        for update, _, scale in updates:
            norm_sq = torch.zeros((), dtype=torch.float64)
            for name in parameter_names:
                values = (
                    update[name].detach().cpu().to(torch.float64) * float(scale)
                )
                norm_sq += values.square().sum()
            update_norms.append(norm_sq.sqrt().clamp_min(1e-12))
        cosine_rows = []
        difference_rows = []
        difference_indices = (
            list(range(len(updates)))
            if gradient_difference_update_indices is None
            else list(gradient_difference_update_indices)
        )
        if need_gradient_difference and not difference_indices:
            raise ValueError("Gradient-Diff requires a target client update.")
        inputs = self.images if candidate_inputs is None else candidate_inputs
        labels = self.labels if candidate_labels is None else candidate_labels
        if inputs.shape[0] != labels.numel():
            raise ValueError("Candidate inputs and labels must have equal length.")
        model.eval()
        for candidate_input, candidate_label in zip(inputs, labels):
            model.zero_grad(set_to_none=True)
            logits = model(candidate_input.unsqueeze(0).to(self.device))
            true_label_loss = F.cross_entropy(
                logits, candidate_label.view(1).to(self.device)
            )
            losses = []
            if need_cosine:
                losses.append(("cosine", true_label_loss))
            if need_gradient_difference:
                sum_label_loss = (
                    logits.shape[1] * torch.logsumexp(logits, dim=1)
                    - logits.sum(dim=1)
                ).sum()
                losses.append(("gradient_difference", sum_label_loss))
            for position, (measurement, differentiated_loss) in enumerate(losses):
                gradients = torch.autograd.grad(
                    differentiated_loss,
                    parameters,
                    retain_graph=position + 1 < len(losses),
                )
                gradient_norm_sq = torch.zeros((), dtype=torch.float64)
                measurement_indices = (
                    list(range(len(updates)))
                    if measurement == "cosine"
                    else difference_indices
                )
                dots = [
                    torch.zeros((), dtype=torch.float64)
                    for _ in measurement_indices
                ]
                for (name, _), gradient in zip(named_parameters, gradients):
                    flat = gradient.detach().flatten().cpu().to(torch.float64)
                    gradient_norm_sq += flat.square().sum()
                    for index, update_index in enumerate(measurement_indices):
                        update, sign, scale = updates[update_index]
                        dots[index] += (
                            flat
                            * update[name]
                            .detach()
                            .flatten()
                            .cpu()
                            .to(torch.float64)
                        ).sum() * float(sign) * float(scale)
                    del flat
                if measurement == "cosine":
                    gradient_norm = gradient_norm_sq.sqrt().clamp_min(1e-12)
                    cosine_rows.append(
                        torch.stack(
                            [
                                dot / (gradient_norm * norm)
                                for dot, norm in zip(dots, update_norms)
                            ]
                        ).to(torch.float32)
                    )
                else:
                    difference_rows.append(
                        torch.stack(
                            [2.0 * dot - gradient_norm_sq for dot in dots]
                        ).to(torch.float32)
                    )
                del gradients
        measurements = {}
        if need_cosine:
            measurements["cosine"] = torch.stack(cosine_rows).t().contiguous()
        if need_gradient_difference:
            measurements["gradient_difference"] = (
                torch.stack(difference_rows).t().contiguous()
            )
        return measurements

    def _raw_input_gradient_cosines(
        self,
        model: torch.nn.Module,
        parameter_names: list[str],
        updates: list[tuple[dict[str, torch.Tensor], float]],
    ) -> torch.Tensor:
        """Backward-compatible wrapper for exact streamed cosine signals."""
        scaled_updates = [(update, sign, 1.0) for update, sign in updates]
        return self._raw_input_gradient_measurements(
            model,
            parameter_names,
            scaled_updates,
            need_cosine=True,
            need_gradient_difference=False,
        )["cosine"]

    @staticmethod
    def _clone_prompt_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in state.items()}

    def _observable_base_state(
        self, base_state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return base_state

    def _observable_client_state(
        self,
        user_id: int,
        updated_state: dict[str, torch.Tensor],
        released_states: dict[int, dict[str, torch.Tensor]] | None,
        *,
        base_state: dict[str, torch.Tensor] | None = None,
        protocol_message: dict | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.audit_view == "full_whitebox":
            return updated_state
        if (
            self.audit_view == "protocol_plus_released_prompts"
            and self.federated_method == "fedsgd"
        ):
            if base_state is None or protocol_message is None:
                raise ValueError(
                    "FedSGD client post-step auditing requires the observable "
                    "base state and target-client protocol message."
                )
            if protocol_message.get("kind") != "gradient":
                raise ValueError(
                    "FedSGD client post-step auditing requires a gradient "
                    "protocol message."
                )
            tensors = protocol_message.get("tensors", {})
            if set(tensors) != set(base_state):
                raise ValueError(
                    "FedSGD gradient tensors must exactly match the "
                    "observable trainable base-state parameters."
                )
            if learning_rate is None or float(learning_rate) <= 0:
                raise ValueError(
                    "FedSGD client post-step reconstruction requires the "
                    "current positive learning rate."
                )
            # Reconstruct only what the server can infer from the uploaded
            # gradient. Do not read the simulator's raw client state.
            return {
                name: base_tensor.detach()
                - float(learning_rate) * tensors[name].detach().to(
                    device=base_tensor.device,
                    dtype=base_tensor.dtype,
                )
                for name, base_tensor in base_state.items()
            }
        if (
            self.audit_view == "protocol_plus_released_prompts"
            and self.federated_method in {"fedavg", "promptfl"}
        ):
            # The full FedAvg model update is itself a protocol message.
            return updated_state
        if not released_states:
            raise ValueError("Released prompt state is required by this audit view.")
        released = released_states.get(user_id, released_states.get(0))
        if released is None:
            raise ValueError(f"No released prompt is available for client {user_id}.")
        return released

    def _attach_www_candidate_scores(
        self,
        observation: dict,
        *,
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        base_states: dict[int, dict[str, torch.Tensor]] | None,
        protocol_messages: dict[int, dict] | None,
        released_states: dict[int, dict[str, torch.Tensor]] | None,
        aggregation_weights: dict[int, float] | None,
        learning_rate: float | None,
    ) -> None:
        """Attach sample-aligned observational WWW scores to one audit round.

        For every fixed FedMIA candidate this computes
        ``L(x; theta_-k) - L(x; theta_k)``. ``theta_k`` is the target client's
        observable post-local-epoch upload and ``theta_-k`` is reconstructed
        from the released FedAvg state after removing that client's configured
        aggregation contribution. No sample is filtered and training is not
        changed.
        """
        if not self.www_candidate_scoring:
            return
        client_ids = observation["client_ids"].tolist()
        if self.target_client_id not in client_ids:
            raise ValueError(
                "WWW candidate scoring requires the target client in the "
                "observed FedAvg round."
            )
        if "confidence" not in observation:
            raise ValueError(
                "WWW candidate scoring requires per-client candidate losses."
            )
        if not released_states or 0 not in released_states:
            raise ValueError(
                "WWW candidate scoring requires the released aggregated state."
            )
        if (
            not aggregation_weights
            or self.target_client_id not in aggregation_weights
        ):
            raise ValueError(
                "WWW candidate scoring requires the target aggregation weight."
            )

        target_position = client_ids.index(self.target_client_id)
        target_base = (
            base_state
            if base_states is None
            else base_states[self.target_client_id]
        )
        target_message = (
            None
            if protocol_messages is None
            else protocol_messages.get(self.target_client_id)
        )
        own_state = self._observable_client_state(
            self.target_client_id,
            updated_states[self.target_client_id],
            released_states,
            base_state=target_base,
            protocol_message=target_message,
            learning_rate=learning_rate,
        )
        own_weight = float(aggregation_weights[self.target_client_id])
        other_state = infer_other_clients_state(
            global_state=released_states[0],
            own_state=own_state,
            own_weight=own_weight,
        )
        own_losses = -observation["confidence"][target_position]
        self.model.load_state_dict(other_state, strict=False)
        _, _, other_losses = self._candidate_outputs(
            self.model,
            require_representation=False,
        )
        scores = other_losses - own_losses
        if scores.numel() != self.labels.numel():
            raise AssertionError("WWW candidate scores lost sample alignment.")
        observation["www_candidate_own_loss"] = own_losses.detach().cpu()
        observation["www_candidate_other_loss"] = other_losses.detach().cpu()
        observation["www_candidate_score"] = scores.detach().cpu()
        observation["www_candidate_aggregation_weight"] = own_weight

    def _observe_exact_batch_round(
        self,
        *,
        round_index: int,
        active_attacks: set[str],
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        base_states: dict[int, dict[str, torch.Tensor]] | None,
        protocol_messages: dict[int, dict] | None,
        released_states: dict[int, dict[str, torch.Tensor]] | None,
        learning_rate: float | None,
    ) -> dict | None:
        """Collect unchanged attack scores on the exact current upload batch."""
        attacks = active_attacks & self.exact_batch_membership_attacks
        if not attacks:
            return None
        target_id = self.target_client_id
        if target_id not in selected_ids:
            raise ValueError(
                "Exact-batch membership cannot audit a round where the target "
                "client did not upload."
            )
        retained_batch = self.users[target_id].last_train_batch
        if (self.defense_name in {"www", "record_dp"}
                and retained_batch is not None and retained_batch[1].numel() == 0):
            # The noise-only upload is a valid accounted step, but has no true
            # batch members on which to define these six membership attacks.
            self.exact_batch_skipped_rounds.append({
                "communication_round": int(round_index) + 1,
                "client_id": int(target_id),
                "attacks": sorted(attacks),
                "reason": "empty_poisson_batch",
            })
            return None
        candidates = self._build_exact_batch_candidates(round_index)
        inputs = candidates["inputs"]
        labels = candidates["labels"]
        client_base = base_state if base_states is None else base_states[target_id]
        observable_base = self._observable_base_state(client_base)
        observable_state = self._observable_client_state(
            target_id,
            updated_states[target_id],
            released_states,
            base_state=client_base,
            protocol_message=(
                None
                if protocol_messages is None
                else protocol_messages.get(target_id)
            ),
            learning_rate=learning_rate,
        )
        observation = {
            "round": int(round_index),
            "client_ids": torch.tensor([target_id], dtype=torch.long),
            "membership": candidates["membership"],
            "candidate_labels": labels,
            "member_local_indices": candidates["member_local_indices"],
            "member_recycled": candidates["member_recycled"],
            "nonmember_pool_indices": candidates["nonmember_pool_indices"],
            "attacks": sorted(attacks),
        }

        loss_attacks = attacks & {
            "blackbox_loss",
            "score_diff",
            "score_ratio",
        }
        if loss_attacks:
            self.model.load_state_dict(observable_state, strict=False)
            self.model.eval()
            _, _, post_losses = self._candidate_outputs(
                self.model,
                require_representation=False,
                candidate_inputs=inputs,
                candidate_labels=labels,
                candidate_inputs_are_features=False,
            )
            observation["confidence"] = (-post_losses).unsqueeze(0)
            if attacks & {"score_diff", "score_ratio"}:
                self.model.load_state_dict(observable_base, strict=False)
                self.model.eval()
                _, _, pre_losses = self._candidate_outputs(
                    self.model,
                    require_representation=False,
                    candidate_inputs=inputs,
                    candidate_labels=labels,
                    candidate_inputs_are_features=False,
                )
                observation["pre_confidence"] = (-pre_losses).unsqueeze(0)

        gradient_attacks = attacks & {"grad_cosine", "gradient_diff"}
        if gradient_attacks:
            names = trainable_names(self.model)
            if (
                self.audit_view == "protocol_plus_released_prompts"
                and protocol_messages is not None
                and target_id in protocol_messages
            ):
                message = protocol_messages[target_id]
                tensors = message.get("tensors", {})
                sign = (
                    -1.0
                    if message.get("kind")
                    in {"model_update", "global_prompt_update"}
                    else 1.0
                )
                scale = (
                    1.0 / float(learning_rate)
                    if message.get("kind")
                    in {"model_update", "global_prompt_update"}
                    and learning_rate is not None
                    else 1.0
                )
            else:
                if self.audit_view == "released_prompt":
                    raise ValueError(
                        "Exact-batch gradient attacks require an observable "
                        "target-client update."
                    )
                tensors = {
                    name: client_base[name] - observable_state[name]
                    for name in names
                }
                sign = 1.0
                scale = (
                    1.0 / float(learning_rate)
                    if learning_rate is not None
                    else 1.0
                )
            self.model.load_state_dict(observable_base, strict=False)
            self.model.eval()
            measurements = self._raw_input_gradient_measurements(
                self.model,
                names,
                [(tensors, sign, scale)],
                need_cosine="grad_cosine" in attacks,
                need_gradient_difference="gradient_diff" in attacks,
                candidate_inputs=inputs,
                candidate_labels=labels,
            )
            if "grad_cosine" in attacks:
                observation["cosine"] = measurements["cosine"]
            if "gradient_diff" in attacks:
                observation["gradient_diff_score"] = measurements[
                    "gradient_difference"
                ]

        projres_payload = None
        if "projres" in attacks:
            projres_scores, projres_diagnostics, projres_payload = (
                self._score_exact_batch_projres(
                    round_index=round_index,
                    member_inputs=candidates["member_inputs"],
                    nonmember_inputs=candidates["nonmember_inputs"],
                    labels=labels,
                    membership=candidates["membership"],
                    member_local_indices=candidates["member_local_indices"],
                    nonmember_pool_indices=candidates[
                        "nonmember_pool_indices"
                    ],
                    base_state=client_base,
                    updated_state=updated_states[target_id],
                    protocol_message=(
                        None
                        if protocol_messages is None
                        else protocol_messages.get(target_id)
                    ),
                    learning_rate=learning_rate,
                )
            )
            observation["projres"] = projres_scores.unsqueeze(0)
            observation["projres_diagnostics"] = projres_diagnostics

        self.exact_batch_observations.append(observation)
        self.exact_batch_candidate_selections.append(candidates["selection"])
        logger.info(
            "Collected exact-batch membership signals for round %d: "
            "attacks=%s, members=%d, nonmembers=%d",
            round_index + 1,
            ",".join(sorted(attacks)),
            int((candidates["membership"] == 1).sum()),
            int((candidates["membership"] == 0).sum()),
        )
        return projres_payload

    def observe_round(
        self,
        round_index: int,
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        base_states: dict[int, dict[str, torch.Tensor]] | None = None,
        protocol_messages: dict[int, dict] | None = None,
        released_states: dict[int, dict[str, torch.Tensor]] | None = None,
        aggregation_weights: dict[int, float] | None = None,
        learning_rate: float | None = None,
    ) -> dict | None:
        active_attacks = self._attacks_for_round(round_index)
        if not self.enabled or not active_attacks:
            return None
        exact_batch_payload = self._observe_exact_batch_round(
            round_index=round_index,
            active_attacks=set(active_attacks),
            base_state=base_state,
            updated_states=updated_states,
            selected_ids=selected_ids,
            base_states=base_states,
            protocol_messages=protocol_messages,
            released_states=released_states,
            learning_rate=learning_rate,
        )
        static_attacks = [
            attack
            for attack in active_attacks
            if attack not in self.exact_batch_membership_attacks
        ]
        full_signals = self.signal_storage == "full"
        needs = _signal_needs(set(static_attacks), full_signals)
        selected_ids = self._audit_client_ids_for_attacks(
            static_attacks, selected_ids
        )
        if not selected_ids:
            return exact_batch_payload
        names = trainable_names(self.model)
        observable_names = names
        needs_gradients = (
            needs["cosine"]
            or needs["gradient_diff_score"]
            or needs["whitebox_features"]
            or needs["promptres"]
        )
        sample_gradients = None
        gradient_diff_gradients = None
        signatures = None
        streaming_text_gradients = bool(
            self.model_type in {"bert_adapter", "bert_lora", "gpt2_adapter"}
            and (needs["cosine"] or needs["gradient_diff_score"])
            and not needs["whitebox_features"]
            and not needs["promptres"]
        )
        observable_base_state = self._observable_base_state(base_state)
        if (
            needs_gradients
            and not self.candidate_inputs_are_features
            and not streaming_text_gradients
        ):
            self.model.load_state_dict(
                observable_base_state, strict=False
            )
            self.model.eval()
            if needs["cosine"] or needs["whitebox_features"] or needs["promptres"]:
                sample_gradients, signatures, _ = self._candidate_gradients(
                    self.model, observable_names
                )
            if needs["gradient_diff_score"]:
                gradient_diff_gradients, _, _ = self._candidate_gradients(
                    self.model,
                    observable_names,
                    gradient_loss="sum_over_labels",
                )

        confidence = []
        pre_confidence = []
        true_label_confidence = []
        pre_true_label_confidence = []
        cosine = []
        gradient_difference = []
        gradient_diff_score = []
        probabilities = []
        representations = []
        promptres_updates = []
        promptres_gradients = []
        cached_feature_updates = []
        streaming_text_updates = []
        audited_ids = set(self.audit_client_ids)
        gradient_diff_positions = [
            position
            for position, user_id in enumerate(selected_ids)
            if user_id in audited_ids
        ]
        observable_states = {}
        for user_id in selected_ids:
            update_is_gradient = False
            client_base = (
                base_state if base_states is None else base_states[user_id]
            )
            state = self._observable_client_state(
                user_id,
                updated_states[user_id],
                released_states,
                base_state=client_base,
                protocol_message=(
                    None
                    if protocol_messages is None
                    else protocol_messages.get(user_id)
                ),
                learning_rate=learning_rate,
            )
            if needs["client_states"]:
                observable_states[user_id] = state
            if needs_gradients:
                if streaming_text_gradients and (
                    self.audit_view == "protocol_plus_released_prompts"
                    and protocol_messages is not None
                    and user_id in protocol_messages
                ):
                    message = protocol_messages[user_id]
                    tensors = message.get("tensors", {})
                    update_is_gradient = message.get("kind") == "gradient"
                    sign = (
                        -1.0
                        if message.get("kind")
                        in {"model_update", "global_prompt_update"}
                        else 1.0
                    )
                    scale = (
                        1.0 / float(learning_rate)
                        if message.get("kind")
                        in {"model_update", "global_prompt_update"}
                        and learning_rate is not None
                        else 1.0
                    )
                    streaming_text_updates.append((tensors, sign, scale))
                    update = None
                elif streaming_text_gradients:
                    if self.audit_view == "released_prompt":
                        raise ValueError(
                            "Text gradient-cosine attacks require an observable "
                            "client update."
                        )
                    streaming_text_updates.append(
                        (
                            {
                                name: client_base[name] - state[name]
                                for name in observable_names
                            },
                            1.0,
                            (
                                1.0 / float(learning_rate)
                                if learning_rate is not None
                                else 1.0
                            ),
                        )
                    )
                    update = None
                elif self.audit_view == "released_prompt":
                    update = torch.ones(1)
                elif (
                    self.audit_view == "protocol_plus_released_prompts"
                    and protocol_messages is not None
                    and user_id in protocol_messages
                ):
                    message = protocol_messages[user_id]
                    tensors = message.get("tensors", {})
                    update_is_gradient = message.get("kind") == "gradient"
                    update = torch.cat(
                        [
                            tensor.detach().flatten().cpu()
                            for tensor in tensors.values()
                        ]
                    )
                    if message.get("kind") in {
                        "model_update",
                        "global_prompt_update",
                    }:
                        # Uploaded deltas use the opposite sign from the descent
                        # gradient used by gradient-similarity measurements.
                        update = -update
                else:
                    update = flatten_state_delta(
                        client_base, state, observable_names
                    )
                if streaming_text_gradients:
                    pass
                elif self.candidate_inputs_are_features:
                    if (
                        needs["gradient_diff_score"]
                        and not update_is_gradient
                        and learning_rate is not None
                    ):
                        update = update / float(learning_rate)
                    cached_feature_updates.append(update)
                else:
                    if sample_gradients is None:
                        raise AssertionError(
                            "Gradient-dependent audit lost its gradients."
                        )
                    compared_gradients = sample_gradients
                    if update.numel() != sample_gradients.shape[1]:
                        width = min(64, update.numel(), sample_gradients.shape[1])
                        update = F.adaptive_avg_pool1d(
                            update.view(1, 1, -1), width
                        ).flatten()
                        compared_gradients = F.adaptive_avg_pool1d(
                            sample_gradients.unsqueeze(1), width
                        ).squeeze(1)
                    update_norm = update.norm().clamp_min(1e-12)
                    sample_norm = compared_gradients.norm(dim=1).clamp_min(1e-12)
                    if needs["cosine"]:
                        cosine.append(
                            (compared_gradients @ update)
                            / (sample_norm * update_norm)
                        )
                    if needs["promptres"]:
                        promptres_updates.append(update)
                        promptres_gradients.append(compared_gradients)
                    if needs["whitebox_features"]:
                        update_sq = update.square().sum()
                        gradient_difference.append(
                            update_sq
                            - (update.unsqueeze(0) - compared_gradients)
                            .square()
                            .sum(dim=1)
                        )
                    if needs["gradient_diff_score"]:
                        if user_id not in audited_ids:
                            gradient_diff_score.append(
                                torch.full(
                                    (self.labels.numel(),), float("nan")
                                )
                            )
                        else:
                            if gradient_diff_gradients is None:
                                raise AssertionError(
                                    "Gradient-Diff lost its all-label gradients."
                                )
                            compared_diff_gradients = gradient_diff_gradients
                            gradient_update = update
                            if learning_rate is not None and not update_is_gradient:
                                gradient_update = (
                                    gradient_update / float(learning_rate)
                                )
                            if (
                                gradient_update.numel()
                                != gradient_diff_gradients.shape[1]
                            ):
                                width = min(
                                    64,
                                    gradient_update.numel(),
                                    gradient_diff_gradients.shape[1],
                                )
                                gradient_update = F.adaptive_avg_pool1d(
                                    gradient_update.view(1, 1, -1), width
                                ).flatten()
                                compared_diff_gradients = F.adaptive_avg_pool1d(
                                    gradient_diff_gradients.unsqueeze(1), width
                                ).squeeze(1)
                            gradient_diff_score.append(
                                2.0 * (compared_diff_gradients @ gradient_update)
                                - compared_diff_gradients.square().sum(dim=1)
                            )
            needs_outputs = (
                needs["confidence"]
                or needs["true_label_confidence"]
                or needs["probabilities"]
                or needs["representations"]
            )
            if needs_outputs:
                self.model.load_state_dict(state, strict=False)
                logits, representation, losses = self._candidate_outputs(
                    self.model,
                    require_representation=needs["representations"],
                )
                if needs["confidence"]:
                    confidence.append(-losses)
                if needs["true_label_confidence"]:
                    true_label_confidence.append(
                        torch.softmax(logits, dim=1)
                        .gather(1, self.labels.detach().cpu().view(-1, 1))
                        .squeeze(1)
                    )
                if needs["probabilities"]:
                    probabilities.append(torch.softmax(logits, dim=1))
                if needs["representations"]:
                    if representation is None:
                        raise AssertionError(
                            "Representation-dependent audit lost its features."
                        )
                    representations.append(representation)

        if needs["pre_confidence"] or needs["pre_true_label_confidence"]:
            for user_id in selected_ids:
                if user_id not in audited_ids:
                    pre_confidence.append(
                        torch.full((self.labels.numel(),), float("nan"))
                    )
                    pre_true_label_confidence.append(
                        torch.full((self.labels.numel(),), float("nan"))
                    )
                    continue
                client_base = base_state if base_states is None else base_states[user_id]
                self.model.load_state_dict(client_base, strict=False)
                logits, _, losses = self._candidate_outputs(
                    self.model, require_representation=False
                )
                pre_confidence.append(-losses)
                pre_true_label_confidence.append(
                    torch.softmax(logits, dim=1)
                    .gather(1, self.labels.detach().cpu().view(-1, 1))
                    .squeeze(1)
                )

        observation = {
            "round": int(round_index),
            "client_ids": torch.tensor(selected_ids, dtype=torch.long),
        }
        if needs["confidence"]:
            observation["confidence"] = torch.stack(confidence)
        if needs["pre_confidence"]:
            observation["pre_confidence"] = torch.stack(pre_confidence)
        if needs["true_label_confidence"]:
            observation["true_label_confidence"] = torch.stack(
                true_label_confidence
            )
        if needs["pre_true_label_confidence"]:
            observation["pre_true_label_confidence"] = torch.stack(
                pre_true_label_confidence
            )
        if needs["cosine"]:
            if self.candidate_inputs_are_features:
                self.model.load_state_dict(observable_base_state, strict=False)
                self.model.eval()
                observation["cosine"] = self._cached_feature_gradient_cosines(
                    self.model,
                    observable_names,
                    cached_feature_updates,
                )
            elif streaming_text_gradients:
                self.model.load_state_dict(observable_base_state, strict=False)
                self.model.eval()
                measurements = self._raw_input_gradient_measurements(
                    self.model,
                    observable_names,
                    streaming_text_updates,
                    need_cosine=True,
                    need_gradient_difference=needs["gradient_diff_score"],
                    gradient_difference_update_indices=gradient_diff_positions,
                )
                observation["cosine"] = measurements["cosine"]
            else:
                observation["cosine"] = torch.stack(cosine)
        if needs["gradient_diff_score"]:
            if self.candidate_inputs_are_features:
                normalized_updates = [
                    (
                        cached_feature_updates[position] / float(learning_rate)
                        if learning_rate is not None
                        else cached_feature_updates[position]
                    )
                    for position in gradient_diff_positions
                ]
                self.model.load_state_dict(observable_base_state, strict=False)
                self.model.eval()
                selected_scores = self._cached_feature_gradient_differences(
                    self.model,
                    observable_names,
                    normalized_updates,
                )
                observation["gradient_diff_score"] = torch.full(
                    (len(selected_ids), self.labels.numel()), float("nan")
                )
                observation["gradient_diff_score"][gradient_diff_positions] = (
                    selected_scores
                )
            elif streaming_text_gradients:
                if not needs["cosine"]:
                    self.model.load_state_dict(observable_base_state, strict=False)
                    self.model.eval()
                    measurements = self._raw_input_gradient_measurements(
                        self.model,
                        observable_names,
                        streaming_text_updates,
                        need_cosine=False,
                        need_gradient_difference=True,
                        gradient_difference_update_indices=(
                            gradient_diff_positions
                        ),
                    )
                observation["gradient_diff_score"] = torch.full(
                    (len(selected_ids), self.labels.numel()), float("nan")
                )
                observation["gradient_diff_score"][gradient_diff_positions] = (
                    measurements["gradient_difference"]
                )
            else:
                observation["gradient_diff_score"] = torch.stack(
                    gradient_diff_score
                )
        if needs["whitebox_features"]:
            if signatures is None:
                raise AssertionError("White-box audit lost its gradient signatures.")
            observation["gradient_difference"] = torch.stack(gradient_difference)
            observation["gradient_signature"] = signatures
            observation["candidate_labels"] = self.labels.detach().cpu().clone()
        if needs["probabilities"]:
            observation["probabilities"] = torch.stack(probabilities)
        if needs["representations"]:
            observation["representations"] = torch.stack(representations)
        if needs["promptres"]:
            updates = torch.stack(promptres_updates)
            configured_rank = int(self.config.get("promptres_background_rank", 0))
            if configured_rank > 0 and updates.shape[0] < 2:
                raise ValueError(
                    "PromptRes background residualization needs another selected client."
                )
            promptres_scores = []
            effective_ranks = []
            for position in range(len(selected_ids)):
                references = torch.cat(
                    (updates[:position], updates[position + 1 :]), dim=0
                )
                rank = configured_rank if references.shape[0] else 0
                scores, used_rank = promptres_round_scores(
                    updates[position],
                    promptres_gradients[position],
                    references if rank else None,
                    background_rank=rank,
                )
                promptres_scores.append(scores)
                effective_ranks.append(used_rank)
            observation["promptres"] = torch.stack(promptres_scores)
            observation["promptres_effective_ranks"] = effective_ranks
        if needs["client_states"]:
            observation["client_states"] = {
                user_id: {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in observable_states[user_id].items()
                }
                for user_id in selected_ids
            }
        if self.signal_storage == "full":
            observation["protocol_messages"] = {
                user_id: {
                    "kind": protocol_messages[user_id].get("kind", "unknown"),
                    "tensors": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in protocol_messages[user_id]
                        .get("tensors", {})
                        .items()
                    },
                }
                for user_id in selected_ids
                if protocol_messages is not None and user_id in protocol_messages
            }
            observation["audit_view"] = self.audit_view
        self._attach_www_candidate_scores(
            observation,
            base_state=base_state,
            updated_states=updated_states,
            base_states=base_states,
            protocol_messages=protocol_messages,
            released_states=released_states,
            aggregation_weights=aggregation_weights,
            learning_rate=learning_rate,
        )
        self.observations.append(observation)
        logger.debug("Collected privacy signals for round %s", round_index)
        return exact_batch_payload

    def _run_fedmia_signal(
        self,
        measurement: str,
        aggregation: str,
        tail: str,
    ) -> AttackResult:
        attack = FEDMIA_MEASUREMENT_NAMES[measurement]
        observations = self._observations_for_attack(attack)
        if not self.pooled_client_audit:
            return run_fedmia(
                observations,
                self.membership,
                self.target_client_id,
                measurement,
                aggregation,
                tail,
                float(self.config.get("fedmia_tail_calibration_fraction", 0.25)),
                self.seed,
            )

        client_results = []
        evaluated_client_ids = []
        skipped_clients = dict(self.skipped_audit_clients)
        global_indices = []
        for client_id in self.audit_client_ids:
            indices = torch.nonzero(
                self.candidate_client_ids == client_id, as_tuple=False
            ).flatten()
            has_usable_round = any(
                client_id in observation["client_ids"].tolist()
                and observation["client_ids"].numel() >= 2
                for observation in observations
            )
            if not has_usable_round and self.allow_partial_client_audit:
                error = (
                    "FedMIA needs a round containing the target and another client."
                )
                skipped_clients[str(client_id)] = error
                logger.warning(
                    "Skipping client %d in pooled %s attack: %s",
                    client_id,
                    measurement,
                    error,
                )
                continue
            result = run_fedmia(
                observations,
                self.membership[indices],
                client_id,
                measurement,
                aggregation,
                tail,
                float(self.config.get("fedmia_tail_calibration_fraction", 0.25)),
                self.seed + 1009 * client_id,
                candidate_indices=indices,
            )
            client_results.append(result)
            evaluated_client_ids.append(client_id)
            global_indices.append(indices[result.sample_indices])
        if not client_results:
            raise ValueError(
                "FedMIA could not evaluate any few-shot client. At least one "
                "audited client must participate in a round with another client."
            )
        name = attack
        per_client_metrics = {
            str(client_id): self._pooled_client_metadata(result)
            for client_id, result in zip(evaluated_client_ids, client_results)
        }
        return AttackResult(
            name=name,
            scores=torch.cat([result.scores for result in client_results]),
            labels=torch.cat([result.labels for result in client_results]),
            sample_indices=torch.cat(global_indices),
            metadata={
                "scope": "pooled_clients",
                "requested_audit_client_ids": self.requested_audit_client_ids,
                "audit_client_ids": evaluated_client_ids,
                "skipped_audit_clients": skipped_clients,
                "few_shot": self.few_shot,
                "fpl_shots": self.fpl_shots,
                "measurement": measurement,
                "round_aggregation": aggregation,
                "tail_policy": tail,
                "score_pooling": "native_fedmia_null_cdf",
                "macro_metrics": self._macro_client_metrics(per_client_metrics),
                "per_client_metrics": per_client_metrics,
                "per_client": {
                    str(client_id): result.metadata
                    for client_id, result in zip(
                        evaluated_client_ids, client_results
                    )
                },
            },
        )

    @staticmethod
    def _slice_candidate_observations(
        observations: list[dict], candidate_indices: torch.Tensor
    ) -> list[dict]:
        """Select one client's candidates without mutating stored round signals."""
        candidate_indices = candidate_indices.detach().cpu().long()
        selected = []
        for observation in observations:
            sliced = {}
            for key, value in observation.items():
                if key in _CLIENT_CANDIDATE_FIELDS:
                    sliced[key] = value.index_select(1, candidate_indices)
                elif key in _CANDIDATE_FIELDS:
                    sliced[key] = value.index_select(0, candidate_indices)
                else:
                    sliced[key] = value
            selected.append(sliced)
        return selected

    @staticmethod
    def _client_rank_scores(scores: torch.Tensor) -> torch.Tensor:
        """Map scores to a label-free within-client empirical CDF."""
        values = scores.detach().to(device="cpu", dtype=torch.float64).flatten()
        _, inverse, counts = torch.unique(
            values,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        upper = counts.cumsum(dim=0).to(torch.float64)
        lower = upper - counts
        average_rank = lower + (counts.to(torch.float64) + 1.0) / 2.0
        return (average_rank[inverse] / values.numel()).to(torch.float32)

    @staticmethod
    def _pooled_client_metadata(result: AttackResult) -> dict:
        summary = result.to_summary()
        return {
            "auc": summary["auc"],
            "reportable_metrics": summary["reportable_metrics"],
            "member_count": summary["member_count"],
            "nonmember_count": summary["nonmember_count"],
            "num_samples": summary["num_samples"],
            "attack_metadata": result.metadata,
        }

    @staticmethod
    def _macro_client_metrics(per_client: dict[str, dict]) -> dict:
        metric_keys = (
            "auc",
            "tpr_at_fpr_0.1",
            "tpr_at_fpr_0.01",
            "tpr_at_fpr_0.001",
        )
        macro = {}
        for key in metric_keys:
            values = []
            for client in per_client.values():
                value = (
                    client["auc"]
                    if key == "auc"
                    else client["reportable_metrics"].get(key)
                )
                if value is not None:
                    values.append(float(value))
            if not values:
                macro[key] = {"mean": None, "std": None, "clients": 0}
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            macro[key] = {
                "mean": float(tensor.mean()),
                "std": float(tensor.std(unbiased=False)),
                "clients": len(values),
            }
        return macro

    def _run_pooled_client_once(
        self,
        attack: str,
        client_id: int,
        indices: torch.Tensor,
    ) -> AttackResult:
        observations = self._slice_candidate_observations(
            self._observations_for_attack(attack), indices
        )
        membership = self.membership[indices]
        labels = self.labels[indices]
        seed = self.seed + 1009 * client_id
        if attack in {
            "blackbox_loss",
            "loss_series",
            "grad_cosine",
            "avg_cosine",
        }:
            return run_fedmia_baseline(
                observations,
                membership,
                client_id,
                attack,
                self.config.get("fedmia_baseline_single_round", "last"),
            )
        if attack in _UPDATE_ATTACKS:
            return run_update_attack(
                observations,
                membership,
                client_id,
                attack,
                score_ratio_damping=float(
                    self.config.get("score_ratio_damping", 1e-6)
                ),
                fta_measurement=str(
                    self.config.get("fta_measurement", "confidence")
                ),
            )
        if attack == "promptres":
            return run_promptres(
                observations,
                membership,
                client_id,
                str(self.config.get("promptres_aggregation", "mean")),
            )
        if attack == "nasr_passive":
            return run_passive_whitebox(
                observations,
                membership,
                client_id,
                self.calibration_fraction,
                seed,
            )
        if attack == "transfer_representation":
            return run_transfer_representation_attack(
                observations,
                membership,
                client_id,
                self.calibration_fraction,
                seed,
            )
        if attack == "rmia":
            return run_rmia(
                observations,
                membership,
                labels,
                client_id,
                float(self.config.get("auxiliary_fraction", 0.5)),
                seed,
                offline_a=float(self.config.get("rmia_offline_a", 0.3)),
                gamma=float(self.config.get("rmia_gamma", 1.0)),
            )
        if attack == "quantile_mia":
            return run_quantile_mia(
                observations,
                membership,
                labels,
                client_id,
                float(self.config.get("auxiliary_fraction", 0.5)),
                seed,
                quantile=float(self.config.get("qmia_quantile", 0.9)),
                epochs=int(self.config.get("qmia_epochs", 200)),
                learning_rate=float(self.config.get("qmia_learning_rate", 0.01)),
            )
        raise AssertionError(f"Unhandled pooled membership attack {attack}")

    def _run_pooled_client_attack(self, attack: str) -> AttackResult:
        client_results = []
        evaluated_client_ids = []
        global_indices = []
        skipped_clients = dict(self.skipped_audit_clients)
        for client_id in self.audit_client_ids:
            indices = torch.nonzero(
                self.candidate_client_ids == client_id, as_tuple=False
            ).flatten()
            try:
                result = self._run_pooled_client_once(
                    attack, client_id, indices
                )
            except Exception as error:
                if not self.allow_partial_client_audit:
                    raise
                skipped_clients[str(client_id)] = f"{type(error).__name__}: {error}"
                logger.warning(
                    "Skipping client %d in pooled %s attack: %s",
                    client_id,
                    attack,
                    error,
                )
                continue
            client_results.append(result)
            evaluated_client_ids.append(client_id)
            global_indices.append(indices[result.sample_indices])
        if not client_results:
            raise ValueError(f"Pooled {attack} could not evaluate any client.")

        per_client_metrics = {
            str(client_id): self._pooled_client_metadata(result)
            for client_id, result in zip(evaluated_client_ids, client_results)
        }
        return AttackResult(
            name=attack,
            scores=torch.cat(
                [self._client_rank_scores(result.scores) for result in client_results]
            ),
            labels=torch.cat([result.labels for result in client_results]),
            sample_indices=torch.cat(global_indices),
            metadata={
                "scope": "pooled_clients",
                "requested_audit_client_ids": self.requested_audit_client_ids,
                "audit_client_ids": evaluated_client_ids,
                "skipped_audit_clients": skipped_clients,
                "few_shot": self.few_shot,
                "fpl_shots": self.fpl_shots,
                "score_pooling": (
                    "per_client_empirical_cdf_without_membership_labels"
                ),
                "macro_metrics": self._macro_client_metrics(per_client_metrics),
                "per_client_metrics": per_client_metrics,
                "per_client": {
                    str(client_id): result.metadata
                    for client_id, result in zip(
                        evaluated_client_ids, client_results
                    )
                },
            },
        )

    def _run_exact_batch_attack(
        self, attack: str, observation: dict
    ) -> AttackResult:
        """Evaluate one round because exact batch identities change each round."""
        membership = observation["membership"]
        if attack == "projres":
            diagnostics = observation["projres_diagnostics"]
            result = AttackResult(
                name="projres",
                scores=observation["projres"][0].detach().cpu(),
                labels=membership.detach().cpu(),
                sample_indices=torch.arange(membership.numel()),
                metadata=dict(diagnostics["metadata"]),
            )
        elif attack in {"blackbox_loss", "grad_cosine"}:
            result = run_fedmia_baseline(
                [observation],
                membership,
                self.target_client_id,
                attack,
                "last",
            )
        else:
            result = run_update_attack(
                [observation],
                membership,
                self.target_client_id,
                attack,
                score_ratio_damping=float(
                    self.config.get("score_ratio_damping", 1e-6)
                ),
            )
        result.metadata.update(
            {
                "membership_definition": (
                    self._exact_batch_membership_definition()
                ),
                "nonmember_training_exposure": "never_trained",
                "label_matching_mode": "exact_batch_histogram_ratio",
                "nonmember_to_member_ratio": (
                    self.exact_batch_nonmember_ratio
                ),
                "communication_round": int(observation["round"]) + 1,
                "cofedmid": self._cofedmid_metadata(),
                "temporal_information": "single_round",
                "round_reduction": "none",
                "member_local_indices": observation[
                    "member_local_indices"
                ].tolist(),
                "nonmember_pool_indices": observation[
                    "nonmember_pool_indices"
                ].tolist(),
            }
        )
        return result

    def _run(self, attack: str, final_model, final_state):
        if attack in self.exact_batch_membership_attacks:
            observations = self._observations_for_attack(attack)
            if not observations:
                raise ValueError(
                    f"Exact-batch {attack} has no observed target-client batch."
                )
            return self._run_exact_batch_attack(attack, observations[-1])
        if self.pooled_client_audit and attack in POOLED_CLIENT_ATTACKS:
            if attack == "fedmia_loss":
                return self._run_fedmia_signal(
                    "confidence",
                    str(self.config.get("fedmia_loss_aggregation", "mean")),
                    str(
                        self.config.get(
                            "fedmia_loss_tail",
                            self.config.get("fedmia_tail", "upper"),
                        )
                    ),
                )
            if attack == "fedmia_cosine":
                return self._run_fedmia_signal(
                    "cosine",
                    str(self.config.get("fedmia_cosine_aggregation", "mean")),
                    str(
                        self.config.get(
                            "fedmia_cosine_tail",
                            self.config.get("fedmia_tail", "upper"),
                        )
                    ),
                )
            return self._run_pooled_client_attack(attack)
        observations = self._observations_for_attack(attack)
        if attack in {
            "blackbox_loss",
            "loss_series",
            "grad_cosine",
            "avg_cosine",
        }:
            return run_fedmia_baseline(
                observations,
                self.membership,
                self.target_client_id,
                attack,
                self.config.get("fedmia_baseline_single_round", "last"),
            )
        if attack in _UPDATE_ATTACKS:
            return run_update_attack(
                observations,
                self.membership,
                self.target_client_id,
                attack,
                score_ratio_damping=float(
                    self.config.get("score_ratio_damping", 1e-6)
                ),
                fta_measurement=str(
                    self.config.get("fta_measurement", "confidence")
                ),
            )
        if attack == "promptres":
            return run_promptres(
                observations,
                self.membership,
                self.target_client_id,
                str(self.config.get("promptres_aggregation", "mean")),
            )
        if attack == "fedmia_loss":
            return self._run_fedmia_signal(
                "confidence",
                str(self.config.get("fedmia_loss_aggregation", "mean")),
                str(
                    self.config.get(
                        "fedmia_loss_tail", self.config.get("fedmia_tail", "upper")
                    )
                ),
            )
        if attack == "fedmia_cosine":
            return self._run_fedmia_signal(
                "cosine",
                str(self.config.get("fedmia_cosine_aggregation", "mean")),
                str(
                    self.config.get(
                        "fedmia_cosine_tail", self.config.get("fedmia_tail", "upper")
                    )
                ),
            )
        if attack == "nasr_passive":
            return run_passive_whitebox(
                observations,
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
                probe_cycles=int(self.config.get("active_probe_cycles", 3)),
                calibration_fraction=self.calibration_fraction,
                seed=self.seed,
            )
        if attack == "transfer_representation":
            return run_transfer_representation_attack(
                observations,
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
                observations,
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
                observations,
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
                    observations, self.target_client_id
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
                observations,
                self.target_client_id,
                self.images,
                self.labels,
                self.membership,
                max_samples=int(self.config.get("query_max_samples", 16)),
                steps=int(self.config.get("yoqo_steps", 20)),
                learning_rate=float(self.config.get("yoqo_learning_rate", 0.01)),
                epsilon=float(self.config.get("query_epsilon", 0.1)),
                distortion_weight=float(self.config.get("yoqo_distortion_weight", 1.0)),
                reference_models=int(self.config.get("query_reference_models", 2)),
                loss_threshold=(
                    None
                    if self.config.get("yoqo_loss_threshold", 0.5) is None
                    else float(self.config.get("yoqo_loss_threshold", 0.5))
                ),
            )
        if attack == "canary":
            return run_canary(
                final_model,
                observations,
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

    def _run_periodic_attack_prefix(
        self, attack: str, observations: list[dict]
    ) -> AttackResult:
        """Evaluate one configured checkpoint without changing final metrics."""
        if attack in self.exact_batch_membership_attacks:
            return self._run_exact_batch_attack(attack, observations[-1])
        if attack in _FEDMIA_BASELINE_ATTACKS:
            selected = (
                observations[-1:]
                if attack in _SINGLE_ROUND_ATTACKS
                else observations
            )
            return run_fedmia_baseline(
                selected,
                self.membership,
                self.target_client_id,
                attack,
                "last",
            )
        if attack in _UPDATE_ATTACKS:
            return run_update_attack(
                observations,
                self.membership,
                self.target_client_id,
                attack,
                score_ratio_damping=float(
                    self.config.get("score_ratio_damping", 1e-6)
                ),
                fta_measurement=str(
                    self.config.get("fta_measurement", "confidence")
                ),
            )
        measurement = "confidence" if attack == "fedmia_loss" else "cosine"
        aggregation = str(
            self.config.get(f"{attack}_aggregation", "mean")
        )
        tail = str(
            self.config.get(
                f"{attack}_tail", self.config.get("fedmia_tail", "upper")
            )
        )
        return run_fedmia(
            observations,
            self.membership,
            self.target_client_id,
            measurement,
            aggregation,
            tail,
            float(self.config.get("fedmia_tail_calibration_fraction", 0.25)),
            self.seed,
        )

    def _paper_balanced_attack_summary(
        self, result: AttackResult
    ) -> dict | None:
        """Evaluate one attack on the fixed balanced subset of its scores."""
        candidate_indices = getattr(
            self, "paper_balanced_candidate_indices", None
        )
        metadata = getattr(self, "paper_balanced_evaluation", None)
        if candidate_indices is None or metadata is None:
            return None
        result_positions = {
            int(candidate_index): position
            for position, candidate_index in enumerate(
                result.sample_indices.detach().cpu().tolist()
            )
        }
        try:
            selected_positions = torch.tensor(
                [
                    result_positions[int(candidate_index)]
                    for candidate_index in candidate_indices.tolist()
                ],
                dtype=torch.long,
            )
        except KeyError as error:
            raise ValueError(
                "Attack result does not contain every paper-balanced candidate."
            ) from error
        balanced = AttackResult(
            name=result.name,
            scores=result.scores[selected_positions],
            labels=result.labels[selected_positions],
            sample_indices=result.sample_indices[selected_positions],
        ).to_summary()
        return {
            "name": metadata["name"],
            "selection_method": metadata["selection_method"],
            "shared_across_attacks_and_rounds": True,
            "member_count": balanced["member_count"],
            "nonmember_count": balanced["nonmember_count"],
            "num_samples": balanced["num_samples"],
            "fpr_resolution": balanced["fpr_resolution"],
            "auc": balanced["auc"],
            "tpr_at_fpr_0.1": balanced["reportable_metrics"][
                "tpr_at_fpr_0.1"
            ],
            "tpr_at_fpr_0.01": balanced["reportable_metrics"][
                "tpr_at_fpr_0.01"
            ],
            "tpr_at_fpr_0.001": balanced["reportable_metrics"][
                "tpr_at_fpr_0.001"
            ],
            "metric_availability": balanced["metric_availability"],
            "score_degenerate": balanced["score_degenerate"],
        }

    def _write_periodic_attack_metrics(self) -> int:
        """Persist one metric row per explicitly scheduled attack checkpoint."""
        if self.pooled_client_audit:
            return 0
        rows = []
        for attack in self.attacks:
            if (
                attack not in _PERIODIC_METRIC_ATTACKS
                or attack not in self.attack_audit_intervals
            ):
                continue
            observations = self._observations_for_attack(attack)
            for position, observation in enumerate(observations):
                completed_round = int(observation["round"]) + 1
                try:
                    result = self._run_periodic_attack_prefix(
                        attack, observations[: position + 1]
                    )
                    summary = result.to_summary(
                        fpr_targets=(
                            _EXACT_BATCH_REPORTED_FPR_TARGETS
                            if attack in self.exact_batch_membership_attacks
                            else (0.1, 0.01, 0.001)
                        )
                    )
                    paper_balanced = (
                        None
                        if attack in self.exact_batch_membership_attacks
                        else self._paper_balanced_attack_summary(result)
                    )
                except Exception as error:
                    error_key = f"round_metrics:{attack}:round_{completed_round}"
                    self.errors[error_key] = f"{type(error).__name__}: {error}"
                    logger.exception(
                        "Periodic metric for %s at round %d failed",
                        attack,
                        completed_round,
                    )
                    continue
                reportable = summary["reportable_metrics"]
                row = {
                    "communication_round": completed_round,
                    "attack": attack,
                    "temporal_scope": (
                        "single_round"
                        if attack in _SINGLE_ROUND_ATTACKS
                        or attack in self.exact_batch_membership_attacks
                        else "cumulative"
                    ),
                    "auc": summary["auc"],
                    "tpr_at_fpr_0.1": reportable["tpr_at_fpr_0.1"],
                    "tpr_at_fpr_0.01": reportable["tpr_at_fpr_0.01"],
                    "num_samples": summary["num_samples"],
                    "member_count": summary["member_count"],
                    "nonmember_count": summary["nonmember_count"],
                    "fpr_resolution": summary["fpr_resolution"],
                    "score_degenerate": summary["score_degenerate"],
                }
                if attack not in self.exact_batch_membership_attacks:
                    row["tpr_at_fpr_0.001"] = reportable[
                        "tpr_at_fpr_0.001"
                    ]
                if paper_balanced is not None:
                    row.update(
                        {
                            "paper_100_100_auc": paper_balanced["auc"],
                            "paper_100_100_tpr_at_fpr_0.1": (
                                paper_balanced["tpr_at_fpr_0.1"]
                            ),
                            "paper_100_100_tpr_at_fpr_0.01": (
                                paper_balanced["tpr_at_fpr_0.01"]
                            ),
                            "paper_100_100_tpr_at_fpr_0.001": (
                                paper_balanced["tpr_at_fpr_0.001"]
                            ),
                        }
                    )
                rows.append(row)
        if not rows:
            return 0
        path = os.path.join(self.results_dir, "attack_round_metrics.csv")
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved %d periodic attack metric rows to %s", len(rows), path)
        return len(rows)

    def _write_www_candidate_artifacts(self) -> dict | None:
        """Persist sample-aligned WWW scores for the fixed FedMIA candidates.

        The raw artifact keeps one score per candidate and audit round.  The
        compact artifact aggregates those scores by sample and joins the final
        FedMIA-Loss score, so downstream analysis never has to infer candidate
        alignment from row order.
        """
        if not self.www_candidate_scoring:
            return None
        observations = [
            observation
            for observation in self.observations
            if "www_candidate_score" in observation
        ]
        if not observations:
            raise ValueError(
                "WWW candidate scoring was enabled but no scored audit round "
                "was collected."
            )

        candidate_count = int(self.labels.numel())
        rounds = [int(observation["round"]) + 1 for observation in observations]

        def stacked_field(name: str) -> torch.Tensor:
            parts = [
                observation[name].detach().cpu().to(torch.float64).flatten()
                for observation in observations
            ]
            if any(part.numel() != candidate_count for part in parts):
                raise ValueError(
                    f"WWW field {name!r} is not aligned with all candidates."
                )
            return torch.stack(parts)

        scores = stacked_field("www_candidate_score")
        own_losses = stacked_field("www_candidate_own_loss")
        other_losses = stacked_field("www_candidate_other_loss")
        weights = [
            float(observation["www_candidate_aggregation_weight"])
            for observation in observations
        ]
        if not torch.isfinite(scores).all():
            raise ValueError("WWW candidate scores contain non-finite values.")

        membership = self.membership.detach().cpu().long().flatten()
        class_labels = self.labels.detach().cpu().long().flatten()
        sources = self.candidate_source_names
        if sources is None:
            sources = ["unknown"] * candidate_count
        if len(sources) != candidate_count:
            raise ValueError("WWW candidate-source metadata is not sample-aligned.")

        fedmia_result = next(
            (result for result in self.results if result.name == "fedmia_loss"),
            None,
        )
        if fedmia_result is None:
            raise ValueError(
                "WWW candidate scoring requires a completed FedMIA-Loss result."
            )
        fedmia_scores = torch.full(
            (candidate_count,), float("nan"), dtype=torch.float64
        )
        fedmia_indices = fedmia_result.sample_indices.detach().cpu().long()
        fedmia_scores[fedmia_indices] = (
            fedmia_result.scores.detach().cpu().to(torch.float64)
        )
        if not torch.isfinite(fedmia_scores).all():
            raise ValueError(
                "FedMIA-Loss did not return one finite score for every WWW candidate."
            )

        round_path = os.path.join(
            self.results_dir, "www_candidate_round_scores.csv"
        )
        with open(round_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    "communication_round",
                    "sample_index",
                    "candidate_source",
                    "membership",
                    "class_label",
                    "own_loss",
                    "other_loss",
                    "www_score",
                    "target_aggregation_weight",
                )
            )
            membership_values = membership.tolist()
            class_values = class_labels.tolist()
            for round_position, communication_round in enumerate(rounds):
                for sample_index, (own_loss, other_loss, score) in enumerate(
                    zip(
                        own_losses[round_position].tolist(),
                        other_losses[round_position].tolist(),
                        scores[round_position].tolist(),
                    )
                ):
                    writer.writerow(
                        (
                            communication_round,
                            sample_index,
                            sources[sample_index],
                            membership_values[sample_index],
                            class_values[sample_index],
                            own_loss,
                            other_loss,
                            score,
                            weights[round_position],
                        )
                    )

        score_mean = scores.mean(dim=0)
        score_std = scores.std(dim=0, unbiased=False)
        score_min = scores.min(dim=0).values
        score_max = scores.max(dim=0).values
        own_loss_mean = own_losses.mean(dim=0)
        other_loss_mean = other_losses.mean(dim=0)
        sample_path = os.path.join(
            self.results_dir, "www_candidate_scores.csv"
        )
        with open(sample_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    "sample_index",
                    "candidate_source",
                    "membership",
                    "class_label",
                    "observation_count",
                    "www_score_mean",
                    "www_score_std",
                    "www_score_min",
                    "www_score_max",
                    "www_score_last",
                    "www_score_last_round",
                    "own_loss_mean",
                    "other_loss_mean",
                    "fedmia_loss_score",
                )
            )
            for sample_index in range(candidate_count):
                writer.writerow(
                    (
                        sample_index,
                        sources[sample_index],
                        membership_values[sample_index],
                        class_values[sample_index],
                        len(rounds),
                        float(score_mean[sample_index]),
                        float(score_std[sample_index]),
                        float(score_min[sample_index]),
                        float(score_max[sample_index]),
                        float(scores[-1, sample_index]),
                        rounds[-1],
                        float(own_loss_mean[sample_index]),
                        float(other_loss_mean[sample_index]),
                        float(fedmia_scores[sample_index]),
                    )
                )

        www_result = AttackResult(
            name="www_candidate_score",
            scores=score_mean.to(torch.float32),
            labels=membership,
            sample_indices=torch.arange(candidate_count, dtype=torch.long),
            metadata={
                "score_formula": "loss_other_clients_minus_loss_target_client",
                "round_aggregation": "mean",
                "rounds": rounds,
                "target_client_id": int(self.target_client_id),
            },
        )
        www_metrics = www_result.to_summary()

        def relationship(mask: torch.Tensor) -> dict:
            count = int(mask.sum())
            return {
                "samples": count,
                "pearson": _pearson(score_mean[mask], fedmia_scores[mask]),
                "spearman": _spearman(score_mean[mask], fedmia_scores[mask]),
                "mean_www_score": (
                    float(score_mean[mask].mean()) if count else None
                ),
                "mean_fedmia_loss_score": (
                    float(fedmia_scores[mask].mean()) if count else None
                ),
            }

        relationships = {
            "all": relationship(torch.ones(candidate_count, dtype=torch.bool)),
            "members": relationship(membership == 1),
            "nonmembers": relationship(membership == 0),
            "by_candidate_source": {},
        }
        for source in dict.fromkeys(sources):
            source_mask = torch.tensor(
                [candidate_source == source for candidate_source in sources],
                dtype=torch.bool,
            )
            relationships["by_candidate_source"][source] = relationship(
                source_mask
            )

        payload = {
            "status": "ok",
            "score_formula": "L(x; theta_-k) - L(x; theta_k)",
            "score_direction": "higher_means_more_target_client_specific",
            "training_effect": "observational_only",
            "target_client_id": int(self.target_client_id),
            "candidate_count": candidate_count,
            "member_count": int((membership == 1).sum()),
            "nonmember_count": int((membership == 0).sum()),
            "audit_rounds": rounds,
            "round_aggregation": "mean",
            "www_membership_metrics": www_metrics,
            "relationship_with_fedmia_loss": relationships,
            "artifacts": {
                "per_round_scores": os.path.basename(round_path),
                "per_sample_scores": os.path.basename(sample_path),
                "relationship_summary": "www_candidate_relationship.json",
            },
        }
        relationship_path = os.path.join(
            self.results_dir, "www_candidate_relationship.json"
        )
        with open(relationship_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
        logger.info(
            "Saved sample-aligned WWW scores for %d candidates across %d rounds",
            candidate_count,
            len(rounds),
        )
        return payload

    def finalize(
        self,
        final_model: torch.nn.Module,
        final_state: dict[str, torch.Tensor],
    ) -> list[dict]:
        if not self.enabled:
            return []
        os.makedirs(self.results_dir, exist_ok=True)
        low_fpr_candidate_selection = getattr(
            self, "low_fpr_candidate_selection", None
        )
        if low_fpr_candidate_selection is not None:
            torch.save(
                low_fpr_candidate_selection,
                os.path.join(self.results_dir, "candidate_selection.pt"),
            )
        if self.exact_batch_candidate_selections:
            torch.save(
                {
                    "membership_definition": (
                        self._exact_batch_membership_definition()
                    ),
                    "nonmember_training_exposure": "never_trained",
                    "attacks": sorted(self.exact_batch_membership_attacks),
                    "rounds": self.exact_batch_candidate_selections,
                },
                os.path.join(
                    self.results_dir, "exact_batch_candidate_selection.pt"
                ),
            )
        final_model = (
            final_model
            if getattr(final_model, "client_scoped_parameters", False)
            else copy.deepcopy(final_model).to(self.device)
        )
        final_model.load_state_dict(final_state, strict=False)
        if self.defense_name == "hamp":
            attach_hamp_output_transform(
                final_model,
                float(self.defense_config.get("hamp_output_temperature", 4.0)),
            )
        for attack in self.attacks:
            try:
                if (attack in self.exact_batch_membership_attacks
                        and not self._observations_for_attack(attack)
                        and any(attack in row["attacks"] for row in self.exact_batch_skipped_rounds)):
                    continue
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                result = self._run(attack, final_model, final_state)
                result.metadata.setdefault("model_type", self.model_type)
                if self.model_type == "clip_mlp":
                    result.metadata.setdefault("trainable_scope", "mlp_only")
                    if attack == "promptmia":
                        result.metadata.setdefault(
                            "signal_space", "class_decision_vectors"
                        )
                elif self.model_type in {"clip_adapter", "visual_adapter"}:
                    result.metadata.setdefault(
                        "trainable_scope", trainable_scope_name(final_model)
                    )
                    if attack == "promptmia":
                        result.metadata.setdefault(
                            "signal_space", "adapter_input_projection_vectors"
                        )
                elif self.model_type in {"clip_lora", "bert_lora"}:
                    result.metadata.setdefault(
                        "trainable_scope", trainable_scope_name(final_model)
                    )
                    result.metadata.setdefault(
                        "lora_aggregation",
                        f"factor_wise_{self.federated_method}",
                    )
                    if self.model_type == "bert_lora":
                        result.metadata.setdefault(
                            "gradient_evaluation",
                            "streamed_exact_full_peft_update",
                        )
                elif self.model_type in {
                    "bert_adapter",
                    "bert_lora",
                    "gpt2_adapter",
                }:
                    result.metadata.setdefault(
                        "trainable_scope", trainable_scope_name(final_model)
                    )
                    result.metadata.setdefault(
                        "gradient_evaluation", "streamed_exact_full_peft_update"
                    )
                self.results.append(result)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as error:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                logger.exception("Membership attack %s failed", attack)
                self.errors[attack] = f"{type(error).__name__}: {error}"
        summaries = []
        record_dp_accounting = getattr(self, "record_dp_accounting", None)
        for result in self.results:
            fpr_targets = (
                _EXACT_BATCH_REPORTED_FPR_TARGETS
                if result.name in self.exact_batch_membership_attacks
                else (0.1, 0.01, 0.001)
            )
            summary = result.to_summary(fpr_targets=fpr_targets)
            if record_dp_accounting is not None:
                epsilon = float(
                    record_dp_accounting["epsilon_upper_bound"]
                )
                delta = float(record_dp_accounting["delta"])
                summary["record_dp_theoretical_tpr_upper_bounds"] = {
                    str(target): min(
                        1.0,
                        (
                            math.exp(epsilon) * float(target) + delta
                            if epsilon < 700
                            else math.inf
                        ),
                    )
                    for target in fpr_targets
                }
            paper_balanced = (
                None
                if result.name in self.exact_batch_membership_attacks
                else self._paper_balanced_attack_summary(result)
            )
            if paper_balanced is not None:
                summary["paper_balanced_evaluation"] = paper_balanced
            summaries.append(summary)
        periodic_metric_rows = self._write_periodic_attack_metrics()
        www_candidate_summary = None
        if self.www_candidate_scoring:
            try:
                www_candidate_summary = self._write_www_candidate_artifacts()
            except Exception as error:
                logger.exception("WWW candidate scoring finalization failed")
                self.errors["www_candidate_scoring"] = (
                    f"{type(error).__name__}: {error}"
                )
                www_candidate_summary = {
                    "status": "error",
                    "error": self.errors["www_candidate_scoring"],
                }
        signal_health_enabled = bool(
            self.config.get("fedmia_signal_health_check", False)
        )
        degenerate_fedmia = [
            item["attack"]
            for item in summaries
            if item["attack"]
            in {
                "fedmia_loss",
                "fedmia_cosine",
            }
            and item["score_degenerate"]
        ]
        if signal_health_enabled:
            for attack in degenerate_fedmia:
                self.errors[f"health:{attack}"] = (
                    "Degenerate FedMIA scores: all evaluation candidates received "
                    "the same score."
                )
        audit_health = {
            "enabled": signal_health_enabled,
            "passed": not degenerate_fedmia,
            "degenerate_fedmia_attacks": degenerate_fedmia,
        }
        candidate_labels = self.labels.detach().cpu().long()
        candidate_membership = self.membership.detach().cpu().long()
        histogram_width = int(candidate_labels.max().item()) + 1
        member_histogram = torch.bincount(
            candidate_labels[candidate_membership == 1], minlength=histogram_width
        ).tolist()
        nonmember_histogram = torch.bincount(
            candidate_labels[candidate_membership == 0], minlength=histogram_width
        ).tolist()
        member_count = int((candidate_membership == 1).sum())
        nonmember_count = int((candidate_membership == 0).sum())
        histograms_exactly_matched = member_histogram == nonmember_histogram
        distributions_exactly_matched = all(
            member * nonmember_count == nonmember * member_count
            for member, nonmember in zip(member_histogram, nonmember_histogram)
        )
        label_tv_distance = 0.5 * sum(
            abs(member / member_count - nonmember / nonmember_count)
            for member, nonmember in zip(member_histogram, nonmember_histogram)
        )
        paper_balanced_evaluation = getattr(
            self, "paper_balanced_evaluation", None
        )
        paper_balanced_summary = (
            None
            if paper_balanced_evaluation is None
            else {
                key: value
                for key, value in paper_balanced_evaluation.items()
                if key != "candidate_indices"
            }
        )
        www_validation = None
        if self.defense_name == "www" and self.defense_config.get("release_private_diagnostics", False):
            try:
                validation = validate_www_attack_relationships(
                    results=[
                        result
                        for result in self.results
                        if result.name
                        not in self.exact_batch_membership_attacks
                    ],
                    users=self.users,
                    candidate_labels=candidate_labels,
                    candidate_membership=candidate_membership,
                    candidate_client_ids=self.candidate_client_ids,
                    candidate_local_indices=self.candidate_local_indices,
                    output_dir=self.results_dir,
                    top_fraction=float(
                        self.defense_config.get(
                            "www_validation_top_fraction", 0.2
                        )
                    ),
                )
                www_validation = {
                    key: value
                    for key, value in validation.items()
                    if key != "relationships"
                }
            except Exception as error:
                logger.exception("WWW specificity validation failed")
                self.errors["www_validation"] = (
                    f"{type(error).__name__}: {error}"
                )
                www_validation = {
                    "status": "error",
                    "error": self.errors["www_validation"],
                }
        with open(
            os.path.join(self.results_dir, "summary.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(
                {
                    "target_client_id": self.target_client_id,
                    "requested_audit_client_ids": self.requested_audit_client_ids,
                    "audit_client_ids": self.audit_client_ids,
                    "audit_scope": (
                        "pooled_clients"
                        if self.pooled_client_audit
                        else "single_client"
                    ),
                    "defense": self.defense_name,
                    "exact_batch_skipped_rounds": self.exact_batch_skipped_rounds,
                    "cofedmid": self._cofedmid_metadata(),
                    "defense_validation_split_sha256": getattr(self.users[0], "defense_validation_split_sha256", None),
                    "federated_method": self.federated_method,
                    "model_type": self.model_type,
                    "audit_view": self.audit_view,
                    "signal_storage": {
                        "mode": self.signal_storage,
                        "collection_mode": "attack_on_demand",
                        "signals_file_written": self.signal_storage != "none",
                        "default_audit_interval": self.audit_interval,
                        "attack_audit_intervals": {
                            attack: self._audit_interval_for_attack(attack)
                            for attack in self.attacks
                        },
                        "attack_schedules": {
                            attack: self._attack_schedule_metadata(attack)
                            for attack in self.attacks
                        },
                        "stored_rounds": [
                            int(observation["round"])
                            for observation in self.observations
                        ],
                        "stored_client_counts": [
                            int(observation["client_ids"].numel())
                            for observation in self.observations
                        ],
                        "stored_observation_fields": sorted(
                            {
                                key
                                for observation in self.observations
                                for key in observation
                            }
                        ),
                        "periodic_metrics_file": (
                            "attack_round_metrics.csv"
                            if periodic_metric_rows
                            else None
                        ),
                        "periodic_metric_rows": periodic_metric_rows,
                        "exact_batch_membership": {
                            "enabled": bool(
                                self.exact_batch_membership_attacks
                            ),
                            "attacks": sorted(
                                self.exact_batch_membership_attacks
                            ),
                            "membership_definition": (
                                self._exact_batch_membership_definition()
                            ),
                            "nonmember_training_exposure": "never_trained",
                            "label_matching_mode": (
                                "exact_batch_histogram_ratio"
                            ),
                            "nonmember_to_member_ratio": (
                                self.exact_batch_nonmember_ratio
                            ),
                            "reported_fpr_targets": list(
                                _EXACT_BATCH_REPORTED_FPR_TARGETS
                            ),
                            "candidate_selection_file": (
                                "exact_batch_candidate_selection.pt"
                                if self.exact_batch_candidate_selections
                                else None
                            ),
                            "stored_rounds": [
                                int(observation["round"])
                                for observation in self.exact_batch_observations
                            ],
                        },
                    },
                    "few_shot": {
                        "enabled": self.few_shot,
                        "fpl_shots": self.fpl_shots,
                        "allow_partial_client_audit": (
                            self.allow_partial_client_audit
                        ),
                        "skipped_audit_clients": self.skipped_audit_clients,
                    },
                    "threat_model": {
                        "protocol_messages": self.audit_view
                        in {"protocol_plus_released_prompts", "full_whitebox"},
                        "fedsgd_client_post_state_source": (
                            "base_minus_learning_rate_times_uploaded_gradient"
                            if self.audit_view
                            == "protocol_plus_released_prompts"
                            and self.federated_method == "fedsgd"
                            else None
                        ),
                        "released_prompt_checkpoints": self.audit_view
                        in {
                            "protocol_plus_released_prompts",
                            "released_prompt",
                            "full_whitebox",
                        },
                        "raw_internal_client_state": self.audit_view == "full_whitebox",
                        "candidate_label_histograms_matched": (
                            self.match_candidate_labels
                            and histograms_exactly_matched
                        ),
                        "candidate_label_distributions_matched": (
                            self.match_candidate_labels
                            and distributions_exactly_matched
                        ),
                        "null_clients_share_candidate_training_labels": bool(
                            self.null_client_candidate_label_overlap
                        ),
                        "null_client_candidate_label_overlap_by_client": (
                            self.null_client_candidate_label_overlap_by_client
                        ),
                    },
                    "candidate_sampling": {
                        "mode": self.candidate_sampling_mode,
                        "full_target_train_members_required": (
                            self.require_full_target_train_members
                        ),
                        "membership_definition": (
                            "global_model_record_membership"
                            if self.candidate_sampling_mode
                            in {
                                "balanced_holdout",
                                "balanced_global_holdout",
                            }
                            else "target_client_membership"
                        ),
                        "nonmember_training_exposure": (
                            "never_trained"
                            if self.candidate_sampling_mode
                            in {
                                "balanced_holdout",
                                "balanced_global_holdout",
                            }
                            else "source_dependent"
                        ),
                        "requested_nonmember_to_member_ratio": (
                            self.nonmember_to_member_ratio
                            if self.candidate_sampling_mode
                            in {"fedmia_mix", "balanced_global_holdout"}
                            else None
                        ),
                        "actual_nonmember_to_member_ratio": (
                            nonmember_count / member_count
                        ),
                        "label_histograms_matched": (
                            self.match_candidate_labels
                            and histograms_exactly_matched
                        ),
                        "label_matching_mode": (
                            "balanced_target_train_test_per_class"
                            if self.candidate_sampling_mode
                            == "balanced_holdout"
                            else "exact_proportional_target_train_global_test"
                            if self.candidate_sampling_mode
                            == "balanced_global_holdout"
                            and not self.require_full_target_train_members
                            else "best_effort_full_target_train_global_test"
                            if self.candidate_sampling_mode
                            == "balanced_global_holdout"
                            else "fedmia_mix_unmatched"
                            if self.candidate_sampling_mode == "fedmia_mix"
                            else "exact_paired_per_client"
                            if self.pooled_client_audit
                            else "exact_proportional"
                            if self.match_candidate_labels
                            else "disabled"
                        ),
                        "label_distributions_matched": (
                            self.match_candidate_labels
                            and distributions_exactly_matched
                        ),
                        "label_total_variation_distance": label_tv_distance,
                        "member_count": member_count,
                        "nonmember_count": nonmember_count,
                        "nonmember_source_priority": self.nonmember_source_priority,
                        "member_label_histogram": member_histogram,
                        "nonmember_label_histogram": nonmember_histogram,
                        "evaluation_views": {
                            "low_fpr": {
                                "member_count": member_count,
                                "nonmember_count": nonmember_count,
                                "fpr_resolution": 1.0 / nonmember_count,
                            },
                            "paper_balanced": paper_balanced_summary,
                        },
                        "candidate_label_support": self.candidate_label_support,
                        "null_client_candidate_label_overlap": (
                            self.null_client_candidate_label_overlap
                        ),
                        "per_client": self.candidate_sampling_by_client,
                    },
                    "audit_health": audit_health,
                    "www_candidate_scoring": www_candidate_summary,
                    "www_validation": www_validation,
                    "record_dp_verification": (
                        None
                        if record_dp_accounting is None
                        else {
                            "privacy_unit": record_dp_accounting[
                                "privacy_unit"
                            ],
                            "epsilon": record_dp_accounting[
                                "epsilon_upper_bound"
                            ],
                            "delta": record_dp_accounting["delta"],
                            "formal_dp_enabled": record_dp_accounting[
                                "formal_dp_enabled"
                            ],
                            "roc_constraint": (
                                "TPR <= min(1, exp(epsilon) * FPR + delta)"
                            ),
                            "scope": "released_model_and_client_uploads",
                            "audit_artifacts_are_private_research_data": True,
                        }
                    ),
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
            writer.writerow(
                (
                    "attack",
                    "sample_index",
                    "audit_client_id",
                    "membership",
                    "score",
                )
            )
            for result in self.results:
                for index, label, score in zip(
                    result.sample_indices.tolist(),
                    result.labels.tolist(),
                    result.scores.tolist(),
                ):
                    writer.writerow(
                        (
                            result.name,
                            index,
                            (
                                self.target_client_id
                                if result.name
                                in self.exact_batch_membership_attacks
                                else int(self.candidate_client_ids[index])
                            ),
                            label,
                            score,
                        )
                    )
        if self.signal_storage != "none":
            torch.save(
                {
                    "candidate_labels": self.labels.detach().cpu(),
                    "candidate_client_ids": self.candidate_client_ids,
                    "membership": self.membership,
                    "observations": self.observations,
                    "exact_batch_observations": self.exact_batch_observations,
                    "storage_mode": self.signal_storage,
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
