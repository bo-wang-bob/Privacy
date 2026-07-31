from __future__ import annotations

import copy
import csv
import json
import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from privacy_attacks.code_poison import run_code_poison_attack
from privacy_attacks.base import AttackResult
from privacy_attacks.features import (
    flatten_state_delta,
    logits_and_representation,
    per_sample_prompt_gradients,
    trainable_names,
)
from privacy_attacks.fedmia import run_fedmia
from privacy_attacks.fedmia_baselines import run_fedmia_baseline
from privacy_attacks.imia import run_imia
from privacy_attacks.model_utils import last_client_states
from privacy_attacks.pipra import run_pipra
from privacy_attacks.promptmia import run_promptmia
from privacy_attacks.promptres import promptres_round_scores, run_promptres
from privacy_attacks.quantile import run_quantile_mia
from privacy_attacks.query_attacks import run_canary, run_yoqo
from privacy_attacks.rmia import run_rmia
from privacy_attacks.transfer import run_transfer_representation_attack
from privacy_attacks.whitebox import run_active_whitebox, run_passive_whitebox
from privacy_defenses import (
    attach_hamp_output_transform,
    attach_output_temperature_transform,
)
from utils.data_loader import group_idx_by_class

logger = logging.getLogger(__name__)

SUPPORTED_ATTACKS = {
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
    "promptres",
}

# These attacks can be evaluated client by client from one shared observation
# trajectory, then combined without treating client identity as membership
# evidence. Active/query attacks remain single-client because their probes must
# be scheduled and accounted for separately for every target user.
POOLED_CLIENT_ATTACKS = {
    "loss_series",
    "grad_cosine",
    "avg_cosine",
    "nasr_passive",
    "fedmia_loss",
    "fedmia_cosine",
    "transfer_representation",
    "rmia",
    "quantile_mia",
    "promptres",
}

_CLIENT_CANDIDATE_FIELDS = {
    "confidence",
    "cosine",
    "gradient_difference",
    "probabilities",
    "representations",
    "promptres",
}
_CANDIDATE_FIELDS = {
    "gradient_signature",
    "candidate_labels",
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
        federated_method: str = "fedavg",
        num_classes: int | None = None,
    ):
        self.config = config or {}
        self.defense_config = dict(defense_config or {"name": "none"})
        self.defense_name = str(self.defense_config.get("name", "none")).lower()
        self.federated_method = str(federated_method).lower()
        self.audit_view = str(
            self.config.get("audit_view", "protocol_plus_released_prompts")
        ).lower()
        if self.audit_view == "protocol_plus_queries":
            self.audit_view = "protocol_plus_released_prompts"
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
        self.signal_storage = str(
            self.config.get("signal_storage", "compact")
        ).lower()
        if self.signal_storage not in {"none", "compact", "full"}:
            raise ValueError("audit.signal_storage must be none, compact, or full.")
        full_signals = self.signal_storage == "full"
        requested_attacks = set(self.attacks)
        self._needs_confidence = full_signals or bool(
            requested_attacks
            & {"blackbox_loss", "loss_series", "nasr_passive", "fedmia_loss"}
        )
        self._needs_cosine = full_signals or bool(
            requested_attacks
            & {"grad_cosine", "avg_cosine", "nasr_passive", "fedmia_cosine"}
        )
        self._needs_promptres = full_signals or "promptres" in requested_attacks
        self._needs_whitebox_features = full_signals or "nasr_passive" in requested_attacks
        self._needs_probabilities = full_signals or bool(
            requested_attacks & {"nasr_passive", "rmia", "quantile_mia"}
        )
        self._needs_representations = full_signals or bool(
            requested_attacks
            & {"nasr_passive", "transfer_representation", "quantile_mia"}
        )
        self._needs_client_states = full_signals or bool(
            requested_attacks & {"pipra", "imia", "yoqo", "canary", "promptmia"}
        )
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
        self.model = copy.deepcopy(model).to(device)
        self.initial_prompt_state = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if self.defense_name == "hamp":
            attach_hamp_output_transform(
                self.model,
                float(self.defense_config.get("hamp_output_temperature", 4.0)),
            )
        elif self.defense_name in {"local_ggeur", "mirage", "veil"} and bool(
            self.defense_config.get("local_ggeur_calibrate_observations", False)
        ):
            margin = self.defense_config.get("local_ggeur_output_margin")
            attach_output_temperature_transform(
                self.model,
                float(self.defense_config.get("local_ggeur_output_temperature", 4.0)),
                margin=None if margin is None else float(margin),
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
        self.audit_interval = int(self.config.get("audit_interval", 1))
        self.audit_batch_size = int(self.config.get("audit_batch_size", 64))
        self.calibration_fraction = float(self.config.get("calibration_fraction", 0.5))
        self.match_candidate_labels = bool(
            self.config.get("match_candidate_labels", False)
        )
        self.candidate_sampling_mode = str(
            self.config.get("candidate_sampling", "legacy")
        ).lower()
        if self.candidate_sampling_mode not in {"legacy", "fedmia_mix"}:
            raise ValueError(
                "audit.candidate_sampling must be legacy or fedmia_mix."
            )
        self.nonmember_to_member_ratio = float(
            self.config.get("nonmember_to_member_ratio", 1.0)
        )
        if self.nonmember_to_member_ratio <= 0:
            raise ValueError(
                "audit.nonmember_to_member_ratio must be positive."
            )
        if self.candidate_sampling_mode == "fedmia_mix" and self.match_candidate_labels:
            raise ValueError(
                "FedMIA mix sampling reproduces the reference evaluation and "
                "therefore requires match_candidate_labels=false."
            )
        if self.audit_interval <= 0 or self.audit_batch_size <= 0:
            raise ValueError("audit_interval and audit_batch_size must be positive.")
        self.observations: list[dict] = []
        self.results = []
        self.errors: dict[str, str] = {}

        if self.enabled:
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
        return torch.cat(image_parts), torch.cat(label_parts)

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

    def should_observe(self, round_index: int) -> bool:
        return self.enabled and round_index % self.audit_interval == 0

    @torch.no_grad()
    def _candidate_outputs(
        self,
        model: torch.nn.Module,
        require_representation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Evaluate large audit candidate pools without one oversized CLIP batch."""
        logits_parts = []
        representation_parts = []
        loss_parts = []
        for start in range(0, self.labels.numel(), self.audit_batch_size):
            stop = start + self.audit_batch_size
            images = self.images[start:stop].to(self.device)
            labels = self.labels[start:stop].to(self.device)
            if require_representation:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample gradients while bounding resident GPU images."""
        gradient_parts = []
        signature_parts = []
        loss_parts = []
        for start in range(0, self.labels.numel(), self.audit_batch_size):
            stop = start + self.audit_batch_size
            gradients, signatures, losses = per_sample_prompt_gradients(
                model,
                self.images[start:stop].to(self.device),
                self.labels[start:stop].to(self.device),
                parameter_names,
            )
            gradient_parts.append(gradients)
            signature_parts.append(signatures)
            loss_parts.append(losses)
        return (
            torch.cat(gradient_parts),
            torch.cat(signature_parts),
            torch.cat(loss_parts),
        )

    @staticmethod
    def _clone_prompt_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in state.items()}

    def _global_prompt_public_state(
        self, source: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Build an explicit global-only model from protocol-visible parameters.

        Initial private parameters are not a protocol message and must not be
        mixed with a later global prompt.  FedOTP duplicates the public global
        prompt into its two OT slots; additive private adapters are neutralized.
        """
        state = self._clone_prompt_state(self.initial_prompt_state)
        global_context = None
        for name, tensor in source.items():
            if name.endswith("global_ctx"):
                state[name] = tensor.detach().clone()
                global_context = tensor.detach().clone()
        if self.federated_method == "fedotp" and global_context is not None:
            for name in state:
                if name.endswith("local_ctx"):
                    state[name] = global_context.clone()
        elif self.federated_method in {"fedpgp", "dpfpl"}:
            for name in state:
                if name.endswith(("local_ctx", "fedpgp_u", "fedpgp_v")):
                    state[name] = torch.zeros_like(state[name])
        return state

    def _observable_base_state(
        self, base_state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.audit_view != "full_whitebox" and self.federated_method in {
            "dpfpl",
            "fedotp",
            "fedpgp",
        }:
            return self._global_prompt_public_state(base_state)
        return base_state

    def _observable_client_state(
        self,
        user_id: int,
        updated_state: dict[str, torch.Tensor],
        released_states: dict[int, dict[str, torch.Tensor]] | None,
    ) -> dict[str, torch.Tensor]:
        if self.audit_view == "full_whitebox":
            return updated_state
        if (
            self.audit_view == "protocol_plus_released_prompts"
            and self.federated_method in {"fedavg", "promptfl"}
        ):
            # The full FedAvg model update is itself a protocol message.
            return updated_state
        if (
            self.audit_view == "protocol_plus_released_prompts"
            and self.federated_method in {"fedotp", "fedpgp"}
        ):
            # The server sees each client's global prompt update, but FedOTP's
            # local_ctx and FedPGP's low-rank U/V never leave that client.
            return self._global_prompt_public_state(updated_state)
        if not released_states:
            raise ValueError("Released prompt state is required by this audit view.")
        released = released_states.get(user_id, released_states.get(0))
        if released is None:
            raise ValueError(f"No released prompt is available for client {user_id}.")
        if self.federated_method == "dpfpl":
            return self._global_prompt_public_state(released)
        if self.federated_method in {"fedotp", "fedpgp"}:
            return self._global_prompt_public_state(released)
        return released

    def observe_round(
        self,
        round_index: int,
        base_state: dict[str, torch.Tensor],
        updated_states: dict[int, dict[str, torch.Tensor]],
        selected_ids: list[int],
        base_states: dict[int, dict[str, torch.Tensor]] | None = None,
        protocol_messages: dict[int, dict] | None = None,
        released_states: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> None:
        if not self.should_observe(round_index):
            return
        names = trainable_names(self.model)
        observable_names = names
        if self.audit_view != "full_whitebox" and self.federated_method in {
            "dpfpl",
            "fedotp",
            "fedpgp",
        }:
            observable_names = [name for name in names if name.endswith("global_ctx")]
        needs_gradients = (
            self._needs_cosine
            or self._needs_whitebox_features
            or self._needs_promptres
        )
        sample_gradients = None
        signatures = None
        if needs_gradients:
            self.model.load_state_dict(
                self._observable_base_state(base_state), strict=False
            )
            self.model.eval()
            sample_gradients, signatures, _ = self._candidate_gradients(
                self.model, observable_names
            )

        confidence = []
        cosine = []
        gradient_difference = []
        probabilities = []
        representations = []
        promptres_updates = []
        promptres_gradients = []
        observable_states = {}
        for user_id in selected_ids:
            state = self._observable_client_state(
                user_id, updated_states[user_id], released_states
            )
            if self._needs_client_states:
                observable_states[user_id] = state
            if needs_gradients:
                if sample_gradients is None:
                    raise AssertionError("Gradient-dependent audit lost its gradients.")
                client_base = (
                    base_state if base_states is None else base_states[user_id]
                )
                if self.audit_view == "released_prompt":
                    update = torch.ones(1)
                elif (
                    self.audit_view == "protocol_plus_released_prompts"
                    and protocol_messages is not None
                    and user_id in protocol_messages
                ):
                    message = protocol_messages[user_id]
                    tensors = message.get("tensors", {})
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
                if self._needs_cosine:
                    cosine.append(
                        (compared_gradients @ update)
                        / (sample_norm * update_norm)
                    )
                if self._needs_promptres:
                    promptres_updates.append(update)
                    promptres_gradients.append(compared_gradients)
                if self._needs_whitebox_features:
                    update_sq = update.square().sum()
                    gradient_difference.append(
                        update_sq
                        - (update.unsqueeze(0) - compared_gradients)
                        .square()
                        .sum(dim=1)
                    )

            needs_outputs = (
                self._needs_confidence
                or self._needs_probabilities
                or self._needs_representations
            )
            if needs_outputs:
                self.model.load_state_dict(state, strict=False)
                logits, representation, losses = self._candidate_outputs(
                    self.model,
                    require_representation=self._needs_representations,
                )
                if self._needs_confidence:
                    confidence.append(-losses)
                if self._needs_probabilities:
                    probabilities.append(torch.softmax(logits, dim=1))
                if self._needs_representations:
                    if representation is None:
                        raise AssertionError(
                            "Representation-dependent audit lost its features."
                        )
                    representations.append(representation)

        observation = {
            "round": int(round_index),
            "client_ids": torch.tensor(selected_ids, dtype=torch.long),
        }
        if self._needs_confidence:
            observation["confidence"] = torch.stack(confidence)
        if self._needs_cosine:
            observation["cosine"] = torch.stack(cosine)
        if self._needs_whitebox_features:
            if signatures is None:
                raise AssertionError("White-box audit lost its gradient signatures.")
            observation["gradient_difference"] = torch.stack(gradient_difference)
            observation["gradient_signature"] = signatures
            observation["candidate_labels"] = self.labels.detach().cpu().clone()
        if self._needs_probabilities:
            observation["probabilities"] = torch.stack(probabilities)
        if self._needs_representations:
            observation["representations"] = torch.stack(representations)
        if self._needs_promptres:
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
        if self._needs_client_states:
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
        self.observations.append(observation)
        logger.debug("Collected privacy signals for round %s", round_index)

    def _run_fedmia_signal(
        self,
        measurement: str,
        aggregation: str,
        tail: str,
    ) -> AttackResult:
        if not self.pooled_client_audit:
            return run_fedmia(
                self.observations,
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
                for observation in self.observations
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
                self.observations,
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
        name = "fedmia_loss" if measurement == "confidence" else "fedmia_cosine"
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
            self.observations, indices
        )
        membership = self.membership[indices]
        labels = self.labels[indices]
        seed = self.seed + 1009 * client_id
        if attack in {"loss_series", "grad_cosine", "avg_cosine"}:
            return run_fedmia_baseline(
                observations,
                membership,
                client_id,
                attack,
                self.config.get("fedmia_baseline_single_round", "last"),
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

    def _run(self, attack: str, final_model, final_state):
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
        if attack in {
            "blackbox_loss",
            "loss_series",
            "grad_cosine",
            "avg_cosine",
        }:
            return run_fedmia_baseline(
                self.observations,
                self.membership,
                self.target_client_id,
                attack,
                self.config.get("fedmia_baseline_single_round", "last"),
            )
        if attack == "promptres":
            return run_promptres(
                self.observations,
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
                probe_cycles=int(self.config.get("active_probe_cycles", 3)),
                calibration_fraction=self.calibration_fraction,
                seed=self.seed,
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
        elif self.defense_name in {"local_ggeur", "mirage", "veil"}:
            margin = self.defense_config.get("local_ggeur_output_margin")
            attach_output_temperature_transform(
                final_model,
                float(self.defense_config.get("local_ggeur_output_temperature", 4.0)),
                margin=None if margin is None else float(margin),
            )
        for attack in self.attacks:
            try:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                self.results.append(self._run(attack, final_model, final_state))
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as error:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                logger.exception("Membership attack %s failed", attack)
                self.errors[attack] = f"{type(error).__name__}: {error}"
        summaries = [result.to_summary() for result in self.results]
        signal_health_enabled = bool(
            self.config.get("fedmia_signal_health_check", False)
        )
        degenerate_fedmia = [
            item["attack"]
            for item in summaries
            if item["attack"] in {"fedmia_loss", "fedmia_cosine"}
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
                    "federated_method": self.federated_method,
                    "audit_view": self.audit_view,
                    "signal_storage": {
                        "mode": self.signal_storage,
                        "signals_file_written": self.signal_storage != "none",
                        "stored_observation_fields": sorted(
                            {
                                key
                                for observation in self.observations
                                for key in observation
                            }
                        ),
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
                        "personalized_public_model_projection": (
                            "global_only_neutral_private_parameters"
                            if self.audit_view != "full_whitebox"
                            and self.federated_method in {"fedotp", "fedpgp", "dpfpl"}
                            else None
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
                        "requested_nonmember_to_member_ratio": (
                            self.nonmember_to_member_ratio
                            if self.candidate_sampling_mode == "fedmia_mix"
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
                            "fedmia_mix_unmatched"
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
                        "candidate_label_support": self.candidate_label_support,
                        "null_client_candidate_label_overlap": (
                            self.null_client_candidate_label_overlap
                        ),
                        "per_client": self.candidate_sampling_by_client,
                    },
                    "audit_health": audit_health,
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
                            int(self.candidate_client_ids[index]),
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
