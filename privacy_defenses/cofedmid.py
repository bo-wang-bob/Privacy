"""Paper-defined CoFedMID adapted to indexed one-batch FedSGD.

The coalition coordinator handles label IDs and aggregation weights only.
Client losses, EXP3 state, and unperturbed local models stay in the simulation's
client-side training path, outside protocol messages and the attack view.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from utils.privacy_accounting import private_generator


DEFAULTS = {
    "cofedmid_clients": "all",
    "cofedmid_partition": True,
    "cofedmid_compensation": True,
    "cofedmid_perturbation": True,
    "cofedmid_max_class_ratio": 0.5,
    "cofedmid_min_class_ratio": 0.2,
    "cofedmid_coverage": "strict",
    "cofedmid_init_round": 10,
    "cofedmid_intervals": 10,
    "cofedmid_recycle_ratio": 0.05,
    "cofedmid_entropy_weight": 0.005,
    "cofedmid_exp3_gamma": 0.2,
    "cofedmid_exp3_learning_rate": 0.3,
    "cofedmid_reward_history": 20,
    "cofedmid_noise_std": 0.01,
    "cofedmid_noise_space": "parameter",
    "cofedmid_perturb_ratio": 0.2,
    "cofedmid_reproducible_noise": False,
    "cofedmid_validation_fraction": 0.1,
}


def coalition_ids(config: dict, total_users: int) -> list[int]:
    value = config.get("cofedmid_clients", "all")
    if value == "all":
        value = list(range(total_users))
    if not isinstance(value, list) or any(
        isinstance(x, bool) or not isinstance(x, int) for x in value
    ):
        raise ValueError("cofedmid_clients must be 'all' or a list of client IDs.")
    if len(value) != len(set(value)) or any(x < 0 or x >= total_users for x in value):
        raise ValueError("cofedmid_clients contains duplicate or invalid IDs.")
    if len(value) < 2:
        raise ValueError("CoFedMID requires at least two coalition clients.")
    return sorted(value)


def validate_cofedmid(config: dict, total_users: int, sample_users: int) -> None:
    enabled = str(config.get("name", "none")).lower() == "cofedmid"
    fraction = float(config.get("cofedmid_validation_fraction", 0.1 if enabled else 0))
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("cofedmid_validation_fraction must be in [0, 1).")
    if not enabled:
        return
    cfg = {**DEFAULTS, **config}
    cfg["cofedmid_validation_fraction"] = fraction
    coalition_ids(cfg, total_users)
    # Complete participation makes the noise cancellation contract explicit.
    if sample_users != total_users:
        raise ValueError("CoFedMID currently requires full client participation.")
    for key in (
        "cofedmid_partition", "cofedmid_compensation", "cofedmid_perturbation",
        "cofedmid_reproducible_noise",
    ):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a boolean.")
    if cfg["cofedmid_compensation"] and fraction <= 0:
        raise ValueError("CoFedMID compensation requires an independent validation split.")
    for key in (
        "cofedmid_max_class_ratio", "cofedmid_min_class_ratio",
        "cofedmid_exp3_gamma", "cofedmid_recycle_ratio", "cofedmid_perturb_ratio",
    ):
        value = float(cfg[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be in [0, 1].")
        cfg[key] = value
    if not 0 < cfg["cofedmid_min_class_ratio"] <= cfg["cofedmid_max_class_ratio"]:
        raise ValueError("CoFedMID class ratios require 0 < min <= max.")
    for key in (
        "cofedmid_noise_std", "cofedmid_entropy_weight",
        "cofedmid_exp3_learning_rate",
    ):
        value = float(cfg[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative.")
        cfg[key] = value
    for key, minimum in (
        ("cofedmid_init_round", 0), ("cofedmid_intervals", 1),
        ("cofedmid_reward_history", 2),
    ):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    for key in ("cofedmid_max_classes", "cofedmid_min_classes"):
        if key in cfg and (
            isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer.")
    if cfg["cofedmid_noise_space"] not in {"parameter", "gradient"}:
        raise ValueError("cofedmid_noise_space must be parameter or gradient.")
    if cfg["cofedmid_coverage"] not in {"strict", "maximize"}:
        raise ValueError("cofedmid_coverage must be strict or maximize.")
    config.update(cfg)


def dataset_labels(dataset) -> torch.Tensor:
    """Read labels without encoding images when the dataset exposes metadata."""
    if isinstance(dataset, Subset):
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return dataset_labels(dataset.dataset)[indices]
    if isinstance(dataset, ConcatDataset):
        return torch.cat([dataset_labels(part) for part in dataset.datasets])
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].detach().cpu().long().reshape(-1)
    for attr in ("targets", "labels", "_labels", "y"):
        values = getattr(dataset, attr, None)
        if values is not None and len(values) == len(dataset):
            return torch.as_tensor(values, dtype=torch.long).reshape(-1)
    return torch.tensor([int(dataset[i][1]) for i in range(len(dataset))], dtype=torch.long)


def reserve_validation(test_sets: list, fraction: float, seed: int):
    """Remove a shared stratified validation set from all audit/evaluation views."""
    if fraction == 0:
        return test_sets, None, None
    pool = ConcatDataset(test_sets)
    labels = dataset_labels(pool)
    generator = torch.Generator().manual_seed(seed + 42017)
    held_out = []
    for label in labels.unique(sorted=True).tolist():
        indices = (labels == label).nonzero().flatten()
        count = min(indices.numel() - 1, max(1, int(indices.numel() * fraction)))
        if count > 0:
            order = torch.randperm(indices.numel(), generator=generator)
            held_out.extend(indices[order[:count]].tolist())
    if not held_out:
        raise ValueError("No independent CoFedMID validation samples can be reserved.")
    held_out.sort()
    held_set = set(held_out)
    evaluation_sets, evaluation_indices = [], []
    offset = 0
    for dataset in test_sets:
        kept = [i for i in range(len(dataset)) if i + offset not in held_set]
        evaluation_sets.append(Subset(dataset, kept))
        evaluation_indices.append(kept)
        offset += len(dataset)
    manifest = {
        "source": "original_independent_evaluation",
        "index_convention": (
            "validation: original client-ordered concatenation; "
            "evaluation: original per-client positions"
        ),
        "seed": seed,
        "fraction": fraction,
        "validation_indices": held_out,
        "evaluation_indices_by_client": evaluation_indices,
        "original_sizes": [len(ds) for ds in test_sets],
        "validation_samples": len(held_out),
        "evaluation_samples": len(pool) - len(held_out),
    }
    manifest["split_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    return evaluation_sets, Subset(pool, held_out), manifest


def assign_classes(
    num_classes: int, clients: list[int], size: int, generator: torch.Generator
) -> dict[int, set[int]]:
    """Balanced class incidence with O(classes * clients^2) bounded work."""
    d = len(clients)
    if not d or not 1 <= size <= num_classes:
        raise ValueError("Invalid CoFedMID class allocation dimensions.")
    sets = [set() for _ in clients]
    loads = [0] * d
    overlaps = [[0] * d for _ in clients]
    repeats, extra = divmod(d * size, num_classes)
    for pos, label in enumerate(torch.randperm(num_classes, generator=generator).tolist()):
        selected = []
        available = torch.randperm(d, generator=generator).tolist()
        max_overlap, sum_overlap = [0] * d, [0] * d
        for _ in range(repeats + (pos < extra)):
            chosen = min(
                available,
                key=lambda i: (loads[i], max_overlap[i], sum_overlap[i]),
            )
            for other in selected:
                overlaps[chosen][other] += 1
                overlaps[other][chosen] += 1
            available.remove(chosen)
            for other in available:
                max_overlap[other] = max(max_overlap[other], overlaps[other][chosen])
                sum_overlap[other] += overlaps[other][chosen]
            selected.append(chosen)
            loads[chosen] += 1
            sets[chosen].add(label)
    if any(len(s) != size for s in sets):
        raise AssertionError("CoFedMID class allocation lost equal row sizes.")
    return dict(zip(clients, sets))


class Exp3:
    def __init__(self, intervals: int, exploration: float, learning_rate: float, history: int):
        self.log_weights = torch.zeros(intervals, dtype=torch.float64)
        self.exploration = exploration
        self.learning_rate = learning_rate
        self.history = deque(maxlen=history)
        self.bounds: torch.Tensor | None = None

    @property
    def probabilities(self) -> torch.Tensor:
        return (
            (1 - self.exploration) * self.log_weights.softmax(0)
            + self.exploration / self.log_weights.numel()
        )

    @staticmethod
    def normalize_losses(losses: torch.Tensor) -> torch.Tensor:
        values = losses.double()
        return (values - values.min()) / (values.max() - values.min()).clamp_min(1e-12)

    def initialize(self, losses: torch.Tensor) -> None:
        values = self.normalize_losses(losses)
        quantiles = torch.linspace(
            0, 1, self.log_weights.numel() + 1, dtype=torch.float64
        )
        self.bounds = torch.quantile(values, quantiles)
        self.bounds[0], self.bounds[-1] = 0.0, 1.0

    def choose(self, losses: torch.Tensor, generator: torch.Generator):
        if self.bounds is None:
            raise RuntimeError("CoFedMID EXP3 intervals have not been initialized.")
        probs = self.probabilities
        arm = int(torch.multinomial(probs, 1, generator=generator))
        values = self.normalize_losses(losses)
        # Right insertion assigns ties once, including the maximum loss.
        assignments = torch.bucketize(values.contiguous(), self.bounds[1:-1], right=True)
        return arm, float(probs[arm]), assignments == arm

    def update(self, reward: float, arm: int | None, probability: float | None) -> float:
        if not math.isfinite(reward):
            raise ValueError("CoFedMID validation reward is not finite.")
        self.history.append(reward)
        lo, hi = torch.quantile(
            torch.tensor(list(self.history), dtype=torch.float64),
            torch.tensor([0.2, 0.8], dtype=torch.float64),
        ).tolist()
        normalized = (
            0.0 if hi - lo < 1e-12
            else max(-1.0, min(1.0, 2 * (reward - lo) / (hi - lo) - 1))
        )
        if arm is not None:
            denominator = max(float(probability) * self.log_weights.numel(), 1e-12)
            self.log_weights[arm] += self.learning_rate * normalized / denominator
            self.log_weights -= self.log_weights.max()
        return normalized


def training_loss(
    logits: torch.Tensor, labels: torch.Tensor,
    recycled: torch.Tensor, entropy_weight: float,
) -> torch.Tensor:
    """Mean over the actual batch of CE + recycled-only KL(q||p) - mu H(p)."""
    loss = F.cross_entropy(logits, labels)
    if recycled.any():
        log_p = F.log_softmax(logits[recycled], dim=1)
        p = log_p.exp()
        y = labels[recycled, None]
        confidence = p.gather(1, y).detach()
        targets = ((1 - confidence) / (logits.shape[1] - 1)).expand_as(p).clone()
        targets.scatter_(1, y, confidence)
        regularizer = F.kl_div(log_p, targets, reduction="none").sum(1)
        regularizer += entropy_weight * (p * log_p).sum(1)
        loss = loss + regularizer.sum() / labels.numel()
    return loss


def perturb_uploads(
    states: dict, weights: dict[int, float], clients: list[int],
    ratio: float, sigma: float, generator: torch.Generator, *,
    gradient_scale: float = 1.0,
) -> dict:
    """Add one projected scalar per client to a common global parameter tail."""
    if len(clients) < 2 or set(clients) - set(states):
        raise ValueError("CoFedMID noise requires the entire coalition's uploads.")
    names = list(states[clients[0]])
    for client in clients:
        if list(states[client]) != names or any(
            states[client][n].shape != states[clients[0]][n].shape for n in names
        ):
            raise ValueError("CoFedMID uploads must have identical parameter scopes.")
    w = torch.tensor([weights[i] for i in clients], dtype=torch.float64)
    if not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError("CoFedMID aggregation weights must be finite and positive.")
    delta = torch.randn(len(clients), generator=generator, dtype=torch.float64) * sigma
    delta -= w * (w @ delta) / (w @ w)
    delta *= gradient_scale
    total = sum(states[clients[0]][n].numel() for n in names)
    count = math.floor(total * ratio)
    start, offset = total - count, 0
    mask = []
    residual = 0.0
    for name in names:
        width = states[clients[0]][name].numel()
        left = max(0, start - offset)
        if left < width:
            mask.append({"parameter": name, "start": left, "count": width - left})
            # Compute the actual rounded update residual without retaining a
            # second copy of every client's full upload.
            weighted_change = torch.zeros(width - left, dtype=torch.float64)
            for position, client in enumerate(clients):
                tensor = states[client][name]
                flat = tensor.reshape(-1).clone()
                before = flat[left:].detach().cpu().double()
                flat[left:] += float(delta[position])
                if not torch.isfinite(flat).all():
                    raise ValueError("CoFedMID noise overflowed the uploaded parameter dtype.")
                weighted_change += float(w[position]) * (
                    flat[left:].detach().cpu().double() - before
                )
                states[client][name] = flat.reshape_as(tensor)
            residual = max(residual, float(weighted_change.abs().max()))
        offset += width
    return {
        "mask": mask, "perturbed_parameters": count,
        "weighted_noise_residual_max": residual,
    }


class CoFedMID:
    def __init__(
        self, config: dict, total_users: int, num_classes: int,
        total_rounds: int, seed: int,
    ):
        self.config = {**DEFAULTS, **config}
        self.clients = coalition_ids(self.config, total_users)
        self.num_classes, self.total_rounds, self.seed = num_classes, total_rounds, seed
        self.assignments: dict[int, set[int]] = {}
        self.round_index = -1
        self.bandits: dict[int, Exp3] = {}
        self.labels: dict[int, torch.Tensor] = {}
        self.exposures: dict[int, torch.Tensor] = {}
        self.recycled_exposures: dict[int, torch.Tensor] = {}
        self.scored_exposures: dict[int, torch.Tensor] = {}
        self.rows: list[dict] = []
        self.noise_rows: list[dict] = []
        self.validation_manifest = None
        self.validation_loader = None
        self._before_validation = None
        self.last_noise_metadata: dict = {}

    def generator(self, client: int, round_index: int, offset: int = 0):
        return torch.Generator().manual_seed(
            self.seed + 1000003 * client + 1009 * round_index + offset
        )

    def prepare(self, selected_ids: list[int], round_index: int):
        if set(self.clients) - set(selected_ids):
            raise ValueError("CoFedMID requires every coalition member in each round.")
        cfg, n = self.config, self.num_classes
        floor = math.ceil(n / len(self.clients)) if cfg["cofedmid_coverage"] == "strict" else 1
        maximum = cfg.get(
            "cofedmid_max_classes", math.ceil(n * cfg["cofedmid_max_class_ratio"])
        )
        minimum = cfg.get(
            "cofedmid_min_classes", math.ceil(n * cfg["cofedmid_min_class_ratio"])
        )
        maximum, minimum = min(n, max(floor, maximum)), min(n, max(floor, minimum))
        if minimum > maximum:
            raise ValueError("CoFedMID minimum class count exceeds maximum.")
        progress = round_index / max(1, self.total_rounds - 1)
        size = max(minimum, round(maximum - (maximum - minimum) * progress))
        if not cfg["cofedmid_partition"]:
            size = n
        self.assignments = assign_classes(
            n, self.clients, size, self.generator(0, round_index, 211)
        )
        self.round_index = round_index
        self._before_validation = None

    @staticmethod
    def losses(model, loader, device) -> torch.Tensor:
        was_training = model.training
        model.eval()
        values = []
        try:
            with torch.no_grad():
                for inputs, labels in loader:
                    loss = F.cross_entropy(
                        model(inputs.to(device)), labels.to(device), reduction="none"
                    )
                    values.append(loss.detach().cpu())
        finally:
            model.train(was_training)
        if not values:
            raise ValueError("CoFedMID loss evaluation received an empty dataset.")
        result = torch.cat(values)
        if not torch.isfinite(result).all():
            raise ValueError("CoFedMID sample losses are not finite.")
        return result

    def train(self, user, model, optimizer, round_index: int, record_step):
        if self.round_index != round_index:
            raise RuntimeError("CoFedMID round has not been prepared.")
        cfg, client = self.config, user.id
        if client not in self.labels:
            self.labels[client] = dataset_labels(user.train_data)
            if ((self.labels[client] < 0) | (self.labels[client] >= self.num_classes)).any():
                raise ValueError("CoFedMID encountered an invalid class label.")
            self.exposures[client] = torch.zeros(user.train_samples, dtype=torch.long)
            self.recycled_exposures[client] = torch.zeros_like(self.exposures[client])
            self.scored_exposures[client] = torch.zeros_like(self.exposures[client])
        labels = self.labels[client]
        assigned = torch.isin(labels, torch.tensor(sorted(self.assignments[client])))
        recycled = torch.zeros_like(assigned)
        completed_round = round_index + 1
        if client not in self.bandits:
            self.bandits[client] = Exp3(
                cfg["cofedmid_intervals"], cfg["cofedmid_exp3_gamma"],
                cfg["cofedmid_exp3_learning_rate"], cfg["cofedmid_reward_history"],
            )
        bandit = self.bandits[client]
        arm, probability = None, None
        compensation = bool(cfg["cofedmid_compensation"])
        local_loader = DataLoader(
            user.train_data, batch_size=user.eval_batch_size,
            collate_fn=user.collate_fn, shuffle=False,
        )
        if compensation:
            if self.validation_loader is None:
                raise ValueError("CoFedMID requires a separate defense validation loader.")
            if self._before_validation is None:
                self._before_validation = float(
                    self.losses(model, self.validation_loader, user.device).mean()
                )
            if completed_round > cfg["cofedmid_init_round"]:
                losses = self.losses(model, local_loader, user.device)
                self.scored_exposures[client] += 1
                if bandit.bounds is None:
                    bandit.initialize(losses)
                generator = self.generator(client, round_index, 307)
                arm, probability, in_interval = bandit.choose(losses, generator)
                candidates = (in_interval & ~assigned).nonzero().flatten()
                cap = math.floor(cfg["cofedmid_recycle_ratio"] * labels.numel())
                order = torch.randperm(candidates.numel(), generator=generator)
                chosen = candidates[order[:cap]]
                recycled[chosen] = True
        eligible = (assigned | recycled).nonzero().flatten()
        if eligible.numel() == 0:
            raise ValueError(
                f"CoFedMID client {client} round {completed_round}: "
                "assigned/recycled pool is empty; "
                f"assigned classes={sorted(self.assignments[client])}."
            )
        samples, recycled_samples = 0, 0
        batches = user.iter_selected_batches(
            eligible, self.generator(client, round_index, 401)
        )
        for inputs, targets, indices in batches:
            batch_recycled = recycled[indices]
            user.last_train_recycled = batch_recycled.clone()
            self.exposures[client][indices] += 1
            self.recycled_exposures[client][indices] += batch_recycled.long()
            samples += indices.numel()
            recycled_samples += int(batch_recycled.sum())
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(
                model(inputs.to(user.device)), targets.to(user.device),
                batch_recycled.to(user.device), cfg["cofedmid_entropy_weight"],
            )
            loss.backward()
            optimizer.step()
            record_step()
        reward, normalized = None, None
        if compensation:
            after_validation = float(
                self.losses(model, self.validation_loader, user.device).mean()
            )
            reward = self._before_validation - after_validation
            normalized = bandit.update(reward, arm, probability)
        self.rows.append({
            "communication_round": completed_round, "client_id": client,
            "assigned_classes": json.dumps(sorted(self.assignments[client])),
            "coalition_class_coverage": len(set().union(*self.assignments.values())),
            "assigned_pool": int(assigned.sum()), "recycled_pool": int(recycled.sum()),
            "training_samples": samples, "recycled_training_samples": recycled_samples,
            "unique_samples_used": int((self.exposures[client] > 0).sum()),
            "arm": arm, "arm_probability": probability, "reward": reward,
            "normalized_reward": normalized,
        })

    def perturb(
        self, states: dict, weights: dict, round_index: int, *,
        learning_rate: float | None = None,
    ):
        cfg = self.config
        sigma = cfg["cofedmid_noise_std"] if cfg["cofedmid_perturbation"] else 0.0
        scale = 1.0
        if learning_rate is not None and cfg["cofedmid_noise_space"] == "parameter":
            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("CoFedMID parameter-space noise requires a positive learning rate.")
            scale = -1.0 / learning_rate
        elif learning_rate is None and cfg["cofedmid_noise_space"] == "gradient":
            raise ValueError("Gradient-space CoFedMID noise requires FedSGD.")
        generator = private_generator(
            torch.device("cpu"), cfg["cofedmid_reproducible_noise"],
            self.seed + 1009 * round_index + 601,
        )
        self.last_noise_metadata = perturb_uploads(
            states, weights, self.clients, cfg["cofedmid_perturb_ratio"],
            sigma, generator, gradient_scale=scale,
        )
        self.noise_rows.append({
            "communication_round": round_index + 1,
            "weighted_noise_residual_max": self.last_noise_metadata["weighted_noise_residual_max"],
            "perturbed_parameters": self.last_noise_metadata["perturbed_parameters"] if sigma else 0,
        })

    def summary(self):
        return {
            "implementation": "paper_modules_fedsgd_v1", "coalition_clients": self.clients,
            "formal_dp": False,
            "configuration": {
                key: value for key, value in self.config.items()
                if key.startswith("cofedmid_")
            },
            "validation_split": None if self.validation_manifest is None else self.validation_manifest["split_sha256"],
            "first_recycling_round": self.config["cofedmid_init_round"] + 1,
            "interval_initialization_model": "global_model_after_init_round_before_first_recycling",
            "loss": "mean(CE_all + recycled * (KL(q||p) - mu*entropy))",
            "upload_view": "perturbed_upload_only", "parameter_mask": self.last_noise_metadata.get("mask", []),
            "max_weighted_noise_residual": max(
                (r["weighted_noise_residual_max"] for r in self.noise_rows),
                default=0.0,
            ),
            "artifacts": ["cofedmid_round_metrics.csv", "cofedmid_noise_metrics.csv", "cofedmid_sample_exposure.pt"],
        }

    def save(self, results_dir: str):
        for name, rows in (
            ("cofedmid_round_metrics.csv", self.rows),
            ("cofedmid_noise_metrics.csv", self.noise_rows),
        ):
            if rows:
                with (Path(results_dir) / name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        torch.save(
            {
                "index_convention": "stable positions in each original client training dataset",
                "training_counts": self.exposures,
                "recycled_training_counts": self.recycled_exposures,
                "loss_scoring_counts": self.scored_exposures,
            },
            Path(results_dir) / "cofedmid_sample_exposure.pt",
        )
