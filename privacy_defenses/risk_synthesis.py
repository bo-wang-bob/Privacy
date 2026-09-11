"""Client-local low-rank token synthesis; empirical defense, no DP claim."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from trainmodel.clip_tokens import token_features
from utils.performance import measure_stage


DEFAULTS = dict(
    replacement_fraction=0.25, noise_scale=0.1, shrinkage=0.5,
    class_rank=5, pooled_rank=16, min_class_samples=3, attempts=2,
    margin_tolerance=0.02, norm_ratio_min=0.5, norm_ratio_max=2.0,
    mode="risk", semantic_filter=True, warmup_rounds=1,
)


def validate_risk_synthesis(config):
    defense = config.get("defense", {})
    if defense.get("name") != "risk_synthesis":
        return
    if config.get("model_type") not in {"clip_adapter", "clip_lora"}:
        raise ValueError("risk_synthesis supports CLIP Adapter/LoRA.")
    if config.get("model_type") == "clip_adapter" and config.get("clip_adapter", {}).get("variant") != "transformer":
        raise ValueError("risk_synthesis requires the transformer Adapter.")
    if config.get("aggregator") != "fedavg":
        raise ValueError("risk_synthesis requires FedAvg original-client membership auditing.")
    if config.get("sample_users", 0) < 2:
        raise ValueError("risk_synthesis requires at least two participating clients.")
    if config.get("code_poison", {}).get("enabled", False):
        raise ValueError("risk_synthesis does not support active code-poison probes.")
    options = {**DEFAULTS, **defense.get("synthesis", {})}
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown synthesis options: {sorted(unknown)}")
    for key in ("replacement_fraction", "shrinkage"):
        if not math.isfinite(float(options[key])) or not 0 <= options[key] <= 1:
            raise ValueError(f"synthesis.{key} must be in [0,1].")
    for key in ("noise_scale", "margin_tolerance", "norm_ratio_min", "norm_ratio_max"):
        if not math.isfinite(float(options[key])) or options[key] < 0:
            raise ValueError(f"synthesis.{key} must be finite and nonnegative.")
    if options["norm_ratio_min"] > options["norm_ratio_max"]:
        raise ValueError("Invalid synthesis norm ratio interval.")
    for key in ("class_rank", "pooled_rank", "min_class_samples", "attempts", "warmup_rounds"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"synthesis.{key} must be a positive integer.")
    if options["min_class_samples"] < 3:
        raise ValueError("synthesis.min_class_samples must be at least 3.")
    if options["mode"] not in {"risk", "shuffled_risk", "mixup"}:
        raise ValueError("synthesis.mode must be risk, shuffled_risk or mixup.")
    if type(options["semantic_filter"]) is not bool:
        raise ValueError("synthesis.semantic_filter must be boolean.")
    defense["synthesis"] = options
    defense.update(target_epsilon=None, delta=None, noise_multiplier=0.0,
                   formal_dp_enabled=False, client_upload_is_private=False)


def low_rank_factor(centered, rank):
    """Factor of the truncated 1/N covariance using a sample Gram matrix."""
    x = centered.double()
    gram = x @ x.T / len(x)
    values, vectors = torch.linalg.eigh(gram)
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    total = float(values.sum())
    numerical_rank = int((values > max(float(values[0]) * 1e-10, 1e-16)).sum())
    used = min(rank, numerical_rank)
    # X.T @ eigenvectors / sqrt(N) = U sqrt(eigenvalues).
    factor = (x.T @ vectors[:, :used] / math.sqrt(len(x))).float().cpu()
    return factor, dict(numerical_rank=numerical_rank, used_rank=used,
                        retained_variance=(float(values[:used].sum()) / total if total else 0.0))


class LocalGeometry:
    def __init__(self, codes, labels, options, device):
        self.codes = codes.detach().cpu().float()
        self.labels = labels.detach().cpu().long()
        self.classes = {}
        residual = torch.empty_like(self.codes)
        for c in self.labels.unique().tolist():
            indices = torch.where(self.labels == c)[0]
            rows = self.codes[indices]
            mean = rows.mean(0)
            centered = rows - mean
            factor, meta = low_rank_factor(centered.to(device), min(options["class_rank"], len(rows)-1))
            self.classes[c] = dict(indices=indices, mean=mean, factor=factor, **meta)
            residual[indices] = centered
        self.pooled, self.pooled_meta = low_rank_factor(residual.to(device), options["pooled_rank"])

    def state(self):
        return dict(labels=self.labels, classes=self.classes, pooled_factor=self.pooled,
                    pooled_metadata=self.pooled_meta, source_sha256=hashlib.sha256(
                        self.codes.numpy().tobytes() + self.labels.numpy().tobytes()).hexdigest())

    def sample(self, original, index, risk, options, generator):
        c = int(self.labels[index])
        group = self.classes[c]
        count = len(group["indices"])
        if count < options["min_class_samples"]:
            return None
        if options["mode"] == "mixup":
            donors = group["indices"][group["indices"] != index]
            donor = int(donors[torch.randint(len(donors), (), generator=generator)])
            anchor = self.codes[donor]
        else:
            anchor = (count * group["mean"] - self.codes[index]) / (count - 1)
        noise = torch.zeros_like(anchor)
        for factor, weight in ((group["factor"], 1-options["shrinkage"]),
                               (self.pooled, options["shrinkage"])):
            noise.add_(factor @ torch.randn(factor.shape[1], generator=generator), alpha=math.sqrt(weight))
        scale = 0.0 if options["mode"] == "mixup" else options["noise_scale"]
        return (1-risk) * original.cpu() + risk * anchor + scale * noise


def select_requests(risk, options, request_generator, control_generator):
    """Couple request counts across controls with the same rank-weight multiset.

    Acceptance and noise draw counts cannot advance the request RNG. Mapping
    uniforms to sorted risk weights also removes sample-order effects on counts.
    """
    used = risk.cpu().float().clone()
    if options["mode"] in {"shuffled_risk", "mixup"}:
        used = used[torch.randperm(len(used), generator=control_generator)]
    order = torch.argsort(used, stable=True)
    requests = order[torch.rand(len(used), generator=request_generator) < options["replacement_fraction"] * used[order]]
    cap = math.floor(options["replacement_fraction"] * len(used))
    if len(requests) > cap:
        requests = requests[torch.randperm(len(requests), generator=request_generator)[:cap]]
    return used, requests


class RiskSynthesis:
    def __init__(self, config, seed):
        self.options = {**DEFAULTS, **config.get("synthesis", {})}
        self.seed = seed
        self.geometry = {}
        self.counts = Counter()
        self.risk_bins = {str(i): Counter() for i in range(5)}
        self.generators = {}
        self.request_generators = {}
        self.control_generators = {}
        self.teacher = None
        self.handle = None
        self.directory = None
        self.exposure = {}

    @torch.no_grad()
    def initialize(self, users, shared_model, directory):
        self.directory = Path(directory) / "risk_synthesis"
        self.directory.mkdir(exist_ok=False)
        device = shared_model.device
        self.teacher = copy.deepcopy(shared_model.clip_model).eval().requires_grad_(False)
        # Remove PEFT residual effects, including nonzero loaded checkpoints.
        for module in self.teacher.modules():
            if hasattr(module, "lora_B"):
                module.lora_B.zero_()
            if module.__class__.__name__ == "BottleneckAdapter":
                module.up.weight.zero_()
                if module.up.bias is not None:
                    module.up.bias.zero_()
        text_args = {"input_ids": shared_model.text_input_ids}
        if hasattr(shared_model, "text_attention_mask"):
            text_args["attention_mask"] = shared_model.text_attention_mask
        self.text = F.normalize(self.teacher.get_text_features(**text_args).float(), dim=-1)
        fields = ["round", "client", "step", "sample_id", "label", "risk", "used_risk",
                  "requested", "accepted", "attempts", "reason", "nearest_distance", "source_round",
                  "norm_ratio", "teacher_margin_delta"]
        self.handle = (self.directory / "synthetic_exposure.csv").open("x", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()
        self.handle.flush()
        for user in users:
            codes, labels, semantic, ids = [], [], [], []
            for images, target, indices in user.iter_www_statistics_batches():
                tokens = shared_model.encode_input_tokens(images)
                codes.append(tokens[:, 1:].flatten(1).cpu())
                labels.append(target.cpu())
                ids.append(indices.cpu())
                semantic.append(F.normalize(token_features(self.teacher, tokens), dim=-1).cpu())
            indices = torch.cat(ids)
            if not torch.equal(indices, torch.arange(user.train_samples)):
                raise ValueError("Synthesis geometry must cover each original local ID exactly once.")
            geometry = LocalGeometry(torch.cat(codes), torch.cat(labels), self.options, device)
            self.geometry[user.id] = geometry
            self.exposure[user.id] = dict(risk_reads=torch.zeros(user.train_samples, dtype=torch.long),
                                          real_steps=torch.zeros(user.train_samples, dtype=torch.long),
                                          synthetic_steps=torch.zeros(user.train_samples, dtype=torch.long))
            state = geometry.state()
            z = torch.cat(semantic)
            state["semantic_class_means"] = {c: z[g["indices"]].mean(0) for c,g in geometry.classes.items()}
            state["representation"] = "frozen_patch_plus_position_without_cls"
            state["covariance_divisor"] = "n"
            state["options"] = self.options
            state["source_membership"] = "original_client_train"
            distribution_path = self.directory / f"client_{user.id}_distribution.pt"
            code_path = self.directory / f"client_{user.id}_source_codes.pt"
            torch.save(state, distribution_path)
            torch.save(geometry.codes, code_path)
            # Keep exact float32 values, but let the OS reclaim inactive clients'
            # pages on machines with much less host RAM than GPU RAM.
            mapped = torch.load(distribution_path, map_location="cpu", weights_only=True, mmap=True)
            geometry.classes = mapped["classes"]
            geometry.pooled = mapped["pooled_factor"]
            geometry.codes = torch.load(code_path, map_location="cpu", weights_only=True, mmap=True)
            self.generators[user.id] = torch.Generator().manual_seed(self.seed + 1000003 * user.id + 9173)
        self.write_summary("initialized")

    @torch.no_grad()
    def margins(self, tokens, labels):
        similarities = F.normalize(token_features(self.teacher, tokens), dim=-1) @ self.text.T
        own = similarities.gather(1, labels[:, None]).squeeze(1)
        similarities.scatter_(1, labels[:, None], -torch.inf)
        return own - similarities.max(1).values

    @torch.no_grad()
    def transform(self, model, user, images, labels, indices, risk, round_index, step, source_round):
        tokens = model.encode_input_tokens(images)
        original = tokens[:, 1:].flatten(1).cpu()
        risk = risk.cpu().float()
        generator = self.generators[user.id]
        if user.id not in self.request_generators:
            self.request_generators[user.id] = torch.Generator().manual_seed(self.seed + 1000003*user.id + 27183)
            self.control_generators[user.id] = torch.Generator().manual_seed(self.seed + 1000003*user.id + 31415)
        used, requests = select_requests(risk, self.options, self.request_generators[user.id],
                                        self.control_generators[user.id])
        if round_index < self.options["warmup_rounds"] or source_round < 0:
            requests = requests[:0]
        accepted = torch.zeros(len(used), dtype=torch.bool)
        tries = torch.zeros(len(used), dtype=torch.long)
        reasons = ["not_requested"] * len(used)
        distances = [None] * len(used)
        norm_ratios = [None] * len(used)
        margin_deltas = [None] * len(used)
        pending = requests.tolist()
        margins = self.margins(tokens[requests.to(tokens.device)], labels[requests.to(labels.device)]) if (
            len(requests) and self.options["semantic_filter"]) else None
        reference_margins = {} if margins is None else dict(zip(pending, margins.tolist()))
        geometry = self.geometry[user.id]
        for _ in range(self.options["attempts"]):
            batch, positions = [], []
            for j in pending:
                tries[j] += 1
                candidate = geometry.sample(original[j], int(indices[j]), float(used[j]), self.options, generator)
                if candidate is None:
                    reasons[j] = "insufficient_class_samples"
                    continue
                ratio = candidate.norm() / original[j].norm().clamp_min(1e-12)
                norm_ratios[j] = float(ratio) if torch.isfinite(ratio) else None
                if not torch.isfinite(candidate).all():
                    reasons[j] = "nonfinite_geometry"
                    continue
                if ratio < self.options["norm_ratio_min"]:
                    reasons[j] = "norm_too_small"
                    continue
                if ratio > self.options["norm_ratio_max"]:
                    reasons[j] = "norm_too_large"
                    continue
                batch.append(candidate.reshape(tokens.shape[1]-1, tokens.shape[2]))
                positions.append(j)
            if not positions:
                break
            candidates = tokens[positions].clone()
            candidates[:, 1:] = torch.stack(batch).to(tokens)
            passed = torch.ones(len(positions), dtype=torch.bool)
            if self.options["semantic_filter"]:
                with measure_stage(self, "train.synthesis_filter"):
                    new_margins = self.margins(candidates, labels[positions]).cpu()
                passed = new_margins >= torch.tensor([reference_margins[j] for j in positions]) - self.options["margin_tolerance"]
                for pos,j in enumerate(positions):
                    margin_deltas[j] = float(new_margins[pos]) - reference_margins[j]
            for pos, j in enumerate(positions):
                if passed[pos]:
                    tokens[j] = candidates[pos]
                    accepted[j] = True
                    reasons[j] = "accepted"
                    group = geometry.classes[int(labels[j])]
                    distances[j] = float((geometry.codes[group["indices"]] - batch[pos].flatten()).norm(dim=1).min())
                else:
                    reasons[j] = "semantic_margin"
            pending = [j for j in pending if not accepted[j] and reasons[j] != "insufficient_class_samples"]
        for j in range(len(used)):
            requested = bool((requests == j).any())
            record = dict(round=round_index+1, client=user.id, step=step, sample_id=int(indices[j]),
                          label=int(labels[j]), risk=float(risk[j]), used_risk=float(used[j]),
                          requested=int(requested), accepted=int(accepted[j]), attempts=int(tries[j]),
                          reason=reasons[j], nearest_distance=distances[j], source_round=source_round,
                          norm_ratio=norm_ratios[j], teacher_margin_delta=margin_deltas[j])
            self.writer.writerow(record)
            count = dict(visits=1, requested=int(requested), accepted=int(accepted[j]),
                         fallback=int(requested and not accepted[j]))
            self.counts.update(count)
            self.risk_bins[str(min(4, int(float(risk[j])*5)))].update(count)
            sid = int(indices[j])
            self.exposure[user.id]["risk_reads"][sid] += int(source_round >= 0)
            self.exposure[user.id]["synthetic_steps" if accepted[j] else "real_steps"][sid] += 1
        self.handle.flush()
        return tokens.detach()

    def summary(self):
        return dict(implementation="local_token_geometry_v2", options=self.options, seed=self.seed,
                    request_sampling="rank_coupled_independent_rng",
                    counts=dict(self.counts), risk_bins={k:dict(v) for k,v in self.risk_bins.items()},
                    formal_dp_enabled=False, client_upload_is_private=False,
                    epsilon=None, delta=None, membership="original_client_train",
                    reference_is_exact_leave_one_out=False, shared_geometry=False)

    def write_summary(self, status):
        if self.directory is not None:
            (self.directory / "synthesis_summary.json").write_text(json.dumps(
                dict(status=status, **self.summary()), indent=2, allow_nan=False))
            torch.save(self.exposure, self.directory / "source_exposure.pt")

    def close(self, status):
        self.write_summary(status)
        if self.handle is not None:
            self.handle.close()
