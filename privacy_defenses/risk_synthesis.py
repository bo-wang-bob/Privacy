"""Risk-guided token synthesis with optional one-time global class geometry."""
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
from privacy_defenses.synthesis_history import ZeroRiskHistory, history_assignment
from utils.performance import measure_stage


LEGACY_DEFAULTS = dict(
    replacement_fraction=0.25, noise_scale=0.1,
    class_rank=5, min_class_samples=3, attempts=2,
    margin_tolerance=0.02, norm_ratio_min=0.5, norm_ratio_max=2.0,
    mode="risk", semantic_filter=True, warmup_rounds=1,
    center_weighting="uniform", replacement_policy="risk_probability",
    risk_history="none", candidate_selection="first_semantic", views_per_record=1,
    global_distribution="disabled",
    center_source="local_class",
)
DEFAULTS = {**{k:v for k,v in LEGACY_DEFAULTS.items() if k not in {"norm_ratio_min", "norm_ratio_max"}},
            "replacement_policy": "all", "replacement_fraction": 1.0, "warmup_rounds": 0}


def synthesis_options(options):
    """Preserve saved partial-replacement configurations lacking a policy field."""
    options = options or {}
    removed = set(options) & {"shrinkage", "pooled_rank"}
    if removed:
        raise ValueError(f"Per-class synthesis removed options {sorted(removed)}; old pooled runs require their original code version.")
    legacy = bool(options) and options.get("replacement_policy", "risk_probability") == "risk_probability"
    if not legacy and {"norm_ratio_min", "norm_ratio_max"} & set(options):
        raise ValueError("All-replacement synthesis removed norm_ratio_min/norm_ratio_max; rebuild the run with the current entrypoint. Historical norm-filter runs require their original code version.")
    return {**(LEGACY_DEFAULTS if legacy else DEFAULTS), **options}


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
    options = synthesis_options(defense.get("synthesis", {}))
    unknown = set(options) - set(LEGACY_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown synthesis options: {sorted(unknown)}")
    for key in ("replacement_fraction",):
        if not math.isfinite(float(options[key])) or not 0 <= options[key] <= 1:
            raise ValueError(f"synthesis.{key} must be in [0,1].")
    for key in ("noise_scale", "margin_tolerance") + tuple(k for k in ("norm_ratio_min", "norm_ratio_max") if k in options):
        if not math.isfinite(float(options[key])) or options[key] < 0:
            raise ValueError(f"synthesis.{key} must be finite and nonnegative.")
    if "norm_ratio_min" in options and options["norm_ratio_min"] > options["norm_ratio_max"]:
        raise ValueError("Invalid synthesis norm ratio interval.")
    for key in ("class_rank", "min_class_samples", "attempts", "views_per_record"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"synthesis.{key} must be a positive integer.")
    if options["min_class_samples"] < 3:
        raise ValueError("synthesis.min_class_samples must be at least 3.")
    if options["mode"] not in {"risk", "shuffled_risk", "mixup"}:
        raise ValueError("synthesis.mode must be risk, shuffled_risk or mixup.")
    if options["center_weighting"] not in {"uniform", "previous_risk"}:
        raise ValueError("synthesis.center_weighting must be uniform or previous_risk.")
    if options["center_weighting"] != "uniform" and options["mode"] == "mixup":
        raise ValueError("Risk-weighted class centers do not apply to a single MixUp donor.")
    if type(options["semantic_filter"]) is not bool:
        raise ValueError("synthesis.semantic_filter must be boolean.")
    if type(options["warmup_rounds"]) is not int or options["warmup_rounds"] < 0:
        raise ValueError("synthesis.warmup_rounds must be a nonnegative integer.")
    if options["replacement_policy"] not in {"all", "risk_probability"}:
        raise ValueError("synthesis.replacement_policy must be all or risk_probability.")
    if options["risk_history"] not in {"none", "zero_risk_frequency"}:
        raise ValueError("synthesis.risk_history must be none or zero_risk_frequency.")
    if options["candidate_selection"] not in {"first_semantic", "least_local_similarity"}:
        raise ValueError("Unknown synthesis.candidate_selection.")
    if options["global_distribution"] not in {"disabled", "share_only", "generate"}:
        raise ValueError("synthesis.global_distribution must be disabled, share_only or generate.")
    if options["global_distribution"] == "generate" and options["mode"] == "mixup":
        raise ValueError("Global geometry generation requires a geometric noise mode, not mixup.")
    if options["center_source"] not in {"local_class", "global_class"}:
        raise ValueError("synthesis.center_source must be local_class or global_class.")
    if options["center_source"] == "global_class" and (
            options["global_distribution"] != "generate" or options["center_weighting"] != "uniform"):
        raise ValueError("Global class centers require global_distribution=generate and uniform center_weighting.")
    advanced = options["risk_history"] != "none" or options["candidate_selection"] != "first_semantic"
    if advanced and options["replacement_policy"] != "all":
        raise ValueError("History and local-neighbor selection require all replacement.")
    if options["views_per_record"] > 1 and options["replacement_policy"] != "all":
        raise ValueError("Multiple training views require all replacement.")
    if options["candidate_selection"] != "first_semantic" and not options["semantic_filter"]:
        raise ValueError("Local-neighbor selection requires the semantic filter.")
    if options["replacement_policy"] == "all":
        if options["replacement_fraction"] != 1 or options["warmup_rounds"] != 0:
            raise ValueError("All replacement requires replacement_fraction=1 and warmup_rounds=0.")
        if options["noise_scale"] <= 0 or options["mode"] == "mixup":
            raise ValueError("All replacement requires positive noise_scale and risk/shuffled_risk mode, including zero-risk records.")
        if options["center_weighting"] != "uniform":
            raise ValueError("All replacement uses uniform centers; risk controls source retention only.")
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
        self.global_distribution = None
        for c in self.labels.unique().tolist():
            indices = torch.where(self.labels == c)[0]
            rows = self.codes[indices]
            mean = rows.mean(0)
            centered = rows - mean
            factor, meta = low_rank_factor(centered.to(device), min(options["class_rank"], len(rows)-1))
            self.classes[c] = dict(indices=indices, mean=mean, factor=factor, **meta)

    def state(self):
        return dict(labels=self.labels, classes=self.classes, geometry_source="local_class_only",
                    source_sha256=hashlib.sha256(
                        self.codes.numpy().tobytes() + self.labels.numpy().tobytes()).hexdigest())

    def leave_source_out_center(self, index, weights=None):
        group = self.classes[int(self.labels[index])]
        donors = group["indices"][group["indices"] != index]
        if not len(donors):
            raise ValueError("A leave-source-out center needs another class record.")
        if weights is not None:
            if (weights.ndim != 1 or len(weights) != len(self.codes) or not torch.isfinite(weights).all()
                    or (weights < 0).any()):
                raise ValueError("Center weights must be finite, nonnegative and source-aligned.")
            selected = weights[donors]
            # Preserve the original arithmetic exactly for uniform anchors,
            # including the bootstrap rounds and an all-zero donor fallback.
            if float(selected.sum()) > 0 and not torch.equal(selected, selected[:1].expand_as(selected)):
                return (self.codes[donors] * selected[:, None]).sum(0) / selected.sum()
        count = len(group["indices"])
        return (count * group["mean"] - self.codes[index]) / (count - 1)

    def global_class_center(self, index):
        if self.global_distribution is None:
            raise RuntimeError("Global distribution must be received before computing a global center.")
        group = self.global_distribution["classes"][int(self.labels[index])]
        # All clients and records of this class use the same broadcast mean.
        # It includes every original training record, including this source.
        return group["mean"].float()

    def sample(self, original, index, risk, options, generator, center_weights=None):
        c = int(self.labels[index])
        group = self.classes[c]
        count = len(group["indices"])
        if count < options["min_class_samples"]:
            return None
        if options["mode"] == "mixup":
            donors = group["indices"][group["indices"] != index]
            donor = int(donors[torch.randint(len(donors), (), generator=generator)])
            anchor = self.codes[donor]
        elif options.get("center_source", "local_class") == "global_class":
            anchor = self.global_class_center(index)
        else:
            anchor = self.leave_source_out_center(index, center_weights)
        factor = group["factor"]
        if options.get("global_distribution", "disabled") == "generate":
            if self.global_distribution is None:
                raise RuntimeError("Global distribution must be received before synthesis training.")
            factor = self.global_distribution["classes"][c]["factor"]
        noise = factor @ torch.randn(factor.shape[1], generator=generator)
        scale = 0.0 if options["mode"] == "mixup" else options["noise_scale"]
        return (1-risk) * original.cpu() + risk * anchor + scale * noise


class PreviousRiskWeights:
    """Freeze 1 - mean(assigned risk) from the previous participating round.

    Missing reference scores retain weight one. Accumulation never changes the
    anchors used by other batches in the current round. Shuffled controls must
    supply their assigned ``used_risk``, not the unshuffled original score.
    """
    def __init__(self):
        self.clients = {}

    def begin(self, client, round_index, size):
        state = self.clients.get(client)
        if state is None:
            state = dict(round=round_index, reference_round=None,
                         weights=torch.ones(size), available=torch.zeros(size, dtype=torch.bool),
                         total=torch.zeros(size, dtype=torch.float64), count=torch.zeros(size, dtype=torch.long))
            self.clients[client] = state
        if len(state["weights"]) != size or round_index < state["round"]:
            raise ValueError("Anchor risk history must retain original identities and increasing rounds.")
        if round_index > state["round"]:
            available = state["count"] > 0
            weights = torch.ones(size)
            weights[available] = (1 - state["total"][available] / state["count"][available]).float()
            state.update(weights=weights, available=available,
                         reference_round=state["round"] + 1 if available.any() else None,
                         round=round_index)
            state["total"].zero_()
            state["count"].zero_()
        return state

    def observe(self, client, indices, assigned_risk, *, available):
        if not available:
            return
        state = self.clients[client]
        indices, risk = indices.cpu().long(), assigned_risk.cpu().double()
        if (indices.ndim != 1 or risk.shape != indices.shape or not torch.isfinite(risk).all()
                or (risk < 0).any() or (risk > 1).any()):
            raise ValueError("Expected finite source-aligned assigned risks in [0,1].")
        state["total"].scatter_add_(0, indices, risk)
        state["count"].scatter_add_(0, indices, torch.ones_like(indices))


def select_requests(risk, options, request_generator, control_generator):
    """Couple request counts across controls with the same rank-weight multiset.

    Acceptance and noise draw counts cannot advance the request RNG. Mapping
    uniforms to sorted risk weights also removes sample-order effects on counts.
    """
    used = risk.cpu().float().clone()
    if options["mode"] in {"shuffled_risk", "mixup"}:
        used = used[torch.randperm(len(used), generator=control_generator)]
    if options.get("replacement_policy", "risk_probability") == "all":
        return used, torch.arange(len(used))
    order = torch.argsort(used, stable=True)
    requests = order[torch.rand(len(used), generator=request_generator) < options["replacement_fraction"] * used[order]]
    cap = math.floor(options["replacement_fraction"] * len(used))
    if len(requests) > cap:
        requests = requests[torch.randperm(len(requests), generator=request_generator)[:cap]]
    return used, requests


class RiskSynthesis:
    def __init__(self, config, seed):
        self.options = synthesis_options(config.get("synthesis", {}))
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
        self.previous_risk_weights = PreviousRiskWeights()
        self.history = ZeroRiskHistory()
        self.pending_history = {}
        self.semantic_sources = {}
        self.candidate_handle = None
        self.view_handle = None
        self.view_counts = Counter()
        self.pending_views = {}
        self.global_exchange = None

    @torch.no_grad()
    def initialize(self, users, shared_model, directory):
        if self.directory is not None:
            raise RuntimeError("Synthesis distributions are initialized exactly once before training.")
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
        if self.options["replacement_policy"] == "all":
            fields += ["quality_passed", "selected_attempt", "retained_original_fraction", "original_distance"]
        if self.options["risk_history"] != "none":
            fields += ["loss_gap", "history_exposure", "history_rounds", "joint_rank_score", "assigned_risk"]
        if self.options["candidate_selection"] != "first_semantic":
            fields += ["nearest_teacher_cosine", "nearest_teacher_source_id"]
            self.candidate_handle = (self.directory / "candidate_choices.csv").open("x", newline="")
            candidate_fields = [
                "round", "client", "step", "sample_id", "attempt", "norm_ratio", "teacher_margin_delta",
                "quality_passed", "nearest_teacher_cosine", "nearest_teacher_source_id"]
            if self.options["views_per_record"] > 1:
                candidate_fields += ["view_index"]
            self.candidate_writer = csv.DictWriter(self.candidate_handle, fieldnames=candidate_fields)
            self.candidate_writer.writeheader()
        if self.options["center_weighting"] == "previous_risk":
            fields += ["anchor_reference_round", "anchor_available_donors", "anchor_source_weight",
                       "anchor_donor_weight_sum", "anchor_uniform_fallback"]
        if self.options["views_per_record"] > 1:
            self.view_handle = (self.directory / "synthetic_views.csv").open("x", newline="")
            self.view_writer = csv.DictWriter(self.view_handle, fieldnames=fields + ["view_index", "loss_weight"])
            self.view_writer.writeheader()
            fields += ["views_per_record", "quality_passed_views", "representative_view_index", "total_attempts"]
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
            if self.options["replacement_policy"] == "all" and any(
                    len(group["indices"]) < self.options["min_class_samples"] for group in geometry.classes.values()):
                raise ValueError("All replacement requires min_class_samples in every local class; original-image fallback is disabled.")
            self.geometry[user.id] = geometry
            self.exposure[user.id] = dict(risk_reads=torch.zeros(user.train_samples, dtype=torch.long),
                                          real_steps=torch.zeros(user.train_samples, dtype=torch.long),
                                          synthetic_steps=torch.zeros(user.train_samples, dtype=torch.long))
            if self.options["views_per_record"] > 1:
                self.exposure[user.id]["synthetic_views"] = torch.zeros(user.train_samples, dtype=torch.long)
            state = geometry.state()
            z = torch.cat(semantic)
            if self.options["candidate_selection"] != "first_semantic":
                self.semantic_sources[user.id] = z
                state["semantic_source_features"] = z
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
            geometry.codes = torch.load(code_path, map_location="cpu", weights_only=True, mmap=True)
            self.generators[user.id] = torch.Generator().manual_seed(self.seed + 1000003 * user.id + 9173)
        if self.options["global_distribution"] != "disabled":
            from privacy_defenses.global_geometry import exchange
            with measure_stage(self, "setup.synthesis_global_distribution"):
                self.global_exchange = exchange(self.geometry, self.directory, self.options, device)
        self.write_summary("initialized")

    @torch.no_grad()
    def margins(self, tokens, labels):
        similarities = F.normalize(token_features(self.teacher, tokens), dim=-1) @ self.text.T
        own = similarities.gather(1, labels[:, None]).squeeze(1)
        similarities.scatter_(1, labels[:, None], -torch.inf)
        return own - similarities.max(1).values

    @torch.no_grad()
    def candidate_metrics(self, tokens, labels, client):
        features = F.normalize(token_features(self.teacher, tokens).float(), dim=-1)
        similarities = features @ self.text.T
        own = similarities.gather(1, labels[:, None]).squeeze(1)
        similarities.scatter_(1, labels[:, None], -torch.inf)
        # Include every local original, including the source and other classes.
        # This avoids explicitly rewarding transfer toward a different donor.
        local = features @ self.semantic_sources[client].to(features).T
        cosine, nearest = local.max(1)
        return (own - similarities.max(1).values).cpu(), cosine.cpu(), nearest.cpu()

    def record_optimized_batch(self, client):
        views = self.pending_views.pop(client, None)
        if views is not None:
            for records in zip(*views):
                # One original visit, with the worst semantic view as the
                # explicitly declared representative. Every view is saved below.
                worst = min(records, key=lambda r: (r["teacher_margin_delta"] or 0., r["view_index"]))
                record = {k: v for k, v in worst.items() if k not in {"view_index", "loss_weight"}}
                record.update(views_per_record=len(records),
                              quality_passed_views=sum(r["quality_passed"] for r in records),
                              representative_view_index=worst["view_index"],
                              total_attempts=sum(r["attempts"] for r in records))
                self.writer.writerow(record)
                count = dict(visits=1, requested=1, accepted=1, fallback=0,
                             quality_failed=int(record["quality_passed_views"] < len(records)))
                self.counts.update(count)
                self.risk_bins[str(min(4, int(record["risk"]*5)))].update(count)
                sid = record["sample_id"]
                self.exposure[client]["risk_reads"][sid] += int(record["source_round"] >= 0)
                self.exposure[client]["synthetic_steps"][sid] += 1
                self.exposure[client]["synthetic_views"][sid] += len(records)
            # View-major order preserves each view's batch boundaries for replay.
            for records in views:
                for record in records:
                    self.view_writer.writerow(record)
                    self.view_counts.update(visits=1, requested=1, accepted=1, fallback=0,
                                            quality_failed=1-record["quality_passed"])
            self.handle.flush()
            self.view_handle.flush()
        pending = self.pending_history.pop(client, None)
        if pending is not None:
            round_index, indices, used = pending
            self.history.observe(client, round_index, indices, used)

    @torch.no_grad()
    def transform_views(self, model, user, images, labels, indices, risk, round_index, step, source_round, *, raw_scores=None):
        """Generate K independently drawn, jointly trained views per original.

        Ranking and shuffled assignment happen once per original batch. All K
        views must be valid before any backward/optimizer work. Source counters
        and history commit once, after the shared optimizer step succeeds.
        """
        if self.options["views_per_record"] == 1:
            return (self.transform(model, user, images, labels, indices, risk, round_index, step,
                                   source_round, raw_scores=raw_scores),)
        if self.options["replacement_policy"] != "all":
            raise ValueError("Multiple training views require all replacement.")
        if user.id in self.pending_views or user.id in self.pending_history:
            raise RuntimeError("Previous synthesis batch was not committed after optimization.")
        assignment = self._all_assignment(user, indices, risk, round_index, source_round, raw_scores)
        outputs, records = [], []
        for view in range(self.options["views_per_record"]):
            tokens, logged = self._transform_all(model, user, images, labels, indices, risk,
                round_index, step, source_round, raw_scores=raw_scores, assignment=assignment,
                view_index=view, previous_views=outputs)
            outputs.append(tokens)
            records.append(logged)
        self.pending_views[user.id] = records
        if self.options["risk_history"] != "none":
            self.pending_history[user.id] = (round_index, indices.detach().cpu().clone(), assignment[-2].clone())
        return tuple(outputs)

    @torch.no_grad()
    def transform(self, model, user, images, labels, indices, risk, round_index, step, source_round, *, raw_scores=None):
        if self.options["views_per_record"] != 1:
            raise ValueError("Multiple training views require transform_views; selecting only one is invalid.")
        if self.options["replacement_policy"] == "all":
            return self._transform_all(model, user, images, labels, indices, risk, round_index, step, source_round,
                                       raw_scores=raw_scores)
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
        anchor_state = None
        if self.options["center_weighting"] == "previous_risk":
            anchor_state = self.previous_risk_weights.begin(user.id, round_index, len(geometry.labels))
            self.previous_risk_weights.observe(user.id, indices, used, available=source_round >= 0)
        for _ in range(self.options["attempts"]):
            batch, positions = [], []
            for j in pending:
                tries[j] += 1
                candidate = geometry.sample(original[j], int(indices[j]), float(used[j]), self.options, generator,
                                            None if anchor_state is None else anchor_state["weights"])
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
            if anchor_state is not None:
                sid = int(indices[j])
                group = geometry.classes[int(labels[j])]["indices"]
                donors = group[group != sid]
                weight_sum = float(anchor_state["weights"][donors].sum())
                record.update(anchor_reference_round=anchor_state["reference_round"],
                              anchor_available_donors=int(anchor_state["available"][donors].sum()),
                              anchor_source_weight=float(anchor_state["weights"][sid]),
                              anchor_donor_weight_sum=weight_sum,
                              anchor_uniform_fallback=int(weight_sum <= 0))
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

    def _all_assignment(self, user, indices, risk, round_index, source_round, raw_scores):
        risk = risk.cpu().float()
        if risk.shape != (len(indices),) or not torch.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any():
            raise ValueError("All replacement requires one finite risk in [0,1] per original record.")
        if source_round < 0:
            risk = torch.zeros_like(risk)
        assigned = risk
        history_values = history_rounds = joint = None
        if self.options["risk_history"] != "none":
            if user.id in self.pending_history:
                raise RuntimeError("Previous synthesis batch was not committed after optimization.")
            history_values, history_rounds = self.history.values(
                user.id, round_index, len(self.geometry[user.id].labels), indices)
            if source_round >= 0:
                if raw_scores is None:
                    raise ValueError("Exposure history needs original loss gaps, before tail truncation.")
                assigned, joint = history_assignment(raw_scores, history_values, risk)
        if user.id not in self.control_generators:
            self.control_generators[user.id] = torch.Generator().manual_seed(self.seed + 1000003*user.id + 31415)
        used, requests = select_requests(assigned, self.options, None, self.control_generators[user.id])
        return risk, assigned, history_values, history_rounds, joint, used, requests

    @torch.no_grad()
    def _transform_all(self, model, user, images, labels, indices, risk, round_index, step, source_round, *,
                       raw_scores=None, assignment=None, view_index=None, previous_views=()):
        """Replace every position, retaining the best generated semantic candidate.

        Semantic failure never restores the original. An absent finite, changed,
        distinct candidate aborts the batch before optimization instead.
        """
        tokens = model.encode_input_tokens(images)
        original = tokens[:, 1:].flatten(1).cpu()
        if assignment is None:
            assignment = self._all_assignment(user, indices, risk, round_index, source_round, raw_scores)
        risk, assigned, history_values, history_rounds, joint, used, requests = assignment
        geometry, generator = self.geometry[user.id], self.generators[user.id]
        semantic = self.options["semantic_filter"]
        choose_local = self.options["candidate_selection"] == "least_local_similarity"
        reference = self.margins(tokens, labels).cpu() if semantic else None
        if reference is not None and not torch.isfinite(reference).all():
            raise ValueError("Nonfinite reference semantics in all-replacement synthesis.")
        pending = requests.tolist()
        tries = [0] * len(tokens)
        best = {}
        invalid = {}
        for attempt in range(1, self.options["attempts"] + 1):
            batch, positions, ratios = [], [], []
            for j in pending:
                tries[j] += 1
                candidate = geometry.sample(original[j], int(indices[j]), float(used[j]), self.options, generator)
                if candidate is None:
                    invalid[j] = "insufficient_class_samples"
                    continue
                candidate = candidate.to(dtype=tokens.dtype)
                if not torch.isfinite(candidate).all():
                    invalid[j] = "nonfinite_geometry"
                    continue
                # Descriptive only: global means can legitimately have much
                # smaller norms than individual inputs. Do not gate, rescale
                # or clip a generated candidate based on this ratio.
                ratio = float(candidate.double().norm() / original[j].double().norm().clamp_min(1e-12))
                if torch.equal(candidate, original[j]):
                    invalid[j] = "unchanged_candidate"
                    continue
                if any(torch.equal(candidate, previous[j, 1:].flatten().cpu()) for previous in previous_views):
                    invalid[j] = "duplicate_training_view"
                    continue
                batch.append(candidate.reshape(tokens.shape[1]-1, tokens.shape[2]))
                positions.append(j)
                ratios.append(ratio)
            if not positions:
                continue
            candidates = tokens[positions].clone()
            candidates[:, 1:] = torch.stack(batch).to(tokens)
            if choose_local:
                with measure_stage(self, "train.synthesis_filter"):
                    margins, cosines, neighbors = self.candidate_metrics(candidates, labels[positions], user.id)
            elif semantic:
                with measure_stage(self, "train.synthesis_filter"):
                    margins = self.margins(candidates, labels[positions]).cpu()
            else:
                margins = torch.zeros(len(positions))
            for pos, j in enumerate(positions):
                delta = float(margins[pos] - reference[j]) if semantic else 0.0
                if not math.isfinite(delta):
                    invalid[j] = "nonfinite_semantics"
                    continue
                quality = not semantic or delta >= -self.options["margin_tolerance"]
                candidate = dict(token=batch[pos], delta=delta, norm_ratio=ratios[pos], attempt=attempt,
                                 quality_passed=quality)
                if choose_local:
                    cosine = float(cosines[pos])
                    if not math.isfinite(cosine):
                        invalid[j] = "nonfinite_neighbor_similarity"
                        continue
                    candidate.update(nearest_teacher_cosine=cosine, nearest_teacher_source_id=int(neighbors[pos]))
                    candidate_record = dict(round=round_index+1, client=user.id, step=step,
                        sample_id=int(indices[j]), attempt=attempt, norm_ratio=ratios[pos],
                        teacher_margin_delta=delta, quality_passed=int(quality),
                        nearest_teacher_cosine=cosine, nearest_teacher_source_id=int(neighbors[pos]))
                    if view_index is not None:
                        candidate_record["view_index"] = view_index
                    self.candidate_writer.writerow(candidate_record)
                    # Semantic feasibility dominates; if all fail, use maximum
                    # margin as requested. Exact ties retain the earlier draw.
                    key = (int(quality), -cosine if quality else delta, delta)
                    candidate["selection_key"] = key
                    improved = j not in best or key > best[j]["selection_key"]
                else:
                    improved = j not in best or delta > best[j]["delta"]
                if improved:
                    best[j] = candidate
            if not choose_local:
                pending = [j for j in pending if j not in best or not best[j]["quality_passed"]]
            if not pending:
                break
        missing = [int(indices[j]) for j in requests.tolist() if j not in best]
        if missing:
            raise RuntimeError(f"All replacement could not produce valid changed tokens for client {user.id}, "
                               f"original IDs {missing}; reasons={invalid}. No original-image fallback.")
        logged_records = []
        for j in requests.tolist():
            chosen = best[j]
            candidate = chosen["token"].flatten().float()
            distance = float((candidate - original[j].float()).norm())
            if not math.isfinite(distance) or distance <= 0:
                raise RuntimeError("All replacement produced an unchanged or nonfinite input.")
            tokens[j, 1:] = chosen["token"].to(tokens)
            group = geometry.classes[int(labels[j])]
            nearest = float((geometry.codes[group["indices"]] - candidate).norm(dim=1).min())
            quality = chosen["quality_passed"]
            record = dict(round=round_index+1, client=user.id, step=step, sample_id=int(indices[j]),
                label=int(labels[j]), risk=float(risk[j]), used_risk=float(used[j]), requested=1, accepted=1,
                attempts=tries[j], reason="accepted" if quality else "best_semantic_candidate", nearest_distance=nearest,
                source_round=source_round, norm_ratio=chosen["norm_ratio"],
                teacher_margin_delta=chosen["delta"] if semantic else None, quality_passed=int(quality),
                selected_attempt=chosen["attempt"], retained_original_fraction=1-float(used[j]), original_distance=distance)
            if history_values is not None:
                record.update(loss_gap=None if source_round < 0 else float(raw_scores[j]),
                              history_exposure=float(history_values[j]), history_rounds=int(history_rounds[j]),
                              joint_rank_score=None if joint is None else float(joint[j]),
                              assigned_risk=float(assigned[j]))
            if choose_local:
                record.update(nearest_teacher_cosine=chosen["nearest_teacher_cosine"],
                              nearest_teacher_source_id=chosen["nearest_teacher_source_id"])
            if view_index is not None:
                record.update(view_index=view_index, loss_weight=1/self.options["views_per_record"])
                logged_records.append(record)
                continue
            self.writer.writerow(record)
            count = dict(visits=1, requested=1, accepted=1, fallback=0, quality_failed=int(not quality))
            self.counts.update(count)
            self.risk_bins[str(min(4, int(float(risk[j])*5)))].update(count)
            sid = int(indices[j])
            self.exposure[user.id]["risk_reads"][sid] += int(source_round >= 0)
            self.exposure[user.id]["synthetic_steps"][sid] += 1
        self.handle.flush()
        if self.candidate_handle is not None:
            self.candidate_handle.flush()
        if view_index is not None:
            return tokens.detach(), logged_records
        if history_values is not None:
            self.pending_history[user.id] = (round_index, indices.detach().cpu().clone(), used.clone())
        return tokens.detach()

    def summary(self):
        shared = self.options["global_distribution"] != "disabled"
        global_center = self.options["center_source"] == "global_class"
        replace_all = self.options["replacement_policy"] == "all"
        return dict(implementation=("local_token_geometry_v11_no_norm_filter" if replace_all else
                                    "local_token_geometry_v10_global_mean" if global_center else
                                    "local_token_geometry_v8_global_class" if shared else
                                    "local_token_geometry_v7_class_only"),
                    norm_ratio_filter_enabled=not replace_all,
                    norm_ratio_role="diagnostic_only" if replace_all else "candidate_constraint",
                    geometry_source=("global_same_class" if self.options["global_distribution"] == "generate"
                                     else "local_class_only"),
                    local_statistics_geometry_source="local_class_only",
                    generation_center=("global_same_class_mean" if global_center else
                                       "local_same_class_leave_source_out"),
                    center_includes_source=global_center,
                    global_distribution=self.global_exchange,
                    options=self.options, seed=self.seed,
                    history_definition=("mean_per_round_zero_assigned_risk_frequency" if
                                        self.options["risk_history"] != "none" else None),
                    history_combination=("equal_midrank_sum_then_loss_gap_then_batch_order" if
                                         self.options["risk_history"] != "none" else None),
                    center_risk_reference=("previous_participating_round_mean_assigned_rank" if
                                           self.options["center_weighting"] == "previous_risk" else None),
                    request_sampling=("every_original_training_position" if self.options["replacement_policy"] == "all"
                                      else "rank_coupled_independent_rng"),
                    semantic_failure_policy=("best_generated_candidate" if self.options["replacement_policy"] == "all"
                                             else "original_input"),
                    counts=dict(self.counts), risk_bins={k:dict(v) for k,v in self.risk_bins.items()},
                    **(dict(view_counts=dict(self.view_counts),
                            observation_unit="original_visit", view_observation_unit="trained_virtual_view",
                            original_row_measurement="worst_semantic_view",
                            loss_normalization="mean_over_views_then_mean_over_original_records",
                            optimizer_steps_per_original_batch=1) if self.options["views_per_record"] > 1 else {}),
                    formal_dp_enabled=False, client_upload_is_private=False,
                    epsilon=None, delta=None, membership="original_client_train",
                    reference_is_exact_leave_one_out=False, shared_geometry=shared)

    def write_summary(self, status):
        if self.directory is not None:
            (self.directory / "synthesis_summary.json").write_text(json.dumps(
                dict(status=status, **self.summary()), indent=2, allow_nan=False))
            torch.save(self.exposure, self.directory / "source_exposure.pt")
            if self.options["risk_history"] != "none":
                torch.save(self.history.clients, self.directory / "history_state.pt")

    def close(self, status):
        self.write_summary(status)
        if self.handle is not None:
            self.handle.close()
        if self.candidate_handle is not None:
            self.candidate_handle.close()
        if self.view_handle is not None:
            self.view_handle.close()
