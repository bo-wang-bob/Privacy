#!/usr/bin/env python3
"""Compare CLIP audit and shared BERT evaluation on local weights/synthetic data."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch.utils.data import TensorDataset
from transformers import CLIPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aggregator.aggregator_builder import build_aggregator
from privacy_attacks.auditor import MembershipAuditor
from scripts.benchmark_fedsgd_gradients import measure, verify
from servers.serverbase import ServerBase
from trainmodel.clip_adapter import CLIPAdapter
from trainmodel.clip_lora import CLIPLoRA
from trainmodel.clip_mlp import CLIPImageMLP
from trainmodel.transformer_adapter import TransformerAdapterClassifier
from utils.performance import StageTimings


def clip_audit(args, device, kind):
    if (args.clip_path / "config.json").is_file():
        clip = CLIPModel.from_pretrained(str(args.clip_path), local_files_only=True)
    else:
        clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=str(args.clip_path), local_files_only=True,
        )
    names = [str(i) for i in range(args.classes)]
    dim = clip.config.projection_dim
    if kind == "clip_mlp":
        model = CLIPImageMLP(clip, num_classes=args.classes, hidden_dim=512, device=device)
    elif kind == "clip_adapter":
        model = CLIPAdapter(clip, text_features=torch.randn(args.classes, dim),
                            classnames=names, device=device)
    else:
        tokens = torch.tensor([[49406, 100+i, 49407] for i in range(args.classes)])
        model = CLIPLoRA(clip, text_inputs={"input_ids": tokens, "attention_mask": torch.ones_like(tokens)},
                        classnames=names, encoder="both", target_modules=["q", "k", "v"],
                        rank=2, alpha=1., dropout=0., device=device)
    model.eval()
    with torch.no_grad():
        # Exercise gradients through the normally zero-initialized LoRA branch.
        for name, p in model.named_parameters():
            if p.requires_grad and name.endswith("lora_B"):
                p.add_(torch.randn_like(p) * .002)
    cached = kind != "clip_lora"
    inputs = (torch.randn(args.audit_candidates, dim) if cached else
              torch.randn(args.audit_candidates, 3, clip.config.vision_config.image_size,
                          clip.config.vision_config.image_size))
    labels = torch.arange(args.audit_candidates) % args.classes
    named = [(n, p) for n,p in model.named_parameters() if p.requires_grad]
    parameter_names = [n for n,_ in named]
    updates = [({n: torch.randn(p.shape) for n,p in named}, 1., 1.) for _ in range(args.clients)]
    flat_updates = [torch.cat([state[n].flatten() for n in parameter_names]) for state,_,_ in updates]
    auditor = MembershipAuditor.__new__(MembershipAuditor)
    auditor.device, auditor.images, auditor.labels = device, inputs, labels
    auditor.candidate_inputs_are_features = cached
    auditor.audit_batch_size = 128
    auditor.config = {"grad_sample_backend": "auto", "grad_sample_chunk_size": args.chunk_size,
                      "low_fpr_gradient_batch_size": 16, "gradient_update_cache_mb": 2048}
    auditor.timings = StageTimings(device)

    def reference():
        if cached:
            return [auditor._cached_feature_gradient_cosines(model, parameter_names, flat_updates)]
        # Original raw CLIP path retained the entire [candidates, parameters] tensor.
        gradients, _, _ = auditor._candidate_gradients(model, parameter_names)
        norms = gradients.norm(dim=1).clamp_min(1e-12)
        return [torch.stack([(gradients @ u) / (norms * u.norm().clamp_min(1e-12))
                             for u in flat_updates])]

    def optimized():
        result = auditor._raw_input_gradient_measurements(
            model, parameter_names, updates, need_cosine=True, need_gradient_difference=False,
            candidate_inputs_are_features=cached,
        )["cosine"]
        auditor.timings.flush()
        return [result]

    error = verify(reference, optimized)
    before = measure(reference, device, args.warmups, args.repeats)
    after = measure(optimized, device, args.warmups, args.repeats)
    params = sum(p.numel() for _,p in named)
    return {
        "stage": kind + "_cosine_audit", "candidates": args.audit_candidates,
        "parameters": params, "cached_features": cached,
        "reference_gradient_tensor_bytes": min(args.audit_candidates, 16 if cached else args.audit_candidates) * params * 4,
        "optimized_gradient_tensor_bytes": min(args.audit_candidates, args.chunk_size) * params * 4,
        "reference": before, "optimized": after,
        "speedup": before["median_seconds"] / after["median_seconds"], "equivalence": error,
    }


def bert_evaluation(args, device):
    model = TransformerAdapterClassifier(str(args.bert_path), architecture="bert", num_classes=2, device=device)
    tokens = torch.randint(100, model.backbone.config.vocab_size, (args.eval_records_per_client, args.sequence_length))
    inputs = torch.stack((tokens, torch.ones_like(tokens)), dim=1)
    data = TensorDataset(inputs, torch.arange(len(tokens)) % 2)
    # A temporary task is used only to instantiate client state and data loaders.
    with tempfile.TemporaryDirectory(prefix="privacy_eval_benchmark_") as path:
        server = ServerBase(
            device=device, dataset_name="cola", model=model,
            train_sets=[data] * args.clients, test_sets=[data] * args.clients,
            class_names=["0", "1"], batch_size=32, eval_batch_size=32,
            learning_rate=.005, num_glob_iters=1, local_epochs=1, total_users=args.clients,
            results_dir=path, user_per_round=args.clients, eval_interval=1,
            aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"),
            audit_config={"enabled": False}, projres_config={"enabled": False},
            defense_config={"name": "none"},
        )
        server.ctx.set_base_model_state(server.ctx.users[0].get_parameters())
        expected = server._evaluate_clients()
        actual = server._evaluate_shared_model()
        if expected[:3] != actual[:3] or not torch.equal(expected[3], actual[3]):
            raise AssertionError("Shared evaluation changed loss, accuracy, sample count, or confusion matrix")
        before = measure(server._evaluate_clients, device, args.warmups, args.repeats)
        after = measure(server._evaluate_shared_model, device, args.warmups, args.repeats)
        return {
            "stage": "bert_adapter_evaluation", "clients": args.clients,
            "records_per_client": args.eval_records_per_client, "eval_batch_size": 32,
            "reference": before, "optimized": after,
            "speedup": before["median_seconds"] / after["median_seconds"],
            "equivalence": "identical loss sum, correct count, sample count and confusion matrix",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-path", type=Path, default=ROOT / "checkpoints/clip-vit-base-patch32")
    parser.add_argument("--bert-path", type=Path, default=ROOT / "checkpoints/bert-base-uncased")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--stages", default="clip_mlp,clip_adapter,clip_lora,bert_evaluation")
    parser.add_argument("--audit-candidates", type=int, default=16)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--clients", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--eval-records-per-client", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("/tmp/runtime_optimization_benchmark.json"))
    args = parser.parse_args()
    stages = args.stages.split(",")
    if not set(stages) <= {"clip_mlp", "clip_adapter", "clip_lora", "bert_evaluation"}:
        parser.error("Unknown benchmark stage")
    if min(args.audit_candidates, args.classes, args.chunk_size, args.eval_records_per_client,
           args.sequence_length, args.repeats) <= 0 or args.clients < 2 or args.warmups < 0:
        parser.error("Sizes must be positive, clients >= 2, and warmups >= 0")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device(args.device)
    report = {"device": str(device), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "repeats": args.repeats, "warmups": args.warmups, "clients": args.clients,
              "sequence_length": args.sequence_length, "chunk_size": args.chunk_size,
              "classes": args.classes, "seed": 42, "cpu_threads": 4,
              "clip_path": str(args.clip_path.resolve()), "bert_path": str(args.bert_path.resolve()),
              "scope": "Local pretrained models with synthetic data. CLIP reference is the pre-optimization path; optimized audit includes default timing instrumentation. Evaluation preserves each client loader and batch. No training or membership-effectiveness experiment.",
              "measurements": []}
    for stage in stages:
        row = bert_evaluation(args, device) if stage == "bert_evaluation" else clip_audit(args, device, stage)
        report["measurements"].append(row)
        print(json.dumps(row), flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
