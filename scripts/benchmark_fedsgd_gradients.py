#!/usr/bin/env python3
"""Reproducible BERT-Adapter compute benchmark; no training data or results edits.

The reference paths reproduce the pre-optimization clipping/audit operations.
Inputs and uploaded vectors are synthetic; model weights are loaded locally.
This measures compute stages, not total training time or privacy/utility scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from privacy_attacks.auditor import MembershipAuditor
from privacy_defenses.www_dp import DEFAULTS, ino_weights, weighted_clipped_sum
from trainmodel.transformer_adapter import TransformerAdapterClassifier
from utils.per_sample_gradients import clipped_sum_from_losses


def legacy_record_sum(model, inputs, labels, parameters, max_norm, chunk_size=16):
    """Old BERT Record-DP: one batch graph, one backward per record."""
    sums = [torch.zeros_like(p) for p in parameters]
    for start in range(0, len(labels), chunk_size):
        losses = F.cross_entropy(model(inputs[start:start+chunk_size]), labels[start:start+chunk_size], reduction="none")
        for i, loss in enumerate(losses):
            gradients = torch.autograd.grad(loss, parameters, retain_graph=i+1 < len(losses))
            norm = sum(g.detach().float().square().sum() for g in gradients).sqrt().clamp_min(1e-12)
            factor = (max_norm / norm).clamp(max=1)
            for destination, gradient in zip(sums, gradients):
                destination.add_(gradient.detach(), alpha=float(factor))
    return sums


def batched_record_sum(model, inputs, labels, parameters, max_norm, chunk_size):
    sums = [torch.zeros_like(p) for p in parameters]
    for start in range(0, len(labels), chunk_size):
        losses = F.cross_entropy(model(inputs[start:start+chunk_size]), labels[start:start+chunk_size], reduction="none")
        partial, _ = clipped_sum_from_losses(losses, parameters, max_norm)
        for destination, value in zip(sums, partial):
            destination.add_(value)
    return sums


def legacy_audit(model, inputs, labels, parameters, updates):
    """Old audit: separate sample graphs and repeated CPU float64 conversions."""
    norms = [sum((u[n].double() * scale).square().sum() for n, _ in parameters).sqrt().clamp_min(1e-12)
             for u, _, scale in updates]
    cosines, differences = [], []
    for x, y in zip(inputs, labels):
        logits = model(x[None])
        losses = [F.cross_entropy(logits, y[None]),
                  (logits.shape[1] * logits.logsumexp(1) - logits.sum(1)).sum()]
        for i, loss in enumerate(losses):
            gradients = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=i == 0)
            norm_sq = torch.zeros((), dtype=torch.float64)
            dots = [torch.zeros((), dtype=torch.float64) for _ in (updates if i == 0 else updates[:1])]
            for (name, _), gradient in zip(parameters, gradients):
                flat = gradient.detach().reshape(-1).cpu().double()
                norm_sq += flat.square().sum()
                for j in range(len(dots)):
                    upload, sign, scale = updates[j]
                    dots[j] += (flat * upload[name].reshape(-1).double()).sum() * sign * scale
            if i == 0:
                cosines.append(torch.stack([d / (norm_sq.sqrt().clamp_min(1e-12) * n) for d, n in zip(dots, norms)]).float())
            else:
                differences.append(torch.stack([2 * d - norm_sq for d in dots]).float())
    return [torch.stack(cosines).T, torch.stack(differences).T]


def measure(fn, device, warmups, repeats):
    for _ in range(warmups):
        fn()
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        values = fn()
        synchronize()
        samples.append(time.perf_counter() - start)
        del values
    return {"median_seconds": statistics.median(samples), "seconds": samples,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}


def verify(reference, optimized):
    expected, actual = reference(), optimized()
    error = max((a-b).abs().max().item() for a, b in zip(expected, actual))
    relative_l2 = (sum((a-b).double().square().sum().item() for a, b in zip(expected, actual)) /
                   max(sum(a.double().square().sum().item() for a in expected), 1e-24)) ** .5
    if relative_l2 > 1e-4:
        raise AssertionError(f"Gradient/signal relative L2 error is too large: {relative_l2}")
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=5e-4, atol=1e-4)
    return {"max_abs_error": error, "relative_l2_error": relative_l2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "checkpoints/bert-base-uncased")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="16,32")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--clients", type=int, default=30)
    parser.add_argument("--audit-candidates", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("/tmp/fedsgd_gradient_benchmark.json"))
    args = parser.parse_args()
    batches = [int(v) for v in args.batch_sizes.split(",")]
    if min(batches + [args.chunk_size, args.sequence_length, args.clients, args.audit_candidates, args.repeats, args.cpu_threads]) < 1 or args.warmups < 0:
        parser.error("Sizes/repeats must be positive; warmups must be nonnegative.")
    if args.output.exists():
        parser.error(f"Output already exists; choose a new path: {args.output}")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    model = TransformerAdapterClassifier(str(args.model_path), architecture="bert", num_classes=2, device=device)
    # Nonzero adapter branches also exercise down-projection gradients.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and ".adapter.up.weight" in name:
                p.add_(torch.randn_like(p) * .002)
    client = model.create_client_model(0)
    report = {"torch": torch.__version__, "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "model_path": str(args.model_path.resolve()), "cpu_threads": args.cpu_threads,
              "sequence_length": args.sequence_length, "chunk_size": args.chunk_size,
              "clients": args.clients, "audit_candidates": args.audit_candidates,
              "www_tail_fraction": DEFAULTS["www_tail_fraction"],
              "warmups": args.warmups, "repeats": args.repeats,
              "scope": "Synthetic compute stages; excludes WWW ranking, DP noise, optimizer, uploads, aggregation and other attacks. Gradient timing uses train mode; numerical checks and audit use eval mode.",
              "measurements": []}

    def data(count):
        tokens = torch.randint(100, model.backbone.config.vocab_size, (count, args.sequence_length), device=device)
        mask = torch.ones_like(tokens)
        # Include padding, as in packed CoLA/IMDb records.
        for i in range(count):
            length = args.sequence_length - i % max(1, args.sequence_length // 2)
            mask[i, length:] = 0
            tokens[i, length:] = 0
        return torch.stack((tokens, mask), dim=1), torch.arange(count, device=device) % 2

    def compare(stage, batch, reference, optimized, training=False):
        client.eval()
        error = verify(reference, optimized)
        client.train(training)
        before = measure(reference, device, args.warmups, args.repeats)
        after = measure(optimized, device, args.warmups, args.repeats)
        row = {"stage": stage, "records": batch, "reference": before, "optimized": after,
               "speedup": before["median_seconds"] / after["median_seconds"],
               "equivalence_eval": error}
        report["measurements"].append(row)
        print(json.dumps(row), flush=True)

    with client.use_shared_model():
        named = [(n, p) for n, p in client.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named]
        report["trainable_parameters"] = sum(p.numel() for p in parameters)
        for batch in batches:
            x, y = data(batch)
            weights = ino_weights(torch.arange(batch).float(), tail_fraction=report["www_tail_fraction"], expected_batch_size=batch)[0].to(device)
            compare("www_clipped_gradient", batch,
                    lambda: weighted_clipped_sum(client, x, y, parameters, 8., weights, backend="loop"),
                    lambda: weighted_clipped_sum(client, x, y, parameters, 8., weights, backend="batched", microbatch_size=args.chunk_size), True)
            compare("record_dp_clipped_gradient", batch,
                    lambda: legacy_record_sum(client, x, y, parameters, 8.),
                    lambda: batched_record_sum(client, x, y, parameters, 8., args.chunk_size), True)
        x, y = data(args.audit_candidates)
        updates = [({n: torch.randn(p.shape) for n, p in named}, -1., 1.) for _ in range(args.clients)]
        auditor = MembershipAuditor.__new__(MembershipAuditor)
        auditor.device, auditor.images, auditor.labels = device, x.cpu(), y.cpu()
        auditor.config = {"grad_sample_backend": "batched", "grad_sample_chunk_size": args.chunk_size}
        def audit():
            result = auditor._raw_input_gradient_measurements(
                client, [n for n, _ in named], updates, need_cosine=True,
                need_gradient_difference=True, gradient_difference_update_indices=[0])
            return [result["cosine"], result["gradient_difference"]]
        compare("text_cosine_and_gradient_difference_audit", len(y),
                lambda: legacy_audit(client, x, y, named, updates), audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"Saved benchmark: {args.output}")


if __name__ == "__main__":
    main()
