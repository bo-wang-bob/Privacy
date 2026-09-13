"""Replay two original records from a real selection run on a frozen CPU teacher.

This numerical spot check bypasses RiskSynthesis.sample/candidate_metrics and
the token_features helper. It is not an attack-efficacy measurement.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import sys

import torch
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainmodel.clip_adapter import build_clip_adapter_text_features
from scripts.analyze_risk_synthesis import digest

SNAPSHOT = ROOT / "checkpoints/clip-vit-base-patch32/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


@torch.no_grad()
def run(directory, output):
    torch.set_num_threads(1)
    config_path = directory / "run_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    options = config["defense"]["synthesis"]
    if (config["model_type"] != "clip_adapter" or config["dataset_name"] != "cifar100" or
            options["candidate_selection"] != "least_local_similarity" or options["attempts"] != 2 or
            config.get("model_load_path") is not None):
        raise ValueError("This spot check targets the original-teacher CIFAR100 Adapter two-candidate run.")
    root = directory / "risk_synthesis"
    with (root / "synthetic_exposure.csv").open() as handle:
        key, group = next(itertools.groupby(csv.DictReader(handle), lambda r: (r["round"], r["client"], r["step"])))
        batch = list(group)
    with (root / "candidate_choices.csv").open() as handle:
        candidate_key, group = next(itertools.groupby(csv.DictReader(handle), lambda r: (r["round"], r["client"], r["step"])))
        choices = list(group)
    assert key == candidate_key and key[0] == "1" and key[2] == "0"
    assert all(float(row["used_risk"]) == 0 for row in batch) and len(batch) >= 2
    client = int(key[1])
    state_path = root / f"client_{client}_distribution.pt"
    code_path = root / f"client_{client}_source_codes.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    codes = torch.load(code_path, map_location="cpu", weights_only=True, mmap=True)
    ids = [int(row["sample_id"]) for row in batch]
    generator = torch.Generator().manual_seed(config["seed"] + 1000003*client + 9173)
    generated = {}
    # Reproduce every draw in the first batch so the second-attempt RNG offset
    # is correct. Only the first two records are forwarded through the teacher.
    for attempt in (1, 2):
        for sid in ids:
            label = int(state["labels"][sid])
            noise = torch.zeros_like(codes[sid])
            for factor, weight in ((state["classes"][label]["factor"], 1-options["shrinkage"]),
                                   (state["pooled_factor"], options["shrinkage"])):
                latent = torch.randn(factor.shape[1], generator=generator)
                noise.add_(factor @ latent, alpha=math.sqrt(weight))
            candidate = codes[sid] + options["noise_scale"] * noise
            if sid in ids[:2]:
                generated[(sid, attempt)] = candidate
    model = CLIPModel.from_pretrained(SNAPSHOT, local_files_only=True).eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(SNAPSHOT, local_files_only=True)
    meta = next((ROOT / "data/CIFAR100/data").rglob("meta"))
    names = pickle.loads(meta.read_bytes(), encoding="latin1")["fine_label_names"]
    text = build_clip_adapter_text_features(model, processor, names, "cifar100", torch.device("cpu"))
    cls = model.vision_model.embeddings.class_embedding + model.vision_model.embeddings.position_embedding.weight[0]
    vectors = torch.stack([codes[sid] for sid in ids[:2]] + list(generated.values()))
    tokens = torch.cat((cls.reshape(1, 1, -1).expand(len(vectors), -1, -1), vectors.reshape(-1, 49, 768)), dim=1)
    # Use the public pixel forward with an embeddings-output hook, independently
    # checking the explicit token forward used in training.
    hook = model.vision_model.embeddings.register_forward_hook(lambda module, inputs, result: tokens)
    try:
        features = F.normalize(model.get_image_features(pixel_values=torch.zeros(len(tokens), 3, 224, 224)).float(), dim=-1)
    finally:
        hook.remove()
    torch.testing.assert_close(features[:2], state["semantic_source_features"][ids[:2]], atol=2e-6, rtol=2e-5)
    scores = features @ text.T
    labels = [int(state["labels"][sid]) for sid in ids[:2]] + [int(state["labels"][sid]) for sid, _ in generated]
    own = scores[torch.arange(len(labels)), labels].clone()
    scores[torch.arange(len(labels)), labels] = -torch.inf
    margins = own - scores.max(1).values
    cosine, nearest = (features @ state["semantic_source_features"].T).max(1)
    original_margin = {sid: float(margins[i]) for i, sid in enumerate(ids[:2])}
    logged = {(int(row["sample_id"]), int(row["attempt"])): row for row in choices}
    results = []
    for offset, ((sid, attempt), vector) in enumerate(generated.items(), start=2):
        row = logged[(sid, attempt)]
        delta = float(margins[offset]) - original_margin[sid]
        ratio = float(vector.norm() / codes[sid].norm())
        similarity = float(cosine[offset])
        errors = dict(norm_ratio=abs(ratio-float(row["norm_ratio"])),
            margin_delta=abs(delta-float(row["teacher_margin_delta"])),
            nearest_cosine=abs(similarity-float(row["nearest_teacher_cosine"])))
        assert max(errors.values()) < 2e-6, errors
        assert int(nearest[offset]) == int(row["nearest_teacher_source_id"])
        assert int(delta >= -options["margin_tolerance"]) == int(row["quality_passed"])
        results.append(dict(sample_id=sid, attempt=attempt, recomputed_margin_delta=delta,
            recomputed_nearest_cosine=similarity, nearest_source_id=int(nearest[offset]), absolute_errors=errors))
    decisions = []
    for row in batch[:2]:
        sid = int(row["sample_id"])
        candidates = [r for r in results if r["sample_id"] == sid]
        feasible = [r for r in candidates if r["recomputed_margin_delta"] >= -options["margin_tolerance"]]
        chosen = (min(feasible, key=lambda r: (r["recomputed_nearest_cosine"], -r["recomputed_margin_delta"], r["attempt"]))
                  if feasible else max(candidates, key=lambda r: (r["recomputed_margin_delta"], -r["attempt"])))
        assert chosen["attempt"] == int(row["selected_attempt"])
        decisions.append(dict(sample_id=sid, selected_attempt=chosen["attempt"]))
    result = dict(status="passed_cpu_numerical_spot_check", scope="First two originals, two candidates each, first batch only; no optimizer or MIA measurement.",
        run=str(directory), client=client, results=results, decisions=decisions,
        sources={str(p): digest(p) for p in (config_path, state_path, code_path, meta, SNAPSHOT/"config.json", SNAPSHOT/"pytorch_model.bin", Path(__file__))},
        logged_prefix_sha256=hashlib.sha256(json.dumps(dict(exposure=batch, choices=choices), sort_keys=True).encode()).hexdigest())
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve())
