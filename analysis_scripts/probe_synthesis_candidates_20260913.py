"""Frozen-teacher candidate-count probe; no training and no attack-effect claim."""
import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import digest
from trainmodel.clip_adapter import build_clip_adapter_text_features
from trainmodel.clip_tokens import token_features

RUN = ROOT / "results/2026-09-12_17-45-01-206560_clip_adapter_cifar100_fedavg_risk_synthesis_seed43_target0_e61674f8b2"
OUTPUT = ROOT / "analysis_scripts/risk_synthesis_candidate_probe_20260913"
SNAPSHOT = ROOT / "checkpoints/clip-vit-base-patch32/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    device = torch.device("cuda:1")
    OUTPUT.mkdir(exist_ok=False)
    plan = dict(scope="All 1000 original records of existing target client 0, frozen original CLIP teacher only.",
        risks=[.5,.9], candidates_per_source=4, seed=20260913,
        comparisons=["first_passing_of_2_else_best_semantic", "first_passing_of_4_else_best_semantic",
                     "minimum_source_identity_margin_among_passing_4_else_best_semantic"],
        identity_margin="cosine(candidate, original) minus maximum cosine(candidate, any other same-class original)",
        constraints="Original semantic margin minus .02; norm ratio [.1,2]; all outputs generated. "
                    "The bank has four independent fresh noise draws per source. These are not replayed training candidates. "
                    "No model updates, new attack candidates, or effectiveness inference.")
    (OUTPUT/"probe_plan.json").write_text(json.dumps(plan,indent=2))
    paths = [RUN/"risk_synthesis/client_0_distribution.pt",RUN/"risk_synthesis/client_0_source_codes.pt",
             SNAPSHOT/"pytorch_model.bin",SNAPSHOT/"config.json",Path(__file__)]
    sources = {str(p):digest(p) for p in paths}
    state = torch.load(paths[0],map_location="cpu",weights_only=True,mmap=True)
    codes = torch.load(paths[1],map_location="cpu",weights_only=True,mmap=True)
    meta = next((ROOT/"data/CIFAR100/data").rglob("meta"))
    sources[str(meta)] = digest(meta)
    classnames = pickle.loads(meta.read_bytes(),encoding="latin1")["fine_label_names"]
    model = CLIPModel.from_pretrained(SNAPSHOT,local_files_only=True).to(device).eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(SNAPSHOT,local_files_only=True)
    text = build_clip_adapter_text_features(model,processor,classnames,"cifar100",device)
    cls = model.vision_model.embeddings.class_embedding+model.vision_model.embeddings.position_embedding.weight[0]

    def features(vectors):
        result=[]
        for chunk in vectors.split(32):
            tokens=torch.cat((cls.reshape(1,1,-1).expand(len(chunk),-1,-1),
                              chunk.to(device).reshape(-1,49,768)),dim=1)
            result.append(F.normalize(token_features(model,tokens),dim=-1).cpu())
        return torch.cat(result)

    original_features=features(codes)
    for c,g in state["classes"].items():
        torch.testing.assert_close(original_features[g["indices"]].mean(0),state["semantic_class_means"][c],
                                   atol=2e-6,rtol=2e-5)
    original_scores=original_features @ text.cpu().T
    original_own=original_scores.gather(1,state["labels"][:,None]).squeeze(1)
    original_scores.scatter_(1,state["labels"][:,None],-torch.inf)
    original_margins=original_own-original_scores.max(1).values
    print("Original frozen-teacher class means reproduce saved training statistics",flush=True)
    rows=[]
    for risk in plan["risks"]:
        for c,g in state["classes"].items():
            ids=g["indices"]
            candidates=[]
            for sid in ids.tolist():
                generator=torch.Generator().manual_seed(plan["seed"]+sid*1009)
                anchor=(len(ids)*g["mean"]-codes[sid])/(len(ids)-1)
                factors=torch.cat((g["factor"],state["pooled_factor"]),dim=1)
                noise=torch.randn(4,factors.shape[1],generator=generator) @ factors.T
                candidates.append((1-risk)*codes[sid]+risk*anchor+.1*(.5**.5)*noise)
            candidates=torch.stack(candidates)
            z=features(candidates.flatten(0,1)).reshape(len(ids),4,-1)
            class_scores=z@text.cpu().T
            own=class_scores[:,:,c].clone();class_scores[:,:,c]=-torch.inf
            margins=own-class_scores.max(2).values
            ratios=candidates.norm(dim=2)/codes[ids].norm(dim=1)[:,None]
            valid=(ratios>=.1)&(ratios<=2)&torch.isfinite(candidates).all(2)
            assert valid.all(),"Probe bank contains invalid geometry; report rather than silently change budget."
            passed=margins>=original_margins[ids,None]-.02
            similarity=z@original_features[ids].T
            identity=torch.empty((len(ids),4))
            for offset,sid in enumerate(ids.tolist()):
                own_similarity=similarity[offset,:,offset].clone()
                similarity[offset,:,offset]=-torch.inf
                identity[offset]=own_similarity-similarity[offset].max(1).values
                for method in plan["comparisons"]:
                    k=2 if method.startswith("first_passing_of_2") else 4
                    eligible=torch.where(passed[offset,:k])[0]
                    if len(eligible):
                        selected=int(eligible[identity[offset,eligible].argmin()]) if method.startswith("minimum") else int(eligible[0])
                    else:
                        selected=int(margins[offset,:k].argmax())
                    rows.append(dict(sample_id=sid,label=c,risk=risk,method=method,
                        quality_passed=int(passed[offset,selected]),source_identity_margin=float(identity[offset,selected]),
                        original_is_nearest_same_class=int(identity[offset,selected]>0),
                        teacher_margin_delta=float(margins[offset,selected]-original_margins[sid]),selected=selected+1,
                        bank_semantic_margin_std=float(margins[offset,:k].std(unbiased=False))))
        print(f"Finished fixed-risk probe r={risk}",flush=True)
    with (OUTPUT/"candidate_metrics.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[]
    for risk in plan["risks"]:
        for method in plan["comparisons"]:
            subset=[r for r in rows if r['risk']==risk and r['method']==method]
            summaries.append(dict(risk=risk,method=method,samples=len(subset),
                **{k:float(np.mean([r[k] for r in subset])) for k in
                   ("quality_passed","original_is_nearest_same_class","source_identity_margin",
                    "teacher_margin_delta","bank_semantic_margin_std")}))
    result=dict(status="completed_frozen_teacher_probe",plan=plan,sources=sources,results=summaries,
                interpretation="Semantic feasibility and identity-proximity diagnostics only. Neither training accuracy "
                               "nor membership-attack effectiveness of a new defense was measured.")
    (OUTPUT/"probe_result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=="__main__":
    main()
