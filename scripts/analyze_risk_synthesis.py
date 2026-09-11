#!/usr/bin/env python
"""Read explicit new experiment directories, verify scores, emit CSV/JSON/Markdown.

This does not select a winning defense or reinterpret incomplete runs as evidence.
Sources are read-only; the output directory must be new.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import yaml


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_synthesis_mechanism(directory, summary, *, complete):
    """Independently reconcile streamed visits with original IDs and counters.

    Norm and margin statistics describe the last attempted candidate per visit,
    rather than all attempts. No distance is interpreted as a privacy guarantee.
    """
    import torch
    directory = Path(directory)
    states = {}
    exposure = {}
    geometry = []
    for path in sorted(directory.glob("client_*_distribution.pt")):
        client = int(path.stem.split("_")[1])
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        states[client] = state["labels"]
        exposure[client] = {key: torch.zeros(len(state["labels"]), dtype=torch.long)
                            for key in ("risk_reads", "real_steps", "synthetic_steps")}
        sizes = [len(group["indices"]) for group in state["classes"].values()]
        geometry.append(dict(client=client, samples=len(state["labels"]),
                             class_count=len(sizes), min_class_samples=min(sizes),
                             max_class_samples=max(sizes), source_sha256=state["source_sha256"],
                             pooled_metadata=state["pooled_metadata"]))
    totals, reasons = Counter(), Counter()
    groups = defaultdict(Counter)
    bins = {str(i): Counter() for i in range(5)}
    measurements = defaultdict(lambda: dict(count=0, sum=0., min=float("inf"), max=-float("inf")))
    path = directory / "synthetic_exposure.csv"
    with path.open() as handle:
        for row in csv.DictReader(handle):
            client, sid = int(row["client"]), int(row["sample_id"])
            if client not in states or not 0 <= sid < len(states[client]):
                raise ValueError("Synthetic exposure contains an unknown original sample ID.")
            if int(row["label"]) != int(states[client][sid]):
                raise ValueError("Synthetic exposure disagrees with the original sample label.")
            risk, used = float(row["risk"]), float(row["used_risk"])
            requested, accepted, attempts = (int(row[k]) for k in ("requested", "accepted", "attempts"))
            if (not 0 <= risk <= 1 or not 0 <= used <= 1 or requested not in (0, 1)
                    or accepted not in (0, 1) or accepted > requested
                    or not 0 <= attempts <= summary["options"]["attempts"]
                    or (requested and not attempts) or (not requested and attempts)
                    or (row["reason"] == "accepted") != bool(accepted)):
                raise ValueError("Inconsistent synthetic exposure decision.")
            counts = dict(visits=1, requested=requested, accepted=accepted,
                          fallback=requested-accepted)
            totals.update(counts)
            risk_bin = min(4, int(risk * 5))
            bins[str(risk_bin)].update(counts)
            group = groups[(int(row["round"]), client, risk_bin)]
            group.update(counts)
            group["used_risk_sum"] += used
            group["accepted_used_risk_sum"] += accepted * used
            if requested:
                reasons[row["reason"]] += 1
            exposure[client]["risk_reads"][sid] += int(int(row["source_round"]) >= 0)
            exposure[client]["synthetic_steps" if accepted else "real_steps"][sid] += 1
            for field in ("norm_ratio", "teacher_margin_delta"):
                if row.get(field) not in (None, ""):
                    value = float(row[field])
                    if not np.isfinite(value):
                        raise ValueError(f"Nonfinite {field} in exposure diagnostics.")
                    measure = measurements[(field, "accepted" if accepted else "fallback")]
                    measure["count"] += 1
                    measure["sum"] += value
                    measure["min"], measure["max"] = min(measure["min"], value), max(measure["max"], value)
    if complete:
        if dict(totals) != summary["counts"] or {k: dict(v) for k, v in bins.items()} != summary["risk_bins"]:
            raise ValueError("Streamed synthesis counts disagree with the completed summary.")
        saved = torch.load(directory / "source_exposure.pt", map_location="cpu", weights_only=True)
        if set(saved) != set(exposure) or any(
                not torch.equal(saved[c][key], values)
                for c, entry in exposure.items() for key, values in entry.items()):
            raise ValueError("Streamed synthesis visits disagree with original-record exposure counters.")
    rows = []
    for (round_number, client, risk_bin), counts in sorted(groups.items()):
        rows.append(dict(round=round_number, client=client, risk_bin=risk_bin, **counts,
                         requested_fraction=counts["requested"] / counts["visits"],
                         accepted_fraction=counts["accepted"] / counts["visits"],
                         acceptance_given_request=(counts["accepted"] / counts["requested"]
                                                   if counts["requested"] else None)))
    return dict(counts=dict(totals), reasons=dict(reasons), geometry=geometry, groups=rows,
                last_attempt_measurements=[dict(field=field, outcome=outcome, count=values["count"],
                    mean=values["sum"] / values["count"], min=values["min"], max=values["max"])
                    for (field, outcome), values in sorted(measurements.items())],
                completed_counters_verified=bool(complete))


def recompute_auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("AUC needs finite scores and both membership classes.")
    members = int(labels.sum())
    nonmembers = len(labels) - members
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - members*(members+1)/2) / (members*nonmembers))


def recompute_tpr(labels, scores, fpr):
    labels, scores = np.asarray(labels,dtype=int), np.asarray(scores,dtype=float)
    order=np.argsort(-scores,kind="stable")
    ordered=scores[order]
    ends=np.r_[np.where(ordered[:-1] != ordered[1:])[0],len(order)-1]
    tp=np.cumsum(labels[order])[ends]/sum(labels)
    fp=np.cumsum(1-labels[order])[ends]/sum(1-labels)
    valid=tp[fp<=fpr]
    return float(valid.max()) if len(valid) else 0.


def selection_digest(path):
    import torch
    def canonical(value):
        if isinstance(value,torch.Tensor):
            return dict(shape=list(value.shape),dtype=str(value.dtype),values=value.tolist())
        if isinstance(value,dict):
            return {str(k):canonical(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            return [canonical(v) for v in value]
        return value
    value=torch.load(path,map_location="cpu",weights_only=True)
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True).encode()).hexdigest()


def read_run(directory):
    directory = Path(directory).resolve()
    config_path = directory / "run_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    sources = {str(config_path): digest(config_path)}
    metrics_path = directory / "training_metrics.csv"
    metrics = list(csv.DictReader(metrics_path.open())) if metrics_path.exists() else []
    if metrics_path.exists():
        sources[str(metrics_path)] = digest(metrics_path)
    last = metrics[-1] if metrics else {}
    actual_round = int(last.get("round", 0))
    summary_path = directory / "privacy_audit" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary_path.exists():
        sources[str(summary_path)] = digest(summary_path)
    if len(summary.get("audit_client_ids", [0])) != 1:
        raise ValueError("This study verifier currently expects one audited client per run.")
    synthesis_path = directory / "risk_synthesis" / "synthesis_summary.json"
    synthesis = json.loads(synthesis_path.read_text()) if synthesis_path.exists() else None
    if synthesis_path.exists():
        sources[str(synthesis_path)] = digest(synthesis_path)
    performance_path=directory/"performance_summary.json"
    performance=json.loads(performance_path.read_text()) if performance_path.exists() else {}
    if performance_path.exists():
        sources[str(performance_path)]=digest(performance_path)
    protocol = {k:config.get(k) for k in (
        "model_type", "dataset_name", "seed", "fpl_shots", "use_full_dataset", "partition_mode",
        "dirichlet_alpha", "total_users", "sample_users", "aggregator", "aggregation_weighting",
        "num_global_iters", "local_epochs", "batch_size", "learning_rate", "learning_rate_decay",
        "clip_adapter", "clip_lora", "fedavg",
    )}
    protocol["audit"] = {k:config.get("audit",{}).get(k) for k in (
        "audit_client_ids", "target_client_id", "attacks", "attack_audit_intervals",
        "candidate_sampling", "nonmember_to_member_ratio",
    )}
    protocol["validation_fraction"] = config.get("defense",{}).get("cofedmid_validation_fraction",0)
    key = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    completed = (actual_round == int(config["num_global_iters"]) and bool(summary)
                 and not summary.get("errors") and performance.get("status")=="completed"
                 and {a["attack"] for a in summary.get("attacks",[])} == set(config.get("audit",{}).get("attacks",[])))
    if synthesis is not None:
        completed = completed and synthesis.get("status") == "completed"
    result = dict(run=directory.name, path=str(directory), protocol=protocol, comparison_key=key,
                  defense=config.get("defense",{}).get("name","none"),
                  synthesis_options=config.get("defense",{}).get("synthesis"),
                  expected_rounds=config["num_global_iters"], actual_round=actual_round,
                  accuracy=float(last["accuracy"]) if last else None,
                  complete=completed, errors=summary.get("errors"), synthesis=synthesis,
                  sources=sources, attacks=[])
    if synthesis is not None and (directory / "risk_synthesis" / "synthetic_exposure.csv").exists():
        result["synthesis_mechanism"] = read_synthesis_mechanism(
            directory / "risk_synthesis", synthesis, complete=completed)
        for name in ("synthetic_exposure.csv", "source_exposure.pt"):
            source = directory / "risk_synthesis" / name
            if source.exists():
                sources[str(source)] = digest(source)
    predictions_path = directory / "privacy_audit" / "predictions.csv"
    predictions = defaultdict(list)
    if predictions_path.exists():
        for row in csv.DictReader(predictions_path.open()):
            predictions[row["attack"]].append(row)
        sources[str(predictions_path)] = digest(predictions_path)
    for attack in summary.get("attacks",[]):
        name = attack["attack"]
        rows = predictions[name]
        ids = [(r["audit_client_id"],r["sample_index"],r["membership"]) for r in rows]
        if not rows or len(set(ids)) != len(ids):
            raise ValueError(f"Missing or duplicate prediction rows: {directory.name}/{name}")
        labels = [int(r["membership"]) for r in rows]
        scores = [float(r["score"]) for r in rows]
        if sum(labels) != attack["member_count"] or len(rows)-sum(labels) != attack["nonmember_count"]:
            raise ValueError(f"Candidate count mismatch: {directory.name}/{name}")
        auc = recompute_auc(labels,scores)
        if not np.isclose(auc,attack["auc"],atol=1e-6,rtol=0):
            raise ValueError(f"Independent AUC verification failed: {directory.name}/{name}")
        reportable = attack.get("reportable_metrics",{})
        for target in (.01,.001):
            reported=reportable.get(f"tpr_at_fpr_{target:g}")
            if reported is not None and not np.isclose(recompute_tpr(labels,scores,target),reported,atol=1e-6,rtol=0):
                raise ValueError(f"Independent TPR verification failed: {directory.name}/{name}")
        result["attacks"].append(dict(
            attack=name,auc=auc,members=sum(labels),nonmembers=len(rows)-sum(labels),
            direction_symmetric_auc=max(auc,1-auc),
            fpr_resolution=1/(len(rows)-sum(labels)),
            tpr_at_1pct=reportable.get("tpr_at_fpr_0.01"),
            tpr_at_01pct=reportable.get("tpr_at_fpr_0.001"),
            independent_auc_verified=True,
        ))
    result["strongest_auc"] = max((a["auc"] for a in result["attacks"]),default=None)
    result["strongest_direction_symmetric_auc"] = max(
        (a["direction_symmetric_auc"] for a in result["attacks"]), default=None)
    # Candidate identities are checked independently of scores, with source/index metadata.
    selection = summary.get("candidate_sampling",{}).get("per_client",{})
    result["candidate_metadata"] = selection
    result["candidate_selection_digests"]={}
    for name in ("candidate_selection.pt","client_train_update_candidate_selection.pt"):
        path=directory/"privacy_audit"/name
        if path.exists():
            result["candidate_selection_digests"][name]=selection_digest(path)
            sources[str(path)]=digest(path)
    return result


def analyze(directories,output):
    runs = [read_run(p) for p in directories]
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for run in runs:
        for attack in run["attacks"]:
            rows.append(dict(run=run["run"],defense=run["defense"],complete=run["complete"],
                             accuracy=run["accuracy"],**attack))
    if rows:
        with (output/"attack_metrics.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    mechanism_rows = [dict(run=run["run"], implementation=run["synthesis"]["implementation"], **group)
                      for run in runs for group in run.get("synthesis_mechanism", {}).get("groups", [])]
    if mechanism_rows:
        with (output / "synthesis_mechanism.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mechanism_rows[0]))
            writer.writeheader()
            writer.writerows(mechanism_rows)
    comparisons=[]
    for run in runs:
        controls=[b for b in runs if b["defense"]=="none" and b["comparison_key"]==run["comparison_key"]]
        if run["defense"]=="none" or not run["complete"] or len(controls)!=1 or not controls[0]["complete"]:
            continue
        base=controls[0]
        if (base["candidate_metadata"] != run["candidate_metadata"] or
                not base["candidate_selection_digests"] or
                base["candidate_selection_digests"] != run["candidate_selection_digests"]):
            raise ValueError("Matched configs but differing candidate metadata; inspect identity before comparison.")
        lookup={a["attack"]:a for a in base["attacks"]}
        comparisons.append(dict(run=run["run"],control=base["run"],
            accuracy_delta=run["accuracy"]-base["accuracy"],
            strongest_auc_delta=run["strongest_auc"]-base["strongest_auc"],
            strongest_direction_symmetric_auc_delta=(run["strongest_direction_symmetric_auc"]
                                                    -base["strongest_direction_symmetric_auc"]),
            attack_auc_deltas={a["attack"]:a["auc"]-lookup[a["attack"]]["auc"] for a in run["attacks"]}))
    payload=dict(runs=runs,matched_comparisons=comparisons,
                 interpretation="Exploratory unless protocol and independent confirmation are separately established.")
    (output/"verified_results.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False))
    lines=["# 风险生成实验核验","","来源为明确列出的任务目录；部分完成任务不能用于效果结论。", "",
           "| 任务 | 已完成轮数 | 状态 | Accuracy | 最强攻击 AUC |",
           "| --- | ---: | --- | ---: | ---: |"]
    for r in runs:
        accuracy="—" if r["accuracy"] is None else f'{r["accuracy"]*100:.2f}%'
        auc="—" if r["strongest_auc"] is None else f'{r["strongest_auc"]:.4f}'
        lines.append(f'| {r["run"]} | {r["actual_round"]}/{r["expected_rounds"]} | {"完成" if r["complete"] else "未完成/失败"} | {accuracy} | {auc} |')
    lines += ["",f"符合配置、候选元数据与完成状态要求的基线配对：{len(comparisons)} 组。",
              "", "所有已输出 AUC 均从 predictions.csv 用平均秩公式独立复算；低 FPR 指标仅采用可报告值。",
              "结论仍需逐攻击比较、效用容差、风险消融和未用于选参的确认实验支持。"]
    for run in runs:
        mechanism = run.get("synthesis_mechanism")
        if mechanism is not None:
            counts = mechanism["counts"]
            lines += ["", f'生成机制 `{run["run"]}`：访问 {counts.get("visits", 0)} 次，'
                      f'请求 {counts.get("requested", 0)} 次，接受 {counts.get("accepted", 0)} 次。',
                      f'请求的最终结果：{json.dumps(mechanism["reasons"], ensure_ascii=False)}。',
                      "范数和 margin 统计仅对应每次访问的最后一次尝试；接受率本身不表示隐私改善。"]
    (output/"readout.md").write_text("\n".join(lines)+"\n")
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",nargs="+",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=analyze(args.runs,args.output)
    print(f'Verified {len(result["runs"])} runs; matched comparisons: {len(result["matched_comparisons"])}; output: {args.output}')


if __name__=="__main__":
    main()
