#!/usr/bin/env python
"""Apply a frozen confirmation plan to all of its completed, verified runs.

This entry only analyzes results. Training remains in run_privacy_experiments.py.
Incomplete or unmatched studies cannot produce an effectiveness decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import analyze, comparison_protocol, digest
from scripts.run_privacy_experiments import load_yaml

ARMS = {"none", "risk", "shuffled_risk", "www"}
PAIRS = (("risk", "none"), ("risk", "shuffled_risk"), ("risk", "www"),
         ("www", "none"), ("shuffled_risk", "none"))


def finite_metric(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"Missing or invalid reportable metric: {name}")
    return float(value)


def validate_design(jobs):
    keys = [(job["seed"], job["arm"]) for job in jobs]
    seeds = sorted({seed for seed, _ in keys})
    if (len(seeds) < 3 or len(keys) != len(set(keys))
            or set(keys) != {(seed, arm) for seed in seeds for arm in ARMS}):
        raise ValueError("Confirmation requires every four-arm seed block, without duplicates, for at least three seeds.")
    return seeds


def summarize_records(jobs, runs, acceptance):
    """Compute seed-level paired effects; never pool seeds as extra candidates."""
    seeds = validate_design(jobs)
    if len(runs) != len(jobs) or len({run["path"] for run in runs}) != len(runs):
        raise ValueError("Exactly one distinct verified run is required for every planned job.")
    indexed, metrics, per_attack = {}, [], []
    for job, run in zip(jobs, runs):
        if (not run["complete"] or run["protocol"] != job["protocol"]
                or not run.get("confirmation_source", {}).get("roles_disjoint")
                or not run["confirmation_source"].get("exploration_records_excluded")):
            raise ValueError("An incomplete, mismatched or unverified-source run cannot enter confirmation.")
        attacks = {attack["attack"]: attack for attack in run["attacks"]}
        if (len(attacks) != len(run["attacks"]) or len(attacks) != 11
                or set(attacks) != set(job["protocol"]["audit"]["attacks"])):
            raise ValueError("Every planned attack must be present exactly once.")
        aucs = [finite_metric(a["auc"], "auc") for a in attacks.values()]
        tprs = [finite_metric(a["tpr_at_1pct"], "tpr_at_1pct") for a in attacks.values()]
        class_aucs = [finite_metric(a["class_conditional_auc"], "class_conditional_auc")
                      for a in attacks.values()]
        metrics.append(dict(seed=job["seed"], arm=job["arm"], run=run["run"],
            accuracy=finite_metric(run["accuracy"], "accuracy"), maximum_auc=max(aucs),
            maximum_direction_symmetric_auc=max(max(a, 1-a) for a in aucs),
            maximum_tpr_at_1pct=max(tprs), maximum_class_conditional_auc=max(class_aucs)))
        indexed[job["seed"], job["arm"]] = (run, metrics[-1], attacks)
    differences = []
    for seed in seeds:
        base = indexed[seed, "none"][0]
        for arm in ARMS:
            candidate = indexed[seed, arm][0]
            if (base["comparison_key"] != candidate["comparison_key"]
                    or not base["candidate_selection_digests"]
                    or base["candidate_selection_digests"] != candidate["candidate_selection_digests"]
                    or base["candidate_metadata"] != candidate["candidate_metadata"]
                    or base["confirmation_source"] != candidate["confirmation_source"]):
                raise ValueError("Paired seed arms have different candidate identities or protocols.")
        for treatment, control in PAIRS:
            _, a, attacks_a = indexed[seed, treatment]
            _, b, attacks_b = indexed[seed, control]
            differences.append(dict(seed=seed, treatment=treatment, control=control,
                **{f"{key}_delta": a[key]-b[key] for key in (
                    "accuracy", "maximum_auc", "maximum_direction_symmetric_auc",
                    "maximum_tpr_at_1pct", "maximum_class_conditional_auc")}))
            for name in sorted(attacks_a):
                per_attack.append(dict(seed=seed, treatment=treatment, control=control, attack=name,
                    auc_delta=attacks_a[name]["auc"]-attacks_b[name]["auc"],
                    tpr_at_1pct_delta=attacks_a[name]["tpr_at_1pct"]-attacks_b[name]["tpr_at_1pct"],
                    class_conditional_auc_delta=attacks_a[name]["class_conditional_auc"]-attacks_b[name]["class_conditional_auc"]))
    aggregates = []
    for treatment, control in PAIRS:
        rows = [row for row in differences if row["treatment"] == treatment and row["control"] == control]
        for metric in [key for key in rows[0] if key.endswith("_delta")]:
            values = [row[metric] for row in rows]
            aggregates.append(dict(treatment=treatment, control=control, metric=metric,
                seeds=len(values), mean=sum(values)/len(values), minimum=min(values), maximum=max(values),
                negative_seeds=sum(value < 0 for value in values), positive_seeds=sum(value > 0 for value in values)))
    primary = [row for row in differences if row["treatment"] == "risk" and row["control"] == "none"]
    mean_auc = sum(row["maximum_auc_delta"] for row in primary)/len(primary)
    mean_tpr = sum(row["maximum_tpr_at_1pct_delta"] for row in primary)/len(primary)
    checks = dict(
        mean_maximum_auc_reduction=mean_auc <= -acceptance["mean_auc_reduction_at_least"] + 1e-12,
        every_seed_maximum_auc_decreases=(not acceptance["each_seed_auc_must_decrease"]
            or all(row["maximum_auc_delta"] < 0 for row in primary)),
        mean_maximum_tpr_decreases=(not acceptance["mean_max_tpr_at_1pct_must_decrease"] or mean_tpr < 0),
        every_seed_accuracy_within_tolerance=all(row["accuracy_delta"] >=
            -acceptance["each_seed_accuracy_drop_at_most"] - 1e-12 for row in primary))
    return dict(seeds=seeds, run_metrics=metrics, paired_seed_effects=differences,
                paired_attack_effects=per_attack, seed_aggregates=aggregates,
                preregistered_checks=checks, preregistered_criteria_met=all(checks.values()),
                orientation_diagnostic_worsens_in_any_seed=any(
                    row["maximum_direction_symmetric_auc_delta"] > 0 for row in primary),
                interpretation="Criteria concern the fixed composite in this model/dataset/client/seed scope. "
                    "They do not prove a risk-ordering benefit, every-attack improvement, population robustness or formal DP. "
                    "Seed averages and ranges are descriptive; overlapping source pools and fixed class counts are retained. "
                    "Candidate resampling, if reported separately, conditions on trained models and is not seed uncertainty.")


def resolve_completed_study(plan_path, state_paths):
    plan_hash = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    validate_design(plan["jobs"])
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported confirmation plan schema.")
    execution = {}
    for state_path in state_paths:
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != plan_hash or Path(state["plan"]).resolve() != plan_path.resolve():
            raise ValueError("Execution state does not identify this frozen plan.")
        if state["status"] != "completed":
            raise ValueError(f"Confirmation lane is not complete: {state_path}")
        for job in state["jobs"]:
            if job["name"] in execution:
                raise ValueError("A planned job appears in multiple execution states.")
            execution[job["name"]] = job
    if set(execution) != {job["name"] for job in plan["jobs"]}:
        raise ValueError("Completed execution states do not cover the entire frozen plan.")
    if digest(Path(plan["confirmation_manifest"])) != plan["confirmation_manifest_sha256"]:
        raise ValueError("Frozen confirmation source manifest changed.")
    directories = []
    for planned in plan["jobs"]:
        actual = execution[planned["name"]]
        if (actual["status"] != "completed" or actual["returncode"] != 0
                or len(actual.get("result_directories", [])) != 1
                or any(actual.get(key) != value for key, value in planned.items() if key != "status")):
            raise ValueError(f"Execution differs from frozen job: {planned['name']}")
        directory = Path(actual["result_directories"][0]).resolve()
        config = load_yaml(directory / "run_config.yaml")
        if (comparison_protocol(config) != planned["protocol"] or config["defense"] != planned["defense"]
                or config["confirmation_split_sha256"] != plan["confirmation_manifest_sha256"]):
            raise ValueError(f"Actual run configuration differs from frozen job: {planned['name']}")
        directories.append(directory)
    if len(set(directories)) != len(directories):
        raise ValueError("One result directory cannot stand for multiple confirmation jobs.")
    return plan, directories


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(plan_path, state_paths, output):
    # Check every lane first. Never create a partial-study effectiveness report.
    plan, directories = resolve_completed_study(plan_path, state_paths)
    output.mkdir(parents=True, exist_ok=False)
    verified = analyze(directories, output / "verified")
    report = summarize_records(plan["jobs"], verified["runs"], plan["acceptance"])
    report.update(plan=str(plan_path.resolve()), plan_sha256=digest(plan_path), scope=plan["scope"],
                  acceptance=plan["acceptance"], sources={str(p.resolve()): digest(p) for p in
                    [plan_path, *state_paths, output/"verified/verified_results.json", Path(__file__)]})
    with (output / "confirmation_report.json").open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    for key in ("run_metrics", "paired_seed_effects", "paired_attack_effects", "seed_aggregates"):
        write_csv(output / f"{key}.csv", report[key])
    outcome = "满足" if report["preregistered_criteria_met"] else "未满足"
    lines = [f"固定范围内的事前验收标准：{outcome}。", "", plan["scope"], "",
             "| 种子 | 对照组 | Accuracy | 最大攻击 AUC | 允许翻转的最大 AUC | 最大 TPR@1%FPR | 最大同类别 AUC |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(report["run_metrics"], key=lambda row: (row["seed"], row["arm"])):
        lines.append(f"| {row['seed']} | {row['arm']} | {row['accuracy']:.2%} | {row['maximum_auc']:.4f} | "
                     f"{row['maximum_direction_symmetric_auc']:.4f} | "
                     f"{row['maximum_tpr_at_1pct']:.2%} | {row['maximum_class_conditional_auc']:.4f} |")
    lines += ["", "各检查项：", ""]
    lines += [f"- {key}: {'通过' if value else '未通过'}" for key, value in report["preregistered_checks"].items()]
    if report["orientation_diagnostic_worsens_in_any_seed"]:
        lines += ["", "至少一个种子在允许翻转攻击分数后，最大 AUC 相对无防御上升。"
                  "即使上述正式攻击的事前标准通过，也不能据此宣称所有可用攻击的识别能力下降；该诊断须单独解释。"]
    lines += ["", "最大值均在每个模型、每个种子的全部 11 种攻击上重新计算，未固定为原最强攻击。"
              "逐攻击和配对种子结果见 CSV；风险/打乱风险与 WWW 比较分别报告，不由整体对无防御的结果代替。", "",
              "三个种子的均值与范围是描述性结果，不是总体置信区间。该验收不代表每种攻击都改善，"
              "不建立形式 DP，也不能外推到其他模型、数据集或目标客户端。", "",
              "所有来源、完整配置、原始候选身份与生成暴露核验见 verified/verified_results.json。"]
    (output / "readout.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(output=str(output), criteria_met=report["preregistered_criteria_met"],
                         checks=report["preregistered_checks"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--states", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.states, args.output)
