"""Verify the two existing full-replacement jobs and compare their fixed controls."""
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import analyze, comparison_protocol, digest
from scripts.confirmation_duplicate_sensitivity import run as duplicate_sensitivity
from scripts.paired_synthesis_uncertainty import run as resample
from scripts.run_privacy_experiments import load_yaml

PLAN = ROOT / "analysis_scripts/risk_synthesis_all_study_plan_20260912.json"
STATE = ROOT / "analysis_scripts/risk_synthesis_all_study_analysis_20260912.json"
OUTPUT = ROOT / "analysis_scripts/risk_synthesis_all_study_verified_20260912"
ARMS = ("none", "legacy_partial_risk", "risk", "shuffled_risk")
LABELS = {"none": "无防御", "legacy_partial_risk": "旧版部分替换", "risk": "新版全部替换：真实风险",
          "shuffled_risk": "新版全部替换：打乱风险"}


def now():
    return datetime.now(timezone.utc).isoformat()


def metrics(arm, record):
    mechanism = record.get("synthesis_mechanism", {})
    performance = json.loads((Path(record["path"]) / "performance_summary.json").read_text())
    return dict(arm=arm, run=record["run"], accuracy=record["accuracy"],
        maximum_auc=record["strongest_auc"],
        maximum_direction_symmetric_auc=record["strongest_direction_symmetric_auc"],
        maximum_class_conditional_auc=record["strongest_class_conditional_auc"],
        maximum_tpr_at_1pct=max(a["tpr_at_1pct"] for a in record["attacks"]),
        server_train_minutes=performance["stages"]["run"]["wall_seconds"] / 60,
        synthesis_counts=mechanism.get("counts"))


def render(outcome, records):
    rows = outcome["arms"]
    lines = ["# 全部替换方案：100 轮探索性效果对照", "",
        "CLIP transformer Adapter / CIFAR100，seed=43，目标客户端 0；每类全局 100 张、"
        "10 个 IID 客户端、FedAvg 100 轮、每轮一个完整本地 epoch。四组原始候选及除防御外的实际配置一致。", "",
        "| 方案 | Accuracy | 11 攻击中最大 AUC | 最大 TPR@1%FPR | 服务器训练/审计时间 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f'| {LABELS[row["arm"]]} | {100*row["accuracy"]:.2f}% | {row["maximum_auc"]:.6f} | '
                     f'{100*row["maximum_tpr_at_1pct"]:.2f}% | {row["server_train_minutes"]:.1f} 分钟 |')
    primary = outcome["comparisons"]["risk_vs_none"]
    checks = outcome["primary_checks"]
    lines += ["", f'真实风险全部替换相对无防御：最大 AUC 变化 {primary["maximum_auc_delta"]:+.6f}，'
        f'最大 TPR@1%FPR 变化 {100*primary["maximum_tpr_at_1pct_delta"]:+.2f} 个百分点，'
        f'准确率变化 {100*primary["accuracy_delta"]:+.2f} 个百分点。', "",
        f'预先固定的三项判据（最大 AUC 降低至少 0.02、最大 TPR 下降、准确率下降不超过 2 个百分点）：'
        f'{"本次全部满足" if all(checks.values()) else "本次未全部满足"}。逐项结果：`{json.dumps(checks, ensure_ascii=False)}`。', "",
        "| 攻击 | 无防御 AUC | 旧部分 AUC | 全替换 AUC | 打乱风险 AUC | 全替换−无防御 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lookup = {arm: {a["attack"]: a for a in record["attacks"]} for arm, record in records.items()}
    for name in lookup["none"]:
        values = [lookup[arm][name]["auc"] for arm in ARMS]
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.6f}' for v in values)
                     + f' | {values[2]-values[0]:+.6f} |')
    lines += ["", "`attack_metrics.csv` 保存全部攻击的三个低 FPR 指标、方向对称 AUC 和类别条件 AUC；"
        "`class_metrics.csv` 保存同类内分数与全局阈值下的逐类别 TPR/FPR。", "",
        "风险与打乱风险的比较用于判断风险排序的额外贡献。旧部分替换与新版同时改变了替换范围、"
        "首轮处理和语义失败时的行为，二者差异属于整体协议比较。", "",
        "每个新任务应有 1,000,000 次原始位置访问，全部替换且原图回退为零；语义未达标仍按用户选择"
        "使用最佳有效虚拟候选，次数单独报告。", ""]
    for row in rows:
        if row["arm"] in ("risk", "shuffled_risk"):
            lines.append(f'- {LABELS[row["arm"]]}：`{json.dumps(row["synthesis_counts"], ensure_ascii=False)}`。')
    lines += ["", "本次为一个种子、一个目标客户端，使用此前已研究的数据来源，不是新的独立确认。"
        "候选重采样只描述固定已训练模型下的候选不确定性，不涵盖训练种子变化或选参。"
        "原始像素重复敏感性单独复算，不改变训练或主结果；筛除后每组 998 条，0.1%FPR 指标不可报告。"
        "无形式 DP 保证。时间来自同一服务器范围，包含训练和审计、不含模型/数据初始化；"
        "并行资源竞争和生成协议均可能影响耗时。", "",
        "![全部攻击 AUC 对照](attack_auc.png)", "",
        "具体任务：", ""]
    lines += [f'- {LABELS[arm]}：`{records[arm]["path"]}`' for arm in ARMS]
    (OUTPUT / "effect_readout.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = list(lookup["none"])
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(names))
    legends = ("None", "Legacy partial", "All: real risk", "All: shuffled risk")
    for offset, (arm, label) in enumerate(zip(ARMS, legends)):
        ax.bar(x + (offset-1.5)*.2, [lookup[arm][name]["auc"] for name in names], .2, label=label)
    ax.axhline(.5, color="black", linewidth=.7, linestyle="--")
    ax.set(ylim=(0, 1), ylabel="Attack AUC (lower is better in the registered direction)",
           title="CIFAR100 / CLIP Adapter / 100-round FedAvg / seed 43 / client 0")
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(OUTPUT / "attack_auc.png", dpi=180)
    plt.close(fig)


def main():
    plan = json.loads(PLAN.read_text())
    fingerprint = digest(PLAN)
    analysis_sources = {str(path): digest(path) for path in (Path(__file__),
        ROOT / "scripts/analyze_risk_synthesis.py", ROOT / "scripts/paired_synthesis_uncertainty.py",
        ROOT / "scripts/confirmation_duplicate_sensitivity.py")}
    state = dict(status="waiting_for_existing_training", pid=os.getpid(), started_at_utc=now(),
                 plan_sha256=fingerprint, analysis_sources=analysis_sources)
    with STATE.open("x") as handle:
        json.dump(state, handle, indent=2)

    def save():
        state["updated_at_utc"] = now()
        temporary = STATE.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(STATE)

    try:
        directories = {arm: Path(plan["controls"][arm]["directory"]) for arm in ARMS[:2]}
        pending = {j["arm"]: j for j in plan["jobs"]}
        while pending:
            for arm, expected in list(pending.items()):
                execution = json.loads((ROOT / f"analysis_scripts/risk_synthesis_all_study_{arm}_execution_20260912.json").read_text())
                if execution["plan_sha256"] != fingerprint:
                    raise RuntimeError("Execution belongs to a different frozen plan.")
                if execution["status"] == "failed":
                    raise RuntimeError(f"Training arm failed: {arm}: {execution.get('error')}")
                if execution["status"] != "completed":
                    os.kill(execution["pid"], 0)
                    continue
                actual = execution["job"]
                if (actual["returncode"] != 0 or len(actual["result_directories"]) != 1
                        or any(actual.get(k) != v for k, v in expected.items() if k != "status")):
                    raise RuntimeError("Completed execution differs from planned intervention.")
                path = Path(actual["result_directories"][0])
                config = load_yaml(path / "run_config.yaml")
                if comparison_protocol(config) != expected["protocol"] or config["defense"] != expected["defense"]:
                    raise RuntimeError("Actual saved protocol differs from plan.")
                directories[arm] = path
                del pending[arm]
            if pending:
                time.sleep(15)
        if digest(PLAN) != fingerprint or any(digest(Path(p)) != h for p, h in analysis_sources.items()):
            raise RuntimeError("Frozen plan or analysis implementation changed.")
        for control in plan["controls"].values():
            if any(digest(Path(p)) != h for p, h in control["sources"].items()):
                raise RuntimeError("Reused baseline artifacts changed.")
        state["status"] = "verifying_results"
        save()
        verified = analyze([directories[arm] for arm in ARMS], OUTPUT)
        if len(verified["matched_comparisons"]) != 3 or not all(r["complete"] for r in verified["runs"]):
            raise RuntimeError("Study is incomplete or has unmatched comparisons.")
        records = dict(zip(ARMS, verified["runs"]))
        for arm in ARMS[2:]:
            record = records[arm]
            counts = record["synthesis_mechanism"]["counts"]
            if (record["synthesis"]["implementation"] != "local_token_geometry_v4_all_replacement"
                    or any(counts[k] != 1000000 for k in ("visits", "requested", "accepted"))
                    or counts["fallback"] != 0):
                raise RuntimeError("Full-replacement visit accounting failed.")
        values = {arm: metrics(arm, records[arm]) for arm in ARMS}
        comparisons = {}
        for treatment, control in (("risk", "none"), ("shuffled_risk", "none"),
                                   ("legacy_partial_risk", "none"), ("risk", "shuffled_risk"),
                                   ("risk", "legacy_partial_risk")):
            comparisons[f"{treatment}_vs_{control}"] = {f"{key}_delta": values[treatment][key]-values[control][key]
                for key in ("accuracy", "maximum_auc", "maximum_tpr_at_1pct", "maximum_direction_symmetric_auc",
                            "maximum_class_conditional_auc")}
        primary = comparisons["risk_vs_none"]
        outcome = dict(plan_sha256=fingerprint, arms=list(values.values()), comparisons=comparisons,
            primary_checks=dict(maximum_auc_reduction=primary["maximum_auc_delta"] <= -.02+1e-12,
                maximum_tpr_reduction=primary["maximum_tpr_at_1pct_delta"] < 0,
                accuracy_tolerance=primary["accuracy_delta"] >= -.02-1e-12),
            interpretation=plan["interpretation"], scope=plan["scope"])
        (OUTPUT / "effect_result.json").write_text(json.dumps(outcome, indent=2))
        render(outcome, records)
        state.update(status="verifying_uncertainty_and_duplicates", output=str(OUTPUT), outcome=outcome)
        save()
        print(json.dumps(outcome, indent=2), flush=True)
        duplicate_sensitivity(OUTPUT / "verified_results.json",
            ROOT / "analysis_scripts/risk_synthesis_confirmation_exact_image_identity_20260912.json",
            ROOT / "analysis_scripts/risk_synthesis_all_study_duplicate_sensitivity_20260912")
        for control in ("none", "shuffled_risk", "legacy_partial_risk"):
            resample(OUTPUT / "verified_results.json",
                ROOT / f"analysis_scripts/risk_synthesis_all_study_vs_{control}_resampling_20260912",
                2000, 20260912, treatment=records["risk"]["run"], control_name=records[control]["run"])
        state.update(status="completed", finished_at_utc=now())
        save()
        print(f"Full-replacement effect verification completed: {OUTPUT}", flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
