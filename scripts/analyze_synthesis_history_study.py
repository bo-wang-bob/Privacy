"""Analyze the complete frozen compact-synthesis study without changing results."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import analyze, digest
from scripts.run_synthesis_history_study import STUDY, check_sources
from scripts.run_synthesis_history_study import save
from scripts.paired_synthesis_uncertainty import run as resample
from scripts.confirmation_duplicate_sensitivity import run as duplicate_sensitivity

LABELS = {"none": "无防御", "baseline": "原全替换", "baseline_shuffle": "原打乱",
          "history": "仅历史因子", "selection": "仅候选选择", "combined": "简化组合",
          "combined_shuffle": "组合整体打乱"}


def metrics(key, record):
    arm, seed = key.rsplit("_", 1)
    performance = json.loads((Path(record["path"]) / "performance_summary.json").read_text())
    mechanism = record.get("synthesis_mechanism", {})
    return dict(key=key, arm=arm, seed=int(seed), run=record["run"], accuracy=record["accuracy"],
        maximum_auc=record["strongest_auc"], maximum_tpr_at_1pct=max(a["tpr_at_1pct"] for a in record["attacks"]),
        maximum_direction_symmetric_auc=record["strongest_direction_symmetric_auc"],
        maximum_class_conditional_auc=record["strongest_class_conditional_auc"],
        server_train_minutes=performance["stages"]["run"]["wall_seconds"] / 60,
        synthesis_counts=mechanism.get("counts"))


def compare(treatment, control, records):
    a, b = metrics(treatment, records[treatment]), metrics(control, records[control])
    if records[treatment]["comparison_key"] != records[control]["comparison_key"] or (
            records[treatment]["candidate_selection_digests"] != records[control]["candidate_selection_digests"]):
        raise ValueError("Comparison has different protocols or original candidate identities.")
    changes = {name + "_delta": a[name] - b[name] for name in
               ("accuracy", "maximum_auc", "maximum_tpr_at_1pct", "server_train_minutes")}
    lhs = {row["attack"]: row for row in records[treatment]["attacks"]}
    rhs = {row["attack"]: row for row in records[control]["attacks"]}
    return dict(treatment=treatment, control=control, **changes,
        auc_increased_attacks=[name for name in lhs if lhs[name]["auc"] > rhs[name]["auc"]],
        tpr_increased_attacks=[name for name in lhs if lhs[name]["tpr_at_1pct"] > rhs[name]["tpr_at_1pct"]],
        overall_vs_none_passed=(changes["maximum_auc_delta"] <= -.02 + 1e-12 and
            changes["maximum_tpr_at_1pct_delta"] < 0 and changes["accuracy_delta"] >= -.02 - 1e-12),
        added_value_passed=(changes["maximum_auc_delta"] < 0 and changes["maximum_tpr_at_1pct_delta"] < 0
                            and changes["accuracy_delta"] >= -.02 - 1e-12))


def render(output, outcome, records):
    rows = outcome["arms"]
    lines = ["# 简化历史暴露与候选选择：百轮验证结果", "", outcome["scope"], "",
        "每类全局100张、10个IID客户端、FedAvg100轮、每轮完整本地epoch、目标客户端0；"
        "每组审计1000个原始成员与1000个独立evaluation非成员。所有正式攻击结果独立复算。", "",
        "| seed | 方案 | Accuracy | 最大AUC | 最大TPR@1%FPR | 训练和审计分钟 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda r: (r["seed"], list(LABELS).index(r["arm"]))):
        lines.append(f'| {row["seed"]} | {LABELS[row["arm"]]} | {100*row["accuracy"]:.2f}% | '
            f'{row["maximum_auc"]:.6f} | {100*row["maximum_tpr_at_1pct"]:.2f}% | {row["server_train_minutes"]:.1f} |')
    lines += ["", "## 预定判据", ""]
    for kind, label in (("vs_none", "组合相对无防御整体标准"), ("vs_baseline", "组合相对原全替换新增收益"),
                        ("vs_shuffle", "组合相对整体打乱排序收益")):
        entry = outcome["three_seed_checks"][kind]
        lines.append(f'- {label}：通过 {entry["passed_seeds"]}/3 个种子；三种子均通过={entry["all_seeds_passed"]}。')
    lines += ["", "整体标准：最大AUC至少降低0.02、最大TPR降低、准确率损失不超过2个百分点。"
        "新增收益与排序收益：每个种子最大AUC和最大TPR均降低、准确率损失不超过2个百分点。"
        "同方向不等于普遍或因果保证；组件独立消融只有seed43。", "",
        "| 比较 | AUC变化 | TPR变化（百分点） | Accuracy变化（百分点） | TPR升高的攻击数 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in outcome["comparisons"]:
        lines.append(f'| {row["treatment"]} − {row["control"]} | {row["maximum_auc_delta"]:+.6f} | '
            f'{100*row["maximum_tpr_at_1pct_delta"]:+.2f} | {100*row["accuracy_delta"]:+.2f} | '
            f'{len(row["tpr_increased_attacks"])} |')
    lines += ["", "## 超参数和解释限制", "",
        "新增可调数值超参数为0：使用历史零风险频率，去掉EMA衰减率、保留幂次和可调融合权重；"
        "固定等权中秩合并，复用现有两候选上限。几何秩、收缩、噪声强度及语义门槛仍沿用旧值，"
        "不能宣称整个方法无超参数。固定等权和零风险事件也是设计假设。", "",
        "候选选择在语义可行者中降低与所有本地原始记录的最近教师相似度；该距离是代理指标，"
        "不代表成员推理成功率。所有候选语义失败仍保留语义最佳有效替身，原图回退必须为零。", "",
        "旧方案合格后可提前停止，新选择必须评价两个候选，时间比较包含这一实际额外工作；"
        "服务器计时含训练/审计，不含模型和数据初始化，资源竞争可能影响时间。", "",
        "来源清单和这些种子此前已研究，本次为预先冻结的新方法对照；不能称未接触数据确认或跨模型验证。"
        "配对候选重采样区间仅条件于固定已训练模型，不包含训练种子变化或方法选择。无形式DP保证。", "",
        "## 核验产物", "",
        "逐攻击和类内指标见 `metrics/attack_metrics.csv`、`metrics/class_metrics.csv`；"
        "完整原始身份和复算见 `metrics/verified_results.json`。每个新组件任务的历史/候选决策重放"
        "在研究目录对应的 `*_mechanism.json`；配对候选区间与精确像素重复敏感性单独保存。", ""]
    for row in rows:
        if row["synthesis_counts"]:
            lines.append(f'- {row["key"]}：`{json.dumps(row["synthesis_counts"], ensure_ascii=False)}`。')
    (output / "readout.md").write_text("\n".join(lines) + "\n")


def run(study, output):
    plan_path = study / "plan.json"
    plan = json.loads(plan_path.read_text())
    check_sources(plan)
    directories = {key: Path(control["directory"]) for key, control in plan["controls"].items()}
    mechanism_evidence = {}
    for job in plan["jobs"]:
        path = study / f'{job["id"]}.json'
        if not path.exists():
            raise RuntimeError(f'Job not started: {job["id"]}')
        state = json.loads(path.read_text())
        if state["status"] != "completed" or state.get("returncode") != 0 or len(state["result_directories"]) != 1:
            raise RuntimeError(f'Job not completed: {job["id"]} ({state["status"]})')
        if state["plan_sha256"] != digest(plan_path):
            raise ValueError("Job belongs to another frozen plan.")
        directories[job["id"]] = Path(state["result_directories"][0])
        if job["arm"] != "baseline":
            source = study / f'{job["id"]}_mechanism.json'
            validation = json.loads(source.read_text())
            if validation["status"] != "verified":
                raise ValueError("Unverified history/selection mechanism.")
            for name, expected in validation["source_hashes"].items():
                if digest(Path(name)) != expected:
                    raise ValueError("Mechanism inputs changed since exact replay.")
            mechanism_evidence[job["id"]] = dict(path=str(source), sha256=digest(source))
    output.mkdir(exist_ok=False)
    verified = analyze(list(directories.values()), output / "metrics")
    by_path = {Path(record["path"]): record for record in verified["runs"]}
    records = {key: by_path[path.resolve()] for key, path in directories.items()}
    if any(not r["complete"] or len(r["attacks"]) != 11 for r in records.values()):
        raise ValueError("Incomplete formal results.")
    comparisons = []
    for seed in (43, 44, 45):
        for control in (f"none_{seed}", f"baseline_{seed}", f"combined_shuffle_{seed}"):
            comparisons.append(compare(f"combined_{seed}", control, records))
    for treatment in ("history_43", "selection_43"):
        for control in ("none_43", "baseline_43"):
            comparisons.append(compare(treatment, control, records))
    checks = {}
    for kind, prefix, field in (("vs_none", "none_", "overall_vs_none_passed"),
                               ("vs_baseline", "baseline_", "added_value_passed"),
                               ("vs_shuffle", "combined_shuffle_", "added_value_passed")):
        selected = [row for row in comparisons if row["treatment"].startswith("combined_") and row["control"].startswith(prefix)]
        assert len(selected) == 3
        checks[kind] = dict(passed_seeds=sum(row[field] for row in selected),
                            all_seeds_passed=all(row[field] for row in selected))
    outcome = dict(status="metrics_verified_supplements_pending", plan=str(plan_path), plan_sha256=digest(plan_path),
        scope=plan["scope"], arms=[metrics(key, record) for key, record in records.items()],
        comparisons=comparisons, three_seed_checks=checks, mechanism_evidence=mechanism_evidence,
        hyperparameters=plan["hyperparameters"])
    (output / "outcome.json").write_text(json.dumps(outcome, indent=2))
    # Conditional intervals for the three predeclared combination comparisons in each seed.
    for row in comparisons[:9]:
        print(f'Resampling {row["treatment"]} vs {row["control"]}', flush=True)
        resample(output / "metrics/verified_results.json", output / f'{row["treatment"]}_vs_{row["control"]}',
                 2000, 20260913, treatment=records[row["treatment"]]["run"], control_name=records[row["control"]]["run"])
    duplicate_sensitivity(output / "metrics/verified_results.json",
        ROOT / "analysis_scripts/risk_synthesis_confirmation_exact_image_identity_20260912.json", output / "duplicates")
    outcome["status"] = "completed"
    (output / "outcome.json").write_text(json.dumps(outcome, indent=2))
    render(output, outcome, records)
    with (output / "comparison_metrics.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps(dict(status="completed", output=str(output), three_seed_checks=checks), indent=2))


def watch(study, output):
    """Wait on existing, live study workers; never restart a missing process."""
    state_path = study / "analysis_execution.json"
    plan_path = study / "plan.json"
    plan = json.loads(plan_path.read_text())
    expected_sha = digest(plan_path)
    state = dict(status="waiting_for_existing_workers", pid=os.getpid(), plan_sha256=expected_sha,
                 output=str(output), analysis_sources={str(p): digest(p) for p in (
                     Path(__file__), ROOT / "scripts/analyze_risk_synthesis.py",
                     ROOT / "scripts/paired_synthesis_uncertainty.py", ROOT / "scripts/confirmation_duplicate_sensitivity.py")})
    save(state_path, state, exclusive=True)
    try:
        while True:
            if digest(plan_path) != expected_sha:
                raise RuntimeError("Frozen plan changed while observing.")
            states = {}
            for job in plan["jobs"]:
                path = study / f'{job["id"]}.json'
                try:
                    states[job["id"]] = json.loads(path.read_text())["status"] if path.exists() else "unclaimed"
                except json.JSONDecodeError:
                    # An exclusive initial creation can be observed before its
                    # write completes; re-read the same job on the next poll.
                    states[job["id"]] = "initializing"
            if "failed" in states.values():
                raise RuntimeError(f"A study job failed: {states}")
            if all(status == "completed" for status in states.values()):
                break
            live = []
            unreadable = False
            for path in study.glob("worker_gpu*.json"):
                try:
                    worker = json.loads(path.read_text())
                except json.JSONDecodeError:
                    unreadable = True
                    continue
                if worker["status"] in ("failed", "finished_claimed_queue"):
                    continue
                proc = Path(f'/proc/{worker["pid"]}/cmdline')
                try:
                    command = proc.read_bytes().split(b"\0")
                except FileNotFoundError:
                    continue
                if any(part.endswith(b"run_synthesis_history_study.py") for part in command):
                    live.append(worker["pid"])
            if not live and not unreadable:
                raise RuntimeError("Incomplete study has no verified live worker; no automatic restart.")
            state.update(jobs=states, verified_live_workers=live)
            save(state_path, state)
            time.sleep(30)
        state["status"] = "analyzing_completed_runs"
        save(state_path, state)
        run(study, output)
        state["status"] = "completed"
        save(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        save(state_path, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, default=STUDY / "analysis")
    parser.add_argument("--watch", action="store_true", help="Observe existing live workers and analyze when all finish.")
    args = parser.parse_args()
    (watch if args.watch else run)(args.study_dir.resolve(), args.output.resolve())
