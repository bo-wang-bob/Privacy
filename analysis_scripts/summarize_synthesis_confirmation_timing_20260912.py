"""Describe completed confirmation timings without changing any experiment."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis_scripts/risk_synthesis_confirmation_timing_20260912"


def main():
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    report = read(ROOT / "analysis_scripts/risk_synthesis_confirmation_report_20260912/confirmation_report.json")
    expected = {row["run"]: row for row in report["run_metrics"]}
    rows = []
    stage_fields = {
        "run_seconds": "run", "client_update_seconds": "train.client_update",
        "audit_seconds": "audit.observe", "evaluation_seconds": "evaluation",
        "geometry_seconds": "setup.synthesis_geometry",
        "generation_seconds": "train.synthesis_generate",
        "filter_seconds": "train.synthesis_filter",
        "synthesis_optimizer_seconds": "train.synthesis_optimizer",
    }
    for lane in (0, 1):
        execution = read(ROOT / f"analysis_scripts/risk_synthesis_confirmation_lane{lane}_20260912.json")
        assert execution["status"] == "completed"
        assert execution["plan_sha256"] == report["plan_sha256"]
        for job in execution["jobs"]:
            assert job["status"] == "completed" and job["returncode"] == 0
            assert len(job["result_directories"]) == 1
            directory = Path(job["result_directories"][0])
            metric = expected[directory.name]
            assert metric["seed"] == job["seed"] and metric["arm"] == job["arm"]
            performance = read(directory / "performance_summary.json")
            assert performance["status"] == "completed"
            stages = performance["stages"]
            elapsed = (datetime.fromisoformat(job["finished_at_utc"])
                       - datetime.fromisoformat(job["started_at_utc"])).total_seconds()
            assert elapsed > 0
            row = dict(run=directory.name, seed=job["seed"], arm=job["arm"],
                       lane=lane, elapsed_seconds=elapsed)
            for field, stage in stage_fields.items():
                row[field] = stages.get(stage, {}).get("wall_seconds")
            ranking = "train.www_ranking" if job["arm"] == "www" else "train.synthesis_ranking"
            row["ranking_seconds"] = stages.get(ranking, {}).get("wall_seconds")
            rows.append(row)
    assert len(rows) == len(expected) == 12
    assert len({row["run"] for row in rows}) == 12
    aggregates = []
    for arm in ("none", "www", "risk", "shuffled_risk"):
        group = [row for row in rows if row["arm"] == arm]
        assert {row["seed"] for row in group} == {43, 44, 45}
        elapsed = [row["elapsed_seconds"] / 60 for row in group]
        ratios = [row["elapsed_seconds"] / next(
            other["elapsed_seconds"] for other in rows
            if other["arm"] == "none" and other["seed"] == row["seed"])
            for row in group]
        aggregates.append(dict(arm=arm, mean_minutes=statistics.mean(elapsed),
                               minimum_minutes=min(elapsed), maximum_minutes=max(elapsed),
                               mean_paired_ratio_to_none=statistics.mean(ratios)))
    result = dict(status="completed", rows=rows, aggregates=aggregates, sources=sources,
        interpretation="Observed local task elapsed times include initialization; machine load and lanes vary. "
        "Performance stages are inclusive, not additive: filter is inside generation, and ranking/generation/optimizer "
        "are inside client_update. CPU wall times can include asynchronous GPU work. These are not latency guarantees.")
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "timing_report.json").write_text(json.dumps(result, indent=2))
    with (OUTPUT / "timing.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
