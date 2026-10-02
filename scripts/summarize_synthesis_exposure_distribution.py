"""Summarize original-record exposure distributions from completed all-replacement runs."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_risk_synthesis import digest


def statistics(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()), minimum=float(x.min()),
                median=float(np.median(x)), p90=float(np.quantile(x, .9)), maximum=float(x.max()))


def summarize(record):
    if not record["complete"]:
        raise ValueError("Exposure distributions require a complete verified run.")
    options = record.get("synthesis_options") or {}
    if options.get("replacement_policy") != "all":
        return None
    direct = options.get("candidate_selection") == "direct"
    mixing_ablation = direct and 'mixing_mode' in options
    views = options.get("views_per_record", 1)
    path = Path(record["path"]) / "risk_synthesis" / ("synthetic_views.csv" if views > 1 else "synthetic_exposure.csv")
    if digest(path) != record["sources"][str(path)]:
        raise ValueError("Exposure stream changed after formal verification.")
    observations = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = int(row["client"]), int(row["sample_id"])
            label = int(row["label"])
            value = observations.setdefault(key, dict(label=label, visits=0, reference_visits=0, zeros=0, zero_mixing=0,
                retained_sum=0., retained_fourth_sum=0., quality_failures=None if direct else 0, attempts=0,
                history_changes=0 if "assigned_risk" in row else None,
                neighbor_cosine_sum=0. if "nearest_teacher_cosine" in row else None,
                source_nearest_count=0 if "nearest_teacher_source_id" in row else None))
            if label != value["label"] or row["requested"] != "1" or row["accepted"] != "1":
                raise ValueError("Original identity or all-replacement protocol changed.")
            used, raw = float(row["used_risk"]), float(row["risk"])
            if not 0 <= used <= 1 or not 0 <= raw <= 1:
                raise ValueError("Invalid recorded risk.")
            value["visits"] += 1
            value["attempts"] += int(row["attempts"])
            if direct:
                if row["quality_passed"] != "":
                    raise ValueError("Unchecked direct views cannot claim semantic quality.")
            else:
                value["quality_failures"] += 1-int(row["quality_passed"])
            # Exclude unavailable-reference bootstrap visits for all retained
            # moments, zero-risk frequency and selection-history comparisons.
            if int(row["source_round"]) < 0:
                continue
            value["reference_visits"] += 1
            # In v15 the fixed endpoints decouple risk from the mixing weight.
            # Preserve historical assigned-risk summaries for older protocols.
            value["zeros"] += int((raw if mixing_ablation else used) == 0)
            value['zero_mixing'] += int(used == 0)
            value["retained_sum"] += 1-used
            value["retained_fourth_sum"] += (1-used)**4
            if value["history_changes"] is not None:
                value["history_changes"] += int(float(row["assigned_risk"]) != raw)
            if value["neighbor_cosine_sum"] is not None:
                value["neighbor_cosine_sum"] += float(row["nearest_teacher_cosine"])
                value["source_nearest_count"] += int(int(row["nearest_teacher_source_id"]) == key[1])
    result = []
    expected = record["protocol"]["num_global_iters"] * record["protocol"]["local_epochs"] * views
    for (client, sid), value in sorted(observations.items()):
        n = value["reference_visits"]
        if value["visits"] != expected or n == 0:
            raise ValueError("Expected complete full-local-epoch original-record exposures.")
        result.append(dict(run=record["run"], seed=record["protocol"]["seed"], client=client, sample_id=sid,
            label=value["label"], visits=value["visits"]//views, reference_visits=n//views,
            trained_views=value["visits"], zero_risk_count=value["zeros"]/views,
            zero_risk_fraction=value["zeros"]/n,
            zero_risk_basis='raw_risk' if mixing_ablation else 'used_risk',
            mixing_mode=options.get('mixing_mode', 'risk'),
            zero_mixing_count=value['zero_mixing']/views, zero_mixing_fraction=value['zero_mixing']/n,
            mean_retained_fraction=value["retained_sum"]/n,
            mean_retained_fourth=value["retained_fourth_sum"]/n,
            semantic_failure_fraction=None if direct else value["quality_failures"]/value["visits"],
            mean_attempts=value["attempts"]/value["visits"],
            history_reassignment_fraction=None if value["history_changes"] is None else value["history_changes"]/n,
            mean_nearest_teacher_cosine=None if value["neighbor_cosine_sum"] is None else value["neighbor_cosine_sum"]/n,
            source_is_nearest_teacher_fraction=None if value["source_nearest_count"] is None else value["source_nearest_count"]/n))
    if sum(row["visits"] for row in result) != record["synthesis_mechanism"]["counts"]["visits"]:
        raise ValueError("Per-original exposures do not reconcile with verified stream counts.")
    target = int(record["protocol"]["audit"]["audit_client_ids"][0])
    distributions = []
    for scope, rows in (("all_clients", result), (f"target_client_{target}", [r for r in result if r["client"] == target])):
        for metric in ("zero_risk_count", "zero_risk_fraction", "zero_mixing_count", "zero_mixing_fraction",
                       "mean_retained_fraction", "mean_retained_fourth",
                       "semantic_failure_fraction", "mean_attempts", "history_reassignment_fraction",
                       "mean_nearest_teacher_cosine", "source_is_nearest_teacher_fraction"):
            present = [row[metric] for row in rows if row[metric] is not None]
            if present:
                if len(present) != len(rows) or not np.isfinite(present).all():
                    raise ValueError("Missing or nonfinite exposure values.")
                distributions.append(dict(run=record["run"], seed=record["protocol"]["seed"],
                                          scope=scope, metric=metric, **statistics(present)))
    return result, distributions, dict(path=str(path), sha256=digest(path))


def run(verified_path, output):
    verified = json.loads(verified_path.read_text())
    output.mkdir(exist_ok=False)
    distributions, sources = [], {str(verified_path): digest(verified_path), str(Path(__file__)): digest(Path(__file__))}
    count = 0
    with (output / "original_record_exposures.csv").open("x", newline="") as handle:
        writer = None
        for record in verified["runs"]:
            result = summarize(record)
            if result is None:
                continue
            rows, summaries, source = result
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            count += len(rows)
            distributions.extend(summaries)
            sources[source["path"]] = source["sha256"]
            print(f'Summarized {record["run"]}: {len(rows)} original records', flush=True)
    if not distributions:
        raise ValueError("No completed all-replacement runs found.")
    with (output / "exposure_distributions.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distributions[0]))
        writer.writeheader()
        writer.writerows(distributions)
    result = dict(status="completed_original_exposure_distribution", original_records=count,
        distributions=distributions, sources=sources,
        interpretation="Distributions across original records, not synthetic members. Retention/zero-risk/neighbor "
            "summaries exclude unavailable-reference visits; quality/attempt summaries include all visits. "
            "zero_risk_basis identifies raw risk in v15 versus historical assigned risk; zero_mixing always uses the actual coefficient. "
            "Source coefficients and teacher similarities are proxies, not information fractions or privacy budgets.")
    (output / "exposure_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result["status"], original_records=count, output=str(output)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verified", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.verified.resolve(), args.output.resolve())
