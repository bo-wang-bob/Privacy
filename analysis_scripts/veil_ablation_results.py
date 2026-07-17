"""Select the three stage-level VEIL ablations and emit attack-level evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from utils.fair_comparison import load_run


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "paper" / "aaai2027" / "evidence" / "ablation.csv"
ALIASES = {"local_ggeur", "mirage", "veil"}
LEGACY_ATTACKS = (
    "fedmia_loss",
    "fedmia_cosine",
    "fedmia_joint",
    "nasr_passive",
    "rmia",
    "quantile_mia",
)
PAPER_ATTACKS = tuple(attack for attack in LEGACY_ATTACKS if attack != "fedmia_joint")
BASE = {
    "local_ggeur_augments": 3,
    "local_ggeur_geometry_scale": 0.60,
    "local_ggeur_anchor_mode": "class_mean",
    "local_ggeur_original_mode": "class_mean",
    "local_ggeur_original_noise": 0.0,
    "local_ggeur_mean_noise_std": 0.0,
    "local_ggeur_fallback_std": 0.02,
    "local_ggeur_entropy_weight": 0.0,
    "local_ggeur_class_balanced": False,
    "local_ggeur_output_temperature": 4.0,
    "local_ggeur_output_margin": None,
    "local_ggeur_calibrate_observations": False,
    "local_ggeur_upload_clip_norm": 0.5,
    "local_ggeur_upload_noise_std": 0.11,
}
OVERRIDES: dict[str, dict[str, Any]] = {
    "Full VEIL": {},
    "w/o Stage I": {"local_ggeur_augments": 0},
    "w/o Stage II": {"local_ggeur_anchor_mode": "sample"},
    "w/o Stage III": {
        "local_ggeur_upload_clip_norm": 0.0,
        "local_ggeur_upload_noise_std": 0.0,
    },
}
VARIANTS = {name: BASE | override for name, override in OVERRIDES.items()}


def matches(config: dict[str, Any], requested: dict[str, Any]) -> bool:
    defense = config.get("defense", {})
    return all(defense.get(key) == value for key, value in requested.items())


def valid_summary(path: Path) -> bool:
    summary = json.loads(path.read_text(encoding="utf-8"))
    candidate = summary.get("candidate_sampling", {})
    member_histogram = candidate.get("member_label_histogram", [])
    attacks = summary.get("attacks", [])
    return bool(
        candidate.get("label_histograms_matched")
        and member_histogram == candidate.get("nonmember_label_histogram")
        and sum(member_histogram) == 64
        and candidate.get("nonmember_source_priority")
        == ["target_test", "other_client_test", "other_client_train"]
        and [item.get("attack") for item in attacks]
        in (list(PAPER_ATTACKS), list(LEGACY_ATTACKS))
        and all(
            0.0 <= float(item.get("tpr_at_fpr_0.01", -1.0)) <= 1.0
            and 0.0 <= float(item.get("auc", -1.0)) <= 1.0
            and int(item.get("num_samples", 0)) > 0
            for item in attacks
        )
        and not summary.get("errors")
    )


def main() -> None:
    candidates: dict[str, list[Path]] = {name: [] for name in VARIANTS}
    for config_path in RESULTS.glob("flowers_*/run_config.yaml"):
        result_dir = config_path.parent
        summary_path = result_dir / "privacy_audit" / "summary.json"
        if not summary_path.exists() or not valid_summary(summary_path):
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if (
            config.get("aggregator") != "fedavg"
            or int(config.get("seed", -1)) != 42
            or not bool(config.get("require_cuda", False))
            or config.get("defense", {}).get("name") not in ALIASES
            or not config.get("audit", {}).get("match_candidate_labels", False)
        ):
            continue
        for name, requested in VARIANTS.items():
            if matches(config, requested):
                candidates[name].append(result_dir)

    missing = [name for name, paths in candidates.items() if not paths]
    if missing:
        raise RuntimeError(f"Missing validated VEIL ablations: {missing}")
    rows = []
    for name in VARIANTS:
        selected = max(
            candidates[name],
            key=lambda path: (path / "privacy_audit" / "summary.json").stat().st_mtime,
        )
        run = load_run(selected)
        summary = json.loads(
            (selected / "privacy_audit" / "summary.json").read_text(encoding="utf-8")
        )
        attack_values = {
            item["attack"]: item["tpr_at_fpr_0.01"] for item in summary["attacks"]
        }
        paper_values = {attack: attack_values[attack] for attack in PAPER_ATTACKS}
        rows.append(
            {
                "variant": name,
                "accuracy": run["accuracy"],
                **paper_values,
                "worst_tpr": max(paper_values.values()),
                "mean_tpr": sum(paper_values.values()) / len(paper_values),
                "result_dir": selected.name,
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "variant",
                "accuracy",
                *PAPER_ATTACKS,
                "worst_tpr",
                "mean_tpr",
                "result_dir",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} validated ablations to {OUTPUT}")


if __name__ == "__main__":
    main()
