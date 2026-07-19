#!/usr/bin/env python3
"""Compact existing privacy-audit signal files without changing attack scores."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


COMMON_FIELDS = {"round", "client_ids"}
ATTACK_FIELDS = {
    "blackbox_loss": {"confidence"},
    "loss_series": {"confidence"},
    "grad_cosine": {"cosine"},
    "avg_cosine": {"cosine"},
    "fedmia_loss": {"confidence"},
    "fedmia_cosine": {"cosine"},
    "nasr_passive": {
        "confidence",
        "cosine",
        "gradient_difference",
        "gradient_signature",
        "candidate_labels",
        "probabilities",
    },
    "transfer_representation": {"representations"},
    "rmia": {"probabilities"},
    "quantile_mia": {"probabilities", "representations"},
    "pipra": {"client_states"},
    "imia": {"client_states"},
    "yoqo": {"client_states"},
    "canary": {"client_states"},
    "promptmia": {"client_states"},
}


def required_observation_fields(attacks: set[str]) -> set[str]:
    fields = set(COMMON_FIELDS)
    for attack in attacks:
        fields.update(ATTACK_FIELDS.get(attack, set()))
    return fields


def compact_payload(payload: dict, attacks: set[str]) -> dict:
    fields = required_observation_fields(attacks)
    observations = [
        {key: value for key, value in observation.items() if key in fields}
        for observation in payload.get("observations", [])
    ]
    return {
        "candidate_labels": payload["candidate_labels"],
        "membership": payload["membership"],
        "observations": observations,
        "storage_mode": "compact",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove observation tensors not required by the attacks recorded in "
            "each privacy_audit/summary.json. Dry-run is the default."
        )
    )
    parser.add_argument(
        "--root",
        default="results/fedmia_prompt_methods",
        help="Result tree containing privacy_audit directories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically replace signals.pt files after validating compact output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    signal_paths = sorted(root.glob("runs/**/privacy_audit/signals.pt"))
    if not signal_paths:
        print(f"No signals.pt files found under {root}")
        return 0
    total_before = 0
    total_after = 0
    for signal_path in signal_paths:
        summary_path = signal_path.with_name("summary.json")
        if not summary_path.is_file():
            print(f"skip missing summary: {signal_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        attacks = {str(item["attack"]) for item in summary.get("attacks", [])}
        before = signal_path.stat().st_size
        total_before += before
        if not args.apply:
            print(
                f"would compact {signal_path}: {before / 1024**2:.2f} MiB; "
                f"keep={sorted(required_observation_fields(attacks))}"
            )
            continue
        payload = torch.load(signal_path, map_location="cpu", weights_only=False)
        compact = compact_payload(payload, attacks)
        temporary = signal_path.with_suffix(".pt.compact.tmp")
        torch.save(compact, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if len(reloaded["observations"]) != len(payload.get("observations", [])):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Observation count changed while compacting {signal_path}")
        after = temporary.stat().st_size
        os.replace(temporary, signal_path)
        total_after += after
        print(
            f"compacted {signal_path}: {before / 1024**2:.2f} -> "
            f"{after / 1024**2:.2f} MiB"
        )
    if args.apply:
        print(
            f"total: {total_before / 1024**3:.3f} -> "
            f"{total_after / 1024**3:.3f} GiB"
        )
    else:
        print(f"dry-run: {len(signal_paths)} files, {total_before / 1024**3:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
