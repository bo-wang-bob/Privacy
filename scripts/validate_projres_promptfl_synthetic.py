#!/usr/bin/env python3
"""Checkpoint-free validation of ProjRes theory adapted to PromptFL logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from privacy_attacks.metrics import membership_metrics
from privacy_attacks.projres_promptfl import (
    principal_angles,
    projection_statistics,
    prompt_gradient_fingerprints,
    prompt_vjp,
    ridge_lift_dense,
    row_subspace_basis,
    text_feature_gradient,
)
from privacy_attacks.promptres import positive_cosine_squared


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _mean(rows: list[dict[str, float | int]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _one_trial(
    classes: int,
    dimension: int,
    batch_size: int,
    prompt_width: int,
    ridge: float,
    seed: int,
) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    text_offset = F.normalize(
        torch.randn(classes, dimension, generator=generator, dtype=dtype), dim=1
    )
    members = F.normalize(
        torch.randn(batch_size, dimension, generator=generator, dtype=dtype), dim=1
    )
    nonmembers = F.normalize(
        torch.randn(batch_size, dimension, generator=generator, dtype=dtype), dim=1
    )
    labels = torch.arange(batch_size) % classes
    nonmember_labels = labels.clone()
    mapping = torch.randn(
        classes * dimension,
        prompt_width,
        generator=generator,
        dtype=dtype,
    ) / (classes * dimension) ** 0.5
    prompt = torch.randn(prompt_width, generator=generator, dtype=dtype) * 0.02

    def feature_function(value: torch.Tensor) -> torch.Tensor:
        raw = text_offset + (mapping @ value).reshape(classes, dimension)
        return F.normalize(raw, dim=1)

    text_features = feature_function(prompt).detach()
    true_gradient, errors = text_feature_gradient(
        text_features, members, labels
    )
    prompt_gradient = prompt_vjp(feature_function, prompt, true_gradient)
    candidate_features = torch.cat((members, nonmembers))
    candidate_labels = torch.cat((labels, nonmember_labels))
    membership = torch.cat(
        (
            torch.ones(batch_size, dtype=torch.long),
            torch.zeros(batch_size, dtype=torch.long),
        )
    )

    oracle_basis, oracle_metadata = row_subspace_basis(true_gradient)
    member_basis, member_metadata = row_subspace_basis(members)
    oracle_stats = projection_statistics(candidate_features, oracle_basis)
    oracle_metrics = membership_metrics(
        membership, oracle_stats["projection_energy"]
    )
    angles = principal_angles(oracle_basis, member_basis)

    # Normalization is part of feature_function, so obtain the exact dense
    # Jacobian through autograd rather than treating mapping as the Jacobian.
    jacobian = torch.autograd.functional.jacobian(
        lambda value: feature_function(value).reshape(-1),
        prompt.detach().clone().requires_grad_(True),
        vectorize=True,
    ).reshape(classes * dimension, prompt_width).detach()
    lifted_gradient, lift_diagnostics = ridge_lift_dense(
        jacobian,
        prompt_gradient,
        (classes, dimension),
        ridge=ridge,
    )
    lifted_basis, lifted_metadata = row_subspace_basis(lifted_gradient)
    lifted_stats = projection_statistics(candidate_features, lifted_basis)
    lifted_metrics = membership_metrics(
        membership, lifted_stats["projection_energy"]
    )

    fingerprints, _, _ = prompt_gradient_fingerprints(
        feature_function,
        prompt,
        candidate_features,
        candidate_labels,
    )
    direct_scores = positive_cosine_squared(
        prompt_gradient.flatten(), fingerprints
    )
    direct_metrics = membership_metrics(membership, direct_scores)
    fingerprint_mean_error = float(
        (fingerprints[:batch_size].mean(dim=0) - prompt_gradient.flatten()).norm()
        / prompt_gradient.norm().clamp_min(1e-12)
    )

    return {
        "classes": classes,
        "dimension": dimension,
        "batch_size": batch_size,
        "prompt_width": prompt_width,
        "theoretical_max_batch": min(classes - 1, dimension - 1),
        "error_rank": int(torch.linalg.matrix_rank(errors)),
        "image_rank": int(member_metadata["numerical_rank"]),
        "oracle_gradient_rank": int(oracle_metadata["numerical_rank"]),
        "lifted_gradient_rank": int(lifted_metadata["numerical_rank"]),
        "maximum_principal_angle_degrees": (
            float(torch.rad2deg(angles).max()) if angles.numel() else 0.0
        ),
        "oracle_member_mean_residual": float(
            oracle_stats["relative_l2_residual"][:batch_size].mean()
        ),
        "oracle_nonmember_mean_residual": float(
            oracle_stats["relative_l2_residual"][batch_size:].mean()
        ),
        "oracle_auc": oracle_metrics["auc"],
        "lifted_auc": lifted_metrics["auc"],
        "direct_prompt_atom_auc": direct_metrics["auc"],
        "candidate_fingerprint_mean_relative_error": fingerprint_mean_error,
        "lift_measurement_relative_residual": (
            lift_diagnostics.measurement_relative_residual
        ),
    }


def run_validation(
    classes: int,
    dimension: int,
    batch_sizes: list[int],
    prompt_widths: list[int],
    trials: int,
    ridge: float,
    seed: int,
) -> dict[str, object]:
    if classes < 2 or dimension < 2 or trials <= 0:
        raise ValueError("classes, dimension, and trials are too small.")
    rows = []
    for prompt_width in prompt_widths:
        for batch_size in batch_sizes:
            trials_for_setting = [
                _one_trial(
                    classes,
                    dimension,
                    batch_size,
                    prompt_width,
                    ridge,
                    seed + 10_007 * trial + 101 * batch_size + prompt_width,
                )
                for trial in range(trials)
            ]
            row = dict(trials_for_setting[0])
            for key in (
                "maximum_principal_angle_degrees",
                "oracle_member_mean_residual",
                "oracle_nonmember_mean_residual",
                "oracle_auc",
                "lifted_auc",
                "direct_prompt_atom_auc",
                "candidate_fingerprint_mean_relative_error",
                "lift_measurement_relative_residual",
            ):
                row[key] = _mean(trials_for_setting, key)
            rows.append(row)
    return {
        "experiment": "synthetic_promptfl_projres_validation",
        "seed": seed,
        "trials": trials,
        "rank_boundary": "batch <= min(classes - 1, dimension - 1)",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument(
        "--batch-sizes", type=_parse_int_list, default=[1, 2, 4, 7, 8, 12, 16]
    )
    parser.add_argument(
        "--prompt-widths", type=_parse_int_list, default=[4, 8, 16, 32]
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_validation(
        args.classes,
        args.dimension,
        args.batch_sizes,
        args.prompt_widths,
        args.trials,
        args.ridge,
        args.seed,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
