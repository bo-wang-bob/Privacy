#!/usr/bin/env python3
"""Build or execute a controlled PromptFL ProjRes validation sweep."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys


def _int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def _float_list(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated numbers")
    return values


def build_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    jobs = []
    for batch_size, local_steps, n_ctx, alpha, seed in itertools.product(
        args.batch_sizes,
        args.local_steps,
        args.n_ctx,
        args.dirichlet_alphas,
        args.seeds,
    ):
        name = (
            f"b{batch_size}_s{local_steps}_ctx{n_ctx}_"
            f"alpha{alpha:g}_seed{seed}"
        )
        output = args.output_dir / f"{name}.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("validate_projres_promptfl_real.py")),
            "--config",
            str(args.config),
            "--batch-size",
            str(batch_size),
            "--local-steps",
            str(local_steps),
            "--n-ctx",
            str(n_ctx),
            "--dirichlet-alpha",
            str(alpha),
            "--seed",
            str(seed),
            "--target-client",
            str(args.target_client),
            "--max-candidates",
            str(args.max_candidates),
            "--ridge",
            str(args.ridge),
            "--lift-iterations",
            str(args.lift_iterations),
            "--output",
            str(output),
        ]
        if args.device:
            command.extend(("--device", args.device))
        if args.skip_lift:
            command.append("--skip-lift")
        jobs.append(
            {
                "name": name,
                "output": str(output),
                "command": command,
            }
        )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/federated_prompt_paper.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=_int_list, default=[1, 4, 8, 16, 32])
    parser.add_argument("--local-steps", type=_int_list, default=[1, 2, 5])
    parser.add_argument("--n-ctx", type=_int_list, default=[4, 8, 16, 32])
    parser.add_argument(
        "--dirichlet-alphas", type=_float_list, default=[0.1, 0.5, 1.0]
    )
    parser.add_argument("--seeds", type=_int_list, default=[42, 43, 44])
    parser.add_argument("--target-client", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--lift-iterations", type=int, default=20)
    parser.add_argument("--device")
    parser.add_argument("--skip-lift", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run jobs sequentially; without this flag only write the manifest.",
    )
    args = parser.parse_args()
    jobs = build_jobs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"jobs": jobs}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(jobs)} jobs to {manifest_path}")
    if args.execute:
        for index, job in enumerate(jobs, start=1):
            output = Path(str(job["output"]))
            if output.exists():
                print(f"[{index}/{len(jobs)}] skip existing {output}")
                continue
            print(f"[{index}/{len(jobs)}] run {job['name']}")
            subprocess.run(job["command"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
