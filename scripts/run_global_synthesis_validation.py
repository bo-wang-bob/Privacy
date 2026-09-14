"""Matched CIFAR100 validation of inclusive-global-mean synthetic views.

One serial queue on one GPU per invocation. Run separate invocations for
Adapter and LoRA to use two GPUs. Execution stays in the unified runner.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / 'analysis_scripts/risk_synthesis_confirmation_data_20260912/split.json'
MANIFEST_SHA256 = '00941791be9727022bb09c4b9d55f8b06d05a1cc3489a99753cbdb393151c178'


def build_arguments(overrides):
    return [
        '--models', 'clip_adapter,clip_lora',
        '--datasets', 'cifar100',
        '--methods', 'fedavg',
        '--defenses', 'none,risk_synthesis',
        '--attacks', 'all',
        '--seeds', '43',
        '--target-clients', '0',
        '--rounds', '100',
        '--local-epochs', '1',
        '--aggregation-weighting', 'sample_count',
        '--partition-mode', 'iid',
        '--gpus', '0', '--jobs', '1',
        '--set', 'use_full_dataset=false',
        '--set', 'fpl_shots=100',
        '--set', 'total_users=10',
        '--set', 'sample_users=10',
        '--set', 'batch_size=32',
        '--set', f'confirmation_split_manifest={MANIFEST}',
        '--set', f'confirmation_split_sha256={MANIFEST_SHA256}',
        '--set', 'defense.synthesis.global_distribution=generate',
        '--set', 'defense.synthesis.center_source=global_class',
        '--set', 'defense.synthesis.replacement_policy=all',
        '--set', 'defense.synthesis.replacement_fraction=1.0',
        '--set', 'defense.synthesis.warmup_rounds=0',
        '--set', 'defense.synthesis.views_per_record=2',
        '--set', 'defense.synthesis.risk_history=none',
        '--set', 'defense.synthesis.candidate_selection=first_semantic',
        '--set', 'defense.synthesis.center_weighting=uniform',
        '--set', 'defense.synthesis.mode=risk',
        *overrides,
    ]


def main(argv=None):
    # Match the existing experiment environment without requiring shell flags.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    from scripts import run_privacy_experiments as runner
    arguments = build_arguments(sys.argv[1:] if argv is None else argv)
    args = runner.parse_args(arguments)
    if args.jobs != 1 or len(runner.parse_int_csv(args.gpus)) != 1:
        raise ValueError('This validation wrapper uses --jobs 1 and one GPU. Use a separate invocation per GPU.')
    os.chdir(ROOT)
    return runner.main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
