"""固定0.2噪声，比较有符号损失差、绝对损失差和批内打乱权重。

默认 Adapter / CIFAR100 / FedSGD 1000轮 / K=2 / seed43 / 全部11种攻击。
单GPU串行运行；--smoke 为5轮无攻击检查，--dry-run 仅核对配置。
"""
from pathlib import Path
import argparse
import copy
import datetime as dt
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_global_synthesis_validation import build_arguments

VARIANTS = {
    'signed': ('signed_loss_gap', 'risk'),
    'absolute': ('absolute_loss_gap', 'risk'),
    'shuffled': ('signed_loss_gap', 'shuffled_risk'),
}
DEFINITIONS = {
    'signed': 'rank(loss_other - loss_own)',
    'absolute': 'rank(abs(loss_other - loss_own))',
    'shuffled': 'batch_permutation(rank(loss_other - loss_own))',
}


def build_study(argv, *, started_at=None):
    from scripts import run_privacy_experiments as runner
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variants', default=','.join(VARIANTS), help='signed,absolute,shuffled；可选择补跑组。')
    parser.add_argument('--smoke', action='store_true', help='5轮、逐轮评估、无攻击。')
    own, forwarded = parser.parse_known_args(argv)
    variants = [v.strip() for v in own.variants.split(',')]
    if len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('--variants 只能包含不重复的 signed,absolute,shuffled。')
    for expression in runner.parse_args(forwarded).set_values:
        if expression.split('=', 1)[0].strip() in ('defense.synthesis.risk_score', 'defense.synthesis.mode'):
            raise ValueError('请用 --variants 选择风险分数与打乱分配，不要覆盖 synthesis.risk_score/mode。')
    overrides = ['--models', 'clip_adapter', '--methods', 'fedsgd', '--rounds', '1000',
                 '--defenses', 'risk_synthesis', '--set', 'defense.synthesis.noise_scale=0.2',
                 '--set', 'defense.synthesis.mixing_mode=risk',
                 '--set', 'defense.synthesis.risk_tail_fraction=1']
    if own.smoke:
        overrides += ['--rounds', '5', '--attacks', 'none', '--set', 'eval_interval=1']
    common = build_arguments([*overrides, *forwarded])
    base = runner.parse_args(common)
    if base.jobs != 1 or len(runner.parse_int_csv(base.gpus)) != 1:
        raise ValueError('本入口使用单GPU、--jobs 1，三组串行运行。')
    if base.list:
        raise ValueError('能力列表请使用 scripts/run_privacy_experiments.py --list。')
    started_at = started_at or dt.datetime.now()
    study_root = runner.resolve_path(base.results_root) / f'synthesis_risk_score_ablation_{started_at:%Y%m%d_%H%M%S_%f}'
    catalog = runner.load_yaml(base.catalog)
    groups, reference = [], None
    for variant in variants:
        score, mode = VARIANTS[variant]
        arguments = [*common, '--results-root', str(study_root/variant),
                     '--set', f'defense.synthesis.risk_score={score}',
                     '--set', f'defense.synthesis.mode={mode}']
        args = runner.parse_args(arguments)
        args.started_at = started_at
        tasks, skipped = runner.build_tasks(catalog, args)
        if skipped or not tasks:
            raise ValueError('消融不应跳过组合或没有任务：' + '; '.join(skipped))
        comparable = []
        for task in tasks:
            options = task.config.get('defense', {}).get('synthesis', {})
            required = dict(noise_scale=.2, global_distribution='generate', center_source='global_class',
                            class_rank='all', candidate_selection='direct', mixing_mode='risk',
                            risk_history='none', risk_score=score, mode=mode, risk_tail_fraction=1,
                            views_per_record=2)
            if (task.defense != 'risk_synthesis' or task.model not in ('clip_adapter', 'clip_lora')
                    or task.config['aggregator'] != 'fedsgd'
                    or any(options.get(k) != v for k, v in required.items())):
                raise ValueError('本消融固定 FedSGD/direct、0.2噪声、K=2、全覆盖风险和全局完整秩几何。')
            config = copy.deepcopy(task.config)
            config.pop('results_dir', None)
            config['defense']['synthesis'].pop('risk_score')
            config['defense']['synthesis'].pop('mode')
            comparable.append(config)
        if reference is not None and comparable != reference:
            raise ValueError('各组除 risk_score、mode 和结果目录外必须完全一致。')
        reference = comparable
        groups.append(dict(variant=variant, definition=DEFINITIONS[variant], arguments=arguments, tasks=tasks))
    return study_root, groups, base.dry_run


def main(argv=None):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    from scripts import run_privacy_experiments as runner
    os.chdir(ROOT)
    study_root, groups, dry_run = build_study(sys.argv[1:] if argv is None else argv)
    print(f'STUDY | {study_root}', flush=True)
    for group in groups:
        print(f"VARIANT | {group['variant']} | {group['definition']}", flush=True)
        runner.print_plan(group['tasks'], [])
    if dry_run:
        print('DRY-RUN | 三组控制变量核对通过；未创建目录，未启动训练。')
        return 0
    study_root.mkdir(parents=True, exist_ok=False)
    plan = dict(noise_scale=.2, generation='(1-r) h + r mu_global_class + 0.2 L epsilon',
                groups=[dict(variant=g['variant'], definition=g['definition'], arguments=g['arguments'],
                    configs=[{k:v for k,v in t.config.items() if k != 'results_dir'} for t in g['tasks']])
                        for g in groups])
    with (study_root/'study_plan.json').open('x') as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False, allow_nan=False)
    for group in groups:
        print(f"RUN | {group['variant']} | {group['definition']}", flush=True)
        # Retain the sole scheduler's process ownership, logs and failure cleanup.
        status = runner.main(group['arguments'])
        if status:
            print(f"STOP | {group['variant']} 未全部完成；后续组未启动。", flush=True)
            return status
    print(f'COMPLETE | {len(groups)}组结果保存在 {study_root}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as error:
        print(f'ERROR | {error}', file=sys.stderr)
        raise SystemExit(2) from error
