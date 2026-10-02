"""固定噪声0.5，比较 source / class_center / risk，并支持 shuffled_risk 对照。

默认：Adapter+LoRA，CIFAR100，FedAvg 100轮，2视图，全部11种攻击。
--smoke：5轮、逐轮评估、关闭攻击。其余参数透传统一实验入口。
例如：python scripts/run_synthesis_center_ablation.py --models clip_adapter --gpus 0
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

DEFAULT_VARIANTS = ('source', 'class_center', 'risk')
VARIANTS = (*DEFAULT_VARIANTS, 'shuffled_risk')
FORMULAS = dict(source='h + 0.5 L epsilon', class_center='mu + 0.5 L epsilon',
                risk='(1-r) h + r mu + 0.5 L epsilon',
                shuffled_risk='(1-r_perm) h + r_perm mu + 0.5 L epsilon')


def build_study(argv, *, started_at=None):
    """Resolve and compare all conditions before writing files or starting a run."""
    from scripts import run_privacy_experiments as runner
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variants', default=','.join(DEFAULT_VARIANTS),
                        help='逗号分隔 source,class_center,risk,shuffled_risk；默认前三组，可单独补跑打乱风险。')
    parser.add_argument('--smoke', action='store_true', help='5轮无攻击检查。')
    own, forwarded = parser.parse_known_args(argv)
    variants = [v.strip() for v in own.variants.split(',')]
    if len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('--variants 只能包含不重复的 source,class_center,risk,shuffled_risk。')
    # Both switches are determined by the named variant; reject conflicting
    # user overrides instead of silently overwriting them later.
    for expression in runner.parse_args(forwarded).set_values:
        if expression.split('=', 1)[0].strip() in ('defense.synthesis.mixing_mode', 'defense.synthesis.mode'):
            raise ValueError('请用 --variants 选择生成位置和风险对照，不要覆盖 synthesis.mixing_mode/mode。')
    overrides = ['--defenses', 'risk_synthesis', '--set', 'defense.synthesis.noise_scale=0.5']
    if own.smoke:
        overrides += ['--rounds', '5', '--attacks', 'none', '--set', 'eval_interval=1']
    common = build_arguments([*overrides, *forwarded])
    base = runner.parse_args(common)
    if base.jobs != 1 or len(runner.parse_int_csv(base.gpus)) != 1:
        raise ValueError('每次调用使用单GPU、--jobs 1；两种模型可分别在两张卡运行。')
    if base.list:
        raise ValueError('能力列表请使用 scripts/run_privacy_experiments.py --list。')
    started_at = started_at or dt.datetime.now()
    study_root = runner.resolve_path(base.results_root) / f'synthesis_center_ablation_{started_at:%Y%m%d_%H%M%S_%f}'
    catalog = runner.load_yaml(base.catalog)
    groups, reference = [], None
    for variant in variants:
        mixing_mode = 'risk' if variant == 'shuffled_risk' else variant
        risk_mode = 'shuffled_risk' if variant == 'shuffled_risk' else 'risk'
        arguments = [*common, '--results-root', str(study_root / variant),
                     '--set', f'defense.synthesis.mixing_mode={mixing_mode}',
                     '--set', f'defense.synthesis.mode={risk_mode}']
        args = runner.parse_args(arguments)
        args.started_at = started_at
        tasks, skipped = runner.build_tasks(catalog, args)
        if skipped:
            raise ValueError('消融不应静默跳过组合：' + '; '.join(skipped))
        comparable = []
        for task in tasks:
            options = task.config.get('defense', {}).get('synthesis', {})
            required = dict(noise_scale=0.5, global_distribution='generate', center_source='global_class',
                            class_rank='all', candidate_selection='direct', mode=risk_mode,
                            mixing_mode=mixing_mode, risk_history='none')
            if (task.defense != 'risk_synthesis' or task.model not in ('clip_adapter', 'clip_lora')
                    or any(options.get(k) != value for k, value in required.items())):
                raise ValueError('本消融固定 risk_synthesis/direct、0.5噪声、全局完整秩几何和当前风险定义。')
            config = copy.deepcopy(task.config)
            config.pop('results_dir', None)
            config['defense']['synthesis'].pop('mixing_mode')
            config['defense']['synthesis'].pop('mode')
            comparable.append(config)
        if reference is not None and comparable != reference:
            raise ValueError('各组配置除 mixing_mode、mode 和结果目录外必须一致。')
        reference = comparable
        groups.append(dict(variant=variant, formula=FORMULAS[variant], arguments=arguments, tasks=tasks))
    return study_root, groups, base.dry_run


def main(argv=None):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    from scripts import run_privacy_experiments as runner
    os.chdir(ROOT)
    study_root, groups, dry_run = build_study(sys.argv[1:] if argv is None else argv)
    print(f'STUDY | {study_root}', flush=True)
    for group in groups:
        print(f"VARIANT | {group['variant']} | {group['formula']}", flush=True)
        runner.print_plan(group['tasks'], [])
    if dry_run:
        print('DRY-RUN | 全组配置匹配通过；未创建目录，未启动训练。')
        return 0
    study_root.mkdir(parents=True, exist_ok=False)
    plan = dict(noise_scale=0.5, groups=[dict(variant=g['variant'], formula=g['formula'],
        arguments=g['arguments'], configs=[{k:v for k,v in t.config.items() if k != 'results_dir'}
                                          for t in g['tasks']]) for g in groups])
    with (study_root / 'study_plan.json').open('x') as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False, allow_nan=False)
    for group in groups:
        print(f"RUN | {group['variant']} | {group['formula']}", flush=True)
        # The sole batch runner owns subprocess scheduling, configs, logs and summaries.
        status = runner.main(group['arguments'])
        if status:
            print(f"STOP | {group['variant']} 未全部完成；后续组未启动。", flush=True)
            return status
    print(f'COMPLETE | {len(groups)}组结果按名称保存在 {study_root}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as error:
        print(f'ERROR | {error}', file=sys.stderr)
        raise SystemExit(2) from error
