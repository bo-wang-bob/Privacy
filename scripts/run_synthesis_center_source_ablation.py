"""固定全局协方差和0.5噪声，只比较包含自身的本地/全局类中心。

默认只补跑 local 的 risk、class_center 两组，Adapter/LoRA各两项；
与已有seed43全局中心结果配对。--centers local,global 可完整匹配重跑。
默认 CIFAR100、FedAvg100轮、K=2、11攻击；--smoke 为5轮无攻击。
其余参数透传唯一批量入口 scripts/run_privacy_experiments.py。
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

CENTERS = {'local': 'local_class_mean', 'global': 'global_class'}
VARIANTS = ('risk', 'class_center')
FORMULAS = dict(risk='(1-r) h + r mu_{center} + 0.5 L_global epsilon',
                class_center='mu_{center} + 0.5 L_global epsilon')


def choices(value, allowed, name):
    selected = [v.strip() for v in value.split(',')]
    if len(set(selected)) != len(selected) or any(v not in allowed for v in selected):
        raise ValueError(f'{name} 只能包含不重复的 {",".join(allowed)}。')
    return selected


def build_study(argv, *, started_at=None):
    """Preflight both scopes, including unscheduled global counterfactuals."""
    from scripts import run_privacy_experiments as runner
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--centers', default='local', help='local,global；默认只补本地中心。')
    parser.add_argument('--variants', default=','.join(VARIANTS), help='risk,class_center；默认两组。')
    parser.add_argument('--smoke', action='store_true', help='5轮、每轮评估、关闭攻击。')
    own, forwarded = parser.parse_known_args(argv)
    centers = choices(own.centers, CENTERS, '--centers')
    variants = choices(own.variants, VARIANTS, '--variants')
    for expression in runner.parse_args(forwarded).set_values:
        if expression.split('=', 1)[0].strip() in (
                'defense.synthesis.center_source', 'defense.synthesis.mixing_mode', 'defense.synthesis.mode'):
            raise ValueError('请用 --centers/--variants 选择中心和混合方式，不要覆盖 center_source/mixing_mode/mode。')
    overrides = ['--defenses', 'risk_synthesis', '--set', 'defense.synthesis.noise_scale=0.5']
    if own.smoke:
        overrides += ['--rounds', '5', '--attacks', 'none', '--set', 'eval_interval=1']
    common = build_arguments([*overrides, *forwarded])
    base = runner.parse_args(common)
    if base.jobs != 1 or len(runner.parse_int_csv(base.gpus)) != 1:
        raise ValueError('每次调用使用单GPU、--jobs 1；双模型可分卡运行。')
    if base.list:
        raise ValueError('能力列表请使用 scripts/run_privacy_experiments.py --list。')
    started_at = started_at or dt.datetime.now()
    root = runner.resolve_path(base.results_root) / f'synthesis_center_source_ablation_{started_at:%Y%m%d_%H%M%S_%f}'
    catalog = runner.load_yaml(base.catalog)
    planned, reference = {}, None
    for variant in variants:
        for center, center_source in CENTERS.items():
            name = f'{variant}_{center}'
            arguments = [*common, '--results-root', str(root/name),
                         '--set', f'defense.synthesis.center_source={center_source}',
                         '--set', f'defense.synthesis.mixing_mode={variant}']
            args = runner.parse_args(arguments)
            args.started_at = started_at
            tasks, skipped = runner.build_tasks(catalog, args)
            if skipped or not tasks:
                raise ValueError('中心消融需要完整可运行的任务组合：' + '; '.join(skipped))
            comparable = []
            for task in tasks:
                options = task.config.get('defense', {}).get('synthesis', {})
                required = dict(noise_scale=0.5, global_distribution='generate', center_source=center_source,
                                class_rank='all', candidate_selection='direct', center_weighting='uniform',
                                mode='risk', mixing_mode=variant, risk_history='none', replacement_policy='all',
                                replacement_fraction=1.0, warmup_rounds=0, semantic_filter=False)
                if (task.defense != 'risk_synthesis' or task.model not in ('clip_adapter', 'clip_lora')
                        or task.config['aggregator'] not in ('fedavg', 'fedsgd')
                        or any(options.get(k) != value for k, value in required.items())):
                    raise ValueError('本消融固定 direct、FedAvg/FedSGD、0.5噪声、全局完整秩协方差、含自身均值和原风险定义。')
                config = copy.deepcopy(task.config)
                config.pop('results_dir', None)
                config['defense']['synthesis'].pop('center_source')
                config['defense']['synthesis'].pop('mixing_mode')
                comparable.append(config)
            if reference is not None and comparable != reference:
                raise ValueError('各组配置除 center_source、mixing_mode 和结果目录外必须一致。')
            reference = comparable
            planned[name] = dict(name=name, variant=variant, center=center,
                                 formula=FORMULAS[variant].format(center=center), arguments=arguments, tasks=tasks)
    groups = [planned[f'{variant}_{center}'] for variant in variants for center in centers]
    return root, groups, base.dry_run


def main(argv=None):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    from scripts import run_privacy_experiments as runner
    os.chdir(ROOT)
    root, groups, dry_run = build_study(sys.argv[1:] if argv is None else argv)
    print(f'STUDY | {root}', flush=True)
    print('PROTOCOL | 两种中心均包含自身；全局协方差、0.5噪声不变。', flush=True)
    for group in groups:
        print(f"VARIANT | {group['name']} | {group['formula']}", flush=True)
        runner.print_plan(group['tasks'], [])
    if dry_run:
        print('DRY-RUN | 两种中心的计划配置匹配通过；未创建目录，未启动训练。')
        return 0
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(noise_scale=0.5, covariance_source='global_same_class', center_includes_source=True,
                groups=[dict(name=g['name'], variant=g['variant'], center=g['center'], formula=g['formula'],
                    arguments=g['arguments'], configs=[{k:v for k,v in t.config.items() if k != 'results_dir'}
                                                      for t in g['tasks']]) for g in groups])
    with (root/'study_plan.json').open('x') as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False, allow_nan=False)
    for group in groups:
        print(f"RUN | {group['name']} | {group['formula']}", flush=True)
        status = runner.main(group['arguments'])
        if status:
            print(f"STOP | {group['name']} 未全部完成；后续组未启动。", flush=True)
            return status
    print(f'COMPLETE | {len(groups)}组结果保存在 {root}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as error:
        print(f'ERROR | {error}', file=sys.stderr)
        raise SystemExit(2) from error
