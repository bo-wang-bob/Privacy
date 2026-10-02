#!/usr/bin/env python3
"""Plan explicit historical cache retirement; deletion requires --apply PLAN.

Never scans the results tree for targets. Metrics, CSVs, models and existing
summaries are retained byte-for-byte. No successful verification is invented.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from privacy_defenses.synthesis_cleanup import atomic_json, digest, plan_cleanup, apply_cleanup, OWNER, process_start


def assert_idle(run_dir):
    """Conservatively reject runs with live processes holding their files open."""
    run_dir = Path(run_dir).resolve(strict=True)
    owner_path = run_dir / 'risk_synthesis' / OWNER
    if owner_path.exists():
        if owner_path.is_symlink():
            raise ValueError('Cache ownership cannot be a symlink.')
        owner = json.loads(owner_path.read_text())
        if owner.get('process_start') is not None and process_start(owner['pid']) == owner['process_start']:
            raise RuntimeError(f"Training process {owner['pid']} is still alive.")
    prefix = str(run_dir) + '/'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            descriptors = list((proc / 'fd').iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            raise RuntimeError('Cannot inspect process file descriptors; run cleanup with sufficient read permissions.')
        for fd in descriptors:
            try:
                target = str(fd.readlink())
            except (FileNotFoundError, ProcessLookupError):
                continue
            if target.startswith(prefix):
                raise RuntimeError(f'Run still has an open file in process {proc.name}: {target}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+')
    parser.add_argument('--plan', type=Path, help='Write a reviewable manifest; no deletion.')
    parser.add_argument('--apply', type=Path, help='Apply exactly the previously saved plan.')
    args = parser.parse_args(argv)
    if args.apply:
        if args.runs or args.plan:
            parser.error('--apply cannot be combined with --runs/--plan.')
        payload = json.loads(args.apply.read_text())
        if payload.get('schema_version') != 1:
            raise ValueError('Unsupported historical cleanup plan.')
        for item in payload['runs']:
            run = Path(item['directory']).parent
            assert_idle(run)
            if digest(run / 'run_config.yaml') != item['run_config_sha256']:
                raise ValueError(f'Run config changed: {run}')
        report = []
        for item in payload['runs']:
            run = Path(item['directory']).parent
            assert_idle(run)
            result = apply_cleanup(item)
            report.append(dict(run=str(run), status=result['status'],
                               removed_bytes=result['removed_bytes'],
                               removed_allocated_bytes=result['removed_allocated_bytes']))
            atomic_json(args.apply.with_suffix('.report.json'), dict(runs=report))
            print(json.dumps(report[-1], ensure_ascii=False), flush=True)
        return report
    if not args.runs or not args.plan:
        parser.error('Use --runs RUN... --plan FILE to preview, or --apply FILE to delete.')
    if args.plan.exists():
        raise FileExistsError(args.plan)
    plans = []
    for supplied in args.runs:
        run = supplied.absolute()
        assert_idle(run)
        config_path = run / 'run_config.yaml'
        config = yaml.safe_load(config_path.read_text())
        if config.get('defense', {}).get('name') != 'risk_synthesis':
            raise ValueError(f'Not a synthesis run: {run}')
        plan = plan_cleanup(run / 'risk_synthesis', range(int(config['total_users'])),
                            reason=dict(trigger='explicit_historical_cleanup'))
        plan['run_config_sha256'] = digest(config_path)
        plans.append(plan)
        print(f"PLAN | {run.name} | files={len(plan['artifacts'])} | "
              f"allocated_GiB={sum(p['allocated_bytes'] for p in plan['artifacts']) / 1024**3:.3f}", flush=True)
    atomic_json(args.plan, dict(schema_version=1, runs=plans))
    return plans


if __name__ == '__main__':
    main()
