"""Exercise real cache lifecycles and failure exits, without touching user results."""
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.risk_synthesis import DEFAULTS
from privacy_defenses import synthesis_cleanup as cleanup
from scripts.analyze_risk_synthesis import read_run
from scripts.verify_synthesis_direct import verify
from scripts.verify_synthesis_global_geometry import verify as verify_global
from scripts import run_privacy_experiments as runner
from servers.serverbase import ServerBase
from test_clip_peft_fedsgd import _audit_config
from test_risk_synthesis import make_model, dataset, deterministic_cpu


def server_at(path, retention='cleanup_on_exit'):
    return ServerBase(device=torch.device('cpu'), dataset_name='toy', model=make_model('clip_adapter'),
        train_sets=[dataset(3, 20), dataset(4, 21)], test_sets=[dataset(12, 30), dataset(13, 31)],
        class_names=['a', 'b', 'c'], batch_size=4, eval_batch_size=8, learning_rate=.05,
        num_glob_iters=2, local_epochs=1, total_users=2, user_per_round=2, eval_interval=1,
        results_dir=str(path), aggregator=build_aggregator('fedsgd', aggregation_weighting='uniform'),
        audit_config=_audit_config(), projres_config={'enabled': True, 'evaluation_interval': 1},
        defense_config={'name': 'risk_synthesis', 'synthesis': {**DEFAULTS, 'views_per_record': 2,
            'statistics_retention': retention, 'global_distribution': 'generate', 'center_source': 'global_class'}},
        method_config={'client_optimizer': 'sgd', 'seed': 42})


def raw_cache(path, policy='cleanup_on_exit'):
    directory = path / 'risk_synthesis'
    directory.mkdir(parents=True)
    cleanup.register_owner(directory, [0], policy)
    for name in ['client_0_source_codes.pt', 'client_0_distribution.pt', 'global_distribution.pt']:
        (directory / name).write_bytes(b'interrupted or corrupt tensor file')
    (directory / 'synthetic_views.csv').write_text('original diagnostic rows\n')
    (directory / 'unrelated.pt').write_bytes(b'model that must survive')
    (directory / 'client_99_distribution.pt').write_bytes(b'foreign client')
    return directory


@pytest.mark.parametrize('failure', ['training', 'audit', 'summary', 'hash'])
def test_new_policy_cleans_failures_and_preserves_metric_evidence(tmp_path, monkeypatch, failure):
    server = server_at(tmp_path)
    synth = server.defense.synthesis
    directory = tmp_path / 'risk_synthesis'
    if failure == 'training':
        def fail():
            raise RuntimeError('original training failure')
        monkeypatch.setattr(server, '_train', fail)
    elif failure == 'audit':
        original = server.auditor.finalize
        def fail(*args, **kwargs):
            result = original(*args, **kwargs)
            server.auditor.errors['test'] = 'audit failure'
            return result
        monkeypatch.setattr(server.auditor, 'finalize', fail)
    elif failure == 'summary':
        original = synth.write_summary
        def fail(status):
            original(status)
            if status == 'completed':
                raise OSError('summary close failure')
        monkeypatch.setattr(synth, 'write_summary', fail)
    else:
        with (directory / 'global_distribution.pt').open('ab') as handle:
            handle.write(b'hash mismatch')
    if failure in ('training', 'summary'):
        with pytest.raises((RuntimeError, OSError), match='failure'):
            server.train()
    else:
        server.train()
    assert not cleanup.owned_caches(directory, [0, 1])
    receipt = cleanup.read_cleanup(directory)
    assert receipt['status'] == 'cleaned_unverified' and receipt['removed_bytes'] > 0
    assert (directory / 'synthetic_views.csv').exists()
    assert (directory / 'synthesis_summary.json').exists()
    with pytest.raises(ValueError, match='without full verification'):
        verify(directory)
    with pytest.raises(ValueError, match='without full verification'):
        verify_global(directory)
    from scripts.verify_synthesis_history import verify as verify_history
    from scripts.verify_synthesis_multiview import verify as verify_multiview
    for check in (verify_history, verify_multiview):
        with pytest.raises(ValueError, match='without full verification'):
            check(directory)
    if failure in ('hash', 'summary'):
        (tmp_path / 'run_config.yaml').write_text(yaml.safe_dump(dict(model_type='clip_adapter',
            aggregator='fedsgd', num_global_iters=2, seed=42, audit=_audit_config(),
            defense=dict(name='risk_synthesis', synthesis=synth.options))))
        result = read_run(tmp_path)
        assert result['training_completed'] and not result['complete']
        assert result['synthesis_verification_status'] == 'unavailable_after_unverified_cleanup'
        assert len(result['attacks']) == 11
        assert all(a['independent_auc_verified'] for a in result['attacks'])


def test_original_exception_survives_failed_diagnostic_close(tmp_path, monkeypatch):
    server = server_at(tmp_path)
    def training():
        raise RuntimeError('original failure')
    def closing(status):
        raise OSError('secondary failure')
    monkeypatch.setattr(server, '_train', training)
    monkeypatch.setattr(server.defense.synthesis, 'close', closing)
    with pytest.raises(RuntimeError, match='original failure'):
        server.train()
    assert not cleanup.owned_caches(tmp_path / 'risk_synthesis', [0, 1])


def test_partial_initialization_is_cleaned(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('exchange failed')
    monkeypatch.setattr('privacy_defenses.global_geometry.exchange', fail)
    with pytest.raises(RuntimeError, match='exchange failed'):
        server_at(tmp_path)
    directory = tmp_path / 'risk_synthesis'
    assert not cleanup.owned_caches(directory, [0, 1])
    assert cleanup.read_cleanup(directory)['reason']['training_status'] == 'initialization_failed'


def test_existing_foreign_directory_is_not_adopted_on_init_error(tmp_path):
    directory = tmp_path / 'risk_synthesis'
    directory.mkdir()
    path = directory / 'global_distribution.pt'
    path.write_bytes(b'previous run')
    with pytest.raises(FileExistsError):
        server_at(tmp_path)
    assert path.read_bytes() == b'previous run' and not (directory / cleanup.OWNER).exists()


@pytest.mark.parametrize('policy', ['keep', 'cleanup_on_success'])
def test_old_policies_keep_failed_files(tmp_path, monkeypatch, policy):
    server = server_at(tmp_path, policy)
    def fail():
        raise RuntimeError('failed')
    monkeypatch.setattr(server, '_train', fail)
    with pytest.raises(RuntimeError):
        server.train()
    assert (tmp_path / 'risk_synthesis/global_distribution.pt').exists()
    assert cleanup.read_cleanup(tmp_path / 'risk_synthesis') is None


def test_success_still_produces_verified_compact_receipt(tmp_path):
    server = server_at(tmp_path)
    server.train()
    directory = tmp_path / 'risk_synthesis'
    assert verify(directory)['trained_views_verified'] == 32
    assert verify_global(directory)['status'] == 'verified_before_cleanup'
    assert cleanup.read_cleanup(directory) is None


def test_raw_cleanup_is_scoped_idempotent_and_does_not_load_tensors(tmp_path, monkeypatch):
    directory = raw_cache(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Do not deserialize failed tensor files')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='alive'):
        cleanup.cleanup_owned(directory, reason='parent too early')
    result = cleanup.cleanup_owned(directory, reason='failure', in_process=True)
    assert result['removed_bytes'] == 3 * len(b'interrupted or corrupt tensor file')
    assert (directory / 'unrelated.pt').read_bytes() == b'model that must survive'
    assert (directory / 'client_99_distribution.pt').exists()
    assert (directory / 'synthetic_views.csv').read_text() == 'original diagnostic rows\n'
    assert cleanup.cleanup_owned(directory, reason='again', in_process=True) == result


@pytest.mark.parametrize('change', ['symlink', 'mutated', 'foreign_name', 'write_failure'])
def test_no_unlink_before_valid_plan_and_durable_receipt(tmp_path, monkeypatch, change):
    directory = raw_cache(tmp_path)
    plan = cleanup.plan_cleanup(directory, [0], reason='test')
    target = directory / 'global_distribution.pt'
    if change == 'symlink':
        target.unlink()
        outside = tmp_path / 'outside.pt'
        outside.write_bytes(b'keep')
        target.symlink_to(outside)
    elif change == 'mutated':
        target.write_bytes(b'new data')
    elif change == 'foreign_name':
        plan['artifacts'][0]['name'] = '../outside.pt'
    else:
        original = cleanup.atomic_json
        def fail(path, value):
            if path.name == cleanup.RECEIPT:
                raise OSError('disk full')
            original(path, value)
        monkeypatch.setattr(cleanup, 'atomic_json', fail)
    with pytest.raises((ValueError, OSError)):
        cleanup.apply_cleanup(plan)
    assert (directory / 'client_0_source_codes.pt').exists()
    assert (directory / 'client_0_distribution.pt').exists()


def test_partial_deletion_retry_keeps_provenance(tmp_path, monkeypatch):
    directory = raw_cache(tmp_path)
    plan = cleanup.plan_cleanup(directory, [0], reason='test')
    original = Path.unlink
    def fail(path, *args, **kwargs):
        if path.name == 'global_distribution.pt':
            raise OSError('temporarily busy')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail)
    with pytest.raises(OSError, match='busy'):
        cleanup.apply_cleanup(plan)
    first = cleanup.read_cleanup(directory)
    assert first['status'] == 'cleanup_incomplete' and len(first['removed_files']) == 2
    monkeypatch.setattr(Path, 'unlink', original)
    second = cleanup.apply_cleanup(plan)
    assert second['status'] == 'cleaned_unverified' and len(second['removed_files']) == 3
    assert second['manifest_sha256'] == first['manifest_sha256']


@pytest.mark.parametrize('exitcode', [7, -signal.SIGBUS])
def test_launcher_cleans_after_real_child_failure(tmp_path, monkeypatch, exitcode):
    run = tmp_path / 'run'
    task = runner.ExperimentTask(run_id='failure', model='clip_adapter', runner='unused', dataset='toy',
        attacks=(), defense='risk_synthesis', seed=43, target_client_id=0,
        config={'defense': {'name': 'risk_synthesis', 'synthesis': {'statistics_retention': 'cleanup_on_exit'}}},
        run_dir=run, config_path=run / 'run_config.yaml')
    code = '''
from pathlib import Path
import os, resource, signal, sys
from privacy_defenses.synthesis_cleanup import register_owner
resource.setrlimit(resource.RLIMIT_CORE, (0,0))
p=Path(sys.argv[1])/'risk_synthesis'; p.mkdir()
register_owner(p,[0],'cleanup_on_exit')
(p/'global_distribution.pt').write_bytes(b'partial tensor')
(p/'synthetic_views.csv').write_text('retain diagnostic')
code=int(sys.argv[2])
if code < 0: os.kill(os.getpid(), -code)
raise SystemExit(code)
'''
    monkeypatch.setattr(runner, 'task_command', lambda task: [sys.executable, '-c', code, str(run), str(exitcode)])
    result = runner.run_task(task)
    assert result.returncode == exitcode
    directory = run / 'risk_synthesis'
    assert not (directory / 'global_distribution.pt').exists()
    assert (directory / 'synthetic_views.csv').read_text() == 'retain diagnostic'
    assert cleanup.read_cleanup(directory)['reason']['returncode'] == exitcode
    assert f'EXIT | returncode={exitcode}' in (run / 'run.log').read_text()


def test_historical_cli_requires_explicit_plan_and_retains_existing_summaries(tmp_path):
    from scripts.cleanup_synthesis_statistics import main, assert_idle
    run = tmp_path / 'old'
    directory = raw_cache(run, policy='keep')
    (directory / cleanup.OWNER).unlink()  # Historical run, before ownership manifests existed.
    config = run / 'run_config.yaml'
    config.write_text(yaml.safe_dump(dict(total_users=1, defense={'name': 'risk_synthesis'})))
    (directory / 'synthesis_summary.json').write_text('{"status":"failed"}')
    with (directory / 'global_distribution.pt').open('rb'):
        with pytest.raises(RuntimeError, match='open file'):
            assert_idle(run)
    plan = tmp_path / 'plan.json'
    main(['--runs', str(run), '--plan', str(plan)])
    assert (directory / 'global_distribution.pt').exists()
    main(['--apply', str(plan)])
    assert not (directory / 'global_distribution.pt').exists()
    assert (directory / 'synthesis_summary.json').read_text() == '{"status":"failed"}'
    assert (directory / 'synthetic_views.csv').exists()
