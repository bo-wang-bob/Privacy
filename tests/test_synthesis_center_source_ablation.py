import copy
import json

import pytest
import torch

from privacy_defenses.risk_synthesis import DEFAULTS, LocalGeometry, validate_risk_synthesis
from privacy_defenses.synthesis_direct import draw_batch
from scripts import run_privacy_experiments as runner
from scripts import run_synthesis_center_ablation as old_study
from scripts import run_synthesis_center_source_ablation as study
from scripts.verify_synthesis_global_geometry import verify_center_metadata


@pytest.mark.parametrize('mixing_mode', ['source', 'risk', 'class_center'])
def test_center_scope_changes_only_weighted_mean_with_identical_global_noise(mixing_mode):
    options = {**DEFAULTS, 'noise_scale':.5, 'mixing_mode':mixing_mode,
               'global_distribution':'generate', 'center_source':'local_class_mean'}
    codes = torch.tensor([[1.,2.], [3.,4.], [8.,9.], [10.,20.]])
    labels = torch.tensor([0,0,0,1])
    geometry = LocalGeometry(codes, labels, options, 'cpu')
    # Different means and covariance ranks make accidental local-noise use visible.
    factor = torch.tensor([[2.,0.], [1.,3.]])
    geometry.global_distribution = {'classes':{
        0:dict(mean=torch.tensor([20.,30.]), factor=factor),
        1:dict(mean=torch.tensor([50.,60.]), factor=factor)}}
    risk = torch.tensor([0.,.5,.9,1.])
    weights = torch.zeros(4) if mixing_mode == 'source' else torch.ones(4) if mixing_mode == 'class_center' else risk
    # Incoming tokens are not used to recompute the fixed reference mean.
    original = codes + .25
    local_means = torch.stack([codes[:3].mean(0)]*3 + [codes[3]])
    global_means = torch.tensor([[20.,30.]]*3 + [[50.,60.]])
    noise_rng = torch.Generator().manual_seed(101)
    noise = torch.cat([torch.randn(n,2,generator=noise_rng) @ factor.T for n in (3,1)])*.5
    outputs = {}
    for center, means in [('local_class_mean',local_means), ('global_class',global_means)]:
        rng = torch.Generator().manual_seed(101)
        outputs[center] = draw_batch(geometry, original, torch.arange(4), risk,
                                    {**options,'center_source':center}, rng)
        torch.testing.assert_close(outputs[center], (1-weights[:,None])*original + weights[:,None]*means + noise)
        assert torch.equal(rng.get_state(), noise_rng.get_state())
    torch.testing.assert_close(outputs['global_class']-outputs['local_class_mean'],
                               weights[:,None]*(global_means-local_means))
    # Legacy local_class still excludes each source, including its singleton fallback.
    legacy_means = torch.stack([codes[torch.arange(4).lt(3) & torch.arange(4).ne(i)].mean(0)
                               for i in range(3)] + [codes[3]])
    legacy = draw_batch(geometry, original, torch.arange(4), risk, {**options,'center_source':'local_class'},
                        torch.Generator().manual_seed(101))
    torch.testing.assert_close(legacy, (1-weights[:,None])*original + weights[:,None]*legacy_means + noise)


def test_inclusive_local_configuration_and_metadata_do_not_reinterpret_legacy():
    config = dict(model_type='clip_lora', aggregator='fedavg', sample_users=2,
                  defense=dict(name='risk_synthesis', synthesis={**DEFAULTS,'center_source':'local_class_mean'}))
    validate_risk_synthesis(config)
    config['defense']['synthesis']['candidate_selection'] = 'first_semantic'
    with pytest.raises(ValueError, match='requires direct'):
        validate_risk_synthesis(config)
    for center, description, inclusive in [('local_class_mean','local_same_class_mean',True),
                                         ('local_class','local_same_class_leave_source_out',False),
                                         ('global_class','global_same_class_mean',True)]:
        summary = dict(options={**DEFAULTS,'center_source':center,'global_distribution':'generate'},
                       implementation='local_token_geometry_v15_direct_mixing',
                       generation_center=description, center_includes_source=inclusive)
        verify_center_metadata(summary)
        with pytest.raises(ValueError, match='inclusion metadata'):
            verify_center_metadata({**summary,'center_includes_source':not inclusive})


@pytest.mark.parametrize('smoke', [False,True])
def test_default_supplement_matches_existing_global_entrypoint(tmp_path, smoke):
    args = ['--results-root',str(tmp_path),'--dry-run', *(['--smoke'] if smoke else [])]
    root, groups, dry = study.build_study(args)
    _, existing, _ = old_study.build_study([*args,'--variants','risk,class_center'])
    assert dry and not root.exists()
    assert [g['name'] for g in groups] == ['risk_local','class_center_local']
    assert all(len(g['tasks']) == 2 for g in groups)
    for group, reference in zip(groups, existing):
        for task, old in zip(group['tasks'], reference['tasks']):
            config, old_config = copy.deepcopy(task.config), copy.deepcopy(old.config)
            opts = config['defense']['synthesis']
            assert opts['center_source'] == 'local_class_mean'
            assert opts['global_distribution'] == 'generate' and opts['class_rank'] == 'all'
            assert opts['noise_scale'] == .5 and opts['views_per_record'] == 2
            assert config['num_global_iters'] == (5 if smoke else 100)
            assert len(task.attacks) == (0 if smoke else 11) and config['seed'] == 43
            assert config['confirmation_split_sha256'] == old_config['confirmation_split_sha256']
            assert task.run_dir.parent == root/group['name']
            config.pop('results_dir'); old_config.pop('results_dir')
            opts['center_source'] = 'global_class'
            assert config == old_config


def test_full_crossed_design_and_new_seeds_stay_paired(tmp_path):
    _, groups, _ = study.build_study(['--results-root',str(tmp_path),'--centers','global,local',
                                    '--seeds','44,45','--dry-run'])
    assert [g['name'] for g in groups] == ['risk_global','risk_local','class_center_global','class_center_local']
    reference = None
    for group in groups:
        assert len(group['tasks']) == 4
        configs = []
        for task in group['tasks']:
            config = copy.deepcopy(task.config)
            config.pop('results_dir')
            opts = config['defense']['synthesis']
            assert opts.pop('center_source') == study.CENTERS[group['center']]
            assert opts.pop('mixing_mode') == group['variant']
            configs.append(config)
        if reference is not None:
            assert configs == reference
        reference = configs


@pytest.mark.parametrize('arguments', [
    ['--centers','local,local'], ['--centers','local_class'], ['--variants','source'],
    ['--variants','risk,risk'], ['--variants','shuffled_risk'], ['--models','clip_mlp'],
    ['--set','defense.synthesis.center_source=local_class'],
    ['--set','defense.synthesis.mixing_mode=risk'], ['--set','defense.synthesis.mode=shuffled_risk'],
    ['--set','defense.synthesis.noise_scale=0.1'], ['--set','defense.synthesis.class_rank=5'],
    ['--set','defense.synthesis.global_distribution=disabled'],
    ['--set','defense.synthesis.semantic_filter=true'], ['--gpus','0,1'], ['--jobs','2'],
    ['--defenses','none,risk_synthesis'], ['--set','aggregator=promptfl']])
def test_invalid_plan_never_writes_or_launches(tmp_path, monkeypatch, arguments):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid study launched training.')
    monkeypatch.setattr(runner, 'main', forbidden)
    with pytest.raises(ValueError):
        study.main(['--results-root',str(tmp_path),*arguments])
    assert not list(tmp_path.iterdir())


def test_dry_run_serial_dispatch_and_failure_stop(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner,'main',lambda args: calls.append(args) or 0)
    args = ['--results-root',str(tmp_path),'--models','clip_lora','--gpus','1']
    assert study.main([*args,'--dry-run']) == 0
    assert not calls and not list(tmp_path.iterdir())
    assert study.main(args) == 0
    assert len(calls) == 2
    plan = json.loads(next(tmp_path.glob('*/study_plan.json')).read_text())
    assert plan['center_includes_source'] and plan['covariance_source'] == 'global_same_class'
    assert [g['name'] for g in plan['groups']] == ['risk_local','class_center_local']
    calls.clear()
    monkeypatch.setattr(runner,'main',lambda args: calls.append(args) or 1)
    assert study.main(args) == 1 and len(calls) == 1
