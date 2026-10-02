import copy
import csv

import pytest
import torch

from privacy_defenses.risk_synthesis import DEFAULTS, LocalGeometry
from privacy_defenses.synthesis_direct import draw_batch
from scripts import run_privacy_experiments as runner
from scripts import run_synthesis_center_ablation as study


@pytest.mark.parametrize('mixing_mode', ['source', 'class_center', 'risk'])
def test_source_dependence_with_frozen_geometry_and_identical_noise(mixing_mode):
    options = {**DEFAULTS, 'noise_scale':.5, 'mixing_mode':mixing_mode,
               'global_distribution':'generate', 'center_source':'global_class'}
    codes = torch.tensor([[1.,2.],[3.,4.]])
    geometry = LocalGeometry(codes, torch.tensor([0,0]), options, 'cpu')
    geometry.global_distribution = {'classes':{0:dict(mean=torch.tensor([2.,3.]),factor=torch.eye(2))}}
    risk = torch.tensor([0.,.75])
    def draw(source):
        return draw_batch(geometry, source, torch.arange(2), risk, options, torch.Generator().manual_seed(19))
    a, b = draw(codes), draw(codes + 7)
    expected = (torch.full_like(codes, 7) if mixing_mode == 'source' else torch.zeros_like(codes)
                if mixing_mode == 'class_center' else 7*(1-risk[:,None]).expand_as(codes))
    torch.testing.assert_close(b-a, expected)


@pytest.mark.parametrize('smoke', [False, True])
def test_study_plan_changes_only_location_and_preserves_protocol(tmp_path, smoke):
    root, groups, dry = study.build_study(['--results-root',str(tmp_path),'--dry-run',
                                         *(['--smoke'] if smoke else [])])
    assert dry and not root.exists()
    assert [g['variant'] for g in groups] == list(study.DEFAULT_VARIANTS)
    normalized = []
    for group in groups:
        assert len(group['tasks']) == 2
        configs = []
        for task in group['tasks']:
            config = copy.deepcopy(task.config)
            options = config['defense']['synthesis']
            assert options.pop('mixing_mode') == group['variant']
            assert options['noise_scale'] == .5 and options['views_per_record'] == 2
            assert options['center_source'] == 'global_class' and options['class_rank'] == 'all'
            assert config['num_global_iters'] == (5 if smoke else 100)
            assert config['eval_interval'] == (1 if smoke else 5)
            assert len(task.attacks) == (0 if smoke else 11)
            assert config['batch_size'] == 32 and config['fpl_shots'] == 100
            assert config['total_users'] == config['sample_users'] == 10
            assert config['local_epochs'] == 1 and config['aggregation_weighting'] == 'sample_count'
            assert config['seed'] == 43
            assert task.run_dir.parent == root/group['variant']
            config.pop('results_dir')
            configs.append(config)
        normalized.append(configs)
    assert normalized[0] == normalized[1] == normalized[2]


@pytest.mark.parametrize('arguments', [
    ['--variants','source,source'], ['--variants','unknown'], ['--models','clip_mlp'],
    ['--set','defense.synthesis.noise_scale=1'], ['--set','defense.synthesis.mixing_mode=source'],
    ['--set','defense.synthesis.center_source=local_class'], ['--gpus','0,1'],
    ['--set','defense.synthesis.mode=shuffled_risk'],
    ['--defenses','none,risk_synthesis']])
def test_invalid_study_fails_before_writing_or_launching(tmp_path, arguments, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid plan launched training.')
    monkeypatch.setattr(runner, 'main', forbidden)
    with pytest.raises(ValueError):
        study.main(['--results-root',str(tmp_path), *arguments])
    assert not list(tmp_path.iterdir())


def test_dry_run_and_serial_runner_routing(tmp_path, monkeypatch):
    invocations = []
    def fake_main(arguments):
        args = runner.parse_args(arguments)
        tasks, skipped = runner.build_tasks(runner.load_yaml(args.catalog), args)
        assert not skipped
        invocations.append(tasks)
        return 0
    monkeypatch.setattr(runner, 'main', fake_main)
    arguments = ['--results-root',str(tmp_path),'--models','clip_lora','--gpus','1']
    assert study.main([*arguments,'--dry-run']) == 0
    assert not invocations and not list(tmp_path.iterdir())
    assert study.main(arguments) == 0
    assert [tasks[0].config['defense']['synthesis']['mixing_mode'] for tasks in invocations] == list(study.DEFAULT_VARIANTS)
    assert all(len(tasks) == 1 for tasks in invocations)
    roots = list(tmp_path.iterdir())
    assert len(roots) == 1 and (roots[0]/'study_plan.json').exists()


def test_study_stops_after_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'main', lambda argv: calls.append(argv) or 1)
    assert study.main(['--results-root',str(tmp_path),'--models','clip_adapter']) == 1
    assert len(calls) == 1


@pytest.mark.parametrize('variants', ['risk,shuffled_risk', 'shuffled_risk', ','.join(study.VARIANTS)])
def test_shuffled_study_preserves_config_except_assigned_variant(tmp_path, variants):
    root, groups, dry = study.build_study(['--variants',variants,'--results-root',str(tmp_path),'--dry-run'])
    assert dry and not root.exists()
    assert [g['variant'] for g in groups] == variants.split(',')
    references = {}
    for group in groups:
        for task in group['tasks']:
            config=copy.deepcopy(task.config)
            options=config['defense']['synthesis']
            assert options['noise_scale']==.5 and options['views_per_record']==2
            assert options.pop('mixing_mode') == ('risk' if group['variant']=='shuffled_risk' else group['variant'])
            assert options.pop('mode') == ('shuffled_risk' if group['variant']=='shuffled_risk' else 'risk')
            config.pop('results_dir')
            assert config==references.setdefault(task.model,config)
            assert task.run_dir.parent==root/group['variant']


def test_shuffled_direct_views_use_one_permutation_and_independent_noise_stream():
    from types import SimpleNamespace
    from test_synthesis_multiview import multiview_mechanism
    draws = {}
    for mode in ('risk','shuffled_risk'):
        torch.manual_seed(1987)  # Match toy model initialization and input images.
        synth,model,images,labels,ids,_ = multiview_mechanism(k=2)
        synth.options={**DEFAULTS,'mixing_mode':'risk','mode':mode,'noise_scale':.5,'views_per_record':2}
        records = []
        views_by_round = []
        for t in range(2):
            risk=torch.linspace(0,.9,len(ids))
            views=synth.transform_views(model,SimpleNamespace(id=0),images,labels,ids,risk,t,t,t-1)
            logged=synth.pending_views[0]
            raw=torch.tensor([r['risk'] for r in logged[0]])
            used=torch.tensor([r['used_risk'] for r in logged[0]])
            assert torch.equal(raw.sort().values,used.sort().values)
            assert [r['used_risk'] for r in logged[0]] == [r['used_risk'] for r in logged[1]]
            assert [r['label'] for r in logged[0]] == labels.tolist()
            if t == 0:
                assert not used.any()
            elif mode=='shuffled_risk':
                assert not torch.equal(raw,used)
            records.append(used);views_by_round.append(views)
            synth.record_optimized_batch(0)
        draws[mode]=(synth.generators[0].get_state(),records,views_by_round)
    assert torch.equal(draws['risk'][0],draws['shuffled_risk'][0])
    # First round has all-zero risk in both modes: even generated noise is equal.
    for left,right in zip(draws['risk'][2][0],draws['shuffled_risk'][2][0]):
        torch.testing.assert_close(left,right,rtol=0,atol=0)


@pytest.mark.parametrize('mixing_mode', ['source', 'class_center'])
def test_exposure_export_distinguishes_zero_risk_from_zero_mixing(tmp_path, mixing_mode):
    from scripts.analyze_risk_synthesis import digest
    from scripts.summarize_synthesis_exposure_distribution import summarize
    directory = tmp_path/'risk_synthesis'; directory.mkdir()
    path = directory/'synthetic_exposure.csv'
    rows = [dict(client=0,sample_id=sid,label=0,requested=1,accepted=1,quality_passed='',attempts=1,
                 used_risk=int(mixing_mode == 'class_center'),risk=0 if t == 0 or sid == 0 else .75,
                 source_round=-1 if t == 0 else 0) for t in range(2) for sid in range(2)]
    with path.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result, distributions, _ = summarize(dict(complete=True,
        synthesis_options={**DEFAULTS,'mixing_mode':mixing_mode},path=str(tmp_path),run='fixture',
        protocol=dict(num_global_iters=2,local_epochs=1,seed=43,audit={'audit_client_ids':[0]}),
        sources={str(path):digest(path)},synthesis_mechanism={'counts':{'visits':4}}))
    assert [r['zero_risk_fraction'] for r in result] == [1.,0.]
    assert all(r['zero_mixing_fraction'] == (1. if mixing_mode == 'source' else 0.) for r in result)
    assert all(r['zero_risk_basis'] == 'raw_risk' and r['reference_visits'] == 1 for r in result)


@pytest.mark.parametrize('noise_options', [{}, {'noise_scale':.5}])
def test_verifier_rejects_wrong_first_round_center_weight(tmp_path, noise_options):
    from scripts.verify_synthesis_direct import read_mechanism
    # Minimal partial snapshot: check coefficient metadata before any cleanup receipt.
    summary = dict(implementation='local_token_geometry_v15_direct_mixing',
        options={**DEFAULTS, 'mixing_mode':'class_center', **noise_options},
        generation_location_mode='class_center', mixing_coefficient_source='constant_one',
        used_risk_role='class_center_mixing_coefficient', risk_ranking_policy='unchanged_when_references_available',
        noise_scale_parameter_enabled=bool(noise_options),
        generation_noise=('scaled_covariance_factor_times_standard_normal' if noise_options
                          else 'covariance_factor_times_standard_normal'),
        norm_ratio_filter_enabled=False, candidate_validity_filter_enabled=False, semantic_filter_enabled=False,
        teacher_initialized=False, semantic_quality_measured=False, norm_ratio_role='not_measured',
        semantic_failure_policy='not_checked', candidate_generation='one_draw_per_training_view',
        candidate_failure_policy='no_rejection_or_retry')
    torch.save(dict(labels=torch.tensor([0]), classes={0:dict(indices=torch.tensor([0]))},
                    geometry_source='local_class_only', source_sha256='fixture'),tmp_path/'client_0_distribution.pt')
    row = dict(round=1,client=0,step=0,sample_id=0,label=0,requested=1,accepted=1,attempts=1,selected_attempt=1,
               reason='direct_generated',quality_passed='',teacher_margin_delta='',norm_ratio='',
               original_distance='',nearest_distance='',risk=0,used_risk=1,retained_original_fraction=0,source_round=-1)
    def write_row():
        with (tmp_path/'synthetic_exposure.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    write_row()
    result = read_mechanism(tmp_path,summary,complete=False)
    assert result['direct_evidence']['original_visits_verified'] == 1
    row.update(used_risk=0, retained_original_fraction=1)
    write_row()
    with pytest.raises(ValueError, match='mixing coefficient'):
        read_mechanism(tmp_path,summary,complete=False)
    # Merely relabeling a pre-v15 summary cannot authorize a new mixing policy.
    summary['implementation'] = 'local_token_geometry_v14_direct_scaled_noise'
    with pytest.raises(ValueError, match='Historical direct'):
        read_mechanism(tmp_path,summary,complete=False)
    summary.update(implementation='local_token_geometry_v15_direct_mixing',
                   generation_location_mode='risk',mixing_coefficient_source='risk')
    summary['options'].update(mixing_mode='risk',mode='shuffled_risk')
    row.update(risk=.75,used_risk=.75,retained_original_fraction=.25,source_round=0)
    write_row()
    read_mechanism(tmp_path,summary,complete=False)
    # All per-row bounds still hold, but this is no longer a permutation.
    row.update(used_risk=.5,retained_original_fraction=.5)
    write_row()
    with pytest.raises(ValueError, match='risk multiset'):
        read_mechanism(tmp_path,summary,complete=False)
