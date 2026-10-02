import csv
import json
from types import SimpleNamespace

import pytest
import torch

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.risk_synthesis import DEFAULTS, LocalGeometry, synthesis_options, validate_risk_synthesis
from privacy_defenses.synthesis_direct import DeviceGeometryCache, draw_batch
from scripts.analyze_risk_synthesis import read_synthesis_mechanism
from servers.serverbase import ServerBase
from test_clip_peft_fedsgd import ATTACKS, _audit_config
from test_risk_synthesis import make_model, dataset, deterministic_cpu
from test_synthesis_multiview import multiview_mechanism


@pytest.mark.parametrize('value', ['unchanged', 'nan', 'huge'])
def test_no_candidate_gate_or_retry_and_input_is_encoded_once(monkeypatch, value):
    synth, model, images, labels, ids, original = multiview_mechanism(k=3)
    synth.options = {**DEFAULTS, 'views_per_record': 3}
    def forbidden(*args, **kwargs):
        raise AssertionError('Teacher or legacy candidate selection must not run.')
    synth.margins = synth.candidate_metrics = forbidden
    synth.geometry[0].sample = forbidden
    calls = []
    def draw(geometry, source, *args):
        calls.append(1)
        return source if value == 'unchanged' else torch.full_like(source, float('nan')) if value == 'nan' else source*100
    monkeypatch.setattr('privacy_defenses.synthesis_direct.draw_batch', draw)
    encoded = []
    original_encode = model.encode_input_tokens
    def encode(images):
        encoded.append(1)
        return original_encode(images)
    monkeypatch.setattr(model, 'encode_input_tokens', encode)
    views = synth.transform_views(model, SimpleNamespace(id=0), images, labels, ids, torch.zeros(8), 0, 0, -1)
    assert len(calls) == len(views) == 3 and len(encoded) == 1
    assert not synth.counts  # Generation alone never commits optimizer exposure.
    for view in views:
        if value == 'nan':
            assert torch.isnan(view[:, 1:]).all()
        else:
            torch.testing.assert_close(view[:, 1:], original[:, 1:]*(1 if value == 'unchanged' else 100))
    if value != 'nan':
        synth.record_optimized_batch(0)
        assert synth.counts == dict(visits=8,requested=8,accepted=8,fallback=0)
        records = list(csv.DictReader(synth.view_handle.getvalue().splitlines()))
        assert len(records) == 24 and all(r['quality_passed'] == r['norm_ratio'] == '' for r in records)


@pytest.mark.parametrize('scale', [None, .1, .5, 1.0])
@pytest.mark.parametrize('mixing_mode', [None, 'source', 'class_center', 'risk'])
def test_class_batched_sampling_matches_formula_and_keeps_global_singleton(scale, mixing_mode):
    options = {**DEFAULTS,'global_distribution':'generate','center_source':'global_class'}
    if scale is not None:
        options['noise_scale'] = scale
    if mixing_mode is not None:
        options['mixing_mode'] = mixing_mode
    codes = torch.tensor([[1.,2.],[2.,4.],[7.,8.]])
    geometry = LocalGeometry(codes,torch.tensor([0,0,1]),options,'cpu')
    geometry.global_distribution = {'classes':{
        0:dict(mean=torch.tensor([3.,5.]),factor=torch.tensor([[1.],[2.]])),
        1:dict(mean=torch.tensor([8.,9.]),factor=torch.empty(2,0))}}
    risk = torch.tensor([0.,.5,.9]); a=torch.Generator().manual_seed(5); b=torch.Generator().manual_seed(5)
    actual = draw_batch(geometry,codes,torch.arange(3),risk,options,a)
    noise = torch.randn(2,1,generator=b) @ torch.tensor([[1.,2.]])
    means = torch.tensor([[3.,5.],[3.,5.],[8.,9.]])
    expected = (codes.clone() if mixing_mode == 'source' else means.clone() if mixing_mode == 'class_center'
                else (1-risk[:,None])*codes + risk[:,None]*means)
    expected[:2] += (1.0 if scale is None else scale)*noise
    torch.testing.assert_close(actual, expected)
    single = geometry.sample(codes[2],2,.9,options,torch.Generator().manual_seed(5))
    torch.testing.assert_close(single,expected[2])


@pytest.mark.parametrize('override', [dict(semantic_filter=True),dict(attempts=2),
    dict(margin_tolerance=.02),dict(min_class_samples=3),dict(risk_history='zero_risk_frequency'),
    dict(noise_scale=0),dict(noise_scale=-.5),dict(noise_scale=float('nan')),
    dict(noise_scale=float('inf')),dict(noise_scale=True),dict(noise_scale='0.5'),dict(noise_scale=None),
    dict(mixing_mode='invalid'), dict(mixing_mode=None), dict(mixing_mode=True),
    dict(mixing_mode='class_center', candidate_selection='first_semantic')])
def test_direct_configuration_rejects_conflicting_old_filter_options(override):
    config=dict(model_type='clip_lora',aggregator='fedavg',sample_users=2,
                defense=dict(name='risk_synthesis',synthesis={**DEFAULTS,**override}))
    with pytest.raises(ValueError):
        validate_risk_synthesis(config)


@pytest.mark.parametrize('kind', ['clip_adapter','clip_lora'])
@pytest.mark.parametrize('views', [1,2,3])
@pytest.mark.parametrize('scale,mixing_mode,risk_mode,center_source', [
    (None,None,'risk','global_class'), (.5,None,'risk','global_class'), (.5,'source','risk','global_class'),
    (.5,'class_center','risk','global_class'), (.5,'risk','risk','global_class'),
    (.5,'risk','shuffled_risk','global_class'), (.5,'risk','risk','local_class_mean'),
    (.5,'class_center','risk','local_class_mean')])
def test_direct_end_to_end_no_teacher_original_membership_and_audit(tmp_path, monkeypatch, kind, views, scale, mixing_mode, risk_mode, center_source):
    def forbidden(*args,**kwargs):
        raise AssertionError('The synthesis teacher must never be evaluated.')
    monkeypatch.setattr('privacy_defenses.risk_synthesis.token_features',forbidden)
    audit = _audit_config(); audit.update(audit_batch_size=4,grad_sample_chunk_size=2)
    noise_options = {'mode':risk_mode, **({} if scale is None else {'noise_scale':scale})}
    if mixing_mode is not None:
        noise_options['mixing_mode'] = mixing_mode
    server = ServerBase(device=torch.device('cpu'),dataset_name='toy',model=make_model(kind),
        train_sets=[dataset(3,20),dataset(4,21)],test_sets=[dataset(12,30),dataset(13,31)],
        class_names=['a','b','c'],batch_size=4,eval_batch_size=8,learning_rate=.05,
        num_glob_iters=3,local_epochs=2,total_users=2,user_per_round=2,eval_interval=1,
        results_dir=str(tmp_path),aggregator=build_aggregator('fedavg',aggregation_weighting='sample_count'),
        audit_config=audit,projres_config={'enabled':True,'evaluation_interval':1},
        defense_config={'name':'risk_synthesis','synthesis':{**DEFAULTS,'views_per_record':views,
            'global_distribution':'generate','center_source':center_source,**noise_options}},
        method_config={'client_optimizer':'sgd','seed':42})
    synth = server.defense.synthesis
    assert synth.teacher is None and not hasattr(synth,'text') and not synth.semantic_sources
    for path in (tmp_path/'risk_synthesis').glob('client_*_distribution.pt'):
        state = torch.load(path,map_location='cpu',weights_only=True)
        assert not {'semantic_class_means','semantic_source_features'} & set(state)
    result = server.train()
    assert not server.auditor.errors and {r['attack'] for r in result} == ATTACKS
    assert all(r['member_count'] == r['nonmember_count'] == 9 for r in result)
    assert synth.counts == dict(visits=126,requested=126,accepted=126,fallback=0)
    assert not synth.pending_views
    summary = json.loads((tmp_path/'risk_synthesis/synthesis_summary.json').read_text())
    assert summary['generation_center'] == ('local_same_class_mean' if center_source == 'local_class_mean'
                                             else 'global_same_class_mean')
    assert summary['center_includes_source'] is True
    assert summary['geometry_source'] == 'global_same_class'
    if scale is None:
        assert summary['implementation'] == 'local_token_geometry_v13_direct_unit_noise'
        assert summary['noise_scale_parameter_enabled'] is False
        assert summary['generation_noise'] == 'covariance_factor_times_standard_normal'
        assert 'noise_scale' not in summary['options']
    else:
        assert summary['implementation'] == ('local_token_geometry_v15_direct_mixing' if mixing_mode is not None
                                             else 'local_token_geometry_v14_direct_scaled_noise')
        assert summary['noise_scale_parameter_enabled'] is True
        assert summary['generation_noise'] == 'scaled_covariance_factor_times_standard_normal'
        assert summary['options']['noise_scale'] == scale
    assert summary['statistics_storage']['status'] == 'cleaned'
    assert not list((tmp_path/'risk_synthesis').glob('client_*_distribution.pt'))
    verified = read_synthesis_mechanism(tmp_path/'risk_synthesis',summary,complete=True)
    assert verified['direct_evidence']['trained_views_verified'] == 126*views
    assert verified['direct_evidence']['full_geometry_replay_available'] is False
    assert verified['global_geometry_evidence']['covariance_recomputed_now'] is False
    assert 'quality_failed' not in verified['counts']
    assert all(e['synthetic_steps'].tolist() == [6]*len(e['synthetic_steps']) for e in synth.exposure.values())
    from scripts.verify_synthesis_history import verify
    assert verify(tmp_path/'risk_synthesis')['direct']['trained_views_verified'] == 126*views
    from scripts.summarize_synthesis_exposure_distribution import summarize
    from scripts.analyze_risk_synthesis import digest
    path = tmp_path/'risk_synthesis'/('synthetic_views.csv' if views > 1 else 'synthetic_exposure.csv')
    rows, distributions, _ = summarize(dict(complete=True,synthesis_options=synth.options,path=str(tmp_path),
        run='direct_toy',protocol={'num_global_iters':3,'local_epochs':2,'seed':42,'audit':{'audit_client_ids':[0]}},
        sources={str(path):digest(path)},synthesis_mechanism=verified))
    assert all(r['semantic_failure_fraction'] is None and r['mean_attempts'] == 1 for r in rows)
    assert 'semantic_failure_fraction' not in {r['metric'] for r in distributions}
    if mixing_mode in ('source', 'class_center'):
        assert all(r['zero_risk_basis'] == 'raw_risk' for r in rows)
        assert all(r['zero_mixing_fraction'] == (1. if mixing_mode == 'source' else 0.) for r in rows)
        # batch<=4 has ceil(.8*n)==n, so every available-reference risk is positive.
        assert sum(r['zero_risk_count'] for r in rows) == 0
    # Auditors must not silently turn unchecked quality into "passed".
    exposure = tmp_path/'risk_synthesis/synthetic_exposure.csv'
    rows = list(csv.DictReader(exposure.open()))
    if mixing_mode is not None:
        assert summary['generation_location_mode'] == mixing_mode
        assert summary['used_risk_role'] == 'class_center_mixing_coefficient'
        assert any(float(r['risk']) > 0 for r in rows if int(r['source_round']) >= 0)
        assert all(float(r['risk']) == 0 for r in rows if int(r['source_round']) < 0)
        for row in rows:
            expected = 0. if mixing_mode == 'source' else 1. if mixing_mode == 'class_center' else float(row['risk'])
            if risk_mode == 'risk':
                assert float(row['used_risk']) == expected
            assert float(row['retained_original_fraction']) == 1-float(row['used_risk'])
        if risk_mode == 'shuffled_risk':
            assert any(float(r['risk']) != float(r['used_risk']) for r in rows if int(r['source_round']) >= 0)
    rows[0]['quality_passed'] = '1'
    with exposure.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError):
        read_synthesis_mechanism(tmp_path/'risk_synthesis',summary,complete=True)


def test_old_saved_filtered_options_keep_their_protocol():
    old = synthesis_options({'replacement_policy':'all','semantic_filter':True})
    assert old['candidate_selection']=='first_semantic' and old['attempts']==2
    assert synthesis_options({})['candidate_selection']=='direct'
    assert synthesis_options({'candidate_selection':'direct'}) == DEFAULTS
    assert DEFAULTS['class_rank'] == 'all'
    assert 'noise_scale' not in DEFAULTS
    assert old['noise_scale'] == .1
    assert old['class_rank'] == 5
    assert synthesis_options({'candidate_selection':'direct', 'class_rank':5})['class_rank'] == 5


def test_historical_scaled_direct_snapshot_remains_readable(tmp_path):
    # An archived v12 snapshot before any optimized batch: no new generation.
    summary = dict(
        implementation='local_token_geometry_v12_direct',
        options={**DEFAULTS, 'noise_scale':.1, 'statistics_retention':'keep'},
        norm_ratio_filter_enabled=False, candidate_validity_filter_enabled=False,
        semantic_filter_enabled=False, teacher_initialized=False, semantic_quality_measured=False,
        norm_ratio_role='not_measured', semantic_failure_policy='not_checked',
        candidate_generation='one_draw_per_training_view',
        candidate_failure_policy='no_rejection_or_retry',
    )
    torch.save(dict(labels=torch.tensor([0]), classes={0:dict(indices=torch.tensor([0]))},
                    geometry_source='local_class_only', source_sha256='snapshot-fixture'),
               tmp_path/'client_0_distribution.pt')
    (tmp_path/'synthetic_exposure.csv').write_text('round,client,step\n')
    (tmp_path/'synthesis_summary.json').write_text(json.dumps(summary))
    evidence = read_synthesis_mechanism(tmp_path, summary, complete=False)['direct_evidence']
    assert evidence['status'] == 'partial_snapshot'
    assert evidence['original_visits_verified'] == evidence['trained_views_verified'] == 0
    # A v12 scale cannot be carried into a v13 summary unnoticed.
    summary.update(implementation='local_token_geometry_v13_direct_unit_noise',
                   generation_noise='covariance_factor_times_standard_normal',
                   noise_scale_parameter_enabled=False)
    with pytest.raises(ValueError, match='unit-noise'):
        read_synthesis_mechanism(tmp_path, summary, complete=False)


@pytest.mark.parametrize('rank', ['all', 5, 0, -1, True, None, 'full', 2.5])
def test_generation_rank_configuration(rank):
    config = dict(model_type='clip_lora', aggregator='fedavg', sample_users=2,
                  defense=dict(name='risk_synthesis', synthesis={**DEFAULTS, 'class_rank':rank}))
    if rank == 'all' or type(rank) is int and rank > 0:
        validate_risk_synthesis(config)
    else:
        with pytest.raises(ValueError, match='class_rank'):
            validate_risk_synthesis(config)


def test_single_view_api_does_not_discard_requested_multiple_views():
    synth, model, images, labels, ids, _ = multiview_mechanism()
    synth.options = {**DEFAULTS,'views_per_record':2}
    with pytest.raises(ValueError,match='transform_views'):
        synth.transform(model,SimpleNamespace(id=0),images,labels,ids,torch.zeros(8),0,0,-1)
    assert not synth.pending_views


def test_device_geometry_cache_reuses_sources_and_evicts_by_bytes():
    cache = DeviceGeometryCache(max_bytes=32)
    a, b, c = (torch.arange(4, dtype=torch.float64) + i for i in range(3))
    cached_a = cache.get(a, 'cpu')
    assert cached_a.dtype == torch.float32
    cache.get(b, 'cpu')
    assert cache.get(a, 'cpu') is cached_a
    cache.get(c, 'cpu')  # b is the least recently used entry.
    assert set(cache.entries) == {id(a), id(c)}
    assert cache.resident_bytes == 32 and cache.evictions == 1
    oversized = torch.ones(9)
    torch.testing.assert_close(cache.get(oversized, 'cpu'), oversized)
    assert id(oversized) not in cache.entries and cache.resident_bytes == 0
    cache.get(a, 'cpu')
    cache.clear()
    assert cache.resident_bytes == 0 and not cache.entries


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')
@pytest.mark.parametrize('center_source', ['global_class', 'local_class', 'local_class_mean'])
@pytest.mark.parametrize('scale', [None, .5])
@pytest.mark.parametrize('mixing_mode', [None, 'source', 'class_center'])
def test_cuda_draws_match_cpu_all_directions_and_preserve_rng(center_source, scale, mixing_mode):
    from privacy_defenses.global_geometry import aggregate_moments, local_moments
    rng = torch.Generator().manual_seed(83)
    codes = torch.randn(25, 8, generator=rng)
    labels = torch.tensor([0]*12 + [1]*12 + [2])
    options = {**DEFAULTS, 'global_distribution':'generate', 'center_source':center_source}
    if mixing_mode is not None:
        options['mixing_mode'] = mixing_mode
    if scale is not None:
        options['noise_scale'] = scale
    geometry = LocalGeometry(codes, labels, options, 'cpu')
    geometry.global_distribution = aggregate_moments({0:local_moments(geometry, 'cpu')}, 'all', 'cpu')
    assert geometry.global_distribution['classes'][0]['used_rank'] > 5
    cache = DeviceGeometryCache(max_bytes=1024**2)
    risk = torch.linspace(0, .95, len(codes))
    cpu_rng = torch.Generator().manual_seed(983)
    gpu_rng = torch.Generator().manual_seed(983)
    for _ in range(3):
        cpu = draw_batch(geometry, codes, torch.arange(len(codes)), risk, options, cpu_rng)
        gpu = draw_batch(geometry, codes.cuda(), torch.arange(len(codes)), risk.cuda(), options, gpu_rng, cache)
        assert gpu.is_cuda
        torch.testing.assert_close(gpu.cpu(), cpu, atol=2e-6, rtol=2e-6)
        assert torch.equal(cpu_rng.get_state(), gpu_rng.get_state())
    assert cache.hits > 0 and all(entry[1].is_cuda for entry in cache.entries.values())
    # A second local client shares the broadcast tensors and must reuse the cache.
    other = LocalGeometry(codes, labels, options, 'cpu')
    other.global_distribution = geometry.global_distribution
    misses = cache.misses
    draw_batch(other, codes.cuda(), torch.arange(len(codes)), risk.cuda(), options, gpu_rng, cache)
    assert cache.misses == misses
    assert cache.summary()['device'].startswith('cuda')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')
def test_cuda_generation_keeps_tokens_on_device_and_cls_unchanged(monkeypatch):
    synth, model, images, labels, ids, original = multiview_mechanism(k=2)
    synth.options = {**DEFAULTS, 'views_per_record':2}
    original = original.cuda()
    monkeypatch.setattr(model, 'encode_input_tokens', lambda _: original)
    original_draw = draw_batch
    observed = []
    def record_draw(geometry, source, indices, risk, *args):
        observed.append((source.device.type, risk.device.type))
        return original_draw(geometry, source, indices, risk, *args)
    monkeypatch.setattr('privacy_defenses.synthesis_direct.draw_batch', record_draw)
    views = synth.transform_views(model, SimpleNamespace(id=0), images, labels.cuda(), ids,
                                  torch.zeros(len(ids)), 0, 0, -1)
    assert observed == [('cuda','cuda')]*2
    for view in views:
        assert view.is_cuda and not view.requires_grad
        torch.testing.assert_close(view[:,0], original[:,0], rtol=0, atol=0)
    synth.record_optimized_batch(0)
    assert synth.counts['accepted'] == len(ids)
    assert synth.view_counts['accepted'] == 2*len(ids)
