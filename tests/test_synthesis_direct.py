import csv
import json
from types import SimpleNamespace

import pytest
import torch

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.risk_synthesis import DEFAULTS, LocalGeometry, synthesis_options, validate_risk_synthesis
from privacy_defenses.synthesis_direct import draw_batch
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


def test_class_batched_sampling_matches_formula_and_keeps_global_singleton():
    options = {**DEFAULTS,'global_distribution':'generate','center_source':'global_class'}
    codes = torch.tensor([[1.,2.],[2.,4.],[7.,8.]])
    geometry = LocalGeometry(codes,torch.tensor([0,0,1]),options,'cpu')
    geometry.global_distribution = {'classes':{
        0:dict(mean=torch.tensor([3.,5.]),factor=torch.tensor([[1.],[2.]])),
        1:dict(mean=torch.tensor([8.,9.]),factor=torch.empty(2,0))}}
    risk = torch.tensor([0.,.5,.9]); a=torch.Generator().manual_seed(5); b=torch.Generator().manual_seed(5)
    actual = draw_batch(geometry,codes,torch.arange(3),risk,options,a)
    noise = torch.randn(2,1,generator=b) @ torch.tensor([[1.,2.]])
    expected = (1-risk[:,None])*codes + risk[:,None]*torch.tensor([[3.,5.],[3.,5.],[8.,9.]])
    expected[:2] += options['noise_scale']*noise
    torch.testing.assert_close(actual, expected)
    single = geometry.sample(codes[2],2,.9,options,torch.Generator().manual_seed(5))
    torch.testing.assert_close(single,expected[2])


@pytest.mark.parametrize('override', [dict(semantic_filter=True),dict(attempts=2),
    dict(margin_tolerance=.02),dict(min_class_samples=3),dict(risk_history='zero_risk_frequency')])
def test_direct_configuration_rejects_conflicting_old_filter_options(override):
    config=dict(model_type='clip_lora',aggregator='fedavg',sample_users=2,
                defense=dict(name='risk_synthesis',synthesis={**DEFAULTS,**override}))
    with pytest.raises(ValueError):
        validate_risk_synthesis(config)


@pytest.mark.parametrize('kind', ['clip_adapter','clip_lora'])
@pytest.mark.parametrize('views', [1,2,3])
def test_direct_end_to_end_no_teacher_original_membership_and_audit(tmp_path, monkeypatch, kind, views):
    def forbidden(*args,**kwargs):
        raise AssertionError('The synthesis teacher must never be evaluated.')
    monkeypatch.setattr('privacy_defenses.risk_synthesis.token_features',forbidden)
    audit = _audit_config(); audit.update(audit_batch_size=4,grad_sample_chunk_size=2)
    server = ServerBase(device=torch.device('cpu'),dataset_name='toy',model=make_model(kind),
        train_sets=[dataset(3,20),dataset(4,21)],test_sets=[dataset(12,30),dataset(13,31)],
        class_names=['a','b','c'],batch_size=4,eval_batch_size=8,learning_rate=.05,
        num_glob_iters=3,local_epochs=2,total_users=2,user_per_round=2,eval_interval=1,
        results_dir=str(tmp_path),aggregator=build_aggregator('fedavg',aggregation_weighting='sample_count'),
        audit_config=audit,projres_config={'enabled':True,'evaluation_interval':1},
        defense_config={'name':'risk_synthesis','synthesis':{**DEFAULTS,'views_per_record':views,
            'global_distribution':'generate','center_source':'global_class'}},
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
    assert summary['implementation'] == 'local_token_geometry_v12_direct'
    verified = read_synthesis_mechanism(tmp_path/'risk_synthesis',summary,complete=True)
    assert verified['direct_evidence']['trained_views_verified'] == 126*views
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
    # Auditors must not silently turn unchecked quality into "passed".
    exposure = tmp_path/'risk_synthesis/synthetic_exposure.csv'
    rows = list(csv.DictReader(exposure.open())); rows[0]['quality_passed'] = '1'
    with exposure.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError):
        read_synthesis_mechanism(tmp_path/'risk_synthesis',summary,complete=True)


def test_old_saved_filtered_options_keep_their_protocol():
    old = synthesis_options({'replacement_policy':'all','semantic_filter':True})
    assert old['candidate_selection']=='first_semantic' and old['attempts']==2
    assert synthesis_options({})['candidate_selection']=='direct'
    assert synthesis_options({'candidate_selection':'direct'}) == DEFAULTS


def test_single_view_api_does_not_discard_requested_multiple_views():
    synth, model, images, labels, ids, _ = multiview_mechanism()
    synth.options = {**DEFAULTS,'views_per_record':2}
    with pytest.raises(ValueError,match='transform_views'):
        synth.transform(model,SimpleNamespace(id=0),images,labels,ids,torch.zeros(8),0,0,-1)
    assert not synth.pending_views
