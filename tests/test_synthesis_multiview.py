import copy
import csv
import io
import json
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.controller import DefenseController
from privacy_defenses.risk_synthesis import DEFAULTS, LocalGeometry, validate_risk_synthesis
from servers.serverbase import ServerBase
from test_clip_peft_fedsgd import ATTACKS, _audit_config
from test_risk_synthesis import make_model, dataset, deterministic_cpu
from test_risk_synthesis_all import rows
from test_synthesis_history_selection import advanced_mechanism


def multiview_mechanism(k=2, history=False):
    synth, *rest = advanced_mechanism(history=history)
    synth.options['views_per_record'] = k
    fields = synth.writer.fieldnames
    synth.view_handle = io.StringIO()
    synth.view_writer = csv.DictWriter(synth.view_handle, fieldnames=fields + ['view_index', 'loss_weight'])
    synth.view_writer.writeheader()
    synth.handle = io.StringIO()
    synth.writer = csv.DictWriter(synth.handle, fieldnames=fields + [
        'views_per_record', 'quality_passed_views', 'representative_view_index', 'total_attempts'])
    synth.writer.writeheader()
    synth.exposure[0]['synthetic_views'] = torch.zeros(8, dtype=torch.long)
    return synth, *rest


def test_class_noise_ignores_other_class_geometry_and_has_no_pooled_directions():
    # Class0 varies only along x; class1 varies strongly along y.
    codes = torch.tensor([[1.,2.],[2.,2.],[3.,2.],[4.,2.],
                          [100.,0.],[100.,100.],[100.,200.],[100.,300.]])
    labels = torch.tensor([0]*4+[1]*4)
    first = LocalGeometry(codes, labels, DEFAULTS, 'cpu')
    changed = codes.clone()
    changed[4:] = torch.randn(4,2)*10000
    second = LocalGeometry(changed, labels, DEFAULTS, 'cpu')
    a, b = torch.Generator().manual_seed(8), torch.Generator().manual_seed(8)
    for _ in range(20):
        x = first.sample(codes[0],0,.6,DEFAULTS,a)
        y = second.sample(codes[0],0,.6,DEFAULTS,b)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert x[1].item()==2., 'Other-class y variation must not enter class0 noise.'
    assert 'pooled_factor' not in first.state() and 'pooled_metadata' not in first.state()
    assert first.state()['geometry_source']=='local_class_only'


@pytest.mark.parametrize('removed', [{'pooled_rank':16},{'shrinkage':0.5}])
def test_removed_pooled_options_fail_instead_of_silently_changing_old_protocol(removed):
    config = dict(model_type='clip_lora',aggregator='fedavg',sample_users=2,
                  defense=dict(name='risk_synthesis',synthesis={**DEFAULTS,**removed}))
    with pytest.raises(ValueError,match='removed options'):
        validate_risk_synthesis(config)


def test_zero_variance_class_does_not_borrow_noise_from_other_classes():
    synth, model, images, labels, ids, original = multiview_mechanism()
    synth.options['semantic_filter'] = False
    factor = synth.geometry[0].classes[0]['factor']
    synth.geometry[0].classes[0]['factor'] = factor[:, :0]
    with pytest.raises(RuntimeError,match='unchanged_candidate'):
        synth.transform_views(model,SimpleNamespace(id=0),images,labels,ids,torch.zeros(8),0,0,-1)
    assert not synth.pending_views and not synth.counts


@pytest.mark.parametrize('k', [2, 3])
def test_all_views_changed_distinct_and_history_counts_original_once(k):
    synth, model, images, labels, ids, original = multiview_mechanism(k, history=True)
    synth.margins = lambda tokens, labels: torch.zeros(len(tokens))
    synth.options['mode'] = 'shuffled_risk'
    views = synth.transform_views(model, SimpleNamespace(id=0), images, labels, ids,
                                  torch.zeros(8), 0, 0, -1)
    assert len(views) == k and not synth.counts and not rows(synth)
    for v, view in enumerate(views):
        assert not view.requires_grad
        torch.testing.assert_close(view[:, 0], original[:, 0], rtol=0, atol=0)
        assert ((view-original).flatten(1).norm(dim=1) > 0).all()
        for other in views[:v]:
            assert ((view-other).flatten(1).norm(dim=1) > 0).all()
    assert synth.history.clients[0]['visits'].sum() == 0
    synth.record_optimized_batch(0)
    assert synth.counts['visits'] == synth.counts['accepted'] == 8
    assert synth.view_counts['visits'] == 8*k
    assert synth.history.clients[0]['visits'].tolist() == [1]*8
    assert synth.exposure[0]['synthetic_steps'].tolist() == [1]*8
    assert synth.exposure[0]['synthetic_views'].tolist() == [k]*8
    logged = list(csv.DictReader(io.StringIO(synth.view_handle.getvalue())))
    assert len(rows(synth)) == 8 and len(logged) == 8*k
    for sid in range(8):
        local = [r for r in logged if int(r['sample_id']) == sid]
        assert sum(float(r['loss_weight']) for r in local) == pytest.approx(1)
        assert {r['used_risk'] for r in local} == {'0.0'}
    # Next round: shared shuffle happens once, not separately per view.
    risk = torch.arange(8.) / 8
    synth.transform_views(model, SimpleNamespace(id=0), images, labels, ids,
                          risk, 1, 1, 0, raw_scores=torch.arange(8.))
    synth.record_optimized_batch(0)
    logged = list(csv.DictReader(io.StringIO(synth.view_handle.getvalue())))
    for sid in range(8):
        assert len({r['used_risk'] for r in logged if r['round']=='2' and int(r['sample_id'])==sid}) == 1
    assert synth.history.clients[0]['visits'].tolist() == [1]*8


def test_each_view_keeps_its_best_semantic_fallback_and_records_all_views():
    synth, model, images, labels, ids, original = multiview_mechanism()
    margins = iter([0., -.4, -.8, 0., -.9, -.2])
    synth.margins = lambda tokens, labels: torch.full((len(tokens),), next(margins))
    views = synth.transform_views(model, SimpleNamespace(id=0), images, labels, ids,
                                  torch.zeros(8), 0, 0, -1)
    synth.record_optimized_batch(0)
    assert len(views) == 2
    assert all(r['quality_passed_views']=='0' and r['representative_view_index']=='0' for r in rows(synth))
    logged = list(csv.DictReader(io.StringIO(synth.view_handle.getvalue())))
    assert {r['selected_attempt'] for r in logged if r['view_index']=='0'} == {'1'}
    assert {r['selected_attempt'] for r in logged if r['view_index']=='1'} == {'2'}
    assert synth.counts['quality_failed'] == 8 and synth.view_counts['quality_failed'] == 16
    assert synth.counts['fallback'] == synth.view_counts['fallback'] == 0


def test_duplicate_second_view_aborts_without_committing_original_visits():
    synth, model, images, labels, ids, original = multiview_mechanism(history=True)
    synth.options['semantic_filter'] = False
    synth.geometry[0].sample = lambda source, *args, **kwargs: source + .01
    with pytest.raises(RuntimeError, match='duplicate_training_view'):
        synth.transform_views(model, SimpleNamespace(id=0), images, labels, ids,
                              torch.zeros(8), 0, 0, -1)
    assert not synth.counts and not synth.view_counts and not rows(synth)
    assert not synth.pending_views and not synth.pending_history
    assert synth.history.clients[0]['visits'].sum() == 0


@pytest.mark.parametrize('value', [0, -1, 1.5, True, '2'])
def test_invalid_view_count_rejected(value):
    config = dict(model_type='clip_lora', aggregator='fedavg', sample_users=2,
                  defense=dict(name='risk_synthesis', synthesis={**DEFAULTS, 'views_per_record': value}))
    with pytest.raises(ValueError, match='views_per_record'):
        validate_risk_synthesis(config)


def test_controller_matches_joint_mean_loss_and_one_momentum_step_per_original_batch():
    class Model(torch.nn.Linear):
        def forward_tokens(self, x):
            return self(x)
    model = Model(3, 2)
    expected = copy.deepcopy(model)
    batches = [(torch.randn(n, 3), torch.arange(n) % 2, torch.arange(n)) for n in (3, 1)]
    views = [(x+.3, x-.7) for x, _, _ in batches]
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.2)
    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=.1, momentum=.9, weight_decay=.2)
    for batch_views, (_, labels, _) in zip(views, batches):
        reference_optimizer.zero_grad()
        F.cross_entropy(expected(torch.cat(batch_views)), labels.repeat(2)).backward()
        reference_optimizer.step()
    controller = DefenseController.__new__(DefenseController)
    controller.device = torch.device('cpu')
    controller._www_pending_states = {}
    controller.steps = defaultdict(int)
    controller._record = lambda *args: None
    iterator = iter(views)
    commits = []
    controller.synthesis = SimpleNamespace(transform_views=lambda *args, **kwargs: next(iterator),
                                           record_optimized_batch=lambda client: commits.append(client))
    user = SimpleNamespace(id=0, iter_www_local_batches=lambda: iter(batches))
    controller._synthesis_training(user, model, optimizer, 0, False)
    assert controller.steps[0] == len(commits) == 2
    for actual, target in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, target, atol=1e-7, rtol=1e-6)
    for actual, target in zip(optimizer.state.values(), reference_optimizer.state.values()):
        torch.testing.assert_close(actual['momentum_buffer'], target['momentum_buffer'], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('kind,history,selection', [
    ('clip_adapter', 'none', 'first_semantic'),
    ('clip_lora', 'zero_risk_frequency', 'least_local_similarity'),
])
@pytest.mark.parametrize('exchange_mode,center_source', [
    ('disabled','local_class'), ('generate','local_class'), ('share_only','local_class'),
    ('generate','global_class')])
def test_multiview_full_training_keeps_original_membership(kind, history, selection, exchange_mode, center_source, tmp_path):
    audit = _audit_config()
    audit.update(audit_batch_size=4, grad_sample_chunk_size=2)
    server = ServerBase(device=torch.device('cpu'), dataset_name='toy', model=make_model(kind),
        train_sets=[dataset(3,20), dataset(4,21)], test_sets=[dataset(12,30), dataset(13,31)],
        class_names=['a','b','c'], batch_size=4, eval_batch_size=8, learning_rate=.05,
        num_glob_iters=3, local_epochs=2, total_users=2, user_per_round=2, eval_interval=1,
        results_dir=str(tmp_path), aggregator=build_aggregator('fedavg', aggregation_weighting='sample_count'),
        audit_config=audit, projres_config={'enabled':True,'evaluation_interval':1},
        defense_config={'name':'risk_synthesis','synthesis':{**DEFAULTS,'views_per_record':2,
            'risk_history':history,'candidate_selection':selection,'mode':'shuffled_risk',
            'global_distribution':exchange_mode,'center_source':center_source}},
        method_config={'client_optimizer':'sgd','seed':42})
    synth = server.defense.synthesis
    if exchange_mode != 'disabled':
        assert synth.global_exchange['aggregation_count']==1
        before = synth.global_exchange['sha256']
        assert all(g.global_distribution is not None for g in synth.geometry.values())
        with pytest.raises(RuntimeError,match='exactly once'):
            synth.initialize(server.ctx.users,server.model,tmp_path)
    result = server.train()
    assert server.auditor.errors == {} and {r['attack'] for r in result} == ATTACKS
    assert all(r['member_count']==r['nonmember_count']==9 for r in result)
    synth = server.defense.synthesis
    assert synth.counts['visits'] == 126 and synth.view_counts['visits'] == 252
    assert not synth.pending_views and not synth.pending_history
    assert all((e['synthetic_steps']==6).all() and (e['synthetic_views']==12).all()
               and (e['real_steps']==0).all() for e in synth.exposure.values())
    assert len(list(csv.DictReader((tmp_path/'risk_synthesis/synthetic_exposure.csv').open()))) == 126
    assert len(list(csv.DictReader((tmp_path/'risk_synthesis/synthetic_views.csv').open()))) == 252
    summary = json.loads((tmp_path/'risk_synthesis/synthesis_summary.json').read_text())
    assert summary['implementation']==('local_token_geometry_v9_global_center' if center_source=='global_class' else
                                       'local_token_geometry_v7_class_only' if exchange_mode=='disabled'
                                       else 'local_token_geometry_v8_global_class')
    if exchange_mode != 'disabled':
        assert synth.global_exchange['sha256']==before
        assert summary['shared_geometry'] and summary['generation_center']==f'{center_source.split("_")[0]}_same_class_leave_source_out'
        from scripts.verify_synthesis_global_geometry import verify as verify_global
        evidence = verify_global(tmp_path/'risk_synthesis')
        assert evidence['recipients']==[0,1] and evidence['classes']==3
    from scripts.analyze_risk_synthesis import read_synthesis_mechanism
    verified = read_synthesis_mechanism(tmp_path/'risk_synthesis', summary, complete=True)
    assert verified['counts']['visits'] == 126
    assert verified['multiview_evidence']['trained_views_verified'] == 252
    from scripts.verify_synthesis_history import verify
    replay = verify(tmp_path/'risk_synthesis')
    assert replay['multiview']['original_visits_verified'] == 126
    assert replay['multiview']['trained_views_verified'] == 252
    if history != 'none':
        assert replay['history_visits_verified'] == 126
        assert replay['multiview']['candidates_verified'] == 504
    from scripts.summarize_synthesis_exposure_distribution import summarize
    from scripts.analyze_risk_synthesis import digest
    view_path = tmp_path/'risk_synthesis/synthetic_views.csv'
    record = dict(complete=True, synthesis_options=synth.options, path=str(tmp_path), run='toy',
                  protocol={'num_global_iters':3,'local_epochs':2,'seed':42,'audit':{'audit_client_ids':[0]}},
                  sources={str(view_path):digest(view_path)}, synthesis_mechanism=verified)
    per_original, _, _ = summarize(record)
    assert len(per_original) == 21
    assert sum(r['visits'] for r in per_original) == 126
    assert sum(r['trained_views'] for r in per_original) == 252
    assert all(r['visits']==6 and r['reference_visits']==4 for r in per_original)
    path = tmp_path/'risk_synthesis/synthetic_views.csv'
    logged = list(csv.DictReader(path.open()))
    logged[-1]['loss_weight'] = '1.0'
    with path.open('w') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(logged[0]))
        writer.writeheader()
        writer.writerows(logged)
    with pytest.raises(ValueError,match='loss weight'):
        verify(tmp_path/'risk_synthesis')
