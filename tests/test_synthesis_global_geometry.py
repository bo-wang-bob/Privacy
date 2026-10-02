import json

import pytest
import torch

from privacy_defenses.global_geometry import local_moments, aggregate_moments, exchange
from privacy_defenses.risk_synthesis import FILTERED_DEFAULTS as DEFAULTS, LocalGeometry, validate_risk_synthesis


def geometry(rows, labels):
    return LocalGeometry(torch.tensor(rows, dtype=torch.float32), torch.tensor(labels), DEFAULTS, 'cpu')


def test_global_moments_equal_centralized_covariance_with_unequal_counts_and_missing_class():
    a = geometry([[0.,1.],[2.,3.],[10.,8.]], [0,0,1])
    b = geometry([[5.,2.],[7.,4.],[9.,6.]], [0,0,0])
    payloads = {3:local_moments(a,'cpu'), 8:local_moments(b,'cpu')}
    global_state = aggregate_moments(payloads,1,'cpu')
    for c in (0,1):
        rows = torch.cat([g.codes[g.labels==c] for g in (a,b)]).double()
        expected = (rows-rows.mean(0)).T@(rows-rows.mean(0))/len(rows)
        group = global_state['classes'][c]
        f = group['covariance_factor'].double()
        torch.testing.assert_close(group['mean'], rows.mean(0))
        torch.testing.assert_close(f@f.T, expected, rtol=2e-6, atol=2e-6)
        assert group['count']==len(rows)
        assert group['factor'].shape[1] <= 1
    assert global_state['classes'][0]['client_counts']=={3:2,8:3}
    assert global_state['classes'][1]['client_counts']=={3:1}
    assert global_state['classes'][1]['numerical_rank']==0


def test_between_client_mean_term_is_preserved_even_when_both_local_covariances_are_zero():
    a = geometry([[0.,1.]]*3,[0]*3)
    b = geometry([[10.,1.]]*9,[0]*9)
    result = aggregate_moments({0:local_moments(a,'cpu'),1:local_moments(b,'cpu')},5,'cpu')['classes'][0]
    torch.testing.assert_close(result['mean'],torch.tensor([7.5,1.],dtype=torch.float64))
    f = result['covariance_factor'].double()
    torch.testing.assert_close(f@f.T,torch.tensor([[18.75,0.],[0.,0.]],dtype=torch.float64))


def test_full_local_covariance_is_uploaded_before_generation_rank_truncation():
    torch.manual_seed(14)
    a = LocalGeometry(torch.randn(12,7),torch.zeros(12,dtype=torch.long),{**DEFAULTS,'class_rank':1},'cpu')
    assert a.classes[0]['factor'].shape[1]==1
    payload = local_moments(a,'cpu')
    assert payload['classes'][0]['covariance_factor'].shape[1]==7
    out = aggregate_moments({0:payload},1,'cpu')['classes'][0]
    assert out['covariance_factor'].shape[1]==7 and out['factor'].shape[1]==1
    x = a.codes.double()-a.codes.double().mean(0)
    f = out['covariance_factor'].double()
    torch.testing.assert_close(f@f.T,x.T@x/len(x),rtol=2e-6,atol=2e-6)


def test_all_directions_generation_recovers_central_covariance_and_samples_beyond_five():
    from privacy_defenses.risk_synthesis import DEFAULTS as DIRECT_DEFAULTS
    from privacy_defenses.synthesis_direct import draw_batch
    generator = torch.Generator().manual_seed(814)
    codes = torch.randn(24, 8, generator=generator)
    options = {**DIRECT_DEFAULTS, 'global_distribution': 'generate', 'center_source': 'global_class'}
    a = LocalGeometry(codes[:10], torch.zeros(10, dtype=torch.long), options, 'cpu')
    b = LocalGeometry(codes[10:], torch.zeros(14, dtype=torch.long), options, 'cpu')
    assert a.classes[0]['used_rank'] == 8
    uploads = {0: local_moments(a, 'cpu'), 1: local_moments(b, 'cpu')}
    a.global_distribution = aggregate_moments(uploads, options['class_rank'], 'cpu')
    group = a.global_distribution['classes'][0]
    assert group['used_rank'] == group['numerical_rank'] == 8
    assert group['retained_variance'] == 1.
    f = group['factor'].double()
    centered = codes.double() - codes.double().mean(0)
    torch.testing.assert_close(f@f.T, centered.T@centered/len(codes), rtol=2e-6, atol=2e-6)
    # Actual sampler must use the additional columns, not silently retain top five.
    sample = draw_batch(a, a.codes[:1], torch.tensor([0]), torch.tensor([.5]), options,
                        torch.Generator().manual_seed(19))
    epsilon = torch.randn(1, 8, generator=torch.Generator().manual_seed(19))
    center = .5*a.codes[:1] + .5*group['mean'].float()
    expected = center + epsilon@group['factor'].T
    torch.testing.assert_close(sample, expected)
    truncated = center + epsilon[:, :5]@group['factor'][:, :5].T
    assert not torch.allclose(sample, truncated)


def test_all_directions_zero_rank_keeps_empty_factor():
    from privacy_defenses.risk_synthesis import DEFAULTS as DIRECT_DEFAULTS
    a = LocalGeometry(torch.ones(1, 8), torch.tensor([0]), DIRECT_DEFAULTS, 'cpu')
    group = aggregate_moments({0: local_moments(a, 'cpu')}, 'all', 'cpu')['classes'][0]
    assert group['factor'].shape == (8, 0)
    assert group['used_rank'] == group['numerical_rank'] == 0


def test_every_client_receives_all_classes_and_global_noise_keeps_local_center(tmp_path):
    # Local class0 varies only in x; remote class0 adds a y direction.
    a = geometry([[0.,0.],[1.,0.],[2.,0.]], [0,0,0])
    b = geometry([[20.,-4.],[20.,0.],[20.,4.],[3.,8.]], [0,0,0,1])
    options = {**DEFAULTS,'global_distribution':'generate'}
    info = exchange({0:a,1:b},tmp_path,options,'cpu')
    assert a.global_distribution is b.global_distribution
    assert set(a.global_distribution['classes'])=={0,1}
    assert info['aggregation_count']==1
    for c in (0,1):
        receipt = json.loads((tmp_path/f'client_{c}_global_receipt.json').read_text())
        assert receipt['sha256']==info['sha256'] and receipt['available_classes']==[0,1]
        uploaded = torch.load(tmp_path/f'client_{c}_moment_upload.pt',weights_only=True)
        assert all(set(g)=={'count','mean','covariance_factor'} for g in uploaded['classes'].values())
    r = .8
    generator = torch.Generator().manual_seed(15)
    reference = torch.Generator().manual_seed(15)
    f = a.global_distribution['classes'][0]['factor']
    candidate = a.sample(a.codes[0],0,r,options,generator)
    center = (1-r)*a.codes[0]+r*a.codes[1:].mean(0)
    torch.testing.assert_close(candidate,center+.1*(f@torch.randn(f.shape[1],generator=reference)))
    assert candidate[1]!=0
    # Merely sharing statistics must leave local generation bitwise unchanged.
    local = a.sample(a.codes[0],0,r,DEFAULTS,torch.Generator().manual_seed(15))
    shared_only = a.sample(a.codes[0],0,r,{**options,'global_distribution':'share_only'},torch.Generator().manual_seed(15))
    torch.testing.assert_close(local,shared_only,atol=0,rtol=0)


def test_global_generation_requires_broadcast_and_valid_mode():
    a=geometry([[0.,0.],[1.,0.],[2.,0.]],[0,0,0])
    with pytest.raises(RuntimeError,match='received'):
        a.sample(a.codes[0],0,0,{**DEFAULTS,'global_distribution':'generate'},torch.Generator())
    with pytest.raises(ValueError,match='global_distribution'):
        validate_risk_synthesis(dict(model_type='clip_lora',aggregator='fedavg',sample_users=2,
            defense=dict(name='risk_synthesis',synthesis={**DEFAULTS,'global_distribution':'bad'})))


def test_global_center_equals_all_same_class_records_including_source_without_new_exchange(tmp_path):
    a = geometry([[1.,2.],[4.,1.],[7.,3.]],[0,0,0])
    b = geometry([[20.,5.],[22.,6.],[24.,7.],[26.,8.],[1000.,9000.]],[0,0,0,0,1])
    options = {**DEFAULTS,'global_distribution':'generate','center_source':'global_class'}
    info = exchange({0:a,1:b},tmp_path,options,'cpu')
    expected_center = torch.cat([a.codes,b.codes[:4]]).mean(0)
    center = a.global_class_center(0)
    torch.testing.assert_close(center,expected_center)
    for g in (a,b):
        for index in torch.where(g.labels==0)[0].tolist():
            torch.testing.assert_close(g.global_class_center(index),center,atol=0,rtol=0)
    assert not torch.allclose(center,a.codes[1:].mean(0))
    f = a.global_distribution['classes'][0]['factor']
    for r in (0.,.4,1.):
        candidate = a.sample(a.codes[0],0,r,options,torch.Generator().manual_seed(9))
        noise = .1*(f@torch.randn(f.shape[1],generator=torch.Generator().manual_seed(9)))
        torch.testing.assert_close(candidate,(1-r)*a.codes[0]+r*expected_center+noise)
    assert info['aggregation_count']==1
    # The source contributes exactly 1/N of the global same-class mean.
    a2=geometry([[999.,-123.],[4.,1.],[7.,3.]],[0,0,0])
    a2.global_distribution=aggregate_moments({0:local_moments(a2,'cpu'),1:local_moments(b,'cpu')},5,'cpu')
    torch.testing.assert_close(a2.global_class_center(0),expected_center+(a2.codes[0]-a.codes[0])/7)


@pytest.mark.parametrize('overrides', [
    {'global_distribution':'disabled'}, {'global_distribution':'share_only'},
    {'center_weighting':'previous_risk'}])
def test_global_center_rejects_missing_exchange_or_unavailable_risk_weighting(overrides):
    options={**DEFAULTS,'global_distribution':'generate','center_source':'global_class',**overrides}
    with pytest.raises(ValueError,match='Global class centers'):
        validate_risk_synthesis(dict(model_type='clip_lora',aggregator='fedavg',sample_users=2,
            defense=dict(name='risk_synthesis',synthesis=options)))


def test_default_single_view_training_and_broadcast_tamper_detection(tmp_path):
    from aggregator.aggregator_builder import build_aggregator
    from servers.serverbase import ServerBase
    from test_risk_synthesis import make_model, dataset
    from test_clip_peft_fedsgd import ATTACKS, _audit_config
    from scripts.verify_synthesis_global_geometry import verify
    from scripts.verify_synthesis_history import verify as verify_history
    server = ServerBase(device=torch.device('cpu'),dataset_name='toy',model=make_model('clip_adapter'),
        train_sets=[dataset(3,20),dataset(4,21)],test_sets=[dataset(12,30),dataset(13,31)],
        class_names=['a','b','c'],batch_size=4,eval_batch_size=8,learning_rate=.05,
        num_glob_iters=1,local_epochs=1,total_users=2,user_per_round=2,eval_interval=1,
        results_dir=str(tmp_path),aggregator=build_aggregator('fedavg',aggregation_weighting='sample_count'),
        audit_config=_audit_config(),projres_config={'enabled':True,'evaluation_interval':1},
        defense_config={'name':'risk_synthesis','synthesis':{**DEFAULTS,'global_distribution':'generate'}},
        method_config={'client_optimizer':'sgd','seed':42})
    results = server.train()
    assert server.auditor.errors=={} and {r['attack'] for r in results}==ATTACKS
    assert all(r['member_count']==r['nonmember_count']==9 for r in results)
    d=tmp_path/'risk_synthesis'
    assert not (d/'synthetic_views.csv').exists()
    assert verify_history(d)['global_geometry']['recipients']==[0,1]
    receipt=d/'client_1_global_receipt.json'
    original=receipt.read_text()
    altered=json.loads(original)
    altered['available_classes']=[0]
    receipt.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='receipt'):
        verify(d)
    receipt.write_text(original)
    # Any modification to the received artifact is caught before tensor replay.
    with (d/'global_distribution.pt').open('ab') as f:
        f.write(b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'):
        verify(d)
