"""Risk-score ablation: signed evidence, actual training and matched launch plans."""
from collections import defaultdict
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from privacy_defenses.controller import DefenseController
from privacy_defenses.risk_synthesis import DEFAULTS, RiskSynthesis, synthesis_options, validate_risk_synthesis
from privacy_defenses.www_dp import risk_regularization_weights
from scripts import run_privacy_experiments as runner
from scripts import run_synthesis_risk_score_ablation as study
from scripts.verify_synthesis_direct import verify_risk_scores


@pytest.mark.parametrize('score', ['signed_loss_gap', 'absolute_loss_gap'])
def test_signed_negative_outlier_and_absolute_ties(score):
    gaps = torch.tensor([-9., -2., 2., 1.], requires_grad=True)
    synth = RiskSynthesis({'synthesis': {**DEFAULTS, 'risk_score': score}}, 43)
    transformed = synth.ranking_scores(gaps)
    assert not transformed.requires_grad
    weights, order, _ = risk_regularization_weights(transformed, 1., expected_batch_size=4)
    expected_order = [0, 1, 3, 2] if score == 'signed_loss_gap' else [3, 1, 2, 0]
    assert order.tolist() == expected_order  # Absolute +/-2 ties preserve source batch order.
    assert weights[0] == (.125 if score == 'signed_loss_gap' else .875)
    assert sorted(weights.tolist()) == [.125, .375, .625, .875]
    assert gaps.tolist() == [-9., -2., 2., 1.]  # Raw signed diagnostics are never overwritten.
    old = RiskSynthesis({'synthesis': DEFAULTS}, 43)
    torch.testing.assert_close(old.ranking_scores(gaps), gaps.detach(), rtol=0, atol=0)
    assert 'risk_score' not in old.options


@pytest.mark.parametrize('score', [None, True, 'abs', 'unknown', 1])
def test_invalid_score_rejected(score):
    config = dict(model_type='clip_lora', aggregator='fedsgd', sample_users=2,
                  defense=dict(name='risk_synthesis', synthesis={**DEFAULTS, 'risk_score': score}))
    with pytest.raises(ValueError, match='risk_score'):
        validate_risk_synthesis(config)


def test_score_option_rejected_for_historical_filtered_generation():
    with pytest.raises(ValueError, match='risk_score'):
        synthesis_options({'candidate_selection': 'first_semantic', 'risk_score': 'absolute_loss_gap'})


@pytest.mark.parametrize('score', ['signed_loss_gap', 'absolute_loss_gap'])
@pytest.mark.parametrize('references', [False, True])
def test_controller_ranks_selected_score_but_passes_original_gaps(monkeypatch, score, references):
    class Model(torch.nn.Linear):
        def forward_tokens(self, x):
            return self(x)
    model = Model(3, 2)
    images, labels, indices = torch.randn(4, 3), torch.arange(4) % 2, torch.arange(4)
    gaps = torch.tensor([-9., -2., 2., 1.])
    ranking = SimpleNamespace(scores=gaps)
    monkeypatch.setattr('privacy_defenses.controller.rank_loss_differences', lambda *a: ranking)
    controller = DefenseController.__new__(DefenseController)
    controller.device = torch.device('cpu')
    controller._www_pending_states = {0: ({}, {})} if references else {}
    controller.steps = defaultdict(int)
    controller._record = lambda *args: None
    controller._record_www_ranking = lambda user, r, *args: torch.testing.assert_close(r.scores, gaps)
    controller.synthesis = RiskSynthesis({'synthesis': {**DEFAULTS, 'risk_score': score,
                                        'risk_tail_fraction': 1}}, 43)
    actual = []
    def transform(model, user, x, y, ids, risk, round_index, step, source_round, *, raw_scores):
        assert source_round == (0 if references else -1)
        assert (raw_scores is gaps) if references else (raw_scores is None)
        actual.append(risk)
        return (images+.1, images-.1)
    controller.synthesis.transform_views = transform
    controller.synthesis.record_optimized_batch = lambda _: None
    user = SimpleNamespace(id=0, www_source_round=0, www_aggregation_weight=.5,
                           get_parameters=lambda: {}, iter_www_local_batches=lambda: iter([(images, labels, indices)]))
    controller._synthesis_training(user, model, torch.optim.SGD(model.parameters(), lr=.01), 1, False)
    expected = ([.125, .375, .875, .625] if score == 'signed_loss_gap' else [.875, .375, .625, .125])
    assert actual[0].tolist() == (expected if references else [0.] * 4)
    assert controller.steps[0] == 1


@pytest.mark.parametrize('score', ['signed_loss_gap', 'absolute_loss_gap'])
def test_verifier_checks_score_transform_and_record_assignment(score):
    gaps = [-9., -2., 2., 1.]
    risks = [.125, .375, .875, .625] if score == 'signed_loss_gap' else [.875, .375, .625, .125]
    rows = [dict(source_round='0', signed_loss_gap=str(d),
                 risk_ranking_score=str(abs(d) if score == 'absolute_loss_gap' else d), risk=str(r))
            for d, r in zip(gaps, risks)]
    options = dict(risk_score=score, risk_tail_fraction=1)
    verify_risk_scores(rows, options)
    swapped = copy.deepcopy(rows)
    swapped[0]['risk'], swapped[1]['risk'] = swapped[1]['risk'], swapped[0]['risk']
    with pytest.raises(ValueError, match='recorded score ranks'):
        verify_risk_scores(swapped, options)  # Same multiset alone is insufficient.
    rows[0]['risk_ranking_score'] = '999'
    with pytest.raises(ValueError, match='score transform'):
        verify_risk_scores(rows, options)
    cold = [dict(source_round='-1', signed_loss_gap='', risk_ranking_score='', risk='0')]
    verify_risk_scores(cold, options)
    cold[0]['signed_loss_gap'] = '0'
    with pytest.raises(ValueError, match='must be empty'):
        verify_risk_scores(cold, options)


@pytest.mark.parametrize('smoke', [False, True])
def test_study_configs_match_and_preserve_requested_protocol(tmp_path, smoke):
    root, groups, dry = study.build_study(['--results-root', str(tmp_path), '--gpus', '1', '--dry-run',
                                         *(['--smoke'] if smoke else [])])
    assert dry and not root.exists() and [g['variant'] for g in groups] == list(study.VARIANTS)
    configs = []
    for group in groups:
        assert len(group['tasks']) == 1
        task = group['tasks'][0]
        assert task.model == 'clip_adapter' and task.dataset == 'cifar100'
        assert task.seed == 43 and task.target_client_id == 0
        assert len(task.attacks) == (0 if smoke else 11)
        config = copy.deepcopy(task.config)
        assert config['num_global_iters'] == (5 if smoke else 1000)
        assert config['aggregation_weighting'] == 'uniform' and config['aggregator'] == 'fedsgd'
        assert config['batch_size'] == 32 and config['fpl_shots'] == 100
        assert config['sample_users'] == config['total_users'] == 10
        opts = config['defense']['synthesis']
        assert (opts.pop('risk_score'), opts.pop('mode')) == study.VARIANTS[group['variant']]
        assert opts['noise_scale'] == .2 and opts['risk_tail_fraction'] == 1 and opts['views_per_record'] == 2
        assert opts['statistics_retention'] == 'cleanup_on_exit'
        config.pop('results_dir')
        configs.append(config)
    assert configs[0] == configs[1] == configs[2]


@pytest.mark.parametrize('arguments', [
    ['--variants', 'signed,signed'], ['--variants', 'bad'], ['--models', 'clip_mlp'],
    ['--methods', 'fedavg'], ['--gpus', '0,1'], ['--defenses', 'none,risk_synthesis'],
    ['--set', 'defense.synthesis.noise_scale=0.3'], ['--set', 'defense.synthesis.risk_tail_fraction=0.8'],
    ['--set', 'defense.synthesis.risk_score=absolute_loss_gap'],
    ['--set', 'defense.synthesis.mode=shuffled_risk'],
    ['--set', 'defense.synthesis.mixing_mode=class_center']])
def test_invalid_study_never_launches_or_writes(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(runner, 'main', lambda *args: pytest.fail('Invalid study launched training.'))
    with pytest.raises(ValueError):
        study.main(['--results-root', str(tmp_path), *arguments])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('exit_code', [0, 1])
def test_study_routes_through_runner_and_stops_on_failure(tmp_path, monkeypatch, exit_code):
    calls = []
    monkeypatch.setattr(runner, 'main', lambda args: calls.append(args) or exit_code)
    assert study.main(['--results-root', str(tmp_path), '--dry-run']) == 0
    assert not calls and not list(tmp_path.iterdir())
    assert study.main(['--results-root', str(tmp_path)]) == exit_code
    assert len(calls) == (3 if exit_code == 0 else 1)
    plan = json.loads(next(tmp_path.glob('*/study_plan.json')).read_text())
    assert plan['noise_scale'] == .2 and [g['variant'] for g in plan['groups']] == list(study.VARIANTS)
