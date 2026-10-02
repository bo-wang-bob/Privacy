"""Generated views must preserve one-step uploads and original batch identity."""
import csv
import json

import pytest
import torch
import yaml
from torch.nn import functional as F

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.risk_synthesis import DEFAULTS, validate_risk_synthesis
from scripts.analyze_risk_synthesis import analyze, read_synthesis_mechanism
from scripts import run_privacy_experiments as runner
from servers.serverbase import ServerBase
from users.user import UserBase
from test_clip_peft_fedsgd import ATTACKS, EXACT_BATCH_ATTACKS, _audit_config
from test_risk_synthesis import make_model, dataset, deterministic_cpu


@pytest.mark.parametrize('kind', ['clip_adapter', 'clip_lora'])
@pytest.mark.parametrize('views', [1, 2, 3])
def test_fedsgd_generated_gradient_original_batches_and_all_attacks(tmp_path, monkeypatch, kind, views):
    audit = _audit_config()
    audit.update(audit_batch_size=8, grad_sample_chunk_size=4)
    model = make_model(kind)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    server = ServerBase(device=torch.device('cpu'), dataset_name='toy', model=model,
        train_sets=[dataset(3, 20), dataset(4, 21)], test_sets=[dataset(24, 30), dataset(24, 31)],
        class_names=['a', 'b', 'c'], batch_size=4, eval_batch_size=8, learning_rate=.05,
        num_glob_iters=4, local_epochs=1, total_users=2, user_per_round=2, eval_interval=1,
        results_dir=str(tmp_path), aggregator=build_aggregator('fedsgd', aggregation_weighting='uniform'),
        audit_config=audit, projres_config={'enabled': True, 'evaluation_interval': 1},
        defense_config={'name': 'risk_synthesis', 'synthesis': {**DEFAULTS, 'views_per_record': views,
            'noise_scale': .5, 'mixing_mode': 'risk', 'global_distribution': 'generate',
            'center_source': 'global_class'}}, method_config={'client_optimizer': 'sgd', 'seed': 42})
    synth = server.defense.synthesis
    generated, captures = {}, []
    transform = synth.transform_views
    def retain_views(model, user, images, labels, indices, *args, **kwargs):
        output = transform(model, user, images, labels, indices, *args, **kwargs)
        generated[user.id] = (torch.cat(output), labels.repeat(views))
        return output
    monkeypatch.setattr(synth, 'transform_views', retain_views)
    capture = UserBase.capture_protocol_gradients
    def check_gradient(user, model):
        capture(user, model)
        tokens, labels = generated[user.id]
        parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        # Independent equivalent objective: a single CE over concatenated views.
        expected = torch.autograd.grad(F.cross_entropy(model.forward_tokens(tokens), labels),
                                       [p for _, p in parameters], allow_unused=True)
        for (name, parameter), gradient in zip(parameters, expected):
            torch.testing.assert_close(user.last_update_gradients[name],
                torch.zeros_like(parameter) if gradient is None else gradient, atol=2e-6, rtol=2e-4)
        captures.append((user.id, user.last_train_indices.tolist(), user.last_update_sample_count))
    monkeypatch.setattr(UserBase, 'capture_protocol_gradients', check_gradient)
    result = server.train()
    assert not server.auditor.errors and {r['attack'] for r in result} == ATTACKS
    assert len(captures) == 8
    target = [entry for entry in captures if entry[0] == 0]
    assert [entry[2] for entry in target] == [4, 4, 1, 4]
    assert sorted(i for entry in target[:3] for i in entry[1]) == list(range(9))
    assert synth.counts == dict(visits=29, requested=29, accepted=29, fallback=0)
    for row in result:
        assert row['member_count'] == (4 if row['attack'] in EXACT_BATCH_ATTACKS else 9)
        assert row['nonmember_count'] == (40 if row['attack'] in EXACT_BATCH_ATTACKS else 9)
    selections = torch.load(tmp_path/'privacy_audit/exact_batch_candidate_selection.pt', weights_only=True)
    assert len(selections['rounds']) == len(target)
    for selection, (_, indices, count) in zip(selections['rounds'], target):
        assert selection['membership_definition'] == 'current_round_original_source_batch'
        assert selection['member_local_indices'].tolist() == indices
        assert selection['member_count'] == count
        assert selection['nonmember_count'] == count * 10
    metadata = next(r['metadata'] for r in result if r['attack'] == 'projres')
    assert metadata['batch_rank_bound'] is None and metadata['paper_fedsgd_exact'] is False
    assert metadata['attacked_parameter_perturbed'] is False
    assert metadata['update_source'] == 'uploaded_client_gradient'
    assert metadata['views_per_record'] == views
    for user in server.ctx.users:
        assert user.last_gradient_capture_count == 1
        message = server.ctx.protocol_messages[user.id]
        assert message['kind'] == 'gradient'
        for name, value in message['tensors'].items():
            torch.testing.assert_close(value, user.last_update_gradients[name])
    base = server.ctx.get_base_model_state()
    for name in server.ctx.trainable_param_names:
        expected = base[name] - .05 * sum(u.last_update_gradients[name] for u in server.ctx.users) / 2
        torch.testing.assert_close(dict(model.named_parameters())[name], expected)
    for name, value in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
    rows = list(csv.DictReader((tmp_path/'risk_synthesis/synthetic_exposure.csv').open()))
    assert all(int(r['source_round']) == -1 and float(r['risk']) == 0 for r in rows if r['round'] == '1')
    assert any(float(r['risk']) > 0 for r in rows if r['round'] != '1')
    summary = json.loads((tmp_path/'risk_synthesis/synthesis_summary.json').read_text())
    assert summary['membership'] == 'original_source_batch' and summary['federated_method'] == 'fedsgd'
    assert summary['statistics_storage']['status'] == 'cleaned'
    evidence = read_synthesis_mechanism(tmp_path/'risk_synthesis', summary, complete=True)
    assert evidence['direct_evidence']['trained_views_verified'] == 29 * views


@pytest.mark.parametrize('script', ['run_global_synthesis_validation', 'run_synthesis_center_ablation',
                                    'run_synthesis_center_source_ablation'])
def test_fedsgd_wrappers_use_method_defaults_and_matching_confirmation_data(script, tmp_path):
    import importlib
    module = importlib.import_module('scripts.' + script)
    arguments = ['--methods', 'fedsgd', '--dry-run', '--results-root', str(tmp_path)]
    if hasattr(module, 'build_study'):
        _, groups, _ = module.build_study(arguments)
        tasks = [t for group in groups for t in group['tasks']]
    else:
        tasks, skipped = runner.build_tasks(runner.load_yaml('configs/experiment_catalog.yaml'),
                                           runner.parse_args(module.build_arguments(arguments)))
        assert not skipped
    for task in tasks:
        config = task.config
        assert config['aggregator'] == 'fedsgd' and config['num_global_iters'] == 1000
        assert config['local_epochs'] == 1 and config['aggregation_weighting'] == 'uniform'
        assert config['audit']['membership_protocol'] == 'exact_batch'
        assert config['projres']['max_candidates'] == 32
        assert config['projres']['min_nonmembers'] == config['projres']['max_nonmembers'] == 320
        assert config['confirmation_split_sha256']
        assert config['defense']['synthesis']['views_per_record'] == 2
    assert not list(tmp_path.iterdir())


def test_fedsgd_rejects_historical_filtered_generation():
    with pytest.raises(ValueError, match='requires candidate_selection=direct'):
        validate_risk_synthesis(dict(model_type='clip_lora', aggregator='fedsgd', sample_users=2,
                                    defense=dict(name='risk_synthesis', synthesis={'replacement_fraction': .25})))


def test_plain_fedsgd_retains_same_shuffled_batches_and_original_ids():
    model = make_model('clip_adapter')
    data = dataset(3, 20)
    user = UserBase(torch.device('cpu'), 0, 'toy', data, data, model, 4, .01, 1,
                    federated_method='fedsgd', method_config={'seed': 42})
    expected = iter(user.trainloader)
    for step in range(7):
        try:
            images, labels = next(expected)
        except StopIteration:
            expected = iter(user.trainloader)
            images, labels = next(expected)
        user.begin_local_update()
        actual_images, actual_labels = user.next_train_batch()
        torch.testing.assert_close(actual_images, images, rtol=0, atol=0)
        torch.testing.assert_close(actual_labels, labels, rtol=0, atol=0)
        ids = user.last_train_indices.tolist()
        torch.testing.assert_close(actual_images, torch.stack([data[i][0] for i in ids]))


def test_fedsgd_analysis_pairs_original_batch_identities(tmp_path):
    directories = []
    for defense in ('none', 'risk_synthesis'):
        torch.manual_seed(23)
        directory = tmp_path/defense
        directories.append(directory)
        audit = _audit_config()
        audit.update(audit_batch_size=8, grad_sample_chunk_size=4, exact_batch_nonmember_to_member_ratio=2)
        config = dict(aggregator='fedsgd', num_global_iters=3, local_epochs=1, seed=42, audit=audit,
                      defense={'name': defense, 'synthesis': {**DEFAULTS, 'views_per_record': 2,
                          'global_distribution': 'generate', 'center_source': 'global_class', 'noise_scale': .5}})
        server = ServerBase(device=torch.device('cpu'), dataset_name='toy', model=make_model('clip_adapter'),
            train_sets=[dataset(3, 20), dataset(4, 21)], test_sets=[dataset(12, 30), dataset(12, 31)],
            class_names=['a', 'b', 'c'], batch_size=4, eval_batch_size=8, learning_rate=.05,
            num_glob_iters=3, local_epochs=1, total_users=2, user_per_round=2, eval_interval=1,
            results_dir=str(directory), aggregator=build_aggregator('fedsgd', aggregation_weighting='uniform'),
            audit_config=audit, projres_config={'enabled': True, 'evaluation_interval': 1},
            defense_config=config['defense'], method_config={'client_optimizer': 'sgd', 'seed': 42})
        server.train()
        (directory/'run_config.yaml').write_text(yaml.safe_dump(config))
    analyze(directories, tmp_path/'analysis')
    report = json.loads((tmp_path/'analysis/verified_results.json').read_text())
    assert len(report['matched_comparisons']) == 1
    assert all(run['complete'] and len(run['attacks']) == 11 for run in report['runs'])
