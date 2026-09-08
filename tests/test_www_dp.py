from __future__ import annotations

import copy
import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import betainc
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import TensorDataset

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.controller import DefenseController
from privacy_defenses.www import WWWRanking, infer_other_clients_state, rank_loss_differences
from privacy_defenses.www_diagnostics import WWWGradientRecorder
from privacy_defenses.www_dp import (
    DEFAULTS, ino_weights, validate_www, weighted_clipped_sum,
    risk_regularization_weights, risk_controlled_losses,
)
from scripts.run_privacy_experiments import build_tasks, load_yaml, parse_args
from servers.serverbase import ServerBase
from users.user import UserBase
from test_cofedmid import tiny_model, toy_dataset


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_exact_loss_difference_sign_order_and_model_restoration():
    model = torch.nn.Linear(2, 2, bias=False)
    restore = copy.deepcopy(model.state_dict())
    own = {"weight": torch.tensor([[2., 0.], [0., 1.]])}
    other = {"weight": torch.tensor([[0., 1.], [2., 0.]])}
    global_state = {"weight": 0.3 * own["weight"] + 0.7 * other["weight"]}
    recovered = infer_other_clients_state(global_state, own, 0.3)
    torch.testing.assert_close(recovered["weight"], other["weight"])
    x, y = torch.tensor([[1., 0.], [0., 1.], [2., 1.]]), torch.tensor([0, 1, 0])
    ranking = rank_loss_differences(model, [(x, y)], own, other, restore,
                                    torch.device("cpu"), torch.tensor([5, 7, 1]))
    expected = (torch.nn.functional.cross_entropy(x @ other["weight"].T, y, reduction="none")
                - torch.nn.functional.cross_entropy(x @ own["weight"].T, y, reduction="none"))
    torch.testing.assert_close(ranking.scores, expected)
    assert torch.equal(ranking.ranked_positions, expected.argsort(stable=True))
    torch.testing.assert_close(model.weight, restore["weight"])
    assert model.training


@pytest.mark.parametrize("count", [1, 2, 5, 16, 32])
def test_risk_loss_weights_increase_only_in_upper_tail_and_preserve_ties(count):
    scores = torch.zeros(count, requires_grad=True)
    weights, order, tail = risk_regularization_weights(scores, expected_batch_size=32)
    m = math.ceil(.8 * count)
    assert order.tolist() == list(range(count))
    assert tail.sum() == m and not weights.requires_grad
    torch.testing.assert_close(weights[:count-m], torch.zeros(count-m, dtype=torch.float64))
    torch.testing.assert_close(weights[count-m:], (torch.arange(m, dtype=torch.float64)+.5)/m)
    if count == 32:
        assert (weights == 0).sum() == 6


@pytest.mark.parametrize("strength", [0., .3, 1., 3.])
def test_prediction_loss_matches_formula_and_detaches_teacher_and_risk(strength):
    logits = torch.tensor([[2., 0.], [0., 1.], [1., -1.]], requires_grad=True)
    labels = torch.tensor([0, 1, 0])
    weights = torch.tensor([0., .5, 1.], requires_grad=True)
    teacher = torch.tensor([.1, .99, .1], requires_grad=True)
    terms = risk_controlled_losses(logits, labels, weights, teacher, strength)
    actual = torch.autograd.grad(terms["total_loss"].sum(), logits)[0]
    p = logits.detach().softmax(-1)
    py = p.gather(1, labels[:, None]).flatten()
    factor = 1 - strength * weights.detach() * py * (py-teacher.detach()).sign()
    expected = (p-torch.nn.functional.one_hot(labels, 2)) * factor[:, None]
    torch.testing.assert_close(actual, expected)
    assert weights.grad is None and teacher.grad is None
    assert not terms["reference_probability"].requires_grad
    assert terms["regularization_loss"][0] == 0
    if strength == 3:
        assert factor[-1] < 0  # Strong regularization can reverse CE; norms remain positive.


@pytest.mark.parametrize("backend", ["loop", "batched"])
@pytest.mark.parametrize("strength", [0., 1e-8, 3.])
@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_diagnostic_norms_match_actual_dropout_graph_and_do_not_change_training(backend, strength, device):
    from privacy_defenses.www_dp import WWWPrivacy
    if device.startswith("cuda") and torch.cuda.device_count() < 2:
        pytest.skip("GPU 1 is unavailable")
    torch.manual_seed(123)
    model = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.Dropout(.3), torch.nn.Linear(5, 2)).to(device)
    x, y = (torch.randn(7, 2)*30).to(device), (torch.arange(7) % 2).to(device)
    weights, _, _ = risk_regularization_weights(torch.arange(7.), expected_batch_size=7)
    teacher = torch.linspace(.05, .95, 7, device=device)
    user = SimpleNamespace(id=0)
    outputs = []
    rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else torch.get_rng_state()
    def restore_rng():
        if device.startswith("cuda"):
            torch.cuda.set_rng_state(rng, device)
        else:
            torch.set_rng_state(rng)
    for enabled in [False, True]:
        student = copy.deepcopy(model)
        trainer = WWWPrivacy({"name": "www", "max_grad_norm": 1e-10,
            "www_regularization_weight": strength, "www_record_diagnostics": enabled,
            "grad_sample_backend": backend, "microbatch_size": 3}, 1, torch.device(device), 42)
        trainer.planned_steps = {0: 1}
        optimizer = torch.optim.SGD(student.parameters(), lr=.01)
        restore_rng()
        result = trainer.step(user, student, optimizer, x, y, weights, 0, reference_probability=teacher)
        outputs.append([p.detach().clone() for p in student.parameters()])
    reference = copy.deepcopy(model)
    restore_rng()
    logits = reference(x)
    ce = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    penalty = strength * weights.to(ce) * (logits.softmax(-1).gather(1,y[:,None]).flatten()-teacher).abs()
    total = ce + penalty
    for field, losses in [("raw_grad_norm", ce), ("regularizer_grad_norm", penalty), ("total_grad_norm", total)]:
        expected_norms = []
        for loss in losses:
            grads = torch.autograd.grad(loss, list(reference.parameters()), retain_graph=True)
            expected_norms.append(sum(g.square().sum() for g in grads).sqrt())
        torch.testing.assert_close(result[field], torch.stack(expected_norms), rtol=2e-5, atol=2e-5)
        if field == "regularizer_grad_norm" and strength == 1e-8:
            torch.testing.assert_close(result[field], torch.stack(expected_norms), rtol=2e-5, atol=1e-12)
    total.mean().backward()
    for before, p, without, with_diagnostics in zip(model.parameters(), reference.parameters(), *outputs):
        torch.testing.assert_close(without, with_diagnostics, rtol=0, atol=0)
        torch.testing.assert_close(with_diagnostics, before-.01*p.grad, rtol=2e-5, atol=2e-6)
    assert result["raw_grad_norm"].max() > 8


def test_disabled_diagnostics_never_call_per_sample_gradients(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("No per-sample gradients are needed to train WWW risk loss")
    monkeypatch.setattr("privacy_defenses.www_dp.gradients_from_losses", unexpected)
    user, _ = make_user({"www_record_diagnostics": False, "www_regularization_weight": 0})
    user.train(round_index=0)
    assert user.last_gradient_capture_count == 1


@pytest.mark.parametrize("count", [1, 2, 5, 10, 16, 32])
def test_highest_eighty_percent_and_linear_tif_exact_integrals(count):
    scores = torch.arange(count, dtype=torch.float64).flip(0)
    weights, positions, tail = ino_weights(scores, expected_batch_size=count)
    m = math.ceil(count * 0.8)
    assert int(tail.sum()) == m
    assert torch.equal(scores[positions], scores.sort().values)
    assert torch.equal(tail, torch.arange(count) < m)
    expected = 1 - (torch.arange(m, dtype=torch.float64) + 0.5) / m
    torch.testing.assert_close(weights[positions[-m:]], expected)
    assert torch.equal(weights[~tail], torch.ones(count - m, dtype=torch.float64))
    _, ties, tied_tail = ino_weights(torch.zeros(count), expected_batch_size=count)
    assert torch.equal(ties, torch.arange(count))
    assert tied_tail[-m:].all()


def test_beta_integral_matches_independent_quadrature():
    weights, positions, _ = ino_weights(torch.arange(19.), 0.4, 2.3, 4.1, expected_batch_size=19)
    m = math.ceil(19 * 0.4)
    expected = torch.tensor([m * quad(lambda u: betainc(2.3, 4.1, 1-u),
                                    j/m, (j+1)/m)[0] for j in range(m)], dtype=torch.float64)
    torch.testing.assert_close(weights[positions[-m:]], expected, atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize("actual", [0, 1, 2, 3, 4, 8, 12, 13, 14, 16, 25])
def test_fixed_tail_uses_expected_batch_and_right_aligns_small_draws(actual):
    weights, order, tail = ino_weights(torch.arange(actual).float(), expected_batch_size=16,
                                      tail_basis="expected_batch")
    m = 13
    expected = 1 - (torch.arange(m, dtype=torch.float64) + 0.5) / m
    assert int(tail.sum()) == min(actual, m)
    assert weights.numel() == actual
    if actual:
        torch.testing.assert_close(weights[order][-min(actual, m):], expected[-min(actual, m):])
        assert (weights[~tail] == 1).all()


@pytest.mark.parametrize("alpha,beta", [(1., 1.), (2.3, 4.1), (4., 0.5)])
def test_add_remove_sensitivity_includes_all_rank_weight_changes(alpha, beta):
    # The sum of absolute changed coefficients is the maximum possible vector
    # change for jointly C-clipped records, attained by aligned unit vectors.
    for actual in (0, 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33):
        scores = torch.arange(actual).double()
        old, _, _ = ino_weights(scores, beta_alpha=alpha, beta_beta=beta,
                                expected_batch_size=16, tail_basis="expected_batch")
        for position in range(actual + 1):
            neighbor = torch.cat((scores, torch.tensor([position - 0.5])))
            new, _, _ = ino_weights(neighbor, beta_alpha=alpha, beta_beta=beta,
                                    expected_batch_size=16, tail_basis="expected_batch")
            total_change = new[-1] + (new[:-1] - old).abs().sum()
            assert total_change <= 1 + 1e-12


def test_joint_clipping_then_weighting_matches_manual_gradients():
    torch.manual_seed(5)
    model = torch.nn.Linear(2, 2)
    parameters = list(model.parameters())
    x, y = torch.tensor([[0.01, 0.], [20., -15.], [0.1, 0.2]]), torch.tensor([0, 1, 1])
    weights = torch.tensor([0.2, 0.7, 1.])
    maximum = 0.8
    expected = [torch.zeros_like(p) for p in parameters]
    for inputs, target, weight in zip(x, y, weights):
        grads = torch.autograd.grad(torch.nn.functional.cross_entropy(model(inputs[None]), target[None]), parameters)
        norm = torch.cat([g.flatten() for g in grads]).norm()
        for total, g in zip(expected, grads):
            total.add_(g * min(1., maximum / norm) * weight)
    actual = weighted_clipped_sum(model, x, y, parameters, maximum, weights)
    for result, manual in zip(actual, expected):
        torch.testing.assert_close(result, manual)


@pytest.mark.parametrize("backend", ["loop", "batched"])
@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_low_risk_large_gradient_is_unclipped_and_norms_match_independent_gradients(backend, device):
    if device.startswith("cuda") and torch.cuda.device_count() < 2:
        pytest.skip("GPU 1 is unavailable")
    model = torch.nn.Linear(2, 2).to(device)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    x = torch.tensor([[100., 0], [0, 50], [3, 4], [.02, .01], [10, -5]], device=device)
    y = torch.tensor([0, 1, 0, 1, 0], device=device)
    weights, _, tail = ino_weights(torch.arange(5.), expected_batch_size=32)
    assert tail.tolist() == [False, True, True, True, True]
    parameters = list(model.parameters())
    sums = [torch.zeros_like(p) for p in parameters]
    norms = []
    for i in range(5):
        gradients = torch.autograd.grad(torch.nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1]), parameters)
        norm = torch.cat([g.flatten() for g in gradients]).double().norm()
        norms.append(float(norm))
        factor = min(1., 8 / float(norm)) if i else 1.
        for total, gradient in zip(sums, gradients):
            total.add_(gradient * (factor * float(weights[i])))
    actual, diagnostics = weighted_clipped_sum(
        model, x, y, parameters, 8, weights, backend=backend, microbatch_size=2,
        clip_mask=tail, return_diagnostics=True,
    )
    for observed, expected in zip(actual, sums):
        torch.testing.assert_close(observed, expected)
    raw = diagnostics["raw_grad_norm"].cpu().numpy()
    np.testing.assert_allclose(raw, norms, rtol=1e-6)
    expected_clipped = np.minimum(norms, 8)
    expected_clipped[0] = norms[0]
    np.testing.assert_allclose(diagnostics["clipped_grad_norm"].cpu(), expected_clipped, rtol=1e-6)
    np.testing.assert_allclose(diagnostics["weighted_grad_norm"].cpu(), expected_clipped * weights.numpy(), rtol=1e-6)
    assert diagnostics["clip_factor"][0] == 1 and raw[0] > 8
    assert diagnostics["clip_factor"][3] == 1
    assert diagnostics["weighted_grad_norm"][3] < diagnostics["raw_grad_norm"][3]


def test_gradient_diagnostics_are_aligned_flushed_and_correlations_are_reproducible(tmp_path):
    scores = torch.tensor([3., 1., 1., 4., 2.])
    norms = torch.tensor([2., 9., 6., 7., 3.])
    weights, order, tail = risk_regularization_weights(scores, expected_batch_size=32)
    labels = torch.tensor([1, 0, 1, 0, 1])
    diagnostics = risk_controlled_losses(torch.randn(5, 2), labels, weights, torch.full((5,), .2), 3.)
    factors = diagnostics["ce_gradient_factor"]
    diagnostics.update(raw_grad_norm=norms, regularizer_grad_norm=(factors-1).abs()*norms,
                       total_grad_norm=factors.abs()*norms, additional_loss=torch.zeros(5))
    ranking = WWWRanking(torch.ones(5), scores + 1, scores, order,
                         torch.tensor([1, 0, 1, 0, 1]), torch.tensor([8, 2, 10, 7, 6]))
    recorder = WWWGradientRecorder(tmp_path)
    recorder.record(user=SimpleNamespace(id=3, batch_size=32), ranking=ranking,
                    diagnostics=diagnostics, weights=weights, tail=tail,
                    round_index=4, client_step=5, source_round=3, has_reference=True, regularization_weight=3.)
    # Available on disk before close, without holding all visits in memory.
    with (tmp_path / "www_diagnostics/sample_gradients.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["local_sample_index"]) for row in rows] == [8, 2, 10, 7, 6]
    assert [float(row["risk_score"]) for row in rows] == scores.tolist()
    assert [float(row["raw_grad_norm"]) for row in rows] == norms.tolist()
    assert rows[1]["group"] == "low_risk" and rows[1]["risk_weight"] == "0.0"
    assert "clip_factor" not in rows[1]
    with (tmp_path / "www_diagnostics/batch_summary.csv").open() as file:
        batches = list(csv.DictReader(file))
    overall = next(row for row in batches if row["group"] == "all")
    for name in ("raw_grad_norm", "regularizer_grad_norm", "total_grad_norm"):
        observed = diagnostics[name].numpy()
        assert float(overall[f"risk_pearson_{name}"]) == pytest.approx(pearsonr(scores.double().numpy(), observed.astype(float))[0])
        assert float(overall[f"risk_spearman_{name}"]) == pytest.approx(spearmanr(scores.numpy(), observed)[0])
        assert float(overall[f"{name}_p99"]) == pytest.approx(np.quantile(observed.astype(float), .99))
    assert next(row for row in batches if row["group"] == "low_risk")["risk_pearson_raw_grad_norm"] == ""
    recorder.close("failed")
    summary = json.loads((tmp_path / "www_diagnostics/summary.json").read_text())
    assert summary["status"] == "failed" and summary["sample_rows"] == 5
    previous = (tmp_path / "www_diagnostics/sample_gradients.csv").read_bytes()
    with pytest.raises(FileExistsError):
        WWWGradientRecorder(tmp_path)
    assert (tmp_path / "www_diagnostics/sample_gradients.csv").read_bytes() == previous


def test_recording_toggle_preserves_updates_and_disabled_mode_creates_no_artifacts(tmp_path):
    updates = []
    for enabled in (True, False):
        torch.manual_seed(41)
        user, controller = make_user({"www_record_diagnostics": enabled}, samples=10)
        directory = tmp_path / str(enabled)
        controller.start_www_gradient_diagnostics(directory)
        for round_index in range(2):
            initial = user.get_parameters()
            if round_index:
                controller.prepare_client_training(user, initial, .5, source_round=0)
            user.train(round_index=round_index)
        updates.append(user.last_update_gradients)
        controller.finish_www_gradient_diagnostics("completed")
    for name in updates[0]:
        torch.testing.assert_close(updates[0][name], updates[1][name], rtol=0, atol=0)
    assert not (tmp_path / "False/www_diagnostics").exists()
    with (tmp_path / "True/www_diagnostics/batch_summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert all(row["risk_pearson_raw_grad_norm"] == "" for row in rows)


def test_training_failure_preserves_completed_batch_diagnostics(tmp_path, monkeypatch):
    from test_runtime_optimization import make_server

    server = make_server(tmp_path, monkeypatch, "bert_adapter", defense="www")
    train = server.defense._www_training
    def fail_after_one_batch(*args, **kwargs):
        train(*args, **kwargs)
        raise RuntimeError("forced failure after recorded batch")
    monkeypatch.setattr(server.defense, "_www_training", fail_after_one_batch)
    with pytest.raises(RuntimeError, match="forced failure"):
        server.train()
    directory = tmp_path / "www_diagnostics"
    with (directory / "sample_gradients.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["sample_rows"] == 2
    assert server.defense.www_gradient_recorder.closed


def test_ordered_sum_replace_one_sensitivity_with_changed_scores_and_gradients():
    generator = torch.Generator().manual_seed(52)
    for n in (2, 5, 16, 32):
        for _ in range(20):
            scores = torch.randn(n, generator=generator)
            vectors = torch.randn(n, 7, generator=generator)
            vectors /= vectors.norm(dim=1, keepdim=True).clamp_min(1)
            neighbor_scores, neighbor_vectors = scores.clone(), vectors.clone()
            index = int(torch.randint(n, (1,), generator=generator))
            neighbor_scores[index] = torch.randn((), generator=generator) * 20
            neighbor_vectors[index] *= -1
            weights, _, _ = ino_weights(scores, beta_alpha=2, beta_beta=3, expected_batch_size=n)
            neighbor_weights, _, _ = ino_weights(neighbor_scores, beta_alpha=2, beta_beta=3, expected_batch_size=n)
            difference = (weights[:, None] * vectors - neighbor_weights[:, None] * neighbor_vectors).sum(0)
            assert difference.norm() <= 2 + 1e-6


def make_user(config=None, rounds=3, samples=10):
    controller = DefenseController({"name": "www", **(config or {})}, torch.device("cpu"), 2, 2, rounds)
    data = TensorDataset(torch.arange(samples * 2).float().reshape(samples, 2) / 20, torch.arange(samples) % 2)
    user = UserBase(torch.device("cpu"), 0, "toy", data, data, torch.nn.Linear(2, 2),
                    5, 0.01, 1, defense_controller=controller, federated_method="fedsgd")
    controller.www_privacy.configure([user])
    return user, controller


def test_missing_reference_is_plain_ce_without_dp_and_schedule_exhaustion_stops():
    user, controller = make_user()
    controller.federated_method = "fedsgd"
    privacy = controller.www_privacy
    assert privacy.config["target_epsilon"] is None
    assert privacy.config["max_grad_norm"] is None
    for index in range(3):
        user.train(round_index=index)
        assert user.last_gradient_capture_count == 1
        assert torch.equal(user.www_risk_weights, torch.zeros(user.last_update_sample_count, dtype=torch.float64))
        assert user.www_tail_mask.sum() == 0
    summary = controller.summary()
    assert not summary["privacy_accounting"]["formal_dp_enabled"]
    assert not summary["privacy_accounting"]["client_upload_is_private"]
    assert summary["privacy_accounting"]["epsilon_upper_bound"] is None
    assert summary["privacy_accounting"]["accountant"] is None
    assert summary["privacy_accounting"]["noise_std_on_sum"] == 0
    assert controller.conservative_dp_epsilon() is None
    assert summary["www"]["training_action"] == "risk_controlled_prediction_loss"
    assert summary["www"]["missing_reference"] == "cross_entropy_only"
    assert summary["metrics"]["www_regularization_loss"] == 0
    assert summary["metrics"]["www_total_loss"] == summary["metrics"]["www_ce_loss"]
    with pytest.raises(RuntimeError, match="schedule"):
        user.train(round_index=3)


def test_fixed_batch_has_no_noise_in_either_reproducibility_mode(monkeypatch):
    # Shuffled sampling uses the experiment seed, independent of obsolete DP flags.
    def gradient(reproducible):
        torch.manual_seed(42)
        user, controller = make_user({"reproducible_dp_noise": reproducible})
        user.train(round_index=0)
        return user.last_update_gradients, controller.summary()["privacy_accounting"]
    a, _ = gradient(False)
    b, _ = gradient(False)
    assert all(torch.equal(a[k], b[k]) for k in a)
    a, metadata = gradient(True)
    b, _ = gradient(True)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not metadata["formal_dp_enabled"]


@pytest.mark.parametrize("backend", ["loop", "batched"])
def test_risk_loss_upload_matches_direct_batch_backward_without_clipping_or_noise(backend, monkeypatch):
    user, controller = make_user({"reproducible_dp_noise": True, "grad_sample_backend": backend,
                                  "www_regularization_weight": 3.})
    privacy = controller.www_privacy
    # A final short batch must use its actual size, rather than five.
    x, y = next(iter(user.trainloader))
    x, y = x[:2], y[:2]
    parameters = list(user.model.parameters())
    initial = [p.detach().clone() for p in parameters]
    x = x * 1000  # Large gradients must not be capped at the historical C=8.
    weights, _, _ = risk_regularization_weights(torch.arange(y.numel()).float(), expected_batch_size=5)
    teacher = torch.tensor([.1, .8], requires_grad=True)
    reference = copy.deepcopy(user.model)
    logits = reference(x)
    loss = (torch.nn.functional.cross_entropy(logits, y, reduction="none")
            + 3 * weights.to(logits) * (logits.softmax(-1).gather(1, y[:, None]).flatten() - teacher.detach()).abs()).mean()
    loss.backward()
    expected = [p.grad.clone() for p in reference.parameters()]
    def disallow_noise(*args, **kwargs):
        raise AssertionError("WWW must not draw Gaussian gradient noise")
    monkeypatch.setattr(torch, "randn", disallow_noise)
    monkeypatch.setattr(torch, "randn_like", disallow_noise)
    monkeypatch.setattr("privacy_defenses.www_dp.weighted_clipped_sum", disallow_noise)
    optimizer = torch.optim.SGD(parameters, lr=user.learning_rate)
    optimizer.register_step_pre_hook(lambda *_: user.capture_protocol_gradients(user.model))
    privacy.step(user, user.model, optimizer, x, y, weights, 0, reference_probability=teacher)
    assert teacher.grad is None
    for p, before, gradient, uploaded in zip(parameters, initial, expected,
                                            user.last_update_gradients.values()):
        torch.testing.assert_close(uploaded, gradient)
        torch.testing.assert_close(p, before - user.learning_rate * gradient)


def test_post_round_diagnostics_do_not_disable_pre_update_defense():
    user, controller = make_user({"www_analysis_timing": "post_round",
                                 "www_analysis_interval": 50})
    own = user.get_parameters()
    other = {name: value + 0.2 for name, value in own.items()}
    global_state = {name: 0.5 * own[name] + 0.5 * other[name] for name in own}
    controller.prepare_client_training(user, global_state, 0.5, source_round=0)
    user.set_parameters(global_state)
    user.train(round_index=1)
    assert int(user.www_tail_mask.sum()) == min(user.last_update_sample_count, 4)
    assert user.www_source_round == 0
    assert int((user.www_risk_weights > 0).sum()) == min(user.last_update_sample_count, 4)


def test_www_uses_shuffling_while_record_dp_retains_poisson_and_noise():
    outputs = {}
    for name in ("www", "record_dp"):
        torch.manual_seed(24)
        controller = DefenseController(
            {"name": name, "target_epsilon": 8., "max_grad_norm": 8.,
             "delta": 1e-5, "reproducible_noise": True, "microbatch_size": 1},
            torch.device("cpu"), 2, 2, 4, samples_num=[12, 25],
        )
        controller.federated_method = "fedsgd"
        users = []
        for i, count in enumerate((12, 25)):
            data = TensorDataset(torch.randn(count, 2), torch.arange(count) % 2)
            users.append(UserBase(torch.device("cpu"), i, "toy", data, data,
                                  torch.nn.Linear(2, 2), 5, 0.01, 1,
                                  defense_controller=controller, federated_method="fedsgd"))
        if name == "www":
            controller.www_privacy.configure(users)
            noise = controller.www_privacy.noise_multiplier
        else:
            controller.configure_record_dp(users)
            noise = controller.record_dp_noise_multiplier
        for user in users:
            user.train(round_index=0)
        outputs[name] = noise, users, controller
    assert outputs["www"][0] == 0
    assert outputs["record_dp"][0] > 0
    for www, dp in zip(outputs["www"][1], outputs["record_dp"][1]):
        assert www.last_update_sample_count == 5
        generator = torch.Generator().manual_seed(42 + 1000003 * dp.id + 17011)
        expected = (torch.rand(dp.train_samples, generator=generator) < dp.record_dp_sample_rate).nonzero().flatten()
        torch.testing.assert_close(dp.last_train_indices, expected)
    assert outputs["www"][2].summary()["privacy_accounting"]["epsilon_upper_bound"] is None
    assert outputs["www"][2].summary()["privacy_accounting"]["sampling"] == "shuffled_batches"
    assert outputs["www"][2].summary()["privacy_accounting"]["normalization"] == "actual_batch_size"
    assert outputs["record_dp"][2].summary()["privacy_accounting"]["epsilon_upper_bound"] > 0


def test_www_shuffled_pass_covers_each_record_once_and_matches_normal_batches(monkeypatch):
    user, controller = make_user(samples=12, rounds=6)
    normal = UserBase(torch.device("cpu"), 0, "toy", user.train_data, user.test_data,
                      torch.nn.Linear(2, 2), 5, .01, 1, federated_method="fedsgd")
    def disallow_poisson(*args, **kwargs):
        raise AssertionError("WWW must not sample Poisson batches")
    monkeypatch.setattr(user, "iter_poisson_batches", disallow_poisson)
    batches = []
    for r in range(6):
        user.train(round_index=r)
        normal_batch = normal.next_train_batch()
        assert user.last_update_sample_count == [5, 5, 2][r % 3]
        assert user.last_gradient_capture_count == 1
        torch.testing.assert_close(user.last_train_batch[0], normal_batch[0])
        batches.append(user.last_train_indices.clone())
    for start in (0, 3):
        assert sorted(torch.cat(batches[start:start + 3]).tolist()) == list(range(12))
    assert not torch.equal(torch.cat(batches[:3]), torch.cat(batches[3:]))


@pytest.mark.parametrize("defense", ["record_dp"])
def test_empty_batch_server_retains_upload_and_skips_only_batch_attacks(defense, monkeypatch, tmp_path):
    model = tiny_model("clip_mlp", monkeypatch)
    server = ServerBase(
        device=torch.device("cpu"), dataset_name="toy", model=model,
        train_sets=[toy_dataset("clip_mlp", 2, 10+i) for i in range(2)],
        test_sets=[toy_dataset("clip_mlp", 20, 20+i) for i in range(2)],
        class_names=["0", "1", "2"], batch_size=2, eval_batch_size=32,
        learning_rate=0.01, num_glob_iters=2, local_epochs=1, total_users=2,
        results_dir=str(tmp_path), user_per_round=2,
        aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"), eval_interval=1,
        audit_config={"enabled": True, "strict": True,
                      "attacks": ["blackbox_loss", "loss_series", "projres"],
                      "candidate_sampling": "balanced_global_holdout", "require_full_target_train_members": True,
                      "nonmember_to_member_ratio": 1, "exact_batch_membership_attacks": ["blackbox_loss", "projres"],
                      "exact_batch_nonmember_to_member_ratio": 10, "paper_balanced_evaluation_size": 0,
                      "audit_batch_size": 32, "attack_audit_intervals": {"blackbox_loss": 1, "loss_series": 1, "projres": 1},
                      "low_fpr_min_nonmembers": 2, "training_health_check": False},
        projres_config={"enabled": True, "evaluation_interval": 1,
                        "max_candidates": 0, "min_nonmembers": 0, "max_nonmembers": 0,
                        "threshold": None, "decision_mode": "ranking"},
        defense_config={"name": defense, "target_epsilon": 8., "max_grad_norm": 8.,
                        "reproducible_noise": True, "release_private_diagnostics": True,
                        "www_analysis_timing": "post_round"},
    )
    monkeypatch.setattr("users.user.torch.rand", lambda n, **kwargs: torch.ones(n))
    summaries = server.train()
    assert {s["attack"] for s in summaries} == {"loss_series"}
    assert server.auditor.errors == {}
    assert server.defense.steps == {0: 2, 1: 2}
    assert server.ctx.update_sample_counts == {0: 2, 1: 2}
    audit = json.loads((tmp_path / "privacy_audit/summary.json").read_text())
    assert len(audit["exact_batch_skipped_rounds"]) == 2
    assert {r["reason"] for r in audit["exact_batch_skipped_rounds"]} == {"empty_poisson_batch"}
    for user in server.ctx.users:
        assert user.last_update_sample_count == 0
        assert user.last_gradient_capture_count == 1
        assert any(g.abs().sum() > 0 for g in user.last_update_gradients.values())


@pytest.mark.parametrize("override", [
    {"www_regularization_weight": -1}, {"www_regularization_weight": float("inf")},
    {"www_regularization_weight": float("nan")}, {"www_regularization_weight": True},
    {"www_regularization_weight": None}, {"www_regularization_weight": "bad"},
    {"www_tail_fraction": 0}, {"www_tail_fraction": float("nan")},
    {"sampling": "poisson"}, {"www_tail_basis": "invalid"},
    {"www_record_diagnostics": "true"},
    {"www_feature_statistics": True},
])
def test_invalid_risk_loss_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        validate_www({"name": "www", **override})


@pytest.mark.parametrize("epsilon,noise", [(3, "auto"), (16, 0.01), (8, 10), (None, 0)])
def test_legacy_privacy_overrides_cannot_reenable_noise_or_accounting(epsilon, noise, monkeypatch):
    def disallow_calibration(*args, **kwargs):
        raise AssertionError("WWW must not calibrate DP noise")
    monkeypatch.setattr("utils.privacy_accounting.calibrate_poisson_sampled_gaussian_noise", disallow_calibration)
    _, controller = make_user({"target_epsilon": epsilon, "noise_multiplier": noise,
                               "delta": 1e-5, "adjacency": "add_remove", "accountant": "rdp"})
    assert controller.config["noise_multiplier"] == 0
    for key in ("target_epsilon", "delta", "adjacency", "accountant"):
        assert controller.config[key] is None
    metadata = controller.summary()["privacy_accounting"]
    assert not metadata["formal_dp_enabled"]
    assert metadata["epsilon_upper_bound"] is None
    json.dumps(metadata, allow_nan=False)


def test_active_probes_are_rejected():
    user, controller = make_user()
    with pytest.raises(ValueError, match="active client probes"):
        controller.www_privacy.configure([user], 1)


def test_unified_runner_www_defaults_and_cli_overrides():
    catalog = load_yaml("configs/experiment_catalog.yaml")
    args = parse_args(["--models", "clip_mlp,clip_adapter,clip_lora,bert_adapter,bert_lora", "--defenses", "www", "--attacks", "all"])
    tasks, skipped = build_tasks(catalog, args)
    assert not skipped
    assert {t.model for t in tasks} == {"clip_mlp", "clip_adapter", "clip_lora", "bert_adapter", "bert_lora"}
    assert all(t.config["defense"] == {"name": "www", **DEFAULTS} for t in tasks)
    args = parse_args(["--models", "clip_mlp", "--defenses", "www", "--set", "defense.target_epsilon=5", "--set", "defense.max_grad_norm=4", "--set", "defense.www_regularization_weight=3"])
    tasks, _ = build_tasks(catalog, args)
    assert all(t.config["defense"]["target_epsilon"] is None and t.config["defense"]["max_grad_norm"] is None
               and t.config["defense"]["www_regularization_weight"] == 3 for t in tasks)
    args = parse_args(["--models", "bert_adapter", "--datasets", "cola", "--defenses", "record_dp,www",
                       "--set", "defense.target_epsilon=16", "--set", "batch_size=32"])
    tasks, skipped = build_tasks(catalog, args)
    assert not skipped
    configs = {t.defense: t.config["defense"] for t in tasks}
    assert configs["record_dp"]["target_epsilon"] == 16
    assert configs["record_dp"]["noise_multiplier"] == "auto"
    assert configs["www"]["target_epsilon"] is None
    assert configs["www"]["noise_multiplier"] == 0


@pytest.mark.parametrize("model_type", ["clip_mlp", "clip_adapter", "clip_lora", "bert_adapter", "bert_lora"])
@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_five_peft_models_risk_loss_training_and_all_attacks(model_type, device, monkeypatch, tmp_path):
    if device.startswith("cuda") and torch.cuda.device_count() < 2:
        pytest.skip("GPU 1 is unavailable")
    torch.manual_seed(42)
    model = tiny_model(model_type, monkeypatch).to(device)
    model.device = torch.device(device)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    catalog = load_yaml("configs/experiment_catalog.yaml")
    attacks = catalog["attacks"]["all"]
    messages = {}
    def observer(**kwargs):
        messages[kwargs["round_index"], kwargs["client_id"]] = copy.deepcopy(kwargs["gradients"])
    server = ServerBase(
        device=torch.device(device), dataset_name="toy", model=model,
        train_sets=[toy_dataset(model_type, 2, 10+i) for i in range(2)],
        test_sets=[toy_dataset(model_type, 24, 20+i) for i in range(2)],
        class_names=["0", "1", "2"], batch_size=5, eval_batch_size=32,
        learning_rate=0.01, num_glob_iters=3, local_epochs=1, total_users=2,
        results_dir=str(tmp_path), user_per_round=2,
        aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"), eval_interval=1,
        audit_config={"enabled": True, "strict": True, "attacks": attacks,
                      "candidate_sampling": "balanced_global_holdout", "require_full_target_train_members": True,
                      "nonmember_to_member_ratio": 1, "exact_batch_membership_attacks": catalog["attacks"]["exact_batch"],
                      "exact_batch_nonmember_to_member_ratio": 10, "paper_balanced_evaluation_size": 0,
                      "audit_batch_size": 32, "attack_audit_intervals": {a: 1 for a in attacks},
                      "low_fpr_min_nonmembers": 2, "training_health_check": False, "seed": 42},
        projres_config={"enabled": True, "evaluation_interval": 1, "token_reduction": "mean",
                        "max_candidates": 0, "min_nonmembers": 0, "max_nonmembers": 0,
                        "threshold": None, "decision_mode": "ranking"},
        defense_config={"name": "www", "reproducible_dp_noise": True},
        method_config={"client_optimizer": "sgd", "momentum": 0, "weight_decay": 0, "max_grad_norm": 0, "seed": 42},
        client_gradient_observer=observer,
    )
    summaries = server.train()
    assert server.auditor.errors == {}
    assert {s["attack"] for s in summaries} == set(attacks)
    assert server.defense.steps == {0: 3, 1: 3}
    for user in server.ctx.users:
        assert user.last_gradient_capture_count == 1
        assert user.last_update_sample_count == user.last_train_indices.numel()
        assert user.www_ranking_round == 2
        assert int(user.www_tail_mask.sum()) == math.ceil(user.last_update_sample_count * .8)
        assert user.www_source_round == 1
        assert torch.equal(user.www_ranked_scores, user.www_scores.sort().values)
        for name, tensor in server.ctx.protocol_messages[user.id]["tensors"].items():
            torch.testing.assert_close(tensor, messages[2, user.id][name], rtol=0, atol=0)
    for selection in server.auditor.exact_batch_candidate_selections:
        assert selection["member_local_indices"].min() >= 0
        assert selection["nonmember_label_histogram"] == [x*10 for x in selection["member_label_histogram"]]
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    defense = json.loads((tmp_path / "defense_summary.json").read_text())
    assert defense["privacy_accounting"]["epsilon_upper_bound"] is None
    assert not defense["privacy_accounting"]["formal_dp_enabled"]
    assert server.auditor.record_dp_accounting is None
    if model_type in {"clip_lora", "bert_lora"}:
        assert any(row["communication_round"] == 1 and row["attacks"] == ["projres"]
                   and row["reason"] == "zero_observed_update"
                   for row in server.auditor.exact_batch_skipped_rounds)
    assert not (tmp_path / "privacy_audit/www_attack_samples.csv").exists()
    with (tmp_path / "www_diagnostics/sample_gradients.csv").open() as file:
        samples = list(csv.DictReader(file))
    assert len(samples) == 2 * (5 + 1 + 5)  # Each client has six records; retain the short batch.
    assert {int(row["communication_round"]) for row in samples} == {1, 2, 3}
    for row in samples:
        client = server.ctx.users[int(row["client_id"])]
        assert int(row["label"]) == int(client.train_data[int(row["local_sample_index"])][1])
        raw, regularizer, total = [float(row[k]) for k in ("raw_grad_norm", "regularizer_grad_norm", "total_grad_norm")]
        factor = float(row["ce_gradient_factor"])
        assert total == pytest.approx(raw * abs(factor), rel=1e-6)
        expected_regularizer_factor = float(row["regularization_weight"]) * float(row["risk_weight"]) * float(row["current_probability"])
        if row["confidence_gap"] == "" or float(row["confidence_gap"]) == 0:
            expected_regularizer_factor = 0
        assert regularizer == pytest.approx(raw * expected_regularizer_factor, rel=1e-6)
        assert float(row["total_loss"]) == pytest.approx(float(row["ce_loss"]) + float(row["regularization_loss"]), rel=1e-6)
        if row["risk_available"] == "1":
            assert float(row["risk_score"]) == pytest.approx(float(row["other_loss"]) - float(row["own_loss"]), abs=1e-7)
        else:
            assert row["risk_score"] == row["risk_rank"] == ""
        if row["group"] == "low_risk":
            assert raw == total and regularizer == 0
            assert row["risk_weight"] == "0.0" and row["regularization_loss"] == "0.0"
        if row["group"] == "warmup":
            assert row["reference_probability"] == ""
            assert row["regularization_loss"] == "0.0"
    diagnostic_summary = json.loads((tmp_path / "www_diagnostics/summary.json").read_text())
    assert diagnostic_summary["status"] == "completed"
    assert diagnostic_summary["sample_rows"] == len(samples)
    assert diagnostic_summary["risk_available_rows"] == 2 * (1 + 5)
    assert diagnostic_summary["schema_version"] == 2
    assert not diagnostic_summary["clipping_enabled"]
    # Prediction-only sample losses preserve the batch rank bound, but still
    # changes the training objective from the unmodified paper protocol.
    prediction_files = list((tmp_path / "privacy_audit").rglob("*.json"))
    metadata = [p.read_text() for p in prediction_files if '"batch_rank_bound"' in p.read_text()]
    assert metadata
    assert all('"batch_rank_bound": null' not in s for s in metadata)
    assert all('"attacked_parameter_perturbed": false' in s for s in metadata)
    assert all('"paper_fedsgd_exact": false' in s for s in metadata)
