from __future__ import annotations

import copy
import json
import math

import pytest
import torch
from scipy.integrate import quad
from scipy.special import betainc
from torch.utils.data import TensorDataset

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.controller import DefenseController
from privacy_defenses.www import infer_other_clients_state, rank_loss_differences
from privacy_defenses.www_dp import (
    DEFAULTS, ino_weights, validate_www, weighted_clipped_sum,
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


@pytest.mark.parametrize("count", [1, 2, 5, 10, 16, 32])
def test_highest_twenty_percent_and_linear_tif_exact_integrals(count):
    scores = torch.arange(count, dtype=torch.float64).flip(0)
    weights, positions, tail = ino_weights(scores, expected_batch_size=count)
    m = math.ceil(count * 0.2)
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


@pytest.mark.parametrize("actual", [0, 1, 2, 3, 4, 8, 16, 25])
def test_fixed_tail_uses_expected_batch_and_right_aligns_small_draws(actual):
    weights, order, tail = ino_weights(torch.arange(actual).float(), expected_batch_size=16)
    m = 4
    expected = torch.tensor([0.875, 0.625, 0.375, 0.125], dtype=torch.float64)
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
                                expected_batch_size=16)
        for position in range(actual + 1):
            neighbor = torch.cat((scores, torch.tensor([position - 0.5])))
            new, _, _ = ino_weights(neighbor, beta_alpha=alpha, beta_beta=beta,
                                    expected_batch_size=16)
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


def test_budget_covers_all_rounds_first_round_is_private_and_exhaustion_stops():
    user, controller = make_user()
    controller.federated_method = "fedsgd"
    privacy = controller.www_privacy
    assert privacy.config["target_epsilon"] == 3
    assert privacy.config["max_grad_norm"] == 8
    assert privacy.epsilon(3) <= 3 + 1e-10
    for index in range(3):
        user.train(round_index=index)
        assert user.last_gradient_capture_count == 1
        assert torch.equal(user.www_importance_weights, torch.ones(user.last_update_sample_count, dtype=torch.float64))
        assert user.www_tail_mask.sum() == 0
    summary = controller.summary()
    assert summary["privacy_accounting"]["formal_dp_enabled"]
    assert summary["privacy_accounting"]["epsilon_upper_bound"] == pytest.approx(3)
    assert summary["privacy_accounting"]["sum_sensitivity"] == 8
    assert summary["metrics"] == {}
    with pytest.raises(RuntimeError, match="schedule"):
        user.train(round_index=3)


def test_noise_is_not_derived_from_public_seed_and_debug_is_not_formal_dp():
    def gradient(reproducible):
        torch.manual_seed(42)
        user, controller = make_user({"reproducible_dp_noise": reproducible})
        user.train(round_index=0)
        return user.last_update_gradients, controller.summary()["privacy_accounting"]
    a, _ = gradient(False)
    b, _ = gradient(False)
    assert any(not torch.equal(a[k], b[k]) for k in a)
    a, metadata = gradient(True)
    b, _ = gradient(True)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not metadata["formal_dp_enabled"]


def test_noisy_upload_exact_scale_and_normalization():
    user, controller = make_user({"reproducible_dp_noise": True})
    privacy = controller.www_privacy
    # Realized batch deliberately differs from the expected size of five.
    x, y = next(iter(user.trainloader))
    x, y = x[:2], y[:2]
    parameters = list(user.model.parameters())
    initial = [p.detach().clone() for p in parameters]
    weights, _, _ = ino_weights(torch.arange(y.numel()).float(), expected_batch_size=user.record_dp_expected_batch_size)
    sums = weighted_clipped_sum(user.model, x, y, parameters, 8., weights)
    expected_generator = privacy.generator(user.id, 0)
    expected = [(s + torch.randn(p.shape, generator=expected_generator)
                 * privacy.noise_multiplier * 8.) / user.record_dp_expected_batch_size
                for p, s in zip(parameters, sums)]
    optimizer = torch.optim.SGD(parameters, lr=user.learning_rate)
    optimizer.register_step_pre_hook(lambda *_: user.capture_protocol_gradients(user.model))
    privacy.step(user, user.model, optimizer, x, y, weights, 0,
                 privacy.generator(user.id, 0))
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
    assert int(user.www_tail_mask.sum()) == min(user.last_update_sample_count, 1)
    assert user.www_source_round == 0
    assert int((user.www_importance_weights < 1).sum()) == min(user.last_update_sample_count, 1)


def test_www_and_record_dp_share_sampling_calibration_and_warmup_upload():
    outputs = {}
    for name in ("www", "record_dp"):
        torch.manual_seed(24)
        controller = DefenseController(
            {"name": name, "target_epsilon": 8., "max_grad_norm": 8.,
             "delta": 1e-5, "reproducible_noise": True, "microbatch_size": 1},
            torch.device("cpu"), 2, 2, 4,
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
        outputs[name] = noise, users
    assert outputs["www"][0] == outputs["record_dp"][0]
    for www, dp in zip(outputs["www"][1], outputs["record_dp"][1]):
        assert torch.equal(www.last_train_indices, dp.last_train_indices)
        assert www.record_dp_sample_rate == dp.record_dp_sample_rate
        assert www.record_dp_expected_batch_size == dp.record_dp_expected_batch_size
        for name in www.last_update_gradients:
            torch.testing.assert_close(www.last_update_gradients[name], dp.last_update_gradients[name], rtol=0, atol=0)


def test_empty_poisson_draw_is_not_resampled_and_uploads_accounted_noise(monkeypatch):
    user, controller = make_user({"reproducible_dp_noise": True,
                                 "release_private_diagnostics": True})
    calls = []
    def empty_draw(*args, **kwargs):
        calls.append(args[0])
        return torch.ones(args[0])
    monkeypatch.setattr("users.user.torch.rand", empty_draw)
    initial = user.get_parameters()
    controller.prepare_client_training(user, initial, 0.5, source_round=0)
    user.train(round_index=1)
    assert calls == [user.train_samples]
    assert user.last_update_sample_count == 0
    assert user.last_train_indices.numel() == 0
    assert user.www_scores.numel() == 0
    assert user.www_importance_weights.numel() == 0
    assert user.last_gradient_capture_count == 1
    assert controller.steps[user.id] == 1
    assert controller.www_privacy.epsilon(1, user.id) > 0
    generator = controller.www_privacy.generator(user.id, 1)
    for name, p in user.model.named_parameters():
        noise = torch.randn(p.shape, generator=generator)
        expected = noise * controller.www_privacy.noise_multiplier * 8 / 5
        torch.testing.assert_close(user.last_update_gradients[name], expected)
        torch.testing.assert_close(p, initial[name] - user.learning_rate * expected)


@pytest.mark.parametrize("defense", ["www", "record_dp"])
def test_empty_batch_server_retains_noise_upload_and_skips_only_batch_attacks(defense, monkeypatch, tmp_path):
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
    {"target_epsilon": 0}, {"target_epsilon": float("nan")},
    {"max_grad_norm": -1}, {"max_grad_norm": float("inf")},
    {"delta": 1}, {"www_tail_fraction": 0}, {"www_tail_fraction": float("nan")},
    {"www_beta_alpha": 0}, {"noise_multiplier": 0},
    {"sampling": "shuffled_batches"}, {"adjacency": "replace_one"},
    {"www_feature_statistics": True},
])
def test_invalid_privacy_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        validate_www({"name": "www", **override})


def test_under_budget_noise_and_active_probes_are_rejected():
    with pytest.raises(ValueError, match="too small"):
        make_user({"noise_multiplier": 0.01})
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
    args = parse_args(["--models", "clip_mlp", "--defenses", "www", "--set", "defense.target_epsilon=5", "--set", "defense.max_grad_norm=4"])
    tasks, _ = build_tasks(catalog, args)
    assert all(t.config["defense"]["target_epsilon"] == 5 and t.config["defense"]["max_grad_norm"] == 4 for t in tasks)


@pytest.mark.parametrize("model_type", ["clip_mlp", "clip_adapter", "clip_lora", "bert_adapter", "bert_lora"])
def test_five_peft_models_private_training_and_all_attacks(model_type, monkeypatch, tmp_path):
    torch.manual_seed(42)
    model = tiny_model(model_type, monkeypatch)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    catalog = load_yaml("configs/experiment_catalog.yaml")
    attacks = catalog["attacks"]["all"]
    messages = {}
    def observer(**kwargs):
        messages[kwargs["round_index"], kwargs["client_id"]] = copy.deepcopy(kwargs["gradients"])
    server = ServerBase(
        device=torch.device("cpu"), dataset_name="toy", model=model,
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
        assert int(user.www_tail_mask.sum()) == min(user.last_update_sample_count, 1)
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
    assert defense["privacy_accounting"]["epsilon_upper_bound"] <= 3 + 1e-10
    assert not (tmp_path / "privacy_audit/www_attack_samples.csv").exists()
    # The exact-batch audit must use a noisy update without a clean batch rank cap.
    prediction_files = list((tmp_path / "privacy_audit").rglob("*.json"))
    metadata = [p.read_text() for p in prediction_files if '"batch_rank_bound"' in p.read_text()]
    assert metadata
    assert all('"batch_rank_bound": null' in s for s in metadata)
    assert all('"paper_fedsgd_exact": false' in s for s in metadata)
