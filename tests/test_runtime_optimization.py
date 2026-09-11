import json

import pytest
import torch
import torch.nn.functional as F

from aggregator.aggregator_builder import build_aggregator
from servers.serverbase import ServerBase
from utils.performance import StageTimings, validate_performance_config
from test_cofedmid import tiny_model, toy_dataset


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_server(tmp_path, monkeypatch, model_type, *, method="fedsgd", lr=.1,
                device="cpu", audit=None, dataset="cola", defense="none", www_diagnostics=False):
    model = tiny_model(model_type, monkeypatch).to(device)
    model.device = torch.device(device)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * .02)
    return ServerBase(
        device=torch.device(device), dataset_name=dataset, model=model,
        train_sets=[toy_dataset(model_type, 2+i, 20+i) for i in range(2)],
        test_sets=[toy_dataset(model_type, 5+i, 30+i) for i in range(2)],
        class_names=["0", "1", "2"], batch_size=2, eval_batch_size=4,
        learning_rate=lr, num_glob_iters=1, local_epochs=1, total_users=2,
        results_dir=str(tmp_path), user_per_round=2, eval_interval=1,
        aggregator=build_aggregator(method, aggregation_weighting="uniform"),
        audit_config=audit or {"enabled": False}, projres_config={"enabled": False},
        defense_config={"name": defense, "target_epsilon": 16., "max_grad_norm": 8.,
                        "reproducible_noise": True, "www_record_diagnostics": www_diagnostics},
        method_config={"client_optimizer": "sgd", "momentum": 0., "weight_decay": 0.},
    )


@pytest.mark.parametrize("model_type,defense", [
    (model, defense)
    for model in ["clip_lora", "bert_adapter", "bert_lora", "gpt2_adapter"]
    for defense in ["none", "cofedmid", "www"]
    if not (model == "gpt2_adapter" and defense == "www")
] + [("bert_adapter", "record_dp")])
def test_multibatch_fedavg_shared_backbones_and_defenses(
    model_type, defense, monkeypatch, tmp_path
):
    from test_clip_peft_fedsgd import _audit_config, ATTACKS
    torch.manual_seed(42)
    model = tiny_model(model_type, monkeypatch)
    # Exercise nonzero LoRA/Adapter inner projections. Tiny first-round BERT
    # query updates otherwise vanish when subtracting float32 model states.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * .02)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    audit = _audit_config()
    audit["audit_batch_size"] = 4
    audit["training_health_check"] = False
    server = ServerBase(
        device=torch.device("cpu"), dataset_name="toy", model=model,
        train_sets=[toy_dataset(model_type, 2+i, 20+i) for i in range(2)],
        test_sets=[toy_dataset(model_type, 12+i, 30+i) for i in range(2)],
        class_names=["0", "1", "2"], batch_size=4, eval_batch_size=8,
        learning_rate=.05, num_glob_iters=2, local_epochs=2, total_users=2,
        results_dir=str(tmp_path), user_per_round=2, eval_interval=1,
        aggregator=build_aggregator("fedavg", aggregation_weighting="sample_count"),
        audit_config=audit,
        projres_config={"enabled": True, "evaluation_interval": 1, "token_reduction": "mean"},
        defense_config={"name": defense, "noise_multiplier": 0.5,
                        "max_grad_norm": 1., "reproducible_noise": True,
                        "www_record_diagnostics": False, "cofedmid_init_round": 1,
                        "cofedmid_intervals": 1, "cofedmid_reproducible_noise": True},
        method_config={"client_optimizer": "sgd", "seed": 42},
    )
    summaries = server.train()
    assert server.auditor.errors == {}
    missing = ATTACKS - {s["attack"] for s in summaries}
    assert missing <= {"projres"}
    if missing:
        skipped = server.auditor.exact_batch_skipped_rounds
        assert len(skipped) == 2
        assert all(row["reason"] == "zero_observed_update" and row["attacks"] == ["projres"] for row in skipped)
    assert all(s["member_count"] == 6 and s["nonmember_count"] == 6 for s in summaries)
    assert server.ctx.aggregation_weights == {0: .4, 1: .6}
    assert all(m["kind"] == "model_update" for m in server.ctx.protocol_messages.values())
    for n, p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
    if defense in {"none", "www"}:
        assert [u.last_update_sample_count for u in server.ctx.users] == [12, 18]
    if defense == "record_dp":
        assert dict(server.defense.steps) == {0: 8, 1: 12}
        assert server.defense.record_dp_planned_steps == {0: 8, 1: 12}
    metadata = json.loads((tmp_path / "federated_method_summary.json").read_text())
    assert metadata["model_update_protocol"]["local_epochs"] == 2


@pytest.mark.parametrize("model_type", ["clip_mlp", "clip_adapter", "clip_lora"])
@pytest.mark.parametrize("method", ["fedsgd", "fedavg"])
@pytest.mark.parametrize("lr", [.1, .005])
def test_fixed_pool_gradient_diff_converts_upload_units_once(model_type, method, lr, tmp_path, monkeypatch):
    torch.manual_seed(7)
    audit = {
        "enabled": True, "strict": True, "attacks": ["gradient_diff", "avg_cosine"],
        "candidate_sampling": "balanced_global_holdout", "require_full_target_train_members": True,
        "nonmember_to_member_ratio": 1, "exact_batch_membership_attacks": [],
        "paper_balanced_evaluation_size": 0, "low_fpr_min_nonmembers": 2,
        "attack_audit_intervals": {"gradient_diff": 1, "avg_cosine": 1},
        "audit_batch_size": 4, "training_health_check": False,
    }
    server = make_server(tmp_path, monkeypatch, model_type, method=method, lr=lr, audit=audit)
    auditor = server.auditor
    original = auditor._raw_input_gradient_measurements
    checks = []

    def capture(model, names, updates, **kwargs):
        target = auditor.target_client_id
        # Build an independent observable gradient directly from the protocol.
        message = server.ctx.protocol_messages[target]
        upload = message["tensors"]
        scale = 1. if method == "fedsgd" else -1. / lr
        expected = []
        parameters = dict(model.named_parameters())
        forward = model.forward_from_image_features if auditor.candidate_inputs_are_features else model
        for image in auditor.images:
            logits = forward(image[None])
            loss = (logits.shape[1] * logits.logsumexp(1) - logits.sum(1)).sum()
            gradients = torch.autograd.grad(loss, [parameters[n] for n in names])
            norm_sq = sum(g.double().square().sum() for g in gradients)
            dot = sum((g.double() * upload[n].to(g).double() * scale).sum()
                      for n, g in zip(names, gradients))
            expected.append(2 * dot - norm_sq)
        actual = original(model, names, updates, **kwargs)
        torch.testing.assert_close(actual["gradient_difference"][0], torch.stack(expected).float(),
                                   rtol=3e-4, atol=2e-5)
        checks.append(True)
        return actual

    monkeypatch.setattr(auditor, "_raw_input_gradient_measurements", capture)
    def disallow_dense(*args, **kwargs):
        raise AssertionError("Normal CLIP audits must not materialize all candidate gradients/signatures")
    monkeypatch.setattr(auditor, "_candidate_gradients", disallow_dense)
    monkeypatch.setattr(auditor, "_cached_feature_gradient_cosines", disallow_dense)
    monkeypatch.setattr(auditor, "_cached_feature_gradient_differences", disallow_dense)
    server.train()
    assert checks == [True]
    timings = json.loads((tmp_path / "performance_summary.json").read_text())
    assert timings["status"] == "completed"
    for name in ("run", "audit.forward", "audit.backward", "audit.gradient_reduce", "outputs.write"):
        assert timings["stages"][name]["calls"] > 0
        assert timings["stages"][name]["wall_seconds"] >= 0


@pytest.mark.parametrize("model_type", ["bert_adapter", "bert_lora", "gpt2_adapter"])
@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
@pytest.mark.parametrize("dataset", ["cola", "imdb"])
def test_shared_evaluation_matches_client_metrics_and_preserves_states(model_type, device, dataset, tmp_path, monkeypatch):
    if device.startswith("cuda") and torch.cuda.device_count() < 2:
        pytest.skip("GPU 1 is unavailable")
    torch.manual_seed(11)
    server = make_server(tmp_path, monkeypatch, model_type, device=device, dataset=dataset)
    server.ctx.set_base_model_state(server.ctx.users[0].get_parameters())
    # Distinct local states must survive evaluation for next-round WWW ranking.
    for user in server.ctx.users:
        user.set_parameters({n: p + .03*(user.id+1) for n, p in user.get_parameters().items()})
    local = [user.get_parameters() for user in server.ctx.users]
    reference = server._evaluate_clients()
    global_before = {n: p.detach().clone() for n, p in server.model.named_parameters()}
    mode_before = server.model.training
    def disallow_binding(*args, **kwargs):
        raise AssertionError("Shared evaluation must not bind or overwrite client parameters")
    for user in server.ctx.users:
        monkeypatch.setattr(user, "set_parameters", disallow_binding)
        monkeypatch.setattr(user.model, "use_shared_model", disallow_binding)
    actual = server._evaluate_shared_model()
    assert actual[:3] == pytest.approx(reference[:3], rel=1e-7, abs=1e-7)
    if dataset == "cola":
        torch.testing.assert_close(actual[3], reference[3], rtol=0, atol=0)
    else:
        assert actual[3] is reference[3] is None
    for user, before in zip(server.ctx.users, local):
        for name, value in user.get_parameters().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for name, value in server.model.named_parameters():
        torch.testing.assert_close(value, global_before[name], rtol=0, atol=0)
    assert server.model.training == mode_before


def test_shared_evaluation_restores_global_state_on_forward_error(tmp_path, monkeypatch):
    server = make_server(tmp_path, monkeypatch, "bert_adapter")
    server.ctx.set_base_model_state({n: p+.2 for n,p in server.ctx.users[0].get_parameters().items()})
    before = {n: p.detach().clone() for n,p in server.model.named_parameters()}
    server.model.train()
    def fail(*args, **kwargs):
        raise RuntimeError("test forward failure")
    monkeypatch.setattr(server.model, "forward", fail)
    with pytest.raises(RuntimeError, match="test forward failure"):
        server._evaluate_shared_model()
    for name, value in server.model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert server.model.training


@pytest.mark.parametrize("defense,diagnostics", [("www", False), ("www", True), ("record_dp", False)])
def test_step_timings_distinguish_www_loss_from_record_dp_noise(defense, diagnostics, tmp_path, monkeypatch):
    server = make_server(tmp_path, monkeypatch, "bert_adapter", defense=defense, www_diagnostics=diagnostics)
    server.train()
    stages = json.loads((tmp_path / "performance_summary.json").read_text())["stages"]
    assert stages["evaluation"]["calls"] == 2
    if defense == "www":
        assert stages["train.www_ranking"]["calls"] == 2
        assert stages["train.www_backward_step"]["calls"] == 2
        if diagnostics:
            assert stages["train.www_gradient_diagnostics"]["calls"] == 2
        else:
            assert "train.www_gradient_diagnostics" not in stages
        assert "train.clipping_step" not in stages and "train.record_gradients" not in stages
        assert "train.noise_and_step" not in stages
    else:
        assert stages["train.record_gradients"]["calls"] == 2
        assert stages["train.noise_and_step"]["calls"] == 2


def test_timings_record_failure_and_can_be_disabled(tmp_path):
    timer = StageTimings("cpu")
    with pytest.raises(RuntimeError):
        with timer.measure("run"):
            with timer.measure("inner"):
                raise RuntimeError("failed")
    timer.save(tmp_path / "failed.json", status="failed")
    report = json.loads((tmp_path / "failed.json").read_text())
    assert report["status"] == "failed"
    assert report["stages"]["run"]["calls"] == 1
    assert report["stages"]["run"]["wall_seconds"] >= report["stages"]["inner"]["wall_seconds"]
    assert report["stages"]["run"]["cuda_stream_seconds"] is None
    disabled = StageTimings("cpu", enabled=False)
    with disabled.measure("run"):
        pass
    disabled.save(tmp_path / "disabled.json", status="completed")
    assert not (tmp_path / "disabled.json").exists() and not disabled.stages


@pytest.mark.parametrize("values", [{"enabled": "false"}, {"cuda_events": 0}, {"evaluation_backend": "typo"}])
def test_invalid_performance_options_fail_before_training(values):
    with pytest.raises(ValueError):
        validate_performance_config({"performance": values})
