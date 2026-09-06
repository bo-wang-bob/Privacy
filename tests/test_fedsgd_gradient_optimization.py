from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from aggregator.aggregator_builder import build_aggregator
from privacy_attacks.auditor import MembershipAuditor
from privacy_defenses.www_dp import weighted_clipped_sum, validate_www
from servers.serverbase import ServerBase
from utils.per_sample_gradients import clipped_sum_from_losses, resolve_grad_sample_backend
from test_cofedmid import tiny_model, toy_dataset


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
@pytest.mark.parametrize("model_type", ["clip_mlp", "clip_adapter", "clip_lora", "bert_adapter", "bert_lora", "gpt2_adapter"])
def test_chunked_joint_clip_matches_individual_model_gradients(model_type, device, monkeypatch):
    if device.startswith("cuda") and torch.cuda.device_count() < 2:
        pytest.skip("GPU 1 is unavailable")
    torch.manual_seed(18)
    model = tiny_model(model_type, monkeypatch).to(device)
    if hasattr(model, "device"):
        model.device = torch.device(device)
    # Exercise inner adapter/LoRA gradients, not just their zero-init branches.
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.02)
    client = model.create_client_model(0) if hasattr(model, "create_client_model") else model
    client.eval()  # Deterministic reference; production dropout remains configured.
    session = client.use_shared_model() if hasattr(client, "use_shared_model") else nullcontext()
    x, y = toy_dataset(model_type, 3, 8).tensors
    x, y = x[:7].to(device), y[:7].to(device)
    weights = torch.tensor([1, .8, .6, 1, .4, .2, 1], device=device)
    with session:
        parameters = [p for p in client.parameters() if p.requires_grad]
        reference = weighted_clipped_sum(client, x, y, parameters, .1, weights, backend="loop")
        for chunk in (1, 4):
            actual = weighted_clipped_sum(client, x, y, parameters, .1, weights,
                                          backend="batched", microbatch_size=chunk)
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, rtol=3e-4, atol=2e-6)
        assert all(p.grad is None for p in parameters)


def test_joint_norm_includes_scalar_and_unused_parameters_and_detaches_weights():
    p = torch.nn.Parameter(torch.tensor(2.))
    unused = torch.nn.Parameter(torch.ones(3))
    weights = torch.tensor([1., .25], requires_grad=True)
    sums, factors = clipped_sum_from_losses(p * torch.tensor([3., 4.]), [p, unused], 2., weights)
    torch.testing.assert_close(sums[0], torch.tensor(2.5))
    torch.testing.assert_close(sums[1], torch.zeros(3))
    torch.testing.assert_close(factors, torch.tensor([2/3, .5]))
    assert not sums[0].requires_grad and weights.grad is None


@pytest.mark.parametrize("defense", ["www", "record_dp"])
@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_chunks_preserve_poisson_noisy_upload_and_uniform_fedsgd(defense, device, monkeypatch, tmp_path):
    if device.startswith("cuda") and torch.cuda.device_count() < 2:
        pytest.skip("GPU 1 is unavailable")
    runs = []
    for backend in ("loop", "batched"):
        torch.manual_seed(8)
        messages = {}
        def observe(**kwargs):
            messages[kwargs["round_index"], kwargs["client_id"]] = {
                n: g.clone() for n, g in kwargs["gradients"].items()
            }
        model = tiny_model("bert_adapter", monkeypatch).to(device)
        model.device = torch.device(device)
        server = ServerBase(
            device=torch.device(device), dataset_name="toy", model=model,
            train_sets=[toy_dataset("bert_adapter", 4+i, 20+i) for i in range(2)],
            test_sets=[toy_dataset("bert_adapter", 4, 30+i) for i in range(2)],
            class_names=["0", "1", "2"], batch_size=7, eval_batch_size=32,
            learning_rate=.01, num_glob_iters=3, local_epochs=1, total_users=2,
            results_dir=str(tmp_path / backend), user_per_round=2, eval_interval=3,
            aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"),
            audit_config={"enabled": False}, projres_config={"enabled": False},
            defense_config={"name": defense, "target_epsilon": 8., "max_grad_norm": .3,
                            "reproducible_noise": True, "grad_sample_backend": backend,
                            "microbatch_size": 4},
            method_config={"client_optimizer": "sgd", "momentum": 0, "weight_decay": 0, "max_grad_norm": 0, "seed": 42},
            client_gradient_observer=observe,
        )
        server.train()
        assert server.defense.steps == {0: 3, 1: 3}
        assert len(messages) == 6
        for user in server.ctx.users:
            assert user.last_gradient_capture_count == 1
        runs.append((messages, server))
    reference, actual = runs
    for key, tensors in reference[0].items():
        for name, value in tensors.items():
            torch.testing.assert_close(actual[0][key][name], value, rtol=2e-5, atol=1e-6)
    for a, b in zip(reference[1].ctx.users, actual[1].ctx.users):
        torch.testing.assert_close(a.last_train_indices, b.last_train_indices)
        assert a.record_dp_expected_batch_size == b.record_dp_expected_batch_size == 7
    for (name, p), (other_name, q) in zip(reference[1].ctx.model.named_parameters(), actual[1].ctx.model.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p, q, rtol=2e-5, atol=1e-6)


def individual_audit_reference(model, x, y, updates, difference_indices):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    cosines, differences = [], []
    norms = [sum((u[n].double() * scale).square().sum() for n, _ in parameters).sqrt().clamp_min(1e-12)
             for u, _, scale in updates]
    for row, label in zip(x, y):
        logits = model(row[None])
        losses = [F.cross_entropy(logits, label[None]),
                  (logits.shape[1] * logits.logsumexp(1) - logits.sum(1)).sum()]
        for i, loss in enumerate(losses):
            gs = [g.cpu() for g in torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=i == 0)]
            norm_sq = sum(g.double().square().sum() for g in gs)
            dots = [sum((g.double() * u[n].double()).sum() for (n, _), g in zip(parameters, gs)) * sign * scale
                    for u, sign, scale in updates]
            if i == 0:
                cosines.append(torch.stack([d / (norm_sq.sqrt().clamp_min(1e-12) * n) for d, n in zip(dots, norms)]).float())
            else:
                differences.append(torch.stack([2 * dots[j] - norm_sq for j in difference_indices]).float())
    return {"cosine": torch.stack(cosines).T, "gradient_difference": torch.stack(differences).T}


@pytest.mark.parametrize("model_type", ["clip_mlp", "clip_adapter", "clip_lora", "bert_adapter", "bert_lora", "gpt2_adapter"])
@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_streamed_audit_preserves_both_losses_signs_scales_and_target_order(model_type, device, monkeypatch):
    if device.startswith("cuda") and torch.cuda.device_count() < 2:
        pytest.skip("GPU 1 is unavailable")
    torch.manual_seed(11)
    model = tiny_model(model_type, monkeypatch).to(device).eval()
    model.device = torch.device(device)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * .02)
    x, y = toy_dataset(model_type, 3, 5).tensors
    x, y = x.to(device), y.to(device)
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    updates = [({n: torch.randn_like(p, device="cpu") for n, p in parameters}, sign, scale)
               for sign, scale in [(1., 1.), (-1., 20.), (-1., .125)]]
    reference = individual_audit_reference(model, x, y, updates, [2, 0])
    auditor = MembershipAuditor.__new__(MembershipAuditor)
    auditor.device, auditor.images, auditor.labels = torch.device(device), x.cpu(), y.cpu()
    for backend, cache in (("loop", 0), ("batched", 0), ("batched", 2048)):
        auditor.config = {"grad_sample_backend": backend, "grad_sample_chunk_size": 4, "gradient_update_cache_mb": cache}
        actual = auditor._raw_input_gradient_measurements(
            model, [n for n, _ in parameters], updates, need_cosine=True,
            need_gradient_difference=True, gradient_difference_update_indices=[2, 0])
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=2e-4, atol=2e-5)
    # A later call must read new uploads, including a true zero update.
    updates = [({n: torch.zeros_like(p, device="cpu") for n, p in parameters}, 1., 1.)]
    actual = auditor._raw_input_gradient_measurements(
        model, [n for n, _ in parameters], updates, need_cosine=True, need_gradient_difference=False)
    assert torch.equal(actual["cosine"], torch.zeros(1, len(y)))


@pytest.mark.parametrize("override", [{"grad_sample_backend": "vmap"}, {"microbatch_size": 0}, {"microbatch_size": 1.5}, {"microbatch_size": True}])
def test_invalid_www_gradient_settings(override):
    with pytest.raises(ValueError):
        validate_www({"name": "www", **override})


def test_checkpointed_shared_transformer_selects_reference_backend():
    class Model:
        gradient_checkpointing = True
    class Client:
        _shared_model = Model()
    assert resolve_grad_sample_backend(Client(), "auto") == "loop"
