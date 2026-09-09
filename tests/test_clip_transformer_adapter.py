import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from aggregator.aggregator_builder import build_aggregator
from main import validate_config
from privacy_attacks.model_utils import trainable_scope_name
from scripts.run_privacy_experiments import build_tasks, load_yaml, parse_args
from servers.serverbase import ServerBase
from trainmodel.clip_transformer_adapter import CLIPTransformerAdapter
from test_clip_peft_fedsgd import ATTACKS, _audit_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


def backbone_and_prompts():
    text = CLIPTextConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=8, eos_token_id=2,
        bos_token_id=1, pad_token_id=0,
    )
    vision = CLIPVisionConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, image_size=4, patch_size=2,
    )
    backbone = CLIPModel(CLIPConfig(
        text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=4,
    ))
    prompts = {"input_ids": torch.tensor([[1, 3, 2], [1, 4, 2], [1, 5, 2]])}
    return backbone, prompts


def tiny_model():
    backbone, prompts = backbone_and_prompts()
    return CLIPTransformerAdapter(backbone, prompts, ["a", "b", "c"])


def dataset(per_class, seed):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(3).repeat(per_class)
    return TensorDataset(torch.randn(len(labels), 3, 4, 4, generator=generator), labels)


def test_identity_initialization_online_encoding_and_visual_gradient_flow():
    backbone, prompts = backbone_and_prompts()
    images = torch.randn(3, 3, 4, 4)
    backbone.eval()
    with torch.no_grad():
        expected = backbone.logit_scale.exp() * F.normalize(
            backbone.get_image_features(pixel_values=images), dim=-1,
        ) @ F.normalize(backbone.get_text_features(**prompts), dim=-1).t()
    model = CLIPTransformerAdapter(backbone, prompts, ["a", "b", "c"])
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    calls = []
    hook = backbone.text_model.register_forward_hook(lambda *_: calls.append(1))
    torch.testing.assert_close(model(images), expected)
    model(images)
    hook.remove()
    assert len(calls) == 2  # No text feature precomputation/cache either.
    with pytest.raises(ValueError, match="raw"):
        model(torch.randn(3, 4))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.05)
    for step in range(2):
        optimizer.zero_grad()
        F.cross_entropy(model(images), torch.arange(3)).backward()
        for index in range(2):
            adapter = backbone.vision_model.encoder.layers[index].adapter
            assert adapter.up.weight.grad.norm() > 0
            assert bool(adapter.down.weight.grad.norm() > 0) == bool(step)
        optimizer.step()
    assert all(".adapter." in n for n, p in model.named_parameters() if p.requires_grad)
    for n, p in model.named_parameters():
        if n in frozen:
            assert p.grad is None
            torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
    assert trainable_scope_name(model) == "clip_visual_transformer_adapters"


def test_client_updates_are_isolated_and_server_parameters_are_restored():
    model = tiny_model()
    clients = [model.create_client_model(i) for i in range(2)]
    original = {n: p.clone() for n, p in model.export_trainable_state().items()}
    with clients[0].use_shared_model():
        optimizer = torch.optim.SGD(clients[0].parameters(), lr=.05)
        F.cross_entropy(clients[0](torch.randn(3, 3, 4, 4)), torch.arange(3)).backward()
        optimizer.step()
    assert model._active_client_model is None
    assert any(not torch.equal(clients[0].state_dict()[n], p) for n, p in original.items())
    for n, p in original.items():
        torch.testing.assert_close(model.export_trainable_state()[n], p, rtol=0, atol=0)
        torch.testing.assert_close(clients[1].state_dict()[n], p, rtol=0, atol=0)
    assert all(p.device.type == "cpu" and p.grad is None for p in clients[0].parameters())
    assert all(".adapter." in n for n in clients[0].state_dict())
    assert trainable_scope_name(clients[0]) == "clip_visual_transformer_adapters"


def test_legacy_checkpoints_are_not_silently_ignored_when_loading_partial_state():
    model = tiny_model()
    client = model.create_client_model(0)
    old_state = {"adapter.net.0.weight": torch.zeros(2, 4)}
    for target in [model, client]:
        with pytest.raises(ValueError, match="legacy feature Adapter checkpoint"):
            target.load_state_dict(old_state, strict=False)
        with pytest.raises(ValueError, match="legacy feature Adapter checkpoint"):
            target.load_trainable_state(old_state, strict=False)


def test_legacy_visual_adapter_block_overrides_new_inline_defaults():
    from main import default_config, normalize_clip_adapter_config
    config = default_config()
    config.update(model_type="visual_adapter", visual_adapter={
        "reduction": 4, "alpha": .2, "text_adapter_enabled": True,
        "precompute_features": True,
    })
    normalize_clip_adapter_config(config)
    assert config["model_type"] == "clip_adapter"
    assert config["clip_adapter"]["variant"] == "feature"
    assert config["clip_adapter"]["reduction"] == 4


@pytest.mark.parametrize("layer_index", [None, 0, 1])
def test_projres_uses_contextualized_tokens_and_total_token_count(layer_index):
    model = tiny_model()
    images = torch.randn(3, 3, 4, 4)
    parameter_name = (None if layer_index is None else
                      f"clip_model.vision_model.encoder.layers.{layer_index}.adapter.down.weight")
    name, layer = model.get_projres_attack_surface(parameter_name)
    expected_index = 1 if layer_index is None else layer_index
    assert name == f"clip_model.vision_model.encoder.layers.{expected_index}.adapter.down.weight"
    assert model.get_audit_key_parameter() is model.clip_model.vision_model.encoder.layers[0].adapter.down.weight
    captured = []
    hook = layer.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
    model(images)
    hook.remove()
    for reduction in ["auto", "mean", "cls"]:
        representations, count = model.get_projres_representations(
            images, parameter_name=parameter_name, token_reduction=reduction,
        )
        expected = captured[0].mean(dim=1) if reduction == "mean" else captured[0][:, 0]
        torch.testing.assert_close(representations, expected)
        assert count == 15 and model.training
        assert not torch.allclose(representations[0], representations[1])
    chunks = [model.get_projres_representations(x[None], parameter_name=parameter_name) for x in images]
    torch.testing.assert_close(torch.cat([x for x, _ in chunks]), captured[0][:, 0])
    assert sum(n for _, n in chunks) == 15


def test_last_adapter_down_gradient_is_reconstructed_by_cls_alone():
    model = tiny_model()
    images, labels = torch.randn(3, 3, 4, 4), torch.arange(3)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.05)
    # The first step moves zero-initialized up weights so down has a signal.
    F.cross_entropy(model(images), labels).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _, layer = model.get_projres_attack_surface()
    captured = []

    def capture(_module, args, output):
        output.retain_grad()
        captured.append((args[0].detach(), output))

    handle = layer.register_forward_hook(capture)
    try:
        F.cross_entropy(model(images), labels).backward()
    finally:
        handle.remove()
    hidden, output = captured[0]
    assert layer.weight.grad.norm() > 0
    assert torch.count_nonzero(output.grad[:, 1:]) == 0
    torch.testing.assert_close(layer.weight.grad, output.grad[:, 0].T @ hidden[:, 0])
    representations, _ = model.get_projres_representations(images)
    torch.testing.assert_close(representations, hidden[:, 0])


@pytest.mark.parametrize("method", ["fedsgd", "fedavg"])
@pytest.mark.parametrize("defense", ["none", "www", "cofedmid"])
def test_raw_image_federated_training_defenses_and_all_attacks(method, defense, tmp_path):
    model = tiny_model()
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    audit = _audit_config()
    audit.update(audit_batch_size=4, grad_sample_chunk_size=2)
    server = ServerBase(
        device=torch.device("cpu"), dataset_name="toy", model=model,
        train_sets=[dataset(2, 20), dataset(3, 21)],
        test_sets=[dataset(12, 30), dataset(13, 31)], class_names=["a", "b", "c"],
        batch_size=3, eval_batch_size=8, learning_rate=.05, num_glob_iters=2,
        local_epochs=1, total_users=2, results_dir=str(tmp_path), user_per_round=2,
        eval_interval=1,
        aggregator=build_aggregator(method, aggregation_weighting="sample_count" if method == "fedavg" else "uniform"),
        audit_config=audit, projres_config={"enabled": True, "evaluation_interval": 1},
        defense_config={"name": defense, "www_record_diagnostics": True,
                        "cofedmid_init_round": 1, "cofedmid_intervals": 1,
                        "cofedmid_reproducible_noise": True},
        method_config={"client_optimizer": "sgd", "seed": 42},
    )
    summaries = server.train()
    assert server.auditor.errors == {}
    assert {s["attack"] for s in summaries} == ATTACKS
    assert not server.auditor.candidate_inputs_are_features
    assert server.auditor.images.ndim == 4
    for result in summaries:
        assert result["metadata"]["adapter_variant"] == "transformer"
        assert result["metadata"]["trainable_scope"] == "clip_visual_transformer_adapters"
    projres = next(s for s in summaries if s["attack"] == "projres")
    meta = projres["metadata"]
    members = 6 if method == "fedavg" else 3
    assert projres["member_count"] == members
    assert projres["nonmember_count"] == (6 if method == "fedavg" else 30)
    assert meta["candidate_hidden_vector_count"] == members * 5
    assert meta["paper_fedsgd_exact"] is False
    assert meta["attacked_parameter"] == "clip_model.vision_model.encoder.layers.1.adapter.down.weight"
    assert meta["sample_representation"] == "cls_token_input_to_visual_transformer_adapter_down_projection"
    assert meta["representation_state"] == "client_post_update_model"
    assert meta["nonmember_to_member_ratio"] == (1 if method == "fedavg" else 10)
    if method == "fedavg" or meta["attacked_parameter_perturbed"]:
        assert meta["batch_rank_bound"] is None
    else:
        assert meta["batch_rank_bound"] == 15
    if method == "fedsgd" and defense in {"none", "www"}:
        assert server.auditor.exact_batch_skipped_rounds[0]["reason"] == "zero_observed_update"
    for n, p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
    assert (tmp_path / model.trainable_state_filename).exists()
    if defense == "www":
        assert (tmp_path / "www_diagnostics" / "batch_summary.csv").exists()
    if method == "fedavg":
        assert server.ctx.aggregation_weights == {0: .4, 1: .6}
        assert (tmp_path / "privacy_audit" / "client_train_update_candidate_selection.pt").exists()


def test_unified_defaults_disable_caches_and_cover_both_methods():
    tasks, skipped = build_tasks(load_yaml("configs/experiment_catalog.yaml"), parse_args([
        "--models", "clip_adapter", "--datasets", "cifar100", "--methods", "fedsgd,fedavg",
        "--defenses", "none,www,cofedmid", "--dry-run",
    ]))
    assert not skipped and len(tasks) == 6
    for task in tasks:
        validate_config(task.config)
        adapter = task.config["clip_adapter"]
        assert adapter["variant"] == "transformer" and adapter["reduction"] == 2
        assert adapter["precompute_features"] is False
        assert adapter["text_adapter_enabled"] is False
        assert task.config["learning_rate"] == .01
        projres = task.config["projres"]
        assert projres["attacked_parameter"] is None
        assert projres["token_reduction"] == "cls"
        assert task.config["audit"]["nonmember_to_member_ratio"] == 1
        assert task.config["audit"]["exact_batch_nonmember_to_member_ratio"] == 10
        assert tuple(projres[key] for key in ["max_candidates", "min_nonmembers", "max_nonmembers"]) == (
            (0, 0, 0) if task.config["aggregator"] == "fedavg" else (32, 320, 320)
        )


@pytest.mark.parametrize("override,match", [
    ({"precompute_features": True}, "precompute_features=false"),
    ({"text_adapter_enabled": True}, "text encoder"),
    ({"activation": "invalid"}, "activation"),
])
def test_incompatible_feature_options_are_rejected(override, match):
    config = load_yaml("configs/models/clip_adapter.yaml")
    config["clip_adapter"].update(override)
    with pytest.raises(ValueError, match=match):
        validate_config(config)


def test_main_constructs_online_adapter_and_preserves_fedavg_weighting(monkeypatch, tmp_path):
    import main
    import trainmodel.clip_feature_cache as cache

    backbone, prompts = backbone_and_prompts()
    class Processor:
        def __call__(self, **kwargs):
            if "text" in kwargs:
                return prompts
            return {"pixel_values": torch.stack(kwargs["images"])}

    def reject_precomputation(*args, **kwargs):
        raise AssertionError("The Transformer variant must not precompute features.")

    observed = {}
    class Server:
        def __init__(self, **kwargs):
            observed.update(kwargs)
        def train(self):
            model = observed["model"]
            images, labels = next(iter(torch.utils.data.DataLoader(
                observed["train_sets"][0], batch_size=3, collate_fn=observed["collate_fn"],
            )))
            F.cross_entropy(model(images), labels).backward()
            assert any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters() if p.requires_grad)
            return []

    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(main, "_load_local_clip", lambda *args, **kwargs: (Processor(), backbone))
    monkeypatch.setattr(main, "generate_iid_split", lambda *args, **kwargs: (
        [dataset(2, 20), dataset(3, 21)], [dataset(2, 30), dataset(3, 31)], ["a", "b", "c"],
    ))
    monkeypatch.setattr(cache, "precompute_federated_clip_features", reject_precomputation)
    monkeypatch.setattr(main, "ServerBase", Server)
    monkeypatch.setattr(main, "_build_logging_handlers", lambda *_: [])
    monkeypatch.setattr(main.logging, "basicConfig", lambda **_: None)
    config = load_yaml("configs/models/clip_adapter.yaml")
    config.update(aggregator="fedavg", aggregation_weighting="sample_count", total_users=2,
                  sample_users=2, num_global_iters=1, require_cuda=False, results_dir=str(tmp_path))
    config["audit"].update(enabled=False, attacks=[], exact_batch_membership_attacks=[])
    config["projres"]["enabled"] = False
    assert main.run(config) == []
    assert isinstance(observed["model"], CLIPTransformerAdapter)
    saved = load_yaml(next(tmp_path.rglob("run_config.yaml")))
    assert saved["aggregation_weighting"] == "sample_count"
    assert saved["clip_adapter"]["precompute_features"] is False
