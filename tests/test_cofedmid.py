from __future__ import annotations

import copy
import json
import math

import pytest
import torch
from torch.utils.data import TensorDataset

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.cofedmid import (
    CoFedMID, Exp3, assign_classes, perturb_uploads, reserve_validation,
    training_loss, validate_cofedmid,
)
from scripts.run_privacy_experiments import build_tasks, load_yaml, parse_args
from servers.serverbase import ServerBase


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("clients,classes,size", [(2, 100, 50), (4, 100, 50), (15, 100, 20), (30, 100, 50), (30, 2, 1)])
def test_class_assignment_has_equal_sizes_and_full_coverage(clients, classes, size):
    sets = assign_classes(classes, list(range(clients)), size, torch.Generator().manual_seed(253))
    assert all(len(values) == size for values in sets.values())
    assert len(set().union(*sets.values())) == classes
    if clients == 4:
        assert max(len(sets[i] & sets[j]) for i in sets for j in sets if i != j) == 17


def test_default_coalition_decay_and_missing_participant():
    controller = CoFedMID({}, 30, 100, 11, 42)
    assert controller.clients == list(range(30))
    controller.prepare(list(range(30)), 0)
    assert {len(x) for x in controller.assignments.values()} == {50}
    controller.prepare(list(range(30)), 10)
    assert {len(x) for x in controller.assignments.values()} == {20}
    with pytest.raises(ValueError, match="every coalition"):
        controller.prepare([0, 1], 11)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weights", [{0: 0.5, 1: 0.5}, {0: 0.2, 1: 0.3, 2: 0.5}])
def test_projected_noise_cancels_with_real_weights_and_one_global_tail(dtype, weights):
    states = {i: {"first": torch.arange(8, dtype=dtype) + i, "last": torch.zeros(2, dtype=dtype)} for i in weights}
    before = copy.deepcopy(states)
    result = perturb_uploads(states, weights, [0, 1], 0.3, 0.2, torch.Generator().manual_seed(3), gradient_scale=-20)
    assert result["mask"] == [{"parameter": "first", "start": 7, "count": 1}, {"parameter": "last", "start": 0, "count": 2}]
    for name in before[0]:
        expected = sum(weights[i] * before[i][name] for i in weights)
        actual = sum(weights[i] * states[i][name] for i in weights)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(before[0]["first"][:7], states[0]["first"][:7])
    assert not torch.equal(before[0]["last"], states[0]["last"])
    if 2 in weights:
        assert torch.equal(before[2]["last"], states[2]["last"])


def test_paper_recycled_loss_and_binary_kl_gradient():
    logits = torch.tensor([[0.3, -0.5], [1.2, 0.2]], requires_grad=True)
    labels = torch.tensor([0, 1])
    recycled = torch.tensor([False, True])
    loss = training_loss(logits, labels, recycled, 0.005)
    log_p = logits.log_softmax(1)
    expected = torch.nn.functional.cross_entropy(logits, labels) + 0.005 * (log_p[1].exp() * log_p[1]).sum() / 2
    actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_multiclass_soft_target_loss_value_and_detached_target_gradient():
    probabilities = torch.tensor([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    loss = training_loss(logits, torch.tensor([1, 0]), torch.tensor([False, True]), 0.005)
    entropy_term = (probabilities[1] * probabilities[1].log()).sum()
    expected_loss = (-math.log(0.5) - math.log(0.6) + 0.2 * math.log(4 / 3) + 0.005 * entropy_term) / 2
    gradient = probabilities.clone()
    gradient[0, 1] -= 1
    gradient[1, 0] -= 1
    gradient[1] += probabilities[1] - torch.tensor([0.6, 0.2, 0.2], dtype=torch.float64)
    gradient[1] += 0.005 * probabilities[1] * (probabilities[1].log() - entropy_term)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], gradient / 2)


def test_exp3_equal_losses_and_rewards_remain_finite_and_probability_updates():
    bandit = Exp3(3, 0.2, 0.3, 20)
    bandit.initialize(torch.ones(6))
    arm, probability, mask = bandit.choose(torch.ones(6), torch.Generator().manual_seed(7))
    assert mask.shape == (6,)
    for _ in range(3):
        assert bandit.update(0.5, arm, probability) == 0
    previous = bandit.probabilities.clone()
    bandit.update(1.0, 1, float(previous[1]))
    assert bandit.probabilities[1] > previous[1]
    assert torch.isfinite(bandit.probabilities).all()
    assert float(bandit.probabilities.sum()) == pytest.approx(1)


def test_validation_is_disjoint_and_controls_share_split():
    datasets = [TensorDataset(torch.arange(40).reshape(20, 2) + 100 * i, torch.arange(20) % 2) for i in range(3)]
    remaining, validation, manifest = reserve_validation(datasets, 0.1, 42)
    seen = set(validation.indices)
    offset = 0
    for original, kept in zip(datasets, remaining):
        assert not seen.intersection(i + offset for i in kept.indices)
        assert len(kept) + sum(offset <= i < offset + len(original) for i in seen) == len(original)
        offset += len(original)
    assert manifest == reserve_validation(datasets, 0.1, 42)[2]
    assert len(validation) == 6
    args = parse_args(["--models", "clip_mlp,bert_lora,gpt2_adapter", "--datasets", "default", "--defenses", "none,cofedmid", "--attacks", "none"])
    tasks, skipped = build_tasks(load_yaml("configs/experiment_catalog.yaml"), args)
    assert not skipped
    assert all(t.config["defense"]["cofedmid_validation_fraction"] == 0.1 for t in tasks)
    assert all(t.config["defense"]["cofedmid_clients"] == "all" for t in tasks if t.defense == "cofedmid")


@pytest.mark.parametrize("override", [{"cofedmid_clients": [0]}, {"cofedmid_clients": [0, 0]}, {"cofedmid_validation_fraction": 0}, {"cofedmid_noise_std": float("nan")}, {"cofedmid_intervals": 0}])
def test_invalid_protocols_are_rejected(override):
    with pytest.raises(ValueError):
        validate_cofedmid({"name": "cofedmid", **override}, 3, 3)


def test_single_client_and_partial_participation_are_rejected():
    with pytest.raises(ValueError, match="at least two"):
        validate_cofedmid({"name": "cofedmid"}, 1, 1)
    with pytest.raises(ValueError, match="full client participation"):
        validate_cofedmid({"name": "cofedmid"}, 3, 2)


def tiny_model(model_type, monkeypatch):
    from transformers import (
        BertConfig, BertModel, GPT2Config, GPT2Model, CLIPConfig,
        CLIPModel, CLIPTextConfig, CLIPVisionConfig,
    )
    if model_type.startswith("clip"):
        from trainmodel.clip_mlp import CLIPImageMLP
        from trainmodel.clip_adapter import CLIPAdapter
        from trainmodel.clip_lora import CLIPLoRA
        text = CLIPTextConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8, eos_token_id=2, bos_token_id=1, pad_token_id=0)
        vision = CLIPVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, image_size=4, patch_size=2)
        clip = CLIPModel(CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=4))
        if model_type == "clip_mlp":
            return CLIPImageMLP(clip_model=clip, num_classes=3, hidden_dim=4)
        if model_type == "clip_adapter":
            return CLIPAdapter(clip_model=clip, text_features=torch.eye(3, 4), classnames=["0", "1", "2"], reduction=2, output_relu=False)
        return CLIPLoRA(clip_model=clip, text_inputs={"input_ids": torch.tensor([[1, 3, 2], [1, 4, 2], [1, 5, 2]]), "attention_mask": torch.ones(3, 3, dtype=torch.long)}, classnames=["0", "1", "2"], encoder="vision", target_modules=["q"], rank=2, dropout=0)
    from trainmodel.transformer_adapter import TransformerAdapterClassifier
    from trainmodel.transformer_lora import TransformerLoRAClassifier
    if model_type == "gpt2_adapter":
        backbone = GPT2Model(GPT2Config(vocab_size=32, n_positions=16, n_embd=8, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    else:
        backbone = BertModel(BertConfig(vocab_size=32, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, hidden_dropout_prob=0, attention_probs_dropout_prob=0))
    monkeypatch.setattr("trainmodel.transformer_adapter.AutoModel.from_pretrained", lambda *a, **k: backbone)
    if model_type == "bert_lora":
        return TransformerLoRAClassifier("unused", num_classes=3, rank=2, dropout=0, classifier_dropout=0)
    return TransformerAdapterClassifier("unused", architecture="gpt2" if model_type == "gpt2_adapter" else "bert", num_classes=3)


def toy_dataset(model_type, per_class, seed):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(3 * per_class) % 3
    if model_type == "clip_lora":
        inputs = torch.randn(len(labels), 3, 4, 4, generator=generator)
    elif model_type.startswith("clip"):
        inputs = torch.randn(len(labels), 4, generator=generator)
    else:
        tokens = torch.randint(1, 32, (len(labels), 4), generator=generator)
        inputs = torch.stack((tokens, torch.ones_like(tokens)), dim=1)
    return TensorDataset(inputs, labels)


@pytest.mark.parametrize("model_type", ["clip_mlp", "clip_adapter", "clip_lora", "bert_adapter", "bert_lora", "gpt2_adapter"])
def test_all_six_peft_models_train_recycle_and_audit_defended_uploads(model_type, monkeypatch, tmp_path):
    torch.manual_seed(42)
    model = tiny_model(model_type, monkeypatch)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    catalog = load_yaml("configs/experiment_catalog.yaml")
    attacks = catalog["attacks"]["all"]
    messages = {}

    def observer(**kwargs):
        messages[kwargs["round_index"], kwargs["client_id"]] = {n: v.clone() for n, v in kwargs["gradients"].items()}

    server = ServerBase(
        device=torch.device("cpu"), dataset_name="toy", model=model,
        train_sets=[toy_dataset(model_type, 4, 10 + i) for i in range(2)],
        test_sets=[toy_dataset(model_type, 24, 20 + i) for i in range(2)],
        class_names=["0", "1", "2"], batch_size=2, eval_batch_size=32,
        learning_rate=0.05, num_glob_iters=3, local_epochs=1, total_users=2,
        results_dir=str(tmp_path), user_per_round=2,
        aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"),
        eval_interval=3,
        audit_config={"enabled": True, "strict": True, "attacks": attacks, "candidate_sampling": "balanced_global_holdout", "require_full_target_train_members": True, "nonmember_to_member_ratio": 1, "exact_batch_membership_attacks": catalog["attacks"]["exact_batch"], "exact_batch_nonmember_to_member_ratio": 10, "paper_balanced_evaluation_size": 0, "audit_batch_size": 32, "attack_audit_intervals": {a: (2 if a == "projres" else 1) for a in attacks}, "low_fpr_min_nonmembers": 2, "training_health_check": False, "seed": 42},
        projres_config={"enabled": True, "evaluation_interval": 2, "token_reduction": "mean", "max_candidates": 2, "min_nonmembers": 20, "max_nonmembers": 20, "threshold": None, "decision_mode": "ranking"},
        defense_config={"name": "cofedmid", "cofedmid_init_round": 1, "cofedmid_intervals": 1, "cofedmid_recycle_ratio": 0.5, "cofedmid_reproducible_noise": True},
        method_config={"client_optimizer": "sgd", "momentum": 0, "weight_decay": 0, "max_grad_norm": 0, "seed": 42},
        client_gradient_observer=observer,
    )
    summaries = server.train()
    assert server.auditor.errors == {}
    assert {s["attack"] for s in summaries} == set(attacks)
    assert server.defense.steps == {0: 3, 1: 3}
    assert server.defense.cofedmid.clients == [0, 1]
    assert int(server.auditor.membership.sum()) == server.ctx.users[0].train_samples
    assert any(r["recycled_pool"] > 0 for r in server.defense.cofedmid.rows)
    assert all(r["recycled_pool"] == 0 for r in server.defense.cofedmid.rows if r["communication_round"] == 1)
    for user in server.ctx.users:
        assert user.last_gradient_capture_count == 1
        assert user.last_update_sample_count == 2
        for name, tensor in server.ctx.protocol_messages[user.id]["tensors"].items():
            torch.testing.assert_close(tensor, messages[2, user.id][name], rtol=0, atol=0)
        assert user.last_train_indices.numel() == 2
    for selection in server.auditor.exact_batch_candidate_selections:
        assert selection["member_local_indices"].min() >= 0
        assert len(selection["member_recycled"]) == 2
        assert selection["nonmember_label_histogram"] == [x * 10 for x in selection["member_label_histogram"]]
    final_selection = server.auditor.exact_batch_candidate_selections[-1]
    assert torch.equal(final_selection["member_local_indices"], server.ctx.users[0].last_train_indices)
    assert server.defense.cofedmid.summary()["max_weighted_noise_residual"] < 1e-5
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    assert (tmp_path / "defense_validation_split.json").exists()
    assert (tmp_path / "cofedmid_sample_exposure.pt").exists()
    audit = json.loads((tmp_path / "privacy_audit" / "summary.json").read_text())
    assert audit["cofedmid"]["upload_perturbed"]
    assert audit["defense_validation_split_sha256"] == server.defense_validation_manifest["split_sha256"]


def small_server(model, tmp_path, defense, *, rounds=12, attacks=(), users=2, per_class=40):
    return ServerBase(
        device=torch.device("cpu"), dataset_name="toy", model=model,
        train_sets=[toy_dataset("clip_mlp", per_class, 10 + i) for i in range(users)],
        test_sets=[toy_dataset("clip_mlp", 24, 20 + i) for i in range(users)],
        class_names=["0", "1", "2"], batch_size=4, eval_batch_size=32,
        learning_rate=0.05, num_glob_iters=rounds, local_epochs=1,
        total_users=users, results_dir=str(tmp_path), user_per_round=users,
        aggregator=build_aggregator("fedsgd", aggregation_weighting="uniform"),
        eval_interval=rounds,
        audit_config={
            "enabled": bool(attacks), "strict": True, "attacks": list(attacks),
            "candidate_sampling": "balanced_global_holdout",
            "require_full_target_train_members": True,
            "exact_batch_membership_attacks": list(attacks),
            "exact_batch_nonmember_to_member_ratio": 10,
            "attack_audit_intervals": {a: 1 for a in attacks},
            "paper_balanced_evaluation_size": 0,
            "low_fpr_min_nonmembers": 2,
            "training_health_check": False,
        },
        projres_config={"enabled": "projres" in attacks, "decision_mode": "ranking"},
        defense_config={"name": "cofedmid", **defense},
    )


def test_default_warmup_and_noise_preserve_actual_server_aggregation(monkeypatch, tmp_path):
    torch.manual_seed(7)
    initial = tiny_model("clip_mlp", monkeypatch)
    servers = []
    for perturbation in (False, True):
        server = small_server(
            copy.deepcopy(initial), tmp_path / str(perturbation),
            {"cofedmid_perturbation": perturbation},
        )
        server.train()
        servers.append(server)
        engine = server.defense.cofedmid
        assert all(row["arm"] is None for row in engine.rows if row["communication_round"] <= 10)
        assert all(row["arm"] is not None for row in engine.rows if row["communication_round"] > 10)
        assert all((counts == 2).all() for counts in engine.scored_exposures.values())
        assert any(row["recycled_pool"] > 0 for row in engine.rows)
        assert sum(counts.sum() for counts in engine.exposures.values()) == 2 * 12 * 4
    clean, noisy = servers
    for name, expected in clean.ctx.new_model_state[0].items():
        torch.testing.assert_close(noisy.ctx.new_model_state[0][name], expected, atol=1e-6, rtol=1e-5)
    assert any(
        not torch.allclose(clean.ctx.protocol_messages[0]["tensors"][name], tensor)
        for name, tensor in noisy.ctx.protocol_messages[0]["tensors"].items()
    )


def test_explicit_coalition_leaves_other_client_unchanged(monkeypatch, tmp_path):
    torch.manual_seed(7)
    initial = tiny_model("clip_mlp", monkeypatch)
    outputs = []
    for perturbation in (False, True):
        torch.manual_seed(19)
        server = small_server(
            copy.deepcopy(initial), tmp_path / str(perturbation),
            {"cofedmid_clients": [0, 1], "cofedmid_perturbation": perturbation},
            rounds=1, users=3,
        )
        server.train()
        outputs.append(server.ctx.protocol_messages[2]["tensors"])
        assert set(server.defense.cofedmid.exposures) == {0, 1}
        assert server.ctx.users[2].last_train_recycled is None
    for name in outputs[0]:
        torch.testing.assert_close(outputs[0][name], outputs[1][name], atol=0, rtol=0)


def test_projres_does_not_apply_clean_rank_bound_to_noisy_parameter(monkeypatch, tmp_path):
    torch.manual_seed(7)
    server = small_server(
        tiny_model("clip_mlp", monkeypatch), tmp_path,
        {"cofedmid_perturb_ratio": 1.0}, rounds=1, attacks=["projres"],
    )
    import privacy_attacks.auditor as auditor_module
    original = auditor_module.strict_mlp_projres
    bounds = []

    def capture(*args, **kwargs):
        bounds.append(kwargs["max_rank"])
        return original(*args, **kwargs)

    monkeypatch.setattr(auditor_module, "strict_mlp_projres", capture)
    summaries = server.train()
    assert bounds == [None]
    assert summaries[0]["attack"] == "projres"


def test_single_sample_pool_keeps_exact_one_to_ten_audit(monkeypatch, tmp_path):
    torch.manual_seed(7)
    attacks = load_yaml("configs/experiment_catalog.yaml")["attacks"]["exact_batch"]
    server = small_server(
        tiny_model("clip_mlp", monkeypatch), tmp_path,
        {"cofedmid_max_classes": 1, "cofedmid_min_classes": 1},
        rounds=1, attacks=attacks, users=3, per_class=1,
    )
    summaries = server.train()
    assert {result["attack"] for result in summaries} == set(attacks)
    assert all(user.last_update_sample_count == 1 for user in server.ctx.users)
    selection = server.auditor.exact_batch_candidate_selections[0]
    assert selection["member_recycled"].shape == (1,)
    assert sum(selection["member_label_histogram"]) == 1
    assert sum(selection["nonmember_label_histogram"]) == 10
