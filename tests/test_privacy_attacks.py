import json
import copy
import itertools
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from aggregator.aggregator_builder import build_aggregator
from aggregator.fedavg_aggregator import aggregate_fedavg_model_states
from main import default_config, validate_config
from privacy_attacks.code_poison import generate_membership_encoding_samples
from privacy_attacks.auditor import MembershipAuditor
from privacy_attacks.fedmia import run_fedmia
from privacy_attacks.metrics import fit_shrinkage_attack, membership_metrics
from privacy_attacks.model_utils import scaled_confidence
from privacy_attacks.promptmia import generate_key_with_similarity
from privacy_defenses import (
    DefenseController,
    attach_hamp_output_transform,
    attach_output_temperature_transform,
)
from servers.serverbase import ServerBase
from utils.privacy_accounting import (
    calibrate_gaussian_noise,
    gaussian_rdp_epsilon,
    planned_private_probe_steps,
)


class ToyPromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = nn.Parameter(
            torch.tensor([[0.2, -0.2, 0.1, 0.3], [-0.1, 0.2, -0.2, 0.1]])
        )

    def forward(self, images, return_intermediate=False):
        signal = images.mean(dim=(1, 2, 3))
        image_features = torch.stack(
            (signal, -signal, signal.square(), torch.ones_like(signal)), dim=1
        )
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(self.prompt, dim=1)
        logits = image_features @ text_features.t()
        if return_intermediate:
            return logits, image_features, text_features
        return logits


def _dataset(offset: float = 0.0):
    images = torch.linspace(-1 + offset, 1 + offset, steps=8 * 3 * 4 * 4).reshape(
        8, 3, 4, 4
    )
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    return TensorDataset(images, labels)


def test_fedavg_uses_sample_weights_and_selected_clients_only():
    ctx = SimpleNamespace(
        samples_num=[1, 3, 100],
        trainable_param_names=["prompt"],
        updated_model_state={
            0: {"prompt": torch.tensor([1.0])},
            1: {"prompt": torch.tensor([3.0])},
            2: {"prompt": torch.tensor([99.0])},
        },
        new_model_state={},
    )
    aggregate_fedavg_model_states(ctx, [0, 1])
    assert torch.allclose(ctx.new_model_state[0]["prompt"], torch.tensor([2.5]))
    assert build_aggregator("fedavg").name == "fedavg"


def test_membership_metrics_and_secret_samples_are_deterministic():
    labels = torch.tensor([1, 1, 0, 0])
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    metrics = membership_metrics(labels, scores)
    assert metrics["tpr_at_fpr_0.01"] == 1.0
    assert metrics["auc"] == 1.0

    images = torch.randn(3, 3, 4, 4)
    first = generate_membership_encoding_samples(images)
    second = generate_membership_encoding_samples(images)
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_cofedmid_class_partition_balances_overlap_and_coverage():
    controller = DefenseController(
        {"name": "cofedmid", "seed": 42},
        device=torch.device("cpu"),
        total_users=4,
        num_classes=100,
        total_rounds=10,
        samples_num=[1, 1, 1, 1],
    )
    controller.prepare_round([0, 1, 2, 3], round_index=0)
    class_sets = [controller._cofedmid_classes(client, 0) for client in range(4)]
    assert [len(class_set) for class_set in class_sets] == [50, 50, 50, 50]
    assert len(set().union(*class_sets)) == 100
    assert (
        max(
            len(left & right)
            for left, right in itertools.combinations(class_sets, 2)
        )
        == 17
    )


def test_small_sample_shrinkage_attack_ignores_distracting_dimensions():
    labels = torch.tensor([1] * 8 + [0] * 8)
    generator = torch.Generator().manual_seed(11)
    distractors = torch.randn(16, 64, generator=generator)
    signal = torch.cat((torch.ones(8), -torch.ones(8))).unsqueeze(1)
    features = torch.cat((signal, distractors), dim=1)
    scores, held_out, _, selected = fit_shrinkage_attack(
        features, labels, calibration_fraction=0.5, seed=7, max_features=1
    )
    assert selected == 1
    assert membership_metrics(held_out, scores)["auc"] == 1.0


def test_fedmia_can_use_max_round_aggregation():
    membership = torch.tensor([1, 1, 0, 0])
    observations = [
        {
            "round": 0,
            "client_ids": torch.tensor([0, 1, 2]),
            "confidence": torch.tensor(
                [
                    [0.2, 0.1, 0.8, 0.7],
                    [0.0, 0.0, 0.9, 0.8],
                    [0.0, 0.0, 0.85, 0.75],
                ]
            ),
        },
        {
            "round": 1,
            "client_ids": torch.tensor([0, 1, 2]),
            "confidence": torch.tensor(
                [
                    [2.0, 2.1, 0.1, 0.2],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        },
    ]
    mean_result = run_fedmia(observations, membership, 0, "confidence", "mean")
    max_result = run_fedmia(observations, membership, 0, "confidence", "max")
    assert max_result.metadata["round_aggregation"] == "max"
    assert torch.all(max_result.scores >= mean_result.scores)
    assert torch.any(max_result.scores > mean_result.scores)


def test_private_noise_calibration_meets_requested_budget():
    multiplier = calibrate_gaussian_noise(
        target_epsilon=3.0,
        steps=20,
        delta=1e-5,
        mechanisms_per_step=2,
    )
    epsilon = gaussian_rdp_epsilon(
        multiplier, steps=20, delta=1e-5, mechanisms_per_step=2
    )
    assert epsilon <= 3.0 + 1e-8
    assert gaussian_rdp_epsilon(multiplier * 0.9, 20, 1e-5, 2) > 3.0


def test_active_client_update_queries_are_counted_in_privacy_budget():
    assert (
        planned_private_probe_steps(
            {
                "enabled": True,
                "attacks": ["nasr_active", "promptmia", "fedmia_loss"],
                "active_max_samples": 6,
                "active_probe_cycles": 3,
                "promptmia_max_samples": 4,
            }
        )
        == 22
    )
    assert planned_private_probe_steps({"enabled": False}) == 0


def test_recent_attack_primitives_match_paper_definitions():
    query = torch.tensor([1.0, 2.0, -1.0, 0.5])
    key = generate_key_with_similarity(query, 0.73, torch.Generator().manual_seed(7))
    cosine = F.cosine_similarity(query, key, dim=0)
    assert torch.allclose(cosine, torch.tensor(0.73), atol=1e-5)
    assert torch.allclose(query.norm(), key.norm(), atol=1e-5)

    probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    scores = scaled_confidence(probabilities, torch.tensor([0, 1]))
    expected = torch.log(torch.tensor([0.7 / 0.2, 0.6 / 0.3]))
    assert torch.allclose(scores, expected)


def test_configuration_rejects_removed_backdoor_and_defense_names():
    config = default_config()
    config["audit"]["attacks"] = ["a3fl"]
    try:
        validate_config(config)
    except ValueError as error:
        assert "Unknown membership attacks" in str(error)
    else:
        raise AssertionError("Removed backdoor attack unexpectedly accepted")

    config = default_config()
    config["audit"]["attacks"] = ["fedmia_joint"]
    try:
        validate_config(config)
    except ValueError as error:
        assert "Unknown membership attacks" in str(error)
    else:
        raise AssertionError("Unpublished composite attack unexpectedly accepted")

    config = default_config()
    config["aggregator"] = "seismograph"
    try:
        validate_config(config)
    except ValueError as error:
        assert "fedavg" in str(error)
    else:
        raise AssertionError("Removed defense unexpectedly accepted")

    config = default_config()
    config["defense"]["name"] = "seismograph"
    try:
        validate_config(config)
    except ValueError as error:
        assert "Unknown privacy defense" in str(error)
    else:
        raise AssertionError("Removed SEISMOGRAPH defense unexpectedly accepted")


class ToyPromptLearner(nn.Module):
    def __init__(self, method: str):
        super().__init__()
        self.method = method
        initial = torch.tensor([[0.2, -0.2, 0.1, 0.3], [-0.1, 0.2, -0.2, 0.1]])
        if method == "dpfpl":
            self.global_ctx = nn.Parameter(initial.clone())
            self.local_ctx = nn.Parameter(torch.zeros_like(initial))
        else:
            self.register_buffer("base_ctx", initial.clone())
            self.fedask_A = nn.Parameter(torch.randn(2, 4) * 0.02)
            self.fedask_B = nn.Parameter(torch.zeros(2, 2))

    def effective_context(self):
        if self.method == "dpfpl":
            return self.global_ctx + self.local_ctx
        return self.base_ctx + self.fedask_B @ self.fedask_A


class ToyPrivateMethodModel(nn.Module):
    def __init__(self, method: str):
        super().__init__()
        self.prompt_learner = ToyPromptLearner(method)

    def forward_with_context(self, images, context, return_intermediate=False):
        signal = images.mean(dim=(1, 2, 3))
        image_features = F.normalize(
            torch.stack(
                (signal, -signal, signal.square(), torch.ones_like(signal)), dim=1
            ),
            dim=1,
        )
        text_features = F.normalize(context, dim=1)
        logits = image_features @ text_features.t()
        if return_intermediate:
            return logits, image_features, text_features
        return logits

    def forward(self, images, return_intermediate=False):
        return self.forward_with_context(
            images,
            self.prompt_learner.effective_context(),
            return_intermediate=return_intermediate,
        )


def test_defense_primitives_preserve_required_invariants():
    model = ToyPromptModel()
    images, _ = next(iter(torch.utils.data.DataLoader(_dataset(), batch_size=8)))
    original_logits = model(images)
    attach_hamp_output_transform(model, temperature=4.0)
    model.eval()
    defended_logits = model(images)
    assert torch.equal(original_logits.argmax(dim=1), defended_logits.argmax(dim=1))
    assert (
        torch.softmax(defended_logits, dim=1).amax(dim=1).mean()
        < torch.softmax(original_logits, dim=1).amax(dim=1).mean()
    )

    local_model = ToyPromptModel()
    attach_output_temperature_transform(local_model, temperature=3.0)
    local_model.eval()
    local_logits = local_model(images)
    assert torch.equal(original_logits.argmax(dim=1), local_logits.argmax(dim=1))
    assert (
        torch.softmax(local_logits, dim=1).amax(dim=1).mean()
        < torch.softmax(original_logits, dim=1).amax(dim=1).mean()
    )

    capped_model = ToyPromptModel()
    attach_output_temperature_transform(capped_model, temperature=1.0, margin=0.01)
    capped_model.eval()
    capped_logits = capped_model(images)
    assert torch.equal(original_logits.argmax(dim=1), capped_logits.argmax(dim=1))
    assert (
        torch.softmax(capped_logits, dim=1).amax(dim=1).mean()
        < torch.softmax(original_logits, dim=1).amax(dim=1).mean()
    )

    controller = DefenseController(
        {
            "name": "cofedmid",
            "cofedmid_noise_std": 0.2,
            "cofedmid_perturb_ratio": 1.0,
            "seed": 7,
        },
        device=torch.device("cpu"),
        total_users=2,
        num_classes=2,
        total_rounds=2,
        samples_num=[1, 3],
    )
    states = {
        0: {"prompt": torch.zeros(4)},
        1: {"prompt": torch.zeros(4)},
    }
    controller._cofedmid_perturb(states, [0, 1], round_index=0)
    weighted = 0.25 * states[0]["prompt"] + 0.75 * states[1]["prompt"]
    assert torch.allclose(weighted, torch.zeros_like(weighted), atol=1e-6)


def _defended_server(
    tmp_path: Path,
    defense: str,
    attack_enabled: bool = True,
    attack: str = "fedmia_loss",
):
    train_sets = [_dataset(index * 0.02) for index in range(4)]
    test_sets = [_dataset(0.4 + index * 0.02) for index in range(4)]
    return ServerBase(
        train_mode="centralized",
        device=torch.device("cpu"),
        dataset_name="toy",
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=["zero", "one"],
        model=ToyPromptModel(),
        batch_size=4,
        eval_batch_size=8,
        learning_rate=0.05,
        num_glob_iters=2,
        local_epochs=1,
        total_users=4,
        results_dir=str(tmp_path),
        user_per_round=4,
        aggregator=build_aggregator("fedavg"),
        eval_interval=1,
        audit_config={
            "enabled": attack_enabled,
            "strict": True,
            "target_client_id": 0,
            "ensure_target_participation": True,
            "attacks": [attack] if attack_enabled else [],
            "max_samples_per_group": 4,
            "audit_interval": 1,
        },
        defense_config={
            "name": defense,
            "seed": 13,
            "dp_max_grad_norm": 0.5,
            "dp_noise_multiplier": 0.2,
            "dp_delta": 1e-5,
            "cofedmid_intervals": 2,
            "cofedmid_recycle_ratio": 0.25,
            "cofedmid_noise_std": 0.02,
            "cofedmid_perturb_ratio": 0.5,
            "mist_cross_steps": 1,
            "soft_obfuscation_strength": 0.5,
            "soft_noise_std": 0.01,
            "hamp_true_probability": 0.6,
            "hamp_entropy_weight": 0.05,
            "hamp_output_temperature": 3.0,
            "local_ggeur_augments": 1,
            "local_ggeur_geometry_scale": 0.2,
            "local_ggeur_anchor_mode": "class_mean",
            "local_ggeur_original_mode": "class_mean_noise",
            "local_ggeur_original_noise": 0.01,
            "local_ggeur_mean_mix": 0.8,
            "local_ggeur_fallback_std": 0.01,
            "local_ggeur_output_temperature": 3.0,
        },
    )


def test_each_defense_runs_independently_with_one_attack():
    root = Path(".test_artifacts") / "defenses"
    for defense in (
        "cofedmid",
        "prompt_dp",
        "mist",
        "soft",
        "hamp",
        "local_ggeur",
        "mirage",
        "veil",
    ):
        path = root / defense
        path.mkdir(parents=True, exist_ok=True)
        summaries = _defended_server(path, defense).train()
        assert [item["attack"] for item in summaries] == ["fedmia_loss"]
        summary = json.loads(
            (path / "defense_summary.json").read_text(encoding="utf-8")
        )
        assert summary["defense"] == defense
        if defense in {"local_ggeur", "mirage", "veil"}:
            assert summary["metrics"]["local_ggeur_private_feature_count"] > 0
        assert (path / "privacy_audit" / "summary.json").exists()


def test_local_ggeur_mean_noise_is_local_deterministic_and_preserves_covariance():
    features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
    )
    labels = torch.tensor([0, 0, 1, 1])

    def geometry(noise):
        controller = DefenseController(
            {"name": "local_ggeur", "seed": 17},
            torch.device("cpu"),
            total_users=2,
            num_classes=2,
            total_rounds=1,
        )
        return controller._local_geometry(
            features,
            labels,
            generator=controller._generator(0, 0, 877),
            mean_noise_std=noise,
        )

    clean = geometry(0.0)
    first = geometry(0.03)
    second = geometry(0.03)
    for class_id in (0, 1):
        assert torch.allclose(first[class_id][0], second[class_id][0])
        assert not torch.allclose(first[class_id][0], clean[class_id][0])
        assert torch.allclose(first[class_id][1], clean[class_id][1])


def test_membership_candidates_can_be_exactly_label_matched():
    auditor = MembershipAuditor.__new__(MembershipAuditor)
    auditor.model = SimpleNamespace(classnames=["zero", "one", "two"])
    auditor.seed = 23
    auditor.collate_fn = None
    member_labels = torch.tensor([0, 0, 1, 2, 2, 2])
    first = TensorDataset(
        torch.arange(5, dtype=torch.float32).view(5, 1),
        torch.tensor([2, 1, 2, 0, 1]),
    )
    second = TensorDataset(
        torch.arange(5, 10, dtype=torch.float32).view(5, 1),
        torch.tensor([0, 2, 0, 2, 1]),
    )
    _images, labels = auditor._collect_label_matched(
        [first, second], member_labels
    )
    assert torch.equal(
        torch.bincount(labels, minlength=3),
        torch.bincount(member_labels, minlength=3),
    )


def test_membership_candidate_loading_does_not_advance_global_torch_rng():
    auditor = MembershipAuditor.__new__(MembershipAuditor)
    auditor.seed = 23
    auditor.collate_fn = None
    dataset = TensorDataset(torch.arange(8).view(8, 1), torch.arange(8) % 2)
    torch.manual_seed(101)
    before = torch.get_rng_state().clone()
    auditor._collect_many([dataset], 6)
    assert torch.equal(torch.get_rng_state(), before)


def test_defense_only_mode_skips_attack_audit():
    path = Path(".test_artifacts") / "defense_only"
    path.mkdir(parents=True, exist_ok=True)
    summaries = _defended_server(path, "hamp", attack_enabled=False).train()
    assert summaries == []
    assert (path / "defense_summary.json").exists()
    assert not (path / "privacy_audit" / "summary.json").exists()


def test_dpfpl_and_fedask_run_as_fedavg_replacements():
    root = Path(".test_artifacts") / "private_methods"
    for method, mode in (("dpfpl", "local"), ("fedask", "centralized")):
        path = root / method
        path.mkdir(parents=True, exist_ok=True)
        method_config = {
            "rank": 2,
            "seed": 17,
            "local_clip_norm": 0.5,
            "local_noise_multiplier": 0.1,
            "global_clip_norm": 0.5,
            "global_noise_multiplier": 0.1,
            "clip_norm": 0.5,
            "noise_multiplier": 0.1,
            "oversampling": 1,
            "local_steps": 1,
            "reproducible_dp_noise": True,
        }
        server = ServerBase(
            train_mode=mode,
            device=torch.device("cpu"),
            dataset_name="toy",
            train_sets=[_dataset(index * 0.02) for index in range(3)],
            test_sets=[_dataset(0.4 + index * 0.02) for index in range(3)],
            class_names=["zero", "one"],
            model=ToyPrivateMethodModel(method),
            batch_size=4,
            eval_batch_size=8,
            learning_rate=0.05,
            num_glob_iters=2,
            local_epochs=1,
            total_users=3,
            results_dir=str(path),
            user_per_round=3,
            aggregator=build_aggregator(method, **method_config),
            eval_interval=1,
            audit_config={
                "enabled": True,
                "strict": True,
                "target_client_id": 0,
                "ensure_target_participation": True,
                "attacks": ["fedmia_loss"],
                "max_samples_per_group": 4,
                "audit_interval": 1,
            },
            defense_config={"name": "hamp", "hamp_output_temperature": 2.0},
            method_config=method_config,
        )
        summaries = server.train()
        assert [item["attack"] for item in summaries] == ["fedmia_loss"]
        summary = json.loads(
            (path / "federated_method_summary.json").read_text(encoding="utf-8")
        )
        assert summary["federated_method"] == method
        assert (
            summary["privacy_accounting"][
                (
                    "local_epsilon_upper_bound"
                    if method == "dpfpl"
                    else "epsilon_upper_bound"
                )
            ]
            is not None
        )
        messages = server.ctx.protocol_messages
        observation_states = server.auditor.observations[-1]["client_states"]
        if method == "dpfpl":
            assert all(
                message["kind"] == "global_prompt_gradient"
                and all(name.endswith("global_ctx") for name in message["tensors"])
                for message in messages.values()
            )
            local_name = next(
                name
                for name in server.auditor.initial_prompt_state
                if name.endswith("local_ctx")
            )
            assert all(
                torch.equal(
                    state[local_name],
                    server.auditor.initial_prompt_state[local_name].cpu(),
                )
                for state in observation_states.values()
            )
        else:
            assert all(
                message["kind"] == "fedask_sketch"
                and set(message["tensors"]) == {"stage1_y", "stage2_p"}
                for message in messages.values()
            )
            released = server.ctx.new_model_state[0]
            assert all(
                all(torch.equal(state[name], released[name].cpu()) for name in released)
                for state in observation_states.values()
            )
        if method == "dpfpl":
            global_name = next(
                name
                for name in server.ctx.new_model_state[0]
                if name.endswith("global_ctx")
            )
            local_name = next(
                name
                for name in server.ctx.new_model_state[0]
                if name.endswith("local_ctx")
            )
            assert torch.equal(
                server.ctx.new_model_state[0][global_name],
                server.ctx.new_model_state[1][global_name],
            )
            assert not torch.equal(
                server.ctx.new_model_state[0][local_name],
                server.ctx.new_model_state[1][local_name],
            )
        else:
            assert summary["last_reconstruction_error"] < 1e-4
            assert "last_pretruncation_error" in summary


def test_target_may_be_absent_when_participation_is_not_forced():
    path = Path(".test_artifacts") / "target_absent"
    path.mkdir(parents=True, exist_ok=True)
    server = _defended_server(path, "none", attack_enabled=False)
    server.ensure_target = False
    server._sample_users = lambda: [1, 2, 3]
    assert server.train() == []


def test_every_defense_uses_private_method_pipeline_without_duplicate_epochs():
    root = Path(".test_artifacts") / "private_method_defenses"
    defense_config = {
        "seed": 23,
        "reproducible_dp_noise": True,
        "dp_max_grad_norm": 0.5,
        "dp_noise_multiplier": 0.2,
        "dp_delta": 1e-5,
        "cofedmid_intervals": 2,
        "cofedmid_recycle_ratio": 0.25,
        "cofedmid_noise_std": 0.01,
        "cofedmid_perturb_ratio": 0.5,
        "mist_cross_steps": 1,
        "mist_cross_weight": 1.0,
        "soft_obfuscation_strength": 0.5,
        "soft_noise_std": 0.01,
        "hamp_true_probability": 0.6,
        "hamp_entropy_weight": 0.05,
        "hamp_output_temperature": 2.0,
    }
    for method, mode in (("dpfpl", "local"), ("fedask", "centralized")):
        for defense in ("cofedmid", "prompt_dp", "mist", "soft", "hamp"):
            path = root / method / defense
            path.mkdir(parents=True, exist_ok=True)
            method_config = {
                "rank": 2,
                "local_steps": 1,
                "seed": 29,
                "reproducible_dp_noise": True,
                "local_clip_norm": 0.5,
                "local_noise_multiplier": 0.2,
                "global_clip_norm": 0.5,
                "global_noise_multiplier": 0.2,
                "clip_norm": 0.5,
                "noise_multiplier": 0.2,
                "oversampling": 1,
                "delta": 1e-5,
            }
            selected_defense = defense_config | {"name": defense}
            server = ServerBase(
                train_mode=mode,
                device=torch.device("cpu"),
                dataset_name="toy",
                train_sets=[_dataset(index * 0.02) for index in range(3)],
                test_sets=[_dataset(0.4 + index * 0.02) for index in range(3)],
                class_names=["zero", "one"],
                model=ToyPrivateMethodModel(method),
                batch_size=8,
                eval_batch_size=8,
                learning_rate=0.05,
                num_glob_iters=1,
                local_epochs=1,
                total_users=3,
                results_dir=str(path),
                user_per_round=3,
                aggregator=build_aggregator(method, **method_config),
                eval_interval=1,
                audit_config={
                    "enabled": False,
                    "attacks": [],
                    "target_client_id": 0,
                },
                defense_config=selected_defense,
                method_config=method_config,
            )
            assert server.train() == []
            expected_steps = 2 if defense == "mist" else 1
            assert set(server.defense.steps.values()) == {expected_steps}
            if defense == "cofedmid":
                assert set(server.defense.cofedmid_selected_arm) == {0, 1, 2}
            if defense == "soft":
                assert (
                    server.defense.summary()["metrics"]["soft_selected_fraction"] == 0.0
                )


def test_fedask_active_probe_keeps_a_fixed_and_updates_b_privately():
    path = Path(".test_artifacts") / "fedask_active_probe"
    path.mkdir(parents=True, exist_ok=True)
    method_config = {
        "rank": 2,
        "local_steps": 1,
        "clip_norm": 0.5,
        "noise_multiplier": 0.2,
        "reproducible_dp_noise": True,
        "seed": 31,
    }
    server = ServerBase(
        train_mode="centralized",
        device=torch.device("cpu"),
        dataset_name="toy",
        train_sets=[_dataset(index * 0.02) for index in range(2)],
        test_sets=[_dataset(0.4 + index * 0.02) for index in range(2)],
        class_names=["zero", "one"],
        model=ToyPrivateMethodModel("fedask"),
        batch_size=8,
        eval_batch_size=8,
        learning_rate=0.05,
        num_glob_iters=1,
        local_epochs=1,
        total_users=2,
        results_dir=str(path),
        user_per_round=2,
        aggregator=build_aggregator("fedask", rank=2, oversampling=1),
        audit_config={"enabled": False, "attacks": [], "target_client_id": 0},
        defense_config={
            "name": "cofedmid",
            "cofedmid_intervals": 2,
            "cofedmid_recycle_ratio": 0.25,
            "cofedmid_noise_std": 0.1,
            "cofedmid_perturb_ratio": 1.0,
        },
        method_config=method_config,
    )
    user = server.ctx.users[0]
    probe = copy.deepcopy(user.model)
    before_a = probe.prompt_learner.fedask_A.detach().clone()
    before_b = probe.prompt_learner.fedask_B.detach().clone()
    user.train_model(probe, privacy_probe=True)
    assert torch.equal(before_a, probe.prompt_learner.fedask_A)
    assert not torch.equal(before_b, probe.prompt_learner.fedask_B)


def test_training_time_attack_and_prompt_dp_can_run_together():
    path = Path(".test_artifacts") / "codepoison_prompt_dp"
    path.mkdir(parents=True, exist_ok=True)
    summaries = _defended_server(
        path,
        "prompt_dp",
        attack_enabled=True,
        attack="codepoison",
    ).train()
    assert [item["attack"] for item in summaries] == ["codepoison"]


def test_active_prompt_probe_runs_under_every_defense():
    root = Path(".test_artifacts") / "promptmia_defenses"
    for defense in ("cofedmid", "prompt_dp", "mist", "soft", "hamp"):
        path = root / defense
        path.mkdir(parents=True, exist_ok=True)
        server = _defended_server(
            path,
            defense,
            attack_enabled=True,
            attack="promptmia",
        )
        server.auditor.config["promptmia_max_samples"] = 4
        server.auditor.config["promptmia_keys"] = 2
        summaries = server.train()
        assert [item["attack"] for item in summaries] == ["promptmia"]
        assert summaries[0]["metadata"]["score"] == "signed_projected_client_update"


def test_all_attacks_run_in_a_toy_federated_prompt_experiment():
    tmp_path = Path(".test_artifacts")
    tmp_path.mkdir(exist_ok=True)
    train_sets = [_dataset(index * 0.02) for index in range(4)]
    test_sets = [_dataset(0.4 + index * 0.02) for index in range(4)]
    server = ServerBase(
        train_mode="centralized",
        device=torch.device("cpu"),
        dataset_name="toy",
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=["zero", "one"],
        model=ToyPromptModel(),
        batch_size=4,
        eval_batch_size=8,
        learning_rate=0.05,
        num_glob_iters=2,
        local_epochs=1,
        total_users=4,
        results_dir=str(tmp_path),
        user_per_round=4,
        aggregator=build_aggregator("fedavg"),
        eval_interval=1,
        audit_config={
            "enabled": True,
            "target_client_id": 0,
            "ensure_target_participation": True,
            "attacks": [
                "nasr_passive",
                "nasr_active",
                "fedmia_loss",
                "fedmia_cosine",
                "transfer_representation",
                "codepoison",
                "pipra",
                "rmia",
                "imia",
                "quantile_mia",
                "yoqo",
                "canary",
                "promptmia",
            ],
            "max_samples_per_group": 4,
            "audit_interval": 1,
            "calibration_fraction": 0.5,
            "active_max_samples": 4,
            "active_ascent_steps": 1,
            "active_probe_cycles": 1,
            "auxiliary_fraction": 0.5,
            "qmia_epochs": 3,
            "pipra_shadow_prompts": 2,
            "pipra_shadow_steps": 1,
            "pipra_attack_epochs": 3,
            "imia_models": 1,
            "imia_warmup_steps": 1,
            "imia_imitation_steps": 1,
            "imia_pivot_steps": 1,
            "query_max_samples": 4,
            "query_reference_models": 1,
            "yoqo_steps": 1,
            "canary_num_queries": 1,
            "canary_steps": 1,
            "canary_shadow_steps": 1,
            "promptmia_max_samples": 4,
            "promptmia_keys": 2,
        },
    )
    summaries = server.train()
    assert {item["attack"] for item in summaries} == {
        "nasr_passive",
        "nasr_active",
        "fedmia_loss",
        "fedmia_cosine",
        "transfer_representation",
        "codepoison",
        "pipra",
        "rmia",
        "imia",
        "quantile_mia",
        "yoqo",
        "canary",
        "promptmia",
    }
    assert (tmp_path / "privacy_audit" / "summary.json").exists()
    assert (tmp_path / "privacy_audit" / "predictions.csv").exists()
    assert (tmp_path / "final_prompt.pt").exists()


def test_veil_and_mirage_are_backward_compatible_local_ggeur_aliases():
    config = default_config()
    for alias in ("mirage", "veil"):
        config["defense"]["name"] = alias
        validate_config(config)
        controller = DefenseController(
            config["defense"],
            device=torch.device("cpu"),
            total_users=2,
            num_classes=3,
            total_rounds=1,
        )
        assert controller.name == alias
        assert controller.enabled
