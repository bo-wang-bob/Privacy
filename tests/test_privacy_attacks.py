import json
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
from privacy_attacks.metrics import membership_metrics
from privacy_attacks.model_utils import scaled_confidence
from privacy_attacks.promptmia import generate_key_with_similarity
from privacy_defenses import DefenseController, attach_hamp_output_transform
from servers.serverbase import ServerBase


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
    assert membership_metrics(labels, scores)["auc"] == 1.0

    images = torch.randn(3, 3, 4, 4)
    first = generate_membership_encoding_samples(images)
    second = generate_membership_encoding_samples(images)
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_recent_attack_primitives_match_paper_definitions():
    query = torch.tensor([1.0, 2.0, -1.0, 0.5])
    key = generate_key_with_similarity(
        query, 0.73, torch.Generator().manual_seed(7)
    )
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
    config["aggregator"] = "seismograph"
    try:
        validate_config(config)
    except ValueError as error:
        assert "FedAvg" in str(error)
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


def test_defense_primitives_preserve_required_invariants():
    model = ToyPromptModel()
    images, _ = next(iter(torch.utils.data.DataLoader(_dataset(), batch_size=8)))
    original_logits = model(images)
    attach_hamp_output_transform(model, temperature=4.0)
    defended_logits = model(images)
    assert torch.equal(original_logits.argmax(dim=1), defended_logits.argmax(dim=1))
    assert torch.softmax(defended_logits, dim=1).amax(dim=1).mean() < torch.softmax(
        original_logits, dim=1
    ).amax(dim=1).mean()

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
        },
    )


def test_each_defense_runs_independently_with_one_attack():
    root = Path(".test_artifacts") / "defenses"
    for defense in ("cofedmid", "prompt_dp", "mist", "soft", "hamp"):
        path = root / defense
        path.mkdir(parents=True, exist_ok=True)
        summaries = _defended_server(path, defense).train()
        assert [item["attack"] for item in summaries] == ["fedmia_loss"]
        summary = json.loads(
            (path / "defense_summary.json").read_text(encoding="utf-8")
        )
        assert summary["defense"] == defense
        assert (path / "privacy_audit" / "summary.json").exists()


def test_defense_only_mode_skips_attack_audit():
    path = Path(".test_artifacts") / "defense_only"
    path.mkdir(parents=True, exist_ok=True)
    summaries = _defended_server(path, "hamp", attack_enabled=False).train()
    assert summaries == []
    assert (path / "defense_summary.json").exists()
    assert not (path / "privacy_audit" / "summary.json").exists()


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
