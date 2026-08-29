from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml
from datasets import Dataset, DatasetDict

from scripts.run_fedllm_adapter import validate_config
from scripts.run_privacy_experiments import load_yaml, resolve_model_config
from servers.serverbase import _scheduled_learning_rate
from utils.text_data_loader import (
    load_federated_text_classification,
    normalize_text_dataset_name,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "models"
    / "gpt2_adapter.yaml"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_yaml("configs/experiment_catalog.yaml")


def _bert_config(dataset):
    profile = CATALOG["models"]["bert_adapter"]
    return resolve_model_config(
        CATALOG,
        model="bert_adapter",
        dataset=dataset,
        attacks=profile["default_attacks"],
        defense=profile["default_defenses"][0],
        seed=42,
        target_client_id=0,
        results_dir="results/test-bert-adapter",
    )


@pytest.mark.parametrize(
    "dataset",
    ["sst5", "cola", "imdb"],
)
def test_bert_dataset_schedule_keeps_all_privacy_evaluations_enabled(
    dataset,
):
    config = _bert_config(dataset)

    assert config["dataset_name"] == dataset
    assert config["num_global_iters"] == 500
    assert config["batch_size"] == 16
    assert config["eval_batch_size"] == 32
    assert config["eval_interval"] == 50
    assert config["learning_rate"] == pytest.approx(0.005)
    assert config["learning_rate_decay"] == pytest.approx(1.0)
    assert config["learning_rate_decay_interval"] == 50
    assert config["adapter"]["zero_init_up"] is True
    assert config["optimization"]["max_grad_norm"] == pytest.approx(0.0)
    assert config["audit"]["enabled"] is True
    assert set(config["audit"]["attacks"]) == {
        "blackbox_loss",
        "loss_series",
        "grad_cosine",
        "avg_cosine",
        "fedmia_loss",
        "fedmia_cosine",
        "gradient_diff",
        "score_diff",
        "score_ratio",
        "fta",
        "projres",
    }
    assert config["audit"]["audit_interval"] == 50
    assert config["audit"]["candidate_sampling"] == (
        "balanced_global_holdout"
    )
    assert config["audit"]["require_full_target_train_members"] is True
    assert config["audit"]["nonmember_to_member_ratio"] == pytest.approx(1.0)
    assert set(config["audit"]["exact_batch_membership_attacks"]) == {
        "blackbox_loss",
        "grad_cosine",
        "gradient_diff",
        "projres",
        "score_diff",
        "score_ratio",
    }
    assert config["audit"]["exact_batch_nonmember_to_member_ratio"] == 10
    assert config["audit"]["paper_balanced_evaluation_size"] == 100
    assert config["audit"]["low_fpr_min_nonmembers"] == 2
    assert config["audit"]["low_fpr_max_members"] == 0
    assert config["audit"]["low_fpr_max_nonmembers"] == 0
    fixed_candidate_attacks = {
        "loss_series",
        "avg_cosine",
        "fedmia_loss",
        "fedmia_cosine",
        "fta",
    }
    assert config["audit"]["attack_audit_intervals"] == {
        attack: 10 if attack in fixed_candidate_attacks else 50
        for attack in config["audit"]["attacks"]
    }
    assert config["projres"]["enabled"] is True
    assert config["projres"]["evaluation_interval"] == 50
    assert "evaluation_round" not in config["projres"]
    assert config["projres"]["max_candidates"] == 16
    assert config["projres"]["min_nonmembers"] == 160
    assert config["projres"]["max_nonmembers"] == 160
    assert config["defense"] == {
        "name": "iclr",
        "iclr_analysis_interval": 50,
        "iclr_analysis_timing": "post_round",
        "iclr_feature_statistics": False,
        "iclr_validation_top_fraction": pytest.approx(0.2),
    }

    validate_config(config)
    assert config["primary_metric"] == ("mcc" if dataset == "cola" else "accuracy")


@pytest.mark.parametrize(
    "dataset",
    ["sst5", "cola", "imdb"],
)
def test_bert_dataset_learning_rate_milestones(
    dataset,
):
    config = _bert_config(dataset)

    rates = [
        _scheduled_learning_rate(
            config["learning_rate"],
            config["learning_rate_decay"],
            round_index,
            config["learning_rate_decay_interval"],
        )
        for round_index in (0, 49, 50, 299, 499)
    ]
    assert rates == pytest.approx([0.005] * 5)


def test_unified_fedllm_dry_run_selects_dataset_specific_bert_configs():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_privacy_experiments.py",
            "--models",
            "bert",
            "--datasets",
            "sst5,cola,imdb",
            "--dry-run",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.count("model=bert_adapter dataset=") == 3
    assert "dataset=sst5" in completed.stdout
    assert "dataset=cola" in completed.stdout
    assert "dataset=imdb" in completed.stdout


def test_gpt2_uses_constant_learning_rate_for_500_rounds():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["num_global_iters"] == 500
    assert config["batch_size"] == 16
    assert config["eval_batch_size"] == 16
    assert config["eval_interval"] == 50
    assert config["learning_rate"] == pytest.approx(0.001)
    assert config["learning_rate_decay"] == pytest.approx(1.0)
    assert config["learning_rate_decay_interval"] == 50
    assert config["adapter"]["zero_init_up"] is True
    assert config["optimization"] == {
        "client_optimizer": "sgd",
        "momentum": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 0.0,
    }
    assert config["audit"]["enabled"] is True
    assert len(config["audit"]["attacks"]) == 11
    assert config["audit"]["audit_batch_size"] == 16
    assert config["audit"]["audit_interval"] == 50
    assert config["audit"]["candidate_sampling"] == (
        "balanced_global_holdout"
    )
    assert config["audit"]["require_full_target_train_members"] is True
    assert config["audit"]["nonmember_to_member_ratio"] == pytest.approx(1.0)
    assert set(config["audit"]["exact_batch_membership_attacks"]) == {
        "blackbox_loss",
        "grad_cosine",
        "gradient_diff",
        "projres",
        "score_diff",
        "score_ratio",
    }
    assert config["audit"]["exact_batch_nonmember_to_member_ratio"] == 10
    assert config["audit"]["paper_balanced_evaluation_size"] == 100
    assert config["audit"]["low_fpr_min_nonmembers"] == 2
    assert config["audit"]["low_fpr_max_members"] == 0
    assert config["audit"]["low_fpr_max_nonmembers"] == 0
    assert config["audit"]["attack_audit_intervals"] == {
        attack: 50 for attack in config["audit"]["attacks"]
    }
    assert config["projres"]["enabled"] is True
    assert config["projres"]["evaluation_interval"] == 50
    assert "evaluation_round" not in config["projres"]
    assert config["projres"]["max_candidates"] == 16
    assert config["projres"]["min_nonmembers"] == 160
    assert config["projres"]["max_nonmembers"] == 160

    rates = [
        _scheduled_learning_rate(
            config["learning_rate"],
            config["learning_rate_decay"],
            round_index,
            config["learning_rate_decay_interval"],
        )
        for round_index in (0, 49, 50, 299, 499)
    ]
    assert rates == pytest.approx([0.001] * 5)


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        sentences,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
        return_attention_mask,
    ):
        assert padding and truncation and return_attention_mask
        assert return_tensors == "pt"
        width = min(max(len(sentence.split()) for sentence in sentences), max_length)
        input_ids = torch.zeros((len(sentences), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, sentence in enumerate(sentences):
            length = min(len(sentence.split()), width)
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _save_sst5(path: Path, *, invalid_label: bool = False) -> None:
    train_labels = [0, 1, 2, 3, 4] * 4
    validation_labels = [0, 1, 2, 3, 4] * 2
    if invalid_label:
        validation_labels[-1] = 5
    DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": [f"train sentence {index}" for index in range(20)],
                    "label": train_labels,
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "text": [
                        f"validation sentence {index}" for index in range(10)
                    ],
                    "label": validation_labels,
                }
            ),
        }
    ).save_to_disk(str(path))


def test_sst5_alias_and_five_class_schema(monkeypatch, tmp_path):
    dataset_path = tmp_path / "sst5"
    model_path = tmp_path / "model"
    model_path.mkdir()
    _save_sst5(dataset_path)
    monkeypatch.setattr(
        "utils.text_data_loader.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )

    data = load_federated_text_classification(
        dataset_name="sst-5",
        dataset_path=dataset_path,
        model_path=model_path,
        num_users=2,
        seed=42,
        max_length=8,
    )

    assert normalize_text_dataset_name("SST-5") == "sst5"
    assert data.dataset_name == "sst5"
    assert data.class_names == [
        "very negative",
        "negative",
        "neutral",
        "positive",
        "very positive",
    ]
    assert [len(split) for split in data.train_sets] == [10, 10]
    assert [len(split) for split in data.test_sets] == [5, 5]
    packed_inputs, labels = data.collate_fn([data.train_sets[0][0]])
    assert packed_inputs.shape[0:2] == (1, 2)
    assert labels.shape == (1,)


def test_sst5_rejects_labels_outside_zero_to_four(monkeypatch, tmp_path):
    dataset_path = tmp_path / "sst5"
    model_path = tmp_path / "model"
    model_path.mkdir()
    _save_sst5(dataset_path, invalid_label=True)
    monkeypatch.setattr(
        "utils.text_data_loader.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )

    with pytest.raises(ValueError, match="labels outside"):
        load_federated_text_classification(
            dataset_name="sst5",
            dataset_path=dataset_path,
            model_path=model_path,
            num_users=2,
        )
