import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import TensorDataset

from aggregator.aggregator_builder import build_aggregator
from main import validate_config as validate_image_config
from privacy_defenses.controller import (
    DefenseController,
    _loop_clipped_gradient_sum,
    _vmap_clipped_gradient_sum,
)
from scripts.run_fedllm_adapter import validate_config as validate_text_config
from servers.serverbase import ServerBase
from users.user import UserBase
from utils.privacy_accounting import (
    calibrate_poisson_sampled_gaussian_noise,
    gaussian_rdp_epsilon,
    poisson_sampled_gaussian_epsilon,
    poisson_sampled_gaussian_rdp,
)


class TinyRecordModel(nn.Module):
    model_type = "tiny"

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs):
        return self.linear(inputs)


def test_record_dp_resnet_and_bert_configs_validate():
    with open(
        "configs/resnet18_cifar100_record_dp.yaml", encoding="utf-8"
    ) as file:
        resnet = yaml.safe_load(file)
    with open(
        "configs/bert_base_sst5_adapter_record_dp.yaml", encoding="utf-8"
    ) as file:
        bert = yaml.safe_load(file)

    validate_image_config(resnet)
    validate_text_config(bert)
    assert resnet["defense"]["name"] == "record_dp"
    assert bert["defense"]["name"] == "record_dp"
    assert bert["projres"]["max_candidates"] == 0


def test_poisson_sampled_rdp_matches_full_gaussian_and_amplifies_privacy():
    full = poisson_sampled_gaussian_epsilon(
        noise_multiplier=1.3,
        sample_rate=1.0,
        steps=20,
        delta=1e-5,
    )
    assert full == pytest.approx(
        gaussian_rdp_epsilon(1.3, steps=20, delta=1e-5)
    )
    sampled = poisson_sampled_gaussian_epsilon(
        noise_multiplier=1.3,
        sample_rate=0.1,
        steps=20,
        delta=1e-5,
    )
    assert 0 < sampled < full
    assert poisson_sampled_gaussian_rdp(1.3, 0.0, 8) == 0.0


def test_poisson_noise_calibration_meets_every_client_budget():
    schedules = [(0.02, 150), (0.08, 40)]
    multiplier = calibrate_poisson_sampled_gaussian_noise(
        target_epsilon=3.0,
        schedules=schedules,
        delta=1e-5,
    )
    epsilons = [
        poisson_sampled_gaussian_epsilon(
            multiplier, sample_rate, steps, delta=1e-5
        )
        for sample_rate, steps in schedules
    ]
    assert max(epsilons) <= 3.0 + 1e-8
    assert max(epsilons) >= 3.0 - 1e-6


def test_loop_and_vmap_apply_the_same_global_per_record_clipping():
    torch.manual_seed(7)
    model = TinyRecordModel()
    inputs = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])
    parameters = [parameter for parameter in model.parameters()]
    losses = F.cross_entropy(model(inputs), labels, reduction="none")
    loop_sum, loop_factors = _loop_clipped_gradient_sum(
        losses, parameters, max_norm=0.2
    )
    vmap_sum, vmap_factors = _vmap_clipped_gradient_sum(
        model, inputs, labels, max_norm=0.2
    )

    assert torch.allclose(loop_factors, vmap_factors, atol=1e-6)
    assert all(
        torch.allclose(loop, vectorized, atol=1e-6)
        for loop, vectorized in zip(loop_sum, vmap_sum)
    )
    assert torch.all(loop_factors <= 1)


def test_record_dp_fedsgd_uses_one_poisson_draw_and_uploads_one_noisy_gradient(
    tmp_path,
):
    data = TensorDataset(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]
        ),
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    controller = DefenseController(
        {
            "name": "record_dp",
            "max_grad_norm": 0.5,
            "noise_multiplier": 1.0,
            "delta": 1e-5,
            "sampling": "poisson",
            "microbatch_size": 2,
            "grad_sample_backend": "loop",
            "reproducible_noise": True,
            "seed": 11,
        },
        device=torch.device("cpu"),
        total_users=1,
        num_classes=2,
        total_rounds=1,
    )
    controller.federated_method = "fedsgd"
    controller.method_config = {
        "client_optimizer": "sgd",
        "momentum": 0.0,
        "weight_decay": 0.0,
    }
    user = UserBase(
        device=torch.device("cpu"),
        id=0,
        dataset_name="toy",
        train_data=data,
        test_data=data,
        model=TinyRecordModel(),
        batch_size=4,
        eval_batch_size=4,
        learning_rate=0.1,
        local_epochs=1,
        defense_controller=controller,
        federated_method="fedsgd",
        method_config=controller.method_config,
    )
    controller.samples_num = [len(data)]
    controller.configure_record_dp([user])

    user.train(round_index=0)

    assert controller.steps[0] == 1
    assert user.last_gradient_capture_count == 1
    assert user.last_update_gradients is not None
    assert set(user.last_update_gradients) == {"linear.weight", "linear.bias"}
    summary = controller.save_summary(str(tmp_path))
    accounting = summary["privacy_accounting"]
    assert accounting["per_client"]["0"]["actual_steps"] == 1
    assert accounting["per_client"]["0"]["sample_rate"] == pytest.approx(0.5)
    assert not accounting["formal_dp_enabled"]
    assert json.loads((tmp_path / "defense_summary.json").read_text())[
        "defense"
    ] == "record_dp"


def test_record_dp_audit_reports_theoretical_roc_envelope(tmp_path):
    generator = torch.Generator().manual_seed(23)
    train_sets = []
    test_sets = []
    for client_id in range(2):
        train_inputs = torch.randn(8, 3, generator=generator) + client_id * 0.1
        test_inputs = torch.randn(8, 3, generator=generator) + 0.5
        labels = torch.tensor([0, 1] * 4)
        train_sets.append(TensorDataset(train_inputs, labels))
        test_sets.append(TensorDataset(test_inputs, labels))
    server = ServerBase(
        device=torch.device("cpu"),
        dataset_name="toy",
        train_sets=train_sets,
        test_sets=test_sets,
        class_names=["zero", "one"],
        model=TinyRecordModel(),
        batch_size=4,
        eval_batch_size=8,
        learning_rate=0.05,
        num_glob_iters=2,
        local_epochs=1,
        total_users=2,
        results_dir=str(tmp_path),
        user_per_round=2,
        aggregator=build_aggregator("fedavg"),
        eval_interval=1,
        audit_config={
            "enabled": True,
            "strict": True,
            "target_client_id": 0,
            "ensure_target_participation": True,
            "attacks": ["fedmia_loss"],
            "max_samples_per_group": 4,
            "audit_interval": 1,
            "training_health_check": False,
        },
        defense_config={
            "name": "record_dp",
            "max_grad_norm": 0.5,
            "noise_multiplier": 1.0,
            "delta": 1e-5,
            "sampling": "poisson",
            "accountant": "rdp",
            "grad_sample_backend": "loop",
            "microbatch_size": 4,
            "reproducible_noise": True,
            "seed": 29,
        },
        method_config={
            "client_optimizer": "sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
    )

    summaries = server.train()

    assert summaries[0]["record_dp_theoretical_tpr_upper_bounds"]
    audit = json.loads(
        (tmp_path / "privacy_audit" / "summary.json").read_text()
    )
    verification = audit["record_dp_verification"]
    assert verification["privacy_unit"] == "record"
    assert verification["roc_constraint"].startswith("TPR <=")
    assert verification["audit_artifacts_are_private_research_data"]
