from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from aggregator.aggregator_builder import build_aggregator
from aggregator.seismograph_aggregator import aggregate_seismograph_model_states
from main import validate_minimal_scope
from servers.serverbase import ServerBase
from users.user import UserBase
from utils.constants import SUPPORTED_FPL_ATTACKS
from utils.seismograph_text_feature_analysis import (
    filter_users_by_raw_top1_svd_history,
)
from utils.poison_func import get_poisoned_sample_count
from utils.trigger import trigger_create_funcs


class ToyPromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = nn.Parameter(torch.tensor([[0.2, -0.2], [0.1, -0.1]]))

    def forward(self, images):
        signal = images.mean(dim=(1, 2, 3))
        logits = torch.stack((signal, -signal), dim=1)
        return logits + self.prompt.mean(dim=0)

    def get_text_features(self, normalize=True):
        if normalize:
            return F.normalize(self.prompt, p=2, dim=-1)
        return self.prompt


def _toy_dataset():
    images = torch.linspace(-0.2, 0.2, steps=4 * 3 * 4 * 4).reshape(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1])
    return TensorDataset(images, labels)


def test_minimal_scope_accepts_only_three_fpl_attacks_and_seismograph():
    assert set(SUPPORTED_FPL_ATTACKS) == {"cerberus", "a3fl", "sabre"}
    for attack_method in SUPPORTED_FPL_ATTACKS:
        validate_minimal_scope(
            attack_method=attack_method,
            defense="seismograph",
            fpl=True,
            train_mode="centralized",
        )

    with pytest.raises(ValueError, match="only supports"):
        validate_minimal_scope(
            attack_method="badpfl",
            defense="seismograph",
            fpl=True,
            train_mode="centralized",
        )
    with pytest.raises(ValueError, match="only supports 'seismograph'"):
        validate_minimal_scope(
            attack_method="a3fl",
            defense="none",
            fpl=True,
            train_mode="centralized",
        )
    with pytest.raises(ValueError, match="only supports FPL"):
        validate_minimal_scope(
            attack_method="a3fl",
            defense="seismograph",
            fpl=False,
            train_mode="centralized",
        )


def test_aggregator_builder_exposes_only_seismograph():
    assert build_aggregator("seismograph").name == "seismograph"
    with pytest.raises(ValueError, match="only supports"):
        build_aggregator("none")


def test_fpl_poison_count_and_trigger_registry():
    assert get_poisoned_sample_count(4, 16) == 4
    assert get_poisoned_sample_count(20, 16) == 16
    with pytest.raises(ValueError, match="integer count"):
        get_poisoned_sample_count(0.5, 16)
    with pytest.raises(ValueError, match="only supports FPL"):
        get_poisoned_sample_count(4, 16, fpl=False)

    assert set(trigger_create_funcs) == set(SUPPORTED_FPL_ATTACKS)
    triggers, patterns = trigger_create_funcs["a3fl"](
        "cifar100",
        torch.device("cpu"),
        malnum=1,
        total_users=2,
        fpl=True,
        trigger_size=2,
    )
    assert len(triggers) == len(patterns) == 2
    assert triggers[0].shape == (3, 224, 224)
    assert len(patterns[0]) == 4


def test_seismograph_weighted_aggregation_uses_only_selected_clients():
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
    aggregate_seismograph_model_states(ctx, [0, 1])
    assert torch.allclose(ctx.new_model_state[0]["prompt"], torch.tensor([2.5]))


def test_seismograph_history_excludes_a_persistent_outlier(tmp_path):
    ctx = SimpleNamespace(glob_iter=0, results_dir=str(tmp_path))
    history = {}
    seismograph_state = {}
    selected_ids = [0, 1, 2]

    first_round = filter_users_by_raw_top1_svd_history(
        ctx=ctx,
        selected_ids=selected_ids,
        valid_user_ids=selected_ids,
        raw_top1_values=torch.tensor([1.0, 1.0, 1.0]),
        device=torch.device("cpu"),
        raw_top1_log_history=history,
        raw_top1_seismograph_state=seismograph_state,
        seismograph_k=0.0,
        seismograph_h=1.0,
    )
    assert first_round == selected_ids

    ctx.glob_iter = 1
    second_round = filter_users_by_raw_top1_svd_history(
        ctx=ctx,
        selected_ids=selected_ids,
        valid_user_ids=selected_ids,
        raw_top1_values=torch.tensor([1.0, 1.0, 10.0]),
        device=torch.device("cpu"),
        raw_top1_log_history=history,
        raw_top1_seismograph_state=seismograph_state,
        seismograph_k=0.0,
        seismograph_h=1.0,
    )
    assert second_round == [0, 1]
    assert seismograph_state[2] > 1.0


@pytest.mark.parametrize("attack_method", ["cerberus", "a3fl", "sabre"])
def test_each_supported_client_attack_runs_on_cpu(attack_method):
    user = UserBase(
        device=torch.device("cpu"),
        fpl=True,
        id=0,
        dataset_name="cifar100",
        train_data=_toy_dataset(),
        test_data=_toy_dataset(),
        model=ToyPromptModel(),
        batch_size=2,
        eval_batch_size=2,
        learning_rate=0.01,
        local_epochs=1,
        local_poison_epochs=1,
        defense="seismograph",
        malicious=True,
    )
    trigger = torch.zeros(3, 4, 4)
    trigger[:, 3, 3] = 0.01
    pattern = [[3, 3]]
    if attack_method == "cerberus":
        user.cerberus_train(
            poison_ratio=1,
            poison_label=1,
            trigger=trigger,
            pattern=pattern,
            poisoned_model_dict={},
        )
    else:
        user.poison_train(
            poison_ratio=1,
            poison_label=1,
            trigger=trigger,
            pattern=pattern,
            attack_method=attack_method,
        )
    assert torch.isfinite(user.model.prompt).all()


def test_server_and_seismograph_aggregator_complete_a_toy_round(tmp_path):
    datasets = [_toy_dataset() for _ in range(4)]
    server = ServerBase(
        train_mode="centralized",
        fpl=True,
        device=torch.device("cpu"),
        dataset_name="cifar100",
        train_sets=datasets,
        test_sets=datasets,
        class_names=["zero", "one"],
        model=ToyPromptModel(),
        batch_size=2,
        eval_batch_size=2,
        learning_rate=0.01,
        num_glob_iters=1,
        local_epochs=1,
        local_poison_epochs=1,
        total_users=4,
        malnum=1,
        malclient_ids=[0],
        poisonratio=1,
        poison_label=1,
        attack_method="a3fl",
        defense="seismograph",
        results_dir=str(tmp_path),
        user_per_round=4,
        aggregator=build_aggregator("seismograph"),
        eval_interval=1,
        trigger_optimization_interval=1,
    )
    trigger_list = [torch.zeros(3, 4, 4) for _ in range(4)]
    pattern_list = [[[3, 3]] for _ in range(4)]
    server.train(
        trigger_list=trigger_list,
        pattern_list=pattern_list,
        attack_rounds=[],
    )

    assert 0 in server.ctx.new_model_state
    assert torch.isfinite(server.ctx.new_model_state[0]["prompt"]).all()
    assert len((tmp_path / "summary_metrics.csv").read_text().splitlines()) == 2
    assert (tmp_path / "text_feature_raw_top1_history" / "round_0.csv").exists()
