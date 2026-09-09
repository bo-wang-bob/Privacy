import pytest
import torch
from types import SimpleNamespace
from torch.utils.data import TensorDataset

from privacy_attacks.auditor import MembershipAuditor


class _ToyExactBatchProjResModel(torch.nn.Module):
    model_type = "bert_adapter"
    architecture = "bert"

    def __init__(self):
        super().__init__()
        self.adapter_down = torch.nn.Linear(2, 2, bias=False)

    def get_projres_attack_surface(self, parameter_name=None):
        expected = "adapter_down.weight"
        if parameter_name not in {None, expected}:
            raise ValueError(parameter_name)
        return expected, self.adapter_down

    def get_projres_representations(
        self,
        packed_inputs,
        parameter_name=None,
        token_reduction="auto",
    ):
        del parameter_name, token_reduction
        representations = packed_inputs[:, 0, :2].float()
        return representations, int(packed_inputs.shape[0])


@pytest.mark.parametrize("method", ["fedsgd", "fedavg"])
@pytest.mark.parametrize("audit_batch_size", [1, 4])
def test_projres_scores_use_public_post_state_for_all_candidates(method, audit_batch_size):
    class ContextualModel(_ToyExactBatchProjResModel):
        def __init__(self):
            super().__init__()
            self.context = torch.nn.Linear(2, 2, bias=False)

        def get_projres_representations(self, inputs, **kwargs):
            return self.context(inputs[:, 0].float()), len(inputs)

    auditor = MembershipAuditor.__new__(MembershipAuditor)
    auditor.model = ContextualModel()
    auditor.model_type = "bert_adapter"
    auditor.federated_method = method
    auditor.exact_batch_projres_config = {}
    auditor.audit_batch_size = audit_batch_size
    auditor.target_client_id = 0
    auditor.exact_batch_nonmember_ratio = 1
    auditor.low_fpr_candidate_selection = {}
    saved = {k: v.clone() for k, v in auditor.model.state_dict().items()}
    base = {k: torch.eye(2) for k in saved}
    post = {
        "adapter_down.weight": torch.tensor([[0., 0.], [0., 1.]]),
        "context.weight": torch.tensor([[1., 2.], [3., 5.]]),
    }
    lr = 0.25
    upload = {k: ((base[k] - v) / lr if method == "fedsgd" else v - base[k])
              for k, v in post.items()}
    inputs = torch.tensor([[[1., 0.]], [[2., 0.]], [[0., 1.]], [[1., 1.]]])
    # Internal post parameters deliberately disagree with the released upload.
    private_post = {k: torch.full_like(v, float("nan")) for k, v in post.items()}
    scores, diagnostics, _ = auditor._score_exact_batch_projres(
        round_index=1, member_inputs=inputs[:2], nonmember_inputs=inputs[2:],
        labels=torch.tensor([0, 1, 0, 1]), membership=torch.tensor([1, 1, 0, 0]),
        member_local_indices=torch.tensor([0, 1]), nonmember_pool_indices=torch.tensor([0, 1]),
        base_state=base, updated_state=private_post,
        protocol_message={"kind": "gradient" if method == "fedsgd" else "model_update",
                          "tensors": upload}, learning_rate=lr,
    )
    # The upload row space is the x axis; post inputs have y = 3x + 5y.
    torch.testing.assert_close(scores, torch.tensor([-3., -6., -5., -8.], dtype=scores.dtype))
    for name, value in auditor.model.state_dict().items():
        torch.testing.assert_close(value, saved[name])
    assert diagnostics["metadata"]["representation_state"] == "client_post_update_model"
    assert diagnostics["metadata"]["paper_fedsgd_exact"] is False


def test_projres_restores_shared_model_after_extractor_failure():
    from privacy_attacks.projres_state import projres_post_model

    model = _ToyExactBatchProjResModel()
    base = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="extractor failed"):
        with projres_post_model(
            model, base_state=base, updated_state={}, learning_rate=0.5,
            protocol_message={"kind": "gradient", "tensors": {k: torch.ones_like(v) for k, v in base.items()}},
            federated_method="fedsgd",
        ):
            raise RuntimeError("extractor failed")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, base[name])


@pytest.mark.parametrize("model_type", ["clip_lora", "bert_adapter", "bert_lora", "gpt2_adapter"])
def test_independent_projres_recomputes_nonmembers_for_each_client(model_type, tmp_path):
    from privacy_attacks.projres_integrated import run_integrated_projres

    class Model(_ToyExactBatchProjResModel):
        def __init__(self):
            super().__init__()
            self.model_type = model_type
            self.adapter_down.rank = 2
            self.context = torch.nn.Linear(2, 2, bias=False)
            self.observed_states = []

        def resolve_projres_token_reduction(self, reduction, parameter_name=None):
            return "cls"

        def get_projres_representations(self, inputs, **kwargs):
            self.observed_states.append(self.context.weight.detach().clone())
            features = inputs.reshape(len(inputs), -1)[:, :2].float()
            return self.context(features), (1 if model_type == "clip_lora" else len(inputs))

    model = Model()
    base = {k: v.clone() for k, v in model.state_dict().items()}
    gradients = {i: {k: torch.full_like(v, 0.2 + i) for k, v in base.items()} for i in range(2)}
    shape = (4, 1, 2, 2) if model_type == "clip_lora" else (4, 2, 2)
    inputs = torch.arange(16).reshape(shape).float()
    labels = torch.tensor([0, 1, 0, 1])
    users = [SimpleNamespace(id=i, last_train_batch=(inputs[:2], labels[:2]),
                             collate_fn=None, test_data=TensorDataset(inputs, labels))
             for i in range(2)]
    private_post = {k: torch.full_like(v, float("nan")) for k, v in base.items()}
    payload = run_integrated_projres(
        model=model, users=users, device=torch.device("cpu"),
        base_states={0: base, 1: base}, updated_states={0: private_post, 1: private_post},
        client_gradients=gradients, learning_rate=0.5, batch_size=2,
        eval_batch_size=2, local_epochs=1, round_index=0, seed=42, dataset_name="toy",
        client_ids=[0, 1], config={"max_candidates": 2, "min_nonmembers": 4, "max_nonmembers": 4},
        output_path=tmp_path / "projres.json", federated_method="fedsgd",
    )
    # Each client gets two nonmember chunks and one member chunk in its own state.
    assert len(model.observed_states) == 6
    for i in range(2):
        expected = base["context.weight"] - 0.5 * gradients[i]["context.weight"]
        for observed in model.observed_states[i * 3:(i + 1) * 3]:
            torch.testing.assert_close(observed, expected)
        result = payload["per_client"][i]
        assert result["attack"]["metadata"]["representation_state"] == "client_post_update_model"
        assert result["threat_model"]["paper_fedsgd_exact"] is False
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, base[name])
