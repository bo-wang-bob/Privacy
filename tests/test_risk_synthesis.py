import copy
import csv
import json
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.risk_synthesis import DEFAULTS, LocalGeometry, RiskSynthesis, low_rank_factor, select_requests
from scripts.run_privacy_experiments import build_tasks, load_yaml, parse_args
from scripts.analyze_risk_synthesis import class_diagnostics, read_synthesis_mechanism, recompute_auc
from scripts.paired_synthesis_uncertainty import paired_resampling, score_metrics, select_pair
from scripts.prepare_synthesis_confirmation import reserve_indices
from servers.serverbase import ServerBase
from trainmodel.clip_lora import CLIPLoRA
from trainmodel.clip_transformer_adapter import CLIPTransformerAdapter
from test_clip_transformer_adapter import backbone_and_prompts, dataset
from test_clip_peft_fedsgd import ATTACKS, _audit_config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(23)
    yield
    torch.set_num_threads(previous)


def make_model(kind):
    backbone, prompts = backbone_and_prompts()
    if kind == "clip_adapter":
        return CLIPTransformerAdapter(backbone, prompts, ["a", "b", "c"])
    return CLIPLoRA(backbone, prompts, ["a", "b", "c"], encoder="both", rank=4, dropout=0)


@pytest.mark.parametrize("kind", ["clip_adapter", "clip_lora"])
def test_token_forward_matches_pixels_and_preserves_all_peft_gradients(kind):
    model = make_model(kind)
    images, labels = torch.randn(3, 3, 4, 4), torch.arange(3)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=.02)
    F.cross_entropy(model(images), labels).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = {n:p.clone() for n,p in model.named_parameters()}
    tokens = model.encode_input_tokens(images)
    assert not tokens.requires_grad
    actual, expected = model.forward_tokens(tokens), model(images)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(F.cross_entropy(actual, labels), parameters)
    b = torch.autograd.grad(F.cross_entropy(expected, labels), parameters)
    for lhs,rhs in zip(a,b):
        torch.testing.assert_close(lhs,rhs)
    assert all(x.norm() > 0 for x in a)
    # Arbitrary synthetic tokens still train visual and, for LoRA, text parameters.
    fake = tokens.clone()
    fake[:, 1:] += torch.randn_like(fake[:, 1:]) * .1
    grads = torch.autograd.grad(F.cross_entropy(model.forward_tokens(fake), labels), parameters)
    assert all(torch.isfinite(x).all() and x.norm() > 0 for x in grads)
    for n,p in model.named_parameters():
        torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    client = model.create_client_model(0)
    torch.testing.assert_close(client.forward_tokens(tokens), model.forward_tokens(tokens))
    with pytest.raises(ValueError,match="input tokens"):
        model.forward_tokens(tokens[:, 1:])


def test_covariance_factor_uses_sample_divisor_and_square_root_eigenvalues():
    x = torch.randn(8, 30)
    x -= x.mean(0)
    factor, meta = low_rank_factor(x, 20)
    torch.testing.assert_close(factor @ factor.T, x.T @ x / len(x), atol=1e-6,rtol=1e-5)
    assert meta["used_rank"] <= 7
    assert meta["retained_variance"] == pytest.approx(1)


def test_independent_auc_counts_tied_member_nonmember_pairs_as_half():
    # Pair wins: 0.5+1 for member score 2, 0+1 for member score 1.
    assert recompute_auc([1,0,1,0],[2,2,1,0]) == pytest.approx(2.5/4)
    assert recompute_auc([1,0],[1,1]) == .5
    with pytest.raises(ValueError,match="finite"):
        recompute_auc([1,0],[float("nan"),1])


def test_conditional_auc_removes_cross_class_score_comparisons():
    # Both within-class pairs are correctly ordered. Cross-class pairs include
    # one score-baseline inversion, so the ordinary AUC is only .75.
    membership, scores, classes = [1, 1, 0, 0], [11., 1., 10., 0.], [0, 1, 0, 1]
    assert recompute_auc(membership, scores) == .75
    result = class_diagnostics(membership, scores, classes)
    assert result["class_conditional_auc"] == result["macro_class_auc"] == 1.
    assert result["conditional_pair_count"] == 2
    assert all(row["tpr_at_global_1pct_fpr"] is None for row in result["classes"])


def test_resampling_metrics_treat_ties_as_indivisible_threshold_groups():
    membership = np.array([1, 1, 0, 0])
    scores = np.array([[1., 2., 1., 1.], [1., 1., 1., 1.]])
    auc, tpr = score_metrics(scores, membership)
    np.testing.assert_array_equal(auc, [.75, .5])
    np.testing.assert_array_equal(tpr, [.5, 0.])


def test_paired_resampling_preserves_null_difference_and_class_membership_strata():
    # Identical model predictions must have zero paired differences for every
    # draw, including two attacks and tied scores. Class-only predictions must
    # retain their AUC when resampling within class/membership strata.
    membership = np.array([1, 1, 1, 0, 0, 0])
    classes = np.array([0, 0, 1, 0, 1, 1])
    attacks = np.array([[.9, .1, .6, .7, .2, .2], classes])
    scores = np.stack([attacks, attacks])
    auc, tpr = paired_resampling(scores, membership, classes, replicates=50, seed=17)
    np.testing.assert_array_equal(auc[:, 1]-auc[:, 0], 0.)
    np.testing.assert_array_equal(tpr[:, 1]-tpr[:, 0], 0.)
    expected, _ = score_metrics(attacks, membership)
    np.testing.assert_array_equal(auc[:, :, 1], expected[1])


def test_explicit_resampling_control_requires_matched_candidate_identities():
    left = dict(run="control", complete=True, comparison_key="same_protocol",
                candidate_metadata={"source":"independent_evaluation"},
                candidate_selection_digests={"fixed":"same_candidates"})
    right = {**copy.deepcopy(left), "run":"treatment"}
    report = dict(runs=[left, right], matched_comparisons=[])
    assert select_pair(report, "treatment", "control") == (left, right)
    right["candidate_selection_digests"]["fixed"] = "other_candidates"
    with pytest.raises(ValueError, match="mismatched"):
        select_pair(report, "treatment", "control")


def test_confirmation_reservation_excludes_exploration_from_both_new_partitions():
    labels = np.repeat([0, 1], 8)
    first = reserve_indices(labels, [0, 8], holdout_per_class=2, train_shots=4, seed=17)
    second = reserve_indices(labels, [0, 8], holdout_per_class=2, train_shots=4, seed=17)
    assert first == second
    training, holdout, old = (set(first[k]) for k in (
        "train_pool_indices", "evaluation_indices", "excluded_exploration_indices"))
    assert not training & holdout and not training & old and not holdout & old
    assert training | holdout | old == set(range(16))
    np.testing.assert_array_equal(np.bincount(labels[sorted(holdout)]), [2, 2])
    np.testing.assert_array_equal(np.bincount(labels[sorted(training)]), [5, 5])


def test_confirmation_reservation_rejects_reusing_or_overallocating_records():
    labels = np.repeat([0, 1], 8)
    with pytest.raises(ValueError, match="unique"):
        reserve_indices(labels, [0, 0], holdout_per_class=2, train_shots=4)
    with pytest.raises(ValueError, match="too few"):
        reserve_indices(labels, [0, 8], holdout_per_class=4, train_shots=4)


def test_generation_center_excludes_source_and_covariance_is_local_within_class():
    codes = torch.randn(8, 12)
    codes[4:] += 100
    labels = torch.tensor([0]*4+[1]*4)
    options = {**DEFAULTS,"noise_scale":0,"class_rank":3,"pooled_rank":8}
    geometry = LocalGeometry(codes,labels,options,"cpu")
    gen = torch.Generator().manual_seed(4)
    torch.testing.assert_close(geometry.sample(codes[0],0,1.,options,gen),codes[1:4].mean(0))
    torch.testing.assert_close(geometry.sample(codes[0],0,0.,options,gen),codes[0])
    residual = torch.cat([codes[:4]-codes[:4].mean(0),codes[4:]-codes[4:].mean(0)])
    torch.testing.assert_close(geometry.pooled @ geometry.pooled.T,residual.T @ residual/8,atol=1e-5,rtol=1e-4)
    assert float((geometry.pooled @ geometry.pooled.T).trace()) < 100


def test_catalog_validates_method_and_keeps_original_membership():
    catalog = load_yaml("configs/experiment_catalog.yaml")
    args = ["--models","clip_adapter,clip_lora","--datasets","cifar100","--defenses","risk_synthesis"]
    with pytest.raises(ValueError,match="没有生成任何兼容任务"):
        build_tasks(catalog,parse_args(args))
    tasks,_ = build_tasks(catalog,parse_args(args+["--methods","fedavg"]))
    assert len(tasks)==2
    direct,_ = build_tasks(catalog,parse_args(args+["--set","aggregator=fedavg"]))
    assert len(direct)==2 and all(t.config["aggregator"]=="fedavg" for t in direct)
    for task in tasks:
        assert task.config["fpl_shots"] == 100
        assert task.config["audit"]["membership_protocol"] == "client_train"
        assert task.config["projres"]["max_candidates"] == 0
        assert task.config["defense"]["formal_dp_enabled"] is False
    with pytest.raises(ValueError,match="replacement_fraction"):
        build_tasks(catalog,parse_args(args+["--methods","fedavg","--set","defense.synthesis.replacement_fraction=2"]))


def test_semantic_rejection_preserves_inputs_and_caps_requests():
    model = make_model("clip_adapter")
    images, labels, indices = torch.randn(12,3,4,4),torch.tensor([0]*6+[1]*6),torch.arange(12)
    tokens = model.encode_input_tokens(images)
    options = {**DEFAULTS,"replacement_fraction":0.25}
    synth = RiskSynthesis({"synthesis":options},42)
    synth.geometry[0] = LocalGeometry(tokens[:,1:].flatten(1),labels,options,"cpu")
    synth.generators[0] = torch.Generator().manual_seed(10)
    synth.request_generators[0] = torch.Generator().manual_seed(10)
    synth.control_generators[0] = torch.Generator().manual_seed(11)
    synth.handle=io.StringIO()
    fields=["round","client","step","sample_id","label","risk","used_risk","requested",
            "accepted","attempts","reason","nearest_distance","source_round","norm_ratio","teacher_margin_delta"]
    synth.writer=csv.DictWriter(synth.handle,fieldnames=fields)
    synth.writer.writeheader()
    synth.exposure[0]={k:torch.zeros(12,dtype=torch.long) for k in ["risk_reads","real_steps","synthetic_steps"]}
    def reject_changed(candidate, _labels):
        exact=(candidate[:,None]==tokens[None,:]).flatten(2).all(2).any(1)
        return torch.where(exact,0.,-1000.)
    synth.margins=reject_changed
    output=synth.transform(model,SimpleNamespace(id=0),images,labels,indices,torch.ones(12),1,0,0)
    torch.testing.assert_close(output,tokens,rtol=0,atol=0)
    rows=list(csv.DictReader(io.StringIO(synth.handle.getvalue())))
    assert 0 < sum(int(r["requested"]) for r in rows) <= 3
    assert all(int(r["accepted"])==0 for r in rows)
    assert all(r["reason"]=="semantic_margin" for r in rows if r["requested"]=="1")
    assert all(int(r["attempts"])==2 for r in rows if r["requested"]=="1")
    assert torch.all(synth.exposure[0]["real_steps"]==1)


def test_request_budget_matches_shuffled_risk_controls_across_batches():
    risk=torch.cat([torch.zeros(6),(torch.arange(26)+.5)/26])
    request=[torch.Generator().manual_seed(42) for _ in range(2)]
    control=[torch.Generator().manual_seed(23) for _ in range(2)]
    differs=False
    for step in range(20):
        left, a=select_requests(risk.roll(step),DEFAULTS,request[0],control[0])
        right,b=select_requests(risk.roll(step),{**DEFAULTS,"mode":"shuffled_risk"},request[1],control[1])
        assert len(a)==len(b)<=8
        torch.testing.assert_close(left[a].sort().values,right[b].sort().values)
        differs=differs or not torch.equal(left,right)
    assert differs


@pytest.mark.parametrize("kind", ["clip_adapter", "clip_lora"])
@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_zero_replacement_matches_ordinary_fedavg_across_rounds(kind, optimizer, tmp_path):
    """The null intervention must preserve batches, optimizer state and uploads.

    Round two executes own/other risk scoring and restoration even when no
    record is replaced; repeated epochs and a short batch exercise that path.
    """
    final_states = []
    for defense in ("none", "risk_synthesis"):
        torch.manual_seed(23)
        model = make_model(kind)
        server = ServerBase(
            device=torch.device("cpu"), dataset_name="toy", model=model,
            train_sets=[dataset(3, 20), dataset(4, 21)],
            test_sets=[dataset(3, 30), dataset(4, 31)], class_names=["a", "b", "c"],
            batch_size=4, eval_batch_size=4, learning_rate=.01,
            num_glob_iters=2, local_epochs=2, total_users=2, user_per_round=2,
            results_dir=str(tmp_path / defense), eval_interval=1,
            aggregator=build_aggregator("fedavg", aggregation_weighting="sample_count"),
            audit_config={"enabled": False, "attacks": []},
            projres_config={"enabled": False},
            defense_config={"name": defense, "synthesis": {"replacement_fraction": 0.}},
            method_config={"client_optimizer": optimizer, "seed": 42,
                           "momentum": .7 if optimizer == "sgd" else 0.,
                           "weight_decay": .01, "max_grad_norm": 0.},
        )
        server.train()
        final_states.append({name: parameter.detach().clone()
                             for name, parameter in model.named_parameters()})
        if defense == "risk_synthesis":
            assert server.defense.synthesis.counts["accepted"] == 0
            assert server.defense.synthesis.counts["visits"] == 84
    assert set(final_states[0]) == set(final_states[1])
    for name in final_states[0]:
        torch.testing.assert_close(final_states[0][name], final_states[1][name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["clip_adapter", "clip_lora"])
def test_full_fedavg_all_attacks_original_membership_and_streamed_exposure(kind,tmp_path):
    model = make_model(kind)
    frozen = {n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    audit = _audit_config()
    audit.update(audit_batch_size=4,grad_sample_chunk_size=2)
    server = ServerBase(
        device=torch.device("cpu"),dataset_name="toy",model=model,
        train_sets=[dataset(3,20),dataset(4,21)],test_sets=[dataset(12,30),dataset(13,31)],
        class_names=["a","b","c"],batch_size=4,eval_batch_size=8,learning_rate=.05,
        num_glob_iters=2,local_epochs=1,total_users=2,results_dir=str(tmp_path),
        user_per_round=2,eval_interval=1,aggregator=build_aggregator("fedavg",aggregation_weighting="sample_count"),
        audit_config=audit,projres_config={"enabled":True,"evaluation_interval":1},
        defense_config={"name":"risk_synthesis","synthesis":{"replacement_fraction":1.,"semantic_filter":False}},
        method_config={"client_optimizer":"sgd","seed":42},
    )
    teacher = {n:p.clone() for n,p in server.defense.synthesis.teacher.named_parameters()}
    summaries = server.train()
    assert server.auditor.errors == {}
    assert {s["attack"] for s in summaries} == ATTACKS
    assert all(s["member_count"]==9 and s["nonmember_count"]==9 for s in summaries)
    projres = next(s for s in summaries if s["attack"]=="projres")
    assert projres["metadata"]["paper_fedsgd_exact"] is False
    assert projres["metadata"]["batch_rank_bound"] is None
    assert server.ctx.aggregation_weights == {0:9/21,1:12/21}
    directory = tmp_path/"risk_synthesis"
    rows = list(csv.DictReader((directory/"synthetic_exposure.csv").open()))
    assert len(rows)==42
    assert sum(int(r["accepted"]) for r in rows if r["round"]=="1")==0
    assert sum(int(r["accepted"]) for r in rows if r["round"]=="2")>0
    assert all(int(r["sample_id"]) < [9,12][int(r["client"])] for r in rows)
    summary = json.loads((directory/"synthesis_summary.json").read_text())
    assert summary["status"]=="completed" and not summary["formal_dp_enabled"]
    mechanism = read_synthesis_mechanism(directory, summary, complete=True)
    assert mechanism["counts"]["visits"] == 42
    assert mechanism["counts"]["accepted"] == sum(int(r["accepted"]) for r in rows)
    assert mechanism["completed_counters_verified"]
    wrong_summary = {**summary, "counts": {**summary["counts"], "visits": 43}}
    with pytest.raises(ValueError, match="disagree"):
        read_synthesis_mechanism(directory, wrong_summary, complete=True)
    state = torch.load(directory/"client_0_distribution.pt",weights_only=False)
    assert len(state["labels"])==9
    assert all(len(g["indices"])==3 for g in state["classes"].values())
    exposure = torch.load(directory/"source_exposure.pt",weights_only=False)
    for v in exposure.values():
        assert torch.all(v["real_steps"]+v["synthetic_steps"]==2)
        assert torch.all(v["risk_reads"]==1)
    for n,p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
    for n,p in server.defense.synthesis.teacher.named_parameters():
        torch.testing.assert_close(p,teacher[n],rtol=0,atol=0)
