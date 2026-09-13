import csv
import io
import json
from types import SimpleNamespace

import pytest
import torch

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.risk_synthesis import DEFAULTS, select_requests, validate_risk_synthesis
from privacy_defenses.synthesis_history import ZeroRiskHistory, history_assignment, midranks
from privacy_defenses.www_dp import risk_regularization_weights
from servers.serverbase import ServerBase
from test_clip_peft_fedsgd import ATTACKS, _audit_config
from test_risk_synthesis import make_model, dataset, deterministic_cpu
from test_risk_synthesis_all import mechanism, rows


def advanced_mechanism(history=False, selection=False):
    synth, *rest = mechanism()
    extra = []
    if history:
        synth.options["risk_history"] = "zero_risk_frequency"
        extra += ["loss_gap", "history_exposure", "history_rounds", "joint_rank_score", "assigned_risk"]
    if selection:
        synth.options["candidate_selection"] = "least_local_similarity"
        extra += ["nearest_teacher_cosine", "nearest_teacher_source_id"]
        synth.candidate_handle = io.StringIO()
        synth.candidate_writer = csv.DictWriter(synth.candidate_handle, fieldnames=[
            "round", "client", "step", "sample_id", "attempt", "norm_ratio", "teacher_margin_delta",
            "quality_passed", "nearest_teacher_cosine", "nearest_teacher_source_id"])
        synth.candidate_writer.writeheader()
    fields = synth.writer.fieldnames + extra
    synth.handle = io.StringIO()
    synth.writer = csv.DictWriter(synth.handle, fieldnames=fields)
    synth.writer.writeheader()
    return (synth, *rest)


def test_midranks_ties_and_uniform_history_preserve_original_ranking():
    scores = torch.tensor([3., 1., 1., 2.])
    torch.testing.assert_close(midranks(scores), torch.tensor([.875, .25, .25, .625], dtype=torch.float64))
    weights, _, _ = risk_regularization_weights(scores, expected_batch_size=4)
    for history in (torch.zeros(4), torch.ones(4)):
        assigned, _ = history_assignment(scores, history, weights)
        torch.testing.assert_close(assigned, weights, rtol=0, atol=0)
    assigned, _ = history_assignment(torch.zeros(4), torch.arange(4.), weights)
    torch.testing.assert_close(assigned, weights.sort().values, rtol=0, atol=0)


def test_history_freezes_rounds_averages_epochs_and_retains_absent_records():
    h = ZeroRiskHistory()
    ids = torch.arange(3)
    initial, counts = h.values(0, 0, 3, ids)
    assert initial.tolist() == [0, 0, 0] and counts.tolist() == [0, 0, 0]
    h.observe(0, 0, torch.tensor([0, 0, 1, 2]), torch.tensor([0., 1., 0., .5]))
    frozen, _ = h.values(0, 0, 3, ids)
    assert frozen.tolist() == [0, 0, 0]
    values, counts = h.values(0, 4, 3, ids)
    assert values.tolist() == [.5, 1, 0] and counts.tolist() == [1, 1, 1]
    # Three repeats in the next participation round still count as one round.
    h.observe(0, 4, torch.tensor([0, 0, 0, 1]), torch.tensor([0., 0., 0., .5]))
    values, counts = h.values(0, 5, 3, ids)
    assert values.tolist() == [.75, .5, 0] and counts.tolist() == [2, 2, 1]
    with pytest.raises(ValueError):
        h.values(0, 3, 3, ids)


@pytest.mark.parametrize("size", [3, 7, 18, 31])
def test_joint_rank_ties_use_loss_gap_even_for_non_power_of_two_batches(size):
    scores = torch.arange(size, dtype=torch.float64)
    weights, _, _ = risk_regularization_weights(scores, expected_batch_size=size)
    assigned, joint = history_assignment(scores, -scores, weights)
    assert joint.unique().tolist() == [1.0]
    torch.testing.assert_close(assigned, weights, rtol=0, atol=0)


def test_candidate_nearest_neighbor_includes_other_sources_and_other_classes(monkeypatch):
    from privacy_defenses import risk_synthesis as module
    synth, *_ = advanced_mechanism(selection=True)
    synth.teacher = object()
    synth.text = torch.tensor([[1., 0.], [0., 1.]])
    # Source0 would have similarity zero; another local original matches exactly.
    synth.semantic_sources[0] = torch.tensor([[0., 1.], [1., 0.], [-1., 0.]])
    monkeypatch.setattr(module, "token_features", lambda teacher, tokens: torch.tensor([[1., 0.]]))
    margin, cosine, nearest = synth.candidate_metrics(torch.zeros(1, 1, 2), torch.tensor([0]), 0)
    assert margin.item() == cosine.item() == 1 and nearest.item() == 1


def test_joint_shuffle_scrambles_whole_assignment_and_history_counts_used_risk():
    h = ZeroRiskHistory()
    indices = torch.arange(8)
    h.begin(0, 0, 8)
    scores = torch.arange(8.)
    weights, _, _ = risk_regularization_weights(scores, expected_batch_size=8)
    assigned, _ = history_assignment(scores, torch.tensor([1., 1., 1., 0., 0., 0., 0., 0.]), weights)
    used, _ = select_requests(assigned, {**DEFAULTS, "mode": "shuffled_risk"}, None,
                              torch.Generator().manual_seed(31415))
    expected = assigned.float()[torch.randperm(8, generator=torch.Generator().manual_seed(31415))]
    torch.testing.assert_close(used, expected, rtol=0, atol=0)
    torch.testing.assert_close(used.sort().values, weights.float().sort().values, rtol=0, atol=0)
    h.observe(0, 0, indices, used)
    actual, _ = h.values(0, 1, 8, indices)
    torch.testing.assert_close(actual, (used == 0).double())


@pytest.mark.parametrize("margins,cosines,chosen,quality", [
    ((.1, .2), (.6, .9), 1, 1),  # A higher semantic margin cannot override proximity.
    ((.1, .2), (.9, .6), 2, 1),
    ((-.1, .1), (.1, .99), 2, 1),  # Feasibility takes precedence.
    ((-.1, -.2), (.99, .1), 1, 0),  # All failed: maximum semantic margin.
    ((.1, .1), (.6, .6), 1, 1),  # Deterministic earlier-candidate tie.
])
def test_candidate_selection_checks_both_candidates_and_obeys_semantic_fallback(margins, cosines, chosen, quality):
    synth, model, images, labels, indices, original = advanced_mechanism(selection=True)
    synth.margins = lambda tokens, targets: torch.zeros(len(tokens))
    values = iter(zip(margins, cosines))
    def metrics(tokens, labels, client):
        margin, cosine = next(values)
        return torch.full((len(tokens),), margin), torch.full((len(tokens),), cosine), torch.zeros(len(tokens), dtype=torch.long)
    synth.candidate_metrics = metrics
    generated = []
    sample = synth.geometry[0].sample
    def generate(*args, **kwargs):
        candidate = sample(*args, **kwargs)
        generated.append(candidate.clone())
        return candidate
    synth.geometry[0].sample = generate
    output = synth.transform(model, SimpleNamespace(id=0), images, labels, indices, torch.zeros(8), 0, 0, -1)
    assert len(generated) == 16
    torch.testing.assert_close(output[:, 1:].flatten(1), torch.stack(generated[(chosen-1)*8:chosen*8]), rtol=0, atol=0)
    assert all(int(row["selected_attempt"]) == chosen and int(row["quality_passed"]) == quality for row in rows(synth))
    assert len(list(csv.DictReader(io.StringIO(synth.candidate_handle.getvalue())))) == 16
    assert torch.all((output[:, 1:] - original[:, 1:]).flatten(1).norm(dim=1) > 0)


def test_history_changes_only_after_successful_optimizer_commit():
    synth, model, images, labels, indices, _ = advanced_mechanism(history=True)
    synth.margins = lambda tokens, targets: torch.zeros(len(tokens))
    synth.transform(model, SimpleNamespace(id=0), images, labels, indices, torch.zeros(8), 0, 0, -1)
    assert synth.history.clients[0]["visits"].sum() == 0
    with pytest.raises(RuntimeError, match="not committed"):
        synth.transform(model, SimpleNamespace(id=0), images, labels, indices, torch.zeros(8), 0, 0, -1)
    synth.record_optimized_batch(0)
    assert synth.history.clients[0]["visits"].tolist() == [1]*8
    assert synth.pending_history == {}


@pytest.mark.parametrize("options", [
    {"risk_history": "invalid"}, {"candidate_selection": "invalid"},
    {"candidate_selection": "least_local_similarity", "semantic_filter": False},
    {"risk_history": "zero_risk_frequency", "replacement_policy": "risk_probability"},
])
def test_new_policies_reject_inconsistent_configuration(options):
    config = dict(model_type="clip_lora", aggregator="fedavg", sample_users=2,
                  defense=dict(name="risk_synthesis", synthesis={**DEFAULTS, **options}))
    with pytest.raises(ValueError):
        validate_risk_synthesis(config)


@pytest.mark.parametrize("kind,mode", [("clip_adapter", "risk"), ("clip_lora", "shuffled_risk")])
def test_history_and_selection_end_to_end_preserve_original_membership_and_steps(kind, mode, tmp_path):
    audit = _audit_config()
    audit.update(audit_batch_size=4, grad_sample_chunk_size=2)
    server = ServerBase(device=torch.device("cpu"), dataset_name="toy", model=make_model(kind),
        train_sets=[dataset(3, 20), dataset(4, 21)], test_sets=[dataset(12, 30), dataset(13, 31)],
        class_names=["a", "b", "c"], batch_size=4, eval_batch_size=8, learning_rate=.05,
        num_glob_iters=3, local_epochs=2, total_users=2, user_per_round=2, eval_interval=1,
        results_dir=str(tmp_path), aggregator=build_aggregator("fedavg", aggregation_weighting="sample_count"),
        audit_config=audit, projres_config={"enabled": True, "evaluation_interval": 1},
        defense_config={"name": "risk_synthesis", "synthesis": {**DEFAULTS,
            "risk_history": "zero_risk_frequency", "candidate_selection": "least_local_similarity", "mode": mode}},
        method_config={"client_optimizer": "sgd", "seed": 42})
    summary = server.train()
    assert server.auditor.errors == {} and {s["attack"] for s in summary} == ATTACKS
    assert all(s["member_count"] == s["nonmember_count"] == 9 for s in summary)
    synth = server.defense.synthesis
    assert synth.pending_history == {}
    assert synth.counts["visits"] == synth.counts["accepted"] == 126 and synth.counts["fallback"] == 0
    assert all((entry["synthetic_steps"] == 6).all() and (entry["real_steps"] == 0).all()
               for entry in synth.exposure.values())
    directory = tmp_path / "risk_synthesis"
    records = list(csv.DictReader((directory / "synthetic_exposure.csv").open()))
    choices = list(csv.DictReader((directory / "candidate_choices.csv").open()))
    assert len(choices) == 252
    assert {row["history_rounds"] for row in records if row["round"] == "3"} == {"2"}
    assert all(row["attempts"] == "2" for row in records)
    assert json.loads((directory / "synthesis_summary.json").read_text())["implementation"] == "local_token_geometry_v5_history_selection"
    from scripts.verify_synthesis_history import verify
    result = verify(directory)
    assert result["history_visits_verified"] == result["selected_visits_verified"] == 126
    # The independent replay must reject a corrupt historical score.
    records[-1]["history_exposure"] = "0.12345"
    with (directory / "synthetic_exposure.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError, match="history"):
        verify(directory)
