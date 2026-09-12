import copy
import csv
import io
import json
from types import SimpleNamespace

import pytest
import torch

from aggregator.aggregator_builder import build_aggregator
from privacy_defenses.risk_synthesis import (
    DEFAULTS, LocalGeometry, RiskSynthesis, select_requests, synthesis_options, validate_risk_synthesis,
)
from scripts.analyze_risk_synthesis import read_synthesis_mechanism
from servers.serverbase import ServerBase
from test_risk_synthesis import make_model, dataset, deterministic_cpu
from test_clip_peft_fedsgd import ATTACKS, _audit_config


def mechanism(semantic=True):
    model = make_model("clip_adapter")
    images = torch.randn(8, 3, 4, 4)
    labels, indices = torch.tensor([0]*4+[1]*4), torch.arange(8)
    tokens = model.encode_input_tokens(images)
    options = {**DEFAULTS, "semantic_filter": semantic, "norm_ratio_min": 0.01, "norm_ratio_max": 10.0}
    synth = RiskSynthesis({"synthesis": options}, 42)
    synth.geometry[0] = LocalGeometry(tokens[:, 1:].flatten(1), labels, options, "cpu")
    synth.generators[0] = torch.Generator().manual_seed(10)
    synth.handle = io.StringIO()
    fields = ["round", "client", "step", "sample_id", "label", "risk", "used_risk", "requested", "accepted",
              "attempts", "reason", "nearest_distance", "source_round", "norm_ratio", "teacher_margin_delta",
              "quality_passed", "selected_attempt", "retained_original_fraction", "original_distance"]
    synth.writer = csv.DictWriter(synth.handle, fieldnames=fields)
    synth.writer.writeheader()
    synth.exposure[0] = {k: torch.zeros(8, dtype=torch.long) for k in ["risk_reads", "real_steps", "synthetic_steps"]}
    return synth, model, images, labels, indices, tokens


def rows(synth):
    return list(csv.DictReader(io.StringIO(synth.handle.getvalue())))


def test_all_request_positions_do_not_depend_on_risk_or_request_rng():
    risk = torch.tensor([0., .1, .5, 1.])
    for mode in ("risk", "shuffled_risk"):
        used, requests = select_requests(risk, {**DEFAULTS, "mode": mode}, None,
                                        torch.Generator().manual_seed(2))
        assert requests.tolist() == [0, 1, 2, 3]
        torch.testing.assert_close(used.sort().values, risk)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_first_round_and_zero_risk_records_are_changed_virtual_inputs(batch_size):
    synth, model, images, labels, indices, original = mechanism(semantic=False)
    output = synth.transform(model, SimpleNamespace(id=0), images[:batch_size], labels[:batch_size],
                             indices[:batch_size], torch.ones(batch_size), 0, 0, -1)
    assert not output.requires_grad
    torch.testing.assert_close(output[:, 0], original[:batch_size, 0], rtol=0, atol=0)
    assert torch.all((output[:, 1:] - original[:batch_size, 1:]).flatten(1).norm(dim=1) > 0)
    for row in rows(synth):
        assert row["requested"] == row["accepted"] == row["quality_passed"] == "1"
        assert float(row["risk"]) == float(row["used_risk"]) == 0
        assert float(row["retained_original_fraction"]) == 1
        assert float(row["original_distance"]) > 0
    assert synth.counts["accepted"] == batch_size and synth.counts["fallback"] == 0
    assert synth.exposure[0]["real_steps"].sum() == synth.exposure[0]["risk_reads"].sum() == 0


@pytest.mark.parametrize("second_margin,selected", [(-.8, 1), (-.2, 2)])
def test_failed_semantics_keeps_best_generated_candidate_not_original_or_last(second_margin, selected):
    synth, model, images, labels, indices, original = mechanism()
    values = iter([0., -.4, second_margin])
    synth.margins = lambda tokens, targets: torch.full((len(tokens),), next(values))
    candidates = []
    original_sample = synth.geometry[0].sample
    def sample(*args, **kwargs):
        candidate = original_sample(*args, **kwargs)
        candidates.append(candidate.clone())
        return candidate
    synth.geometry[0].sample = sample
    risks = torch.tensor([0., .1, .3, .9, 0., .5, .8, 1.])
    output = synth.transform(model, SimpleNamespace(id=0), images, labels, indices, risks, 1, 1, 0)
    torch.testing.assert_close(output[:, 1:].flatten(1), torch.stack(candidates[(selected-1)*8:selected*8]), rtol=0, atol=0)
    assert torch.all((output[:, 1:] - original[:, 1:]).flatten(1).norm(dim=1) > 0)
    assert synth.counts == dict(visits=8, requested=8, accepted=8, fallback=0, quality_failed=8)
    for row, risk in zip(rows(synth), risks):
        assert row["reason"] == "best_semantic_candidate" and row["quality_passed"] == "0"
        assert int(row["selected_attempt"]) == selected and row["attempts"] == "2"
        assert float(row["retained_original_fraction"]) == pytest.approx(1-float(risk))


def test_invalid_first_geometry_is_retried_and_unchanged_candidates_abort():
    synth, model, images, labels, indices, original = mechanism(semantic=False)
    calls = 0
    def sample(source, *args, **kwargs):
        nonlocal calls
        calls += 1
        return source * 100 if calls <= 8 else source + .01
    synth.geometry[0].sample = sample
    output = synth.transform(model, SimpleNamespace(id=0), images, labels, indices, torch.zeros(8), 0, 0, -1)
    assert calls == 16
    torch.testing.assert_close(output[:, 1:], original[:, 1:] + .01)
    assert all(row["selected_attempt"] == "2" for row in rows(synth))
    synth, model, images, labels, indices, original = mechanism(semantic=False)
    synth.geometry[0].sample = lambda source, *args, **kwargs: source.clone()
    with pytest.raises(RuntimeError, match="No original-image fallback"):
        synth.transform(model, SimpleNamespace(id=0), images, labels, indices, torch.zeros(8), 0, 0, -1)
    assert not rows(synth) and not synth.counts


@pytest.mark.parametrize("override", [dict(replacement_fraction=.25), dict(warmup_rounds=1),
                                     dict(noise_scale=0), dict(mode="mixup"), dict(center_weighting="previous_risk")])
def test_all_replacement_rejects_options_that_reintroduce_original_positions(override):
    config = dict(model_type="clip_lora", aggregator="fedavg", sample_users=2,
                  defense=dict(name="risk_synthesis", synthesis={**DEFAULTS, **override}))
    with pytest.raises(ValueError):
        validate_risk_synthesis(config)


def test_saved_partial_options_remain_legacy_and_empty_new_options_replace_all():
    assert synthesis_options({"replacement_fraction": .25})["replacement_policy"] == "risk_probability"
    assert synthesis_options({"replacement_fraction": .25})["warmup_rounds"] == 1
    assert synthesis_options({}) == DEFAULTS


@pytest.mark.parametrize("kind", ["clip_adapter", "clip_lora"])
def test_all_replacement_fedavg_all_attacks_and_independent_exposure_reconciliation(kind, tmp_path):
    model = make_model(kind)
    audit = _audit_config()
    audit.update(audit_batch_size=4, grad_sample_chunk_size=2)
    server = ServerBase(device=torch.device("cpu"), dataset_name="toy", model=model,
        train_sets=[dataset(3, 20), dataset(4, 21)], test_sets=[dataset(12, 30), dataset(13, 31)],
        class_names=["a", "b", "c"], batch_size=4, eval_batch_size=8, learning_rate=.05,
        num_glob_iters=2, local_epochs=2, total_users=2, user_per_round=2, eval_interval=1,
        results_dir=str(tmp_path), aggregator=build_aggregator("fedavg", aggregation_weighting="sample_count"),
        audit_config=audit, projres_config={"enabled": True, "evaluation_interval": 1},
        defense_config={"name": "risk_synthesis", "synthesis": copy.deepcopy(DEFAULTS)},
        method_config={"client_optimizer": "sgd", "seed": 42})
    teacher = {n: p.clone() for n, p in server.defense.synthesis.teacher.named_parameters()}
    summaries = server.train()
    assert server.auditor.errors == {} and {s["attack"] for s in summaries} == ATTACKS
    assert all(s["member_count"] == s["nonmember_count"] == 9 for s in summaries)
    directory = tmp_path / "risk_synthesis"
    summary = json.loads((directory / "synthesis_summary.json").read_text())
    verified = read_synthesis_mechanism(directory, summary, complete=True)
    assert verified["counts"]["visits"] == verified["counts"]["accepted"] == 84
    assert verified["counts"]["fallback"] == 0
    assert verified["measurement_scope"] == "selected_candidate"
    exposure = torch.load(directory / "source_exposure.pt", weights_only=True)
    for entry in exposure.values():
        assert torch.all(entry["real_steps"] == 0)
        assert torch.all(entry["synthetic_steps"] == 4)
        assert torch.all(entry["risk_reads"] == 2)
    for name, parameter in server.defense.synthesis.teacher.named_parameters():
        torch.testing.assert_close(parameter, teacher[name], rtol=0, atol=0)
    records = list(csv.DictReader((directory / "synthetic_exposure.csv").open()))
    records[0]["original_distance"] = "0"
    with (directory / "synthetic_exposure.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError, match="changed virtual inputs"):
        read_synthesis_mechanism(directory, summary, complete=True)
