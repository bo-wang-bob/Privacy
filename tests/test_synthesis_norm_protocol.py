import copy
from types import SimpleNamespace

import pytest
import torch

from privacy_defenses.risk_synthesis import DEFAULTS, synthesis_options
from scripts.synthesis_norm_protocol import norm_filter_enabled, valid_norm_diagnostic
from test_risk_synthesis_all import mechanism, rows
from test_synthesis_multiview import multiview_mechanism


@pytest.mark.parametrize('scale', [.01, 100.])
def test_changed_candidates_outside_old_bounds_are_used_without_rescaling(scale):
    synth, model, images, labels, ids, original = mechanism(semantic=False)
    synth.geometry[0].sample = lambda source, *args, **kwargs: source * scale
    output = synth.transform(model, SimpleNamespace(id=0), images, labels, ids,
                             torch.zeros(8), 0, 0, -1)
    torch.testing.assert_close(output[:, 1:], original[:, 1:] * scale, rtol=0, atol=0)
    assert synth.counts['accepted'] == 8 and synth.counts['fallback'] == 0
    for row in rows(synth):
        assert float(row['norm_ratio']) == pytest.approx(scale)
        assert row['selected_attempt'] == '1'
        assert valid_norm_diagnostic(float(row['norm_ratio']), synth.summary())


def test_small_distinct_views_pass_semantics_and_preserve_pending_commit():
    synth, model, images, labels, ids, original = multiview_mechanism()
    synth.options['candidate_selection'] = 'first_semantic'
    synth.margins = lambda tokens, labels: torch.zeros(len(tokens))
    calls = 0
    def sample(source, *args, **kwargs):
        nonlocal calls
        calls += 1
        return source * (.01 if calls <= 8 else .02)
    synth.geometry[0].sample = sample
    views = synth.transform_views(model, SimpleNamespace(id=0), images, labels, ids,
                                  torch.zeros(8), 0, 0, -1)
    assert not synth.counts and len(views) == 2
    for view, scale in zip(views, [.01, .02]):
        torch.testing.assert_close(view[:, 1:], original[:, 1:] * scale, rtol=0, atol=0)
    synth.record_optimized_batch(0)
    assert synth.counts['accepted'] == 8 and synth.view_counts['accepted'] == 16
    assert synth.view_counts['quality_failed'] == 0


@pytest.mark.parametrize('key', ['norm_ratio_min', 'norm_ratio_max'])
def test_removed_bounds_rejected_only_for_current_all_replacement(key):
    with pytest.raises(ValueError, match='removed norm_ratio'):
        synthesis_options({**DEFAULTS, key: 1.})
    assert synthesis_options({'replacement_policy': 'risk_probability', key: 1.})[key] == 1.


def test_auditing_preserves_historical_constraint_and_checks_new_metadata():
    old = dict(implementation='local_token_geometry_v10_global_mean',
               options=dict(replacement_policy='all', norm_ratio_min=.1, norm_ratio_max=2.))
    new = dict(implementation='local_token_geometry_v11_no_norm_filter',
               norm_ratio_filter_enabled=False, norm_ratio_role='diagnostic_only',
               options=dict(replacement_policy='all'))
    assert norm_filter_enabled(old) and not norm_filter_enabled(new)
    for ratio in [.01, 100.]:
        assert not valid_norm_diagnostic(ratio, old)
        assert valid_norm_diagnostic(ratio, new)
    for ratio in [-1., float('nan'), float('inf')]:
        assert not valid_norm_diagnostic(ratio, new)
    with pytest.raises(ValueError, match='Historical'):
        norm_filter_enabled({**old, 'norm_ratio_filter_enabled': False})
    for override in [dict(norm_ratio_filter_enabled=True), dict(norm_ratio_role='candidate_constraint'),
                     dict(options={**new['options'], 'norm_ratio_min': .1})]:
        with pytest.raises(ValueError, match='Inconsistent'):
            norm_filter_enabled({**copy.deepcopy(new), **override})
