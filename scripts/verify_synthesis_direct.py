"""Reconcile direct-generation identities and weights, without candidate filtering."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import itertools
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_synthesis_history import batch_key, fingerprint, require

UNMEASURED = ('quality_passed', 'teacher_margin_delta', 'norm_ratio',
              'original_distance', 'nearest_distance')


def read_mechanism(directory, summary, *, complete):
    directory = Path(directory)
    options = summary['options']
    from scripts.verify_synthesis_global_geometry import verify_center_metadata
    # Historical partial snapshots may lack center metadata entirely.
    if complete or 'generation_center' in summary or options.get('center_source') == 'local_class_mean':
        verify_center_metadata(summary)
    require(summary['implementation'] in {'local_token_geometry_v12_direct',
                                         'local_token_geometry_v13_direct_unit_noise',
                                         'local_token_geometry_v14_direct_scaled_noise',
                                         'local_token_geometry_v15_direct_mixing'}
            and options['replacement_policy'] == 'all' and options['candidate_selection'] == 'direct'
            and options['semantic_filter'] is False and options['risk_history'] == 'none'
            and not {'attempts','margin_tolerance','min_class_samples','norm_ratio_min','norm_ratio_max'} & set(options),
            'Invalid direct-generation options.')
    ablation = summary['implementation'] == 'local_token_geometry_v15_direct_mixing'
    mixing_mode = options.get('mixing_mode', 'risk')
    if ablation:
        require(mixing_mode in ('source', 'class_center', 'risk') and 'mixing_mode' in options
                and summary.get('generation_location_mode') == mixing_mode
                and summary.get('mixing_coefficient_source') == {
                    'risk':'risk', 'source':'constant_zero', 'class_center':'constant_one'}[mixing_mode]
                and summary.get('used_risk_role') == 'class_center_mixing_coefficient'
                and summary.get('risk_ranking_policy') == 'unchanged_when_references_available',
                'Invalid direct mixing metadata.')
    else:
        require('mixing_mode' not in options, 'Historical direct generation cannot declare a mixing ablation.')
    if summary['implementation'] == 'local_token_geometry_v13_direct_unit_noise' or (ablation and 'noise_scale' not in options):
        require('noise_scale' not in options
                and summary.get('noise_scale_parameter_enabled') is False
                and summary.get('generation_noise') == 'covariance_factor_times_standard_normal',
                'Invalid unit-noise direct-generation metadata.')
    else:
        scale = options.get('noise_scale')
        require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0,
                'Invalid direct noise_scale.')
        if summary['implementation'] == 'local_token_geometry_v14_direct_scaled_noise' or ablation:
            require(summary.get('noise_scale_parameter_enabled') is True
                    and summary.get('generation_noise') == 'scaled_covariance_factor_times_standard_normal',
                    'Invalid scaled direct-generation metadata.')
    require(all(summary.get(k) is False for k in ('norm_ratio_filter_enabled', 'candidate_validity_filter_enabled',
                'semantic_filter_enabled', 'teacher_initialized', 'semantic_quality_measured'))
            and summary['norm_ratio_role'] == 'not_measured'
            and summary['semantic_failure_policy'] == 'not_checked'
            and summary['candidate_generation'] == 'one_draw_per_training_view'
            and summary['candidate_failure_policy'] == 'no_rejection_or_retry', 'Incorrect direct-generation metadata.')
    k = options['views_per_record']
    require(type(k) is int and k >= 1, 'Invalid view count.')
    if 'risk_tail_fraction' in options:
        fraction = options['risk_tail_fraction']
        require(type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction <= 1
                and summary.get('risk_tail_fraction') == fraction
                and summary.get('risk_tail_basis') == 'actual_batch', 'Invalid risk tail metadata.')
    if k > 1:
        require(summary['original_row_measurement'] == 'first_generated_view'
                and summary['loss_normalization'] == 'mean_over_views_then_mean_over_original_records'
                and summary['optimizer_steps_per_original_batch'] == 1, 'Incorrect direct view weighting metadata.')
    sources = [directory/'synthesis_summary.json', directory/'synthetic_exposure.csv']
    states, exposure, view_exposure, geometry = {}, {}, {}, []
    from privacy_defenses.synthesis_storage import retained_receipt, RECEIPT
    receipt = retained_receipt(directory, summary)
    if receipt is not None:
        sources.append(directory/RECEIPT)
        local_states = []
        for client, compact in sorted(receipt['clients'].items(), key=lambda item:int(item[0])):
            labels = torch.tensor(compact['labels'], dtype=torch.long)
            local_states.append((int(client), dict(labels=labels,
                classes={int(c):dict(indices=torch.where(labels == int(c))[0]) for c in compact['classes']},
                source_sha256=compact['source_sha256'], geometry_source=compact['geometry_source'])))
    else:
        local_states = []
        for path in sorted(directory.glob('client_*_distribution.pt')):
            local_states.append((int(path.stem.split('_')[1]),
                torch.load(path, map_location='cpu', weights_only=True, mmap=True)))
            sources.append(path)
    require(bool(local_states), 'Missing original-record class metadata.')
    for client, state in local_states:
        require(state['geometry_source']=='local_class_only'
                and not {'semantic_source_features','semantic_class_means','pooled_factor'} & set(state),
                'Direct statistics must not contain teacher or pooled geometry.')
        states[client] = state['labels']
        exposure[client] = {name:torch.zeros(len(state['labels']), dtype=torch.long)
                            for name in ('risk_reads','real_steps','synthetic_steps')}
        view_exposure[client] = torch.zeros(len(state['labels']), dtype=torch.long)
        sizes = [len(group['indices']) for group in state['classes'].values()]
        geometry.append(dict(client=client,samples=len(state['labels']),class_count=len(sizes),
            min_class_samples=min(sizes),max_class_samples=max(sizes),source_sha256=state['source_sha256'],
            geometry_source=state['geometry_source'],pooled_metadata=None))
    totals, view_totals, reasons = Counter(), Counter(), Counter()
    bins = {str(i):Counter() for i in range(5)}
    groups = defaultdict(Counter)
    seen = set()
    with ExitStack() as stack:
        originals = itertools.groupby(csv.DictReader(stack.enter_context(sources[1].open())), batch_key)
        views = None
        if k > 1:
            path = directory/'synthetic_views.csv'; sources.append(path)
            views = itertools.groupby(csv.DictReader(stack.enter_context(path.open())), batch_key)
        for key, rows in originals:
            require(key not in seen, 'Duplicate original batch.')
            seen.add(key)
            rows = list(rows)
            ids = [int(r['sample_id']) for r in rows]
            require(len(set(ids)) == len(ids), 'Duplicate source within a batch.')
            if 'risk_tail_fraction' in options:
                available = [int(r['source_round']) >= 0 for r in rows]
                require(all(available) or not any(available), 'Mixed risk reference availability within a batch.')
                width = math.ceil(options['risk_tail_fraction'] * len(rows)) if all(available) else 0
                expected = [0.] * (len(rows)-width) + [(j+.5)/width for j in range(width)]
                actual = sorted(float(r['risk']) for r in rows)
                require(all(math.isclose(a,b,rel_tol=0,abs_tol=1e-7) for a,b in zip(actual,expected)),
                        'Risk weights disagree with the configured tail fraction.')
            if views is not None:
                view_key, records = next(views, (None, [])); records = list(records)
                require(view_key == key and [(int(r['view_index']),int(r['sample_id'])) for r in records]
                        == [(v,sid) for v in range(k) for sid in ids], 'Missing or misaligned direct views.')
            else:
                records = rows
            for row in records:
                client, sid = int(row['client']), int(row['sample_id'])
                require(client in states and 0 <= sid < len(states[client])
                        and int(row['label']) == int(states[client][sid]), 'Unknown source identity or label.')
                require(row['requested'] == row['accepted'] == row['attempts'] == row['selected_attempt'] == '1'
                        and row['reason'] == 'direct_generated' and all(row[f] == '' for f in UNMEASURED),
                        'Direct draws must not claim retries or quality measurements.')
                risk, used = float(row['risk']), float(row['used_risk'])
                require(0 <= risk <= 1 and 0 <= used <= 1
                        and math.isclose(float(row['retained_original_fraction']),1-used,abs_tol=1e-7)
                        and (int(row['source_round']) >= 0 or risk == 0), 'Invalid risk assignment.')
                expected_used = (0.0 if mixing_mode == 'source' else 1.0 if mixing_mode == 'class_center'
                                 else risk if options['mode'] == 'risk' or int(row['source_round']) < 0 else None)
                require(expected_used is None or math.isclose(used, expected_used, abs_tol=1e-7),
                        'Invalid class-center mixing coefficient.')
                if k > 1:
                    require(math.isclose(float(row['loss_weight']),1/k,abs_tol=1e-12), 'Wrong view loss weight.')
                    view_totals.update(visits=1,requested=1,accepted=1,fallback=0)
                view_exposure[client][sid] += 1
            if mixing_mode == 'risk' and options['mode'] == 'shuffled_risk':
                raw_values = sorted(float(row['risk']) for row in rows)
                used_values = sorted(float(row['used_risk']) for row in rows)
                require(all(math.isclose(a, b, abs_tol=1e-7) for a,b in zip(raw_values, used_values)),
                        'Shuffled risk must preserve the per-batch risk multiset.')
            for j, row in enumerate(rows):
                client, sid = int(row['client']), int(row['sample_id'])
                if k > 1:
                    first = records[j]
                    extras = {'views_per_record','quality_passed_views','representative_view_index','total_attempts'}
                    require(all(row[f] == first[f] for f in row if f not in extras)
                            and row['quality_passed_views'] == '' and row['representative_view_index'] == '0'
                            and int(row['views_per_record']) == int(row['total_attempts']) == k,
                            'Original row must describe the first draw without semantic quality claims.')
                    require(all(all(records[v*len(rows)+j][f] == row[f] for f in
                            ('risk','used_risk','source_round','label')) for v in range(k)), 'Risk differs across views.')
                count = dict(visits=1,requested=1,accepted=1,fallback=0)
                totals.update(count); reasons[row['reason']] += 1
                risk_bin = min(4,int(float(row['risk'])*5)); bins[str(risk_bin)].update(count)
                group = groups[(int(row['round']),client,risk_bin)]; group.update(count)
                group['used_risk_sum'] += float(row['used_risk'])
                group['accepted_used_risk_sum'] += float(row['used_risk'])
                exposure[client]['risk_reads'][sid] += int(int(row['source_round']) >= 0)
                exposure[client]['synthetic_steps'][sid] += 1
        require(views is None or next(views,None) is None, 'Unmatched direct views.')
    global_evidence = None
    if complete:
        require(summary['status']=='completed' and dict(totals)==summary['counts']
                and {key:dict(value) for key,value in bins.items()} == summary['risk_bins'], 'Direct count mismatch.')
        saved_path = directory/'source_exposure.pt'; sources.append(saved_path)
        saved = torch.load(saved_path,map_location='cpu',weights_only=True)
        require(set(saved)==set(exposure) and all(torch.equal(saved[c][key],value)
                for c, fields in exposure.items() for key,value in fields.items()), 'Source count mismatch.')
        if k > 1:
            require(dict(view_totals)==summary['view_counts'] and all(torch.equal(saved[c]['synthetic_views'],value)
                    for c,value in view_exposure.items()), 'View count mismatch.')
        if summary['shared_geometry']:
            from scripts.verify_synthesis_global_geometry import verify
            global_evidence = verify(directory)
    evidence = dict(status='verified' if complete else 'partial_snapshot',batches_verified=len(seen),
        original_visits_verified=totals['visits'],trained_views_verified=totals['visits']*k,
        candidates_verified=0,scope='Direct source/view identities, weights and counters; no candidate validity or semantic test.',
        source_hashes={str(path):fingerprint(path) for path in sources} if complete else {})
    if receipt is not None:
        evidence.update(statistics_storage='compact_receipt', full_geometry_replay_available=False)
    return dict(counts=dict(totals),reasons=dict(reasons),geometry=geometry,
        groups=[dict(round=t,client=c,risk_bin=r,**count,requested_fraction=1.,accepted_fraction=1.,
                     acceptance_given_request=1.) for (t,c,r),count in sorted(groups.items())],
        global_geometry_evidence=global_evidence,measurement_scope='direct_unchecked_views',
        last_attempt_measurements=[],completed_counters_verified=complete,anchor_history_rows_verified=None,
        **(dict(multiview_evidence=evidence,view_counts=dict(view_totals)) if k > 1 else {}),
        direct_evidence=evidence)


def verify(directory):
    directory = Path(directory)
    summary = json.loads((directory/'synthesis_summary.json').read_text())
    return read_mechanism(directory,summary,complete=True)['direct_evidence']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    if args.output:
        with args.output.open('x') as handle:
            json.dump(result,handle,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
