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
    require(summary['implementation'] == 'local_token_geometry_v12_direct'
            and options['replacement_policy'] == 'all' and options['candidate_selection'] == 'direct'
            and options['semantic_filter'] is False and options['risk_history'] == 'none'
            and not {'attempts','margin_tolerance','min_class_samples','norm_ratio_min','norm_ratio_max'} & set(options),
            'Invalid direct-generation options.')
    require(all(summary.get(k) is False for k in ('norm_ratio_filter_enabled', 'candidate_validity_filter_enabled',
                'semantic_filter_enabled', 'teacher_initialized', 'semantic_quality_measured'))
            and summary['norm_ratio_role'] == 'not_measured'
            and summary['semantic_failure_policy'] == 'not_checked'
            and summary['candidate_generation'] == 'one_draw_per_training_view'
            and summary['candidate_failure_policy'] == 'no_rejection_or_retry', 'Incorrect direct-generation metadata.')
    k = options['views_per_record']
    require(type(k) is int and k >= 1, 'Invalid view count.')
    if k > 1:
        require(summary['original_row_measurement'] == 'first_generated_view'
                and summary['loss_normalization'] == 'mean_over_views_then_mean_over_original_records'
                and summary['optimizer_steps_per_original_batch'] == 1, 'Incorrect direct view weighting metadata.')
    sources = [directory/'synthesis_summary.json', directory/'synthetic_exposure.csv']
    states, exposure, view_exposure, geometry = {}, {}, {}, []
    for path in sorted(directory.glob('client_*_distribution.pt')):
        client = int(path.stem.split('_')[1])
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
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
        sources.append(path)
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
                        and (int(row['source_round']) >= 0 or risk == used == 0), 'Invalid risk assignment.')
                if k > 1:
                    require(math.isclose(float(row['loss_weight']),1/k,abs_tol=1e-12), 'Wrong view loss weight.')
                    view_totals.update(visits=1,requested=1,accepted=1,fallback=0)
                view_exposure[client][sid] += 1
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
