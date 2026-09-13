"""Reconcile trained views with original visits, weights and selection logs."""
import argparse
from collections import Counter
import csv
from contextlib import ExitStack
import itertools
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_synthesis_history import batch_key, fingerprint, require


def verify(directory):
    directory = Path(directory)
    summary_path = directory / 'synthesis_summary.json'
    summary = json.loads(summary_path.read_text())
    options = summary['options']
    k = options.get('views_per_record', 1)
    require(type(k) is int and k > 1 and options['replacement_policy']=='all', 'Not an all-replacement multiview run.')
    require(summary['status']=='completed', 'Cannot verify uncompleted multiview training.')
    require(summary['observation_unit']=='original_visit'
            and summary['original_row_measurement']=='worst_semantic_view'
            and summary['loss_normalization']=='mean_over_views_then_mean_over_original_records'
            and summary['optimizer_steps_per_original_batch']==1, 'Unknown multiview training protocol.')
    originals_path, views_path = directory/'synthetic_exposure.csv', directory/'synthetic_views.csv'
    sources = [summary_path, originals_path, views_path, directory/'source_exposure.pt']
    saved = torch.load(sources[-1], map_location='cpu', weights_only=True)
    labels, counts = {}, {}
    for path in sorted(directory.glob('client_*_distribution.pt')):
        client = int(path.stem.split('_')[1])
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        labels[client] = state['labels']
        counts[client] = torch.zeros(len(labels[client]), dtype=torch.long)
        sources.append(path)
    totals, view_totals = Counter(), Counter()
    batches, candidates_verified, seen = 0, 0, set()
    selection = options['candidate_selection']=='least_local_similarity'
    with ExitStack() as stack:
        originals = itertools.groupby(csv.DictReader(stack.enter_context(originals_path.open())), batch_key)
        views = itertools.groupby(csv.DictReader(stack.enter_context(views_path.open())), batch_key)
        if selection:
            path = directory/'candidate_choices.csv'
            sources.append(path)
            choices = itertools.groupby(csv.DictReader(stack.enter_context(path.open())), batch_key)
        for key, group in originals:
            require(key not in seen, 'Duplicate original batch.')
            seen.add(key)
            original = list(group)
            ids = [int(r['sample_id']) for r in original]
            require(len(set(ids))==len(ids), 'Duplicate original identity in batch.')
            view_key, view_group = next(views, (None, []))
            require(view_key==key, 'Trained view batch missing or reordered.')
            records = list(view_group)
            require([(int(r['view_index']), int(r['sample_id'])) for r in records]
                    == [(v,sid) for v in range(k) for sid in ids], 'Each original requires exactly K ordered views.')
            by_source = {sid:[] for sid in ids}
            candidate_groups = {}
            if selection:
                choice_key, choice_group = next(choices, (None, []))
                require(choice_key==key, 'Missing view-aware candidate batch.')
                for c in choice_group:
                    candidate_key = int(c['view_index']), int(c['sample_id'])
                    require(candidate_key[0] in range(k) and candidate_key[1] in by_source, 'Unknown candidate view/source.')
                    candidate_groups.setdefault(candidate_key, []).append(c)
                    candidates_verified += 1
            for row in records:
                client, sid = int(row['client']), int(row['sample_id'])
                require(client in labels and 0 <= sid < len(labels[client])
                        and int(row['label'])==int(labels[client][sid]), 'View identity/label differs from original source.')
                used, risk = float(row['used_risk']), float(row['risk'])
                require(0 <= used <= 1 and 0 <= risk <= 1 and row['requested']==row['accepted']=='1', 'Invalid replacement/risk.')
                require(math.isclose(float(row['loss_weight']), 1/k, rel_tol=0, abs_tol=1e-12), 'View loss weight is not 1/K.')
                require(math.isclose(float(row['retained_original_fraction']),1-used,rel_tol=0,abs_tol=1e-7), 'Incorrect retention.')
                distance, ratio = float(row['original_distance']), float(row['norm_ratio'])
                require(math.isfinite(distance) and distance > 0
                        and options['norm_ratio_min'] <= ratio <= options['norm_ratio_max'], 'Invalid trained view geometry.')
                attempts, selected = int(row['attempts']), int(row['selected_attempt'])
                require(1 <= selected <= attempts <= options['attempts'], 'Invalid view attempt budget.')
                quality = int(row['quality_passed'])
                margin = float(row['teacher_margin_delta']) if options['semantic_filter'] else 0.
                require(math.isfinite(margin) and quality==int(margin >= -options['margin_tolerance'])
                        and row['reason']==('accepted' if quality else 'best_semantic_candidate'), 'Invalid view semantic fallback.')
                if selection:
                    candidates = candidate_groups.get((int(row['view_index']),sid), [])
                    require(candidates and attempts==options['attempts'], 'Candidate selection stopped early.')
                    attempt_ids = [int(c['attempt']) for c in candidates]
                    require(len(set(attempt_ids))==len(attempt_ids) and min(attempt_ids)>=1
                            and max(attempt_ids)<=attempts, 'Invalid/duplicate candidate attempts.')
                    for c in candidates:
                        delta, cosine = float(c['teacher_margin_delta']), float(c['nearest_teacher_cosine'])
                        require(math.isfinite(delta) and math.isfinite(cosine) and -1.00001<=cosine<=1.00001
                                and 0 <= int(c['nearest_teacher_source_id']) < len(labels[client])
                                and int(c['quality_passed'])==int(delta >= -options['margin_tolerance'])
                                and options['norm_ratio_min']<=float(c['norm_ratio'])<=options['norm_ratio_max'],
                                'Invalid view candidate metrics.')
                    feasible = [c for c in candidates if c['quality_passed']=='1']
                    chosen = (min(feasible,key=lambda c:(float(c['nearest_teacher_cosine']),-float(c['teacher_margin_delta']),int(c['attempt'])))
                              if feasible else max(candidates,key=lambda c:(float(c['teacher_margin_delta']),-int(c['attempt']))))
                    require(selected==int(chosen['attempt']) and all(row[f]==chosen[f] for f in (
                        'norm_ratio','teacher_margin_delta','quality_passed','nearest_teacher_cosine','nearest_teacher_source_id')),
                        'Trained view disagrees with candidate selection.')
                by_source[sid].append(row)
                counts[client][sid] += 1
                view_totals.update(visits=1, requested=1, accepted=1, fallback=0, quality_failed=1-quality)
            for record in original:
                local = by_source[int(record['sample_id'])]
                require(all(all(r[f]==record[f] for f in ('label','risk','used_risk','source_round')) for r in local),
                        'Risk assignment changed between views of one original.')
                worst = min(local,key=lambda r:(float(r['teacher_margin_delta'] or 0),int(r['view_index'])))
                extra = {'views_per_record','quality_passed_views','representative_view_index','total_attempts'}
                require(all(record[f]==worst[f] for f in record if f not in extra), 'Original summary is not its worst semantic view.')
                passed = sum(int(r['quality_passed']) for r in local)
                require(int(record['views_per_record'])==k and int(record['quality_passed_views'])==passed
                        and record['representative_view_index']==worst['view_index']
                        and int(record['total_attempts'])==sum(int(r['attempts']) for r in local), 'Invalid original/view reconciliation.')
                totals.update(visits=1, requested=1, accepted=1, fallback=0, quality_failed=int(passed<k))
            batches += 1
        require(next(views,None) is None and (not selection or next(choices,None) is None), 'Unmatched view/candidate batches remain.')
    require(dict(totals)==summary['counts'] and dict(view_totals)==summary['view_counts'], 'Original/view totals disagree.')
    require(set(saved)==set(counts) and all(torch.equal(saved[c]['synthetic_views'], counts[c])
            and torch.equal(saved[c]['synthetic_steps']*k, counts[c]) for c in counts), 'Source view counters disagree.')
    return dict(status='verified', batches_verified=batches, original_visits_verified=totals['visits'],
                trained_views_verified=view_totals['visits'], candidates_verified=candidates_verified,
                source_hashes={str(p):fingerprint(p) for p in sources},
                scope='Saved view decisions, per-original weights and counters; gradient normalization is tested separately, not inferred from logs.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    if args.output:
        with args.output.open('x') as handle:
            json.dump(result,handle,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
