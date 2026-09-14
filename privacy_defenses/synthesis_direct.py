"""Generate every training view once, without candidate checks or a teacher."""
import torch


def draw_batch(geometry, original, indices, risk, options, generator):
    """Class-batched draws; zero-rank and identical candidates are retained."""
    result = torch.empty_like(original)
    labels = geometry.labels[indices]
    for label in labels.unique(sorted=True).tolist():
        positions = torch.where(labels == label)[0]
        ids = indices[positions]
        local = geometry.classes[label]
        if options['center_source'] == 'global_class':
            center = geometry.global_distribution['classes'][label]['mean'].float()
        else:
            n = len(local['indices'])
            center = ((n * local['mean'] - geometry.codes[ids]) / (n - 1)
                      if n > 1 else local['mean'])
        factor = (geometry.global_distribution['classes'][label]['factor']
                  if options['global_distribution'] == 'generate' else local['factor'])
        latent = torch.randn(len(positions), factor.shape[1], generator=generator)
        weight = risk[positions, None]
        result[positions] = ((1-weight)*original[positions] + weight*center
                             + options['noise_scale']*(latent @ factor.T))
    return result


@torch.no_grad()
def generate_views(synth, model, user, images, labels, indices, risk,
                   round_index, step, source_round, raw_scores):
    if user.id in synth.pending_views:
        raise RuntimeError('Previous direct synthesis batch was not committed after optimization.')
    assignment = synth._all_assignment(user, indices, risk, round_index, source_round, raw_scores)
    raw, assigned, _, _, _, used, _ = assignment
    # Encode once and share the original input across all views. No teacher,
    # norm/distance calculation, finite check, equality test, retry or selection.
    tokens = model.encode_input_tokens(images)
    original = tokens[:, 1:].flatten(1).cpu().float()
    ids = indices.detach().cpu().long()
    targets = labels.detach().cpu().long()
    views, logs = [], []
    for view_index in range(synth.options['views_per_record']):
        generated = draw_batch(synth.geometry[user.id], original, ids, used,
                               synth.options, synth.generators[user.id])
        view = tokens.clone()
        view[:, 1:] = generated.reshape_as(tokens[:, 1:]).to(tokens)
        views.append(view.detach())
        logs.append([dict(round=round_index+1, client=user.id, step=step, sample_id=int(sid),
            label=int(targets[j]), risk=float(raw[j]), used_risk=float(used[j]),
            requested=1, accepted=1, attempts=1, selected_attempt=1, reason='direct_generated',
            source_round=source_round, retained_original_fraction=1-float(used[j]),
            nearest_distance=None, norm_ratio=None, original_distance=None,
            teacher_margin_delta=None, quality_passed=None,
            view_index=view_index, loss_weight=1/synth.options['views_per_record'])
            for j, sid in enumerate(ids)])
    synth.pending_views[user.id] = logs
    return tuple(views)


def record_optimized(synth, client):
    views = synth.pending_views.pop(client, None)
    if views is None:
        return
    count = dict(visits=1, requested=1, accepted=1, fallback=0)
    for records in zip(*views):
        first = records[0]
        row = {**first, 'views_per_record': len(records), 'quality_passed_views': None,
               'representative_view_index': 0, 'total_attempts': len(records)}
        synth.writer.writerow({key: row.get(key) for key in synth.writer.fieldnames})
        synth.counts.update(count)
        synth.risk_bins[str(min(4, int(first['risk']*5)))].update(count)
        sid = first['sample_id']
        synth.exposure[client]['risk_reads'][sid] += int(first['source_round'] >= 0)
        synth.exposure[client]['synthetic_steps'][sid] += 1
        if 'synthetic_views' in synth.exposure[client]:
            synth.exposure[client]['synthetic_views'][sid] += len(records)
    if synth.view_handle is not None:
        for records in views:
            for row in records:
                synth.view_writer.writerow({key: row.get(key) for key in synth.view_writer.fieldnames})
                synth.view_counts.update(count)
        synth.view_handle.flush()
    synth.handle.flush()
