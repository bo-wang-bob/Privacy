"""Generate every training view once, without candidate checks or a teacher."""
from collections import OrderedDict
import math

import torch


class DeviceGeometryCache:
    """Run-scoped LRU of immutable factors/means shared across clients and views.

    CUDA residency is capped at min(2 GiB, one quarter of free device memory
    on first use). This is a memory policy, not a generation hyperparameter.
    Oversized entries are transferred for the current draw without retention.
    """
    def __init__(self, max_bytes=None):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.resident_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.device = None

    def get(self, source, device):
        device = torch.device(device)
        if self.device != device:
            self.clear()
            self.device = device
        if self.max_bytes is None:
            free, _ = torch.cuda.mem_get_info(device)
            self.max_bytes = min(2 * 1024**3, free // 4)
        key = id(source)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][1]
        self.misses += 1
        size = source.numel() * 4  # All generation arithmetic uses float32 factors.
        while self.entries and self.resident_bytes + size > self.max_bytes:
            _, (_, _, old_size) = self.entries.popitem(last=False)
            self.resident_bytes -= old_size
            self.evictions += 1
        value = source.to(device=device, dtype=torch.float32).contiguous()
        if size <= self.max_bytes:
            # Hold the source reference as well: Python IDs must not be reused.
            self.entries[key] = (source, value, size)
            self.resident_bytes += size
        return value

    def clear(self):
        self.entries.clear()
        self.resident_bytes = 0

    def summary(self):
        return dict(policy='lru_min_2gib_quarter_initial_free_cuda_memory',
                    device=str(self.device) if self.device is not None else None,
                    budget_bytes=self.max_bytes, resident_bytes=self.resident_bytes,
                    hits=self.hits, misses=self.misses, evictions=self.evictions)


def mixing_weights(risk, options):
    """Actual class-center coefficients; endpoints also apply without risk history."""
    mode = options.get('mixing_mode', 'risk')
    if mode == 'source':
        return torch.zeros_like(risk)
    if mode == 'class_center':
        return torch.ones_like(risk)
    if mode != 'risk':
        raise ValueError('Direct synthesis.mixing_mode must be source, class_center or risk.')
    return risk


def generation_noise_scale(options):
    """Explicit scales multiply the noise factor; an absent scale preserves v13."""
    scale = options.get('noise_scale', 1.0)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Direct synthesis.noise_scale must be a finite positive number.')
    return float(scale)


def draw_batch(geometry, original, indices, risk, options, generator, cache=None):
    """Class-batched draws; zero-rank and identical candidates are retained."""
    scale = generation_noise_scale(options)
    device = original.device
    risk = mixing_weights(risk, options).to(device)

    def on_device(tensor):
        if cache is not None and device.type == 'cuda':
            return cache.get(tensor, device)
        return tensor.to(device=device, dtype=torch.float32)

    result = torch.empty_like(original)
    labels = geometry.labels[indices]
    for label in labels.unique(sorted=True).tolist():
        positions = torch.where(labels == label)[0]
        ids = indices[positions]
        device_positions = positions.to(device)
        local = geometry.classes[label]
        if options['center_source'] == 'global_class':
            center = on_device(geometry.global_distribution['classes'][label]['mean'])
        elif options['center_source'] == 'local_class_mean':
            # Inclusive local mean isolates center scope from source exclusion.
            # Keep the historical local_class leave-source-out path unchanged.
            center = local['mean'].to(device)
        else:
            n = len(local['indices'])
            center = ((n * local['mean'] - geometry.codes[ids]) / (n - 1)
                      if n > 1 else local['mean'])
            center = center.to(device)
        factor = (geometry.global_distribution['classes'][label]['factor']
                  if options['global_distribution'] == 'generate' else local['factor'])
        factor = on_device(factor)
        # Keep the existing per-client CPU RNG and view/class/record draw order.
        # Only this small [class_batch, rank] matrix crosses to CUDA per draw.
        latent = torch.randn(len(positions), factor.shape[1], generator=generator).to(device)
        weight = risk[device_positions, None]
        result[device_positions] = ((1-weight)*original[device_positions] + weight*center
                             + scale * (latent @ factor.T))
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
    original = tokens[:, 1:].flatten(1).float()
    used = mixing_weights(used, synth.options)
    device_risk = used.to(original.device)
    ids = indices.detach().cpu().long()
    targets = labels.detach().cpu().long()
    views, logs = [], []
    for view_index in range(synth.options['views_per_record']):
        generated = draw_batch(synth.geometry[user.id], original, ids, device_risk,
                               synth.options, synth.generators[user.id], synth.device_geometry_cache)
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
