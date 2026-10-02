"""One-time, per-class moment exchange (Ma et al., CVPR 2025, Eq. 4).

Covariances are represented by spectral factors, never dense D x D matrices.
The server consumes only count, mean and covariance factors, not source codes.
This module simulates communication within the repository's single process.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def spectral_factor(matrix):
    """Return an untruncated numerical spectral factor of matrix @ matrix.T.

    Compute in float64; serialize factors in float32 like the source tokens.
    Only numerical zero directions are discarded, not a configured local rank.
    """
    b = matrix.double()
    if b.ndim != 2 or not torch.isfinite(b).all():
        raise ValueError('Expected a finite covariance factor matrix.')
    if b.shape[1] == 0:
        return b.float().cpu(), torch.empty(0, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(b.T @ b)
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    keep = values > max(float(values[0]) * 1e-10, 1e-16)
    return (b @ vectors[:, keep]).float().cpu(), values[keep].cpu()


def local_moments(geometry, device):
    """Client-side upload. No labels, record IDs or per-record codes in payload."""
    classes = {}
    for label, group in geometry.classes.items():
        rows = geometry.codes[group['indices']].to(device=device, dtype=torch.float64)
        mean = rows.mean(0)
        factor, _ = spectral_factor((rows - mean).T / math.sqrt(len(rows)))
        classes[label] = dict(count=len(rows), mean=mean.cpu(), covariance_factor=factor)
    return dict(schema_version=1, covariance_divisor='n',
                representation='frozen_patch_plus_position_without_cls', classes=classes)


def aggregate_moments(payloads, class_rank, device):
    """Server: weighted within-client covariance PLUS between-client means.

    Full numerical covariance is retained for distribution access. class_rank
    'all' also uses every numerical direction for generation; an integer keeps
    the historical truncation after the complete global aggregation.
    """
    if not payloads:
        raise ValueError('Global distribution requires client statistics.')
    classes = {}
    dimension = None
    for payload in payloads.values():
        if (payload['covariance_divisor'] != 'n' or
                payload['representation'] != 'frozen_patch_plus_position_without_cls'):
            raise ValueError('Incompatible uploaded covariance conventions.')
        for group in payload['classes'].values():
            mean, factor = group['mean'], group['covariance_factor']
            if dimension is None:
                dimension = mean.numel()
            if (type(group['count']) is not int or group['count'] < 1 or mean.shape != (dimension,)
                    or factor.ndim != 2 or factor.shape[0] != dimension
                    or not torch.isfinite(mean).all() or not torch.isfinite(factor).all()):
                raise ValueError('Invalid uploaded class moments.')
    labels = sorted({c for payload in payloads.values() for c in payload['classes']})
    for label in labels:
        groups = {client: p['classes'][label] for client, p in payloads.items() if label in p['classes']}
        count = sum(g['count'] for g in groups.values())
        mean = sum(g['count'] * g['mean'].double() for g in groups.values()) / count
        columns = []
        for group in groups.values():
            weight = math.sqrt(group['count'] / count)
            columns.append(weight * group['covariance_factor'].to(device=device, dtype=torch.float64))
            columns.append((weight * (group['mean'].double() - mean)).to(device)[:, None])
        factor, values = spectral_factor(torch.cat(columns, dim=1))
        used = len(values) if class_rank == 'all' else min(class_rank, len(values))
        classes[label] = dict(count=count, mean=mean, covariance_factor=factor,
            eigenvalues=values, factor=factor[:, :used], used_rank=used,
            numerical_rank=len(values), retained_variance=(float(values[:used].sum()/values.sum()) if len(values) else 0.),
            client_counts={client:g['count'] for client,g in groups.items()})
    return dict(schema_version=1, geometry_source='global_same_class', classes=classes,
        covariance_divisor='n', representation='frozen_patch_plus_position_without_cls',
        aggregation='count_weighted_within_covariance_plus_between_client_mean_covariance',
        covariance_representation='spectral_factor_full_numerical_rank',
        local_rank_truncation_before_aggregation=False,
        generation_rank=class_rank, source='original_training_partition_only',
        contributing_clients=sorted(payloads), computed_before_training_round=1,
        aggregation_count=1, refresh='never', formal_dp_enabled=False)


def exchange(geometries, directory, options, device):
    """One simulated upload/aggregate/broadcast; bind identical data to all clients."""
    directory = Path(directory)
    uploads, upload_hashes = {}, {}
    for client in sorted(geometries):
        path = directory / f'client_{client}_moment_upload.pt'
        torch.save(local_moments(geometries[client], device), path)
        upload_hashes[str(client)] = digest(path)
        uploads[client] = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    state = aggregate_moments(uploads, options['class_rank'], device)
    state['upload_sha256'] = upload_hashes
    path = directory / 'global_distribution.pt'
    torch.save(state, path)
    sha = digest(path)
    shared = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    for client, geometry in geometries.items():
        # Shared read-only by convention, as with the frozen shared backbone.
        # Do not serialize one enormous duplicate per receiving client.
        geometry.global_distribution = shared
        receipt = dict(client=client, artifact=path.name, sha256=sha,
            available_classes=sorted(shared['classes']), received_before_training_round=1,
            generation_uses_global=options['global_distribution'] == 'generate')
        with (directory / f'client_{client}_global_receipt.json').open('x') as handle:
            json.dump(receipt, handle, indent=2)
    return dict(artifact=path.name, sha256=sha, recipient_clients=sorted(geometries),
        available_classes=sorted(shared['classes']), aggregation_count=1,
        computed_before_training_round=1, refresh='never', upload_sha256=upload_hashes,
        shared_fields=['class_count', 'class_mean', 'class_covariance_factor'],
        uploaded_source_codes=False, transport='single_process_simulated_broadcast',
        covariance_representation=shared['covariance_representation'],
        local_rank_truncation_before_aggregation=False,
        statistics_protected_by_dp=False, statistics_in_existing_attack_view=False)
