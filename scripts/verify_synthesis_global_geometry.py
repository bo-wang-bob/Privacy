"""Verify one-time global class distribution and per-client broadcast receipts.

Covariance equality is checked on eight deterministic probe directions, not by
materializing a dense token covariance or claiming a full source-data replay.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    directory = Path(directory)
    summary_path = directory/'synthesis_summary.json'
    summary = json.loads(summary_path.read_text())
    center_source = summary['options'].get('center_source', 'local_class')
    require(center_source in {'local_class', 'global_class'}, 'Unknown center source.')
    # Preserve verification of historical v9 artifacts without reinterpreting
    # their leave-source-out protocol as the new shared mean.
    inclusive = center_source == 'global_class' and summary['implementation'] in {
        'local_token_geometry_v10_global_mean', 'local_token_geometry_v11_no_norm_filter',
        'local_token_geometry_v12_direct'}
    expected_center = ('global_same_class_mean' if inclusive else
                       f'{center_source.split("_")[0]}_same_class_leave_source_out')
    require(summary['generation_center'] == expected_center,
            'Generation center metadata mismatch.')
    require(summary.get('center_includes_source', False) == inclusive, 'Center inclusion metadata mismatch.')
    if center_source == 'global_class':
        require(summary['options']['global_distribution'] == 'generate' and
                summary['options']['center_weighting'] == 'uniform', 'Invalid global center protocol.')
    evidence = summary['global_distribution']
    require(summary['shared_geometry'] and evidence is not None, 'No global geometry exchange recorded.')
    require(evidence['artifact'] == 'global_distribution.pt', 'Unexpected global artifact.')
    path = directory/evidence['artifact']
    sha = digest(path)
    require(sha == evidence['sha256'], 'Global distribution hash mismatch.')
    state = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    require(state['aggregation_count'] == evidence['aggregation_count'] == 1 and
            state['computed_before_training_round'] == 1 and state['refresh'] == 'never',
            'Global distribution must be computed once before round one.')
    require(state['covariance_divisor'] == 'n' and not state['local_rank_truncation_before_aggregation'],
            'Wrong covariance convention.')
    require(state['upload_sha256'] == evidence['upload_sha256'], 'Upload provenance mismatch.')
    require(state['contributing_clients'] == evidence['recipient_clients'], 'Incomplete recipients.')
    clients = evidence['recipient_clients']
    local_paths = sorted(directory.glob('client_*_distribution.pt'))
    require(sorted(int(p.stem.split('_')[1]) for p in local_paths) == clients, 'Missing client statistics.')
    sources = {str(summary_path):digest(summary_path), str(path):sha}
    uploads = {}
    for client in clients:
        upload_path = directory/f'client_{client}_moment_upload.pt'
        h = digest(upload_path)
        require(h == evidence['upload_sha256'][str(client)], 'Local moment upload hash mismatch.')
        sources[str(upload_path)] = h
        upload = torch.load(upload_path, weights_only=True, map_location='cpu', mmap=True)
        local_path = directory/f'client_{client}_distribution.pt'
        local = torch.load(local_path, weights_only=True, map_location='cpu', mmap=True)
        sources[str(local_path)] = digest(local_path)
        require(set(upload['classes']) == set(local['classes']), 'Uploaded class set mismatch.')
        for c,g in upload['classes'].items():
            require(set(g) == {'count','mean','covariance_factor'}, 'Unexpected uploaded per-class fields.')
            require(g['count'] == len(local['classes'][c]['indices']), 'Uploaded class count mismatch.')
        uploads[client] = upload
        receipt_path = directory/f'client_{client}_global_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        sources[str(receipt_path)] = digest(receipt_path)
        require(receipt == dict(client=client, artifact=path.name, sha256=sha,
            available_classes=sorted(state['classes']), received_before_training_round=1,
            generation_uses_global=summary['options']['global_distribution'] == 'generate'),
            'Invalid or incomplete global broadcast receipt.')
    require(set(state['classes']) == {c for p in uploads.values() for c in p['classes']}, 'Global class union mismatch.')
    generator = torch.Generator().manual_seed(20260913)
    max_relative_error = 0.
    for c,group in state['classes'].items():
        contributors = {k:p['classes'][c] for k,p in uploads.items() if c in p['classes']}
        total = sum(g['count'] for g in contributors.values())
        require(group['count'] == total and group['client_counts'] == {k:g['count'] for k,g in contributors.items()},
                'Incorrect per-class aggregation weights.')
        mean = sum(g['count']*g['mean'].double() for g in contributors.values())/total
        torch.testing.assert_close(group['mean'], mean, atol=1e-10, rtol=1e-10)
        full, factor = group['covariance_factor'].double(), group['factor']
        require(torch.isfinite(full).all() and torch.isfinite(mean).all(), 'Non-finite global statistics.')
        require(factor.shape[1] == min(summary['options']['class_rank'], full.shape[1]), 'Wrong generation rank.')
        torch.testing.assert_close(factor, group['covariance_factor'][:, :factor.shape[1]], atol=0, rtol=0)
        probes = torch.randn(len(mean),8, generator=generator, dtype=torch.float64)
        expected = torch.zeros_like(probes)
        for g in contributors.values():
            f = g['covariance_factor'].double()
            delta = (g['mean'].double()-mean)[:,None]
            expected += (g['count']/total)*(f@(f.T@probes)+delta@(delta.T@probes))
        actual = full@(full.T@probes)
        error = float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
        require(error < 2e-5, 'Global covariance projection mismatch.')
        max_relative_error = max(max_relative_error, error)
    return dict(status='verified', recipients=clients, classes=len(state['classes']),
        generation_center=summary['generation_center'],
        covariance_check='eight_fixed_probe_directions_per_class_from_uploaded_moments',
        max_covariance_relative_error=max_relative_error, source_hashes=sources,
        statistics_in_existing_attack_view=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
