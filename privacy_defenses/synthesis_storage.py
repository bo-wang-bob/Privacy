"""Retain compact evidence before retiring this run's large synthesis caches.

Verified successful-run cleanup. Unverified terminal cleanup lives separately
in synthesis_cleanup and must never be interpreted as geometry verification.
"""
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from time import perf_counter

import torch

LOGGER = logging.getLogger(__name__)
RECEIPT = 'statistics_receipt.json'
RETIRED_STATES = {'prepared', 'cleaned', 'cleanup_incomplete'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summary_identity(summary):
    values = {k:v for k,v in summary.items() if k != 'statistics_storage'}
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def retained_receipt(directory, summary):
    """Read verified compact metadata; never treat an unexplained loss as cleanup."""
    storage = summary.get('statistics_storage') or {}
    if storage.get('status') not in RETIRED_STATES:
        return None
    directory = Path(directory)
    if storage.get('receipt') != RECEIPT:
        raise ValueError('Unexpected statistics cleanup receipt.')
    path = directory/RECEIPT
    if digest(path) != storage.get('receipt_sha256'):
        raise ValueError('Statistics receipt hash mismatch.')
    receipt = json.loads(path.read_text())
    if (receipt.get('schema_version') != 1 or receipt.get('summary_sha256') != summary_identity(summary)
            or summary.get('status') != 'completed'):
        raise ValueError('Statistics receipt does not match the completed run.')
    for name, expected in receipt['retained_file_sha256'].items():
        if Path(name).name != name or digest(directory/name) != expected:
            raise ValueError('Retained statistics evidence hash mismatch.')
    return receipt


def compact_global_evidence(directory, summary):
    receipt = retained_receipt(directory, summary)
    if receipt is None:
        return None
    evidence = receipt.get('global_verification_before_cleanup')
    if not evidence or evidence.get('status') != 'verified':
        raise ValueError('Missing pre-cleanup global geometry verification.')
    # Archived hashes describe deleted files; only existing files are sources
    # for this invocation. Do not claim to have re-read deleted covariances.
    return dict(status='verified_before_cleanup', recipients=evidence['recipients'],
                classes=evidence['classes'], generation_center=evidence['generation_center'],
                covariance_check=evidence['covariance_check'],
                max_covariance_relative_error=evidence['max_covariance_relative_error'],
                covariance_recomputed_now=False, full_geometry_replay_available=False,
                historical_source_hashes=evidence['source_hashes'],
                source_hashes={str(Path(directory)/name):digest(Path(directory)/name)
                               for name in (RECEIPT, 'synthesis_summary.json', *receipt['retained_file_sha256'])},
                statistics_in_existing_attack_view=False)


def _prepare(synth, summary):
    """Finish all validation and build metadata before any file is removed."""
    directory = synth.directory
    from scripts.verify_synthesis_global_geometry import verify
    global_evidence = verify(directory) if summary['shared_geometry'] else None
    clients, paths = {}, []
    exposure = torch.load(directory/'source_exposure.pt', weights_only=True, map_location='cpu')
    if set(exposure) != set(synth.geometry):
        raise ValueError('Incomplete original-record counters before cleanup.')
    total, views = 0, 0
    for client in sorted(synth.geometry):
        path = directory/f'client_{client}_distribution.pt'
        state = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        labels = state['labels']
        counts = {str(c): len(g['indices']) for c,g in state['classes'].items()}
        if sum(counts.values()) != len(labels):
            raise ValueError('Invalid class counts before cleanup.')
        for c,group in state['classes'].items():
            if not torch.equal(group['indices'], torch.where(labels == c)[0]):
                raise ValueError('Class identity mismatch before cleanup.')
        e = exposure[client]
        if any(e[k].shape != labels.shape for k in ('synthetic_steps', 'real_steps', 'risk_reads')):
            raise ValueError('Source counter shape mismatch before cleanup.')
        if e['real_steps'].count_nonzero():
            raise ValueError('Direct synthesis unexpectedly trained original records.')
        total += int(e['synthetic_steps'].sum())
        if summary['options']['views_per_record'] > 1:
            if not torch.equal(e['synthetic_views'], e['synthetic_steps']*summary['options']['views_per_record']):
                raise ValueError('View counters mismatch before cleanup.')
            views += int(e['synthetic_views'].sum())
        clients[str(client)] = dict(labels=labels.tolist(), source_sha256=state['source_sha256'],
            geometry_source=state['geometry_source'], classes={str(c):dict(count=len(g['indices']),
                numerical_rank=g['numerical_rank'], used_rank=g['used_rank'],
                retained_variance=g['retained_variance']) for c,g in state['classes'].items()})
        paths.extend([path, directory/f'client_{client}_source_codes.pt'])
        if summary['shared_geometry']:
            paths.append(directory/f'client_{client}_moment_upload.pt')
    if total != summary['counts']['accepted'] or total != summary['counts']['visits']:
        raise ValueError('Original counters mismatch before cleanup.')
    if summary['options']['views_per_record'] > 1 and views != summary['view_counts']['accepted']:
        raise ValueError('Total view counters mismatch before cleanup.')
    global_classes = {}
    if summary['shared_geometry']:
        paths.append(directory/'global_distribution.pt')
        global_state = torch.load(paths[-1], weights_only=True, map_location='cpu', mmap=True)
        global_classes = {str(c):dict(count=g['count'], numerical_rank=g['numerical_rank'],
            used_rank=g['used_rank'], eigenvalues=g['eigenvalues'].tolist(),
            retained_variance=g['retained_variance'], client_counts=g['client_counts'])
            for c,g in global_state['classes'].items()}
    artifacts = []
    known_hashes = (global_evidence or {}).get('source_hashes', {})
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.parent != directory:
            raise ValueError('Cleanup may only remove owned regular statistics files.')
        artifacts.append(dict(name=path.name, bytes=path.stat().st_size,
                              sha256=known_hashes.get(str(path)) or digest(path)))
    kept = [directory/'source_exposure.pt', *sorted(directory.glob('client_*_global_receipt.json'))]
    return dict(schema_version=1, summary_sha256=summary_identity(summary),
        scope='Original identities/counters and archived geometry checks; no candidate filtering.',
        full_geometry_replay_available=False, clients=clients, global_classes=global_classes,
        global_verification_before_cleanup=global_evidence,
        counter_verification=dict(original_visits=total, trained_views=views if views else total),
        artifacts=artifacts, retained_file_sha256={p.name:digest(p) for p in kept}), paths


def cleanup_on_success(synth, *, audit_succeeded):
    if (synth.directory is None or synth.options.get('statistics_retention', 'keep') not in {'cleanup_on_success', 'cleanup_on_exit'}
            or synth.options['candidate_selection'] != 'direct'):
        return
    directory = synth.directory
    summary_path = directory/'synthesis_summary.json'
    summary = json.loads(summary_path.read_text())
    if summary.get('status') != 'completed' or not audit_succeeded or synth.pending_views:
        return
    if summary.get('statistics_storage'):
        return  # Do not repeat deletion or reinterpret a partially finished cleanup.
    prepared = False
    removed = []
    started = perf_counter()
    try:
        receipt, paths = _prepare(synth, summary)
        atomic_json(directory/RECEIPT, receipt)
        storage = dict(status='prepared', policy=synth.options['statistics_retention'], receipt=RECEIPT,
                       receipt_sha256=digest(directory/RECEIPT), removed_files=[], removed_bytes=0)
        atomic_json(summary_path, {**summary, 'statistics_storage':storage})
        prepared = True
        # Release all live memory maps and GPU references before unlinking.
        synth.device_geometry_cache.clear()
        for geometry in synth.geometry.values():
            geometry.codes = None
            geometry.classes = {}
            geometry.global_distribution = None
        synth.geometry.clear()
        gc.collect()
        for path, entry in zip(paths, receipt['artifacts']):
            path.unlink()
            removed.append(entry)
        storage.update(status='cleaned', removed_files=[r['name'] for r in removed],
                       removed_bytes=sum(r['bytes'] for r in removed), elapsed_seconds=perf_counter()-started)
        atomic_json(summary_path, {**summary, 'statistics_storage':storage})
        synth.statistics_storage = storage
        LOGGER.info('Synthesis statistics cleaned | files=%d | freed=%.2f GiB | receipt=%s',
                    len(removed), storage['removed_bytes']/1024**3, RECEIPT)
    except Exception as error:
        # Never mask a valid completed training result with a housekeeping error.
        # If interrupted after preparation, the compact evidence remains usable.
        storage = (storage if prepared else dict(policy=synth.options['statistics_retention']))
        storage.update(status='cleanup_incomplete' if prepared else 'retained_error',
                       error=f'{type(error).__name__}: {error}',
                       removed_files=[r['name'] for r in removed],
                       removed_bytes=sum(r['bytes'] for r in removed), elapsed_seconds=perf_counter()-started)
        synth.statistics_storage = storage
        try:
            atomic_json(summary_path, {**summary, 'statistics_storage':storage})
        except Exception:
            LOGGER.warning('Could not persist statistics cleanup status.', exc_info=True)
        LOGGER.warning('Statistics cleanup incomplete; see synthesis summary: %s', error)
