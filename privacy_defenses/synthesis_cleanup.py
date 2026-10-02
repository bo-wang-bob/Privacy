"""Retire only explicitly owned synthesis caches, including after process failure.

This module uses no tensor deserialization: interrupted/corrupt .pt files are
hashed as bytes. Deletion receipts never certify training or geometry validity.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time

OWNER = 'statistics_owner.json'
MANIFEST = 'statistics_cleanup_manifest.json'
RECEIPT = 'statistics_cleanup.json'
CACHE_NAME = re.compile(r'(?:global_distribution|client_(\d+)_(?:source_codes|distribution|moment_upload))\.pt\Z')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Cleanup metadata must not be a symlink.')
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checked_directory(directory):
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError('Cleanup cannot traverse a symlink directory.')
    if directory.name != 'risk_synthesis' or not directory.is_dir():
        raise ValueError('Expected an existing run/risk_synthesis directory.')
    return directory.resolve(strict=True)


def process_start(pid):
    try:
        # comm may contain spaces or parentheses; field 22 is starttime.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def register_owner(directory, client_ids, policy):
    directory = checked_directory(directory)
    info = directory.stat()
    path = directory / OWNER
    if path.exists():
        raise ValueError('Synthesis cache ownership must be registered once.')
    value = dict(schema_version=1, directory=str(directory), device=info.st_dev, inode=info.st_ino,
                 client_ids=sorted(int(c) for c in client_ids), policy=policy,
                 pid=os.getpid(), process_start=process_start(os.getpid()))
    atomic_json(path, value)
    return value


def owned_caches(directory, client_ids):
    directory = checked_directory(directory)
    clients = set(int(c) for c in client_ids)
    paths = []
    for path in sorted(directory.iterdir()):
        match = CACHE_NAME.fullmatch(path.name)
        if not match or (match.group(1) is not None and int(match.group(1)) not in clients):
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f'Cache must be a regular file, not a symlink: {path.name}')
        paths.append(path)
    return paths


def file_identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Cache is no longer a regular file: {path}')
    return dict(device=info.st_dev, inode=info.st_ino, bytes=info.st_size,
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                allocated_bytes=info.st_blocks * 512, links=info.st_nlink)


def plan_cleanup(directory, client_ids, *, reason):
    directory = checked_directory(directory)
    info = directory.stat()
    artifacts = []
    for path in owned_caches(directory, client_ids):
        identity = file_identity(path)
        sha = digest(path)
        if file_identity(path) != identity:
            raise ValueError(f'Cache changed while planning: {path.name}')
        artifacts.append(dict(name=path.name, sha256=sha, **identity))
    # Preserve hashes of existing summaries; do not overwrite historical results.
    retained = {}
    for name in ('synthesis_summary.json', 'source_exposure.pt', 'statistics_receipt.json'):
        path = directory / name
        if path.is_file() and not path.is_symlink():
            retained[name] = digest(path)
    return dict(schema_version=1, directory=str(directory), device=info.st_dev, inode=info.st_ino,
                client_ids=sorted(int(c) for c in client_ids), reason=reason,
                verification_status='not_performed', full_geometry_replay_available=False,
                artifacts=artifacts, retained_file_sha256=retained)


def apply_cleanup(plan):
    directory = checked_directory(plan['directory'])
    info = directory.stat()
    if (info.st_dev, info.st_ino) != (plan['device'], plan['inode']):
        raise ValueError('Run directory identity changed since cleanup planning.')
    clients = set(plan['client_ids'])
    for name, expected in plan.get('retained_file_sha256', {}).items():
        if Path(name).name != name or (directory / name).is_symlink() or digest(directory / name) != expected:
            raise ValueError('Retained experiment evidence changed since cleanup planning.')
    names = [item['name'] for item in plan['artifacts']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate cache in cleanup plan.')
    # Validate every path and byte stream before the first unlink. No glob delete.
    for item in plan['artifacts']:
        name = item['name']
        match = CACHE_NAME.fullmatch(name)
        if (Path(name).name != name or not match or
                (match.group(1) is not None and int(match.group(1)) not in clients)):
            raise ValueError('Cleanup plan contains a non-cache or foreign-client path.')
        path = directory / name
        if path.exists() or path.is_symlink():
            if file_identity(path) != {key: item[key] for key in file_identity(path)}:
                raise ValueError(f'Cache changed since cleanup planning: {name}')
            if digest(path) != item['sha256']:
                raise ValueError(f'Cache hash changed since cleanup planning: {name}')
    manifest_path = directory / MANIFEST
    if manifest_path.exists():
        if manifest_path.is_symlink() or json.loads(manifest_path.read_text()) != plan:
            raise ValueError('An incompatible cleanup manifest already exists.')
    else:
        atomic_json(manifest_path, plan)
    receipt = dict(schema_version=1, status='prepared', manifest=MANIFEST,
                   manifest_sha256=digest(manifest_path), verification_status='not_performed',
                   reason=plan['reason'], removed_files=[], already_absent_files=[],
                   removed_bytes=0, removed_allocated_bytes=0, full_geometry_replay_available=False)
    old_path = directory / RECEIPT
    if old_path.exists():
        old = read_cleanup(directory)
        if old['manifest_sha256'] != receipt['manifest_sha256']:
            raise ValueError('Cleanup receipt refers to another manifest.')
        receipt = old
    # A durable manifest and prepared receipt must exist before deleting anything.
    atomic_json(old_path, receipt)
    started = time.monotonic()
    try:
        for item in plan['artifacts']:
            path = directory / item['name']
            if not path.exists() and not path.is_symlink():
                if item['name'] not in receipt['removed_files'] and item['name'] not in receipt['already_absent_files']:
                    receipt['already_absent_files'].append(item['name'])
                continue
            identity = file_identity(path)
            if identity != {key: item[key] for key in identity}:
                raise ValueError(f'Cache changed during cleanup: {path.name}')
            path.unlink()
            receipt['removed_files'].append(item['name'])
            receipt['removed_bytes'] += item['bytes']
            receipt['removed_allocated_bytes'] += item['allocated_bytes'] if item['links'] == 1 else 0
            atomic_json(old_path, receipt)
        receipt['status'] = 'cleaned_unverified'
        receipt.pop('error', None)
    except Exception as error:
        receipt.update(status='cleanup_incomplete', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        atomic_json(old_path, receipt)
    return receipt


def read_cleanup(directory):
    directory = Path(directory)
    path = directory / RECEIPT
    if not path.exists():
        return None
    directory = checked_directory(directory)
    if path.is_symlink():
        raise ValueError('Cleanup receipt must not be a symlink.')
    receipt = json.loads(path.read_text())
    manifest = directory / MANIFEST
    if (receipt.get('schema_version') != 1 or receipt.get('manifest') != MANIFEST
            or manifest.is_symlink() or digest(manifest) != receipt.get('manifest_sha256')
            or receipt.get('verification_status') != 'not_performed'):
        raise ValueError('Invalid unverified statistics cleanup receipt.')
    return receipt


def cleanup_owned(directory, *, reason, in_process=False):
    directory = checked_directory(directory)
    owner_path = directory / OWNER
    if not owner_path.exists():
        return None  # Automatic cleanup never adopts historical or foreign files.
    if owner_path.is_symlink():
        raise ValueError('Cache ownership cannot be a symlink.')
    owner = json.loads(owner_path.read_text())
    info = directory.stat()
    if (owner.get('directory'), owner.get('device'), owner.get('inode')) != (str(directory), info.st_dev, info.st_ino):
        raise ValueError('Cache ownership does not match this run directory.')
    if owner.get('policy') != 'cleanup_on_exit':
        return None
    same_process = owner['pid'] == os.getpid() and owner.get('process_start') == process_start(os.getpid())
    alive = owner.get('process_start') is not None and process_start(owner['pid']) == owner['process_start']
    if alive and not (in_process and same_process):
        raise ValueError('Cannot clean caches while their training process is alive.')
    paths = owned_caches(directory, owner['client_ids'])
    if not paths:
        return read_cleanup(directory)
    manifest = directory / MANIFEST
    plan = (json.loads(manifest.read_text()) if manifest.exists() else
            plan_cleanup(directory, owner['client_ids'], reason=reason))
    return apply_cleanup(plan)


def unverified_cleanup(directory, summary=None):
    """Distinguish intentionally discarded evidence from fully verified results."""
    receipt = read_cleanup(directory)
    if receipt is None:
        return None
    # A valid pre-existing success receipt remains sufficient after a retry of
    # an interrupted verified deletion. Its reader validates all retained hashes.
    if summary is not None and (summary.get('statistics_storage') or {}).get('status') in {
            'prepared', 'cleaned', 'cleanup_incomplete'}:
        from privacy_defenses.synthesis_storage import retained_receipt
        if retained_receipt(directory, summary) is not None:
            return None
    return receipt


def require_geometry_available(directory, summary):
    if unverified_cleanup(directory, summary) is not None:
        raise ValueError('Synthesis statistics were intentionally cleaned without full verification; '
                         'see statistics_cleanup.json. Geometry cannot be certified or replayed.')
