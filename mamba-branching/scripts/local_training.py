#!/usr/bin/env python3
"""Mirror immutable training inputs locally; keep the canonical training tree shared.

Uses only the standard library so staging does not import Torch from shared storage.
The existing trainer still checks all source/environment/data/checkpoint hashes.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another trainer/staging process holds ' + str(path))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def assert_trainer_idle(run):
    # Probe, then release. The unchanged trainer acquires this same canonical
    # lock itself after validation; that remains the authoritative race guard.
    with exclusive(run / 'training/.lock'):
        pass


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Expected a relative immutable input: ' + str(value))
    return path


def copy_checked(source, target, expected):
    if target.is_symlink():
        raise RuntimeError('Cached inputs must be regular files: ' + str(target))
    if target.is_file() and digest(target) == expected:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.' + target.name + '.' + uuid.uuid4().hex)
    value = hashlib.sha256()
    try:
        with source.open('rb') as reader, temporary.open('wb') as writer:
            for block in iter(lambda: reader.read(2 ** 20), b''):
                writer.write(block)
                value.update(block)
            writer.flush()
            os.fsync(writer.fileno())
        if value.hexdigest() != expected:
            raise RuntimeError('Source input hash mismatch: ' + str(source))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def stage(run, cache_root):
    started = time.monotonic()
    record = read_json(run / 'run.json')
    manifest_path = run / 'dataset/manifest.json'
    manifest = read_json(manifest_path)
    run_hash, dataset_hash = digest(run / 'run.json'), digest(manifest_path)
    if manifest['run_sha256'] != run_hash or manifest['plan_sha256'] != record['plan_sha256']:
        raise RuntimeError('Dataset provenance mismatch')
    for relative, expected in [('config.json', record['config_sha256']),
                               ('plan.json', record['plan_sha256'])]:
        if digest(run / relative) != expected:
            raise RuntimeError('Canonical RUN metadata changed: ' + relative)
    for relative, expected in record['sources'].items():
        if digest(ROOT / relative_path(relative)) != expected:
            raise RuntimeError('Frozen source changed: ' + relative)
    inputs = {'config.json': record['config_sha256'], 'plan.json': record['plan_sha256'],
              'run.json': run_hash, 'dataset/manifest.json': dataset_hash}
    for row in manifest['trajectories']:
        for field, hash_field in [('path', 'sha256'), ('index', 'index_sha256')]:
            relative = str(relative_path(row[field]))
            if relative in inputs and inputs[relative] != row[hash_field]:
                raise RuntimeError('Conflicting input digests: ' + relative)
            inputs[relative] = row[hash_field]
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if cache_root.stat().st_dev == run.stat().st_dev:
        raise RuntimeError('Cache and RUN share a filesystem; choose node-local /tmp or /dev/shm')
    key = hashlib.sha256((str(run) + '\n' + dataset_hash).encode()).hexdigest()[:20]
    view = cache_root / key
    with exclusive(cache_root / (key + '.lock')):
        view.mkdir(exist_ok=True)
        sizes = {relative: (run / relative).stat().st_size for relative in inputs}
        required = sum(size for relative, size in sizes.items() if not (view / relative).is_file())
        if shutil.disk_usage(cache_root).free < required + 64 * 2 ** 20:
            raise RuntimeError('Insufficient cache space for ' + str(sum(sizes.values())) + ' input bytes')
        copied = 0
        for number, (relative, expected) in enumerate(inputs.items(), 1):
            copied += copy_checked(run / relative, view / relative, expected)
            if number % 100 == 0:
                print('Staging: {}/{} files verified'.format(number, len(inputs)), flush=True)
        # Never copy checkpoints locally: both launch paths see the same files,
        # locks, sampler orders, logs, selections and atomic checkpoint pointers.
        training = run / 'training'
        training.mkdir(exist_ok=True)
        link = view / 'training'
        if link.is_symlink():
            if link.resolve() != training.resolve():
                raise RuntimeError('Cache training symlink points at another RUN')
        elif link.exists():
            raise RuntimeError('Cache training must be a symlink to the shared RUN')
        else:
            link.symlink_to(training, target_is_directory=True)
        # Detect metadata changes during a long copy rather than launching a
        # stale mirror; the trainer subsequently checks every cached input too.
        if digest(run / 'run.json') != run_hash or digest(manifest_path) != dataset_hash:
            raise RuntimeError('RUN metadata changed during staging')
        report = {'schema': 1, 'canonical_run': str(run), 'local_view': str(view),
                  'host': socket.gethostname(), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                  'run_sha256': run_hash, 'dataset_sha256': dataset_hash,
                  'files_verified': len(inputs), 'input_bytes': sum(sizes.values()),
                  'files_copied': copied, 'staging_seconds': time.monotonic() - started,
                  'launcher_sha256': digest(Path(__file__)),
                  'training_outputs': str(training), 'sources_and_config_unchanged': True}
        atomic_json(view / 'local-training.json', report)
    return view, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--cache-root', default='/tmp/mamba-branching-' + str(os.getuid()))
    parser.add_argument('--prepare-only', action='store_true', help='Copy inputs without starting a trainer')
    parser.add_argument('--stage', choices=['all', 'cl', 'il'], default='all')
    parser.add_argument('--stop-after-updates', type=int, default=0)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    if not args.prepare_only:
        assert_trainer_idle(run)
    view, report = stage(run, Path(args.cache_root).resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return 0
    assert_trainer_idle(run)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    atomic_json(run / 'training/local_launches' / (attempt + '.json'), report)
    command = ['bash', str(ROOT / 'custom_job.sh'), 'train', '--run-dir', str(view),
               '--resume', 'auto', '--stage', args.stage,
               '--stop-after-updates', str(args.stop_after_updates)]
    print('Launching the unchanged trainer; shared RUN = ' + str(run), flush=True)
    os.execvp(command[0], command)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('local-training: ' + str(error), file=sys.stderr)
        sys.exit(2)
