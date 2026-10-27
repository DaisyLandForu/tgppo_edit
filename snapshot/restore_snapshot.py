#!/usr/bin/env python3
"""Verify/recover the complete TGPPO snapshot from its GitHub Release assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import threading


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe archive path: ' + value)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True,
                        help='Directory containing PROJECT_MANIFEST.json and all archive parts')
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().with_name('PROJECT_MANIFEST.json'),
                        help='Trusted manifest from the Git checkout (default: beside this script)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path, help='New or empty directory for the original project')
    group.add_argument('--verify-archive', action='store_true', help='Verify all contents without extracting')
    args = parser.parse_args()
    if not shutil.which('zstd'):
        raise SystemExit('Install the zstd command before restoring this archive.')
    manifest = json.loads(args.manifest.read_text())
    if sha256(args.manifest) != sha256(args.assets / 'PROJECT_MANIFEST.json'):
        raise SystemExit('The asset manifest differs from the trusted Git manifest.')
    expected = {e['path']: e for e in manifest['entries']}
    if len(expected) != len(manifest['entries']):
        raise SystemExit('Duplicate manifest entries.')
    parts = []
    for part in manifest['parts']:
        name = part['name']
        if Path(name).name != name:
            raise SystemExit('Invalid part filename.')
        path = args.assets / name
        if path.stat().st_size != part['size'] or sha256(path) != part['sha256']:
            raise SystemExit('Part verification failed: ' + name)
        print('PART_VERIFIED', name, flush=True)
        parts.append(path)
    output = args.output.resolve() if args.output else None
    if output:
        if output.exists() and any(output.iterdir()):
            raise SystemExit('Output must be empty. Existing experiments will not be overwritten.')
        output.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(['zstd', '-d', '-q', '-c'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    feeder_errors = []

    def feed():
        try:
            with process.stdin:
                for path in parts:
                    with path.open('rb') as stream:
                        shutil.copyfileobj(stream, process.stdin, length=1024 * 1024)
        except BaseException as exc:
            feeder_errors.append(exc)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    seen = set()
    links = []
    file_count = 0
    byte_count = 0
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                relative = safe_relative(member.name).as_posix()
                entry = expected.get(relative)
                if entry is None or relative in seen:
                    raise ValueError('Unexpected or duplicate archive member: ' + relative)
                seen.add(relative)
                target = output / relative if output else None
                if member.isdir() and entry['type'] == 'directory':
                    if target:
                        target.mkdir(parents=True, exist_ok=True)
                elif member.issym() and entry['type'] == 'symlink':
                    if member.linkname != entry['target']:
                        raise ValueError('Symlink target mismatch: ' + relative)
                    links.append((target, member.linkname))
                elif member.isfile() and entry['type'] == 'file':
                    if member.size != entry['size']:
                        raise ValueError('File size mismatch: ' + relative)
                    digest = hashlib.sha256()
                    source = archive.extractfile(member)
                    if target:
                        target.parent.mkdir(parents=True, exist_ok=True)
                    dest = target.open('xb') if target else None
                    try:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            digest.update(block)
                            if dest:
                                dest.write(block)
                    finally:
                        if dest:
                            dest.close()
                        source.close()
                    if digest.hexdigest() != entry['sha256']:
                        raise ValueError('File hash mismatch: ' + relative)
                    if target:
                        os.chmod(target, entry['mode'])
                        os.utime(target, ns=(entry['mtime_ns'], entry['mtime_ns']))
                    file_count += 1
                    byte_count += member.size
                    if file_count % 1000 == 0:
                        print('FILES_VERIFIED', file_count, 'BYTES', byte_count, flush=True)
                else:
                    raise ValueError('Archive member type mismatch: ' + relative)
        # Drain the final tar padding, so zstd and the feeder exit normally.
        while process.stdout.read(1024 * 1024):
            pass
        process.stdout.close()
        feeder.join()
        status = process.wait()
        if status or feeder_errors:
            raise RuntimeError('Decompression failed: status %s, feeder %r' % (status, feeder_errors))
        if seen != set(expected):
            raise ValueError('Missing archive members: %s' % sorted(set(expected) - seen)[:10])
        if byte_count != manifest['source_file_bytes']:
            raise ValueError('Total source size mismatch.')
        if output:
            # Create links last, after extracting regular files without following links.
            for target, link in links:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(link)
            for entry in reversed(manifest['entries']):
                if entry['type'] == 'directory':
                    path = output / entry['path']
                    os.chmod(path, entry['mode'])
                    os.utime(path, ns=(entry['mtime_ns'], entry['mtime_ns']))
        print(json.dumps({'SNAPSHOT_VERIFIED': True, 'regular_files': file_count,
                          'symlinks': len(links), 'source_bytes': byte_count,
                          'output': str(output) if output else None}), flush=True)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


if __name__ == '__main__':
    main()
