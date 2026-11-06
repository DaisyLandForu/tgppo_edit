#!/usr/bin/env python3
"""Read-only audit of the original archive and frozen dataset (stdlib only)."""
import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    manifest_path = root / 'migration/2026-10-22/ARCHIVE_CONTENT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    errors, total = [], 0
    for row in manifest['files']:
        path = root / row['path']
        if 'symlink' in row:
            if not path.is_symlink() or os.readlink(path) != row['symlink']:
                errors.append(dict(path=row['path'], error='symlink mismatch'))
        elif not path.is_file():
            errors.append(dict(path=row['path'], error='missing'))
        else:
            size, digest = path.stat().st_size, sha(path)
            total += size
            if size != row['bytes'] or digest != row['sha256']:
                errors.append(dict(path=row['path'], error='size/hash mismatch',
                                   actual_bytes=size, actual_sha256=digest,
                                   expected_sha256=row['sha256']))
    original = json.loads((root / 'docs/stage1/protocol_draft.json').read_text())['data']
    cutoffs_path = root / 'tgppo/datasets/cutoffs.json'
    cutoffs = json.loads(cutoffs_path.read_text())
    downloads = json.loads((root / 'tgppo/datasets/download_manifest.json').read_text())
    download_by_name = {row['name']: row for row in downloads}
    test = json.loads((root / 'migration/2026-10-22/server-run/test66.json').read_text())
    sets, data_errors = {}, []
    for label, directory in [('train_dev', 'train_instances'), ('test', 'test_instances')]:
        sets[label] = []
        for path in sorted((root / 'tgppo/datasets' / directory).glob('*.mps.gz')):
            name, digest = path.name[:-7], sha(path)
            try:
                with gzip.open(path) as stream:
                    while stream.read(1024 * 1024):
                        pass
            except Exception as exc:
                data_errors.append(dict(name=name, error='gzip: ' + str(exc)))
            cutoff = cutoffs.get(name, {})
            if (cutoff.get('status') != 'optimal' or not cutoff.get('source') or
                    not math.isfinite(float(cutoff.get('cutoff', math.nan)))):
                data_errors.append(dict(name=name, error='missing finite optimal cutoff/provenance'))
            downloaded = download_by_name.get(name, {})
            if downloaded.get('sha256') != digest or downloaded.get('bytes') != path.stat().st_size:
                data_errors.append(dict(name=name, error='download manifest mismatch'))
            sets[label].append(dict(name=name, sha256=digest, **cutoff))
    train, dev = original['train'], original['development']
    if {row['name'] for row in sets['train_dev']} != set(train + dev):
        data_errors.append(dict(error='train/development coverage mismatch'))
    for row in sets['train_dev']:
        if row['sha256'] != original['hashes'][row['name']]:
            data_errors.append(dict(name=row['name'], error='frozen instance hash mismatch'))
    if {row['name'] for row in sets['test']} != set(test):
        data_errors.append(dict(error='test coverage mismatch'))
    if set(train) & set(dev) or set(train + dev) & set(test):
        data_errors.append(dict(error='split name overlap'))
    if {row['sha256'] for row in sets['train_dev']} & {row['sha256'] for row in sets['test']}:
        data_errors.append(dict(error='split content overlap'))
    if (len(manifest['files']) != manifest['file_count_excluding_manifest'] or
            total != manifest['total_file_bytes_excluding_manifest']):
        errors.append(dict(error='manifest count/total bytes mismatch'))
    return dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                manifest_sha256=sha(manifest_path), checked_count=len(manifest['files']),
                total_bytes=total, archive_exact=not errors, manifest_errors=errors,
                data_passed=not data_errors, data_errors=data_errors,
                split_counts=dict(train=len(train), development=len(dev), test=len(test)),
                counts={key: len(value) for key, value in sets.items()},
                cutoffs_sha256=sha(cutoffs_path), instances=sets,
                cutoff_scope='packaged optimality records and hashes; no independent re-proof')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier audits; choose a new output path')
    report = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'instances'}, indent=2))
    raise SystemExit(0 if report['archive_exact'] and report['data_passed'] else 1)
