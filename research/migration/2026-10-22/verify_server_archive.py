#!/usr/bin/env python3
"""Verify every regular file in an extracted TGPPO handoff package."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "ARCHIVE_CONTENT_MANIFEST.json"


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / MANIFEST.relative_to(ROOT)).read_text())
    bad = []
    for row in manifest["files"]:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            bad.append(str(rel) + ": unsafe path")
            continue
        p = root / rel
        if "symlink" in row:
            if not p.is_symlink() or os.readlink(p) != row["symlink"]:
                bad.append(str(rel) + ": symlink mismatch")
        elif not p.is_file() or p.is_symlink() or p.stat().st_size != row["bytes"] or hash_file(p) != row["sha256"]:
            bad.append(str(rel) + ": size/hash mismatch")
    if bad:
        print("\n".join(bad[:30]))
        raise SystemExit("verification failed: {} files".format(len(bad)))
    print("verified {} files; train MPS {}, test MPS {}".format(
        len(manifest["files"]), manifest["train_mps_count"], manifest["test_mps_count"]))


if __name__ == "__main__":
    main()
