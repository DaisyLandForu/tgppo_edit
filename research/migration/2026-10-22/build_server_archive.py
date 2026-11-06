#!/usr/bin/env python3
"""Build a portable research handoff archive without old checkpoints or build artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DATE_DIR = Path(__file__).resolve().parent
ARCHIVE = DATE_DIR / "tgppo_research_handoff_2026-10-22.tar.gz"
MANIFEST = DATE_DIR / "ARCHIVE_CONTENT_MANIFEST.json"
RUN_SUFFIXES = {".json", ".csv", ".md", ".log", ".txt"}
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "build", "dist", ".tox", ".venv"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".pt", ".pkl"}
EXCLUDED_NAMES = {ARCHIVE.name, ARCHIVE.name + ".sha256", MANIFEST.name}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected():
    roots = [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "migration/2026-10-22",
        ROOT / "tgppo/repo",
        ROOT / "tgppo/experiments",
        ROOT / "tgppo/setup",
        ROOT / "tgppo/datasets",
        ROOT / "tgppo/configs",
        ROOT / "tgppo/activate.sh",
        ROOT / "tgppo/run_train.sh",
        ROOT / "tgppo/prepare_baselines.py",
        ROOT / "tgppo/runs",
    ]
    out = []
    for base in roots:
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDED_DIRS for part in rel.parts) or path.name in EXCLUDED_NAMES:
                continue
            if not path.is_file() and not path.is_symlink():
                continue
            if path.is_symlink():
                # Keep only source-tree relative links; never copy absolute old-machine links.
                target = os.readlink(path)
                if os.path.isabs(target):
                    continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            if rel.parts[:2] == ("tgppo", "runs"):
                if path.suffix not in RUN_SUFFIXES or path.stat().st_size > 2_000_000:
                    continue
            if rel.parts[:2] == ("tgppo", "setup") and path.name.endswith((".tar", ".tar.gz", ".zip")):
                continue
            out.append(path)
    return sorted(set(out), key=lambda p: p.relative_to(ROOT).as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    archive = args.output.resolve()
    if archive != ARCHIVE.resolve():
        raise SystemExit("Use the frozen archive output path")
    files = selected()
    mps_train = [p for p in files if p.relative_to(ROOT).parts[:3] == ("tgppo", "datasets", "train_instances") and p.name.endswith(".mps.gz")]
    mps_test = [p for p in files if p.relative_to(ROOT).parts[:3] == ("tgppo", "datasets", "test_instances") and p.name.endswith(".mps.gz")]
    assert len(mps_train) == 25 and len(mps_test) == 66, (len(mps_train), len(mps_test))
    required = [
        "tgppo/runs/stage4-calibration/final_train_only_statistics.json",
        "tgppo/experiments/server-20261022/code/server.py",
        "tgppo/experiments/stage4/metrics_binding/src/pyscipopt/scip.pyx",
        "migration/2026-10-22/server-run/train.sbatch",
        "migration/2026-10-22/PROMPT_服务器Codex.md",
        "tgppo/setup/scip-6.0.1/CMakeLists.txt",
        "tgppo/setup/soplex-4.0.1/CMakeLists.txt",
    ]
    names = {p.relative_to(ROOT).as_posix() for p in files}
    absent = sorted(set(required) - names)
    if absent:
        raise SystemExit("Required files missing: " + ", ".join(absent))
    rows = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if p.is_symlink():
            rows.append({"path": rel, "symlink": os.readlink(p)})
        else:
            rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifest = {
        "schema": "tgppo-server-handoff-v1",
        "date": "2026-10-22",
        "root_in_archive": "research",
        "file_count_excluding_manifest": len(rows),
        "train_mps_count": len(mps_train),
        "test_mps_count": len(mps_test),
        "total_file_bytes_excluding_manifest": sum(row.get("bytes", 0) for row in rows),
        "excluded": [
            "old .pt/.pkl checkpoints and trajectories",
            "compiled .so/.o/.a files, build directories, Git metadata, Python caches",
            "tgppo/downloads duplicate tarball and tgppo/solver binaries",
            "source-machine Conda environment and Codex account sessions",
        ],
        "files": rows,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with tarfile.open(archive, mode="w:gz", compresslevel=5, dereference=False) as tf:
        for p in files + [MANIFEST]:
            tf.add(p, arcname="research/" + p.relative_to(ROOT).as_posix(), recursive=False)
    digest = sha256(archive)
    (archive.parent / (archive.name + ".sha256")).write_text(digest + "  " + archive.name + "\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest, "file_count": len(files) + 1, "train_mps": len(mps_train), "test_mps": len(mps_test)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
