#!/usr/bin/env python3
"""Record reviewed working-copy changes against the pinned upstream commit."""
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    upstream = ROOT / "upstream/Mamba-Branching"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    assert commit == "66ab2d6572eaf83c279a07c21445272e8e3cc34f"
    files, patch = {}, []
    for original in sorted(upstream.glob("*.py")):
        working = ROOT / "code" / original.name
        before, after = original.read_bytes(), working.read_bytes()
        files[original.name] = {"upstream_sha256": hashlib.sha256(before).hexdigest(),
                                "working_sha256": hashlib.sha256(after).hexdigest()}
        patch.extend(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                          fromfile="upstream/" + original.name, tofile="code/" + original.name))
    record = {"url": "https://github.com/doctor-watson626/Mamba-Branching", "commit": commit,
              "files": files, "purpose": "Mamba-only imports and explicit paper-protocol fixes; see SOURCE_AUDIT.md",
              "changes": ["Remove unused dependencies in the Mamba path",
                          "Disable all SCIP-8 primal-heuristic frequencies, including adaptivediving and padm",
                          "Validate HDF5 float-stored action indices and explicitly convert them to integer tensor labels"]}
    adapter = ROOT / 'code/paper_training.py'
    if adapter.exists():
        record['additional_files'] = {'paper_training.py': {
            'working_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
            'purpose': 'Diagnostic loss adapter: real-candidate CL and packed causal IL, no formal training entry point'}}
        patch.extend(difflib.unified_diff([], adapter.read_text().splitlines(True),
                                         fromfile='/dev/null', tofile='code/paper_training.py'))
    (ROOT / "setup/source-lock.json").write_text(json.dumps(record, indent=2) + "\n")
    (ROOT / "setup/working-source.patch").write_text("".join(patch))
    print("Recorded", len(files), "source files; upstream", commit)


if __name__ == "__main__":
    main()
