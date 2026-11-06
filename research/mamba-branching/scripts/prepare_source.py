#!/usr/bin/env python3
"""Make a traceable working copy; remove imports unused by Mamba's path."""
import difflib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/Mamba-Branching"
WORK = ROOT / "code"


def main():
    if WORK.exists():
        raise SystemExit(f"Working tree already exists: {WORK}")
    WORK.mkdir()
    changes = []
    hashes = {}
    for path in sorted(UPSTREAM.glob("*.py")):
        before = path.read_text()
        after = before
        if path.name == "main.py":
            for line in ["from peft import LoraConfig, TaskType", "from peft import get_peft_model, PeftModel",
                         "from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer",
                         "from transformers import Trainer,TrainingArguments, DataCollatorForLanguageModeling",
                         "from datasets import load_dataset", "from transformers import get_linear_schedule_with_warmup"]:
                after = after.replace(line + "\n", "")
            after = after.replace("model: PeftModel", "model: torch.nn.Module")
        if path.name == "get_samples.py":
            for line in ["import ecole.observation", "import torch_geometric", "import ecole", "from observation import ExploreThenStrongBranch, TreeFeature"]:
                after = after.replace(line + "\n", "")
        if path.name == "evaluate_all.py":
            for line in ["from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig",
                         "import ecole", "from observation import TreeFeature"]:
                after = after.replace(line + "\n", "")
        if path.name == "convert_pkl_to_h5.py":
            after = after.replace("from sklearn.model_selection import train_test_split\n", "")
        if path.name == "utils.py":
            after += ("\n\n        # SCIP 8 adds heuristics absent from the upstream hard-coded list.\n"
                      "        # The paper disables ALL primal heuristics, including these new plugins.\n"
                      "        for name in model.getParams():\n"
                      "            if name.startswith('heuristics/') and name.endswith('/freq'):\n"
                      "                model.setIntParam(name, -1)\n")
        if path.name == "dataset_scip.py":
            after = after.replace('import h5py\n', 'import h5py\n\n\ndef _read_action_index(value):\n    label = float(value)\n    if not np.isfinite(label) or label < 0 or not label.is_integer():\n        raise ValueError(f"Invalid HDF5 action label: {value!r}")\n    return int(label)\n')
            after = after.replace('torch.LongTensor([x[0]])', 'torch.tensor([_read_action_index(x[0])], dtype=torch.long)')
        # Official Mamba loop uses these bytes names with its custom binding.
        # Keep training math unchanged; explicitly disable SCIP-8 heuristics as in the paper.
        (WORK / path.name).write_text(after)
        hashes[path.name] = {"upstream_sha256": hashlib.sha256(before.encode()).hexdigest(),
                             "working_sha256": hashlib.sha256(after.encode()).hexdigest()}
        changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                           fromfile="upstream/" + path.name, tofile="code/" + path.name))
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    record = {"url": "https://github.com/doctor-watson626/Mamba-Branching", "commit": commit,
              "files": hashes, "purpose": "Mamba-only imports; other model families are outside this working copy's scope"}
    (ROOT / "setup/source-lock.json").write_text(json.dumps(record, indent=2) + "\n")
    (ROOT / "setup/mamba-only-imports.patch").write_text("".join(changes))
    print("Frozen upstream:", commit)
    print("Working Python files:", len(hashes))


if __name__ == "__main__":
    main()
