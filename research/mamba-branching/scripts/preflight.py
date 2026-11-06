#!/usr/bin/env python3
"""Independent CPU checks and a real V100/A100 kernel/update probe.

The GPU probe is synthetic and is explicitly NOT a complete MILP training loop.
Every invocation writes separate evidence, including failures.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_check():
    import numpy as np
    import pyscipopt as scip
    expected = Path(os.environ.get("MAMBA_ENV", "/home/duweiyue25/conda/envs/mamba-branching")).resolve()
    assert Path(sys.prefix).resolve() == expected, (sys.prefix, str(expected))
    assert expected in Path(scip.__file__).resolve().parents, scip.__file__
    lock = json.loads((ROOT / "setup/source-lock.json").read_text())
    for name, hashes in lock["files"].items():
        assert sha256(ROOT / "code" / name) == hashes["working_sha256"], name
    for name, hashes in lock.get("additional_files", {}).items():
        assert sha256(ROOT / "code" / name) == hashes["working_sha256"], name
    specs = json.loads((ROOT / "configs/milp-s-instances.json").read_text())
    manifest = json.loads((ROOT / "data/milp-s/manifest.json").read_text())
    assert len(specs["train"]) == 19 and len(specs["test"]) == 8
    assert not set(specs["train"]) & set(specs["test"])
    assert len(manifest) == 27
    for item in manifest:
        assert sha256(ROOT / item["path"]) == item["sha256"], item["name"]
    cutoff = json.loads((ROOT / "data/milp-s/cutoffs.json").read_text())
    assert set(cutoff) == set(specs["test"])
    for item in cutoff.values():
        assert item["status"] == "optimal" and np.isfinite(item["cutoff"])
    model = scip.Model()
    version = model.version()
    assert np.isclose(version, 8.04), version
    methods = ["getCandsState", "getNodeState", "getMIPState", "getChildren", "executeBranchRule", "getFairNNodes", "getPrimalDualIntegral"]
    for name in methods:
        assert hasattr(model, name), name
    evidence = []

    class Probe(scip.Branchrule):
        def branchexeclp(self, allowaddcons):
            candidates, positions, features = self.model.getCandsState(25, len(evidence) + 1)
            node = self.model.getNodeState(8)
            mip = self.model.getMIPState(53)
            assert features.shape == (len(candidates), 25)
            assert len(positions) == len(candidates) and len(set(positions)) == len(positions)
            assert node.shape == (8,) and mip.shape == (53,)
            assert np.isfinite(features).all() and np.isfinite(node).all() and np.isfinite(mip).all()
            # Same candidate count across native PySCIPOpt API and custom features.
            assert len(candidates) == self.model.getLPBranchCands()[3]
            assert np.isclose(node[6], len(candidates) / self.model.getNDiscreteVars())
            evidence.append({"depth": self.model.getCurrentNode().getDepth(), "candidates": len(candidates)})
            result = self.model.executeBranchRule("relpscost", allowaddcons)
            if result == scip.SCIP_RESULT.BRANCHED:
                for child in self.model.getChildren():
                    assert len(child.getBranchInfos()) == 5
            return {"result": result}

    model.hideOutput()
    model.readProblem(str(ROOT / "data/milp-s/train_instances/misc03.mps.gz"))
    model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    model.setParam("limits/time", 10.0)
    model.setParam("limits/nodes", 30)
    model.setParam("randomization/permutationseed", 0)
    model.includeBranchrule(Probe(), "mamba_binding_probe", "SCIP-8 feature probe", 100000000, -1, 1.0)
    model.optimize()
    assert len(evidence) > 0, "No real branching callback was exercised"
    assert any(item["depth"] > 0 for item in evidence), "No non-root features were exercised"
    fair_nodes = model.getFairNNodes(b"relpscost")
    assert fair_nodes >= model.getNNodes()
    result = {"data_hashes_verified": 27, "scip_version": version,
              "pyscipopt_path": scip.__file__, "callback_count": len(evidence),
              "callback_examples": evidence[:6], "solver_status": str(model.getStatus()),
              "nodes": model.getNNodes(), "fair_nodes_relpscost": fair_nodes,
              "primal_dual_integral": model.getPrimalDualIntegral(),
              "scope": "short native binding diagnostic; no learning or formal baseline"}
    model.freeProb()
    return result


def gpu_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No visible CUDA GPU. Run this command inside the allocated V100/A100 node.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one allocated GPU with CUDA_VISIBLE_DEVICES.")
    props = torch.cuda.get_device_properties(0)
    supported = (("V100" in props.name and (props.major, props.minor) == (7, 0))
                 or ("A100" in props.name and (props.major, props.minor) == (8, 0)))
    if not supported:
        raise RuntimeError(f"Expected V100 (sm70) or A100 (sm80), got {props}")
    import causal_conv1d_cuda
    import selective_scan_cuda
    from mamba_ssm import Mamba
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Exercise actual dispatch for the paper's Mamba-1 dimensions and FP32.
    model = Mamba(d_model=8, d_state=64, d_conv=4, expand=2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    x = torch.randn(2, 128, 8, device="cuda")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    loss = out.square().mean()
    assert torch.isfinite(out).all() and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    model.eval()
    with torch.no_grad():
        expected = model(x)
    with tempfile.TemporaryDirectory(prefix="mamba-gpu-probe-") as tmp:
        file = Path(tmp) / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, file)
        saved = torch.load(file, map_location="cuda")
        restored = Mamba(d_model=8, d_state=64, d_conv=4, expand=2).cuda().eval()
        restored.load_state_dict(saved["model"])
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(x), rtol=0, atol=0)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
        restored_optimizer.load_state_dict(saved["optimizer"])
    torch.cuda.synchronize()
    return {"gpu": props.name, "compute_capability": [props.major, props.minor],
            "total_vram_gib": props.total_memory / 2**30, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "loss": loss.item(),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "forward_backward_update_checkpoint": "passed",
            "scope": "synthetic kernel probe; full 100-state batch and MILP loop still required"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cpu", "gpu"])
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    report = {"mode": args.mode, "timestamp_utc": stamp, "host": platform.node(),
              "python": sys.executable, "status": "failed", "training_started": False}
    report["fingerprints"] = {name: sha256(ROOT / name) for name in
                              ["configs/paper-milp-s.json", "setup/source-lock.json", "data/milp-s/manifest.json"]}
    report["cpu_model"] = next((line.split(":", 1)[1].strip() for line in
                                Path("/proc/cpuinfo").read_text().splitlines()
                                if line.startswith("model name")), platform.processor())
    report["cgroup_limits"] = {}
    for filename in ["cpu.max", "memory.max"]:
        file = Path("/sys/fs/cgroup") / filename
        report["cgroup_limits"][filename] = file.read_text().strip() if file.exists() else None
    for name in ["numpy", "torch", "pyscipopt", "mamba-ssm", "causal-conv1d"]:
        try:
            report.setdefault("versions", {})[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report.setdefault("versions", {})[name] = None
    try:
        report["checks"] = cpu_check() if args.mode == "cpu" else gpu_check()
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
    directory = ROOT / "runs/preflight"
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{args.mode}-{stamp}.json"
    file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(file), "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
