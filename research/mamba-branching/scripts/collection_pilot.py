#!/usr/bin/env python3
"""Run one bounded CPU expert-collection diagnostic on a training instance."""
import hashlib
import argparse
import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def main():
    parser = argparse.ArgumentParser()
    train = json.loads((ROOT / 'configs/milp-s-instances.json').read_text())['train']
    parser.add_argument('--instance', choices=train, default='misc03')
    parser.add_argument('--solver-seed', type=int, choices=range(5), default=0)
    parser.add_argument('--random-prefix', type=int, choices=[0, 1, 5, 10, 15], default=0)
    parser.add_argument('--node-limit', type=int, default=30)
    parser.add_argument('--time-limit', type=float, default=10.0)
    args = parser.parse_args()
    if not 1 <= args.node_limit <= 1000 or not 0 < args.time_limit <= 60:
        parser.error('Pilot limits must be at most 1000 nodes and 60 seconds')
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = ROOT / "runs/collection-pilot" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "scope": "bounded CPU diagnostic, not formal collection",
              "limits": {"nodes": args.node_limit, "seconds": args.time_limit},
              "instance": args.instance, "solver_seed": args.solver_seed, "random_prefix_steps": args.random_prefix,
              "training_started": False, "timestamp_utc": stamp}
    try:
        from env import SCIPCollectEnv
        stats, trajectory = SCIPCollectEnv().run_episode(
            instance=str(ROOT / "data/milp-s/train_instances" / (args.instance + '.mps.gz')),
            name=args.instance, explorer="random", expert="relpscost", k=args.random_prefix,
            state_dims={"var_dim": 25, "node_dim": 8, "mip_dim": 53},
            scip_seed=args.solver_seed, cutoff_value=None,
            scip_limits={"node_limit": args.node_limit, "time_limit": args.time_limit},
            scip_params={"heuristics": False, "cutoff": False, "conflict_usesb": False,
                         "probing_bounds": False, "checksol": False, "reevalage": 0},
            verbose=False,
        )
        assert trajectory, "No expert samples collected"
        for sample in trajectory.values():
            cands = sample["cands_state_mat"]
            assert cands.ndim == 2 and cands.shape[1] == 25
            assert sample["node_state"].shape == (8,)
            assert sample["mip_state"].shape == (53,)
            assert 0 <= sample["varRELpos"] < len(cands)
            assert all(np.isfinite(sample[name]).all() for name in
                       ["cands_state_mat", "node_state", "mip_state"])
        file = directory / "data.pkl"
        file.write_bytes(pickle.dumps(trajectory, protocol=4))
        restored = pickle.loads(file.read_bytes())
        assert len(restored) == len(trajectory)
        report.update(status="passed", samples=len(trajectory), solver_stats=stats,
                      trajectory_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    except Exception:
        report["error"] = traceback.format_exc()
    file = directory / "report.json"
    file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(file), "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
