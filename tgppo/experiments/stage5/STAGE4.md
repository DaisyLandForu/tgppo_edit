# Stage 4: fixed-policy search information diagnosis

Authorization: user `进入阶段4`, 2026-10-21. Stage 5 is not authorized here.

This detached worktree includes the accepted stage-3 correctness and metric fixes.
The upstream Actor, Critic, modules, reward and solver-settings sources stay unchanged.
New `project/policy/search_critic.py` implements the Critic-only extension.

`docs/stage4/build_search_binding.py` copies the stage-3 isolated binding and adds
a read-only `getOpenNodeSnapshot()` (all leaves+children+siblings, internal lower
bounds and depths). Installed binding and SCIP shared library stay untouched.

Run with the existing environment and explicit worktree precedence:

```bash
source /root/research/tgppo/activate.sh
export PYTHONPATH=/root/research/tgppo/experiments/stage4/metrics_binding/src:/root/research/tgppo/experiments/stage4/code
python /root/research/tgppo/experiments/stage4/code/stage4.py bridge
python /root/research/tgppo/experiments/stage4/code/stage4.py collect
python /root/research/tgppo/experiments/stage4/code/stage4.py cache
python /root/research/tgppo/experiments/stage4/code/stage4.py fit
python /root/research/docs/stage4/summarize.py
python /root/research/docs/stage4/value_scale_diagnostics.py
```

Existing output is deliberately protected. `--run-root /absolute/new/path` on
`stage4.py` provides a separate new run; `summarize.py --run-root PATH --report-dir PATH`
audits an independent run. Native stage-3 baselines and the fixed accepted stage-3 checkpoint
are explicit hashed dependencies, not rerun or development-selected.

Collection is CPU-only with the original Actor and B0 Critic. Search features
are an independently versioned sidecar of the original episode-preserving
trajectory; each row must match action index, action and candidate count.
Absent branching decisions produce zero samples and retained task records.

Cache uses the frozen B0 Critic encoder in eval mode on CPU, sorted by candidate
length, microbatch 8. Fits use cached pure tensors (existing GPU allowed).
24 variants: R0/Rcap/Rprod/Rjoint × MSE/HL-Gauss × seeds 11/22/33.
All use 500 updates of 256 instance-balanced train-only samples and identical
index schedules for a given fitting seed. The dev split never selects updates.

Final adapter checkpoints store all trainable parameters and the common base
checkpoint/cache hashes. Reconstruct `SearchCritic(base, group, loss)` and load
`trainable_state_dict` with `strict=False`; missing keys must be frozen encoder
or constant return-support buffers only. The delivered audit checks every
adapter against saved predictions and a raw-state end-to-end evaluation.

The result is prediction on a fixed, weakly trained Actor's trajectory
distribution with a frozen encoder. It does not reproduce the paper's result
and does not demonstrate online PPO or solver performance gains.

The completed 500-update fits were worse than a train-mean constant predictor.
Read-only output-range diagnosis found a shared value-scale underfitting issue.
The tiny paired differences must not be interpreted as validated information
benefit. No further fitting or online stage was started after these results.
