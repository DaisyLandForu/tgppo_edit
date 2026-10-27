# Stage 2 corrected original-input TGPPO

Base: `4f90ccddd7999779c93ff2425ca8dd056113a523`. The upstream checkout is preserved at `/root/research/tgppo/repo`.

This worktree fixes episode GAE, reward time configuration, result semantics and Actor-only evaluation. Network, reward formula and SCIP settings sources are unchanged. No additional search features or new Critic are included. Rollout and differentiable PPO probability recomputation both use Actor eval mode (effective dropout zero). Critic uses eval for collection and train for optimization.

Activate the existing environment, then **override** its upstream PYTHONPATH:

```bash
source /root/research/tgppo/activate.sh
export PYTHONPATH=/root/research/tgppo/experiments/stage2/code
python /root/research/tgppo/experiments/stage2/code/experiment.py --help
python -m unittest discover -s /root/research/tgppo/experiments/stage2/tests -v
```

`prepare_baselines.py`, `train.py` and `test.py` wrap the prepare/train/eval commands. They use explicit instance lists, cutoff sources, seeds, time limits and separate output directories. Baselines must match instance hash, solver seed, limits, settings and cutoff provenance. Task failures remain in job results; a failed collection batch is refused. Job processes have an enforceable wall watchdog, and successful trajectories are merged by job ID. Hard-killed workers do not produce invented terminal rewards.

`latest.pt` contains architecture, protocol, separate networks and optimizers, RNG states, policy version, pending complete episodes and batch progress. Training resumes with `--checkpoint` plus the same explicit config/data/seed/time settings and a **new** output directory; `--batches` is the number of additional batches. Actor-only evaluation reads the stored architecture without constructing a Critic or optimizer. Legacy checkpoint helpers and `tune.py` are outside this protocol; the legacy tuning CLI is disabled.

SCIP time limits are finite task terminals, not proof completion. A verified optimal cutoff can produce `infeasible` with no stored feasible solution; this is a cutoff proof, not infeasibility of the original MILP. Raw gap/bounds and SCIP infinity validity flags are retained. PDI is currently the native integral until actual stop, with no asserted fixed-horizon correction. `fallback_action_count` counts explicit action substitution (disabled); any DIDNOTRUN also sets `native_fallback_possible`, since SCIP may then invoke another branch rule before noticing an interrupt.

Checkpoints restore at whole-episode/batch boundaries, not inside a SCIP search. The generic memory API requires an actual next state for a nonterminal fragment; the current collector only exports complete finite tasks.

Stage-2 short-loop reproduction and Chinese acceptance report:

```bash
bash /root/research/docs/stage2/reproduce.sh /root/research/tgppo/runs/stage2-review-new
```

See `/root/research/docs/阶段2_正确性修复与验收报告_TGPPO_2026-10-21.md`. These are correctness checks, not paper-scale results or method improvements. Stage 3 requires separate user acceptance of stage 2.
