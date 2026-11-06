# Stage 5: calibrated online PPO

Authorized 2026-10-21: “验收通过，进入下一个阶段”. Isolated from pristine upstream and all prior experiment worktrees. Complete stage5 and stop before stage6.

- New `project/online.py` retains time/frontier in schema3 on-policy trajectories; raw-unit rollouts and GAE, normalized residual MSE only.
- The original Actor/Critic/reward/settings source files remain byte-identical to upstream. `CalibratedCritic` is a separate fully trainable subclass. No frozen cached encoder is used online.
- `Agent` uses backward compatible Critic-forward/loss hooks; default legacy behavior is unchanged.
- Collection adds Critic context before action, without adding Actor inputs. R0 avoids frontier scan. Actor-only loading constructs no Critic, optimizer, or extra extractor.
- Schema3 checkpoints retain configuration, fixed return statistics, both optimizers, RNG, trajectories and policy version. Schema2 remains supported for old artifacts.
- `code/stage5.py` freezes initialization/scheduling/budgets and performs complete-batch checkpoint selection before development evaluation.

Entry points and evidence: `/root/research/docs/stage5/`. See the execution record and frozen JSON before reproducing; use a new output directory for any rerun. Do not overwrite prior results.
