# Six independent server training jobs

User-selected scope on 2026-10-22: B0/R0/Rjoint × train seeds11/22, 500 finite-time episodes each, milestones every100, choose best on the frozen development split. Production entry is `code/server.py`; Slurm/portable wrappers and full instructions are in `migration/2026-10-22/server-run/` at the project root.

This is an isolated extension of stage5; historical sources/results are preserved. CUDA learner uses CPU SCIP workers and CPU Actor-only evaluation. The loader saves/restores CUDA RNG and CUDA optimizer state; a target V100 bridge is required before production training. Latest recovery checkpoints precede learning and retain complete raw-unit trajectories; incomplete PPO updates are discarded. Cached jobs are reused only under matching immutable policy/configuration manifests. All attempts/errors remain recorded.

Original Actor/Critic/reward/settings remain byte-identical to upstream. Device/checkpoint changes are in runtime.py; PPO uses the accepted microbatch accumulation, with a wall-clock progress heartbeat. The original optimizer remains Adam across these short-horizon comparisons. B0 is not claimed as an exact AdamW/3600s reproduction.

Local CPU regression checks, real interrupted updates/resume, development selection and Actor-only exports were tested. Target V100 execution and Slurm are pending, not claimed successful. No full500-episode job has been submitted by this delivery. Held-out tests and stage6 are not automatically queued.
