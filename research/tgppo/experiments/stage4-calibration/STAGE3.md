# Stage 3: measured 120-second B0 pilot

Preserved upstream base: `4f90ccddd7999779c93ff2425ca8dd056113a523`.
No new Actor/Critic state features, architecture or loss are included.
Actor, Critic, network modules, H3 formula and SCIP settings sources remain identical to upstream.

Activate the existing environment and select the isolated read-only statistics binding:

```bash
source /root/research/tgppo/activate.sh
export PYTHONPATH=/root/research/tgppo/experiments/stage3/metrics_binding/src:/root/research/tgppo/experiments/stage3/code
python /root/research/tgppo/experiments/stage3/code/stage3.py --help
```

The isolated binding is rebuilt by `/root/research/docs/stage3/build_metrics_binding.py`.
It adds only `getIntegralSnapshot()` and the declaration of native integral update time.
It links the same SCIP library and does not overwrite the installed binding.
`project/measurement.py` computes the native tail without modifying solver statistics.
The raw getter still supplies H3. Horizon overshoots receive an explicit integral interval.

`stage3.py prepare|pilot|evaluate --run-root NEW_PATH` implements the stage workflow.
The default pilot retains the historical full-minibatch path; its recorded allocation failure
is preserved. `repair_pilot.py --run-root PATH` reloads the pre-update checkpoint and reuses
the full second batch, with length sorting *within* each original minibatch and weighted
microbatch gradients. Actor and Critic optimizers still step once per logical minibatch.
Use `Agent.learn(microbatch_size=16)` to select this adapter for future common comparisons.
Critic dropout stays enabled: stochastic masks need not be bitwise identical to full batching.

The final delivered model is the last complete 3-epoch update (first 4 episodes, 770 transitions).
The large batch was preserved but its partial updates were discarded after a forecast budget stop.
No development outcomes were used to select the checkpoint.
Training was not sufficient to claim reproduction of paper performance.

Reproduction, including preservation of failures and the explicit forecast CPU guard:

```bash
bash /root/research/docs/stage3/reproduce.sh /root/research/tgppo/runs/stage3-review-new
```

The Chinese report and all result paths are indexed at
`/root/research/docs/阶段3_基线复现与验收报告_TGPPO_2026-10-21.md`.
Stage 4 requires explicit user acceptance of stage 3.
