# Fixed100 test66 CPU evaluation, 2026-11-10

Independent seed11 B0 / B0_new / MLP / MLP+BC plus fresh relpscost. Full66, solver0, 330 tasks, horizon3600, watchdog4200. Four CPU Custom jobs, each4CPU/32GiB/0GPU. No total execution cap; explicit resumability; no original training-source changes.

Launch commands and expected costs: [docs/CUSTOM_JOBS.md](docs/CUSTOM_JOBS.md), [commands.txt](commands.txt).

`plan.json` records exact instance hashes, frozen actor hashes, full-checkpoint provenance and configurations. `prepare.py` verifies existing fixed100 exports and copies them read-only; it never trains or optimizes a SCIP problem. Real frozen-Actor CPU inference and parent/child fork equivalence is checked by `probe.py` before actual evaluation. `tests/test_evaluation.py` uses synthetic results to test coverage, recovery, environment pairing and reporting.

Output: `research/tgppo/runs/paper100-test66-20261110` and `research/docs/methods/tgppo_paper100_test66_20261110`. Original fixed500 evaluation gate is unchanged. No outcome claims before330/330 valid results.
