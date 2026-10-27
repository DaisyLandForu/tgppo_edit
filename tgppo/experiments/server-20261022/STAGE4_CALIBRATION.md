# Stage 4 supplement: common Critic fitting calibration

User authorization: `先统一校准各组 Critic 的拟合，再开展在线 PPO 对照。`
This worktree preserves the completed stage-4 experiment and its frozen cache.
Calibration is delivered for acceptance before the online stage.

`CalibratedCritic` inherits the stage-4 search Critic without changing its gated
topology, encoder, extra features, Actor, reward, or data split. MSE standardizes
training targets and decodes EVERY value into raw return units. Its last linear
layer starts at zero, so the common initial value is the train-only mean.
HL-Gauss keeps the frozen raw support and Gaussian smoothing; zero last weights
and train-only marginal log-probability biases provide a common prior.

R0-only hyperparameter selection uses training solver seed 0 for fitting and 1
for internal validation. Two learning rates and two update counts were fixed
before those fits. Settings are selected separately by loss, frozen to disk,
and then reused by every group/seed. Development outcomes never select them.

```bash
source /root/research/tgppo/activate.sh
export PYTHONPATH=/root/research/tgppo/experiments/stage4/metrics_binding/src:/root/research/tgppo/experiments/stage4-calibration/code
python /root/research/tgppo/experiments/stage4-calibration/code/calibrate.py select
python /root/research/tgppo/experiments/stage4-calibration/code/calibrate.py fit
python /root/research/docs/stage4_calibration/audit.py
```

Use `docs/stage4_calibration/reproduce.sh` with a fresh output directory for a
separate run. Existing directories are protected. All runs use the original
hashed stage-4 cache and B0 checkpoint; no SCIP data recollection is needed.
The original shared four CPU-core-hour offline cap includes stage-4 costs and
this supplement's pilot/fits, including any failed attempts. GPU is used only
for offline cached tensors; online training is not launched here.

Adapters include all trained parameters, return scale/support buffers, shared
base/cache hashes, selection hash, calibration statistics, and optimizer state.
The read-only audit restores all 24, compares raw decoded values against saved
predictions, and checks train-only statistics and index provenance.

Completed calibration passed the declared MSE training-fit gate. Development
MSE means are R0 57.894 and Rjoint 50.392, versus constant 50.613. The joint-vs-
marginal difference is not stable across the small held-out set and fitting
seeds. These remain fixed-Actor prediction results; online PPO was not run.
