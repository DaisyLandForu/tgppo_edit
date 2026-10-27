"""Read-only upstream diagnostics. No training or source modification.

Run after sourcing tgppo/activate.sh. Output stays under docs/stage1.
The short native solves below are diagnostics, not normalization baselines.
"""
import hashlib
import inspect
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from pyscipopt import Model
from project.agent import Agent
from project.environment import Environment
from project.policy import Actor, Critic
from project.utils import init_params, scip_limits, settings

ROOT = Path('/root/research')
OUT = ROOT / 'docs/stage1'
REPO = ROOT / 'tgppo/repo'
torch.set_num_threads(1)
torch.manual_seed(20261021)


def snapshot():
    files = subprocess.check_output(
        ['git', '-C', str(REPO), 'ls-files', '-z']).decode().split('\0')
    return {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest()
            for f in files if f and (REPO / f).is_file()}


before = snapshot()
report = {'purpose': 'stage1_diagnostics_only', 'training_performed': False,
          'commit': subprocess.check_output(
              ['git', '-C', str(REPO), 'rev-parse', 'HEAD']).decode().strip()}
kwargs = dict(var_dim=25, node_dim=8, mip_dim=53, hidden_dim=256,
              num_heads=8, num_layers=5, dropout=0.05)
actor, critic = Actor(**kwargs), Critic(**kwargs)
report['architecture'] = {
    'actor_parameters': sum(p.numel() for p in actor.parameters()),
    'critic_parameters': sum(p.numel() for p in critic.parameters()),
    'shared_parameter_objects': len({id(p) for p in actor.parameters()} &
                                    {id(p) for p in critic.parameters()})}

context = SimpleNamespace(gamma=1.0, gae_lambda=1.0)
_, returns = Agent._compute_gae(
    context, torch.tensor([1., 2.]), torch.tensor([10., 20.]),
    torch.tensor([0., 1.]), torch.tensor(0.))
report['gae_terminal_case'] = {
    'rewards': [1, 2], 'values': [10, 20], 'dones': [False, True],
    'gamma': 1, 'lambda': 1, 'expected_returns': [3, 2],
    'upstream_returns': returns.tolist(),
    'matches_reference': torch.allclose(returns, torch.tensor([3., 2.]))}

try:
    inspect.signature(Environment.run_episode).bind(None, learn=False)
    signature_error = None
except TypeError as e:
    signature_error = str(e)
report['validation_signature'] = {
    'environment_signature': str(inspect.signature(Environment.run_episode)),
    'training_call_uses_learn_false': 'run_episode(learn=False)' in
        (REPO / 'code/train.py').read_text(), 'bind_error': signature_error}

candidates = torch.randn(1, 9, 25)
node, mip = torch.randn(1, 8), torch.randn(1, 53)
mask = torch.zeros(1, 9, dtype=torch.bool)
with torch.no_grad():
    actor.eval()
    old = actor(candidates, node, mip, mask)
    same = actor(candidates, node, mip, mask)
    actor.train()
    new = actor(candidates, node, mip, mask)
report['dropout_without_update'] = {
    'eval_repeat_max_abs_diff': (old - same).abs().max().item(),
    'eval_vs_train_max_abs_diff': (old - new).abs().max().item(),
    'probability_ratio_min': (new / old).min().item(),
    'probability_ratio_max': (new / old).max().item(),
    'note': 'Synthetic fixed input; no optimizer step. This initialization has nearly uniform actions; the observed probability difference is only numerical scale and does not establish a material PPO bias.'}

class CountCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, cands, node, mip, padding_mask=None):
        self.calls += 1
        return torch.zeros(cands.shape[0], 1)

counter = CountCritic()
logger = logging.getLogger('stage1_diagnostic')
logger.addHandler(logging.NullHandler())
agent = Agent(actor, None, counter, None, .16, .003, .97, .92, 256, 3,
              'cpu', {'var_dim': 25, 'node_dim': 8, 'mip_dim': 53}, logger)
agent.choose_action(candidates, mip, node, deterministic=True)
report['evaluation_critic_calls'] = counter.calls

def json_number(x):
    x = float(x)
    return x if math.isfinite(x) else ('Infinity' if x > 0 else '-Infinity')

cutoffs = json.loads((ROOT / 'tgppo/datasets/cutoffs.json').read_text())
native = []
for name in ['lseu', 'stein27', 'air04', 'neos20']:
    m = Model()
    init_params(m, dict(scip_limits, time_limit=1.0), settings['sandbox'])
    m.setBoolParam('randomization/permutevars', True)
    m.setIntParam('randomization/permutationseed', 0)
    m.readProblem(str(ROOT / 'tgppo/datasets/train_instances' / (name + '.mps.gz')))
    cutoff = cutoffs[name]['cutoff']
    m.setObjlimit(cutoff)
    start = time.perf_counter()
    m.optimize()
    row = dict(instance=name, time_limit=1.0, cutoff=cutoff,
               solver_status=str(m.getStatus()), nodes=m.getNNodes(),
               n_stored_solutions=len(m.getSols()), raw_gap=json_number(m.getGap()),
               gap_is_scip_infinite=abs(float(m.getGap())) >= m.infinity(),
               primal_bound=json_number(m.getPrimalbound()),
               dual_bound=json_number(m.getDualbound()),
               pdi=json_number(m.getPrimalDualIntegral()),
               solver_seconds=m.getSolvingTime(),
               optimize_wall_seconds=time.perf_counter() - start)
    native.append(row)
    print(name, row['solver_status'], 'gap=', row['raw_gap'], 'pdi=', row['pdi'], flush=True)
    m.freeProb()
report['native_one_second_diagnostics'] = native
after = snapshot()
report['tracked_files_changed'] = sorted(
    f for f in set(before) | set(after) if before.get(f) != after.get(f))
report['upstream_status_porcelain'] = subprocess.check_output(
    ['git', '-C', str(REPO), 'status', '--porcelain']).decode()
(OUT / 'upstream_file_hashes.json').write_text(json.dumps(before, indent=2) + '\n')
(OUT / 'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: v for k, v in report.items()
                  if k != 'native_one_second_diagnostics'}, indent=2), flush=True)
