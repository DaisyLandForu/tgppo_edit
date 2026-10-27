"""Read-only raw-upstream bridge. Run with PYTHONPATH=tgppo/repo/code."""
import json
import argparse
import logging
from pathlib import Path
import random
import numpy as np
import torch
from project.agent import Agent
from project.environment import Environment
from project.policy import Actor, Critic
from project.reward import RewardH3
from project.utils.settings import state_dims, settings

ROOT = Path('/root/research')
RUN = ROOT / 'tgppo/runs/stage2-verification'


class ObserveEnvironment(Environment):
    def _is_solved(self):
        # Observe the official environment's pre-free state, don't alter its result.
        self.observed_gap = float(self.model.getGap())
        return super()._is_solved()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/stage2/upstream-bridge.json')
    args = parser.parse_args()
    run = args.run_dir
    torch.set_num_threads(1)
    payload = torch.load(run / 'train-short/before-batch-000.pt', map_location='cpu')
    config = payload['config']
    records = json.loads((ROOT / 'tgppo/datasets/cutoffs.json').read_text())
    baselines = list(json.loads((run / 'native-short/baselines.json').read_text()).values())
    jobs = json.loads((run / 'train-short/batch-000/results.json').read_text())
    rows = []
    for j in jobs:
        name = j['instance_name']
        arch = dict(state_dims, **{k: config[k] for k in ('hidden_dim', 'num_heads', 'num_layers', 'dropout')})
        actor, critic = Actor(**arch), Critic(**arch)
        actor.load_state_dict(payload['actor'])
        critic.load_state_dict(payload['critic'])
        a = Agent(actor, None, critic, None, device='cpu', state_dims=state_dims,
            **{k: config[k] for k in ('policy_clip', 'entropy_weight', 'gamma', 'gae_lambda', 'batch_size', 'n_epochs')})
        random.seed(j['sampling_seed'])
        np.random.seed(j['sampling_seed'])
        torch.manual_seed(j['sampling_seed'])
        baseline = next(b['result'] for b in baselines if b['result']['instance_name'] == name)
        reward = RewardH3()
        env = ObserveEnvironment('cpu', a, state_dims, dict(time_limit=2., node_limit=-1),
                                 settings['sandbox'], 0, reward, logging.getLogger('bridge'))
        env.reset(str(ROOT / 'tgppo/datasets/train_instances' / (name + '.mps.gz')),
                  records[name]['cutoff'], baseline['nnodes'], baseline['gap'],
                  baseline['primalDualIntegral'], baseline['status'])
        _, info, _ = env.run_episode()
        info.update(instance_name=name, observed_raw_gap=env.observed_gap,
                    reward_time_limit=reward.time_limit, num_transitions=len(a.memory),
                    purpose='raw_upstream_bridge_no_optimizer_update')
        rows.append(info)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
