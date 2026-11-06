"""Bounded one-thread CPU checks using existing data; no SCIP optimization."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from unittest.mock import patch
import numpy as np
import torch
import server
import launch
from engineering import bridge, same_state
from expert_data import OUTPUT, batch, load_arrays
from project import runtime as rt
from project.agent.expert_supervision import ExpertSupervision
from project.utils.protocol import write_json

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
OUT = ROOT / 'tgppo/runs/supervised-ppo-20261105/preparation'


def fill(agent):
    agent.memory.begin_episode('synthetic-local-only', 0)
    rng = np.random.RandomState(72)
    for i, length in enumerate((3, 7, 5, 9)):
        x = rng.normal(size=(length, 25)).astype('float32')
        node = rng.normal(size=8).astype('float32')
        mip = rng.normal(size=53).astype('float32')
        action, value, logp = agent.choose_action(x, mip, node)
        agent.remember(cands_state=x, node_state=node, mip_state=mip, action=action,
            value=value, log_prob=logp, reward=(1., -.5, .25, -.75)[i], done=i == 3)
    agent.seal_episode(task_terminal=True)


def state(agent):
    return copy.deepcopy((agent.actor_network.state_dict(), agent.critic_network.state_dict(),
        agent.actor_optimizer.state_dict(), agent.critic_optimizer.state_dict()))


def main():
    budget = json.loads((OUT / 'budget.json').read_text())
    started = budget['started_epoch']
    remaining = 600 - (time.time() - started)
    def expired(*args):
        raise TimeoutError('600-second local preparation cap')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, remaining - 10))
    checks = {}
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        torch.set_num_threads(1)
        cfg = server.params(ROOT, launch.protocol('expert')[0], 'B0', 11, 'cpu')
        cfg.update(batch_size=4, n_epochs=1)
        supervision = ExpertSupervision(cfg)
        ids = np.concatenate([supervision.sample_ids(i, 0) for i in range(100)])
        checks['only_legal_train_experts'] = (set(supervision.data['instances'][ids]) == set(supervision.names)
            and bool(np.all(supervision.data['split'][ids] == 'train')))
        before = rt.rng_state()
        checks['sampler_stateless_and_cursor_sensitive'] = (np.array_equal(supervision.sample_ids(3, 7), supervision.sample_ids(3, 7))
            and not np.array_equal(supervision.sample_ids(3, 7), supervision.sample_ids(3, 8))
            and same_state(before, rt.rng_state()))
        # Current BC-selected weights; no BC fitting or selection is repeated.
        raw = ROOT / 'tgppo/runs/light-actor-20261105/mlp_bc/training/B0-s11/pretrain/actor.pt'
        actor = rt.make_actor(dict(cfg['state_dims'], **{k: cfg[k] for k in rt.ARCH_KEYS}), 'mlp').eval()
        actor.load_state_dict(torch.load(raw, map_location='cpu')['actor'])
        before = rt.rng_state()
        loss, info = supervision.loss(actor, 'cpu', 0, 0)
        x, node, mip, mask, label = batch(supervision.data, supervision.sample_ids(0, 0))
        expected = .01 * (torch.nn.functional.cross_entropy(actor.logits(x, node, mip, mask), label,
            reduction='none') / (~mask).sum(1).float().log()).mean()
        checks['masked_normalized_ce_matches_formula'] = torch.equal(loss, expected)
        actor.zero_grad(set_to_none=True)
        loss.backward()
        checks['supervision_reaches_variable_tree_and_head'] = all(any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for name, p in actor.named_parameters() if name.startswith(prefix))
            for prefix in ('var_embedding.', 'tree_embedding.', 'head.'))
        checks['expert_forward_backward_preserves_global_rng'] = same_state(before, rt.rng_state())
        supervised_norm = sum(p.grad.double().square().sum().item() for p in actor.parameters()) ** .5
        actor.zero_grad(set_to_none=True)
        probs = actor(x, node, mip, mask)
        logp = torch.distributions.Categorical(probs).log_prob(label)
        advantages = torch.linspace(-1., 1., len(label))
        synthetic_ppo = -(torch.exp(logp - logp.detach()) * advantages).mean() - .003 * torch.distributions.Categorical(probs).entropy().mean()
        synthetic_ppo.backward()
        ppo_norm = sum(p.grad.double().square().sum().item() for p in actor.parameters()) ** .5
        gradient_probe = dict(fixed_lambda=.01, weighted_expert_gradient_norm=supervised_norm,
            synthetic_ppo_gradient_norm=ppo_norm, ratio=supervised_norm / ppo_norm,
            label='engineering magnitude only; artificial advantages; not on-policy or solver evidence')
        # Construct ONE common synthetic buffer, then compare identical replay paths.
        control_cfg = dict(cfg, expert_supervision_weight=0.)
        rt.seed_all(101)
        control = rt.build_agent(control_cfg, device='cpu')
        control.actor_network.load_state_dict(torch.load(raw, map_location='cpu')['actor'])
        fill(control)
        directory = Path(tempfile.mkdtemp(prefix='tgppo-supervised-check-'))
        before_path = directory / 'before.pt'
        rt.save_checkpoint(before_path, control, control_cfg)
        control.learn(microbatch_size=2)
        expected_state, expected_rng = state(control), rt.rng_state()
        old_file = ROOT / 'tgppo/experiments/light-actor-20261105/code/project/agent/agent.py'
        spec = importlib.util.spec_from_file_location('frozen_original_agent', old_file)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        with patch.object(rt, 'Agent', old.Agent):
            original, _ = rt.load_agent(before_path, device='cpu')
        original.learn(microbatch_size=2)
        checks['lambda_zero_exact_original_models_optimizers_rng'] = (same_state(expected_state, state(original))
            and same_state(expected_rng, rt.rng_state()))
        expert, _ = rt.load_agent(before_path, device='cpu')
        expert.expert_supervision = ExpertSupervision(cfg)
        metrics = expert.learn(microbatch_size=2)
        after = state(expert)
        checks['aux_once_per_logical_minibatch'] = metrics['expert_batches'] == metrics['num_updates'] == 1 and metrics['expert_samples'] == 16
        checks['aux_changes_actor_preserves_critic_and_rng'] = (not same_state(expected_state[0], after[0])
            and same_state(expected_state[1], after[1]) and same_state(expected_state[3], after[3])
            and same_state(expected_rng, rt.rng_state()))
        result = bridge(cfg, 'cpu', directory / 'expert-recovery', actor_state=torch.load(raw, map_location='cpu')['actor'])
        checks['supervised_checkpoint_exact_recovery'] = result['passed'] and result['metrics']['expert_batches'] == 1
        export = directory / 'expert-recovery/actor.pt'
        with patch.object(rt, 'Critic', side_effect=AssertionError('deployment constructed Critic')), \
             patch('project.agent.expert_supervision.ExpertSupervision', side_effect=AssertionError('deployment read expert data')):
            policy, _ = rt.load_actor(export)
        checks['deployment_actor_only_no_expert_dependency'] = all(p.device.type == 'cpu' for p in policy.actor_network.parameters())
        try:
            with launch.deadline(.02):
                time.sleep(.04)
        except launch.BudgetExhausted:
            checks['budget_timeout'] = True
        checks['dead_attempt_charges_allocation'] = launch.charged_seconds([dict(allocated_seconds=50), dict(allocated_seconds=20, elapsed_seconds=3)]) == 53
        checks['unchanged_actor_critic_reward_settings'] = all(launch.sha256(PACKAGE / 'code/project' / name) ==
            launch.sha256(ROOT / 'tgppo/experiments/light-actor-20261105/code/project' / name)
            for name in ('policy/light_actor.py', 'policy/actor.py', 'policy/critic.py', 'reward/reward.py', 'utils/settings.py'))
        checks['cuda_uninitialized'] = not torch.cuda.is_initialized()
        value = dict(passed=all(bool(x) for x in checks.values()), checks={k: bool(v) for k, v in checks.items()},
            gradient_probe=gradient_probe, selected_lambda=.01, expert_training_instances=12, expert_training_states=1468,
            heldout_states_not_used_in_gradients=288, cpu_threads=1, cuda_initialized=torch.cuda.is_initialized(),
            elapsed_seconds=time.time() - started, new_scip_solves=0, real_ppo_episodes=0,
            scope='synthetic CPU interface/gradient/recovery checks; no solver performance evidence', fixture_directory=str(directory))
        write_json(OUT / 'checks.json', value)
        if not value['passed']:
            raise RuntimeError('Local checks failed: ' + str(checks))
        print(json.dumps(value, ensure_ascii=False, indent=2))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        budget.update(charged_seconds=time.time() - started, finished=True)
        write_json(OUT / 'budget.json', budget)


if __name__ == '__main__':
    main()
