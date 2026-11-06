"""Hand calculations, protocol failures, real binding and resumable PPO checks."""
import copy
import json
import logging
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from pyscipopt import Model
from project.agent.agent import ActorPolicy
from project.brancher import Brancher
from project.environment import Environment
from project.runtime import (config_with_protocol, seed_all, build_agent, load_agent, load_actor,
    save_checkpoint, run_jobs, make_job, job_identity, import_trajectories)
from project.utils.protocol import validate_baseline, solver_stats, sha256
from project.utils.settings import state_dims, settings

ROOT = Path('/root/research')
CONFIG = config_with_protocol(dict(hidden_dim=32, num_layers=1, num_heads=4,
                                 batch_size=2, n_epochs=1, dropout=.2))


def agent():
    seed_all(17)
    return build_agent(CONFIG)


def state(n=3):
    rng = np.random.RandomState(n)
    return dict(cands_state=rng.randn(n, 25).astype('float32'),
                mip_state=rng.randn(53).astype('float32'),
                node_state=rng.randn(8).astype('float32'))


def populate(a, episode='first', rewards=(1., 2.)):
    a.memory.begin_episode(episode, a.policy_version)
    for i, reward in enumerate(rewards):
        s = state(i + 2)
        action, value, log_prob = a.choose_action(**s)
        a.remember(**s, action=action, reward=reward, done=i == len(rewards) - 1,
                   value=value, log_prob=log_prob)
    a.seal_episode(True)


class CorrectnessTests(unittest.TestCase):
    def test_gae_terminal_hand_calculation(self):
        a = agent()
        a.gamma = a.gae_lambda = 1.
        _, returns = a._compute_gae(torch.tensor([1., 2.]), torch.tensor([10., 20.]),
                                    torch.tensor([0., 1.]), torch.tensor(0.))
        torch.testing.assert_close(returns, torch.tensor([3., 2.]))

    def test_gae_truncation_hand_calculation(self):
        a = agent()
        a.gamma, a.gae_lambda = .9, 1.
        _, returns = a._compute_gae(torch.tensor([1., 2.]), torch.tensor([10., 20.]),
                                    torch.tensor([0., 0.]), torch.tensor(30.))
        torch.testing.assert_close(returns, torch.tensor([27.1, 29.]))

    def test_two_episodes_do_not_contaminate_each_other(self):
        a = agent()
        a.gamma = a.gae_lambda = 1.
        populate(a)
        populate(a, 'second', (3., 4.))
        torch.testing.assert_close(a.compute_targets()[1], torch.tensor([3., 2., 7., 4.]))
        a.memory.episodes[1]['transitions'][0]['reward'] = 900.
        torch.testing.assert_close(a.compute_targets()[1][:2], torch.tensor([3., 2.]))

    def test_truncation_needs_actual_tail_and_uses_critic(self):
        a = agent()
        a.memory.begin_episode('tail', 0)
        with self.assertRaisesRegex(ValueError, 'Missing actual'):
            a.seal_episode(False, collector_truncated=True)
        s = state()
        with torch.no_grad():
            a.critic_network.eval()
            c, m, n, mask = a._inputs(**s)
            expected = a.critic_network(c, n, m, mask).item()
        a.seal_episode(False, collector_truncated=True, bootstrap_state=s)
        self.assertEqual(a.memory.episodes[0]['bootstrap_value'], expected)

    def test_buffer_roundtrip_empty_duplicate_and_stale_policy(self):
        a = agent()
        populate(a)
        payload = a.memory.export_dict()
        b = agent()
        b.memory.import_dict(payload)
        torch.testing.assert_close(a.compute_targets()[1], b.compute_targets()[1])
        with self.assertRaises(ValueError):
            b.memory.import_dict(payload)
        b.policy_version += 1
        with self.assertRaisesRegex(ValueError, 'Stale'):
            b.learn()
        a.memory.clear()
        a.memory.begin_episode('zero-actions', 0)
        a.seal_episode(True)
        self.assertEqual(a.compute_targets()[1].numel(), 0)
        self.assertFalse(a.learn()['updated'])

    def test_independent_parameters_and_gradients(self):
        a = agent()
        self.assertFalse({id(p) for p in a.actor_network.parameters()} &
                         {id(p) for p in a.critic_network.parameters()})
        c, m, n, mask = a._inputs(**state())
        a.critic_network(c, n, m, mask).sum().backward()
        self.assertTrue(all(p.grad is None for p in a.actor_network.parameters()))
        self.assertTrue(any(p.grad is not None for p in a.critic_network.parameters()))

    def test_padding_ratio_and_real_actor_update(self):
        a = agent()
        populate(a)
        before = [p.detach().clone() for p in a.actor_network.parameters()]
        metrics = a.learn()
        self.assertTrue(metrics['updated'])
        self.assertLess(metrics['initial_ratio_max_error'], 1e-4)
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(before, a.actor_network.parameters())))
        self.assertEqual(len(a.memory), 0)
        self.assertEqual(a.policy_version, 1)

    def test_actor_only_actions_match_collection_and_old_eval_mode(self):
        a = agent()
        policy = ActorPolicy(a.actor_network)
        for seed in range(10):
            seed_all(seed)
            collected = a.choose_action(**state())
            seed_all(seed)
            deployed = policy.act(**state())
            self.assertEqual((collected[0], collected[2]), deployed)
        # Same unmodified Actor forward + Categorical as the original selection path.
        seed_all(39)
        c, m, n, mask = a._inputs(**state())
        a.actor_network.eval()
        dist = torch.distributions.Categorical(a.actor_network(c, n, m, mask))
        expected = dist.sample().item()
        seed_all(39)
        self.assertEqual(policy.act(**state())[0], expected)
        with self.assertRaises(ValueError):
            policy.act(**state(), padding_mask=np.ones(3, dtype=bool))

    def test_actual_upstream_choose_action_matches_actor_only(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('upstream_agent',
            ROOT / 'tgppo/repo/code/project/agent/agent.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a = agent()
        original = module.Agent(a.actor_network, None, a.critic_network, None,
            device='cpu', state_dims=state_dims, **{k: CONFIG[k] for k in
            ('policy_clip', 'entropy_weight', 'gamma', 'gae_lambda', 'batch_size', 'n_epochs')})
        deployed = ActorPolicy(a.actor_network)
        for seed in range(10):
            seed_all(seed)
            expected = original.choose_action(**state())
            seed_all(seed)
            actual = deployed.act(**state())
            self.assertEqual(int(expected[0]), actual[0])
            self.assertAlmostEqual(float(expected[2]), actual[1], places=7)

    def test_invalid_transition_and_incomplete_episode_rejected(self):
        a = agent()
        a.memory.begin_episode('bad', 0)
        with self.assertRaises(ValueError):
            a.remember(**state(), action=900, value=0., log_prob=0., reward=1., done=False)
        with self.assertRaises(ValueError):
            a.remember(**state(), action=0, value=0., log_prob=0., reward=float('nan'), done=False)
        with self.assertRaises(RuntimeError):
            a.learn()
        with self.assertRaises(RuntimeError):
            a.memory.export_dict()
        a.memory.abort_episode()
        self.assertEqual(len(a.memory), 0)

    def test_checkpoint_architecture_rng_optimizer_buffer_and_exact_next_update(self):
        a = agent()
        populate(a)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(path, a, CONFIG, dict(next_batch=2))
            expected_rng = (np.random.rand(), torch.rand(1), __import__('random').random())
            b, payload = load_agent(path)
            actual_rng = (np.random.rand(), torch.rand(1), __import__('random').random())
            self.assertEqual(expected_rng[0], actual_rng[0])
            torch.testing.assert_close(expected_rng[1], actual_rng[1])
            self.assertEqual(expected_rng[2], actual_rng[2])
            self.assertEqual(payload['config']['hidden_dim'], 32)
            self.assertEqual(payload['progress']['next_batch'], 2)
            a, _ = load_agent(path)
            progress = []
            a.learn(progress_callback=progress.append)
            self.assertTrue(progress)
            b, _ = load_agent(path)
            b.learn()
            for k, v in a.actor_network.state_dict().items():
                torch.testing.assert_close(v, b.actor_network.state_dict()[k], rtol=0, atol=0)
            for k, v in a.critic_network.state_dict().items():
                torch.testing.assert_close(v, b.critic_network.state_dict()[k], rtol=0, atol=0)
            # Optimizer state must also be nonempty and recoverable after an update.
            save_checkpoint(path, a, CONFIG)
            recovered, _ = load_agent(path)
            self.assertTrue(recovered.actor_optimizer.state_dict()['state'])
            with patch('project.runtime.Critic', side_effect=AssertionError('Critic constructed')):
                policy, _ = load_actor(path)
                policy.act(**state())

    def test_finalize_then_free_is_idempotent_and_zero_actions_not_invented(self):
        a = agent()
        a.memory.begin_episode('one', 0)
        reward = Mock()
        reward.compute.return_value = 5.
        brancher = Brancher(Model(), state_dims, a, 'collect', reward)
        s = state()
        action, value, lp = a.choose_action(**s)
        brancher.prev = (s, action, value, lp)
        brancher.finalize_episode('timelimit')
        brancher.finalize_episode('timelimit')
        brancher.branchfree()
        brancher.branchfree()
        self.assertEqual(len(a.memory), 1)
        self.assertEqual(reward.compute.call_count, 1)
        self.assertTrue(a.memory.episodes[0]['transitions'][0]['done'])
        a.memory.begin_episode('empty', 0)
        brancher = Brancher(Model(), state_dims, a, 'collect', reward)
        brancher.finalize_episode('optimal')
        self.assertEqual(len(a.memory.episodes[1]['transitions']), 0)

    def test_original_branchfree_already_idempotent(self):
        # Load only the original Brancher source under another module name.
        import importlib.util
        spec = importlib.util.spec_from_file_location('upstream_brancher',
            ROOT / 'tgppo/repo/code/project/brancher/brancher.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reward, collector = Mock(), Mock()
        reward.compute.return_value = 1.
        b = module.Brancher(Model(), state_dims, 'cpu', collector, reward, 1., logging.getLogger('quiet'))
        b.prev_state = state()
        b.prev_action, b.prev_value, b.prev_log_prob = 0, 0., 0.
        b.branchfree()
        b.branchfree()
        self.assertEqual(collector.remember.call_count, 1)

    def test_reward_and_solver_receive_120_and_3600_and_baseline_mismatch_rejected(self):
        import project.environment.environment as env_module
        cutoffs = json.loads((ROOT / 'tgppo/datasets/cutoffs.json').read_text())
        for limit in (120, 3600):
            j = make_job('clock', ROOT / 'tgppo/datasets/train_instances', 'lseu', limit,
                         0, 11, cutoffs, 'collect')
            identity = job_identity(j)
            base = dict(identity=identity, result=dict(valid=True, task_terminal=True,
                nnodes=8, gap=0., primalDualIntegral=2., status='infeasible'))
            reward = Mock()
            model = Mock()
            with patch.object(env_module.scip, 'Model', return_value=model):
                env = Environment('cpu', agent(), state_dims, j['limits'], j['params'], 0, reward)
                env.reset(j['instance'], j['cutoff'], base, j['cutoff_provenance'])
                model.setRealParam.assert_any_call('limits/time', limit)
                self.assertEqual(reward.reset.call_args.kwargs['time_limit'], limit)
            for field, value in [('solver_seed', 9), ('instance_sha256', 'bad'),
                                 ('limits', dict(time_limit=999, node_limit=-1)),
                                 ('params', settings['default'])]:
                wrong = copy.deepcopy(identity)
                wrong[field] = value
                with self.assertRaisesRegex(ValueError, 'mismatched'):
                    validate_baseline(base, wrong)
            with self.assertRaises(ValueError):
                validate_baseline(None, identity)

    def test_scip_sentinel_timeout_and_cutoff_semantics(self):
        model = Mock()
        model.getStatus.return_value = 'timelimit'
        model.getSols.return_value = []
        model.infinity.return_value = 1e20
        for name in ('getGap', 'getPrimalbound'):
            getattr(model, name).return_value = 1e20
        model.getDualbound.return_value = -5.
        model.getPrimalDualIntegral.return_value = 4.
        model.getNNodes.return_value = 10
        model.getSolvingTime.return_value = 120.
        info = solver_stats(model, True, True)
        self.assertEqual(info['gap'], 1e20)
        self.assertFalse(info['gap_finite'])
        self.assertTrue(info['task_terminal'])
        self.assertFalse(info['proof_completed'])
        print('SIMULATED_SENTINEL_EVIDENCE=' + json.dumps(info, sort_keys=True))
        model.getStatus.return_value = 'infeasible'
        model.getGap.return_value = 0.
        info = solver_stats(model, True, True)
        self.assertTrue(info['cutoff_proof'])
        self.assertFalse(info['original_optimal'])
        self.assertFalse(info['has_feasible_solution'])
        self.assertFalse(solver_stats(model, True, False)['valid'])
        model.getGap.return_value = float('nan')
        malformed = solver_stats(model, True, True)
        self.assertFalse(malformed['valid'])
        self.assertIsNone(malformed['gap'])
        self.assertEqual(malformed['gap_ieee_nonfinite'], 'nan')

    def test_worker_failure_watchdog_and_order_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = [dict(job_id='z-failure', mode='eval', _test_failure=True),
                    dict(job_id='a-timeout', mode='eval', _test_sleep=10)]
            # Startup is inside the watchdog, allow imports but not the injected sleep.
            rows = run_jobs(jobs, directory, max_workers=2, watchdog_seconds=5.)
            self.assertEqual([r['job_id'] for r in rows], ['a-timeout', 'z-failure'])
            self.assertEqual(rows[0]['status'], 'watchdog_timeout')
            self.assertEqual(rows[1]['status'], 'worker_error')
            self.assertTrue(all(not r['valid'] for r in rows))
            self.assertFalse(mp.active_children())
            with self.assertRaises(RuntimeError):
                import_trajectories(agent(), rows, directory)

    def test_unchanged_networks_reward_settings_and_original_repository(self):
        hashes = json.loads((ROOT / 'docs/stage1/upstream_file_hashes.json').read_text())
        for relative, expected in hashes.items():
            self.assertEqual(sha256(ROOT / 'tgppo/repo' / relative), expected, relative)
        for relative in ('policy/actor.py', 'policy/critic.py', 'policy/modules.py',
                         'reward/reward.py', 'utils/settings.py'):
            self.assertEqual(sha256(ROOT / 'tgppo/repo/code/project' / relative),
                             sha256(ROOT / 'tgppo/experiments/stage2/code/project' / relative))


class RealSolverTests(unittest.TestCase):
    def job(self, name, mode):
        cutoffs = json.loads((ROOT / 'tgppo/datasets/cutoffs.json').read_text())
        return make_job('real-' + name, ROOT / 'tgppo/datasets/train_instances', name, 1,
                        0, 11, cutoffs, mode)

    def test_real_air04_timeout_retains_nonzero_raw_gap(self):
        j = self.job('air04', 'native')
        env = Environment('cpu', None, state_dims, j['limits'], j['params'], 0, mode='native')
        env.reset(j['instance'], j['cutoff'], cutoff_provenance=j['cutoff_provenance'])
        _, info, _ = env.run_episode()
        self.assertEqual(info['status'], 'timelimit')
        self.assertGreater(info['gap'], 0.)
        self.assertTrue(info['valid'])
        self.assertFalse(info['proof_completed'])
        print('REAL_AIR04_TIMEOUT_EVIDENCE=' + json.dumps(info, sort_keys=True))

    def test_real_eval_never_calls_critic_reward_or_memory_and_preserves_weights(self):
        a = agent()
        populate(a)
        old = copy.deepcopy(a.actor_network.state_dict())
        old_size = len(a.memory)
        j = self.job('lseu', 'eval')
        with patch.object(a.critic_network, 'forward', side_effect=AssertionError('Critic called')), \
             patch.object(a, 'remember', side_effect=AssertionError('remember called')), \
             patch.object(a, 'learn', side_effect=AssertionError('learn called')):
            reward = Mock()
            reward.compute.side_effect = AssertionError('reward called')
            env = Environment('cpu', a, state_dims, j['limits'], j['params'], 0, reward, mode='eval')
            env.reset(j['instance'], j['cutoff'], cutoff_provenance=j['cutoff_provenance'])
            valid, info, _ = env.run_episode()
        self.assertTrue(valid, info)
        self.assertGreater(info['branch_count'], 0)
        self.assertEqual(info['callback_seconds']['critic'], 0.)
        self.assertEqual(info['reward_count'], 0)
        self.assertEqual(len(a.memory), old_size)
        for key, value in old.items():
            torch.testing.assert_close(value, a.actor_network.state_dict()[key], rtol=0, atol=0)

    def test_real_policy_exception_interrupts_and_is_invalid(self):
        a = agent()
        j = self.job('lseu', 'eval')
        with patch.object(a, 'act', side_effect=ValueError('injected illegal policy')):
            env = Environment('cpu', a, state_dims, j['limits'], j['params'], 0, mode='eval')
            env.reset(j['instance'], j['cutoff'], cutoff_provenance=j['cutoff_provenance'])
            valid, info, _ = env.run_episode()
        self.assertFalse(valid)
        self.assertIn('injected illegal policy', info['error'])
        self.assertEqual(info['fallback_action_count'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
