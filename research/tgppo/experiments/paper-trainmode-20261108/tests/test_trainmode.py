import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
import paper
from project import runtime as rt
from project.agent.agent import Agent
from project.utils.settings import state_dims


class DropoutActor(nn.Module):
    """Strong stochastic response makes the old train-ratio guard fail reliably."""
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(.8)
        self.linear = nn.Linear(25, 1)

    def forward(self, cands, node, mip, mask=None):
        logits = self.linear(self.dropout(cands)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))
        return logits.softmax(-1)


class SmallCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(.1))

    def forward(self, cands, node, mip, mask=None):
        return self.value.expand(len(cands))


def toy_agent():
    rt.seed_all(123)
    actor, critic = DropoutActor(), SmallCritic()
    agent = Agent(actor, torch.optim.AdamW(actor.parameters(), lr=.00024),
        critic, torch.optim.AdamW(critic.parameters(), lr=.00012),
        policy_clip=.16, entropy_weight=.003, gamma=.9, gae_lambda=.8,
        batch_size=256, n_epochs=3, device='cpu', state_dims=state_dims)
    agent.attention_cell_budget = 1048576
    return agent


def fill(agent, name='toy', seed=7, n=4):
    rng = np.random.RandomState(seed)
    agent.memory.begin_episode(name, agent.policy_version)
    for i in range(n):
        x, node, mip = [rng.randn(*shape).astype('float32')
                        for shape in [(5+i,25),(8,),(53,)]]
        action, value, log_prob = agent.choose_action(x, mip, node)
        agent.remember(cands_state=x, node_state=node, mip_state=mip,
            action=action, value=value, log_prob=log_prob, reward=float(i), done=i==n-1)
    agent.seal_episode(True)


class TrainModeTests(unittest.TestCase):
    def test_dropout_ratio_noise_is_reported_not_rejected(self):
        a = toy_agent()
        fill(a)
        modes = []
        handle = a.actor_network.register_forward_pre_hook(
            lambda net, inputs: modes.append((net.training, torch.is_grad_enabled())))
        result = a.learn(microbatch_size=2)
        handle.remove()
        self.assertTrue(result['updated'])
        self.assertLess(result['initial_ratio_max_error'], 1e-4)
        self.assertGreater(result['initial_train_ratio_max_error'], 1e-4)
        self.assertGreater(result['initial_dropout_clip_fraction'], 0.)
        self.assertEqual(result['num_updates'], 3)
        self.assertTrue(all(training for training, grad in modes if grad))
        self.assertTrue(a.actor_network.training)
        self.assertTrue(np.isfinite(result['deployment_tv_before_after_mean']))

    def test_corrupt_old_probabilities_still_fail_before_any_step(self):
        a = toy_agent()
        fill(a)
        for row in a.memory.episodes[0]['transitions']:
            row['log_prob'] += .6
        with self.assertRaisesRegex(ValueError, 'Eval rollout/recompute'):
            a.learn(microbatch_size=2)
        self.assertEqual(len(a.actor_optimizer.state), 0)
        self.assertEqual(len(a.critic_optimizer.state), 0)

    def test_gae_matches_hand_calculation_and_ignores_terminal_bootstrap(self):
        a = toy_agent()
        r, v, d = map(torch.tensor, ([1.,2.],[.5,.7],[0.,1.]))
        adv, ret = a._compute_gae(r, v, d, torch.tensor(999.))
        torch.testing.assert_close(adv, torch.tensor([2.066,1.3]))
        torch.testing.assert_close(ret, torch.tensor([2.566,2.]))
        # The public extra done[t+1] mask would instead give A0=1.436.
        self.assertGreater(abs(float(adv[0])-1.436), .5)

    def test_gae_uses_real_next_state_value_at_truncation(self):
        a = toy_agent()
        r, v, d = map(torch.tensor, ([1.,2.],[.5,.7],[0.,0.]))
        adv, ret = a._compute_gae(r, v, d, torch.tensor(.8))
        torch.testing.assert_close(adv, torch.tensor([2.5844,2.02]))
        torch.testing.assert_close(ret, torch.tensor([3.0844,2.72]))

    def test_episode_targets_do_not_leak_across_boundaries(self):
        a = toy_agent()
        fill(a, 'one', n=2)
        first = a.compute_targets()[0].clone()
        fill(a, 'two', seed=99, n=3)
        a.memory.episodes[1]['transitions'][0]['reward'] = 100000.
        torch.testing.assert_close(a.compute_targets()[0][:2], first)

    def test_eval_reference_preserves_mode_and_rng(self):
        a = toy_agent()
        fill(a)
        b = next(a.memory.get_batch_generator())
        a.actor_network.train()
        before = torch.get_rng_state().clone()
        ratios = a._eval_ratios(b['cands'], b['node_state'], b['mip_state'],
            b['mask'], b['action'], b['log_prob'])
        self.assertTrue(a.actor_network.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.testing.assert_close(ratios, torch.ones_like(ratios))

    def test_production_b0_uses_training_dropout_with_gradients(self):
        c = paper.protocol()
        rt.seed_all(11)
        a = rt.build_agent(paper.configuration(c, 'b0', 11, 'cpu'))
        fill(a)
        modes = []
        dropouts = [m for m in a.actor_network.modules() if isinstance(m, nn.Dropout)]
        self.assertTrue(dropouts and all(m.p == .05 for m in dropouts))
        h = a.actor_network.register_forward_pre_hook(lambda net, inputs:
            modes.append((torch.is_grad_enabled(), net.training, all(m.training for m in dropouts))))
        result = a.learn(microbatch_size=2)
        h.remove()
        self.assertTrue(result['updated'])
        self.assertTrue(all(train and drops for grad, train, drops in modes if grad))
        self.assertEqual(result['actor_update_mode'], 'train')
        for variant, expected in [('b0',.05),('mlp',0.),('mlp_bc',0.)]:
            cfg = paper.configuration(c, variant, 11, 'cpu')
            self.assertEqual(cfg['effective_actor_dropout'], expected)

    def test_old_checkpoint_and_unselected_protocol_are_rejected(self):
        c = paper.protocol()
        bad = copy.deepcopy(c)
        bad['alignment_choice'] = 'author_code'
        with mock.patch.object(paper, 'read', return_value=bad):
            with self.assertRaises(ValueError):
                paper.protocol()
        old = rt.build_agent
        cfg = paper.configuration(c,'mlp',11,'cpu')
        cfg['experiment'] = 'paper-aligned-20261106'
        with self.assertRaisesRegex(ValueError,'Old experiment'):
            old(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'old.pt'
            torch.save({'config':cfg},p)
            with self.assertRaisesRegex(ValueError,'different experiment'):
                rt.read_checkpoint(p)

    def test_b0_interrupted_inside_update_replays_dropout_and_adam_exactly(self):
        from engineering import same_state
        rt.seed_all(22)
        cfg = paper.configuration(paper.protocol(),'b0',22,'cpu')
        a = rt.build_agent(cfg)
        fill(a)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pending.pt'
            rt.save_checkpoint(path,a,cfg,{'phase':'update_pending','next_episode':0})
            a.learn(microbatch_size=2)
            expected = copy.deepcopy((a.actor_network.state_dict(),a.critic_network.state_dict(),
                a.actor_optimizer.state_dict(),a.critic_optimizer.state_dict(),rt.rng_state()))
            interrupted, _ = rt.load_agent(path)
            step = interrupted.actor_optimizer.step
            def interrupt():
                step()
                raise InterruptedError('Interrupted between Actor and Critic optimizer steps')
            with mock.patch.object(interrupted.actor_optimizer,'step',side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    interrupted.learn(microbatch_size=2)
            resumed, _ = rt.load_agent(path)
            resumed.learn(microbatch_size=2)
            actual = (resumed.actor_network.state_dict(),resumed.critic_network.state_dict(),
                resumed.actor_optimizer.state_dict(),resumed.critic_optimizer.state_dict(),rt.rng_state())
            self.assertTrue(same_state(expected,actual))
            self.assertEqual(resumed.update_counter,1)
            self.assertEqual(resumed.policy_version,1)

    def test_normalization_reuse_is_exact_and_readonly(self):
        c = copy.deepcopy(paper.protocol())
        job = paper.make_job(c,'vpm2',0,0,'native','native-vpm2-s0')
        ident = rt.job_identity(job)
        info = dict(stack={'software':'synthetic'}, host='new')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root/'old-normalization'
            c['normalization_readonly_sources'] = [str(source_root)]
            source = source_root/paper.fp(info['stack'])/paper.fp(ident)/'baseline.json'
            bundle = dict(signature=dict(identity=ident,stack=info['stack']),identity=ident,
                horizon_seconds=3600.,host='old',result=dict(valid=True,task_terminal=True,
                    nnodes=5,gap=.1,primalDualIntegral=100.))
            paper.write(source,bundle)
            before = (source.read_bytes(),source.stat().st_mtime_ns)
            with mock.patch.object(paper,'RUN',root/'new-run'), \
                 mock.patch.object(paper,'cached_job',side_effect=AssertionError('Unexpected SCIP task')):
                found = paper.native(c,'vpm2',0,info)
            self.assertEqual(found,bundle)
            self.assertEqual(before,(source.read_bytes(),source.stat().st_mtime_ns))
            self.assertEqual(list(source_root.rglob('*.lock')),[])
            self.assertEqual(len(list((root/'new-run').rglob('reuse_provenance.json'))),1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
