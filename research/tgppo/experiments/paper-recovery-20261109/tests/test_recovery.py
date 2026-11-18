import copy
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import recover
paper = recover.boot(os.environ['PAPER_RECOVERY_TEST_PROTOCOL'])
import numpy as np
import torch
from project import runtime as rt
from engineering import same_state
import checks
import server


def fill(agent, name='synthetic'):
    rng = np.random.RandomState(7)
    agent.memory.begin_episode(name, agent.policy_version)
    for i in range(4):
        x, n, m = [rng.randn(*shape).astype('float32') for shape in [(3+i,25),(8,),(53,)]]
        action, value, logp = agent.choose_action(x, m, n)
        agent.remember(cands_state=x, node_state=n, mip_state=m, action=action,
            value=value, log_prob=logp, reward=float(i), done=i==3)
    agent.seal_episode(True)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_run, self.target_run = self.root/'parent', self.root/'cpu'
        self.source = self.source_run/'training/mlp-s11'
        self.source.mkdir(parents=True)
        self.c = paper.protocol()

    def source_checkpoint(self, with_pending=False):
        cfg = paper.configuration(self.c, 'mlp', 11, 'cuda')
        rt.seed_all(11)
        a = rt.build_agent(cfg, device='cpu')
        fill(a, 'warm-adamw')
        a.learn(microbatch_size=2)
        progress = dict(next_episode=41, phase='ready', experiment_protocol_sha256=paper.fp(self.c))
        rt.save_checkpoint(self.source/'latest.pt', a, cfg, progress)
        paper.write(self.source/'budget.json', dict(attempts=[dict(started_epoch=1,
            allocated_seconds=172800, elapsed_seconds=100., finished_epoch=101., host='parent',pid=1)],
            charged_seconds=100.))
        if with_pending:
            fill(a, 'pending')
            directory = self.source/'source-collection'
            directory.mkdir()
            (directory/'trajectory.pkl').write_bytes(b'preserve original trajectory')
            pending = dict(next_episode=41, phase='update_pending',
                experiment_protocol_sha256=paper.fp(self.c), task_result=dict(valid=True),
                collection_directory=str(directory))
            rt.save_checkpoint(self.source/'pending.pt', a, cfg, pending)
        return a

    def test_cpu_import_keeps_weights_adamw_counters_rng_budget(self):
        self.source_checkpoint(True)
        before = {p.name:recover.sha(p) for p in self.source.glob('*.pt')}
        original = rt.read_checkpoint(self.source/'latest.pt')
        out = recover.cpu_fork(paper, self.source_run, self.target_run, self.c, 'mlp', 11)
        migrated = rt.read_checkpoint(out/'latest.pt')
        for key in ['actor','critic','actor_optimizer','critic_optimizer','policy_version',
                    'update_counter','memory','rng','progress']:
            self.assertTrue(same_state(original[key], migrated[key]), key)
        expected_cfg = dict(original['config'], learner_device='cpu')
        self.assertEqual(migrated['config'], expected_cfg)
        self.assertEqual(paper.read(out/'budget.json'), paper.read(self.source/'budget.json'))
        pending = rt.read_checkpoint(out/'pending.pt')
        self.assertTrue(str(pending['progress']['collection_directory']).startswith(str(out)))
        self.assertEqual(before, {p.name:recover.sha(p) for p in self.source.glob('*.pt')})
        self.assertEqual(recover.cpu_fork(paper, self.source_run, self.target_run, self.c,'mlp',11),out)
        self.assertFalse(torch.cuda.is_initialized())

    def test_live_parent_refused(self):
        self.source_checkpoint()
        with paper.lock(self.source/'execution.lock'):
            with self.assertRaises(BlockingIOError):
                recover.cpu_fork(paper,self.source_run,self.target_run,self.c,'mlp',11)
        self.assertFalse((self.target_run/'training/mlp-s11').exists())

    def test_unfinished_parent_accounting_refused(self):
        self.source_checkpoint()
        ledger = paper.read(self.source/'budget.json')
        del ledger['attempts'][0]['elapsed_seconds']
        paper.write(self.source/'budget.json', ledger)
        with self.assertRaisesRegex(ValueError, 'unfinished budget'):
            recover.cpu_fork(paper,self.source_run,self.target_run,self.c,'mlp',11)

    def test_cpu_pending_interruption_replays_exactly_once(self):
        self.source_checkpoint(True)
        source_hashes = {str(p):recover.sha(p) for p in self.source.rglob('*') if p.is_file()}
        out = recover.cpu_fork(paper,self.source_run,self.target_run,self.c,'mlp',11)
        pending = rt.read_checkpoint(out/'pending.pt')
        a, _ = rt.load_agent(out/'pending.pt',device='cpu')
        rt.seed_all(rt.action_seed(11,'paper-ppo-update-e00042'))
        a.learn(microbatch_size=self.c['microbatch_size'])
        expected = copy.deepcopy((a.actor_network.state_dict(),a.critic_network.state_dict(),
            a.actor_optimizer.state_dict(),a.critic_optimizer.state_dict(),rt.rng_state()))
        real_learn = rt.Agent.learn
        def interrupted(a, **kwargs):
            real_learn(a, **kwargs)
            raise paper.Stopped('synthetic interruption after steps before commit')
        def no_next_solver(*args, **kwargs):
            raise paper.Stopped('synthetic stop before next solve')
        with mock.patch.object(paper,'RUN',paper.RUN), mock.patch.object(paper,'configuration',paper.configuration), \
             mock.patch.object(paper,'native',no_next_solver), \
             mock.patch.object(server,'preflight',return_value=dict(stack={'synthetic':True},gpu=None,host='cpu')), \
             mock.patch.object(checks,'target_checks'):
            recover.install_cpu(paper,self.source_run,self.target_run)
            def train():
                if os.environ['PAPER_RECOVERY_TEST_PROTOCOL']=='aligned':
                    paper.train(self.c,'mlp',11,True)
                else:
                    paper.train(self.c,'mlp',11,True,'cpu')
            with mock.patch.object(rt.Agent,'learn',interrupted):
                with self.assertRaises(paper.Stopped):train()
            self.assertEqual(rt.read_checkpoint(out/'latest.pt')['progress']['next_episode'],41)
            with self.assertRaises(paper.Stopped):train()
        committed = rt.read_checkpoint(out/'latest.pt')
        self.assertEqual(committed['progress']['next_episode'],42)
        self.assertFalse((out/'pending.pt').exists())
        actual = (committed['actor'],committed['critic'],committed['actor_optimizer'],
                  committed['critic_optimizer'],committed['rng'])
        self.assertTrue(same_state(expected,actual))
        self.assertEqual(source_hashes,{str(p):recover.sha(p) for p in self.source.rglob('*') if p.is_file() and p.name!='execution.lock'})
        self.assertEqual(pending['config']['learner_device'],'cpu')
        self.assertFalse(torch.cuda.is_initialized())

    def test_explicit_unlimited_budget_preserves_prior_cost_and_lock(self):
        folder = self.root/'budget'
        first = dict(started_epoch=1,allocated_seconds=172800,elapsed_seconds=200000.,finished_epoch=200001.,pid=1)
        paper.write(folder/'budget.json',dict(attempts=[first],charged_seconds=200000.))
        with recover.unlimited_budget(paper)(folder,172800,True) as end:
            self.assertTrue(math.isinf(end))
            with self.assertRaises(BlockingIOError):
                with recover.unlimited_budget(paper)(folder,172800,True):pass
        ledger = paper.read(folder/'budget.json')
        self.assertEqual(ledger['attempts'][0],first)
        self.assertGreaterEqual(ledger['charged_seconds'],200000.)
        self.assertIsNone(ledger['attempts'][-1]['wall_limit_seconds'])
        self.assertEqual(len(ledger['attempts']),2)

    def test_cpu_cache_routes_and_rejects_policy_or_software_changes(self):
        folder = self.root/'episode'
        old = dict(checkpoint_sha256='policy',sampling_seed=4,task={'solver':3},
            stack=dict(cpu_model='old',torch='same'))
        paper.write(folder/'identity.json',old)
        current = copy.deepcopy(old)
        current['stack']['cpu_model']='new'
        worker = mock.Mock(return_value=({'valid':True},'synthetic'))
        cached = recover.cohort_cached_job(paper,worker)
        cached(dict(mode='collect'),folder,current,self.c,True)
        self.assertIn('recovery-cohorts',str(worker.call_args[0][1]))
        self.assertEqual(paper.read(folder/'identity.json'),old)
        for changed in (dict(current,checkpoint_sha256='wrong'),dict(current,sampling_seed=5),
                        dict(current,stack=dict(cpu_model='new',torch='wrong'))):
            with self.assertRaises(ValueError):cached(dict(mode='collect'),folder,changed,self.c,True)

    def test_duplicate_cannot_overwrite_owner_state_even_if_container_pid_matches(self):
        folder = self.root/'duplicate'
        state = dict(status='updating',next_episode=4)
        paper.write(folder/'state.json',state)
        guarded = recover.guarded_budget(paper,paper.budget,dict(device='cpu'))
        with paper.lock(folder/'execution.lock'):
            with self.assertRaises(BlockingIOError):
                with guarded(folder,172800,True):pass
        self.assertEqual(paper.read(folder/'state.json'),state)
        self.assertFalse(list(folder.glob('recovery-attempt-*')))

    def test_original_full_model_cpu_device_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            result = checks.target_checks(self.c,11,Path(directory),'cpu')
            self.assertTrue(result['passed'])
            self.assertEqual(result['scip_solves'],0)
            self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
