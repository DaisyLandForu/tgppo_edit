import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
import paper
from paper import protocol, schedule, configuration, eval_instances, write, fp
from project import runtime as rt
from engineering import same_state
from checks import short_episode_check, target_checks


class ProtocolTests(unittest.TestCase):
    def test_coverage_and_paired_schedule(self):
        c = protocol()
        expected = {(n, s) for n in c['train'] for s in range(5)}
        for seed in (11, 22):
            plan = schedule(seed, c)
            self.assertEqual(len(plan), 500)
            self.assertTrue(all(set(plan[i:i+125]) == expected for i in range(0, 500, 125)))
            self.assertNotEqual(plan[:125], plan[125:250])
            self.assertEqual(plan, schedule(seed, c))
        self.assertNotEqual(schedule(11, c), schedule(22, c))

    def test_test_shards_and_author_coverage(self):
        c = protocol()
        groups = [eval_instances(c, s) for s in range(17)]
        self.assertEqual(groups[-1], ['mspp16'])
        self.assertEqual(sorted(sum(groups, [])), sorted(c['test']))
        self.assertEqual(sum(35*len(g) for g in groups), 2310)
        self.assertLessEqual(max(map(len, groups)), 5)
        a = paper.read(paper.PACKAGE/'docs/author_subset_audit.json')
        self.assertEqual(a['finished_instances.xlsx']['count'], 33)
        self.assertEqual(a['timelimit_instances.xlsx']['count'], 32)
        self.assertEqual(a['missing_test_instances'], ['opm2-z7-s2'])

    def test_budget_is_not_reset_by_explicit_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with paper.budget(folder, 120, False):
                pass
            with self.assertRaises(ValueError):
                with paper.budget(folder, 120, False):
                    pass
            with paper.budget(folder, 120, True):
                pass
            record = paper.read(folder/'budget.json')
            self.assertEqual(len(record['attempts']), 2)
            self.assertGreaterEqual(record['charged_seconds'], 0.)
            record['attempts'][0]['elapsed_seconds'] = 119.
            write(folder/'budget.json', record)
            with self.assertRaises(paper.Stopped):
                with paper.budget(folder, 120, True):
                    self.fail('Exhausted budget was reset')

    def test_old_120_second_normalization_rejected(self):
        from project.utils.protocol import validate_baseline
        c = protocol()
        j = paper.make_job(c, 'vpm2', 0, 11, 'native', 'synthetic-identity')
        ident = rt.job_identity(j)
        old = copy.deepcopy(ident)
        old['limits']['time_limit'] = 120.
        record = dict(identity=old, result=dict(valid=True, task_terminal=True, nnodes=1, gap=0., primalDualIntegral=1.))
        with self.assertRaises(ValueError):
            validate_baseline(record, ident)

    def test_legal_experts_and_dataset(self):
        from expert_data import load_arrays, OUTPUT
        d = load_arrays()
        c = protocol()
        self.assertEqual(paper.sha(OUTPUT/'expert.npz'), c['expert_dataset_sha256'])
        self.assertTrue(set(d['instances']) <= set(c['train']))
        self.assertFalse(set(d['instances']) & set(c['test']))
        self.assertEqual(int(sum(d['split']=='train')), 1468)
        self.assertEqual(int(sum(d['split']=='validation')), 288)
        self.assertTrue(np.all((d['actions']>=0)&(d['actions']<np.diff(d['offsets']))))

    def test_short_full_episode_and_exact_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            short_episode_check(protocol(), Path(tmp))

    def test_full_cpu_update_export_bridges(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = target_checks(protocol(), 11, Path(tmp), 'cpu')
            self.assertTrue(result['passed'])
            self.assertEqual(result['scip_solves'], 0)

    def test_optimizer_and_actor_parameters(self):
        c = protocol()
        values = []
        for variant in ('b0', 'mlp', 'mlp_bc'):
            rt.seed_all(11)
            a = rt.build_agent(configuration(c, variant, 11, 'cpu'))
            self.assertIsInstance(a.actor_optimizer, torch.optim.AdamW)
            self.assertEqual(a.actor_optimizer.param_groups[0]['weight_decay'], .01)
            values.append((sum(p.numel() for p in a.actor_network.parameters()),
                           paper.state_digest(a.critic_network.state_dict()), rt.rng_state()))
        self.assertEqual([v[0] for v in values], [4738895, 24493, 24493])
        self.assertEqual(len({v[1] for v in values}), 1)
        self.assertTrue(same_state(values[0][2], values[1][2]))
        self.assertTrue(same_state(values[1][2], values[2][2]))

    def test_pending_replay_commits_once(self):
        """Interrupt inside an update, resume pending, never collect/step twice."""
        import server
        import checks
        c = protocol()
        built = rt.build_agent
        loader = rt.load_agent
        expected_state = []
        collector_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            info = dict(stack={'test': True}, host='synthetic', gpu=None)
            def initialize(cc, variant, seed, out):
                cfg = configuration(cc, variant, seed)
                rt.seed_all(seed)
                a = built(cfg, device='cpu')
                progress = dict(next_episode=0, phase='ready', experiment_protocol_sha256=fp(cc))
                paper.ready_save(out, a, cfg, progress, cc)
                return a, cfg, progress
            def cpu_load(path, **kwargs):
                return loader(path, device='cpu', **kwargs)
            def collection(job, directory, identity, cc, resume=False):
                if 'e00002' in job['job_id']:
                    raise paper.Stopped('Synthetic stop after one committed episode')
                collector_calls.append(job['job_id'])
                a, _ = cpu_load(job['checkpoint'], training=False, restore_random_state=False)
                rt.seed_all(job['sampling_seed'])
                a.memory.begin_episode(job['job_id'], a.policy_version)
                rng = np.random.RandomState(7)
                for i in range(4):
                    x, n, m = [rng.randn(*shape).astype('float32') for shape in [(3+i, 25), (8,), (53,)]]
                    act, v, lp = a.choose_action(x, m, n)
                    a.remember(cands_state=x, node_state=n, mip_state=m, action=act,
                               value=v, log_prob=lp, reward=float(i), done=i==3)
                a.seal_episode(True)
                directory = Path(directory)/'attempt-000'
                directory.mkdir(parents=True)
                import pickle
                with (directory/(job['job_id']+'.pkl')).open('wb') as f:
                    pickle.dump(a.memory.export_dict(), f)
                return dict(valid=True, job_id=job['job_id'], policy_version=a.policy_version), directory
            original_learn = rt.Agent.learn
            def interrupting_learn(a, **kw):
                result = original_learn(a, **kw)
                expected_state.append(copy.deepcopy((a.actor_network.state_dict(), a.critic_network.state_dict(),
                    a.actor_optimizer.state_dict(), a.critic_optimizer.state_dict(), rt.rng_state())))
                raise paper.Stopped('Synthetic interruption after optimizer steps before commit')
            with mock.patch.object(paper, 'RUN', run), mock.patch.object(server, 'preflight', return_value=info), \
                 mock.patch.object(checks, 'target_checks'), mock.patch.object(paper, 'initial', side_effect=initialize), \
                 mock.patch.object(paper, 'native', return_value={}), mock.patch.object(rt, 'job_identity', return_value={}), \
                 mock.patch.object(rt, 'load_agent', side_effect=cpu_load), mock.patch.object(paper, 'cached_job', side_effect=collection):
                with mock.patch.object(rt.Agent, 'learn', interrupting_learn):
                    with self.assertRaises(paper.Stopped):
                        paper.train(c, 'mlp', 11, False)
                out = run/'training/mlp-s11'
                self.assertEqual(rt.read_checkpoint(out/'latest.pt')['progress']['next_episode'], 0)
                self.assertTrue((out/'pending.pt').exists())
                with self.assertRaises(paper.Stopped):
                    paper.train(c, 'mlp', 11, True)
                committed = rt.read_checkpoint(out/'latest.pt')
                self.assertEqual(committed['progress']['next_episode'], 1)
                self.assertEqual(len(collector_calls), 1)
                self.assertFalse((out/'pending.pt').exists())
                a, _ = loader(out/'latest.pt', device='cpu')
                self.assertTrue(same_state(expected_state[0][:4], (a.actor_network.state_dict(), a.critic_network.state_dict(),
                    a.actor_optimizer.state_dict(), a.critic_optimizer.state_dict())))
                self.assertTrue(same_state(expected_state[0][4], committed['rng']))
                self.assertEqual(len(paper.read(out/'budget.json')['attempts']), 2)

    def test_summarizer_exact_coverage_cost_and_partial_refusal(self):
        import analyze
        c = protocol()
        models = [dict(label=v+'-s'+str(s)) for s in (11, 22) for v in ('b0', 'mlp', 'mlp_bc')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'runs'
            with mock.patch.object(analyze, 'ROOT', root), mock.patch.object(analyze, 'RUN', run), \
                 mock.patch.object(analyze, 'frozen_models', return_value=models):
                analyze.summarize(c)
                self.assertFalse((root/'docs/methods/tgppo_paper_aligned_20261106/results/summary.csv').exists())
                for shard in range(17):
                    for name in eval_instances(c, shard):
                        rows = []
                        for ss in range(5):
                            for label in ['native']+[m['label'] for m in models]:
                                calls = 2 if label!='native' else 0
                                if name == c['test'][0] or (name == c['test'][1] and label == 'b0-s11' and ss == 0):
                                    calls = 0
                                rows.append(dict(job_id='eval-{}-{}-s{}'.format(label, name, ss), instance_name=name,
                                    solver_seed=ss, valid=True, proof_completed=True, scip_solve_time=10.,
                                    pdi_horizon_midpoint=100., pdi_horizon_lower=99., pdi_horizon_upper=101.,
                                    nnodes=5, optimize_wall_seconds=11., branch_count=calls,
                                    callback_seconds={'actor': .05*calls} if label!='native' else {}))
                        write(run/'evaluation'/('shard-'+str(shard))/name/'results.json', rows)
                analyze.summarize(c)
                docs = root/'docs/methods/tgppo_paper_aligned_20261106/results'
                with (docs/'summary.csv').open() as stream:
                    summary = list(csv.DictReader(stream))
                self.assertEqual(len(summary), 7)
                with (docs/'paired.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 792)
                self.assertTrue(all(abs(float(x['actor_mean_ms'])-50.) < 1e-8 for x in summary if x['label']!='native'))
                self.assertEqual(paper.read(docs/'results.json')['actor_participation_counts'],
                                 dict(all66=66, common_actor=64, no_actor=1, mixed=1))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
