import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

LAUNCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAUNCH))
spec = importlib.util.spec_from_file_location('parallel_post', LAUNCH/'parallel_post.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class ParallelPostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'runs'
        self.post = self.run/'post-training-20261023'
        self.post.mkdir(parents=True)
        self.c = dict(milestones=[100, 200, 300, 400, 500], solver_seeds=[0, 1, 2],
                      development=['d1', 'd2', 'd3', 'd4', 'd5'], workers=4,
                      worker_address_space_bytes=8589934592, time_limit=120., watchdog=180.)
        self.audit = dict(stack={'cpu_model': 'Gold 6148'}, jobs=[],
                          datasets={'test66': {'t{:02d}'.format(i): {} for i in range(66)}})
        p.write_json(self.post/'training_audit.json', self.audit)

    def rows(self, names=None, pdi=10.):
        return [dict(instance_name=n, solver_seed=s, valid=True, task_terminal=True,
                     worker_cpu_complete=True, cuda_initialized=False,
                     callback_seconds={'actor': .1, 'critic': 0.},
                     pdi_horizon_midpoint=pdi, pdi_horizon_lower=pdi, pdi_horizon_upper=pdi,
                     scip_solve_time=120., proof_completed=False, nnodes=10,
                     worker_cpu_seconds=120., worker_wall_seconds=121.)
                for n in (names or self.c['development']) for s in self.c['solver_seeds']]

    def checkpoint_fixture(self):
        out = p.server.group_dir(self.run, 'B0', 11)
        out.mkdir(parents=True)
        checkpoints = {}
        for e in self.c['milestones']:
            cp = out/('episode-{:04d}.pt'.format(e))
            cp.write_bytes(('fixture checkpoint '+str(e)).encode())
            checkpoints[str(e)] = {'sha256': p.sha256(cp)}
        row = dict(group='B0', seed=11, files={}, checkpoints=checkpoints)
        self.audit['jobs'] = [row]
        return out, row

    def test_shards_are_disjoint_complete_and_independent_of_input_order(self):
        names = list(self.audit['datasets']['test66'])
        shards = p.partitions(names)
        self.assertEqual([len(s) for s in shards], [11]*6)
        self.assertEqual(set(sum(shards, [])), set(names))
        self.assertEqual(len(sum(shards, [])), 66)
        self.assertEqual(shards, p.partitions(list(reversed(names))))

    def test_missing_seed_duplicate_cuda_and_critic_are_not_filtered_out(self):
        rows = self.rows()
        for invalid in [rows[:-1], rows+[rows[0]],
                        [dict(r, cuda_initialized=True) for r in rows],
                        [dict(r, callback_seconds={'critic': .1}) for r in rows]]:
            with self.assertRaises(ValueError):
                p.valid_deployment(invalid, self.c['development'], self.c['solver_seeds'])

    def test_partial_report_waits_for_all_models_and_a_fixed_prefix(self):
        complete = {(m, s, 1): [] for m, s in p.METHODS}
        self.assertEqual(p.completed_prefix(complete), 0)
        complete.update({(m, s, 0): [] for m, s in p.METHODS[:-1]})
        self.assertEqual(p.completed_prefix(complete), 0)
        complete[(*p.METHODS[-1], 0)] = []
        self.assertEqual(p.completed_prefix(complete), 2)

    def test_repetitions_are_averaged_before_instance_mean(self):
        # Different repetition counts must not turn one instance into extra samples.
        rows = self.rows(['a'], 10.) + self.rows(['b'], 30.)*2
        result = p.metrics(rows, ['a', 'b'], 120.)
        self.assertEqual(result['mean_pdi'], 20.)
        self.assertEqual(result['mean_par2'], 240.)

    def test_wrong_cpu_is_rejected_before_baseline_or_solver(self):
        with patch.object(p.server, 'preflight', return_value={'stack': {'cpu_model': 'E5'}}), \
                patch.object(p.server, 'baseline_bundle') as baseline:
            with self.assertRaisesRegex(ValueError, 'NODE_MISMATCH'):
                p.preflight(self.root, self.c, self.run, self.post, self.audit)
            baseline.assert_not_called()

    def test_select_shard_preserves_original_job_identity_rng_and_budget(self):
        out, row = self.checkpoint_fixture()
        cutoffs = {n: dict(cutoff=10., status='optimal', source='fixture') for n in self.c['development']}
        p.write_json(self.root/'tgppo/datasets/cutoffs.json', cutoffs)
        base = out/'development/episode-0100'

        def cache(jobs, destination, identity, workers, watchdog):
            self.assertEqual(len(jobs), 15)
            self.assertEqual((workers, watchdog), (4, 180.))
            self.assertEqual(identity, p.development_identity(self.c, self.audit, row, 100))
            for job in jobs:
                self.assertEqual(job['sampling_seed'], p.server.action_seed(11, job['job_id']))
                self.assertEqual(job['limits']['time_limit'], 120.)
                self.assertEqual(job['mode'], 'eval')
                self.assertEqual(job['address_space_limit_bytes'], 8589934592)
            rows = self.rows()
            p.write_json(destination/'selected_results.json', rows)
            return [(r, None) for r in rows]

        with patch.object(p, 'preflight', return_value={'stack': self.audit['stack']}), \
                patch.object(p.server, 'cached_jobs', side_effect=cache), \
                patch.object(p, 'finalize_one', side_effect=p.post.Pending('siblings pending')):
            p.select_shard(self.root, self.c, self.run, self.post, self.audit, 'B0', 11, 100)
        self.assertTrue((base/'parallel_verified.json').exists())
        self.assertFalse((out/'selection.json').exists())

    def test_finalize_uses_the_same_dev_rule_and_runs_no_solver(self):
        out, row = self.checkpoint_fixture()
        for episode in self.c['milestones']:
            base = out/'development'/('episode-{:04d}'.format(episode))
            p.write_json(base/'identity.json', p.development_identity(self.c, self.audit, row, episode))
            p.write_json(base/'selected_results.json', self.rows(pdi=5. if episode >= 400 else 10.))
        selected = dict(group='B0', seed=11, best={'episode': 400})
        exports = []

        def export(source, target):
            exports.append(source.name)
            target.write_bytes(b'fixture actor')

        with patch.object(p.server, 'cached_jobs') as solve, \
                patch.object(p.server, 'export_actor', side_effect=export), \
                patch.object(p.post, 'selected_record', return_value=selected):
            p.finalize_one(self.root, self.c, self.run, self.post, self.audit, 'B0', 11)
            solve.assert_not_called()
        selection = p.read(out/'selection.json')
        self.assertEqual(selection['best']['episode'], 400)  # Exact tie at 500 favors earlier.
        self.assertEqual(exports, ['episode-0400.pt', 'episode-0500.pt'])
        self.assertEqual(selection['stack'], self.audit['stack'])

    def test_test_shards_cannot_start_before_all_selections_freeze(self):
        with patch.object(p.post, 'freeze', side_effect=p.post.Pending('not selected')), \
                patch.object(p, 'preflight') as preflight, patch.object(p.server, 'evaluate') as solve:
            with self.assertRaises(p.post.Pending):
                p.evaluate_shard(self.root, self.c, self.run, self.post, self.audit, {}, 'B0', 11, 0)
            preflight.assert_not_called()
            solve.assert_not_called()

    def test_manifest_cannot_change_after_creation(self):
        p.manifest(self.c, self.post, self.audit)
        changed = dict(self.audit)
        changed['datasets'] = {'test66': {'different{:02d}'.format(i): {} for i in range(66)}}
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            p.manifest(self.c, self.post, changed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
