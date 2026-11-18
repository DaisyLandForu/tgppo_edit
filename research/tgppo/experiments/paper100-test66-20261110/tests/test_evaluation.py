"""Synthetic scheduler/provenance/report tests. No torch, SCIP or solves."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c
import analyze


def fake_row(spec, pdi=100.):
    action_key = str(0 if spec['label'] == 'native' else 11) + ':' + spec['canonical_sampling_job_id']
    row = dict(job_id=spec['job_id'], instance_name=spec['instance'], solver_seed=0,
        mode='native' if spec['label'] == 'native' else 'eval', valid=True,
        time_limit_seconds=3600, address_space_limit_bytes=c.task_resources(spec['instance'])['address_space_limit_bytes'],
        worker_cpu_complete=True, cuda_initialized=False, pdi_horizon_lower=pdi-1,
        pdi_horizon_midpoint=pdi, pdi_horizon_upper=pdi+1, scip_solve_time=100.,
        optimize_wall_seconds=101., worker_wall_seconds=102., nnodes=4, proof_completed=True,
        cpu_affinity=[0, 1, 2, 3], cohort_id='fake')
    row.update(sampling_seed=int(hashlib.sha256(action_key.encode()).hexdigest()[:8], 16),
               measurement_version='native-pdi-snapshot-horizon-v2', pdi_horizon_seconds=3600)
    if row['mode'] == 'eval':
        row.update(fallback_action_count=0, callback_seconds=dict(actor=.002, critic=0), branch_count=2)
    return row


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.specs = c.specs(0)[:5]
        self.env = dict(stack={'cpu': 'a'}, host='node-a', cores=[0, 1, 2, 3])

    def tearDown(self):
        self.temp.cleanup()

    def attempt(self, name, specs, environment=None, pdi=100.):
        env = environment or self.env
        path = self.folder / 'attempts' / name
        c.write(path / 'manifest.json', dict(plan_sha256=c.sha(c.PACKAGE / 'plan.json'),
            cohort_id=c.fingerprint(env)[:24], environment=env,
            tasks=[dict(job_id=s['job_id'], spec=s) for s in specs]))
        for spec in specs:
            c.write(path / 'raw' / (spec['job_id']+'.json'), fake_row(spec, pdi))
        return path

    def test_complete330_and_no_duplicate(self):
        all_specs = [s for shard in c.SHARDS for s in c.specs(shard)]
        self.assertEqual(330, len(all_specs))
        self.assertEqual(330, len({(s['instance'], s['label'], s['solver_seed']) for s in all_specs}))
        self.assertEqual({0}, {s['solver_seed'] for s in all_specs})

    def test_common_action_stream(self):
        self.assertEqual(1, len({s['canonical_sampling_job_id'] for s in self.specs}))
        self.assertEqual(set(c.LABELS), {s['label'] for s in self.specs})

    def test_large_instance_serial(self):
        self.assertEqual(1, c.task_resources('mspp16')['workers'])
        self.assertEqual(24*1024**3, c.task_resources('mspp16')['address_space_limit_bytes'])
        self.assertEqual(4200, c.task_resources('mspp16')['watchdog_seconds'])
        self.assertEqual(4, c.task_resources('blend2')['workers'])

    def test_resume_shared_results_without_tmp(self):
        self.attempt('001', self.specs[:3])
        panels, cohorts = c.scan(0, self.folder)
        self.assertFalse(panels)
        self.assertEqual(3, len(next(iter(cohorts.values()))['rows']))
        self.attempt('002', self.specs[3:])
        panels, _ = c.scan(0, self.folder)
        self.assertEqual(5, len(panels[self.specs[0]['instance']]))

    def test_no_cross_environment_partial_panel(self):
        self.attempt('001', self.specs[:3])
        other = dict(self.env, host='node-b')
        self.attempt('002', self.specs[3:], other)
        self.assertFalse(c.scan(0, self.folder)[0])
        self.attempt('003', self.specs, other)
        self.assertEqual(1, len(c.scan(0, self.folder)[0]))

    def test_first_valid_not_best_metric(self):
        self.attempt('001', self.specs, pdi=200.)
        self.attempt('002', self.specs, pdi=10.)
        rows = next(iter(c.scan(0, self.folder)[0].values()))
        self.assertTrue(all(r['pdi_horizon_midpoint'] == 200. for r in rows))

    def test_bad_rows_cannot_complete(self):
        path = self.attempt('001', self.specs)
        c.write(path / 'raw' / (self.specs[0]['job_id']+'.json'), dict(valid=False, status='watchdog_timeout'))
        self.assertFalse(c.scan(0, self.folder)[0])

    def test_horizon_and_critic_guards(self):
        spec = next(s for s in self.specs if s['label'] != 'native')
        row = fake_row(spec)
        row['time_limit_seconds'] = 120
        with self.assertRaises(AssertionError):
            c.validate_row(row, spec)
        row = fake_row(spec)
        row['callback_seconds']['critic'] = 1
        with self.assertRaises(AssertionError):
            c.validate_row(row, spec)

    def test_pdi_bounds_and_proof_deadline(self):
        rows = []
        for spec in self.specs:
            row = fake_row(spec, 50 if spec['label'] == 'mlp_bc' else 100)
            rows.append(dict(row, label=spec['label']))
        effect = next(e for e in analyze.paired(rows) if e['effect'] == 'BC')
        self.assertEqual(-50, effect['mean_pdi_difference'])
        self.assertEqual(-52, effect['pdi_difference_lower'])
        self.assertEqual(-48, effect['pdi_difference_upper'])
        native = copy.deepcopy(rows[0])
        native['scip_solve_time'] = 3601
        self.assertEqual(0, analyze.statistics([native], 'native')['proof_rate'])
        self.assertEqual(7200, analyze.statistics([native], 'native')['par2'])

    def test_partial_matrix_has_no_ranking(self):
        old_docs, old_scan = analyze.DOCS, analyze.scan
        try:
            analyze.DOCS = self.folder / 'report'
            analyze.scan = lambda shard: ({}, {})
            self.assertFalse(analyze.summarize()['complete'])
            self.assertFalse((analyze.DOCS / 'summary.csv').exists())
            self.assertIsNone(c.read(analyze.DOCS / 'progress.json')['ranking'])
        finally:
            analyze.DOCS, analyze.scan = old_docs, old_scan

    def test_full330_reporting_including_native_without_actor_fields(self):
        old_docs, old_scan = analyze.DOCS, analyze.scan
        try:
            analyze.DOCS = self.folder / 'report'
            def synthetic_scan(shard):
                panels = {}
                for spec in c.specs(shard):
                    row = fake_row(spec)
                    panels.setdefault(spec['instance'], []).append(dict(row, label=spec['label'],
                        result_path='synthetic', result_sha256='synthetic'))
                return panels, {'fake': {'environment': self.env,
                    'rows': {r['job_id']: r for values in panels.values() for r in values}}}
            analyze.scan = synthetic_scan
            self.assertTrue(analyze.summarize()['complete'])
            result = c.read(analyze.DOCS / 'results.json')
            self.assertEqual(330, result['tasks'])
            self.assertEqual(5, len(result['summary']))
            self.assertIsNone(result['summary'][0]['actor_mean_ms'])
            self.assertTrue((analyze.DOCS / 'REPORT.md').exists())
        finally:
            analyze.DOCS, analyze.scan = old_docs, old_scan


if __name__ == '__main__':
    unittest.main(verbosity=2)
