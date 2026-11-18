"""Exercise real frozen cache routing with a synthetic worker; never solve/train."""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import resume as adapter


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)/'episodes/92'
        self.job = dict(mode='collect', job_id='collect-e00092-test-s3')
        self.identity = dict(checkpoint_sha256='same-checkpoint', sampling_seed=123,
            task=dict(instance_sha256='same-instance', solver_seed=3, time_limit=3600),
            stack=dict(cpu_model='old CPU', machine='x86_64', torch_version='1.13.1',
                       binding_sha256='same-binding'))
        self.current = copy.deepcopy(self.identity)
        self.current['stack']['cpu_model'] = 'new CPU'
        adapter.paper.write(self.folder/'identity.json', self.identity)
        adapter.paper.write(self.folder/'attempt-000/job.json', self.job)
        self.old_bytes = {p.relative_to(self.folder): p.read_bytes()
                          for p in self.folder.rglob('*') if p.is_file()}
        self.calls = []
        runtime = types.ModuleType('project.runtime')

        def run_jobs(jobs, attempt, workers, watchdog):
            self.calls.append((jobs, attempt, workers, watchdog))
            row = dict(valid=True, job_id=jobs[0]['job_id'])
            adapter.paper.write(attempt/(jobs[0]['job_id']+'.json'), row)
            (attempt/(jobs[0]['job_id']+'.pkl')).write_bytes(b'synthetic trajectory')
            return [row]

        runtime.run_jobs = run_jobs
        module_patch = patch.dict(sys.modules, {'project.runtime': runtime})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        self.cached = adapter.cohort_cached_job(adapter.paper.cached_job)

    def call(self, identity=None, explicit_resume=True):
        return self.cached(self.job, self.folder, identity or self.current,
                           dict(watchdog=4200), explicit_resume)

    def assert_old_preserved(self):
        for relative, content in self.old_bytes.items():
            self.assertEqual((self.folder/relative).read_bytes(), content)

    def test_same_cpu_uses_original_cache(self):
        _, attempt = self.call(self.identity)
        self.assertEqual(attempt, self.folder/'attempt-001')
        self.assertFalse((self.folder/'resume-cohorts').exists())
        self.assert_old_preserved()

    def test_cpu_move_preserves_old_records_and_routes_new_attempt(self):
        row, attempt = self.call()
        destination = self.folder/'resume-cohorts'/adapter.paper.fp(self.current)
        self.assertEqual(attempt, destination/'attempt-000')
        self.assertTrue(row['valid'])
        self.assertEqual(adapter.paper.read(destination/'identity.json'), self.current)
        provenance = adapter.paper.read(destination/'migration.json')
        self.assertFalse(provenance['optimizer_or_protocol_changed'])
        self.assertEqual(provenance['checkpoint_sha256'], 'same-checkpoint')
        self.assertEqual(provenance['sampling_seed'], 123)
        self.assertEqual(self.calls[0][2:], (1, 4200))
        self.assert_old_preserved()

    def test_repeated_resume_reuses_valid_new_cohort(self):
        first = self.call()
        self.assertEqual(self.call(), first)
        self.assertEqual(len(self.calls), 1)
        self.assert_old_preserved()

    def test_task_checkpoint_and_rng_changes_rejected(self):
        for field, value in [('checkpoint_sha256', 'different-policy'),
                             ('sampling_seed', 124), ('task', dict(solver_seed=4))]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.current)
                changed[field] = value
                with self.assertRaisesRegex(ValueError, 'task/checkpoint/action'):
                    self.call(changed)
        self.assertFalse(self.calls)
        self.assert_old_preserved()

    def test_software_architecture_or_missing_cpu_rejected(self):
        for field, value in [('binding_sha256', 'different-binding'),
                             ('torch_version', 'different-torch'),
                             ('machine', 'aarch64'), ('cpu_model', '')]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.current)
                changed['stack'][field] = value
                with self.assertRaisesRegex(ValueError, 'software/architecture'):
                    self.call(changed)
        self.assertFalse(self.calls)

    def test_native_cache_identity_guard_remains(self):
        self.job['mode'] = 'native'
        with self.assertRaisesRegex(ValueError, 'Cached task identity changed'):
            self.call()
        self.assertFalse(self.calls)
        self.assert_old_preserved()

    def test_migration_requires_resume(self):
        with self.assertRaisesRegex(ValueError, 'explicit resume'):
            self.call(explicit_resume=False)
        self.assertFalse(self.calls)

    def test_completed_old_attempt_preserved_on_cpu_change(self):
        old_result = self.folder/'attempt-000'/ (self.job['job_id']+'.json')
        old_pickle = old_result.with_suffix('.pkl')
        adapter.paper.write(old_result, dict(valid=True, old_cpu=True))
        old_pickle.write_bytes(b'old uncommitted trajectory')
        self.call()
        self.assertTrue(adapter.paper.read(old_result)['old_cpu'])
        self.assertEqual(old_pickle.read_bytes(), b'old uncommitted trajectory')
        self.assertEqual(len(self.calls), 1)


if __name__ == '__main__':
    unittest.main()
