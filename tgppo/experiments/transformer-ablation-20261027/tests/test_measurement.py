import unittest
import json
from pathlib import Path
from project.measurement import percent_gap, horizon_integral
from project.environment import Environment
from project.runtime import make_job
from project.utils.settings import state_dims


class MeasurementTests(unittest.TestCase):
    def test_percent_gap_is_not_scip_relative_gap(self):
        self.assertAlmostEqual(percent_gap(100., 80.), 20.)
        self.assertAlmostEqual(percent_gap(-80., -100.), 20.)
        self.assertEqual(percent_gap(0., 0.), 0.)
        self.assertEqual(percent_gap(2., -1.), 100.)
        self.assertEqual(percent_gap(100., 1e20), 100.)

    def test_early_proof_zero_extension_and_timeout_hold_extension(self):
        self.assertEqual(horizon_integral(12., 2., 120., 0., True)['pdi_horizon_midpoint'], 12.)
        self.assertEqual(horizon_integral(12., 119., 120., 20., False)['pdi_horizon_midpoint'], 32.)

    def test_overshoot_is_an_interval_not_fabricated_exact_integral(self):
        r = horizon_integral(300., 120.01, 120., 20., False)
        self.assertAlmostEqual(r['pdi_horizon_lower'], 299.)
        self.assertAlmostEqual(r['pdi_horizon_upper'], 300.)
        self.assertFalse(r['pdi_horizon_exact'])

    def test_real_solved_presolve_and_timeout_flush(self):
        root = Path(__file__).resolve().parents[4]
        cutoffs = json.loads((root / 'tgppo/datasets/cutoffs.json').read_text())
        for name in ('misc03', 'lseu', 'air04'):
            job = make_job('metric-' + name, root / 'tgppo/datasets/train_instances',
                           name, 1., 0, 11, cutoffs, 'native')
            env = Environment('cpu', None, state_dims, job['limits'], job['params'], 0, mode='native')
            env.reset(job['instance'], job['cutoff'], cutoff_provenance=job['cutoff_provenance'])
            valid, info, _ = env.run_episode()
            self.assertTrue(valid, info)
            self.assertGreaterEqual(info['pdi_native_complete_at_stop'] + 1e-6,
                                    info['pdi_native_raw_before_flush'])
            self.assertLessEqual(info['pdi_horizon_upper'], 100. + 1e-5)
            self.assertGreaterEqual(info['pdi_horizon_lower'], 0.)
            snapshot = info['integral_snapshot']
            expected = snapshot['integral'] + (snapshot['solving_time'] - snapshot['last_update_time']) * snapshot['previous_percent_gap']
            self.assertAlmostEqual(info['pdi_native_complete_at_stop'], expected)
            if name == 'misc03':
                self.assertEqual(info['nnodes'], 0)
                self.assertEqual(snapshot['integral'], 0.)
                self.assertGreater(info['pdi_native_complete_at_stop'], 0.)
                self.assertEqual(info['pdi_extension_basis'], 'verified_proof_certificate')
            print('MEASUREMENT_EVIDENCE=' + json.dumps(info, sort_keys=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
