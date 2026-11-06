import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import experiment as ex
import reporting


class AutomationTests(unittest.TestCase):
    def test_missing_gpu_is_recorded_without_starting_solves(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(ex,'MATRIX',Path(d)), patch.object(ex,'verify_package'), \
                    patch.object(ex.server,'preflight',side_effect=RuntimeError('CUDA unavailable')), \
                    patch.object(ex,'prepare') as prepare, patch.object(ex.server,'train') as train, \
                    patch.dict(ex.os.environ,{'CR_LAUNCH_EPOCH':str(time.time())}):
                with self.assertRaises(RuntimeError):ex.run('candidate',11)
                prepare.assert_not_called(); train.assert_not_called()
                record=ex.server.read(ex.status_path('candidate',11))
                self.assertEqual(record['status'],'failed')
                self.assertGreater(record['charged_wall_seconds'],0)
                with self.assertRaises(FileExistsError):ex.run('candidate',11)

    def test_instance_weights_are_not_solver_repeat_weights(self):
        def row(name,value):
            return dict(instance_name=name,pdi_horizon_midpoint=value,pdi_horizon_lower=value-.1,
                pdi_horizon_upper=value+.1,proof_completed=False,scip_solve_time=120,nnodes=10)
        result=reporting.aggregate([row('a',10),row('a',10),row('b',30)])
        self.assertEqual(result['pdi'],20)
        self.assertAlmostEqual(result['pdi_lower'],19.9)
        self.assertAlmostEqual(result['pdi_upper'],20.1)

    def test_complete_report_and_pairing_on_temporary_fixtures(self):
        # Fixtures live only in a temporary directory, never among method results.
        with tempfile.TemporaryDirectory() as d:
            matrix=Path(d)
            output=matrix/'evaluation/episode-0100';output.mkdir(parents=True)
            names=['a','b','c','d','e']
            models=[dict(variant=v,seed=s) for v in ex.VARIANTS for s in ex.SEEDS]
            ex.write_json(output/'complete.json',{'complete':True})
            ex.write_json(output/'frozen_models.json',dict(development=names,solver_seeds=[0,1],models=models,stack={}))
            def rows(offset):
                return [dict(instance_name=n,solver_seed=s,pdi_horizon_midpoint=100+i+offset,
                    pdi_horizon_lower=99.9+i+offset,pdi_horizon_upper=100.1+i+offset,
                    proof_completed=False,scip_solve_time=120,nnodes=10,branch_count=5,
                    callback_seconds={'actor':.1,'critic':0,'callback':.2},valid=True)
                    for i,n in enumerate(names) for s in (0,1)]
            ex.write_json(output/'native_rows.json',rows(-10))
            for m in models:
                ex.write_json(output/(m['variant']+'-s'+str(m['seed']))/'selected_results.json',
                    rows({'b0':0,'score':-1,'candidate':-2}[m['variant']]))
                train=matrix/m['variant']/'training'/('B0-s'+str(m['seed']))
                ex.write_json(train/'updates.json',[])
            with patch.object(reporting.subprocess,'run') as run:
                run.return_value.returncode=0;run.return_value.stdout='temporary fixture'
                reporting.report(Path(d),matrix,100,{})
            result=ex.server.read(matrix/'results/summary.json')
            self.assertTrue(result['complete'])
            self.assertEqual(len(result['models']),7)
            self.assertEqual(len(result['paired']),10)
            self.assertTrue(all(p['candidate_minus_b0']==-2 and p['candidate_minus_score']==-1
                                for p in result['paired']))


if __name__=='__main__':unittest.main()
