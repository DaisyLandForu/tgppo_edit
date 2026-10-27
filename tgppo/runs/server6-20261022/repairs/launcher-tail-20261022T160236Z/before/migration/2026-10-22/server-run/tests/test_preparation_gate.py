import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LAUNCH=Path(__file__).resolve().parents[1]
ROOT=LAUNCH.parents[2]
spec=importlib.util.spec_from_file_location('preparation_gate',LAUNCH/'check_preparation.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


class PreparationGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        self.base=dict(protocol_sha256='frozen-protocol',stack={'cpu_model':'Gold 6148','binding':'same'},
                       baselines={str(i):{'result':{'valid':True}} for i in range(75)})
        self.proof=dict(parent_protocol_sha256='frozen-protocol',stack=self.base['stack'],
                        gpu='Tesla V100-SXM2-32GB',passed=True)

    def put(self,name,payload):
        relative=dict(gate.ARTIFACTS)[name];p=self.run/relative
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(payload))
        return p

    def complete(self):
        self.put('native_baselines',self.base)
        self.put('gpu_bridge',self.proof);self.put('throughput',self.proof)

    def test_missing_artifacts_are_pending_and_no_state_is_created(self):
        result=gate.inspect(self.run)
        self.assertFalse(result['ready']);self.assertEqual(len(result['pending_files']),3)
        self.assertEqual(list(self.run.iterdir()),[])

    def test_completed_baseline_does_not_allow_early_training(self):
        path=self.put('native_baselines',self.base);before=path.read_bytes()
        result=gate.inspect(self.run)
        self.assertEqual(result['stages']['native_baselines'],'passed')
        self.assertFalse(result['ready']);self.assertEqual(len(result['pending_files']),2)
        self.assertEqual(path.read_bytes(),before)

    def test_all_three_matching_passed_artifacts_are_ready(self):
        self.complete();self.assertTrue(gate.inspect(self.run)['ready'])

    def test_failed_gpu_proof_and_incomplete_baseline_are_rejected(self):
        self.complete();self.put('gpu_bridge',dict(self.proof,passed=False))
        self.assertEqual(gate.inspect(self.run)['stages']['gpu_bridge'],'failed')
        self.put('gpu_bridge',self.proof)
        broken=copy.deepcopy(self.base);broken['baselines'].pop('0')
        self.put('native_baselines',broken)
        self.assertEqual(gate.inspect(self.run)['stages']['native_baselines'],'failed')

    def test_mismatched_protocol_cpu_and_gpu_are_rejected(self):
        for key,value in [('parent_protocol_sha256','other-protocol'),('stack',{'cpu_model':'different'}),('gpu','other-GPU')]:
            with self.subTest(key=key):
                self.complete();self.put('throughput',dict(self.proof,**{key:value}))
                result=gate.inspect(self.run)
                self.assertFalse(result['ready']);self.assertTrue(result['errors'])

    def test_corrupt_json_and_null_baseline_result_are_rejected(self):
        path=self.put('gpu_bridge',self.proof);path.write_text('{broken')
        self.assertEqual(gate.inspect(self.run)['stages']['gpu_bridge'],'invalid')
        self.complete();broken=copy.deepcopy(self.base);broken['baselines']['0']['result']=None
        self.put('native_baselines',broken)
        self.assertEqual(gate.inspect(self.run)['stages']['native_baselines'],'failed')

    def test_command_reports_pending_without_a_traceback(self):
        result=subprocess.run([sys.executable,str(LAUNCH/'check_preparation.py'),'--run-root',str(self.run)],capture_output=True,text=True)
        self.assertEqual(result.returncode,75)
        self.assertFalse(json.loads(result.stdout)['ready'])
        self.assertNotIn('Traceback',result.stdout+result.stderr)

    def test_launcher_stops_before_creating_a_training_directory(self):
        env=dict(os.environ,RESEARCH_ROOT=str(ROOT),SERVER_RUN_ROOT=str(self.run),TGPPO_PYTHON=sys.executable)
        result=subprocess.run(['bash',str(LAUNCH/'custom_job.sh'),'train','B0','11'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,75,result.stdout+result.stderr)
        self.assertIn('Preparation is still running/incomplete',result.stdout+result.stderr)
        self.assertNotIn('Traceback',result.stdout+result.stderr)
        self.assertFalse((self.run/'training').exists())


if __name__=='__main__':unittest.main(verbosity=2)
