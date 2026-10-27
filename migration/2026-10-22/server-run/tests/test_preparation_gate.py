import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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

    def run_preparation_during_edit(self,source,replacement,pilot_exit=0):
        """Exercise the real shell entrypoint with CPU-only fixture commands."""
        self.complete()
        launch=self.run/'launch';launch.mkdir()
        script=launch/'custom_job.sh';script.write_text(source)
        for name in ['env.sh','check_preparation.py']:
            shutil.copy2(LAUNCH/name,launch/name)
        for name in ['run.sh','gpu_bridge.sh']:
            helper=launch/name;helper.write_text('#!/usr/bin/env bash\nexit 0\n');helper.chmod(0o755)
        (launch/'verify_environment.py').write_text('# fixture environment checks\n')
        pilot=self.run/'tgppo/experiments/server-20261022/code/server_pilot.py'
        pilot.parent.mkdir(parents=True)
        pilot.write_text("import os\nfrom pathlib import Path\n"
            "fd=os.open(os.environ['PROBE_RELEASE'],os.O_RDWR)\n"
            "Path(os.environ['PROBE_READY']).write_text('ready')\n"
            "os.read(fd,1)\nos.close(fd)\n"
            "raise SystemExit(int(os.environ['PROBE_PILOT_EXIT']))\n")
        ready=self.run/'probe-ready';release=self.run/'probe-release';os.mkfifo(release)
        env=dict(os.environ,RESEARCH_ROOT=str(self.run),SERVER_RUN_ROOT=str(self.run),
                 TGPPO_PYTHON=sys.executable,PROBE_READY=str(ready),PROBE_RELEASE=str(release),
                 PROBE_PILOT_EXIT=str(pilot_exit))
        process=subprocess.Popen(['bash',str(script),'prepare'],env=env,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:
            deadline=time.monotonic()+8
            while not ready.exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists(),'Fixture pilot did not start')
            script.write_text(replacement)
            with release.open('w') as handle:handle.write('x')
            stdout,stderr=process.communicate(timeout=8)
            return process.returncode,stdout+stderr
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL);process.communicate(timeout=8)

    def test_prepare_finishes_without_rereading_an_edited_launcher(self):
        source=(LAUNCH/'custom_job.sh').read_text()
        code,output=self.run_preparation_during_edit(source,source+'\nthis_must_not_execute_after_the_job\n')
        self.assertEqual(code,0,output)
        self.assertIn('Preparation completed: all three evidence stages passed.',output)
        self.assertNotIn('command not found',output)

    def test_prepare_preserves_a_real_stage_failure_during_launcher_edit(self):
        source=(LAUNCH/'custom_job.sh').read_text()
        code,output=self.run_preparation_during_edit(source,source+'\nthis_must_not_execute_after_the_job\n',pilot_exit=23)
        self.assertEqual(code,23,output)
        self.assertNotIn('Preparation completed:',output)


if __name__=='__main__':unittest.main(verbosity=2)
