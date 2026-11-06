import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import launcher as flex
base=flex.base


def info(cpu):
    return dict(stack=dict(cpu_model=cpu,machine='x86_64',torch_version='1.13.1+cu117',
        binding_sha256='same',libscip_sha256='same',python_abi='3.8'),
        gpu='Tesla V100-SXM2-32GB',host='temporary-test',workers=4)


class CpuFlexTests(unittest.TestCase):
    def test_cpu_model_and_v100_variant_allowed_but_software_not_relaxed(self):
        old,new=info('Gold6148'),info('other CPU')
        new['gpu']='Tesla V100-PCIE-16GB'
        flex.check_platform(new,old)
        new['stack']['binding_sha256']='different'
        with self.assertRaises(ValueError):flex.check_platform(new,old)
        new=info('other CPU');new['gpu']='A100'
        with self.assertRaises(ValueError):flex.check_platform(new,old)

    def test_training_package_and_checkpoint_protocol_stay_compatible(self):
        base.verify_package()
        for variant in base.VARIANTS:
            p=base.status_path(variant,11)
            if p.exists():
                record=base.server.read(p)
                self.assertEqual(base.fingerprint(base.protocol(variant)[0]),record['protocol_sha256'])

    def test_prepare_isolates_cpu_normalizers_and_keeps_canonical_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            matrix=Path(d)/'matrix'
            root=Path(d)/'root'
            def protocol(v):return dict(variant=v),matrix/v
            calls=[]
            def baseline(root,c,run):
                calls.append(str(run))
                base.write_json(run/'normalization/baselines.json',dict(
                    stack=flex.CURRENT_INFO['stack'],protocol_sha256=base.fingerprint(c)))
            def bundle(root,c,run):
                value=base.server.read(run/'normalization/baselines.json')
                self.assertEqual(value['stack'],flex.CURRENT_INFO['stack'])
                self.assertEqual(value['protocol_sha256'],base.fingerprint(c))
                return value
            def adopt(source,c,target,stack):
                value=base.server.read(source)
                if value['stack']!=stack:return False
                base.write_json(target/'normalization/baselines.json',dict(value,protocol_sha256=base.fingerprint(c)))
                return True
            with patch.object(base,'MATRIX',matrix),patch.object(base,'ROOT',root), \
                    patch.object(base,'protocol',side_effect=protocol), \
                    patch.object(base,'adopt_native',side_effect=adopt), \
                    patch.object(base.server,'baseline',side_effect=baseline), \
                    patch.object(base.server,'baseline_bundle',side_effect=bundle), \
                    patch.object(base.server,'params',return_value={}), \
                    patch('engineering.bridge',return_value={'passed':True}):
                base.write_json(matrix/'matrix_environment.json',info('CPU-A'))
                contexts=[]
                for variant,seed,cpu in [('score',11,'CPU-A'),('score',22,'CPU-B'),('candidate',11,'CPU-B')]:
                    flex.CURRENT_JOB=(variant,seed)
                    c,run=protocol(variant);before=copy.deepcopy(c)
                    flex.prepare(variant,c,run,info(cpu))
                    self.assertEqual(c,before)
                    contexts.append(flex.CONTEXT)
                    self.assertEqual((flex.CONTEXT/'training').resolve(),(run/'training').resolve())
                    self.assertFalse((run/'normalization/baselines.json').exists())
                self.assertEqual(len(calls),2)
                self.assertNotEqual(contexts[0],contexts[1])
                self.assertEqual(base.server.read(contexts[1]/'normalization/baselines.json')['stack']['cpu_model'],'CPU-B')
                self.assertTrue((matrix/'results/TRAINING_ENVIRONMENTS.md').exists())

    def test_training_uses_context_without_relabeling_config_or_resetting_resume(self):
        config={'variant':'score','unchanged':True}
        with patch.object(flex,'CONTEXT',Path('/temporary-context')), \
                patch.object(flex,'CURRENT_JOB',('score',22)),patch.object(flex,'_TRAIN',return_value='ok') as train:
            self.assertEqual(flex.train('root',config,'original-run','B0',22,True,'cuda'),'ok')
            self.assertIs(train.call_args.args[1],config)
            self.assertEqual(train.call_args.args[2],Path('/temporary-context'))
            self.assertTrue(train.call_args.args[5])

    def test_cross_cpu_collection_cache_separation(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'e00040'
            selected=[({'job_id':'fixture','valid':True},Path(d)/'trajectory.pkl')]
            with patch.object(flex,'_CACHED',return_value=selected) as cached:
                for cpu in ('A','B'):
                    with patch.object(flex,'CURRENT_INFO',info(cpu)):
                        flex.cached([{'mode':'collect'}],folder,{'policy':'same'},4,180)
                self.assertNotEqual(cached.call_args_list[0].args[1],cached.call_args_list[1].args[1])
                self.assertEqual(base.server.read(folder/'selected_results.json'),[selected[0][0]])

    def test_preflight_failure_retry_archives_failure_and_preserves_spent_budget(self):
        with tempfile.TemporaryDirectory() as d:
            matrix=Path(d)
            def protocol(v):return dict(gpu_job_budget_seconds=8400,finalization_budget_seconds=3600,
                freeze_deadline='2099-01-01T00:00:00+08:00'),matrix/v
            old=dict(status='failed',error='ValueError(\'Matrix CPU/software/GPU differs\')',
                attempts=[dict(id='old-attempt',wall_seconds=12.,started_epoch=1.)],charged_wall_seconds=12.)
            with patch.object(base,'MATRIX',matrix),patch.object(base,'protocol',side_effect=protocol):
                path=base.status_path('candidate',11);base.write_json(path,old)
                self.assertTrue(flex.automatic_resume('candidate',11))
                archived=matrix/'cpu-flex/prior-failures/candidate-s11/old-attempt.json'
                self.assertEqual(base.server.read(archived),old)
                with patch.object(base,'verify_package'),patch.object(base.server,'preflight',side_effect=RuntimeError('test GPU unavailable')), \
                        patch.dict(base.os.environ,{'CR_LAUNCH_EPOCH':str(time.time())}):
                    with self.assertRaises(RuntimeError):base.run('candidate',11,True)
                new=base.server.read(path)
                self.assertEqual(len(new['attempts']),2)
                self.assertGreater(new['charged_wall_seconds'],12.)
                self.assertEqual(new['attempts'][0],old['attempts'][0])

    def test_existing_checkpoint_is_not_automatically_restarted(self):
        with tempfile.TemporaryDirectory() as d:
            matrix=Path(d)
            with patch.object(base,'MATRIX',matrix),patch.object(base,'protocol',side_effect=lambda v:({},matrix/v)):
                base.write_json(base.status_path('b0',11),dict(status='failed',error='Matrix CPU/software/GPU differs'))
                out=base.server.group_dir(matrix/'b0','B0',11);out.mkdir(parents=True)
                (out/'latest.pt').touch()
                self.assertFalse(flex.automatic_resume('b0',11))


if __name__=='__main__':unittest.main()
