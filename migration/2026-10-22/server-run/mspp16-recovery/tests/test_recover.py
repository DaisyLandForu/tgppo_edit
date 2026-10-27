import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ENTRY = Path(__file__).resolve().parents[1]/'recover.py'
spec = importlib.util.spec_from_file_location('mspp16_recover', ENTRY)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'runs'
        self.work = self.run/'test/mspp16-recovery-20261023'
        self.work.mkdir(parents=True)
        self.names = ['n{:02d}'.format(i) for i in range(65)]+['mspp16']
        self.c = dict(solver_seeds=[0, 1, 2], time_limit=120., worker_address_space_bytes=8*1024**3)
        self.audit = dict(stack={'cpu_model': 'Gold 6148'}, datasets={'test66': {
            n: dict(sha256='fixture-'+n) for n in self.names}})
        self.actor = self.root/'actor.pt'
        self.actor.write_bytes(b'fixture actor')
        self.frozen = dict(models=[dict(group='B0', seed=11, exports={'best': {
            'path':'actor.pt', 'sha256':r.p.sha256(self.actor)}})])
        self.cutoffs = {n:dict(cutoff=1., status='optimal', source='fixture') for n in self.names}
        r.p.write_json(self.root/'tgppo/datasets/cutoffs.json', self.cutoffs)

    def rows(self, names, mode='eval', allowance=8*1024**3):
        return [dict(job_id='test-{}-s{}'.format(n,s), instance_name=n, solver_seed=s,
            valid=True, task_terminal=True, worker_cpu_complete=True, cuda_initialized=False,
            mode=mode, callback_seconds={'critic':0., 'actor':.1}, time_limit_seconds=120.,
            pdi_horizon_midpoint=100., pdi_horizon_lower=99.9, pdi_horizon_upper=100.1,
            worker_cpu_seconds=120., worker_wall_seconds=121., scip_solve_time=120.,
            nnodes=10, proof_completed=False, address_space_limit_bytes=allowance)
            for n in names for s in (0,1,2)]

    def original(self):
        key = r.p.sha256(self.actor)
        base = self.run/'test/official66-best'/key[:16]
        directory = base/'attempt-000'
        directory.mkdir(parents=True)
        lock = self.run/'post-training-20261023/.evaluate-B0-s11.lock'
        lock.parent.mkdir(parents=True)
        lock.write_text('')
        identity = dict(actor_sha256=key, stack=self.audit['stack'], protocol=r.p.fingerprint(self.c),
                        names=self.names, instance_hashes={n:self.audit['datasets']['test66'][n]['sha256'] for n in self.names})
        r.p.write_json(base/'identity.json', identity)
        jobs=[]
        for n in self.names:
            for s in (0,1,2):
                jobs.append(r.p.server.bound_job(self.root,self.c,self.cutoffs,n,s,11,'eval',self.actor,
                    job_id='test-{}-s{}'.format(n,s),instances=self.root/'tgppo/datasets/test_instances'))
        r.p.write_json(directory/'manifest.json',dict(identity=identity,jobs=jobs))
        for row in self.rows(self.names[:-1]):
            r.p.write_json(directory/(row['job_id']+'.json'),row)
        for s in (0,1,2):
            r.p.write_json(directory/('test-mspp16-s{}.json'.format(s)),dict(job_id='test-mspp16-s{}'.format(s),
                instance_name='mspp16',solver_seed=s,valid=False,task_terminal=False,status='worker_exit_error',exitcode=-11))
        return base

    def test_snapshot_preserves_195_sources_and_exactly_three_failed_jobs(self):
        self.original()
        snapshot=r.original_snapshot(self.root,self.c,self.run,self.audit,self.frozen,'B0',11)
        self.assertEqual(len(snapshot['valid_sources']),195)
        self.assertEqual(len(snapshot['failures']),3)
        self.assertEqual(len(snapshot['retry_jobs']),3)
        self.assertTrue(all(j['instance_name']=='mspp16' for j in snapshot['retry_jobs']))

    def test_other_instance_failures_are_never_silently_excluded(self):
        base=self.original()
        path=base/'attempt-000/test-n00-s0.json'
        row=r.p.read(path);row['valid']=False;r.p.write_json(path,row)
        with self.assertRaisesRegex(ValueError,'Unexpected failed instance'):
            r.original_snapshot(self.root,self.c,self.run,self.audit,self.frozen,'B0',11)

    def test_retry_changes_only_address_space_not_rng_or_solver_budget(self):
        self.original()
        snapshot=r.original_snapshot(self.root,self.c,self.run,self.audit,self.frozen,'B0',11)
        original=copy.deepcopy(snapshot['retry_jobs'])
        jobs=r.retry_jobs(snapshot)
        self.assertEqual(snapshot['retry_jobs'],original)
        for before,after in zip(original,jobs):
            self.assertEqual(after,dict(before,address_space_limit_bytes=24*1024**3))
            self.assertEqual(after['sampling_seed'],r.p.server.action_seed(11,after['job_id']))

    def test_retry_rejects_different_instance_or_increased_time_budget(self):
        self.original()
        model=r.original_snapshot(self.root,self.c,self.run,self.audit,self.frozen,'B0',11)
        for field,value in [('instance_name','n00'),('limits',{'time_limit':3600.})]:
            changed=copy.deepcopy(model);changed['retry_jobs'][0][field]=value
            with self.assertRaises(ValueError):r.retry_jobs(changed)

    def test_changed_original_result_is_rejected(self):
        base=self.original()
        model=r.original_snapshot(self.root,self.c,self.run,self.audit,self.frozen,'B0',11)
        path=base/'attempt-000/test-n00-s0.json';row=r.p.read(path);row['nnodes']=11;r.p.write_json(path,row)
        with self.assertRaisesRegex(ValueError,'Recorded source changed'):
            r.original_rows(self.root,model)

    def test_running_original_job_blocks_snapshot(self):
        self.original()
        path=self.run/'post-training-20261023/.evaluate-B0-s11.lock'
        with path.open('r') as handle:
            r.fcntl.flock(handle,r.fcntl.LOCK_EX|r.fcntl.LOCK_NB)
            with self.assertRaises(r.p.Pending):
                r.original_snapshot(self.root,self.c,self.run,self.audit,self.frozen,'B0',11)

    def test_no_recovery_without_global_selection_freeze(self):
        with patch.object(r.p,'load_audit',return_value=self.audit):
            with self.assertRaises(r.p.Pending):r.frozen_inputs(self.root,self.c,self.run)

    def test_report_stays_on_common_65_until_every_model_has_three_recovered_rows(self):
        for all_done in (False,True):
            models=[dict(method=g,seed=s) for g,s in r.MATRIX]
            policy=dict(models=models,frozen_selection_sha256='fixture')
            r.p.write_json(self.work/'policy.json',policy)
            for m in models:m['failures']=[]
            def original(root,model):return self.rows(self.names[:-1], 'native' if model['method']=='native' else 'eval')
            def recovered(root,c,work,model):
                if not all_done and model['method']=='native':return None
                return self.rows(['mspp16'],'native' if model['method']=='native' else 'eval',24*1024**3)
            with patch.object(r,'load_policy',return_value=(policy,self.audit,self.frozen)), \
                    patch.object(r,'original_rows',side_effect=original),patch.object(r,'recovered_rows',side_effect=recovered):
                snapshot=r.report(self.root,self.c,self.run,self.work)
            self.assertEqual(snapshot['complete66'],all_done)
            self.assertEqual(snapshot['instances'],66 if all_done else 65)
            actor=next(x for x in snapshot['combined_training_seeds'] if x['method']=='B0')
            self.assertEqual(actor['runs'],396 if all_done else 390)
            self.assertEqual(len(snapshot['per_training_seed']),7)

    def test_cpu_mismatch_blocks_recovery_before_solver(self):
        model=dict(method='B0',seed=11)
        with patch.object(r,'load_policy',return_value=({'models':[model]},self.audit,self.frozen)), \
                patch.object(r.p,'target_cpu',side_effect=ValueError('wrong CPU')),patch.object(r.p.server,'cached_jobs') as solve:
            with self.assertRaisesRegex(ValueError,'wrong CPU'):
                r.retry(self.root,self.c,self.run,self.work,'B0',11)
            solve.assert_not_called()

    def test_retry_runs_only_three_tasks_with_one_worker_and_no_gpu_requirement(self):
        self.original()
        model=r.original_snapshot(self.root,self.c,self.run,self.audit,self.frozen,'B0',11)
        policy=dict(models=[model,dict(method='R0',seed=11)])
        r.p.write_json(self.work/'policy.json',policy)
        info=dict(stack=self.audit['stack'],effective_cpu_count=8.,memory_limit_bytes=32*1024**3,gpu=None)
        calls=[]
        def cached(jobs,base,identity,workers,watchdog):
            calls.append((jobs,workers,watchdog))
            rows=self.rows(['mspp16'],allowance=24*1024**3)
            r.p.write_json(base/'selected_results.json',rows)
            return [(row,None) for row in rows]
        with patch.object(r,'load_policy',return_value=(policy,self.audit,self.frozen)), \
                patch.object(r.p,'target_cpu',return_value=info),patch.object(r.p,'validate_dataset'), \
                patch.object(r.p.server,'cached_jobs',side_effect=cached):
            r.retry(self.root,self.c,self.run,self.work,'B0',11)
        self.assertEqual(len(calls),1)
        jobs,workers,watchdog=calls[0]
        self.assertEqual((len(jobs),workers,watchdog),(3,1,300.))
        self.assertTrue((self.work/'tasks/B0-s11/complete.json').exists())
        self.assertFalse((self.run/'test/official66-best'/r.p.sha256(self.actor)[:16]/'selected_results.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
