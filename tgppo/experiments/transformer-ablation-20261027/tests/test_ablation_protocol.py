"""Protocol/summary checks; no solver collection or training."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import ablation
import server
from project.utils.protocol import write_json


class ProtocolTests(unittest.TestCase):
    def test_only_architecture_flag_and_output_labels_differ(self):
        ablation.check_pair()
        for variant in ablation.VARIANTS:
            c,_=ablation.protocol(variant)
            p=server.params(ablation.ROOT,c,'B0',11,'cpu')
            self.assertNotIn('online_group',p)
            self.assertEqual(p['hidden_dim'],256);self.assertEqual(p['batch_size'],256)
            self.assertEqual(p['use_transformer'],variant=='transformer')
        left,_=ablation.protocol('transformer');right,_=ablation.protocol('bimatching')
        for seed in (11,22,33):self.assertEqual(server.schedule(seed,left),server.schedule(seed,right))

    def test_confounded_config_is_rejected(self):
        original=ablation.protocol
        def changed(variant,smoke=False):
            c,r=original(variant,smoke)
            if variant=='bimatching':c['time_limit']=42.
            return c,r
        with patch.object(ablation,'protocol',changed):
            with self.assertRaisesRegex(ValueError,'Unexpected experimental'):ablation.check_pair()

    def test_timeouts_unknown_gaps_and_all_solver_seed_proof_count_are_preserved(self):
        rows=[]
        for name in ('a','b'):
            for seed in (0,1,2):
                solved=name=='a' or seed==0
                rows.append(dict(instance_name=name,solver_seed=seed,proof_completed=solved,
                    scip_solve_time=5. if solved else 120.1,status='infeasible' if solved else 'timelimit',
                    inference_seconds=[.001,.003],end_to_end_wall_seconds=6. if solved else 121.,
                    nnodes=100,fair_nodes=100,pdi_horizon_midpoint=10,pdi_horizon_lower=9,pdi_horizon_upper=11,
                    gap=0. if solved else 1e20,gap_finite=solved,callback_seconds=dict(features=.1,callback=.3)))
        result=ablation.aggregate(rows,120.)
        self.assertEqual(result['solved_instances'],1)
        self.assertEqual(result['solved'],4);self.assertEqual(result['timeouts'],2)
        self.assertEqual(result['gap_unknown_count'],2)
        self.assertGreater(result['solve_time'],40.)
        self.assertGreater(result['final_gap'],1e19)
        self.assertEqual(result['inference_mean_ms'],2.)

    def test_missing_freeze_and_changed_frozen_model_are_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ablation,'ROOT',Path(d)):
            current=dict(models=[dict(actor_sha256='original')],stack={})
            with patch.object(ablation,'selection_manifest',return_value=current):
                with self.assertRaises(FileNotFoundError):ablation.freeze(verify=True)
                ablation.freeze();ablation.freeze(verify=True)
            with patch.object(ablation,'selection_manifest',return_value=dict(current,models=[])):
                with self.assertRaisesRegex(ValueError,'modified'):ablation.freeze(verify=True)

    def test_mspp16_resource_repair_is_serial_and_keeps_solver_budget(self):
        c,_=ablation.protocol('bimatching')
        c.update(train=[],development=[],instance_hashes={})
        calls=[]
        def cached(jobs,base,identity,workers,watchdog):
            calls.append((jobs,workers,watchdog))
            return [(dict(job_id=j['job_id'],instance_name=j['instance_name'],solver_seed=j['solver_seed'],
                valid=True,pdi_horizon_midpoint=1.,proof_completed=False,scip_solve_time=120.,nnodes=1,
                callback_seconds=dict(critic=0.)),None) for j in jobs]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);actor=path/'actor.pt';actor.write_bytes(b'placeholder, loading is mocked')
            for name in ('lseu','mspp16'):(path/(name+'.mps.gz')).write_bytes(name.encode())
            with patch.object(server,'load_actor',return_value=(object(),dict(config=dict(server_training_seed=11,server_group='B0')))),\
                 patch.object(server,'stack',return_value={'cpu':'same'}),\
                 patch.object(server,'resource_limits',return_value=dict(memory_limit_bytes=32*1024**3)),\
                 patch.object(server,'cached_jobs',side_effect=cached):
                server.evaluate(ablation.ROOT,c,path/'run',actor,path,['lseu','mspp16'],'resources',11)
            self.assertEqual([call[1] for call in calls],[4,1])
            self.assertEqual([call[2] for call in calls],[180.,300.])
            for job in calls[1][0]:
                self.assertEqual(job['address_space_limit_bytes'],24*1024**3)
                self.assertEqual(job['limits']['time_limit'],120.)
            self.assertEqual(len(list((path/'run/test/resources').glob('*/selected_results.json'))),1)


if __name__=='__main__':unittest.main(verbosity=2)
