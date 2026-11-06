import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import launch
from local_preflight import configuration
from pretrain import fit,signature,load_initialization
from project import runtime as rt
from project.policy.light_actor import LightActor
from expert_data import OUTPUT,sha256


class Readiness(unittest.TestCase):
    def test_dead_attempt_cannot_reset_budget(self):
        self.assertEqual(launch.charged_seconds([{'allocated_seconds':80},
            {'allocated_seconds':70,'elapsed_seconds':9}]),89)

    def test_budget_timeout(self):
        with self.assertRaises(launch.BudgetExhausted):
            with launch.deadline(.03):
                import time
                time.sleep(.08)

    def test_cpu_model_can_change_but_software_cannot(self):
        a={'cpu_model':'one','torch':'1.13'};b={'cpu_model':'two','torch':'1.13'}
        self.assertEqual(launch.software(a),launch.software(b))
        b['torch']='2.0';self.assertNotEqual(launch.software(a),launch.software(b))

    def test_mismatching_native_stack_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'native.json';p.write_text(json.dumps({'stack':{'cpu_model':'one'}}))
            self.assertFalse(launch.adopt_native(p,{},Path(td)/'target',{'cpu_model':'two'}))
            self.assertFalse((Path(td)/'target/normalization/baselines.json').exists())

    def test_full_native_identity_adoption_without_solver(self):
        source=launch.ROOT/'tgppo/runs/server6-20261022/normalization/baselines.json'
        fixture=json.loads(source.read_text());cfg,_=launch.protocol('b0')
        # Validate cache-adoption semantics using the fixture's measured stack;
        # this does not assert that this local host has the same CPU.
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(launch.adopt_native(source,cfg,Path(td),fixture['stack']))
            adopted=json.loads((Path(td)/'normalization/baselines.json').read_text())
            self.assertEqual(len(adopted['baselines']),75)
            self.assertEqual(adopted['protocol_sha256'],launch.fingerprint(cfg))

    def test_bc_selection_recovery_and_fresh_ppo_optimizer(self):
        cfg=configuration('mlp');cfg.update(bc_steps=8,bc_selection_milestones=[0,4,8])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rt.seed_all(11);full=rt.build_agent(cfg,training=True,device='cpu')
            initial_critic=copy.deepcopy(full.critic_network.state_dict())
            expected=fit(full.actor_network,cfg,root/'full')
            rt.seed_all(11);interrupted=rt.build_agent(cfg,training=True,device='cpu')
            def stop(row):
                if row['step']==4:raise InterruptedError('Test only: interrupt after atomic BC commit')
            with self.assertRaises(InterruptedError):fit(interrupted.actor_network,cfg,root/'resume',heartbeat=stop)
            rt.seed_all(11);recovered=rt.build_agent(cfg,training=True,device='cpu')
            result=fit(recovered.actor_network,cfg,root/'resume',resume=True)
            self.assertEqual(result['selected_step'],expected['selected_step'])
            self.assertEqual(result['final'],expected['final'])
            for name,value in full.actor_network.state_dict().items():
                self.assertTrue(torch.equal(value,recovered.actor_network.state_dict()[name]),name)
            self.assertFalse(recovered.actor_optimizer.state)
            load_initialization(recovered,cfg,root/'resume')
            for name,value in initial_critic.items():self.assertTrue(torch.equal(value,recovered.critic_network.state_dict()[name]),name)
            changed=dict(cfg,expert_dataset_sha256='wrong')
            with self.assertRaises(ValueError):load_initialization(recovered,changed,root/'resume')

    def test_three_original_source_network_files_are_untouched(self):
        original=launch.ROOT/'tgppo/experiments/server-20261022/code/project'
        current=launch.HERE/'code/project'
        for relative in ['policy/actor.py','policy/critic.py','agent/agent.py','reward/reward.py']:
            self.assertEqual(sha256(original/relative),sha256(current/relative),relative)

    def test_large_candidate_set_is_not_truncated(self):
        torch.set_num_threads(1);torch.manual_seed(2)
        x=torch.randn(1,695,25);node=torch.randn(1,8);mip=torch.randn(1,53)
        perm=torch.randperm(695)
        for kind in ('mlp','set','isab'):
            model=LightActor(kind).eval()
            with torch.no_grad():p=model(x,node,mip);q=model(x[:,perm],node,mip)[:,perm.argsort()]
            self.assertEqual(p.shape,(1,695))
            self.assertTrue(torch.isfinite(p).all())
            self.assertTrue(torch.allclose(p,q,atol=2e-6,rtol=2e-5))

    def test_existing_export_is_preserved_and_corruption_is_rejected(self):
        cfg=configuration('mlp');rt.seed_all(11);agent=rt.build_agent(cfg,device='cpu')
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'checkpoint.pt';actor=Path(td)/'actor.pt'
            rt.save_checkpoint(cp,agent,cfg,{'scope':'test only'})
            launch.ensure_export(cp,actor);before=sha256(actor)
            launch.ensure_export(cp,actor);self.assertEqual(sha256(actor),before)
            payload=torch.load(actor,map_location='cpu')
            key=next(iter(payload['actor']));payload['actor'][key]+=1.
            torch.save(payload,actor)
            with self.assertRaises(ValueError):launch.ensure_export(cp,actor)

    def test_bc_and_structure_pairing_are_separate_and_noactor_is_flagged(self):
        from results import report
        import csv
        models=[]
        pdis={'b0':100.,'b0_bc':95.,'mlp':90.,'mlp_bc':80.}
        for variant,pdi in pdis.items():
            row=dict(instance_name='case',solver_seed=0,pdi_horizon_midpoint=pdi,nnodes=10,
                proof_completed=True,scip_solve_time=1.,optimize_wall_seconds=1.,branch_count=0,
                callback_seconds={'actor':0.})
            models.append(dict(variant=variant,seed=11,rows=[row],updates=1,progress=dict(next_episode=100,trained_episodes=100)))
        with tempfile.TemporaryDirectory() as td:
            report(td,models,[models[0]['rows'][0]],100,[11],{'stack':{'cpu_model':'test'}})
            with (Path(td)/'paired.csv').open() as f:rows=list(csv.DictReader(f))
            actual={r['effect']:float(r['pdi_delta']) for r in rows if r['variant']=='mlp_bc'}
            self.assertEqual(actual,{'structure_at_same_pretraining':-15.,'bc_within_structure':-10.})
            self.assertTrue(all(r['both_actor_active']=='False' for r in rows))


if __name__=='__main__':unittest.main()
