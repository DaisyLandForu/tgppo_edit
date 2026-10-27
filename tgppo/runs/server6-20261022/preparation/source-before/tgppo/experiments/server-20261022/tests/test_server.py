import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import server
from project.runtime import config_with_protocol,build_agent,seed_all,save_checkpoint,load_agent,load_actor
from project.agent.agent import Agent
from project.utils.protocol import write_json,fingerprint

ROOT=Path(__file__).resolve().parents[4]


def tiny(group):
    p=config_with_protocol(dict(hidden_dim=32,num_layers=1,num_heads=4,dropout=0.,batch_size=4,n_epochs=1,learner_device='cpu'))
    if group!='B0':p.update(online_group=group,target_mean=-13.,target_std=8.,extra_initialization_seed=1011)
    return p


def populate(a):
    a.memory.begin_episode('synthetic',0)
    for i in range(8):
        s=dict(cands_state=np.full((2+i%3,25),.1*(i+1),np.float32),mip_state=np.full(53,.2,np.float32),node_state=np.zeros(8,np.float32))
        if getattr(a,'online_context',False):s.update(remaining_time=np.float32(.8),frontier=np.r_[np.full(64,1/64),np.ones(5)].astype(np.float32))
        action,value,lp=a.choose_action(**s)
        a.remember(**s,action=action,value=value,log_prob=lp,reward=-float(i+1),done=i==7)
    a.seal_episode(True)


class ServerTests(unittest.TestCase):
    def test_three_groups_public_actor_pairing_and_cpu_worker(self):
        actors=[]
        for g in ('B0','R0','Rjoint'):
            seed_all(11);a=build_agent(tiny(g));actors.append(server.digest_actor(a.actor_network))
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'initial.pt';save_checkpoint(p,a,tiny(g))
                worker,_=load_agent(p,training=False,restore_random_state=False)
                self.assertEqual(worker.device,'cpu');self.assertIsNone(worker.actor_optimizer)
                self.assertEqual(getattr(worker,'online_context',False),g!='B0')
        self.assertEqual(len(set(actors)),1)

    def test_complete_update_restore_and_actor_export_all_groups(self):
        for g in ('B0','R0','Rjoint'):
            seed_all(11);a=build_agent(tiny(g));populate(a)
            self.assertTrue(a.learn(microbatch_size=2)['updated'])
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'full.pt';export=Path(d)/'actor.pt'
                save_checkpoint(p,a,tiny(g),dict(next_episode=100,phase='ready'))
                b,payload=load_agent(p);self.assertEqual(b.policy_version,1)
                self.assertTrue(b.actor_optimizer.state);self.assertTrue(b.critic_optimizer.state)
                server.export_actor(p,export)
                with patch('project.runtime.build_agent',side_effect=AssertionError('No Critic construction')):
                    deployed,_=load_actor(export)
                self.assertFalse(hasattr(deployed,'critic_network'))
                self.assertEqual(server.digest_actor(a.actor_network),server.digest_actor(deployed.actor_network))
                with self.assertRaises(ValueError):load_agent(export)

    def test_dev_selection_coverage_ties_and_failure(self):
        def row(name,seed,pdi):return dict(instance_name=name,solver_seed=seed,valid=True,
            pdi_horizon_midpoint=pdi,pdi_horizon_lower=pdi,pdi_horizon_upper=pdi)
        values={100:[row('a',0,3),row('a',1,5),row('b',0,10),row('b',1,2)],
                200:[row('a',0,6),row('a',1,2),row('b',0,2),row('b',1,10)]}
        best,scores=server.selection(values,['a','b'],[0,1]);self.assertEqual(best['episode'],100)
        self.assertEqual(best['mean_pdi'],5.)
        broken=copy.deepcopy(values);broken[200][0]['valid']=False
        with self.assertRaises(ValueError):server.selection(broken,['a','b'],[0,1])
        broken=copy.deepcopy(values);broken[200].pop()
        with self.assertRaises(ValueError):server.selection(broken,['a','b'],[0,1])

    def test_real_collection_interrupted_update_resumes_without_lost_or_duplicate_tasks(self):
        c=server.plan(ROOT,ROOT/'migration/2026-10-22/server-run/protocol.json')
        c.update(engineering_mode=True,groups=['B0','R0','Rjoint'],training_seeds=[11],
            train=['lseu','stein27','misc03','pp08aCUTS'],development=['vpm2'],solver_seeds=[0],
            max_episodes=4,milestones=[4],workers=2,time_limit=2.,watchdog=30.,microbatch_size=4,
            model_overrides=dict(hidden_dim=32,num_layers=1,num_heads=4,dropout=0.,batch_size=16,n_epochs=1))
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);server.baseline(ROOT,c,run)
            normal=Agent.learn
            def fail_after_first_step(agent,progress_callback=None,**kwargs):
                def crash(p):
                    if progress_callback:progress_callback(p)
                    raise server.Interrupted('Injected post-optimizer interruption')
                return normal(agent,progress_callback=crash,**kwargs)
            for g in c['groups']:
                with patch.object(Agent,'learn',fail_after_first_step):
                    with self.assertRaises(server.Interrupted):server.train(ROOT,c,run,g,11,False,'cpu')
                out=server.group_dir(run,g,11);a,p=load_agent(out/'latest.pt')
                self.assertEqual(p['progress']['phase'],'collected');self.assertEqual(a.policy_version,0)
                self.assertEqual(p['progress']['next_episode'],4)
                self.assertGreater(len(a.memory),16)
                collected={f: f.stat().st_mtime_ns for f in out.glob('collection/**/*.pkl')}
                a.learn(microbatch_size=4);expected_actor=server.digest_actor(a.actor_network);expected_critic=server.digest_actor(a.critic_network)
                state=server.train(ROOT,c,run,g,11,True,'cpu')
                b,p=load_agent(out/'latest.pt')
                self.assertEqual(expected_actor,server.digest_actor(b.actor_network))
                self.assertEqual(expected_critic,server.digest_actor(b.critic_network))
                self.assertEqual(collected,{f: f.stat().st_mtime_ns for f in out.glob('collection/**/*.pkl')})
                self.assertEqual(state['trained_episodes'],4);self.assertEqual(b.policy_version,1)
                self.assertTrue((out/'episode-0004.pt').exists())
                self.assertEqual(json.loads((out/'summary.json').read_text())['episodes_seen'],4)
                server.select(ROOT,c,run,g,11)
                self.assertEqual(server.read(out/'selection.json')['best']['episode'],4)
                self.assertTrue((out/'actor_best.pt').exists());self.assertTrue((out/'actor_final.pt').exists())
                for p in out.glob('development/**/dev*.json'):
                    r=server.read(p);self.assertEqual(r['callback_seconds']['critic'],0.)
                    self.assertNotIn('search_extraction_wall_seconds',r)
                if g=='Rjoint':
                    server.evaluate(ROOT,c,run,out/'actor_best.pt',ROOT/'tgppo/datasets/train_instances',
                        ['neos-476283'],'heldout-test',11)
                    server.evaluate(ROOT,c,run,None,ROOT/'tgppo/datasets/train_instances',
                        ['neos-476283'],'heldout-native',11,native=True)
                    self.assertEqual(len(list((run/'test/heldout-test').glob('*-summary.json'))),1)
                    with self.assertRaises(ValueError):
                        server.evaluate(ROOT,c,run,out/'actor_best.pt',ROOT/'tgppo/datasets/train_instances',
                            ['vpm2'],'leak-forbidden',11)
                for p in out.glob('collection/**/e*.json'):
                    self.assertFalse(json.loads(p.read_text())['cuda_initialized'])

if __name__=='__main__':unittest.main(verbosity=2)
