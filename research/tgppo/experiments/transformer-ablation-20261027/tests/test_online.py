import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from project.runtime import (seed_all,config_with_protocol,build_agent,save_checkpoint,
                             load_agent,load_actor,rng_state)


def config(group='R0'):
    return config_with_protocol(dict(hidden_dim=32,num_layers=1,num_heads=4,dropout=0.,
        batch_size=4,n_epochs=3,online_group=group,target_mean=-13.166822661165169,
        target_std=8.840326545776836,extra_initialization_seed=1011))


def populate(agent):
    agent.memory.begin_episode('online-unit',agent.policy_version)
    for i in range(8):
        state=dict(cands_state=np.full((2+i%3,25),.1*(i+1),np.float32),
            mip_state=np.full(53,.1,np.float32),node_state=np.zeros(8,np.float32),
            remaining_time=np.float32(1-i/8),frontier=np.r_[np.full(64,1/64),np.ones(5)].astype(np.float32))
        action,value,lp=agent.choose_action(**state)
        agent.remember(**state,action=action,value=value,log_prob=lp,reward=-float(i+1),done=i==7)
    agent.seal_episode(task_terminal=True)


class OnlineTests(unittest.TestCase):
    def test_pairing_public_weights_rng_and_extra_shapes(self):
        agents=[];rngs=[]
        for g in ('R0','Rprod','Rjoint'):
            seed_all(11);agents.append(build_agent(config(g)));rngs.append(rng_state())
        common=agents[0].critic_network.state_dict()
        for a in agents[1:]:
            for k,v in agents[0].actor_network.state_dict().items():
                torch.testing.assert_close(v,a.actor_network.state_dict()[k],rtol=0,atol=0)
            for k,v in common.items():
                torch.testing.assert_close(v,a.critic_network.state_dict()[k],rtol=0,atol=0)
        for r in rngs[1:]:
            self.assertTrue(torch.equal(rngs[0]['torch'],r['torch']))
        for k,v in agents[1].critic_network.state_dict().items():
            torch.testing.assert_close(v,agents[2].critic_network.state_dict()[k],rtol=0,atol=0)

    def test_context_roundtrip_minibatch_and_reject_corruption(self):
        seed_all(11);a=build_agent(config('Rjoint'));populate(a)
        payload=a.memory.export_dict();self.assertEqual(payload['schema_version'],3)
        b=build_agent(config('Rjoint'));b.memory.import_dict(payload)
        for batch in b.memory.get_batch_generator():
            for j,index in enumerate(batch['indices']):
                row=a.memory.rows()[index]
                np.testing.assert_array_equal(batch['frontier'][j],row['frontier'])
                self.assertEqual(float(batch['remaining_time'][j]),float(row['remaining_time']))
        for bad in ('missing','nan','shape'):
            broken=copy.deepcopy(payload)
            row=broken['episodes'][0]['transitions'][0]
            if bad=='missing':del row['frontier']
            if bad=='nan':row['frontier'][0]=float('nan')
            if bad=='shape':row['frontier']=np.zeros(68)
            with self.assertRaises((ValueError,KeyError)):build_agent(config('Rjoint')).memory.import_dict(broken)

    def test_raw_value_gae_and_only_loss_standardized(self):
        seed_all(11);a=build_agent(config());populate(a)
        rows=a.memory.rows();mu=config()['target_mean'];std=config()['target_std']
        for r in rows:self.assertAlmostEqual(r['value'],mu,places=5)
        adv,ret=a.compute_targets()
        gae=0.;expected=[]
        for i in reversed(range(len(rows))):
            r=rows[i];nextv=0. if r['done'] else rows[i+1]['value']
            delta=r['reward']+.97*(1-r['done'])*nextv-r['value']
            gae=delta+.97*.92*(1-r['done'])*gae;expected.append(gae+r['value'])
        np.testing.assert_allclose(ret.numpy(),expected[::-1],rtol=1e-5)
        x=torch.tensor([-20.,-3.],requires_grad=True);y=torch.tensor([-5.,8.])
        loss=a._value_loss(x,y);torch.testing.assert_close(loss,((x-y)/std).square().mean())
        self.assertAlmostEqual(ret[-1].item(),rows[-1]['reward'],places=5)

    def test_full_update_restore_rng_optimizer_and_actor_only(self):
        seed_all(11);a=build_agent(config('Rjoint'));populate(a)
        before={k:v.detach().clone() for k,v in a.critic_network.named_parameters()}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt';save_checkpoint(p,a,config('Rjoint'))
            a,_=load_agent(p);m=a.learn(microbatch_size=2)
            self.assertTrue(m['updated']);self.assertLess(m['initial_ratio_max_error'],1e-4)
            for name in ('var_embedding','transformer','tree_embedding','tree_refinement','aggregation'):
                self.assertTrue(any(not torch.equal(v,before[k]) for k,v in a.critic_network.named_parameters() if k.startswith(name)),name)
            self.assertGreater(a.critic_network.extra_projection.weight.norm().item(),0.)
            save_checkpoint(p,a,config('Rjoint'));b,payload=load_agent(p)
            self.assertEqual(b.update_counter,1);self.assertEqual(b.policy_version,1)
            self.assertEqual(len(b.actor_optimizer.state),len(a.actor_optimizer.state))
            self.assertEqual(len(b.critic_optimizer.state),len(a.critic_optimizer.state))
            with patch('project.runtime.build_agent',side_effect=AssertionError('Critic forbidden')):
                policy,_=load_actor(p)
            self.assertFalse(hasattr(policy,'critic_network'))
            row=dict(cands_state=np.ones((2,25)),mip_state=np.zeros(53),node_state=np.zeros(8))
            state=torch.get_rng_state();aa=policy.act(**row);torch.set_rng_state(state);bb=b.act(**row)
            self.assertEqual(aa,bb)

    def test_online_microbatch_equivalence(self):
        seed_all(11);a=build_agent(config('Rprod'));populate(a)
        for opt in (a.actor_optimizer,a.critic_optimizer):
            for g in opt.param_groups:g['lr']=0.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.pt';save_checkpoint(p,a,config('Rprod'))
            full,_=load_agent(p);mf=full.learn()
            small,_=load_agent(p);ms=small.learn(microbatch_size=1)
            for net in ('actor_network','critic_network'):
                for x,y in zip(getattr(full,net).parameters(),getattr(small,net).parameters()):
                    torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=3e-4)
            self.assertAlmostEqual(mf['critic_loss'],ms['critic_loss'],places=5)

if __name__=='__main__':unittest.main(verbosity=2)
