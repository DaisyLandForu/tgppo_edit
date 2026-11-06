import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import engineering
import experiment as ex
from project import runtime as rt


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base = rt.config_with_protocol({})
        g = torch.Generator().manual_seed(77)
        cls.inputs = (torch.randn(2,7,25,generator=g), torch.randn(2,8,generator=g),
            torch.randn(2,53,generator=g), torch.tensor([[False]*7,[False]*4+[True]*3]))

    def actor(self, mode):
        rt.seed_all(11)
        return rt.build_agent(dict(self.base,candidate_residual_mode=mode),training=False).actor_network.eval()

    def test_paired_initial_weights_rng_and_zero_policy_at_production_size(self):
        for seed in (11,22):
            with self.subTest(training_seed=seed):
                self._check_paired_initialization(seed)

    def _check_paired_initialization(self,seed):
        models, rngs = {}, {}
        for mode in ('none','score','candidate'):
            rt.seed_all(seed)
            a = rt.build_agent(dict(self.base,candidate_residual_mode=mode), training=False)
            rngs[mode] = torch.get_rng_state().clone()
            models[mode] = a
        reference = models['none']
        self.assertEqual(sum(p.numel() for p in reference.actor_network.parameters()), 4738895)
        with torch.no_grad():
            expected = reference.actor_network.eval()(*self.inputs)
            for mode in ('score','candidate'):
                a = models[mode]
                self.assertTrue(torch.equal(rngs['none'],rngs[mode]))
                for k,v in reference.actor_network.state_dict().items():
                    self.assertTrue(torch.equal(v,a.actor_network.state_dict()[k]),k)
                self.assertTrue(engineering.same_state(reference.critic_network.state_dict(),a.critic_network.state_dict()))
                self.assertEqual(sum(p.numel() for p in a.actor_network.parameters()),4738895+256)
                self.assertTrue(torch.equal(expected,a.actor_network.eval()(*self.inputs)))
                self.assertTrue((a.actor_network.residual_weight == 0).all())

    def test_nonzero_residual_matches_declared_formula(self):
        for mode in ('score','candidate'):
            actor = self.actor(mode)
            with torch.no_grad():
                actor.residual_weight.copy_(torch.linspace(-.03,.03,256))
            capture={}
            handles=[]
            for key, module in [('e',actor.var_embedding),('m',actor.tree_refinement),('s',actor.output_layer)]:
                handles.append(module.register_forward_hook(lambda m,i,o,k=key:capture.__setitem__(k,o.detach().clone())))
            with torch.no_grad():
                probs=actor(*self.inputs)
                features=capture['m']+(capture['e'] if mode=='candidate' else 0)
                logits=capture['s']+features@actor.residual_weight
                expected=torch.softmax(logits.masked_fill(self.inputs[-1],float('-inf')),dim=-1)
                self.assertTrue(torch.equal(probs,expected))
            for h in handles:h.remove()

    def test_nonuniform_policy_permutation_and_padding(self):
        actor=self.actor('candidate')
        with torch.no_grad():actor.residual_weight.copy_(torch.linspace(-.1,.1,256))
        c,n,m,mask=self.inputs
        permutation=torch.tensor([4,1,6,0,5,3,2])
        with torch.no_grad():
            reference=actor(c,n,m,mask)
            reordered=actor(c[:,permutation],n,m,mask[:,permutation])[:,torch.argsort(permutation)]
            self.assertTrue(torch.allclose(reference,reordered,atol=1e-6,rtol=0))
            padded=actor(torch.cat([c,torch.ones(2,3,25)*99],1),n,m,
                torch.cat([mask,torch.ones(2,3,dtype=torch.bool)],1))
            self.assertTrue(torch.allclose(reference,padded[:,:7],atol=1e-6,rtol=0))
            self.assertTrue((padded[:,7:]==0).all())
            self.assertGreater(float((reference[0]-1/7).abs().sum()),.01)
            one=actor(c[:1,:1],n[:1],m[:1])
            self.assertEqual(float(one),1.)

    def test_full_update_resume_actor_export_all_modes(self):
        with tempfile.TemporaryDirectory() as d:
            for mode in ('none','score','candidate'):
                config=dict(self.base,candidate_residual_mode=mode)
                result=engineering.bridge(config,'cpu',Path(d)/mode)
                self.assertTrue(result['passed'])
                if mode!='none':
                    restored,_=rt.load_agent(Path(d)/mode/'after.pt',device='cpu',restore_random_state=False)
                    self.assertGreater(float(restored.actor_network.residual_weight.norm()),0.)

    def test_legacy_actor_config_and_bad_modes(self):
        rt.seed_all(11)
        actor=rt.build_agent(self.base,training=False).actor_network
        self.assertFalse(hasattr(actor,'residual_weight'))
        with self.assertRaises(ValueError):
            rt.build_agent(dict(self.base,candidate_residual_mode='typo'),training=False)


class ProtocolTests(unittest.TestCase):
    def test_only_declared_variables_and_schedule_prefix(self):
        ex.check_protocols()
        for seed in (11,22):
            c,_=ex.protocol('b0')
            short=ex.server.schedule(seed,c)
            old=ex.server.schedule(seed,dict(c,max_episodes=500))
            self.assertEqual(short,old[:100])
            self.assertEqual(set(short[:20]),set(c['train']))
            self.assertEqual(len(short),100)

    def test_common_endpoint_no_mixed_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            def protocol(v):return {},Path(temp)/v
            states={v+'-s'+str(s):{'status':'training_complete'} for v in ex.VARIANTS for s in ex.SEEDS}
            with patch.object(ex,'protocol',protocol):
                self.assertEqual(ex.choose_endpoint(states),0)
                for v in ex.VARIANTS:
                    for s in ex.SEEDS:
                        out=ex.server.group_dir(protocol(v)[1],'B0',s)
                        out.mkdir(parents=True)
                        (out/'episode-0020.pt').touch()
                        (out/'episode-0100.pt').touch()
                self.assertEqual(ex.choose_endpoint(states),100)
                states['candidate-s22']['status']='budget_exhausted'
                self.assertEqual(ex.choose_endpoint(states),20)
                states['candidate-s22']['status']='running'
                self.assertIsNone(ex.choose_endpoint(states))

    def test_native_adoption_rejects_changed_machine(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'baseline.json'
            source.write_text(json.dumps({'stack':{'cpu':'old'}}))
            self.assertFalse(ex.adopt_native(source,{},Path(temp)/'target',{'cpu':'new'}))
            self.assertFalse((Path(temp)/'target/normalization/baselines.json').exists())


if __name__=='__main__':unittest.main()
