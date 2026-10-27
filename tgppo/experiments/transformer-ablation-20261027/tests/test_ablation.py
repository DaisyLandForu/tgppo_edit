"""Architecture/gradient tests only; no MILP training jobs are launched."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from project.policy import Actor, Critic
from project.agent.agent import ActorPolicy
from project.runtime import (build_agent, config_with_protocol, seed_all,
                             save_checkpoint, load_agent, load_actor)
from project.utils.settings import state_dims
import server

ROOT = Path(__file__).resolve().parents[4]


def config(flag=True, **extra):
    return config_with_protocol(dict(use_transformer=flag, **extra))


def inputs():
    seed_all(19)
    return (torch.randn(3, 7, 25), torch.randn(3, 8), torch.randn(3, 53),
            torch.arange(7)[None, :] >= torch.tensor([1, 3, 7])[:, None])


class AblationTests(unittest.TestCase):
    def test_true_matches_original_parameters_and_forward_exactly(self):
        c, n, m, mask = inputs()
        for cls, name in ((Actor, 'actor'), (Critic, 'critic')):
            spec = importlib.util.spec_from_file_location('project.policy.original_'+name,
                ROOT / ('tgppo/repo/code/project/policy/'+name+'.py'))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            args = dict(state_dims, hidden_dim=256, num_heads=8, num_layers=5, dropout=.05)
            seed_all(11); original = getattr(module, cls.__name__)(**args).eval()
            seed_all(11); actual = cls(**args, use_transformer=True).eval()
            self.assertEqual(original.state_dict().keys(), actual.state_dict().keys())
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, actual.state_dict()[key]), key)
            with torch.no_grad():
                self.assertTrue(torch.equal(original(c,n,m,mask), actual(c,n,m,mask)))

    def test_retained_weights_and_rng_are_paired_no_removed_parameters(self):
        seed_all(11); a = build_agent(config(True)); rng = torch.get_rng_state()
        seed_all(11); b = build_agent(config(False))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for x, y in ((a.actor_network,b.actor_network),(a.critic_network,b.critic_network)):
            self.assertIsNone(y.transformer)
            self.assertFalse(any(isinstance(t, torch.nn.TransformerEncoder) for t in y.modules()))
            self.assertFalse(any(k.startswith('transformer.') for k in y.state_dict()))
            for key, value in y.state_dict().items():
                self.assertTrue(torch.equal(value, x.state_dict()[key]), key)
            self.assertLess(sum(p.numel() for p in y.parameters()), sum(p.numel() for p in x.parameters()))

    def test_shapes_logits_probabilities_and_bimatching_interface(self):
        c,n,m,mask = inputs()
        for flag in (True,False):
            a = build_agent(config(flag), training=False)
            a.actor_network.eval(); a.critic_network.eval()
            seen = {}
            def raw_logits(module, args, output): seen['logits'] = output.detach()
            def bm_input(module,args): seen['bm'] = args[1].detach()
            handles = [a.actor_network.output_layer.register_forward_hook(raw_logits),
                       a.actor_network.tree_refinement.register_forward_pre_hook(bm_input)]
            with torch.no_grad():
                p = a.actor_network(c,n,m,mask); v = a.critic_network(c,n,m,mask)
                self.assertEqual(tuple(p.shape),(3,7)); self.assertEqual(tuple(v.shape),(3,1))
                self.assertEqual(tuple(seen['logits'].shape),(3,7))
                self.assertEqual(tuple(seen['bm'].shape),(3,7,256))
                expected = torch.softmax(seen['logits'].masked_fill(mask,float('-inf')),dim=-1)
                torch.testing.assert_close(p, expected)
                torch.testing.assert_close(p.sum(-1), torch.ones(3))
                self.assertTrue(torch.isfinite(p).all() and torch.isfinite(v).all())
                self.assertTrue((p[mask]==0).all())
                if not flag:
                    actor = a.actor_network
                    tree = actor.tree_embedding(torch.cat([n,m],-1))
                    embedded = actor.global_embedding(torch.cat([
                        actor.var_embedding(c),tree[:,None,:].expand(-1,7,-1)],-1))
                    self.assertTrue(torch.equal(embedded,seen['bm']))
            for h in handles:h.remove()

    def test_padding_does_not_change_valid_outputs(self):
        c,n,m,mask = inputs()
        for flag in (True,False):
            a = build_agent(config(flag), training=False)
            for net in (a.actor_network,a.critic_network):
                net.eval()
                with torch.no_grad():
                    batched = net(c,n,m,mask)
                    altered = c.clone(); altered[mask] = 1e3
                    torch.testing.assert_close(batched,net(altered,n,m,mask),rtol=2e-5,atol=2e-6)
                    for i,length in enumerate([1,3,7]):
                        single = net(c[i:i+1,:length],n[i:i+1],m[i:i+1])
                        expected = batched[i:i+1,:length] if net is a.actor_network else batched[i:i+1]
                        torch.testing.assert_close(single,expected,rtol=2e-5,atol=2e-6)

    def test_candidate_permutation_and_original_index_correspondence(self):
        c,n,m,mask = inputs(); permutation = torch.tensor([4,0,6,2,1,5,3])
        a = build_agent(config(False), training=False)
        for net in (a.actor_network,a.critic_network):
            net.eval()
            with torch.no_grad():
                old = net(c,n,m,mask); new = net(c[:,permutation],n,m,mask[:,permutation])
                expected = old[:,permutation] if net is a.actor_network else old
                torch.testing.assert_close(new,expected,rtol=2e-5,atol=2e-6)
                if net is a.actor_network:
                    # Exclude ties from deterministic argmax correspondence.
                    for row in (1,2):
                        original_candidates = ['SCIP-variable-'+str(i) for i in range(7)]
                        reordered = [original_candidates[i] for i in permutation]
                        self.assertEqual(original_candidates[old[row].argmax()],reordered[new[row].argmax()])

    def test_actor_critic_finite_nonzero_gradients_independently(self):
        c,n,m,mask = inputs()
        a = build_agent(config(False))
        p = a.actor_network(c,n,m,mask)
        (-p[1,0].log()-p[2,1].log()).backward()
        self.assertTrue(all(p.grad is None for p in a.critic_network.parameters()))
        for name in ('var_embedding','tree_embedding','global_embedding','tree_refinement','output_layer'):
            gradients = [p.grad for p in getattr(a.actor_network,name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients),name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients),0,name)
        a.critic_network(c,n,m,mask).square().sum().backward()
        for name in ('var_embedding','tree_embedding','tree_refinement','aggregation','value_head'):
            gradients = [p.grad for p in getattr(a.critic_network,name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients),name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients),0,name)

    def test_empty_all_padding_nonfinite_rejected_by_policy(self):
        policy = ActorPolicy(build_agent(config(False),training=False).actor_network)
        s = dict(cands_state=np.zeros((3,25),np.float32),node_state=np.zeros(8,np.float32),mip_state=np.zeros(53,np.float32))
        with self.assertRaisesRegex(ValueError,'Empty'):policy.act(**s,padding_mask=[True]*3)
        with self.assertRaisesRegex(ValueError,'Empty'):policy.act(**dict(s,cands_state=np.zeros((0,25))))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):policy.act(**dict(s,cands_state=np.full((3,25),np.nan)))

    def test_config_bool_checkpoint_and_actor_only_roundtrip(self):
        with self.assertRaisesRegex(ValueError,'boolean'):config('false')
        for flag in (True,False):
            c = config(flag); seed_all(11); a = build_agent(c)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d)/'checkpoint.pt'; save_checkpoint(p,a,c)
                b, payload = load_agent(p)
                self.assertEqual(b.actor_network.use_transformer,flag)
                self.assertEqual(server.digest_actor(a.actor_network),server.digest_actor(b.actor_network))
                with patch('project.runtime.Critic',side_effect=AssertionError('Critic constructed')):
                    policy,_ = load_actor(p)
                self.assertEqual(policy.actor_network.use_transformer,flag)
                self.assertFalse(hasattr(policy,'critic_network'))

    def test_legacy_b0_export_loads_with_default_true(self):
        path = ROOT/'tgppo/runs/server6-20261022/training/B0-s11/actor_best.pt'
        policy,payload = load_actor(path)
        self.assertNotIn('use_transformer',payload['config'])
        self.assertTrue(policy.actor_network.use_transformer)


if __name__ == '__main__': unittest.main(verbosity=2)
