"""Tests for causal diagnostics, including a counterexample to naive LN deletion."""
import unittest
import torch
from torch import nn
from diagnose_cause import FixedVariance,intervene
from project.policy.modules import TreeGateBranchingNet
from uniform_control import UniformActor


class CauseTests(unittest.TestCase):
    def test_ln2_formula_and_derivative(self):
        x=torch.tensor([[[-.4,-.8],[-.3,-.8],[.2,-.6]]],dtype=torch.float64,requires_grad=True)
        norm=nn.LayerNorm(2).double()
        with torch.no_grad():norm.weight.copy_(torch.tensor([.7,1.3]));norm.bias.copy_(torch.tensor([.1,-.2]))
        d=x[...,0]-x[...,1];eps=norm.eps
        a=d/(d.square()+4*eps).sqrt()
        expected=torch.stack((a,-a),-1)*norm.weight+norm.bias
        self.assertTrue(torch.allclose(norm(x),expected,atol=1e-14,rtol=1e-14))
        grad,=torch.autograd.grad(norm(x)[...,0].sum(),x)
        derivative=4*eps/(d.square()+4*eps).pow(1.5)*norm.weight[0]
        self.assertTrue(torch.allclose(grad,torch.stack((derivative,-derivative),-1),atol=1e-12,rtol=1e-10))

    def test_fixed_variance_preserves_reference_and_restores_difference(self):
        x=torch.tensor([[[-.4,-.8],[-.3,-.8]]],dtype=torch.float64)
        norm=nn.LayerNorm(2).double();fixed=FixedVariance(norm);fixed.anchor(x)
        reference=x.mean(1,keepdim=True)
        self.assertTrue(torch.allclose(fixed(reference),norm(reference),atol=1e-14,rtol=1e-14))
        original_span=(norm(x)[0,0]-norm(x)[0,1]).abs().max()
        restored_span=(fixed(x)[0,0]-fixed(x)[0,1]).abs().max()
        self.assertGreater(restored_span,1000*original_span)
        self.assertEqual(sum(p.numel() for p in fixed.parameters()),0)

    def test_naive_identity_can_kill_relu(self):
        # Real seed-11 counterfactuals have this sign pattern: deleting LN does
        # not preserve the trained activation interval.
        x=torch.tensor([[[-.4,-.8],[-.3,-.8]]],dtype=torch.float64)
        self.assertEqual(int((torch.relu(x)>0).sum()),0)
        self.assertGreater(int((torch.relu(nn.LayerNorm(2).double()(x))>0).sum()),0)

    def test_intervention_does_not_mutate_original(self):
        torch.manual_seed(11)
        head=TreeGateBranchingNet(256,256,2,infimum=1,norm='layer',hidden_size=256)
        original={k:v.clone() for k,v in head.state_dict().items()}
        other=intervene(head,(2,),fixed_variance=True)
        self.assertIsInstance(head.BranchingNet[6][1],nn.LayerNorm)
        self.assertIsInstance(other.BranchingNet[6][1],FixedVariance)
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in original.items()))
        self.assertTrue(torch.equal(other.BranchingNet[6][1].weight,original['BranchingNet.6.1.weight'].double()))

    def test_fixed_variance_gradient_matches_finite_difference(self):
        x=torch.tensor([[[-.4,-.8],[-.3,-.8]]],dtype=torch.float64,requires_grad=True)
        fixed=FixedVariance(nn.LayerNorm(2).double());fixed.anchor(x.detach())
        grad,=torch.autograd.grad(fixed(x)[...,0].sum(),x)
        direction=torch.tensor([[[.3,.2],[-.2,.4]]],dtype=torch.float64)
        step=1e-5
        fd=(fixed(x.detach()+step*direction)[...,0].sum()-fixed(x.detach()-step*direction)[...,0].sum())/(2*step)
        self.assertAlmostEqual(float(fd),float((grad*direction).sum()),places=9)

    def test_uniform_control_respects_padding(self):
        c=torch.randn(2,4,25);mask=torch.tensor([[False,False,True,True],[False,True,True,True]])
        p=UniformActor()(c,torch.randn(2,8),torch.randn(2,53),mask)
        self.assertTrue(torch.equal(p[mask],torch.zeros_like(p[mask])))
        self.assertTrue(torch.equal(p.sum(-1),torch.ones(2)))
        self.assertTrue(torch.equal(p[0,:2],torch.tensor([.5,.5])))


if __name__=='__main__':unittest.main()
