import unittest
import numpy as np
import torch
from project.policy.critic import Critic
from project.policy.calibrated_critic import (CalibratedCritic,balanced_return_statistics,make_calibration)


class CalibrationChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.base=Critic(25,8,53,32,4,1,.05).eval()

    def args(self):
        return (torch.randn(6,32),torch.randn(6,32),torch.randn(6,61),torch.rand(6,1),torch.rand(6,69))

    def test_statistics_are_instance_balanced(self):
        s=balanced_return_statistics([0,2,10],['a','a','b'])
        self.assertAlmostEqual(s['mean'],5.5)
        self.assertAlmostEqual(s['variance'],20.75)
        np.testing.assert_allclose(s['weights'],[.25,.25,.5])
        with self.assertRaises(ValueError):balanced_return_statistics([],[])

    def test_mse_all_groups_start_at_same_raw_train_mean(self):
        args=self.args()
        for group in ('R0','Rcap','Rprod','Rjoint'):
            model=CalibratedCritic(self.base,group,'mse',mean=-13.,std=8.).eval()
            prediction=model.forward_cached(*args)
            torch.testing.assert_close(prediction,torch.full((6,),-13.))
            logits=model.logits_cached(*args)
            target=torch.tensor([-20.,-15.,-13.,-3.,0.,5.])
            torch.testing.assert_close(model.training_loss(logits,target),((prediction-target)/8.).square().mean())

    def test_raw_decoder_roundtrip_and_negative_values(self):
        model=CalibratedCritic(self.base,mean=-13.,std=8.)
        z=torch.tensor([[-2.],[0.],[2.]])
        torch.testing.assert_close(model.decode(z),torch.tensor([-29.,-13.,3.]))
        with self.assertRaises(ValueError):CalibratedCritic(self.base,std=0.)

    def test_hl_prior_initialization_is_common_and_train_only(self):
        target=np.array([-20.,-12.,-4.],dtype=np.float32)
        stats,prior=make_calibration(self.base,target,np.array(['a','a','b']))
        means=[]
        for group in ('R0','Rcap','Rprod','Rjoint'):
            model=CalibratedCritic(self.base,group,'hl_gauss',stats['mean'],stats['std'],prior).eval()
            logits=model.logits_cached(*self.args())
            torch.testing.assert_close(logits.softmax(-1),logits[0].softmax(-1).expand_as(logits))
            means.append(float(model.decode(logits)[0]))
        np.testing.assert_allclose(means,means[0])
        self.assertLess(abs(means[0]-stats['mean']),.005)

    def test_parameter_matching_and_gradient_reaches_encoder_boundary(self):
        counts=[]
        for group in ('Rcap','Rprod','Rjoint'):
            model=CalibratedCritic(self.base,group,mean=-13.,std=8.)
            model.freeze_encoder()
            args=self.args()
            # Zero final weight is intentional: the first step updates readout;
            # subsequent steps can update earlier gates and added branches.
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
            for _ in range(4):
                opt.zero_grad();loss=model.training_loss(model.logits_cached(*args),torch.linspace(-20.,1.,6));loss.backward();opt.step()
            self.assertGreater(model.extra_projection.weight.norm().item(),0.)
            self.assertIsNone(model.var_embedding[1].weight.grad)
            counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.assertEqual(len(set(counts)),1)

    def test_state_restore_retains_raw_value_units(self):
        model=CalibratedCritic(self.base,mean=-13.,std=8.).eval()
        args=self.args();state=model.state_dict()
        restored=CalibratedCritic(self.base).eval();restored.load_state_dict(state)
        torch.testing.assert_close(model.forward_cached(*args),restored.forward_cached(*args))


if __name__=='__main__':unittest.main()
