import tempfile
from pathlib import Path
import unittest
import torch
from project.runtime import seed_all,config_with_protocol,build_agent,save_checkpoint,load_agent
from test_correctness import populate


class MicrobatchTests(unittest.TestCase):
    def test_same_logical_batch_gradient_without_dropout(self):
        seed_all(17)
        config=config_with_protocol(dict(hidden_dim=32,num_layers=1,num_heads=4,
                                        batch_size=4,n_epochs=1,dropout=0.))
        a=build_agent(config)
        populate(a)
        populate(a,'second',(3.,4.))
        # Zero optimizer rates expose matching accumulated/clipped gradients.
        for opt in (a.actor_optimizer,a.critic_optimizer):
            for group in opt.param_groups: group['lr']=0.
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'same.pt'
            save_checkpoint(path,a,config)
            full,_=load_agent(path)
            mf=full.learn()
            split,_=load_agent(path)
            ms=split.learn(microbatch_size=1)
            self.assertEqual(mf['num_updates'],ms['num_updates'])
            self.assertEqual(mf['num_samples'],ms['num_samples'])
            for x,y in zip(full.actor_network.parameters(),split.actor_network.parameters()):
                torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-4)
            for x,y in zip(full.critic_network.parameters(),split.critic_network.parameters()):
                torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-4)
            self.assertAlmostEqual(mf['actor_loss'],ms['actor_loss'],places=5)
            self.assertAlmostEqual(mf['critic_loss'],ms['critic_loss'],places=5)


if __name__=='__main__': unittest.main(verbosity=2)
