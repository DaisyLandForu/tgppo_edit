"""Scientific invariants, update weighting, and delivery/duplicate guards."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import os
import signal
import subprocess
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.distributions import Categorical

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'code'))
import common as c
import runner
from frozen_model.actor import Actor
from frozen_model.modules import TreeGateBranchingNet

torch.set_num_threads(1)


def fixture():
    torch.manual_seed(8)
    original=TreeGateBranchingNet(8,8,2,1,norm='layer',depth=2,hidden_size=8)
    rows=[]
    for i,L in enumerate([2,5,3,4,6]):
        p=torch.full((L,),1/L)
        rows.append(dict(features=torch.randn(L,8),tree=torch.randn(8),action=i%L,
                         old_logprob=float(Categorical(p).log_prob(torch.tensor(i%L))),
                         advantage=[1.,-.7,.4,-1.2,.8][i],instance=f'i{i%2}',key=f'r{i}'))
    return dict(seed=11,hidden_dim=8,head_state=original.state_dict(),train=rows,monitor=copy.deepcopy(rows[:2]))


class HeadProbeTests(unittest.TestCase):
    def test_full_actor_cache_exact(self):
        torch.manual_seed(11)
        actor=Actor(3,2,2,8,2,1,.05).eval()
        row=dict(cands_state=np.random.RandomState(11).normal(size=(5,3)).astype('float32'),
                 node_state=np.array([.1,.4],dtype='float32'),mip_state=np.array([.2,.7],dtype='float32'),action=2)
        features,tree,logp=runner.capture(actor,row)
        self.assertEqual(features.shape,(5,8));self.assertEqual(tree.shape,(8,));self.assertTrue(np.isfinite(logp))

    def test_paired_uniform_initial_policy(self):
        cache=fixture();p=c.protocol()
        a=c.evaluate(c.make_head(cache,'original'),cache['train'],p)
        b=c.evaluate(c.make_head(cache,'linear'),cache['train'],p)
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x['probability'],y['probability'])
            self.assertLess(x['tv_uniform'],1e-12)
            self.assertLess(abs(x['old_action_ratio']-1),1e-6)

    def test_nonuniform_permutation_padding_and_mask(self):
        cache=fixture();head=c.make_head(cache,'linear')
        with torch.no_grad():head.readout.weight.copy_(torch.arange(8,dtype=torch.float32)[None]/8)
        row=cache['train'][1];base=c.collate([row],[0]);_,p=c.forward(head,base)
        self.assertGreater(float((p[0]-1/p.shape[1]).abs().sum()),.01)
        perm=torch.tensor([2,0,4,1,3]);changed=copy.deepcopy(row);changed['features']=row['features'][perm]
        _,pp=c.forward(head,c.collate([changed],[0]))
        torch.testing.assert_close(pp[0][torch.argsort(perm)],p[0],atol=1e-7,rtol=1e-6)
        padded=dict(base);padded['features']=torch.cat([base['features'],torch.randn(1,3,8)],1)
        padded['mask']=torch.tensor([[False]*5+[True]*3]);_,pad=c.forward(head,padded)
        torch.testing.assert_close(pad[:,:5],p,atol=1e-7,rtol=1e-6);self.assertEqual(float(pad[:,5:].sum()),0.)
        padded['mask'][:]=True
        with self.assertRaises(ValueError):c.forward(head,padded)

    def test_microbatch_gradient_equivalence_fp32(self):
        cache=fixture();p=c.protocol()
        for variant in ['original','linear']:
            a=c.make_head(cache,variant)
            # Nonzero output exercises upstream gradients instead of passing a
            # trivial zero-readout test.
            last=a.BranchingNet[-1][0] if variant=='original' else a.readout
            with torch.no_grad():last.weight.fill_(.31)
            b=copy.deepcopy(a)
            for net,mb in [(a,5),(b,2)]:
                net.zero_grad(set_to_none=True);c.backward_batch(net,cache['train'],p,microbatch=mb)
            norms=[]
            for x,y in zip(a.parameters(),b.parameters()):
                if x.grad is not None:
                    torch.testing.assert_close(x.grad,y.grad,atol=2e-7,rtol=2e-5);norms.append(float(x.grad.norm()))
            self.assertGreater(sum(norms),1e-5)

    def test_fp64_adam_accumulation_equivalence(self):
        # Near-zero FP32 gradients can be amplified differently by fresh Adam.
        # Verify the accumulation/update math separately in double precision;
        # production jobs all retain the same fixed FP32 microbatch=16.
        cache=fixture();p=c.protocol()
        for rows in [cache['train'],cache['monitor']]:
            for r in rows:r['features']=r['features'].double();r['tree']=r['tree'].double()
        for variant in ['original','linear']:
            a=c.make_head(cache,variant).double()
            last=a.BranchingNet[-1][0] if variant=='original' else a.readout
            with torch.no_grad():last.weight.fill_(.31)
            b=copy.deepcopy(a)
            for net,mb in [(a,5),(b,2)]:
                opt=torch.optim.Adam(net.parameters(),lr=p['actor_lr'])
                net.zero_grad(set_to_none=True);c.backward_batch(net,cache['train'],p,microbatch=mb)
                torch.nn.utils.clip_grad_norm_(net.parameters(),p['gradient_clip']);opt.step()
            for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-9)

    def test_only_head_changes_and_checkpoint_step_cap(self):
        cache=fixture();before=copy.deepcopy(cache);p=c.protocol();p['max_actor_steps']=2
        with tempfile.TemporaryDirectory() as d:
            result=runner.run_head_fit(cache,'linear',p,Path(d),dict(test=True))
            self.assertEqual(result['completed_steps'],2);self.assertFalse(result['full_actor_training'])
            checkpoint=torch.load(Path(d)/'latest.pt',map_location='cpu')
            self.assertEqual(checkpoint['step'],2);self.assertEqual(len(json.loads((Path(d)/'history.json').read_text())),3)
            self.assertFalse((Path(d)/'inflight.json').exists())
            self.assertTrue(any(float(v.abs().sum())>0 for k,v in checkpoint['head'].items()))
        for old,new in zip(before['train'],cache['train']):
            torch.testing.assert_close(old['features'],new['features']);self.assertIsNone(new['features'].grad)
        for k,v in before['head_state'].items():torch.testing.assert_close(v,cache['head_state'][k])

    def test_four_configs_are_paired(self):
        for seed in [11,22]:
            a=c.job_config(f'original-s{seed}');b=c.job_config(f'linear-s{seed}')
            self.assertEqual(a['seed'],b['seed']);self.assertEqual(a['device'],b['device'])
        with self.assertRaises(ValueError):c.job_config('original-s33')

    def test_corrupt_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'prepared').mkdir();cache=root/'prepared/x.pt';cache.write_bytes(b'valid')
            m=dict(protocol_sha256=c.canonical(c.protocol()),code_fingerprint=c.fingerprint(),
                   caches={'11':dict(path='prepared/x.pt',sha256=c.sha(cache))},torch_version=torch.__version__)
            c.save(root/'prepared/manifest.json',m);cache.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'corrupted'):c.load_prepared(root)

    def test_aggregate_refuses_missing_jobs(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'load_prepared',return_value=(c.protocol(),{})),patch.object(runner,'verify_job',side_effect=ValueError('missing')):
            with self.assertRaises(ValueError):runner.aggregate(Path(d))
            m=json.loads((Path(d)/'summary/completeness.json').read_text())
            self.assertFalse(m['complete']);self.assertEqual(len(m['missing_or_failed']),4)

    def test_duplicate_completed_job_does_not_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'jobs/original-s11';path.mkdir(parents=True);c.save(path/'budget.json',dict(status='completed'))
            with patch.object(runner,'identity'),patch.object(runner,'load_prepared'),patch.object(runner,'verify_job'),patch.object(runner.subprocess,'Popen') as popen:
                runner.supervised(root,'run','original-s11');popen.assert_not_called()

    def test_failed_job_does_not_silently_reset_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'jobs/original-s11';path.mkdir(parents=True);c.save(path/'budget.json',dict(status='failed'))
            with patch.object(runner,'identity'),patch.object(runner,'load_prepared'),patch.object(runner.subprocess,'Popen') as popen:
                with self.assertRaisesRegex(ValueError,'previous attempt'):runner.supervised(root,'run','original-s11')
                popen.assert_not_called()

    def test_guardian_enforces_timeout_and_reaps_child(self):
        real_popen=subprocess.Popen;children=[]
        def sleeping_child(*args,**kwargs):
            child=real_popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
            children.append(child);return child
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'START',time.monotonic()),patch.object(runner.subprocess,'Popen',side_effect=sleeping_child):
            with self.assertRaises(SystemExit) as exc:runner.supervised_locked(Path(d),'run','toy',Path(d),5.1,[])
            self.assertEqual(exc.exception.code,124);self.assertIsNotNone(children[0].poll())
            self.assertEqual(json.loads((Path(d)/'budget.json').read_text())['status'],'budget_timeout')

    def test_guardian_forwards_platform_sigterm(self):
        real_popen=subprocess.Popen;children=[];timers=[]
        def sleeping_child(*args,**kwargs):
            child=real_popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
            children.append(child)
            timer=threading.Timer(.1,lambda:os.kill(os.getpid(),signal.SIGTERM));timer.start();timers.append(timer)
            return child
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'START',time.monotonic()),patch.object(runner.subprocess,'Popen',side_effect=sleeping_child):
            with self.assertRaises(SystemExit) as exc:runner.supervised_locked(Path(d),'run','toy',Path(d),10,[])
            self.assertEqual(exc.exception.code,143);self.assertIsNotNone(children[0].poll())
            self.assertEqual(json.loads((Path(d)/'budget.json').read_text())['exit_code'],143)
        for timer in timers:timer.join()


if __name__=='__main__':unittest.main()
