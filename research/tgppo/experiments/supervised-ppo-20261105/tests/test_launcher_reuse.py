"""Exercise actual BC-reuse plumbing with GPU hooks mocked; no CUDA or solves."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import torch
import launch
import server
from pretrain import load_initialization
from project import runtime as rt
from project.utils.protocol import write_json


def main():
    started=time.monotonic()
    checks=[]
    for variant in ('control','expert'):
        c,_=launch.protocol(variant)
        with tempfile.TemporaryDirectory(prefix='tgppo-reuse-check-') as td:
            context=Path(td)
            info=dict(stack={'fixture':'not_actual_target'},gpu='MOCK-NO-GPU')
            def fake_bridge(cfg,device,directory,actor_state=None):
                assert device=='cuda' and actor_state is not None
                return dict(passed=True,metrics=dict(expert_batches=int(variant=='expert')),
                    scope='mock only; no target CUDA was checked')
            with patch('engineering.bridge',side_effect=fake_bridge) as bridge, \
                 patch('target_checks.check_rng_pairing',return_value={'mock':True}):
                launch.prepare_model(c,context,11,info,False)
                launch.prepare_model(c,context,11,info,False)
                assert bridge.call_count==1
            cfg=server.params(launch.ROOT,c,'B0',11,'cpu')
            actor=rt.make_actor(dict(cfg['state_dims'],**{k:cfg[k] for k in rt.ARCH_KEYS}),'mlp')
            target=SimpleNamespace(actor_network=actor,device='cpu',actor_optimizer=SimpleNamespace(state={}))
            out=server.group_dir(context,'B0',11)
            load_initialization(target,cfg,out/'pretrain')
            original=server.read(launch.ROOT/'tgppo/runs/light-actor-20261105/mlp_bc/training/B0-s11/initialization.json')
            assert server.digest_actor(actor)==original['actor_fingerprint'] and not target.actor_optimizer.state
            assert server.read(out/'reused_initialization.json')['new_bc_training_steps']==0
            checks.append(variant+': actual BC reuse/signature/original fingerprint/fresh optimizer/cached bridge')
    assert not torch.cuda.is_initialized()
    write_json(launch.MATRIX/'preparation/launcher_checks.json',dict(passed=True,checks=checks,
        mocked_cuda_hooks=True,actual_cuda_checked=False,scip_solves=0,new_bc_steps=0,
        elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(passed=True,checks=checks,actual_cuda_checked=False,scip_solves=0),indent=2))


if __name__=='__main__':main()
