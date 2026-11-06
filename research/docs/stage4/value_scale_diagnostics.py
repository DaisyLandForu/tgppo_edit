"""Read-only diagnosis of fitted head output ranges; no additional updates."""
from pathlib import Path
import json
import csv
import argparse
import math
import numpy as np
import torch
from project.runtime import load_agent
from project.policy.search_critic import SearchCritic
from project.utils.protocol import write_json

ROOT=Path('/root/research');RUN=ROOT/'tgppo/runs/stage4';DOC=ROOT/'docs/stage4'


def expected_bound(low,high,centers,maximize):
    # Independently varying each logit over its interval gives a valid outer
    # bound even though actual logits share two gated features.
    left,right=float(centers.min()),float(centers.max())
    for _ in range(80):
        mid=(left+right)/2
        if maximize: z=np.where(centers>=mid,high,low)
        else: z=np.where(centers>=mid,low,high)
        f=np.sum(np.exp(z-z.max())*(centers-mid))
        if f>0:left=mid
        else:right=mid
    return (left+right)/2


def main():
    global RUN, DOC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=RUN)
    parser.add_argument('--report-dir',type=Path,default=DOC)
    args=parser.parse_args();RUN=args.run_root.resolve();DOC=args.report_dir.resolve()
    DOC.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    cached=torch.load(RUN/'offline/cache.pt',map_location='cpu')['arrays']
    protocol=json.loads((ROOT/'docs/stage1/protocol_draft.json').read_text())
    train=np.isin(cached['instance'],protocol['data']['train'])
    dev=np.isin(cached['instance'],protocol['data']['development'])
    agent,_=load_agent(ROOT/'tgppo/runs/stage3/pilot-repaired/latest.pt',training=False,restore_random_state=False)
    rows=[]
    for path in sorted((RUN/'offline/fits').glob('*.pt')):
        p=torch.load(path,map_location='cpu')
        model=SearchCritic(agent.critic_network,p['group'],p['loss'])
        model.load_state_dict(p['trainable_state_dict'],strict=False)
        norm=model.value_head.BranchingNet[-2][1]
        if not isinstance(norm,torch.nn.LayerNorm):raise AssertionError('Unexpected head topology')
        dim=norm.weight.numel()
        # A coordinate of standardized d-dimensional LayerNorm is in
        # [-sqrt(d-1),sqrt(d-1)]. Then ReLU and sigmoid gate preserve [0,upper].
        upper=(norm.bias.detach().numpy()+np.abs(norm.weight.detach().numpy())*math.sqrt(dim-1)).clip(min=0)
        linear=model.value_head.BranchingNet[-1][0]
        w,b=linear.weight.detach().numpy(),linear.bias.detach().numpy()
        low=b+np.minimum(w,0).dot(upper)
        high=b+np.maximum(w,0).dot(upper)
        if p['loss']=='mse':lo,hi=float(low[0]),float(high[0])
        else:
            centers=model.return_centers.numpy()
            lo=expected_bound(low,high,centers,False);hi=expected_bound(low,high,centers,True)
        pred=np.load(path.with_name(path.stem+'-predictions.npy'))
        if pred.min()<lo-1e-4 or pred.max()>hi+1e-4:raise AssertionError('Output range bound violated')
        rows.append(dict(group=p['group'],loss=p['loss'],seed=p['seed'],
            possible_output_lower=lo,possible_output_upper=hi,
            actual_prediction_min=float(pred.min()),actual_prediction_max=float(pred.max()),
            actual_prediction_std=float(pred.std()),
            train_targets_outside_possible_range_fraction=float(np.mean((cached['target'][train]<lo)|(cached['target'][train]>hi))),
            dev_targets_outside_possible_range_fraction=float(np.mean((cached['target'][dev]<lo)|(cached['target'][dev]>hi))),
            extra_projection_l2=float(model.extra_projection.weight.norm()) if p['group']!='R0' else None))
    with (DOC/'value_scale_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    names=sorted(set(cached['instance'][dev]))
    weights=np.zeros(len(cached['target']),dtype=float)
    for name in names:
        ids=cached['instance']==name
        weights[ids]=1./len(names)/ids.sum()
    bins=np.linspace(-32,16,97)
    histogram=dict(center=(bins[:-1]+bins[1:])/2,
        target=np.histogram(cached['target'],bins=bins,weights=weights)[0])
    for loss in ('mse','hl_gauss'):
        predictions=np.mean([np.load(RUN/'offline/fits'/('Rjoint-{}-s{}-predictions.npy'.format(loss,seed)))
            for seed in (11,22,33)],axis=0)
        histogram['prediction_'+loss]=np.histogram(predictions,bins=bins,weights=weights)[0]
    with (DOC/'value_scale_histogram.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(histogram));writer.writeheader()
        writer.writerows({key:float(values[i]) for key,values in histogram.items()} for i in range(96))
    write_json(DOC/'value_scale_diagnostics.json',dict(rows=rows,
        interpretation='all groups underfit return scale; output interval of the CURRENT fitted weights, not a fixed architectural impossibility',
        protocol_changed=False, additional_training_updates=0,
        train_target_min=float(cached['target'][train].min()),train_target_max=float(cached['target'][train].max()),
        dev_target_min=float(cached['target'][dev].min()),dev_target_max=float(cached['target'][dev].max())))
    print(json.dumps(dict(passed=True,heads=len(rows),additional_training_updates=0)))


if __name__=='__main__':main()
