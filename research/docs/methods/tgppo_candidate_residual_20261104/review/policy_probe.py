"""Read-only trained-policy behavior on fixed saved training anchors; no labels."""
import csv
import json
from pathlib import Path
import pickle
import statistics
import time
import numpy as np
import torch
from project import runtime as rt
from project.utils.protocol import sha256

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=ROOT/'tgppo/runs/candidate-residual-20261104'

def main():
    start=time.monotonic();torch.set_num_threads(1)
    collection=RUN/'b0/training/B0-s11/collection'
    anchors=[];unavailable=[];sources={}
    # Fixed rule: first actual decision of every instance in the first20
    # episodes of B0-s11, shared by all six trained policies. No optimization.
    for batch in range(0,20,4):
        folder=collection/('e{:05d}'.format(batch))
        rows=json.loads((folder/'selected_results.json').read_text())
        for row in rows:
            files=list(folder.rglob(row['job_id']+'.pkl'));assert len(files)==1
            p=files[0];sources[str(p.relative_to(ROOT))]=sha256(p)
            with p.open('rb') as stream:payload=pickle.load(stream)
            transitions=[t for e in payload['episodes'] for t in e['transitions']]
            if not transitions:
                unavailable.append(row['instance_name']);continue
            anchors.append((row['instance_name'],transitions[0]))
    manifest=json.loads((RUN/'evaluation/episode-0100/frozen_models.json').read_text())
    all_probabilities={};state_rows=[];summary=[]
    public=json.loads((ROOT/'tgppo/configs/paper_hparams.json').read_text())
    for m in manifest['models']:
        v,s=m['variant'],m['seed']
        assert sha256(m['actor'])==m['actor_sha256']
        assert sha256(m['checkpoint'])==m['checkpoint_sha256']
        policy,export=rt.load_actor(m['actor'])
        actor=policy.actor_network.eval()
        cp=rt.read_checkpoint(m['checkpoint'])
        assert cp['progress']['next_episode']==100 and cp['progress']['phase']=='ready'
        assert cp['policy_version']==cp['update_counter']==25
        assert cp['memory']['num_transitions']==0
        assert all(cp['config'][k]==value for k,value in public.items())
        assert all(torch.equal(t,export['actor'][k]) for k,t in cp['actor'].items())
        steps={name:sorted({int(state['step']) for state in cp[name+'_optimizer']['state'].values()})
            for name in ('actor','critic')}
        assert all(value==[m['logical_minibatch_updates']] for value in steps.values())
        assert all(torch.isfinite(t).all() for net in ('actor','critic') for t in cp[net].values())
        del cp
        probabilities=[];multi=[]
        with torch.no_grad():
            for name,row in anchors:
                c=torch.as_tensor(row['cands_state']).unsqueeze(0)
                n=torch.as_tensor(row['node_state']).unsqueeze(0)
                g=torch.as_tensor(row['mip_state']).unsqueeze(0)
                p=actor(c,n,g)[0].double();p=p/p.sum()
                probabilities.append(p.numpy())
                count=len(p);tv=float(.5*(p-1/count).abs().sum())
                entropy=float(-(p[p>0]*p[p>0].log()).sum())/np.log(count) if count>1 else None
                record=dict(variant=v,training_seed=s,instance=name,candidates=count,
                    tv_uniform=tv,normalized_entropy=entropy)
                state_rows.append(record)
                if count>1:multi.append(record)
        all_probabilities[v,s]=probabilities
        weight=getattr(actor,'residual_weight',None)
        summary.append(dict(variant=v,training_seed=s,states=len(anchors),multi_candidate_states=len(multi),
            tv_uniform_median=statistics.median(r['tv_uniform'] for r in multi),
            normalized_entropy_median=statistics.median(r['normalized_entropy'] for r in multi),
            states_tv_at_least_0_01=sum(r['tv_uniform']>=.01 for r in multi),
            residual_weight_l2=float(weight.double().norm()) if weight is not None else None,
            actor_parameters=sum(t.numel() for t in actor.parameters()),optimizer_step_counts=steps))
        del actor,policy,export
    for r in summary:
        ps=all_probabilities[r['variant'],r['training_seed']]
        reference=all_probabilities['b0',r['training_seed']]
        r['tv_vs_b0_median']=statistics.median(.5*np.abs(a-b).sum() for a,b in zip(ps,reference) if len(a)>1)
    with (HERE/'policy_states.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(state_rows[0]));w.writeheader();w.writerows(state_rows)
    result=dict(models=summary,anchor_instances=[n for n,r in anchors],no_decision_instances=unavailable,
        source_hashes=sources,wall_seconds=time.monotonic()-start,optimizer_steps_executed=0,new_scip_solves=0,
        scope='one saved training anchor per participating instance; descriptive policy response, no action-value truth or generalization evidence')
    (HERE/'policy_probe.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(models=summary,wall_seconds=result['wall_seconds']),indent=2))

if __name__=='__main__':main()
