"""Shared, solver-free code for the offline scoring-head experiment."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from frozen_model.modules import TreeGateBranchingNet

PACKAGE=Path(__file__).resolve().parents[1]
RESEARCH=PACKAGE.parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_csv(path):return list(csv.DictReader(Path(path).open()))


def fingerprint():
    paths=[PACKAGE/'protocol.json',PACKAGE/'custom_job.sh']
    paths+=sorted((PACKAGE/'code').rglob('*.py'))+sorted((PACKAGE/'configs').glob('*.json'))
    return {str(p.relative_to(PACKAGE)):sha(p) for p in paths}


def protocol():
    p=json.loads((PACKAGE/'protocol.json').read_text())
    if p['schema']!='tgppo-offline-head-probe-v1':raise ValueError('Wrong experiment schema')
    if p['variants']!=['original','linear'] or p['seeds']!=[11,22]:raise ValueError('Unexpected matrix')
    if (p['max_actor_steps'],p['logical_batch_size'],p['microbatch_size'])!=(20,256,16):raise ValueError('Frozen step/batch scope changed')
    if any(p[k] for k in ['solver_tasks','full_actor_updates','critic_updates']):raise ValueError('Outside offline-head scope')
    return p


def job_config(job_id):
    if job_id not in [f'{v}-s{s}' for s in [11,22] for v in ['original','linear']]:raise ValueError('Unknown job')
    c=json.loads((PACKAGE/'configs'/f'{job_id}.json').read_text())
    if c!={'job_id':job_id,'variant':job_id.split('-s')[0],'seed':int(job_id.split('-s')[1]),'device':'cpu'}:raise ValueError('Job differs from frozen matrix')
    return c


def set_threads():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


class LinearHead(nn.Module):
    def __init__(self,hidden):
        super().__init__();self.readout=nn.Linear(hidden,1)
    def forward(self,features,tree):
        return self.readout(features).squeeze(-1)


def make_head(cache,variant):
    hidden=cache['hidden_dim']
    if variant=='original':
        head=TreeGateBranchingNet(hidden,hidden,2,1,norm='layer',depth=2,hidden_size=hidden)
        head.load_state_dict(cache['head_state'])
        last=head.BranchingNet[-1][0]
    elif variant=='linear':
        head=LinearHead(hidden);last=head.readout
    else:raise ValueError('Unknown scoring-head variant')
    with torch.no_grad():last.weight.zero_();last.bias.zero_()
    return head.eval()


def collate(rows,indices,device='cpu'):
    chosen=[rows[int(i)] for i in indices]
    lengths=torch.tensor([len(r['features']) for r in chosen],device=device)
    features=nn.utils.rnn.pad_sequence([r['features'].to(device) for r in chosen],batch_first=True)
    tree=torch.stack([r['tree'] for r in chosen]).to(device)
    mask=torch.arange(features.shape[1],device=device)[None]>=lengths[:,None]
    return dict(features=features,tree=tree,mask=mask,lengths=lengths,
                actions=torch.tensor([r['action'] for r in chosen],dtype=torch.long,device=device),
                old_logprob=torch.tensor([r['old_logprob'] for r in chosen],dtype=torch.float32,device=device),
                advantages=torch.tensor([r.get('advantage',0.) for r in chosen],dtype=torch.float32,device=device))


def chunks(rows,size):
    order=np.argsort([len(r['features']) for r in rows],kind='stable')
    for start in range(0,len(order),size):yield order[start:start+size]


def forward(head,batch):
    logits=head(batch['features'],batch['tree'])
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
    probs=torch.softmax(logits.masked_fill(batch['mask'],float('-inf')),-1)
    if not torch.isfinite(probs).all() or probs[batch['mask']].abs().sum()!=0:raise ValueError('Invalid candidate mask')
    return logits,probs


def backward_batch(head,rows,config,microbatch=None):
    head.eval()
    totals=dict(surrogate_loss=0.,entropy=0.,actor_loss=0.,clip_fraction=0.)
    size=microbatch or config['microbatch_size']
    for ids in chunks(rows,size):
        b=collate(rows,ids)
        logits,probs=forward(head,b);dist=Categorical(probs)
        ratio=torch.exp(dist.log_prob(b['actions'])-b['old_logprob'])
        clip=config['ppo_clip'];adv=b['advantages']
        surrogate=-torch.min(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
        entropy=dist.entropy().mean();loss=surrogate-config['entropy_weight']*entropy
        if not torch.isfinite(loss):raise ValueError('Nonfinite Actor objective')
        weight=len(ids)/len(rows);(weight*loss).backward()
        for k,v in dict(surrogate_loss=surrogate,entropy=entropy,actor_loss=loss,
                        clip_fraction=((ratio-1).abs()>clip).float().mean()).items():totals[k]+=weight*float(v.detach())
    return totals


def evaluate(head,rows,config,initial=None):
    head.eval();records=[None]*len(rows)
    with torch.no_grad():
        for ids in chunks(rows,config['microbatch_size']):
            b=collate(rows,ids);logits,probs=forward(head,b);dist=Categorical(probs)
            logps=dist.log_prob(b['actions']);ent=dist.entropy()
            ratio=torch.exp(logps-b['old_logprob'])
            for j,index in enumerate(ids):
                L=int(b['lengths'][j]);p=probs[j,:L].double().numpy();p=p/p.sum()
                l=logits[j,:L].double().numpy();u=np.full(L,1/L)
                old=initial[int(index)] if initial is not None else None
                p0=old['probability'] if old is not None else u
                delta=l-old['logits'] if old is not None else l-l.mean()
                records[int(index)]=dict(probability=p,logits=l,L=L,
                                         tv_uniform=float(.5*np.abs(p-u).sum()),
                                         tv_initial=float(.5*np.abs(p-p0).sum()),
                                         kl_initial=max(0.,float(np.sum(p0*np.log(p0/p)))),
                                         normalized_entropy=(math.log(L)-max(0.,float(np.sum(p*np.log(p*L)))))/math.log(L) if L>1 else None,
                                         old_action_ratio=float(ratio[j]),log_prob=float(logps[j]),entropy=float(ent[j]),
                                         common_logit_shift=float(delta.mean()),
                                         centered_logit_delta_rms=float(np.sqrt(np.mean((delta-delta.mean())**2))),
                                         logit_spread=float(np.ptp(l)))
    return records


def summarize_records(rows,records):
    groups=defaultdict(list)
    for row,r in zip(rows,records):groups[row['instance']].append(r)
    return dict(states=len(records),instances=len(groups),
                tv_uniform_p50=float(np.median([r['tv_uniform'] for r in records])),
                tv_uniform_max=max(r['tv_uniform'] for r in records),
                tv_initial_p50=float(np.median([r['tv_initial'] for r in records])),
                tv_initial_max=max(r['tv_initial'] for r in records),
                instance_median_tv_initial_p50=float(np.median([np.median([r['tv_initial'] for r in rr]) for rr in groups.values()])),
                mean_kl_initial=float(np.mean([r['kl_initial'] for r in records])),
                old_ratio_max_error=max(abs(r['old_action_ratio']-1) for r in records),
                numerical_response_states=sum(r['tv_initial']>1e-6 for r in records),
                visible_nonuniform_states=sum(r['tv_uniform']>=.01 for r in records),
                centered_logit_delta_rms_p50=float(np.median([r['centered_logit_delta_rms'] for r in records])),
                abs_common_logit_shift_p50=float(np.median([abs(r['common_logit_shift']) for r in records])))


def objective(records,rows,config):
    ratio=torch.tensor([r['old_action_ratio'] for r in records],dtype=torch.float32)
    adv=torch.tensor([r['advantage'] for r in rows],dtype=torch.float32)
    clip=config['ppo_clip'];surrogate=float(-torch.min(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean())
    entropy=float(np.mean([r['entropy'] for r in records]))
    return dict(surrogate_loss=surrogate,entropy=entropy,actor_loss=surrogate-config['entropy_weight']*entropy,
                clip_fraction=float(((ratio-1).abs()>clip).float().mean()))


def load_prepared(run,check_sources=False):
    p=protocol();manifest_path=run/'prepared/manifest.json'
    if not manifest_path.exists():raise ValueError('Run prepare before submitting the four jobs')
    m=json.loads(manifest_path.read_text())
    if m['protocol_sha256']!=canonical(p) or m['code_fingerprint']!=fingerprint():raise ValueError('Prepared code/protocol drift; do not mix runs')
    for seed,entry in m['caches'].items():
        if sha(run/entry['path'])!=entry['sha256']:raise ValueError('Prepared cache corrupted: '+seed)
    if check_sources:
        for rel,h in m['input_source_hashes'].items():
            if sha(RESEARCH/rel)!=h:raise ValueError('Source artifact changed: '+rel)
    if m['torch_version']!=torch.__version__:raise ValueError('Use the same Torch version as cache preparation')
    return p,m
