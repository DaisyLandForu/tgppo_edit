"""Read-only reuse of audited Mamba expert files, only TGPPO training instances."""
import hashlib
import json
from pathlib import Path
import difflib
import re
import time
import h5py
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
OUTPUT = ROOT/'tgppo/runs/light-actor-20261105/offline'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def feature_audit():
    files=[ROOT/'tgppo/experiments/stage4/metrics_binding/src/pyscipopt/scip.pyx',
           ROOT/'mamba-branching/setup/PySCIPOpt-4.3.0-features/src/pyscipopt/scip.pyx']
    texts=[p.read_text() for p in files]
    differences={}
    for name in ('relDistance','relPosition','getVarScore','gNormMax','getNodeState','getMIPState','getCandsState'):
        def extract(text):
            match=re.search(r'^    def '+name+r'\(',text,re.M)
            if not match: raise ValueError('Feature function missing: '+name)
            tail=text[match.start():];end=re.search(r'^    (?:c?def) ',tail[1:],re.M)
            if end: tail=tail[:end.start()+1]
            return [re.sub(r'\s+',' ',s.split('#')[0].strip()) for s in tail.splitlines()
                    if s.split('#')[0].strip() and not s.lstrip().startswith('@cython')]
        a,b=map(extract,texts)
        differences[name]=list(difflib.unified_diff(a,b,fromfile='SCIP6',tofile='SCIP8',n=1))
    if differences['getVarScore'] or differences['gNormMax'] or differences['relDistance']:
        raise ValueError('Feature scaling changed; review before reuse')
    # getLPBranchCands in this particular 6.0.1 binding returns a variable list.
    original=texts[0].split('    def getLPBranchCands(self):',1)[1].split('\n    def ',1)[0]
    if 'return variables' not in original: raise ValueError('Candidate count semantics require review')
    return dict(source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},
        normalized_source_differences=differences,
        interpretation=['25 candidate features have the same formulas and order; SCIP8 adds dimension checks.',
            'Node candidate-count list length in this SCIP6 binding equals SCIP8 getNumLPBranchCands.',
            'SCIP8 uses node getter APIs for matching MIP features and guards ub==lb in relPosition.',
            'Collector stores node[0:8], then mip[0:53]; action is pre-action candidate-relative position.',
            'Candidate features13/14 use callback count in expert collector vs successful branch count in TGPPO; values are preserved as transfer data, not claimed identical.',
            'SCIP8/no-cutoff native states differ in distribution from SCIP6/cutoff PPO states; no on-policy equivalence claimed.'])


def prepare():
    started=time.monotonic();OUTPUT.mkdir(parents=True,exist_ok=True)
    target=OUTPUT/'expert.npz';audit_path=OUTPUT/'data_audit.json'
    if target.exists() or audit_path.exists():
        if not target.exists() or not audit_path.exists():raise RuntimeError('Incomplete dataset; inspect before retry')
        audit=json.loads(audit_path.read_text())
        if sha256(target)!=audit['dataset_sha256']:raise ValueError('Frozen expert data changed')
        return audit
    frozen=json.loads((ROOT/'docs/stage1/protocol_draft.json').read_text())['data']
    native=json.loads((ROOT/'mamba-branching/configs/milp-s-instances.json').read_text())
    legal=sorted(set(frozen['train'])&set(native['train']))
    excluded=sorted(set(native['train'])-set(frozen['train']))
    assert len(legal)==15 and not set(legal)&set(frozen['development'])
    heldout=sorted(legal,key=lambda n:hashlib.sha256(('light-actor-offline-v1:'+n).encode()).hexdigest())[:3]
    source=ROOT/'mamba-branching/runs/milp-s-seed0-20261027'
    manifest=json.loads((source/'dataset/manifest.json').read_text())
    rows=sorted([r for r in manifest['trajectories'] if any(r['id']==n+'-s'+str(s)+'-r0'
        for n in legal for s in range(4))],key=lambda r:r['id'])
    assert len(rows)==60
    features=[];states=[];actions=[];instances=[];splits=[];raw_indices=[];source_ids=[];sources=[]
    counts_by_instance={};filtered=0
    for row in rows:
        name=row['id'].rsplit('-s',1)[0];path=source/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('Source file hash mismatch: '+str(path))
        result=json.loads((path.parent/'result.json').read_text())
        if result['task']['instance']!=name or result['task']['random_prefix_steps']!=0 or result['status']!='passed':
            raise ValueError('Unexpected expert provenance')
        assert result['cutoff'] is None
        with h5py.File(path,'r') as f:
            counts=f['counts'][:];labels=f['actions'][:]
            assert len(f['states'])==len(counts)==len(labels)==row['raw_states']
            assert ((labels>=0)&(labels<counts)).all()
            eligible=np.flatnonzero((counts>1)&(counts<=256))
            filtered+=int((counts>256).sum())
            limit=24 if name in heldout else 32
            chosen=eligible[np.linspace(0,len(eligible)-1,min(limit,len(eligible)),dtype=int)] if len(eligible) else []
            for index in chosen:
                c=np.asarray(f['candidates'][index],np.float32).reshape(-1,25)
                state=np.asarray(f['states'][index],np.float32)
                if c.shape!=(int(counts[index]),25) or state.shape!=(61,) or not np.isfinite(c).all() or not np.isfinite(state).all():
                    raise ValueError('Invalid expert state; not silently filtered')
                features.append(c);states.append(state);actions.append(int(labels[index]));instances.append(name)
                splits.append('validation' if name in heldout else 'train');raw_indices.append(int(index));source_ids.append(row['id'])
            counts_by_instance[name]=counts_by_instance.get(name,0)+len(chosen)
        sources.append(dict(id=row['id'],path=str(path.relative_to(ROOT)),sha256=row['sha256'],selected=len(chosen)))
    if not all(counts_by_instance.get(n,0)>0 for n in legal):raise ValueError('An instance has no eligible states')
    lengths=np.asarray([len(c) for c in features],np.int64)
    offsets=np.concatenate(([0],lengths.cumsum()))
    np.savez(target,candidates=np.concatenate(features),states=np.stack(states),actions=np.asarray(actions,np.int64),
        offsets=offsets,instances=np.asarray(instances),split=np.asarray(splits),raw_indices=np.asarray(raw_indices),source_ids=np.asarray(source_ids))
    audit=dict(schema='light-expert-v1',dataset_sha256=sha256(target),source_manifest_sha256=sha256(source/'dataset/manifest.json'),
        feature_audit=feature_audit(),legal_instances=legal,excluded_instances=excluded,validation_instances=heldout,
        train_instances=sorted(set(legal)-set(heldout)),states_by_instance=counts_by_instance,
        train_states=splits.count('train'),validation_states=splits.count('validation'),
        candidate_counts=dict(min=int(lengths.min()),median=float(np.median(lengths)),max=int(lengths.max())),
        source_files=sources,whole_states_with_L_gt_256_excluded=filtered,
        sampling='seed0-3; random-prefix0 only; evenly spaced <=32 train / <=24 validation states per source; no candidate truncation',
        domain='SCIP8 relpscost expert offline transfer only; no PPO replay; no dev/test fitting',
        seconds=time.monotonic()-started,new_scip_solves=0)
    write_json(audit_path,audit);return audit


def load_arrays(path=None):
    return dict(np.load(path or OUTPUT/'expert.npz',allow_pickle=False))


def batch(data, ids, device='cpu'):
    import torch
    ids=np.asarray(ids,dtype=int);lengths=data['offsets'][ids+1]-data['offsets'][ids]
    maximum=int(lengths.max());x=np.zeros((len(ids),maximum,25),np.float32)
    mask=np.arange(maximum)[None,:]>=lengths[:,None]
    for i,j in enumerate(ids):x[i,:lengths[i]]=data['candidates'][data['offsets'][j]:data['offsets'][j+1]]
    state=data['states'][ids]
    return tuple(torch.as_tensor(a,device=device) for a in (x,state[:,:8],state[:,8:],mask,data['actions'][ids]))


if __name__=='__main__':
    a=prepare();print(json.dumps({k:a[k] for k in ('train_states','validation_states','validation_instances','candidate_counts','seconds')},indent=2))
