"""Read-only causal head interventions on fixed real features; no optimizer or solve."""
import argparse,copy,hashlib,json,math,pickle,statistics,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import ablation,server
from project.runtime import load_actor
from project.utils.protocol import write_json,sha256

HERE=Path(__file__).resolve().parent
RUN=ablation.ROOT/'tgppo/runs/transformer-ablation-20261027'

class ExistingAffine(nn.Module):
    def __init__(self,norm):
        super().__init__();self.register_buffer('weight',norm.weight.detach().clone());self.register_buffer('bias',norm.bias.detach().clone())
    def forward(self,x):return x*self.weight+self.bias

class FixedVariance(nn.Module):
    """Diagnostic only: fix LN's denominator at the real candidate mean.

    Keeps centering, gamma/beta and the reference activation exactly; removes
    variance cancellation of the magnitude of the two-channel difference.
    Never installed in a deployed/trained policy.
    """
    def __init__(self,norm):
        super().__init__()
        self.register_buffer('weight',norm.weight.detach().double().clone())
        self.register_buffer('bias',norm.bias.detach().double().clone())
        self.eps=norm.eps
        self.reference=None
    def anchor(self,x):
        reference=x.double().mean(dim=1,keepdim=True)
        self.reference=(reference-reference.mean(-1,keepdim=True)).square().mean(-1,keepdim=True).add(self.eps).sqrt()
    def forward(self,x):
        if self.reference is None:raise RuntimeError('Real-state anchor required')
        return (x-x.mean(-1,keepdim=True))/self.reference*self.weight+self.bias

def sample(out):
    states=[]
    for f in sorted(out.glob('collection/*/selected_results.json'))[-5:]:
        for row in server.read(f):
            if len(states)>=48:break
            if not any(2<=n<=512 for n in row.get('candidate_counts',[])):continue
            path=next(f.parent.glob('attempt-*/'+row['job_id']+'.pkl'))
            with path.open('rb') as stream:p=pickle.load(stream)
            valid=[(idx,r) for ep in p['episodes'] for idx,r in enumerate(ep['transitions']) if 2<=len(r['cands_state'])<=512]
            for i in sorted(set([0,len(valid)//2,len(valid)-1])) if valid else []:
                idx,r=valid[i];states.append(dict(instance=row['instance_name'],job_id=row['job_id'],transition=idx,**{k:np.array(r[k],copy=True) for k in ('cands_state','node_state','mip_state')}))
    return states

def intervene(head,widths,affine=True,dtype=torch.float64,epsilon=None,fixed_variance=False):
    h=copy.deepcopy(head).to(dtype).eval()
    for layer in h.BranchingNet:
        for i,sub in enumerate(layer):
            if isinstance(sub,nn.LayerNorm) and sub.normalized_shape[0] in widths:
                if fixed_variance:layer[i]=FixedVariance(sub)
                elif epsilon is not None:sub.eps=epsilon
                else:layer[i]=ExistingAffine(sub) if affine else nn.Identity()
    return h.requires_grad_(False)

def distribution(logits):
    p=logits.double().softmax(-1)[0];n=p.numel();u=torch.full_like(p,1/n)
    return dict(logit_span=float(logits.max()-logits.min()),tv_uniform=float((p-u).abs().sum()/2),normalized_entropy=float(-(p*p.clamp_min(1e-300).log()).sum()/math.log(n)))

def grad_probe(head,z,tree,i,j):
    q=z.double().detach().requires_grad_(True);t=tree.double()
    logits=head(q,t);contrast=logits[0,i]-logits[0,j]
    grad,=torch.autograd.grad(contrast,q,retain_graph=True);norm=float(grad.norm())
    lp=logits.log_softmax(-1)[0,i]
    gprob,=torch.autograd.grad(lp,q)
    result=dict(contrast_gradient_l2=norm,logprob_gradient_l2=float(gprob.norm()))
    if norm>0:
        direction=grad.detach()/norm;errors=[]
        with torch.no_grad():
            for step in (1e-3,1e-4,1e-5,1e-6):
                plus=head(q.detach()+step*direction,t);minus=head(q.detach()-step*direction,t)
                fd=float(((plus[0,i]-plus[0,j])-(minus[0,i]-minus[0,j]))/(2*step))
                errors.append(dict(step=step,value=fd,absolute_error=abs(fd-norm)))
        best=min(errors,key=lambda r:r['absolute_error'])
        result.update(finite_difference=best['value'],relative_fd_error=best['absolute_error']/max(norm,1e-30),finite_differences=errors)
    return result

def run():
    torch.set_num_threads(1);records=[];summaries=[];checks={};protected={}
    variants={'original32':((),True,torch.float32),'original64':((),True,torch.float64),
        'bypass_ln2_affine':((2,),True,torch.float64),'bypass_ln4_affine':((4,),True,torch.float64),
        'bypass_ln2_4_affine':((2,4),True,torch.float64),'bypass_ln2_identity':((2,),False,torch.float64)}
    variants.update(ln2_eps_1e_3=((2,),True,torch.float64,1e-3),ln2_eps_1e_2=((2,),True,torch.float64,1e-2),
                    ln2_fixed_variance=((2,),True,torch.float64,None,True))
    gradient_keys=('original64','bypass_ln2_affine','bypass_ln4_affine','bypass_ln2_4_affine','ln2_eps_1e_2','ln2_fixed_variance')
    for variant in ('transformer','bimatching'):
        for seed in (11,22):
            name=variant+'/s'+str(seed);out=RUN/variant/'training'/('B0-s'+str(seed));path=out/'latest.pt'
            protected[str(path)]=sha256(path);policy,payload=load_actor(path);actor=policy.actor_network.eval().requires_grad_(False)
            source_digest=server.digest_actor(actor);states=sample(out);heads={key:intervene(actor.output_layer,*args) for key,args in variants.items()}
            own=[];cache={}
            def bm(module,args,output):cache['z']=output.detach().clone();cache['tree']=args[0].detach().clone()
            def ln(module,args,output):cache['ln2_in']=args[0].detach().clone();cache['ln2_out']=output.detach().clone()
            handles=[actor.tree_refinement.register_forward_hook(bm),actor.output_layer.BranchingNet[6][1].register_forward_hook(ln)]
            for idx,state in enumerate(states):
                with torch.no_grad():
                    c,n,m=[torch.as_tensor(state[k]).unsqueeze(0) for k in ('cands_state','node_state','mip_state')]
                    original=actor(c,n,m);z=cache['z'];tree=cache['tree'];x=cache['ln2_in'].double();norm=actor.output_layer.BranchingNet[6][1]
                    d=x[...,0]-x[...,1];eps=norm.eps;a=d/(d.square()+4*eps).sqrt()
                    predicted=torch.stack([a,-a],-1)*norm.weight.double()+norm.bias.double()
                    actual=nn.functional.layer_norm(x,(2,),norm.weight.double(),norm.bias.double(),eps)
                    formula_error=float((actual-predicted).abs().max())
                    derivative=4*eps/(d.square()+4*eps).pow(1.5)
                    is_same_sign=bool((d>0).all() or (d<0).all())
                    heads['ln2_fixed_variance'].BranchingNet[6][1].anchor(x)
                    reference=x.mean(1,keepdim=True)
                    anchor_error=float((heads['ln2_fixed_variance'].BranchingNet[6][1](reference)-
                        nn.functional.layer_norm(reference,(2,),norm.weight.double(),norm.bias.double(),eps)).abs().max())
                    stats={key:distribution(head(z.to(next(head.parameters()).dtype),tree.to(next(head.parameters()).dtype))) for key,head in heads.items()}
                    replay=heads['original32'](z,tree).softmax(-1)
                    parity=bool(torch.equal(original,replay))
                # Validate analytical Jacobian on actual two-channel inputs in double precision.
                xx=x.detach().requires_grad_(True)
                yy=nn.functional.layer_norm(xx,(2,),norm.weight.double(),norm.bias.double(),eps)
                g,=torch.autograd.grad(yy[...,0].sum(),xx)
                expected=derivative*norm.weight.double()[0]
                jacobian_error=float((g-torch.stack([expected,-expected],-1)).abs().max())
                row=dict(model=name,instance=state['instance'],job_id=state['job_id'],transition=state['transition'],candidates=len(state['cands_state']),
                    bitwise_replay=parity,formula_error=formula_error,jacobian_error=jacobian_error,anchor_error=anchor_error,ln2_same_channel_order=is_same_sign,
                    ln2_abs_channel_difference_min=float(d.abs().min()),ln2_abs_channel_difference_max=float(d.abs().max()),
                    ln2_saturation_derivative_median=float(derivative.median()),ln2_input_candidate_span=float((x.max(1).values-x.min(1).values).abs().max()),
                    ln2_output_candidate_span=float((actual.max(1).values-actual.min(1).values).abs().max()),variants=stats)
                if idx%4==0:
                    i=0;j=int((z[0]-z[0,0]).norm(dim=-1).argmax())
                    if j!=i:row['gradients']={key:grad_probe(heads[key],z,tree,i,j) for key in gradient_keys}
                own.append(row);records.append(row)
            for handle in handles:handle.remove()
            summary=dict(model=name,states=len(own),same_channel_order_fraction=sum(r['ln2_same_channel_order'] for r in own)/len(own),
                min_abs_channel_difference=min(r['ln2_abs_channel_difference_min'] for r in own),max_formula_error=max(r['formula_error'] for r in own),
                max_jacobian_error=max(r['jacobian_error'] for r in own),variants={},gradients={})
            for key in variants:
                summary['variants'][key]={field:dict(median=float(np.median([r['variants'][key][field] for r in own])),max=max(r['variants'][key][field] for r in own)) for field in ('logit_span','tv_uniform','normalized_entropy')}
            for key in gradient_keys:
                values=[r['gradients'][key]['contrast_gradient_l2'] for r in own if 'gradients' in r]
                summary['gradients'][key]=dict(contrast_gradient_l2_median=float(np.median(values)),nonzero=sum(v>0 for v in values),samples=len(values),finite_difference_max_abs_error=max(abs(r['gradients'][key].get('finite_difference',0)-r['gradients'][key]['contrast_gradient_l2']) for r in own if 'gradients' in r))
            checks[name+'_replay']=all(r['bitwise_replay'] for r in own)
            checks[name+'_ln_formula']=summary['max_formula_error']<1e-10
            checks[name+'_ln_jacobian']=summary['max_jacobian_error']<1e-8
            checks[name+'_original_actor_unchanged']=server.digest_actor(actor)==source_digest
            checks[name+'_checkpoint_unchanged']=sha256(path)==protected[str(path)]
            checks[name+'_all_same_channel_order']=all(r['ln2_same_channel_order'] for r in own)
            checks[name+'_fixed_variance_anchor_parity']=max(r['anchor_error'] for r in own)<1e-10
            checks[name+'_fixed_variance_restores_gradient']=summary['gradients']['ln2_fixed_variance']['contrast_gradient_l2_median']>100*summary['gradients']['original64']['contrast_gradient_l2_median']
            checks[name+'_fixed_variance_finite_difference']=summary['gradients']['ln2_fixed_variance']['finite_difference_max_abs_error']<1e-7
            summaries.append(summary);print(json.dumps(summary),flush=True)
            write_json(HERE/'causal-results.json',dict(checks=checks,summaries=summaries,records=records,protected_checkpoint_sha256=protected,scope='Fixed-real-feature counterfactual head diagnostics only; neither trained repair nor solve performance.'))
            del actor,policy,payload,heads
    if not all(checks.values()):raise RuntimeError('Causal diagnostic validation failed')

if __name__=='__main__':run()
