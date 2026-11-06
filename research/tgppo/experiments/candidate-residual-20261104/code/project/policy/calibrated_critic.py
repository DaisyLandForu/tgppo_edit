"""Common static return calibration, separate from the search-information method.

MSE learns standardized targets and always decodes values to raw return units.
HL-Gauss keeps the fixed raw support and starts at the train-only target prior.
Neither changes reward, discounts, the frozen encoder, or the gated topology.
"""
import numpy as np
import torch
from torch import nn
from .search_critic import SearchCritic


def balanced_return_statistics(target, instances):
    target=np.asarray(target,dtype=np.float64)
    instances=np.asarray(instances)
    if len(target)==0 or instances.shape!=target.shape or not np.isfinite(target).all():
        raise ValueError('Nonempty finite train-only target array required')
    names=sorted(set(instances))
    weight=np.zeros(len(target),dtype=np.float64)
    for name in names:
        ids=instances==name
        weight[ids]=1./len(names)/ids.sum()
    mean=float(np.dot(weight,target))
    variance=float(np.dot(weight,(target-mean)**2))
    return dict(mean=mean,std=max(variance**.5,1e-6),variance=variance,weights=weight,
                instance_names=names)


class CalibratedCritic(SearchCritic):
    def __init__(self,base,group='R0',loss='mse',mean=0.,std=1.,prior=None):
        super().__init__(base,group,loss)
        if not np.isfinite(mean) or not np.isfinite(std) or std<=0:
            raise ValueError('Invalid return scale')
        self.register_buffer('target_mean',torch.tensor(float(mean),dtype=torch.float32))
        self.register_buffer('target_std',torch.tensor(float(std),dtype=torch.float32))
        last=self.value_head.BranchingNet[-1][0]
        nn.init.zeros_(last.weight)
        if loss=='mse':
            nn.init.zeros_(last.bias)
        else:
            p=torch.ones(64,dtype=torch.float32)/64 if prior is None else torch.as_tensor(prior,dtype=torch.float32)
            if p.shape!=(64,) or not torch.isfinite(p).all() or (p<0).any() or p.sum()<=0:
                raise ValueError('Invalid categorical prior')
            p=p.clamp(min=1e-6);p=p/p.sum()
            with torch.no_grad():last.bias.copy_(p.log())

    def decode(self,output):
        if self.loss=='mse':return self.target_mean+self.target_std*output.squeeze(-1)
        return super().decode(output)

    def training_loss(self,output,targets):
        if self.loss=='mse':
            standardized=(targets-self.target_mean)/self.target_std
            return (output.squeeze(-1)-standardized).square().mean()
        return super().training_loss(output,targets)


def make_calibration(base,target,instances):
    statistics=balanced_return_statistics(target,instances)
    probe=SearchCritic(base,loss='hl_gauss')
    with torch.no_grad():
        labels=probe.gaussian_targets(torch.as_tensor(target,dtype=torch.float32)).double()
        prior=(labels*torch.from_numpy(statistics['weights'])[:,None]).sum(0).float().numpy()
    return {k:v for k,v in statistics.items() if k!='weights'},prior
