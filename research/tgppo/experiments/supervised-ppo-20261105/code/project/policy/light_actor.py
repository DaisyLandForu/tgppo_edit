"""Three stateless, permutation-equivariant Actors over existing 25/8/53 features.

No historical policy, graph features, or residual addition to the original logits.
"""
import torch
from torch import nn
from .actor import Actor


class MAB(nn.Module):
    """Pre-normalized cross attention with unmodified query residual path."""
    def __init__(self, width, heads=4):
        super().__init__()
        self.qnorm = nn.LayerNorm(width)
        self.knorm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.ffnorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, query, keys, mask=None):
        q, k = self.qnorm(query), self.knorm(keys)
        out = query + self.attn(q, k, k, key_padding_mask=mask, need_weights=False)[0]
        return out + self.ff(self.ffnorm(out))


class LightActor(nn.Module):
    def __init__(self, kind, width=64, inducing=8):
        super().__init__()
        if kind not in ('mlp', 'set', 'isab', 'tiny_transformer'):
            raise ValueError('Unknown light Actor: '+kind)
        self.kind, self.width = kind, width
        self.var_embedding = nn.Sequential(nn.LayerNorm(25), nn.Linear(25,width),
            nn.GELU(), nn.Linear(width,width), nn.GELU())
        self.tree_embedding = nn.Sequential(nn.LayerNorm(61), nn.Linear(61,width),
            nn.GELU(), nn.Linear(width,width), nn.GELU())
        if kind == 'isab':
            self.inducing = nn.Parameter(torch.empty(1,inducing,width))
            nn.init.normal_(self.inducing, std=.02)
            self.compress, self.readback = MAB(width), MAB(width)
        elif kind == 'tiny_transformer':
            self.interaction = nn.TransformerEncoderLayer(width,4,2*width,dropout=0.,
                activation='gelu',batch_first=True,norm_first=True)
        head_input = 4*width+1 if kind == 'set' else 2*width
        self.head = nn.Sequential(nn.Linear(head_input,width), nn.GELU(),
            nn.Linear(width,32), nn.GELU(), nn.Linear(32,1))

    def logits(self, cands, node, mip, padding_mask=None):
        h = self.var_embedding(cands)
        t = self.tree_embedding(torch.cat((node,mip),-1))
        mask = (torch.zeros(cands.shape[:2], dtype=torch.bool, device=cands.device)
                if padding_mask is None else padding_mask)
        valid = (~mask).unsqueeze(-1)
        if self.kind == 'set':
            count = valid.sum(1).clamp_min(1)
            mean = h.masked_fill(~valid,0.).sum(1)/count
            maximum = h.masked_fill(~valid,float('-inf')).max(1).values
            context = torch.cat((mean,maximum,torch.log1p(count.to(h.dtype))),-1)
            features = torch.cat((h,t[:,None,:].expand_as(h),
                context[:,None,:].expand(-1,h.shape[1],-1)),-1)
        else:
            if self.kind == 'isab':
                conditioned = h+t[:,None,:]
                summaries = self.compress(self.inducing.expand(h.shape[0],-1,-1),conditioned,mask)
                h = self.readback(conditioned,summaries)
            elif self.kind == 'tiny_transformer':
                h = self.interaction(h+t[:,None,:],src_key_padding_mask=mask)
            features = torch.cat((h,t[:,None,:].expand_as(h)),-1)
        return self.head(features).squeeze(-1).masked_fill(mask,float('-inf'))

    def forward(self, cands_state_mat, node_state, mip_state, padding_mask=None):
        return self.logits(cands_state_mat,node_state,mip_state,padding_mask).softmax(-1)


def make_actor(arch, kind='b0', width=64, inducing=8):
    if kind == 'b0':
        return Actor(**arch)
    return LightActor(kind,width,inducing)
