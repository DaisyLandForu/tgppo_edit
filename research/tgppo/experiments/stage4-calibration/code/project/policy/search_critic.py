"""Critic-only search extension; the upstream Actor/Critic modules stay untouched.

Frozen encoder caching is the stage-4 diagnostic. forward() also provides the
same input-to-value chain for checking future end-to-end use; no PPO is run here.
"""
import copy
import torch
from torch import nn
from .critic import Critic


class SearchCritic(Critic):
    def __init__(self, base, group='R0', loss='mse'):
        nn.Module.__init__(self)
        if group not in ('R0', 'Rcap', 'Rprod', 'Rjoint') or loss not in ('mse', 'hl_gauss'):
            raise ValueError('Unknown critic variant')
        for name, module in base.named_children():
            self.add_module(name, copy.deepcopy(module))
        self.hidden_dim = base.hidden_dim
        self.group, self.loss = group, loss
        self.time_projection = nn.Linear(1, self.hidden_dim, bias=False)
        nn.init.zeros_(self.time_projection.weight)
        if loss == 'hl_gauss':
            previous = self.value_head.BranchingNet[-1][0]
            self.value_head.BranchingNet[-1][0] = nn.Linear(previous.in_features, 64)
        if group != 'R0':
            self.extra = nn.Sequential(nn.Linear(69, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
            self.extra_projection = nn.Linear(32, self.hidden_dim, bias=False)
            nn.init.zeros_(self.extra_projection.weight)
        self.register_buffer('return_edges', torch.linspace(-36., 52., 65))
        self.register_buffer('return_centers', (self.return_edges[:-1]+self.return_edges[1:])/2)

    def freeze_encoder(self):
        for name in ('var_embedding', 'tree_embedding', 'transformer', 'tree_refinement', 'aggregation'):
            for p in getattr(self, name).parameters():
                p.requires_grad_(False)

    def encode(self, cands_state_mat, node_state, mip_state, padding_mask=None):
        batch, count, _ = cands_state_mat.shape
        if padding_mask is None:
            padding_mask = torch.zeros((batch, count), dtype=torch.bool, device=cands_state_mat.device)
        var = self.var_embedding(cands_state_mat).transpose(0, 1)
        var = self.transformer(var, src_key_padding_mask=padding_mask).transpose(0, 1)
        tree = self.tree_embedding(torch.cat([node_state, mip_state], dim=-1))
        refined = self.tree_refinement(tree, var, padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (refined*valid).sum(1)/valid.sum(1).clamp(min=1)
        aggregated = self.aggregation(torch.cat([pooled, tree], dim=-1))
        return aggregated, tree

    def extra_input(self, tree_state, remaining_time, frontier):
        if self.group == 'Rcap':
            return torch.cat([tree_state, remaining_time, torch.zeros_like(frontier[:, :7])], dim=-1)
        if self.group == 'Rprod':
            h = frontier[:, :64].reshape(-1, 8, 8)
            h = h.sum(2).unsqueeze(2)*h.sum(1).unsqueeze(1)
            return torch.cat([h.flatten(1), frontier[:, 64:]], dim=-1)
        return frontier

    def logits_cached(self, aggregated, tree, tree_state, remaining_time, frontier):
        z = aggregated + self.time_projection(remaining_time)
        if self.group != 'R0':
            z = z + self.extra_projection(self.extra(self.extra_input(tree_state, remaining_time, frontier)))
        gates = self.value_head.GatingNet(tree)
        start = 0
        for index, layer in enumerate(self.value_head.BranchingNet):
            stop = start + self.value_head.n_units_dict[index]
            z = layer(z*gates[:, start:stop])
            start = stop
        return z

    def decode(self, output):
        return (output.softmax(-1)*self.return_centers).sum(-1) if self.loss == 'hl_gauss' else output.squeeze(-1)

    def forward_cached(self, *args):
        return self.decode(self.logits_cached(*args))

    def forward(self, cands_state_mat, node_state, mip_state, remaining_time, frontier, padding_mask=None):
        a, t = self.encode(cands_state_mat, node_state, mip_state, padding_mask)
        return self.forward_cached(a, t, torch.cat([node_state, mip_state], -1), remaining_time, frontier)

    def gaussian_targets(self, targets):
        if not torch.isfinite(targets).all() or (targets < -36).any() or (targets > 52).any():
            raise ValueError('Return outside frozen HL-Gauss support')
        sigma = .75*(88./64)
        cdf = .5*(1+torch.erf((self.return_edges[None, :]-targets[:, None])/(sigma*2**.5)))
        probabilities = (cdf[:, 1:]-cdf[:, :-1]).clamp(min=0)
        return probabilities/probabilities.sum(-1, keepdim=True)

    def training_loss(self, output, targets):
        if self.loss == 'mse':
            return (output.squeeze(-1)-targets).square().mean()
        return -(self.gaussian_targets(targets)*output.log_softmax(-1)).sum(-1).mean()
