"""Zero-initialized score shortcuts; the original Actor remains untouched."""
import torch
from torch import nn
from torch.nn import functional as F
from .actor import Actor


MODES = ('none', 'score', 'candidate')


class ResidualActor(Actor):
    def __init__(self, *args, residual_mode, **kwargs):
        if residual_mode not in MODES[1:]:
            raise ValueError('ResidualActor requires score or candidate mode')
        super().__init__(*args, **kwargs)
        self.residual_mode = residual_mode
        # No RNG consumption: all original Actor/Critic initial weights and the
        # subsequent training RNG stream match the unmodified constructor.
        self.residual_weight = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, cands_state_mat, node_state, mip_state, padding_mask=None):
        batch, length, _ = cands_state_mat.shape
        candidate_features = self.var_embedding(cands_state_mat)
        tree_features = self.tree_embedding(torch.cat([node_state, mip_state], dim=-1))
        combined = torch.cat([
            candidate_features, tree_features.unsqueeze(1).expand(-1, length, -1)
        ], dim=-1)
        fused = self.global_embedding(combined).transpose(0, 1)
        mask = padding_mask
        if mask is None:
            mask = torch.zeros((batch, length), dtype=torch.bool, device=fused.device)
        encoded = self.transformer(fused, src_key_padding_mask=mask).transpose(0, 1)
        matched = self.tree_refinement(tree_features, encoded, mask)
        logits = self.output_layer(matched, tree_features)
        shortcut = matched
        if self.residual_mode == 'candidate':
            shortcut = shortcut + candidate_features
        logits = logits + torch.matmul(shortcut, self.residual_weight)
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask, float('-inf'))
        return F.softmax(logits, dim=-1)


def make_actor(arch, mode='none'):
    if mode not in MODES:
        raise ValueError('Unknown candidate residual mode: ' + str(mode))
    if mode == 'none':
        return Actor(**arch)
    return ResidualActor(**arch, residual_mode=mode)
