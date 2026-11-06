import time
import torch
from torch.distributions import Categorical

class ActorPolicy:
    """Deployment path: no Critic, optimizer, reward function or training buffer."""
    def __init__(self, actor_network, device='cpu'):
        self.actor_network = actor_network.to(device)
        self.device = device
        self.last_actor_seconds = self.last_critic_seconds = 0.0

    def _inputs(self, cands_state, mip_state, node_state, padding_mask=None):
        cands, mip, node = [torch.as_tensor(x, dtype=torch.float32, device=self.device)
                            for x in (cands_state, mip_state, node_state)]
        if cands.ndim == 2:
            cands = cands.unsqueeze(0)
        if mip.ndim == 1:
            mip = mip.unsqueeze(0)
        if node.ndim == 1:
            node = node.unsqueeze(0)
        mask = None if padding_mask is None else torch.as_tensor(
            padding_mask, dtype=torch.bool, device=self.device)
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if not all(torch.isfinite(x).all() for x in (cands, mip, node)):
            raise ValueError('Nonfinite policy input')
        if cands.shape[1] == 0 or (mask is not None and mask.all(dim=1).any()):
            raise ValueError('Empty candidate set')
        return cands, mip, node, mask

    @torch.no_grad()
    def _act(self, inputs, deterministic):
        cands, mip, node, mask = inputs
        self.actor_network.eval()  # Also used during differentiable PPO recomputation.
        probs = self.actor_network(cands, node, mip, mask)
        if not torch.isfinite(probs).all():
            raise ValueError('Nonfinite action distribution')
        dist = Categorical(probs)
        action = probs.argmax(-1) if deterministic else dist.sample()
        if mask is not None and mask.gather(1, action[:, None]).any():
            raise ValueError('Selected padded candidate')
        return action, dist.log_prob(action)

    def act(self, cands_state, mip_state, node_state, padding_mask=None, deterministic=False):
        start = time.perf_counter()
        inputs = self._inputs(cands_state, mip_state, node_state, padding_mask)
        action, log_prob = self._act(inputs, deterministic)
        self.last_actor_seconds = time.perf_counter() - start
        self.last_critic_seconds = 0.0
        return int(action.item()), float(log_prob.item())
