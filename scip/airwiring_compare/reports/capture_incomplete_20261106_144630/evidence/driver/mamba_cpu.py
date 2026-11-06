"""Inference-only CPU adapter for the frozen real-valued Mamba-1 block.

Uses the v2.2.2 unfused forward and selective_scan_ref equations:
https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/modules/mamba_simple.py
https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/ops/selective_scan_interface.py

The recurrence is scripted to avoid Python overhead per token and keeps only
one B x D x N state, rather than allocating B x D x L x N scan intermediates.
Original parameters, GroupedMamba positions, and history handling are retained.
No CUDA kernels, GRU substitution, training, or persistent inference cache.
"""
from types import MethodType

import torch
import torch.nn.functional as F


@torch.jit.script
def selective_scan_cpu(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                       B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                       z: torch.Tensor, delta_bias: torch.Tensor) -> torch.Tensor:
    """Real FP32 Mamba-1 scan, with input-dependent B/C and zero initial state."""
    delta = F.softplus(delta + delta_bias.unsqueeze(-1))
    state = A.new_zeros((u.shape[0], u.shape[1], A.shape[1]))
    output = torch.empty_like(u)
    for i in range(u.shape[2]):
        dt = delta[:, :, i].unsqueeze(-1)
        transition = torch.exp(dt * A)
        injection = (dt * B[:, :, i].unsqueeze(1)) * u[:, :, i].unsqueeze(-1)
        state = transition * state + injection
        output[:, :, i] = torch.sum(state * C[:, :, i].unsqueeze(1), dim=-1)
    return (output + u * D.unsqueeze(-1)) * F.silu(z)


def mamba_cpu_forward(self, hidden_states, inference_params=None):
    if inference_params is not None:
        raise ValueError('CPU aviation adapter recomputes each bounded history; no cached state')
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('CPU aviation adapter requires eval() and inference_mode()/no_grad()')
    if hidden_states.device.type != 'cpu' or hidden_states.dtype != torch.float32:
        raise ValueError('CPU aviation adapter requires FP32 CPU inputs')
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model or hidden_states.shape[1] < 1:
        raise ValueError('Expected nonempty [batch, tokens, d_model]')
    batch, length, _ = hidden_states.shape
    # Preserve the upstream projection layout, including its bias placement.
    flattened = hidden_states.permute(2, 0, 1).reshape(self.d_model, batch * length)
    xz = (self.in_proj.weight @ flattened).reshape(2 * self.d_inner, batch, length).permute(1, 0, 2)
    if self.in_proj.bias is not None:
        xz = xz + self.in_proj.bias.view(1, -1, 1)
    x, z = xz.chunk(2, dim=1)
    x = F.silu(self.conv1d(x)[..., :length])
    projected = self.x_proj(x.transpose(1, 2).reshape(batch * length, self.d_inner))
    dt, B, C = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    delta = (self.dt_proj.weight @ dt.t()).reshape(self.d_inner, batch, length).permute(1, 0, 2)
    B = B.reshape(batch, length, self.d_state).transpose(1, 2).contiguous()
    C = C.reshape(batch, length, self.d_state).transpose(1, 2).contiguous()
    y = selective_scan_cpu(x, delta, -torch.exp(self.A_log.float()), B, C,
                           self.D.float(), z, self.dt_proj.bias.float())
    return self.out_proj(y.transpose(1, 2))


def install_cpu_backend(sequence):
    """Override this instance's execution only; state_dict and upstream stay intact."""
    block = sequence.mamba
    if (type(block).__name__ != 'Mamba' or block.d_model != 8 or block.d_state != 64
            or block.d_inner != 16 or block.d_conv != 4 or block.activation != 'silu'):
        raise ValueError('Expected the frozen aviation Mamba-1 8/64/16/4 architecture')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in sequence.parameters()):
        raise ValueError('CPU backend requires all sequence weights on CPU in FP32')
    block.forward = MethodType(mamba_cpu_forward, block)
    return sequence
