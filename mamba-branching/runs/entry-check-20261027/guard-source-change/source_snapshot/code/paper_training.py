"""Padding-safe loss adapters for the pinned upstream Mamba-1 networks.

CL follows Eq. (2), without upstream's irrelevant +2 constant. IL follows
the paper's sum over real states and mean over trajectories, rather than
upstream's mean over states (a factor of 100 for full windows).
Candidate tokens and previous expert-action tokens
are packed before Mamba; only the END of a sequence is padded. No candidates
or states are dropped. This module is exercised by the diagnostic pilot,
not yet a formal training/resume entry point.
"""
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def contrastive_loss(model, batch):
    actions, states, candidates, valid = batch
    if not torch.isfinite(states).all() or not torch.isfinite(candidates).all():
        raise ValueError('Nonfinite CL input')
    if not (valid.sum(-1) > 1).all():
        raise ValueError('CL requires at least two real candidates per state')
    rows = torch.arange(len(actions), device=actions.device)
    if not valid[rows, actions].all():
        raise ValueError('Expert label selects padding')
    embeddings, _ = model(candidates, states)
    anchor = embeddings.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(-2)
    similarities = F.cosine_similarity(anchor.unsqueeze(-2), embeddings, dim=-1)
    positive = similarities[rows, actions]
    negative = valid.clone()
    negative[rows, actions] = False
    negative_mean = similarities.masked_fill(~negative, 0).sum(-1) / negative.sum(-1)
    return (-positive + negative_mean).mean()


def sequence_batch(sequences, device, extra_candidate_padding=0):
    """Prepare a batch from chronological lists of native expert samples.

    Explicit lengths avoid relying on the inconsistent upstream mask polarity.
    All padding is private to this adapter; real candidate order is unchanged.
    """
    import numpy as np

    if not sequences or any(not seq or len(seq) > 100 for seq in sequences):
        raise ValueError('Expected nonempty windows of at most 100 states')
    counts = [[len(s['cands_state_mat']) for s in seq] for seq in sequences]
    b, t = len(sequences), max(map(len, sequences))
    n = max(max(row) for row in counts) + extra_candidate_padding
    features = torch.zeros(b, t, n, 25)
    states = torch.zeros(b, t, 61)
    actions = torch.zeros(b, t, dtype=torch.long)
    valid = torch.zeros(b, t, n, dtype=torch.bool)
    steps = torch.zeros(b, t, dtype=torch.bool)
    positions = torch.zeros(b, t, n, dtype=torch.long)
    token_indices, position_ids = [], []
    for i, seq in enumerate(sequences):
        indices, ids, offset = [], [], 0
        for j, sample in enumerate(seq):
            count = counts[i][j]
            label = int(sample['varRELpos'])
            if not count or not 0 <= label < count:
                raise ValueError('Invalid candidate count or expert label')
            features[i, j, :count] = torch.as_tensor(sample['cands_state_mat'], dtype=torch.float32)
            states[i, j] = torch.as_tensor(np.concatenate((sample['node_state'], sample['mip_state'])), dtype=torch.float32)
            actions[i, j], valid[i, j, :count], steps[i, j] = label, True, True
            positions[i, j, :count] = torch.arange(offset, offset + count)
            indices.extend(range(j * n, j * n + count))
            ids.extend([j] * count)
            offset += count
            # Last expert action is a future token relative to the final target.
            if j + 1 < len(seq):
                indices.append(j * n + label)
                ids.append(j)
                offset += 1
        token_indices.append(torch.tensor(indices, device=device))
        position_ids.append(torch.tensor(ids, device=device))
    tensors = {'features': features, 'states': states, 'actions': actions,
               'valid': valid, 'steps': steps, 'positions': positions}
    if not torch.isfinite(features).all() or not torch.isfinite(states).all():
        raise ValueError('Nonfinite IL input')
    return {**{k: v.to(device) for k, v in tensors.items()}, 'token_indices': token_indices,
            'position_ids': pad_sequence(position_ids, batch_first=True),
            'token_counts': [len(x) for x in token_indices],
            'state_counts': list(map(len, sequences)), 'candidate_counts': counts}


def sequence_logits(embedding, model, batch):
    embedded, _ = embedding(batch['features'], batch['states'])
    flat = embedded.flatten(1, 2)
    tokens = pad_sequence([flat[i, idx] for i, idx in enumerate(batch['token_indices'])], batch_first=True)
    outputs = model(tokens, batch['position_ids']).mean(-1)
    b, t, n = batch['valid'].shape
    logits = outputs.gather(1, batch['positions'].reshape(b, t * n)).view(b, t, n)
    return logits.masked_fill(~batch['valid'], -torch.inf)


def imitation_loss(embedding, model, batch):
    logits = sequence_logits(embedding, model, batch)
    return F.cross_entropy(logits[batch['steps']], batch['actions'][batch['steps']], reduction='sum') / len(logits)


def checked_step(loss, models, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    checked = 0
    for model in models:
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f'Missing/nonfinite gradient: {name}')
            checked += 1
    optimizer.step()
    for model in models:
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite updated parameter')
    return {'loss': loss.item(), 'gradient_tensors_checked': checked}
