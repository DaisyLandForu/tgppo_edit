"""Same FP32 objectives and logical updates; prepared CPU inputs and CUDA gate."""
import gc

import numpy as np
import torch

import formal_train as original
from fast_data import MemoryData, OrderedBatches, sequence_batch_cpu, to_device
from paper_training import contrastive_loss, imitation_loss, sequence_logits


def all_finite(tensors):
    return bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())


def update(stage, embedding, sequence, optimizer, parts, grad_clip, device='cuda:0'):
    models = [embedding] if stage == 'cl' else [embedding, sequence]
    for model in models: model.train()
    optimizer.zero_grad(set_to_none=True)
    total = sum(count for count, _ in parts)
    weighted = None
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for count, cpu_batch in parts:
        batch = to_device(cpu_batch, device)
        loss = contrastive_loss(embedding, batch) if stage == 'cl' else imitation_loss(embedding, sequence, batch)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; no samples skipped')
        fraction = count / total
        (loss * fraction).backward()
        value = loss.detach() * fraction
        weighted = value if weighted is None else weighted + value
    parameters = [p for model in models for p in model.parameters()]
    if any(p.grad is None for p in parameters) or not all_finite([p.grad for p in parameters]):
        raise FloatingPointError('Missing/nonfinite gradient')
    norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
    optimizer.step()
    if not all_finite(parameters): raise FloatingPointError('Nonfinite updated parameter')
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    return {'loss': float(weighted), 'gradient_norm_before_clip': float(norm),
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if str(device).startswith('cuda') else 0,
            'microbatch': max(c for c, _ in parts), 'logical_batch': total}


def validate(stage, embedding, sequence, data, microbatch, prefetch, stopped, device='cuda:0'):
    embedding.eval()
    if sequence is not None: sequence.eval()
    numerator, total, states, correct = 0., 0, 0, 0
    with torch.inference_mode(), OrderedBatches(data, np.arange(len(data)), 0, microbatch,
                                               microbatch, prefetch) as batches:
        for _, ids, parts in batches:
            count, cpu_batch = parts[0]
            batch = to_device(cpu_batch, device)
            if stage == 'cl':
                value = contrastive_loss(embedding, batch)
                states += count
            else:
                logits = sequence_logits(embedding, sequence, batch)
                value = torch.nn.functional.cross_entropy(logits[batch['steps']],
                    batch['actions'][batch['steps']], reduction='sum') / count
                states += int(batch['steps'].sum())
                correct += int((logits.argmax(-1)[batch['steps']] == batch['actions'][batch['steps']]).sum())
            if not torch.isfinite(value): raise FloatingPointError('Nonfinite validation loss')
            numerator += float(value) * count
            total += count
            if stopped(): return None
    result = {'loss': numerator / total, 'examples': total, 'states': states, 'solver_seed': 4,
              'selection_metric': 'mean CL loss' if stage == 'cl' else 'mean trajectory-sum NLL'}
    if stage == 'il': result.update(per_state_nll=numerator / states, action_accuracy=correct / states)
    return result


def compare_tensors(a, b, tolerance=32 * torch.finfo(torch.float32).eps):
    errors = []
    def visit(a, b):
        if isinstance(a, torch.Tensor):
            assert a.shape == b.shape
            error = float((a - b).abs().max()) / max(1., float(a.abs().max()), float(b.abs().max()))
            if error > tolerance: raise ArithmeticError('Tensor scaled error {} > {}'.format(error, tolerance))
            errors.append(error)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a: visit(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b): visit(x, y)
        else: assert a == b
    visit(a, b)
    return max(errors, default=0.)


def cuda_gate(store, requested, event):
    """Discarded probes, before formal updates. OOM never skips training data."""
    saved = original.rng_state()
    attempts, chosen = [], None
    train, valid = MemoryData(store, 'train', 'il'), MemoryData(store, 'valid', 'il')
    pool = [data[i] for data in [train, valid] for i in data.hardest_windows()]
    items = [pool[i % len(pool)] for i in range(8)]
    # Combine the globally widest state with the longest real window. Padding
    # dimensions and packed token count bound a real microbatch's shapes.
    widest = max((MemoryData.sample(a, int(a[2].argmax())) for a in store.arrays if len(a[2])),
                 key=lambda s: len(s['cands_state_mat']))
    longest = max(pool, key=lambda seq: sum(len(s['cands_state_mat']) for s in seq))
    capacity = list(longest)
    while len(capacity) < 100: capacity.append(longest[0])
    capacity[min(range(len(capacity)), key=lambda i: len(capacity[i]['cands_state_mat']))] = widest
    candidates = [8, 4, 2, 1] if requested == 'auto' else [int(requested)]

    def models():
        original.seed_all(90210)
        return original.TreeGateNet(infimum=8).cuda(), original.GroupedMamba(8, 100).cuda()

    try:
        for microbatch in candidates:
            e1 = m1 = e2 = m2 = optimizer1 = optimizer2 = None
            try:
                e1, m1 = models()
                optimizer1 = torch.optim.AdamW(list(e1.parameters()) + list(m1.parameters()), lr=.001, weight_decay=.01)
                baseline = original.optimizer_update('il', e1, m1, optimizer1, items, 1, 1.)
                e2, m2 = models()
                optimizer2 = torch.optim.AdamW(list(e2.parameters()) + list(m2.parameters()), lr=.001, weight_decay=.01)
                parts = [(len(items[j:j + microbatch]), sequence_batch_cpu(items[j:j + microbatch]))
                         for j in range(0, len(items), microbatch)]
                result = update('il', e2, m2, optimizer2, parts, 1.)
                errors = [compare_tensors(e1.state_dict(), e2.state_dict()),
                          compare_tensors(m1.state_dict(), m2.state_dict()),
                          compare_tensors(optimizer1.state_dict(), optimizer2.state_dict())]
                if abs(baseline['loss'] - result['loss']) > 1e-4 * max(1., abs(baseline['loss'])):
                    raise ArithmeticError('Microbatch loss mismatch')
                # One synthetic worst-shape capacity probe; discarded afterward.
                capacity_parts = [(microbatch, sequence_batch_cpu([capacity] * microbatch))]
                measured = update('il', e2, m2, optimizer2, capacity_parts, 1.)
                attempt = {'microbatch': microbatch, 'status': 'passed',
                           'max_tensor_scaled_error': max(errors), 'capacity_peak_mib': measured['peak_allocated_mib'],
                           'capacity_probe_is_synthetic': True, 'real_windows_used_for_equivalence': len(items)}
                chosen = microbatch
            except (torch.cuda.OutOfMemoryError, ArithmeticError) as error:
                attempt = {'microbatch': microbatch, 'status': 'rejected', 'error': str(error)}
            finally:
                e1 = m1 = e2 = m2 = optimizer1 = optimizer2 = None
                gc.collect()
                torch.cuda.empty_cache()
            attempts.append(attempt)
            event({'event': 'cuda_microbatch_probe', **attempt})
            if chosen is not None: break
        if chosen is None: raise RuntimeError('No IL microbatch passed CUDA numerical/capacity checks')
        return {'selected_il_microbatch': chosen, 'gpu': torch.cuda.get_device_name(), 'attempts': attempts}
    finally:
        original.restore_rng(saved)
