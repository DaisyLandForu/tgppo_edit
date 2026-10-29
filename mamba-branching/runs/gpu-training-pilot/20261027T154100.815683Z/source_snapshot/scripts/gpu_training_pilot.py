#!/usr/bin/env python3
"""Bounded real-data GPU validation; never uses MILP-S test instances.

One 100-state training window is repeated 32 times ONLY for capacity testing.
This is not a dataset, convergence experiment, or formal training launcher.
"""
import copy
import hashlib
import json
import pickle
import random
import shutil
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def memory_measure(operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = operation()
    torch.cuda.synchronize()
    return {**result, 'seconds': time.monotonic() - started,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def compare_state(left, right, rtol=1e-5, atol=1e-7):
    differences = []
    def visit(a, b):
        if isinstance(a, torch.Tensor):
            a, b = a.cpu(), b.cpu()
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
            differences.append((a - b).abs().max().item())
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                visit(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                visit(x, y)
        else:
            assert a == b
    visit(left, right)
    return {'bitwise_equal': equal_state(left, right), 'max_absolute_difference': max(differences, default=0),
            'rtol': rtol, 'atol': atol}


def find_trajectory(instance, seed, min_samples):
    for path in sorted((ROOT / 'runs/collection-pilot').glob('*/report.json'), reverse=True):
        item = json.loads(path.read_text())
        if (item['status'] == 'passed' and item['instance'] == instance
                and item['solver_seed'] == seed and item['random_prefix_steps'] == 0
                and item['samples'] >= min_samples):
            data = path.parent / 'data.pkl'
            assert sha(data) == item['trajectory_sha256']
            values = list(pickle.loads(data.read_bytes()).values())
            return values, {'trajectory': str(data.relative_to(ROOT)), 'sha256': sha(data),
                            'samples': len(values), 'instance': instance, 'solver_seed': seed}
    raise ValueError(f'Collect a diagnostic {instance} / seed {seed} trajectory with >= {min_samples} samples first')


def evaluate(embedding, sequence, instance='misc03'):
    from env import MambaEvalBrancher
    from utils import init_params
    import pyscipopt as scip

    embedding.eval()
    sequence.eval()
    model = scip.Model()
    init_params(model, {'node_limit': 1000, 'time_limit': 30.0},
                {'heuristics': False, 'conflict_usesb': False, 'probing_bounds': False,
                 'checksol': False, 'reevalage': 0})
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', 0)
    model.readProblem(str(ROOT / 'data/milp-s/train_instances' / (instance + '.mps.gz')))
    # Diagnostic deliberately uses no cutoff; this is not a main-table result.
    neural_calls, max_position, max_tokens = 0, 0, 0

    def inspect_inputs(module, inputs, kwargs):
        nonlocal neural_calls, max_position, max_tokens
        positions = kwargs.get('position_ids', inputs[1] if len(inputs) > 1 else None)
        assert positions is not None and int(positions.max()) <= 24
        assert torch.isfinite(inputs[0]).all()
        neural_calls += 1
        max_position = max(max_position, int(positions.max()))
        max_tokens = max(max_tokens, inputs[0].shape[1])

    hook = sequence.register_forward_pre_hook(inspect_inputs, with_kwargs=True)
    brancher = MambaEvalBrancher(model, 'cuda:0', embedding, sequence,
                                {'var_dim': 25, 'node_dim': 8, 'mip_dim': 53},
                                False, 49, True, 1)
    model.includeBranchrule(brancher, 'MambaPilot', 'bounded GPU diagnostic', 999999, -1, 1)
    try:
        started = time.monotonic()
        with torch.inference_mode():
            model.optimize()  # No upstream broad except: native errors must fail the pilot.
        torch.cuda.synchronize()
        assert neural_calls > 0 and brancher.branch_count > 1
        result = {'instance': instance, 'split': 'train', 'cutoff': None,
                  'root_relpscost_steps': 1, 'history_steps': 24, 'deque_groups': 49,
                  'neural_calls': neural_calls, 'max_position_id': max_position,
                  'max_tokens': max_tokens, 'branch_count': brancher.branch_count,
                  'nodes': model.getNNodes(), 'status': model.getStatus(),
                  'gap': model.getGap(), 'primal_bound': model.getPrimalbound(),
                  'dual_bound': model.getDualbound(), 'seconds': time.monotonic() - started}
    finally:
        hook.remove()
        model.freeProb()
    return result


def main():
    from dataset_scip import clip_collate
    from network import TreeGateNet, GroupedMamba
    from paper_training import contrastive_loss, sequence_batch, sequence_logits, imitation_loss, checked_step
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    train_instances = json.loads((ROOT / 'configs/milp-s-instances.json').read_text())['train']
    parser.add_argument('--capacity-instance', choices=train_instances, default='lseu')
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = ROOT / 'runs/gpu-training-pilot' / stamp
    directory.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'timestamp_utc': stamp, 'host': socket.gethostname(),
              'python': sys.executable, 'formal_training_started': False,
              'scope': 'bounded real-data CL/IL, capacity, fixed-data resume and train-instance SCIP diagnostic',
              'fingerprints': {str(p.relative_to(ROOT)): sha(p) for p in
                               [Path(__file__), ROOT / 'setup/source-lock.json', ROOT / 'configs/paper-milp-s.json',
                                *sorted((ROOT / 'code').glob('*.py'))]}, 'checks': {}}
    for relative in report['fingerprints']:
        saved = directory / 'source_snapshot' / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, saved)

    def record(name, value):
        report['checks'][name] = value
        (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'check': name, 'result': value}, allow_nan=False), flush=True)

    try:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['gpu'] = torch.cuda.get_device_name()
        report['torch'] = torch.__version__
        samples, source = find_trajectory(args.capacity_instance, 0, 100)
        record('source', source)
        # Traverse the actual converter and HDF5 reader, including the author's
        # explicit >1-candidate filter before splitting the trajectory.
        from argparse import Namespace
        from convert_pkl_to_h5 import convert_files
        from dataset_scip import transformer_dataset_h5
        convert_files([str(ROOT / source['trajectory'])],
                      Namespace(network_mode='transformer', out_dir=str(directory), max_seq_length=100), 'train')
        h5_path = directory / 'transformer_train_100.h5'
        dataset = transformer_dataset_h5(str(h5_path), max_seq_length=100)
        entry = dataset[0]
        assert len(entry['states']) == 100
        window = []
        for i in range(100):
            valid = ~entry['candidate_masks'][i]
            window.append({'node_state': entry['states'][i, :8].numpy(),
                           'mip_state': entry['states'][i, 8:].numpy(),
                           'cands_state_mat': entry['candidate_features'][i, valid].numpy(),
                           'varRELpos': int(entry['actions'][i])})
        eligible = sum(len(s['cands_state_mat']) > 1 for s in samples)
        record('hdf5_pipeline', {'raw_states': len(samples), 'eligible_states': eligible,
                                'upstream_single_candidate_filter': len(samples) - eligible,
                                'sequence_lengths': [len(dataset[i]['states']) for i in range(len(dataset))],
                                'hdf5_sha256': sha(h5_path)})
        dataset.h5_file.close()
        # Real CL states; singleton-candidate states have no negative candidate.
        cl_samples = [s for s in samples if len(s['cands_state_mat']) > 1][:32]
        assert len(cl_samples) == 32
        raw = [[torch.tensor([int(s['varRELpos'])]), torch.tensor(s['node_state'], dtype=torch.float32),
                torch.tensor(s['mip_state'], dtype=torch.float32),
                torch.tensor(s['cands_state_mat'], dtype=torch.float32)] for s in cl_samples]
        cl_batch = tuple(x.cuda() for x in clip_collate(raw))
        embedding = TreeGateNet(infimum=8).cuda()
        sequence = GroupedMamba(8, 100).cuda()
        cl_optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-4)

        # Independently verify padding cannot change CL loss or gradients.
        actions, states, features, valid = cl_batch
        padded = (actions, states, torch.cat((features, torch.full_like(features[:, :7], 123)), dim=1),
                  torch.cat((valid, torch.zeros_like(valid[:, :7])), dim=1))
        def cl_value_grad(batch):
            embedding.zero_grad(set_to_none=True)
            loss = contrastive_loss(embedding, batch)
            loss.backward()
            return loss.detach(), [p.grad.clone() for p in embedding.parameters()]
        value, grads = cl_value_grad(cl_batch)
        padded_value, padded_grads = cl_value_grad(padded)
        torch.testing.assert_close(value, padded_value, rtol=1e-5, atol=1e-6)
        for a, b in zip(grads, padded_grads):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
        record('cl_padding_invariance', {'passed': True, 'loss_difference': abs((value - padded_value).item())})
        before = copy.deepcopy(embedding.state_dict())
        record('contrastive_update', memory_measure(lambda: checked_step(
            contrastive_loss(embedding, cl_batch), [embedding], cl_optimizer)))
        assert not equal_state(before, embedding.state_dict())

        # A short real window + a full window exercise unequal tail lengths.
        small = sequence_batch([window[:8], window[:3]], 'cuda:0')
        extra = sequence_batch([window[:8], window[:3]], 'cuda:0', extra_candidate_padding=7)
        extra['features'][~extra['valid']] = 123
        def il_value_grad(batch):
            embedding.zero_grad(set_to_none=True)
            sequence.zero_grad(set_to_none=True)
            loss = imitation_loss(embedding, sequence, batch)
            loss.backward()
            return loss.detach(), [p.grad.clone() for m in [embedding, sequence] for p in m.parameters()]
        value, grads = il_value_grad(small)
        padded_value, padded_grads = il_value_grad(extra)
        torch.testing.assert_close(value, padded_value, rtol=1e-5, atol=1e-6)
        for a, b in zip(grads, padded_grads):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
        # Batched causal outputs equal the separately evaluated unpadded window.
        with torch.no_grad():
            batched = sequence_logits(embedding, sequence, small)
            solo = sequence_batch([window[:3]], 'cuda:0')
            single = sequence_logits(embedding, sequence, solo)
            real = solo['valid'][0]
            torch.testing.assert_close(batched[1, :3, :single.shape[-1]][real], single[0][real],
                                       rtol=1e-4, atol=1e-5)
        record('il_padding_invariance', {'passed': True, 'candidate_padding': 7,
                                         'unequal_state_lengths': [8, 3], 'solo_vs_batch': 'passed'})
        modified = [dict(s) for s in window[:8]]
        modified[3]['varRELpos'] = (modified[3]['varRELpos'] + 1) % len(modified[3]['cands_state_mat'])
        original = sequence_batch([window[:8]], 'cuda:0')
        changed = sequence_batch([modified], 'cuda:0')
        with torch.no_grad():
            a, b = sequence_logits(embedding, sequence, original), sequence_logits(embedding, sequence, changed)
            mask = original['valid'][:, :4]
            torch.testing.assert_close(a[:, :4][mask], b[:, :4][mask], rtol=0, atol=0)
        record('causality', {'passed': True, 'changed_action_step': 3, 'current_and_prior_logits_unchanged': True})

        optimizer = torch.optim.AdamW(list(embedding.parameters()) + list(sequence.parameters()), lr=1e-3, weight_decay=.01)
        capacity = sequence_batch([window] * 32, 'cuda:0')
        record('capacity_shape', {'batch_size': 32, 'states_per_sequence': 100,
                                 'unique_windows': 1, 'repeated_for_memory_test': True,
                                 'candidate_min': min(map(len, [s['cands_state_mat'] for s in window])),
                                 'candidate_max': capacity['features'].shape[2],
                                 'tokens_per_sequence': capacity['token_counts'][0],
                                 'dtype': 'float32', 'candidate_truncation': False})
        before_embedding, before_sequence = copy.deepcopy(embedding.state_dict()), copy.deepcopy(sequence.state_dict())
        record('full_batch_update', memory_measure(lambda: checked_step(
            imitation_loss(embedding, sequence, capacity), [embedding, sequence], optimizer)))
        assert not equal_state(before_embedding, embedding.state_dict())
        assert not equal_state(before_sequence, sequence.state_dict())

        # Fixed-data replay includes optimizer, sampler cursor, every RNG.
        # Custom CUDA backward kernels may use floating point atomic reductions;
        # record numerical agreement separately from bitwise reproducibility.
        order = list(range(4))
        random.shuffle(order)
        state = {'embedding': embedding.state_dict(), 'sequence': sequence.state_dict(),
                 'optimizer': optimizer.state_dict(), 'cl_optimizer': cl_optimizer.state_dict(),
                 'rng': rng_state(), 'progress': {'il_updates': 1, 'order': order, 'cursor': 1},
                 'source': source, 'fingerprints': report['fingerprints']}
        checkpoint = directory / 'pilot.pt'
        torch.save(state, checkpoint)
        replay_batches = [window[:8], window[8:16], window[16:24], window[24:32]]

        def next_update(emb, seq, opt, progress):
            # Consume every saved RNG; only random training-window order affects the loss.
            draws = (random.random(), float(np.random.rand()), torch.rand(1).item(), torch.rand(1, device='cuda').item())
            index = progress['order'][progress['cursor']]
            batch = sequence_batch([replay_batches[index]], 'cuda:0')
            outcome = checked_step(imitation_loss(emb, seq, batch), [emb, seq], opt)
            progress['cursor'] += 1
            progress['il_updates'] += 1
            return draws, index, outcome['loss'], copy.deepcopy(progress)

        expected = next_update(embedding, sequence, optimizer, copy.deepcopy(state['progress']))
        restored = torch.load(checkpoint, map_location='cpu')
        embedding2, sequence2 = TreeGateNet(infimum=8).cuda(), GroupedMamba(8, 100).cuda()
        embedding2.load_state_dict(restored['embedding'])
        sequence2.load_state_dict(restored['sequence'])
        optimizer2 = torch.optim.AdamW(list(embedding2.parameters()) + list(sequence2.parameters()), lr=1e-3, weight_decay=.01)
        optimizer2.load_state_dict(restored['optimizer'])
        assert equal_state(embedding2.state_dict(), restored['embedding'])
        assert equal_state(sequence2.state_dict(), restored['sequence'])
        assert equal_state(optimizer2.state_dict(), restored['optimizer'])
        restore_rng(restored['rng'])
        actual = next_update(embedding2, sequence2, optimizer2, restored['progress'])
        assert expected == actual, (expected, actual)
        embedding_diff = compare_state(embedding.state_dict(), embedding2.state_dict())
        sequence_diff = compare_state(sequence.state_dict(), sequence2.state_dict())
        optimizer_diff = compare_state(optimizer.state_dict(), optimizer2.state_dict())
        record('fixed_data_resume', {'passed': True, 'next_loss': actual[2],
                                     'checkpoint_loaded_exactly': True, 'embedding_after_update': embedding_diff,
                                     'sequence_after_update': sequence_diff, 'optimizer_after_update': optimizer_diff,
                                     'next_progress': actual[3],
                                     'rng_draws_and_next_window': 'exact', 'checkpoint_sha256': sha(checkpoint),
                                     'scope': 'controlled pilot replay, not full formal multi-epoch resume'})
        record('scip_gpu_inference', memory_measure(lambda: evaluate(embedding2, sequence2)))
        record('scip_full_history', memory_measure(lambda: evaluate(embedding2, sequence2, 'lseu')))
        assert report['checks']['scip_full_history']['max_position_id'] == 24
        report['status'] = 'passed'
    except Exception:
        report['error'] = traceback.format_exc()
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(directory / 'report.json'),
                      'error': report.get('error')}, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
