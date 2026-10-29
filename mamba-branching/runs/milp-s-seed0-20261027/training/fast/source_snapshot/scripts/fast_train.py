#!/usr/bin/env python3
"""Audited fast execution of the frozen CL/IL protocol and checkpoint schema."""
import argparse
import atexit
import copy
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import uuid

import numpy as np
import torch

from formal_common import (ROOT, atomic_file, hardware, load_run, lock, read_json, sha, stamp, write_json)
from fast_data import MemoryStore, MemoryData, OrderedBatches
from fast_ops import update as prepared_update, validate as prepared_validate, cuda_gate
from dataset_scip import clip_collate
from network import GroupedMamba, TreeGateNet
from paper_training import contrastive_loss, imitation_loss, sequence_batch, sequence_logits

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(directory, pointer):
    pointer = read_json(pointer)
    path = directory / pointer['checkpoint']
    assert sha(path) == pointer['sha256'], 'Checkpoint digest mismatch'
    return torch.load(path, map_location='cpu')


def train_stage(directory, cfg, provenance, dataset_digest, stage, resume, event, stop_after, store, execution, runtime):
    options = dict(cfg['training'])
    options['il_microbatch'] = runtime['selected_il_microbatch']
    options['checkpoint_every_updates'] = execution['settings']['checkpoint_every_updates']
    stage_dir = directory / 'training' / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    latest = stage_dir / 'latest.json'
    epochs = options[stage + '_epochs']
    train, valid = MemoryData(store, 'train', stage), MemoryData(store, 'valid', stage)
    assert len(train) > 0 and len(valid) > 0
    # Phase initialization is independent of whether CL and IL share a process.
    seed_all(options['seed'] + (0 if stage == 'cl' else 100000))
    embedding = TreeGateNet(infimum=8).cuda()
    sequence = GroupedMamba(8, 100).cuda() if stage == 'il' else None
    optimizer = (torch.optim.Adam(embedding.parameters(), lr=1e-4) if stage == 'cl' else
                 torch.optim.AdamW(list(embedding.parameters()) + list(sequence.parameters()), lr=1e-3, weight_decay=.01))
    generator = np.random.default_rng(options['seed'] + (0 if stage == 'cl' else 100000))
    progress = {'epoch': 0, 'cursor': 0, 'updates': 0, 'order': None, 'order_sha256': None,
                'loss_numerator': 0., 'examples': 0, 'best_loss': None, 'best_epoch': None,
                'best_checkpoint': None, 'completed': False}

    last_save = time.monotonic()
    execution_hash = sha(directory / 'training/fast/execution.json')

    def save():
        nonlocal last_save
        save_started = time.monotonic()
        name = f'{stage}/checkpoints/step-{progress["updates"]:09d}-{uuid.uuid4().hex}.pt'
        relative = 'training/' + name
        state = {'schema': 1, 'stage': stage, 'fast_execution_sha256': execution_hash,
                 'effective_il_microbatch': options['il_microbatch'], 'config_sha256': provenance['config_sha256'],
                 'dataset_sha256': dataset_digest, 'run_sha256': sha(directory / 'run.json'),
                 'embedding': embedding.state_dict(), 'sequence': sequence.state_dict() if sequence is not None else None,
                 'optimizer': optimizer.state_dict(), 'progress': copy.deepcopy(progress),
                 'rng': rng_state(), 'sampler_rng': copy.deepcopy(generator.bit_generator.state)}
        with atomic_file(directory / relative) as stream:
            torch.save(state, stream)
        write_json(latest, {'checkpoint': relative, 'sha256': sha(directory / relative)})
        last_save = time.monotonic()
        event({'event': 'checkpoint_committed', 'stage': stage, 'update': progress['updates'],
               'checkpoint_seconds': last_save - save_started})
        return relative

    def export_stage():
        best = directory / progress['best_checkpoint']
        assert sha(best) == read_json(str(best) + '.sha256.json')['sha256'], 'Best checkpoint changed'
        selected = {'stage': stage, 'checkpoint': progress['best_checkpoint'], 'sha256': sha(best),
                    'epoch_1based': progress['best_epoch'], 'validation_loss': progress['best_loss'],
                    'completed_epochs': progress['epoch'], 'updates': progress['updates'],
                    'dataset_sha256': dataset_digest, 'validation_solver_seed': 4, 'tie_break': 'earliest epoch',
                    'fast_execution_sha256': execution_hash, 'effective_il_microbatch': options['il_microbatch']}
        write_json(stage_dir / 'selected.json', selected)
        return selected

    try:
        if latest.exists():
            assert resume == 'auto', 'Existing checkpoints require --resume auto'
            state = load_checkpoint(directory, latest)
            if 'fast_execution_sha256' in state:
                assert state['fast_execution_sha256'] == execution_hash, 'Fast execution profile changed'
                assert state['effective_il_microbatch'] == options['il_microbatch']
            else:
                event({'event': 'execution_migration', 'stage': stage,
                       'from_checkpoint': read_json(latest), 'to_execution_sha256': execution_hash,
                       'optimizer_rng_sampler_cursor_preserved': True})
            assert state['stage'] == stage and state['config_sha256'] == provenance['config_sha256']
            assert state['dataset_sha256'] == dataset_digest and state['run_sha256'] == sha(directory / 'run.json')
            embedding.load_state_dict(state['embedding'])
            if sequence is not None:
                sequence.load_state_dict(state['sequence'])
            optimizer.load_state_dict(state['optimizer'])
            progress = state['progress']
            if progress['best_checkpoint'] is not None:
                best = directory / progress['best_checkpoint']
                assert sha(best) == read_json(str(best) + '.sha256.json')['sha256'], 'Best checkpoint changed'
            generator.bit_generator.state = state['sampler_rng']
            restore_rng(state['rng'])
            event({'event': 'resume', 'stage': stage, 'progress': progress})
        else:
            if stage == 'il':
                selected = read_json(directory / 'training/cl/selected.json')
                assert selected['completed_epochs'] == options['cl_epochs']
                assert sha(directory / selected['checkpoint']) == selected['sha256']
                embedding.load_state_dict(torch.load(directory / selected['checkpoint'], map_location='cpu')['embedding'])
            save()
        if progress['completed']:
            return export_stage(), 0
        invocation_updates = 0
        microbatch = options['cl_microbatch' if stage == 'cl' else 'il_microbatch']
        batch_size = options['batch_size']
        order = None
        while progress['epoch'] < epochs:
            if progress['order'] is None:
                indices = generator.permutation(len(train))
                order_path = stage_dir / 'orders' / f'epoch-{progress["epoch"]:04d}.npy'
                if order_path.exists():
                    assert np.array_equal(indices, np.load(order_path)), 'Sampler order changed'
                else:
                    with atomic_file(order_path) as stream:
                        np.save(stream, indices)
                progress['order'] = str(order_path.relative_to(directory))
                progress['order_sha256'] = sha(order_path)
                order = indices
            elif order is None:
                assert sha(directory / progress['order']) == progress['order_sha256']
                order = np.load(directory / progress['order'], mmap_mode='r')
                assert len(order) == len(train)
            with OrderedBatches(train, order, progress['cursor'], batch_size, microbatch,
                                execution['settings']['prefetch']) as batches:
                for first, indices, parts in batches:
                    if STOP:
                        save()
                        return None, 75
                    assert first == progress['cursor'], 'Prefetch changed the committed cursor'
                    started = time.monotonic()
                    try:
                        outcome = prepared_update(stage, embedding, sequence, optimizer, parts, options['gradient_clip'])
                    except Exception:
                        event({'event': 'failed_batch', 'stage': stage, 'epoch': progress['epoch'],
                               'cursor': first, 'indices': [int(i) for i in indices], 'error': traceback.format_exc()})
                        raise
                    progress['cursor'] += len(indices)
                    progress['updates'] += 1
                    invocation_updates += 1
                    progress['loss_numerator'] += outcome['loss'] * len(indices)
                    progress['examples'] += len(indices)
                    event({'event': 'update', 'stage': stage, 'epoch_1based': progress['epoch'] + 1,
                           'update': progress['updates'], 'cursor': progress['cursor'],
                           'seconds': time.monotonic() - started, **outcome})
                    if (progress['updates'] % options['checkpoint_every_updates'] == 0 or STOP
                            or time.monotonic() - last_save >= execution['settings']['checkpoint_max_seconds']
                            or (stop_after and invocation_updates >= stop_after)):
                        save()
                    if STOP or (stop_after and invocation_updates >= stop_after):
                        return None, 75
            # Commit cursor=N before validation; interruption repeats only validation.
            save()
            metric = prepared_validate(stage, embedding, sequence, valid, microbatch,
                                       execution['settings']['prefetch'], lambda: STOP)
            if metric is None:
                return None, 75
            epoch = progress['epoch'] + 1
            event({'event': 'validation', 'stage': stage, 'epoch_1based': epoch, 'validation': metric,
                   'train_loss': progress['loss_numerator'] / progress['examples']})
            improved = progress['best_loss'] is None or metric['loss'] < progress['best_loss']
            # Save the immutable epoch checkpoint first. Only the atomic latest
            # pointer commits the best-checkpoint choice and next-epoch cursor.
            if improved:
                progress['best_loss'], progress['best_epoch'] = metric['loss'], epoch
            progress.update(epoch=epoch, cursor=0, order=None, order_sha256=None,
                            loss_numerator=0., examples=0, completed=epoch == epochs)
            relative = f'training/{stage}/epochs/epoch-{epoch:04d}-{uuid.uuid4().hex}.pt'
            if improved:
                progress['best_checkpoint'] = relative
            state = {'schema': 1, 'stage': stage, 'fast_execution_sha256': execution_hash,
                 'effective_il_microbatch': options['il_microbatch'], 'config_sha256': provenance['config_sha256'],
                     'dataset_sha256': dataset_digest, 'run_sha256': sha(directory / 'run.json'),
                     'embedding': embedding.state_dict(), 'sequence': sequence.state_dict() if sequence is not None else None,
                     'optimizer': optimizer.state_dict(), 'progress': copy.deepcopy(progress), 'validation': metric,
                     'rng': rng_state(), 'sampler_rng': copy.deepcopy(generator.bit_generator.state)}
            with atomic_file(directory / relative) as stream:
                torch.save(state, stream)
            digest = sha(directory / relative)
            write_json(str(directory / relative) + '.sha256.json', {'sha256': digest})
            write_json(latest, {'checkpoint': relative, 'sha256': digest})
            order = None
        return export_stage(), invocation_updates
    finally:
        pass  # RAM store is shared across phases; OrderedBatches owns its worker lifecycle.


def freeze_execution(directory, args):
    files = ['train_fast.sh', 'scripts/fast_launch.py', 'scripts/local_training.py',
             'scripts/fast_train.py', 'scripts/fast_data.py', 'scripts/fast_ops.py']
    settings = {'il_microbatch': args.il_microbatch, 'prefetch': args.prefetch,
                'checkpoint_every_updates': args.checkpoint_every_updates,
                'checkpoint_max_seconds': args.checkpoint_max_seconds, 'console_every': args.console_every}
    record = {'schema': 1, 'engine': 'ram-prefetch-v1', 'settings': settings,
              'sources': {name: sha(ROOT / name) for name in files},
              'protocol': 'FP32, same logical batch32, objectives, ordering, optimizer, epochs and seed4 selection',
              'migration': 'preserve baseline config/run/data hashes and checkpoint RNG/optimizer/cursor; record execution overrides'}
    path = directory / 'training/fast/execution.json'
    if path.exists():
        assert read_json(path) == record, 'Fast execution sources/settings changed; explicit migration required'
    else:
        for name in files:
            with atomic_file(directory / 'training/fast/source_snapshot' / name) as stream:
                stream.write((ROOT / name).read_bytes())
        write_json(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stage', choices=['all', 'cl', 'il'], default='all')
    parser.add_argument('--resume', choices=['auto', 'never'], default='auto')
    parser.add_argument('--stop-after-updates', type=int, default=0,
                        help='Checkpoint and exit 75 for controlled interruption testing')
    parser.add_argument('--il-microbatch', choices=['auto', '1', '2', '4', '8'], default='auto')
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--checkpoint-every-updates', type=int, default=250)
    parser.add_argument('--checkpoint-max-seconds', type=float, default=60.)
    parser.add_argument('--console-every', type=int, default=100)
    args = parser.parse_args()
    assert 0 <= args.prefetch <= 4 and args.checkpoint_every_updates > 0
    assert args.checkpoint_max_seconds > 0 and args.console_every > 0
    directory, cfg, provenance = load_run(args.run_dir)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.__version__ == '2.2.2+cu118'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(cfg['training']['seed'])
    manifest_path = directory / 'dataset/manifest.json'
    manifest = read_json(manifest_path)
    assert manifest['run_sha256'] == sha(directory / 'run.json')
    for row in manifest['trajectories']:
        assert sha(directory / row['path']) == row['sha256']
        assert sha(directory / row['index']) == row['index_sha256']
    dataset_digest = sha(manifest_path)
    attempt = stamp()
    log_path = directory / 'training/logs' / (attempt + '.jsonl')
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_stream = open(log_path, 'a', buffering=1)
    atexit.register(log_stream.close)

    def event(value):
        value.update(timestamp_utc=stamp(), attempt=attempt)
        log_stream.write(json.dumps(value, allow_nan=False) + '\n')
        if value['event'] != 'update' or value['update'] % args.console_every == 0:
            print(json.dumps(value, allow_nan=False), flush=True)

    with lock(directory / 'training/.lock'):
        execution = freeze_execution(directory, args)
        execution_hash = sha(directory / 'training/fast/execution.json')
        event({'event': 'start', 'hardware': {**hardware(), 'gpu': torch.cuda.get_device_name()},
               'config_sha256': provenance['config_sha256'], 'dataset_sha256': dataset_digest})
        try:
            store = MemoryStore(directory, event)
            event({'event': 'ram_ready', 'bytes': store.bytes})
            gate_path = directory / 'training/fast/il-microbatch.json'
            if gate_path.exists():
                runtime = read_json(gate_path)
                assert runtime['fast_execution_sha256'] == execution_hash
                if runtime['gpu'] != torch.cuda.get_device_name():
                    runtime = {**cuda_gate(store, str(runtime['selected_il_microbatch']), event),
                               'fast_execution_sha256': execution_hash}
                    write_json(directory / 'training/fast/gpu' / (attempt + '.json'), runtime)
                else:
                    event({'event': 'cuda_gate_reused', 'gpu': runtime['gpu'],
                           'selected_il_microbatch': runtime['selected_il_microbatch']})
            else:
                runtime = {**cuda_gate(store, args.il_microbatch, event),
                           'fast_execution_sha256': execution_hash}
                write_json(gate_path, runtime)
                write_json(directory / 'training/fast/gpu' / (attempt + '.json'), runtime)
            for stage in (['cl', 'il'] if args.stage == 'all' else [args.stage]):
                selected, count = train_stage(directory, cfg, provenance, dataset_digest, stage,
                                              args.resume, event, args.stop_after_updates, store, execution, runtime)
                if selected is None:
                    event({'event': 'checkpointed_stop', 'stage': stage, 'exit_code': 75})
                    return 75
                event({'event': 'stage_complete', 'selection': selected})
            cl_path, il_path = directory / 'training/cl/selected.json', directory / 'training/il/selected.json'
            if cl_path.exists() and il_path.exists():
                cl, il = read_json(cl_path), read_json(il_path)
                assert cl['completed_epochs'] == cfg['training']['cl_epochs']
                assert il['completed_epochs'] == cfg['training']['il_epochs']
                selected = {'frozen': True, 'profile': cfg['profile'], 'seed': cfg['training']['seed'],
                            'config_sha256': provenance['config_sha256'], 'dataset_sha256': dataset_digest,
                            'cl_epoch_1based': cl['epoch_1based'], 'il_epoch_1based': il['epoch_1based'],
                            'validation_solver_seed': 4, 'checkpoint': il['checkpoint'],
                            'checkpoint_sha256': il['sha256'], 'validation_loss': il['validation_loss'],
                            'tie_break': 'earliest epoch', 'test_used_for_selection': False,
                    'fast_execution_sha256': execution_hash,
                    'effective_il_microbatch': runtime['selected_il_microbatch']}
                target = directory / 'training/selected.json'
                if target.exists():
                    assert read_json(target) == selected, 'Frozen selection would change'
                else:
                    write_json(target, selected)
                event({'event': 'selection_frozen', 'selection': selected})
            return 0
        except Exception:
            event({'event': 'failed', 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
