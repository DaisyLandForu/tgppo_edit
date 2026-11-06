#!/usr/bin/env python3
"""CPU checks of inputs, prefetch and checkpoint migration; GPU gate runs on Custom.

Control-flow tests use the real TreeGate and a causal GRU stand-in for Mamba,
explicitly not a test of CUDA/Mamba kernels or formal training performance.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from benchmark_training_io import equal
from dataset_scip import clip_collate
from fast_data import MemoryStore, MemoryData, OrderedBatches, sequence_batch_cpu
from formal_common import ROOT, read_json, sha, write_json
from formal_data import IndexedData
import formal_train as baseline
import fast_train as fast
import fast_ops
from paper_training import sequence_batch as reference_sequence_batch


class CPUSequence(torch.nn.Module):
    def __init__(self, dim, length):
        super().__init__()
        self.positions = torch.nn.Embedding(length, dim)
        self.gru = torch.nn.GRU(dim, dim, batch_first=True)

    def forward(self, x, positions):
        return self.gru(x + self.positions(positions))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Diagnostic data, preferably staged locally')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    source, output = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'scope': __doc__, 'checks': {}, 'cuda_validated': False}
    def passed(name, result=True):
        report['checks'][name] = result
        print(name + ': ' + json.dumps(result), flush=True)
    print('Preloading RAM inputs', flush=True)
    store = MemoryStore(source)
    passed('ram_preload_bytes', store.bytes)
    rng = random.Random(123)
    for split in ['train', 'valid']:
        for mode in ['cl', 'il']:
            old, new = IndexedData(source, split, mode), MemoryData(store, split, mode)
            assert len(old) == len(new)
            ids = [rng.randrange(len(new)) for _ in range(128 if mode == 'cl' else 8)]
            assert all(equal(old[i], new[i]) for i in ids)
            if mode == 'cl':
                assert equal(clip_collate([old[i] for i in ids]), new.cl_batch(ids))
            else:
                for first in range(0, len(ids), 4):
                    seqs = [old[i] for i in ids[first:first + 4]]
                    assert equal(reference_sequence_batch(seqs, 'cpu'), sequence_batch_cpu(seqs))
            old.close()
            passed(split + '-' + mode + '-data-and-collator-exact')
    data = MemoryData(store, 'train', 'cl')
    order = np.random.default_rng(99).permutation(len(data))
    def stream(cursor, prefetch):
        with OrderedBatches(data, order, cursor, 32, 32, prefetch) as batches:
            return [(first, ids.copy(), parts) for first, ids, parts in batches]
    sequential = stream(0, 0)
    assert equal(sequential, stream(0, 2))
    committed = 64
    with OrderedBatches(data, order, committed, 32, 32, 2) as batches:
        first, ids, parts = next(batches)
        assert first == committed
        # Deliberately abandon prefetched batches; only the consumer owns cursor.
    assert equal(stream(committed, 2), sequential[2:])
    passed('prefetch-order-and-abandoned-lookahead-resume')
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    stream(0, 2)
    assert torch.equal(torch_state, torch.get_rng_state())
    assert equal(numpy_state, np.random.get_state()) and python_state == random.getstate()
    passed('prefetch-does-not-consume-training-rng')
    with patch.object(data, 'cl_batch', side_effect=ValueError('injected loader failure')):
        try: stream(0, 2)
        except ValueError: passed('prefetch-failure-propagates')
        else: raise AssertionError('Loader failure was swallowed')

    if args.data_only:
        report.update(status='passed', script_sha256=sha(Path(__file__)))
        write_json(output / 'verification.json', report)
        print('REPORT ' + str(output / 'verification.json'), flush=True)
        return

    manifest = read_json(source / 'dataset/manifest.json')
    def fixture(name):
        target = output / name
        files = ['config.json', 'plan.json', 'run.json', 'dataset/manifest.json']
        files += [r[k] for r in manifest['trajectories'] for k in ['path', 'index']]
        for name in files:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
        return target
    continuous, migrated = fixture('continuous'), fixture('migrated')
    cfg, provenance = read_json(source / 'config.json'), read_json(source / 'run.json')
    assert cfg['profile'] == 'diagnostic' and cfg['training']['cl_epochs'] <= 2 and cfg['training']['il_epochs'] <= 2
    settings = SimpleNamespace(il_microbatch='4', prefetch=2, checkpoint_every_updates=250,
                               checkpoint_max_seconds=60., console_every=100)
    execution = fast.freeze_execution(migrated, settings)
    runtime = {'selected_il_microbatch': 4}
    events = []
    def event(value): events.append(copy.deepcopy(value))
    original_update, original_validate = fast_ops.update, fast_ops.validate
    # Patch only these diagnostic processes. Production code still requires a
    # real GPU and real Mamba and runs its numerical/capacity gate automatically.
    with patch.object(torch.nn.Module, 'cuda', lambda self, *a, **k: self), \
         patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self), \
         patch.object(torch.cuda, 'manual_seed_all', lambda *a: None), \
         patch.object(torch.cuda, 'get_rng_state_all', lambda: []), \
         patch.object(torch.cuda, 'set_rng_state_all', lambda *a: None), \
         patch.object(torch.cuda, 'reset_peak_memory_stats', lambda: None), \
         patch.object(torch.cuda, 'max_memory_allocated', lambda: 0), \
         patch.object(torch.cuda, 'synchronize', lambda: None), \
         patch.object(baseline, 'GroupedMamba', CPUSequence), \
         patch.object(fast, 'GroupedMamba', CPUSequence), \
         patch.object(baseline, 'sequence_batch', lambda seqs, *a: reference_sequence_batch(seqs, 'cpu')), \
         patch.object(fast, 'prepared_update', lambda *a: original_update(*a, device='cpu')), \
         patch.object(fast, 'prepared_validate', lambda *a: original_validate(*a, device='cpu')):
        for stage in ['cl', 'il']:
            baseline.train_stage(continuous, cfg, provenance, sha(source / 'dataset/manifest.json'), stage,
                                 'auto', event, 0)
        result, code = baseline.train_stage(migrated, cfg, provenance, sha(source / 'dataset/manifest.json'),
                                           'cl', 'auto', event, 2)
        assert result is None and code == 75
        for stage, stop in [('cl', 0), ('il', 1), ('il', 0)]:
            result, code = fast.train_stage(migrated, cfg, provenance, sha(source / 'dataset/manifest.json'),
                                           stage, 'auto', event, stop, store, execution, runtime)
            assert (result is None and code == 75) if stop else result is not None
    assert any(x['event'] == 'execution_migration' for x in events)
    passed('baseline-checkpoint-migration-and-fast-il-stop-resume')
    for stage in ['cl', 'il']:
        a = baseline.load_checkpoint(continuous, continuous / 'training' / stage / 'latest.json')
        b = fast.load_checkpoint(migrated, migrated / 'training' / stage / 'latest.json')
        errors = [fast_ops.compare_tensors(a[field], b[field]) for field in
                  ['embedding', 'optimizer'] + (['sequence'] if stage == 'il' else [])]
        assert equal(a['sampler_rng'], b['sampler_rng']) and equal(a['rng'], b['rng'])
        for key in ['epoch', 'cursor', 'updates', 'completed', 'best_epoch']:
            assert a['progress'][key] == b['progress'][key], key
        passed(stage + '-continuous-vs-migrated-model-optimizer-rng-cursor', {'max_scaled_error': max(errors)})
    report.update(status='passed', script_sha256=sha(Path(__file__)))
    write_json(output / 'verification.json', report)
    print('REPORT ' + str(output / 'verification.json'), flush=True)


if __name__ == '__main__':
    main()
