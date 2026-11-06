#!/usr/bin/env python3
"""Check real pilot trajectories through upstream HDF5 readers without training."""
import hashlib
import json
import sys
import traceback
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = ROOT / 'runs/data-pilot' / stamp
    directory.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'timestamp_utc': stamp, 'training_started': False,
              'scope': 'pilot-only HDF5/read/batch/loss diagnostic; no parameter updates or GPU execution'}
    try:
        from convert_pkl_to_h5 import convert_files
        from dataset_scip import clip_dataset_h5, transformer_dataset_h5, clip_collate, pad_collate
        from main import real_clip_process
        from network import TreeGateNet
        candidates = []
        for path in sorted((ROOT / 'runs/collection-pilot').glob('*/report.json')):
            item = json.loads(path.read_text())
            if item['status'] == 'passed' and item['instance'] == 'misc03':
                candidates.append((path, item))
        selected = {}
        for split, seed, prefix in [('train', 0, 0), ('valid', 4, 5)]:
            matches = [(p, r) for p, r in candidates if r['solver_seed'] == seed and r['random_prefix_steps'] == prefix]
            assert matches, split
            path, item = matches[-1]
            source = path.parent / 'data.pkl'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item['trajectory_sha256']
            selected[split] = {'solver_seed': seed, 'random_prefix_steps': prefix,
                               'trajectory': str(source.relative_to(ROOT)), 'sha256': item['trajectory_sha256']}
            for mode in ['clip', 'transformer']:
                convert_files([str(source)], Namespace(network_mode=mode, out_dir=str(directory), max_seq_length=100), split)
        train = clip_dataset_h5(str(directory / 'clip_train.h5'))
        valid = clip_dataset_h5(str(directory / 'clip_valid.h5'))
        sequences = transformer_dataset_h5(str(directory / 'transformer_train_100.h5'), max_seq_length=100)
        assert len(train) >= 2 and len(valid) > 0 and len(sequences) > 0
        # All features in this diagnostic batch come from the training pilot.
        batch = clip_collate([train[i] for i in range(len(train))])
        actions, states, features, mask = batch
        assert mask.any(dim=1).all() and torch.all(mask.sum(dim=1) > 1)
        assert mask[torch.arange(len(actions)), actions].all()
        assert torch.isfinite(features).all() and torch.isfinite(states).all()
        torch.manual_seed(0)
        model = TreeGateNet(infimum=8).eval()
        with torch.no_grad():
            original_loss = real_clip_process(batch, model, Namespace(device='cpu'))
            embeds, _ = model(features, states)
            anchor = embeds.masked_fill(~mask.unsqueeze(-1), float('-inf')).amax(dim=1)
            similarity = F.cosine_similarity(anchor.unsqueeze(1), embeds, dim=-1)
            positive = similarity[torch.arange(len(actions)), actions]
            negative_mask = mask.clone()
            negative_mask[torch.arange(len(actions)), actions] = False
            negative = (similarity * negative_mask).sum(dim=1) / negative_mask.sum(dim=1)
            # Add the same harmless constant 2 used in the upstream code so
            # that this comparison isolates padding rather than a constant shift.
            masked_loss = (2 - positive + negative).mean()
        sequence = sequences[0]
        padded = pad_collate([sequence])
        assert padded['states'].shape[-1] == 61
        assert padded['candidate_features'].shape[-1] == 25
        assert padded['states'].shape[1] <= 100
        assert torch.isfinite(padded['states']).all()
        report.update(status='passed', sources=selected, clip_samples={'train': len(train), 'valid': len(valid)},
                      sequence_count=len(sequences), actual_sequence_states=len(sequence['states']),
                      requested_max_sequence_states=100, full_100_state_batch_tested=False,
                      padded_candidates=int((~mask).sum()), min_candidates=int(mask.sum(dim=1).min()),
                      max_candidates=int(mask.sum(dim=1).max()), original_contrastive_loss=float(original_loss),
                      masked_contrastive_loss=float(masked_loss),
                      loss_difference=float(original_loss - masked_loss),
                      note='The upstream padding issue remains open; this diagnostic does not modify the training loss.')
    except Exception:
        report['error'] = traceback.format_exc()
    file = directory / 'report.json'
    file.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(file), 'error': report.get('error')}, indent=2))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
