"""Immutable RAM trajectories, vectorized collators and ordered CPU prefetch."""
from bisect import bisect_right
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

import h5py
import numpy as np
import torch

from formal_common import read_json


class MemoryStore:
    def __init__(self, directory, event=None):
        self.directory = directory
        self.records = read_json(directory / 'dataset/manifest.json')['trajectories']
        self.arrays = []
        started = time.monotonic()
        for number, row in enumerate(self.records, 1):
            with h5py.File(directory / row['path'], 'r') as f:
                states, actions, counts = f['states'][:], f['actions'][:], f['counts'][:]
                pieces = f['candidates'][:]
            offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
            candidates = (np.concatenate(pieces).reshape(-1, 25).astype(np.float32, copy=False)
                          if len(pieces) else np.empty((0, 25), np.float32))
            indices = np.load(directory / row['index'])
            assert states.shape == (len(actions), 61) and offsets[-1] == len(candidates)
            assert np.isfinite(states).all() and np.isfinite(candidates).all()
            assert ((actions >= 0) & (actions < counts)).all()
            assert len(indices) == row['eligible_states'] and (counts[indices] > 1).all()
            self.arrays.append((states, actions, counts, offsets, candidates, indices))
            if event and (number % 50 == 0 or number == len(self.records)):
                event({'event': 'ram_preload', 'files': number, 'total': len(self.records),
                       'seconds': time.monotonic() - started})
        self.bytes = sum(a.nbytes for record in self.arrays for a in record)


class MemoryData:
    def __init__(self, store, split, mode):
        self.store, self.mode = store, mode
        self.records = [i for i, r in enumerate(store.records) if r['split'] == split]
        self.ends, count = [], 0
        for i in self.records:
            row = store.records[i]
            count += row['eligible_states'] if mode == 'cl' else row['windows']
            self.ends.append(count)
        self.count = count

    def __len__(self):
        return self.count

    def locate(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        file_id = bisect_right(self.ends, index)
        local = index - (self.ends[file_id - 1] if file_id else 0)
        return self.store.arrays[self.records[file_id]], local

    @staticmethod
    def sample(arrays, raw):
        states, actions, counts, offsets, candidates, indices = arrays
        return {'node_state': states[raw, :8], 'mip_state': states[raw, 8:],
                'varRELpos': int(actions[raw]),
                'cands_state_mat': candidates[offsets[raw]:offsets[raw + 1]]}

    def __getitem__(self, index):
        arrays, local = self.locate(index)
        indices = arrays[-1]
        if self.mode == 'il':
            return [self.sample(arrays, int(raw)) for raw in indices[local * 100:(local + 1) * 100]]
        item = self.sample(arrays, int(indices[local]))
        return [torch.tensor([item['varRELpos']], dtype=torch.long),
                torch.from_numpy(item['node_state']), torch.from_numpy(item['mip_state']),
                torch.from_numpy(item['cands_state_mat'])]

    def cl_batch(self, indices):
        samples = []
        for index in indices:
            arrays, local = self.locate(int(index))
            samples.append(self.sample(arrays, int(arrays[-1][local])))
        counts = np.array([len(x['cands_state_mat']) for x in samples])
        features = np.zeros((len(samples), int(counts.max()), 25), np.float32)
        for i, sample in enumerate(samples):
            features[i, :counts[i]] = sample['cands_state_mat']
        states = np.concatenate((np.stack([x['node_state'] for x in samples]),
                                 np.stack([x['mip_state'] for x in samples])), axis=-1)
        return tuple(torch.from_numpy(a) for a in
                     (np.array([x['varRELpos'] for x in samples], dtype=np.int64), states,
                      features, np.arange(features.shape[1])[None, :] < counts[:, None]))

    def hardest_windows(self):
        assert self.mode == 'il'
        dense, longest, shortest = (-1, 0), (-1, 0), (101, 0)
        first = 0
        for file_id in self.records:
            row, arrays = self.store.records[file_id], self.store.arrays[file_id]
            counts, indices = arrays[2], arrays[-1]
            for j in range(row['windows']):
                selected = counts[indices[j * 100:(j + 1) * 100]]
                index = first + j
                if int(selected.max()) > dense[0]: dense = int(selected.max()), index
                tokens = int(selected.sum()) + len(selected) - 1
                if tokens > longest[0]: longest = tokens, index
                if len(selected) < shortest[0]: shortest = len(selected), index
            first += row['windows']
        return [dense[1], longest[1], shortest[1]]


def sequence_batch_cpu(sequences):
    """Exactly the reference token ordering, with NumPy array operations."""
    assert sequences and all(0 < len(s) <= 100 for s in sequences)
    counts = [[len(s['cands_state_mat']) for s in seq] for seq in sequences]
    b, t, n = len(sequences), max(map(len, sequences)), max(max(c) for c in counts)
    features = np.zeros((b, t, n, 25), np.float32)
    states = np.zeros((b, t, 61), np.float32)
    actions = np.zeros((b, t), np.int64)
    valid, steps = np.zeros((b, t, n), bool), np.zeros((b, t), bool)
    positions = np.zeros((b, t, n), np.int64)
    token_indices, ids_list = [], []
    for i, seq in enumerate(sequences):
        c = np.array(counts[i], dtype=np.int64)
        labels = np.array([s['varRELpos'] for s in seq], dtype=np.int64)
        assert ((c > 0) & (labels >= 0) & (labels < c)).all()
        length, total = len(c), int(c.sum())
        rows = np.repeat(np.arange(length), c)
        columns = np.arange(total) - np.repeat(np.cumsum(c) - c, c)
        starts = np.cumsum(c + 1) - c - 1
        candidate_positions = starts[rows] + columns
        features[i, rows, columns] = np.concatenate([s['cands_state_mat'] for s in seq])
        states[i, :length] = np.concatenate((np.stack([s['node_state'] for s in seq]),
                                              np.stack([s['mip_state'] for s in seq])), axis=-1)
        actions[i, :length], steps[i, :length] = labels, True
        valid[i, rows, columns], positions[i, rows, columns] = True, candidate_positions
        tokens = np.empty(total + length - 1, np.int64)
        ids = np.empty_like(tokens)
        tokens[candidate_positions], ids[candidate_positions] = rows * n + columns, rows
        action_positions = starts[:-1] + c[:-1]
        tokens[action_positions] = np.arange(length - 1) * n + labels[:-1]
        ids[action_positions] = np.arange(length - 1)
        token_indices.append(torch.from_numpy(tokens))
        ids_list.append(torch.from_numpy(ids))
    tensors = dict(features=features, states=states, actions=actions, valid=valid, steps=steps,
                   positions=positions)
    return {**{k: torch.from_numpy(v) for k, v in tensors.items()}, 'token_indices': token_indices,
            'position_ids': torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True),
            'token_counts': list(map(len, token_indices)), 'state_counts': list(map(len, sequences)),
            'candidate_counts': counts}


def prepare(data, indices, microbatch):
    parts = []
    for first in range(0, len(indices), microbatch):
        selected = indices[first:first + microbatch]
        batch = (data.cl_batch(selected) if data.mode == 'cl' else
                 sequence_batch_cpu([data[int(i)] for i in selected]))
        parts.append((len(selected), batch))
    return parts


class OrderedBatches:
    """Bounded CPU-only lookahead; consumed cursor belongs exclusively to trainer."""
    def __init__(self, data, order, cursor, batch_size, microbatch, prefetch=2):
        assert prefetch >= 0
        self.data, self.order = data, order
        self.next_cursor, self.batch_size, self.microbatch = cursor, batch_size, microbatch
        self.limit = prefetch
        self.executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.pending = deque()

    def _fill(self):
        while self.executor and len(self.pending) < self.limit and self.next_cursor < len(self.order):
            first = self.next_cursor
            ids = self.order[first:first + self.batch_size].copy()
            self.next_cursor += len(ids)
            self.pending.append((first, ids, self.executor.submit(prepare, self.data, ids, self.microbatch)))

    def __enter__(self):
        self._fill()
        return self

    def __iter__(self): return self

    def __next__(self):
        if self.executor:
            if not self.pending: raise StopIteration
            first, ids, future = self.pending.popleft()
            result = future.result()
            self._fill()
            return first, ids, result
        if self.next_cursor >= len(self.order): raise StopIteration
        first = self.next_cursor
        ids = self.order[first:first + self.batch_size]
        self.next_cursor += len(ids)
        return first, ids, prepare(self.data, ids, self.microbatch)

    def __exit__(self, *args):
        for _, _, future in self.pending: future.cancel()
        if self.executor: self.executor.shutdown(wait=True, cancel_futures=True)


def to_device(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(to_device(v, device) for v in value)
    if isinstance(value, list): return [to_device(v, device) for v in value]
    return value
