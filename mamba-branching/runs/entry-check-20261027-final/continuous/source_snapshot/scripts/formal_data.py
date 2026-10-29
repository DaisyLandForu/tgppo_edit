"""Streaming native trajectories and bounded-memory indexed training data."""
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch

from formal_common import read_json


class TrajectorySink:
    def __init__(self, path):
        self.file = h5py.File(path, 'w')
        self.states = self.file.create_dataset('states', (0, 61), maxshape=(None, 61), chunks=(128, 61), dtype='f4')
        self.actions = self.file.create_dataset('actions', (0,), maxshape=(None,), chunks=(128,), dtype='i8')
        self.counts = self.file.create_dataset('counts', (0,), maxshape=(None,), chunks=(128,), dtype='i4')
        self.candidates = self.file.create_dataset('candidates', (0,), maxshape=(None,), chunks=(128,),
                                                   dtype=h5py.vlen_dtype(np.dtype('float32')))
        self.buffer = []
        self.total = 0

    def __setitem__(self, index, sample):
        candidates = np.asarray(sample['cands_state_mat'], dtype=np.float32)
        states = np.concatenate([sample['node_state'], sample['mip_state']]).astype('float32')
        label = sample['varRELpos']
        if (candidates.ndim != 2 or candidates.shape[1] != 25 or states.shape != (61,)
                or int(label) != label or not 0 <= label < len(candidates)
                or not np.isfinite(candidates).all() or not np.isfinite(states).all()):
            raise ValueError(f'Invalid native expert sample {index}')
        self.buffer.append((states, int(label), candidates.copy()))
        if len(self.buffer) >= 128:
            self.flush()

    def __len__(self):
        return self.total + len(self.buffer)

    def flush(self):
        if not self.buffer:
            return
        end = self.total + len(self.buffer)
        for dataset in [self.states, self.actions, self.counts, self.candidates]:
            dataset.resize(end, axis=0)
        self.states[self.total:end] = np.stack([s[0] for s in self.buffer])
        self.actions[self.total:end] = [s[1] for s in self.buffer]
        self.counts[self.total:end] = [len(s[2]) for s in self.buffer]
        for i, (_, _, candidates) in enumerate(self.buffer, self.total):
            self.candidates[i] = candidates.reshape(-1)
        self.total = end
        self.buffer.clear()
        self.file.flush()

    def close(self):
        self.flush()
        self.file.close()


class IndexedData:
    """No DataLoader workers/prefetch: cursor is exactly the committed order."""
    def __init__(self, directory, split, mode):
        self.directory = Path(directory)
        manifest = read_json(self.directory / 'dataset/manifest.json')
        self.records = [row for row in manifest['trajectories'] if row['split'] == split]
        self.mode = mode
        self.ends, count = [], 0
        for row in self.records:
            count += row['eligible_states'] if mode == 'cl' else row['windows']
            self.ends.append(count)
        self.count = count
        self.files, self.indices = OrderedDict(), OrderedDict()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        file_id = bisect_right(self.ends, index)
        local = index - (self.ends[file_id - 1] if file_id else 0)
        record = self.records[file_id]
        if file_id not in self.files:
            self.files[file_id] = h5py.File(self.directory / record['path'], 'r')
            self.indices[file_id] = np.load(self.directory / record['index'], mmap_mode='r')
            if len(self.files) > 8:
                old, handle = self.files.popitem(last=False)
                handle.close()
                del self.indices[old]
        self.files.move_to_end(file_id)
        handle, indices = self.files[file_id], self.indices[file_id]

        def sample(raw):
            raw = int(raw)
            state = handle['states'][raw]
            return {'node_state': state[:8], 'mip_state': state[8:],
                    'varRELpos': int(handle['actions'][raw]),
                    'cands_state_mat': handle['candidates'][raw].reshape(-1, 25)}

        if self.mode == 'cl':
            item = sample(indices[local])
            return [torch.tensor([item['varRELpos']], dtype=torch.long),
                    torch.from_numpy(item['node_state']), torch.from_numpy(item['mip_state']),
                    torch.from_numpy(item['cands_state_mat'])]
        return [sample(raw) for raw in indices[local * 100:(local + 1) * 100]]

    def close(self):
        for handle in self.files.values():
            handle.close()
