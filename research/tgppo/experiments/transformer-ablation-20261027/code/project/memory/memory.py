"""Episode-preserving on-policy storage (stage 2; no new state features)."""
import copy
import numpy as np
import torch


def state_arrays(cands_state, mip_state, node_state, state_dims):
    result = {}
    for key, value, shape in [
        ('cands_state', cands_state, (None, state_dims['var_dim'])),
        ('mip_state', mip_state, (state_dims['mip_dim'],)),
        ('node_state', node_state, (state_dims['node_dim'],)),
    ]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.array(value, dtype=np.float32, copy=True)
        if value.ndim != len(shape) or any(
                n is not None and value.shape[i] != n for i, n in enumerate(shape)):
            raise ValueError('Invalid shape for ' + key)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite state: ' + key)
        result[key] = value
    if result['cands_state'].shape[0] < 1:
        raise ValueError('A transition needs at least one candidate')
    return result


class Memory:
    schema_version = 2

    def __init__(self, batch_size, state_dims, device, logger=None):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.batch_size, self.state_dims, self.device = batch_size, state_dims, device
        self.episodes = []
        self.active = None

    def clear(self):
        self.episodes = []
        self.active = None

    def begin_episode(self, episode_id, policy_version):
        if self.active is not None:
            raise RuntimeError('Previous episode has not been sealed')
        if any(e['episode_id'] == str(episode_id) for e in self.episodes):
            raise ValueError('Duplicate episode ID')
        self.active = dict(episode_id=str(episode_id), policy_version=int(policy_version),
                           transitions=[])

    def store(self, cands_state, mip_state, node_state, action, reward, done,
              value, log_prob):
        if self.active is None:
            raise RuntimeError('begin_episode must precede store')
        row = state_arrays(cands_state, mip_state, node_state, self.state_dims)
        if not 0 <= int(action) < len(row['cands_state']):
            raise ValueError('Invalid action')
        if not all(np.isfinite(float(v)) for v in (reward, value, log_prob)):
            raise ValueError('Nonfinite reward/value/log probability')
        if self.active['transitions'] and self.active['transitions'][-1]['done']:
            raise ValueError('Transition after terminal')
        row.update(action=int(action), reward=float(reward), done=bool(done),
                   value=float(value), log_prob=float(log_prob))
        self.active['transitions'].append(row)

    def seal_episode(self, task_terminal, collector_truncated=False,
                     bootstrap_value=0.0, bootstrap_state=None, end_reason=None):
        if self.active is None:
            raise RuntimeError('No active episode')
        episode = copy.deepcopy(self.active)
        episode.update(task_terminal=bool(task_terminal),
                       collector_truncated=bool(collector_truncated),
                       bootstrap_value=float(bootstrap_value),
                       bootstrap_state=bootstrap_state, end_reason=end_reason)
        self._validate_episode(episode)
        self.episodes.append(episode)
        self.active = None

    def abort_episode(self):
        self.active = None

    def _validate_episode(self, episode):
        terminal, truncated = episode['task_terminal'], episode['collector_truncated']
        if terminal == truncated:
            raise ValueError('Must be either a task terminal or a collector truncation')
        if not np.isfinite(episode['bootstrap_value']):
            raise ValueError('Nonfinite bootstrap')
        if terminal and episode['bootstrap_value'] != 0:
            raise ValueError('Terminal tasks cannot bootstrap')
        if truncated:
            if episode['bootstrap_state'] is None:
                raise ValueError('Truncation requires the actual next decision state')
            state_arrays(**episode['bootstrap_state'], state_dims=self.state_dims)
        rows = episode['transitions']
        for i, row in enumerate(rows):
            state_arrays(row['cands_state'], row['mip_state'], row['node_state'], self.state_dims)
            if not 0 <= row['action'] < len(row['cands_state']):
                raise ValueError('Imported action out of range')
            if not all(np.isfinite(row[k]) for k in ['reward', 'value', 'log_prob']):
                raise ValueError('Imported nonfinite transition')
            if row['done'] != (terminal and i == len(rows) - 1):
                raise ValueError('Terminal flags do not match episode boundary')

    def __len__(self):
        return sum(len(e['transitions']) for e in self.episodes)

    def is_empty(self):
        return len(self) == 0

    def export_dict(self):
        if self.active is not None:
            raise RuntimeError('Cannot export an unsealed episode')
        return dict(schema_version=self.schema_version, state_dims=self.state_dims.copy(),
                    episodes=copy.deepcopy(self.episodes), num_transitions=len(self))

    def import_dict(self, payload):
        if payload.get('schema_version') != self.schema_version:
            raise ValueError('Unsupported trajectory schema; episode boundaries required')
        if payload['state_dims'] != self.state_dims:
            raise ValueError('Trajectory state dimensions mismatch')
        incoming = copy.deepcopy(payload['episodes'])
        ids = {e['episode_id'] for e in self.episodes}
        for episode in incoming:
            if episode['episode_id'] in ids:
                raise ValueError('Duplicate imported episode')
            ids.add(episode['episode_id'])
            self._validate_episode(episode)
        if sum(len(e['transitions']) for e in incoming) != payload['num_transitions']:
            raise ValueError('Trajectory count mismatch')
        self.episodes.extend(incoming)

    def rows(self):
        if self.active is not None:
            raise RuntimeError('Cannot train on unfinished episode')
        return [r for e in self.episodes for r in e['transitions']]

    def get_batch_generator(self, batch_size=None):
        rows = self.rows()
        indices = np.random.permutation(len(rows))
        for start in range(0, len(rows), batch_size or self.batch_size):
            ids = indices[start:start + (batch_size or self.batch_size)]
            selected = [rows[i] for i in ids]
            states = [torch.as_tensor(r['cands_state'], device=self.device) for r in selected]
            cands = torch.nn.utils.rnn.pad_sequence(states, batch_first=True)
            lengths = torch.tensor([len(s) for s in states], device=self.device)
            mask = torch.arange(cands.shape[1], device=self.device)[None, :] >= lengths[:, None]
            batch = dict(indices=ids, cands=cands, mask=mask)
            for name in ['mip_state', 'node_state', 'action', 'value', 'log_prob']:
                array = np.asarray([r[name] for r in selected])
                dtype = torch.long if name == 'action' else torch.float32
                batch[name] = torch.as_tensor(array, dtype=dtype, device=self.device)
            yield batch
