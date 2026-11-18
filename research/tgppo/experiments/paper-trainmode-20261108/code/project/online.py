"""Stage 5: raw-unit online PPO with calibrated, fully trainable Critics."""
import numpy as np
import torch
from project.agent.agent import Agent
from project.memory.memory import Memory


def context_arrays(remaining_time, frontier):
    t = np.array(remaining_time, dtype=np.float32).reshape(-1)
    f = np.array(frontier, dtype=np.float32, copy=True)
    if t.shape != (1,) or f.shape != (69,) or not np.isfinite(t).all() or not np.isfinite(f).all():
        raise ValueError('Malformed online Critic context')
    if not 0. <= t[0] <= 1.:
        raise ValueError('Remaining time outside [0,1]')
    return dict(remaining_time=t[0], frontier=f)


class OnlineMemory(Memory):
    schema_version = 3

    def store(self, remaining_time, frontier, **kwargs):
        extra = context_arrays(remaining_time, frontier)
        super().store(**kwargs)
        self.active['transitions'][-1].update(extra)

    def _validate_episode(self, episode):
        # This experiment always finishes a complete finite-time SCIP task.
        if episode['collector_truncated']:
            raise ValueError('Stage5 requires complete finite tasks, no collector truncation')
        super()._validate_episode(episode)
        for row in episode['transitions']:
            context_arrays(row['remaining_time'], row['frontier'])

    def get_batch_generator(self, batch_size=None):
        rows = self.rows()
        for batch in super().get_batch_generator(batch_size):
            for key in ('remaining_time', 'frontier'):
                batch[key] = torch.as_tensor(np.asarray([rows[i][key] for i in batch['indices']]),
                                              dtype=torch.float32, device=self.device)
            batch['remaining_time'] = batch['remaining_time'].reshape(-1, 1)
            yield batch


class OnlineAgent(Agent):
    online_context = True

    def __init__(self, *args, task_time_limit=120., **kwargs):
        super().__init__(*args, **kwargs)
        self.task_time_limit = float(task_time_limit)
        self.memory = OnlineMemory(self.batch_size, self.state_dims, self.device, self.logger)
        if not all(p.requires_grad for p in self.critic_network.parameters()):
            raise ValueError('Online Critic must train the full encoder')

    def choose_action(self, cands_state, mip_state, node_state, remaining_time, frontier,
                      padding_mask=None, deterministic=False):
        import time
        extra = context_arrays(remaining_time, frontier)
        start = time.perf_counter()
        inputs = self._inputs(cands_state, mip_state, node_state, padding_mask)
        action, log_prob = self._act(inputs, deterministic)
        self.last_actor_seconds = time.perf_counter()-start
        start = time.perf_counter()
        with torch.no_grad():
            self.critic_network.eval()
            c, m, n, mask = inputs
            t = torch.as_tensor([[extra['remaining_time']]], dtype=torch.float32, device=self.device)
            f = torch.as_tensor(extra['frontier'][None], device=self.device)
            value = self.critic_network(c, n, m, t, f, mask).reshape(-1)
        self.last_critic_seconds = time.perf_counter()-start
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite raw value')
        return int(action.item()), float(value.item()), float(log_prob.item())

    def seal_episode(self, task_terminal, collector_truncated=False,
                     bootstrap_state=None, end_reason=None):
        if not task_terminal or collector_truncated or bootstrap_state is not None:
            raise ValueError('Stage5 collector must finish its finite task')
        self.memory.seal_episode(True, False, 0., None, end_reason)

    def _learning_values(self, cands, node, mip, mask, batch, chunk):
        return self.critic_network(cands, node, mip, batch['remaining_time'][chunk],
                                   batch['frontier'][chunk], mask).reshape(-1)

    def _value_loss(self, values, targets):
        # Raw value -> GAE; standardized residual -> Critic loss only.
        return ((values-targets)/self.critic_network.target_std).square().mean()
