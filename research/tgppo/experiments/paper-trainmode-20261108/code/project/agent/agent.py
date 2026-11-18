"""Original networks and PPO-Clip, with explicit episode boundaries and modes."""
import time
import logging
import numpy as np
import torch
from torch.distributions import Categorical
from project.memory import Memory
from project.memory.memory import state_arrays


class ActorPolicy:
    """Deployment path: no Critic, optimizer, reward function or training buffer."""
    def __init__(self, actor_network, device='cpu'):
        self.actor_network = actor_network.to(device)
        self.device = device
        self.last_actor_seconds = self.last_critic_seconds = 0.0

    def _inputs(self, cands_state, mip_state, node_state, padding_mask=None):
        cands, mip, node = [torch.as_tensor(x, dtype=torch.float32, device=self.device)
                            for x in (cands_state, mip_state, node_state)]
        if cands.ndim == 2:
            cands = cands.unsqueeze(0)
        if mip.ndim == 1:
            mip = mip.unsqueeze(0)
        if node.ndim == 1:
            node = node.unsqueeze(0)
        mask = None if padding_mask is None else torch.as_tensor(
            padding_mask, dtype=torch.bool, device=self.device)
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if not all(torch.isfinite(x).all() for x in (cands, mip, node)):
            raise ValueError('Nonfinite policy input')
        if cands.shape[1] == 0 or (mask is not None and mask.all(dim=1).any()):
            raise ValueError('Empty candidate set')
        return cands, mip, node, mask

    @torch.no_grad()
    def _act(self, inputs, deterministic):
        cands, mip, node, mask = inputs
        self.actor_network.eval()  # Rollout and deployment follow the public author code.
        probs = self.actor_network(cands, node, mip, mask)
        if not torch.isfinite(probs).all():
            raise ValueError('Nonfinite action distribution')
        dist = Categorical(probs)
        action = probs.argmax(-1) if deterministic else dist.sample()
        if mask is not None and mask.gather(1, action[:, None]).any():
            raise ValueError('Selected padded candidate')
        return action, dist.log_prob(action)

    def act(self, cands_state, mip_state, node_state, padding_mask=None, deterministic=False):
        start = time.perf_counter()
        inputs = self._inputs(cands_state, mip_state, node_state, padding_mask)
        action, log_prob = self._act(inputs, deterministic)
        self.last_actor_seconds = time.perf_counter() - start
        self.last_critic_seconds = 0.0
        return int(action.item()), float(log_prob.item())


class Agent(ActorPolicy):
    def __init__(self, actor_network, actor_optimizer, critic_network, critic_optimizer,
                 policy_clip, entropy_weight, gamma, gae_lambda, batch_size, n_epochs,
                 device, state_dims, logger=None, policy_version=0):
        super().__init__(actor_network, device)
        self.actor_optimizer, self.critic_optimizer = actor_optimizer, critic_optimizer
        self.critic_network = critic_network.to(device)
        self.policy_clip, self.entropy_weight = policy_clip, entropy_weight
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.batch_size, self.n_epochs = batch_size, n_epochs
        self.state_dims = state_dims
        self.logger = logger or logging.getLogger(__name__)
        self.memory = Memory(batch_size, state_dims, device, self.logger)
        self.policy_version = policy_version
        self.update_counter = 0
        if {id(p) for p in self.actor_network.parameters()} & {
                id(p) for p in self.critic_network.parameters()}:
            raise ValueError('Stage2 baseline requires disjoint Actor/Critic parameters')

    def choose_action(self, cands_state, mip_state, node_state, padding_mask=None, deterministic=False):
        start = time.perf_counter()
        inputs = self._inputs(cands_state, mip_state, node_state, padding_mask)
        action, log_prob = self._act(inputs, deterministic)
        self.last_actor_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with torch.no_grad():
            self.critic_network.eval()
            cands, mip, node, mask = inputs
            value = self.critic_network(cands, node, mip, mask).reshape(-1)
        self.last_critic_seconds = time.perf_counter() - start
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite value prediction')
        return int(action.item()), float(value.item()), float(log_prob.item())

    def remember(self, **kwargs):
        self.memory.store(**kwargs)  # Fail closed, never log-and-drop a transition.

    def seal_episode(self, task_terminal, collector_truncated=False,
                     bootstrap_state=None, end_reason=None):
        value = 0.0
        if collector_truncated:
            if bootstrap_state is None:
                raise ValueError('Missing actual next decision state')
            bootstrap_state = state_arrays(**bootstrap_state, state_dims=self.state_dims)
            inputs = self._inputs(**bootstrap_state)
            with torch.no_grad():
                self.critic_network.eval()
                c, m, n, mask = inputs
                value = self.critic_network(c, n, m, mask).item()
        self.memory.seal_episode(task_terminal, collector_truncated, value,
                                 bootstrap_state, end_reason)

    @torch.no_grad()
    def _compute_gae(self, rewards, values, dones, bootstrap_value):
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), device=rewards.device)
        for t in reversed(range(len(rewards))):
            next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nonterminal * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
        return advantages, advantages + values

    def compute_targets(self):
        advantages, returns = [], []
        for episode in self.memory.episodes:
            if episode['policy_version'] != self.policy_version:
                raise ValueError('Stale policy trajectory')
            rows = episode['transitions']
            # Avoid launching one CUDA kernel per scalar GAE recurrence step.
            # Full trajectories and GAE stay on CPU; only targets are transferred.
            args = [torch.tensor([r[k] for r in rows], dtype=torch.float32)
                    for k in ['reward', 'value', 'done']]
            a, ret = self._compute_gae(*args, torch.tensor(
                episode['bootstrap_value'], dtype=torch.float32))
            advantages.append(a)
            returns.append(ret)
        if not advantages:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)
        return torch.cat(advantages).to(self.device), torch.cat(returns).to(self.device)

    def _learning_values(self, cands, node, mip, mask, batch, chunk):
        return self.critic_network(cands, node, mip, mask).reshape(-1)

    def _value_loss(self, values, targets):
        return torch.nn.functional.mse_loss(values, targets)

    @torch.no_grad()
    def _eval_ratios(self, cands, node, mip, mask, actions, old_log_probs):
        """Check rollout provenance without dropout or changing the training RNG."""
        previous_mode = self.actor_network.training
        try:
            self.actor_network.eval()
            dist = Categorical(self.actor_network(cands, node, mip, mask))
            return torch.exp(dist.log_prob(actions) - old_log_probs)
        finally:
            self.actor_network.train(previous_mode)

    @torch.no_grad()
    def _make_deployment_probe(self, cands, node, mip, mask):
        """At most eight first-chunk states; a monitor, not a validation set."""
        n = min(8, len(cands))
        inputs = [x[:n].detach().clone() for x in (cands, node, mip, mask)]
        previous_mode = self.actor_network.training
        try:
            self.actor_network.eval()
            before = self.actor_network(*inputs).detach().clone()
        finally:
            self.actor_network.train(previous_mode)
        return inputs, before

    @torch.no_grad()
    def _deployment_change(self, probe):
        inputs, before = probe
        previous_mode = self.actor_network.training
        try:
            self.actor_network.eval()
            after = self.actor_network(*inputs)
        finally:
            self.actor_network.train(previous_mode)
        mask = inputs[-1]
        count = (~mask).sum(1)
        uniform = (~mask).to(after.dtype) / count[:, None]
        return dict(deployment_probe_states=len(before),
            deployment_probe_scope='first chunk of first PPO minibatch; at most8 states',
            deployment_tv_before_after_mean=float(.5*(after-before).abs().sum(1).mean()),
            deployment_tv_to_uniform_before_mean=float(.5*(before-uniform).abs().sum(1).mean()),
            deployment_tv_to_uniform_after_mean=float(.5*(after-uniform).abs().sum(1).mean()),
            deployment_argmax_change_fraction=float((after.argmax(1)!=before.argmax(1)).float().mean()))

    def learn(self, progress_callback=None, microbatch_size=None):
        if self.memory.active is not None:
            raise RuntimeError('Unfinished episode')
        # Check stale data even when the buffer is too small for an update.
        raw_advantages, returns = self.compute_targets()
        n = len(raw_advantages)
        if n == 0:
            # Complete zero-decision episodes count toward500, with no fake update.
            self.memory.clear()
            return dict(updated=False, num_samples=0, num_updates=0,
                        reason='complete_episode_without_decisions', policy_version=self.policy_version)
        if self.actor_optimizer is None or self.critic_optimizer is None:
            raise RuntimeError('This collector has no training optimizers')
        advantages = (raw_advantages - raw_advantages.mean()) / raw_advantages.std(
            unbiased=False).clamp_min(1e-8)
        self.actor_network.train()  # Public author update mode; B0 dropout=.05.
        self.critic_network.train()
        totals = dict(actor_loss=0., critic_loss=0., entropy=0., clip_fraction=0.)
        updates = 0
        supervised_totals = dict(normalized_nll=0., weighted_loss=0., samples=0, batches=0)
        initial_ratio_error = None
        initial_train_ratio_error = 0.
        initial_dropout_clip_fraction = 0.
        probe = None
        last_progress_cpu = time.process_time()
        last_progress_wall = time.monotonic()
        for epoch in range(self.n_epochs):
            for batch in self.memory.get_batch_generator():
                size = len(batch['indices'])
                lengths = (~batch['mask']).sum(1).cpu().numpy()
                order = (np.argsort(lengths, kind='stable') if microbatch_size
                         else np.arange(size))
                chunk_size = int(microbatch_size or size)
                if chunk_size < 1:
                    raise ValueError('Positive microbatch size required')
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                # The outer minibatch and its optimizer step stay unchanged.
                # Sort only inside this batch, trim padding per chunk, weight
                # each mean by chunk_size / actual_outer_batch_size.
                begin = 0
                while begin < size:
                    largest = int(lengths[order[begin:min(size, begin+chunk_size)]].max())
                    bounded_size = max(1, min(chunk_size, self.attention_cell_budget // (largest*largest)))
                    chunk = order[begin:begin+bounded_size]
                    begin += len(chunk)
                    width = int(lengths[chunk].max())
                    cands = batch['cands'][chunk, :width].to(self.device)
                    mask = batch['mask'][chunk, :width].to(self.device)
                    node = batch['node_state'][chunk].to(self.device)
                    mip = batch['mip_state'][chunk].to(self.device)
                    ids = batch['indices'][chunk]
                    weight = len(chunk) / size
                    if probe is None:
                        probe = self._make_deployment_probe(cands, node, mip, mask)
                    probs = self.actor_network(cands, node, mip, mask)
                    dist = Categorical(probs)
                    ratios = torch.exp(dist.log_prob(batch['action'][chunk].to(self.device))
                                       - batch['log_prob'][chunk].to(self.device))
                    if updates == 0:
                        error = (ratios.detach()-1).abs().max().item()
                        initial_train_ratio_error = max(initial_train_ratio_error, error)
                        initial_dropout_clip_fraction += weight * float(
                            ((ratios.detach()-1).abs()>self.policy_clip).float().mean())
                        reference = self._eval_ratios(cands, node, mip, mask,
                            batch['action'][chunk].to(self.device),
                            batch['log_prob'][chunk].to(self.device))
                        reference_error = (reference-1).abs().max().item()
                        initial_ratio_error = max(initial_ratio_error or 0., reference_error)
                        if reference_error > 1e-4:
                            raise ValueError('Eval rollout/recompute probabilities disagree before update')
                    loss = -torch.min(ratios*advantages[ids], torch.clamp(
                        ratios,1-self.policy_clip,1+self.policy_clip)*advantages[ids]).mean()
                    entropy = dist.entropy().mean()
                    actor_loss = loss - self.entropy_weight*entropy
                    if not torch.isfinite(actor_loss):
                        raise ValueError('Nonfinite actor loss')
                    (weight*actor_loss).backward()
                    values = self._learning_values(cands,node,mip,mask,batch,chunk)
                    critic_loss = self._value_loss(values,returns[ids])
                    if not torch.isfinite(critic_loss):
                        raise ValueError('Nonfinite critic loss')
                    (weight*critic_loss).backward()
                    for key,val in dict(actor_loss=actor_loss,critic_loss=critic_loss,
                        entropy=entropy,clip_fraction=((ratios-1).abs()>self.policy_clip).float().mean()).items():
                        totals[key] += weight*float(val.detach())
                # Exactly once per logical PPO minibatch, after weighted PPO
                # microbatches and before the SAME Actor clipping/Adam step.
                supervision = getattr(self, 'expert_supervision', None)
                if supervision is not None:
                    auxiliary, info = supervision.loss(self.actor_network, self.device,
                        self.update_counter, updates)
                    auxiliary.backward()
                    for key in ('normalized_nll', 'weighted_loss', 'samples'):
                        supervised_totals[key] += info[key]
                    supervised_totals['batches'] += 1
                torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), .5,
                                               error_if_nonfinite=True)
                self.actor_optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), .5,
                                               error_if_nonfinite=True)
                self.critic_optimizer.step()
                updates += 1
                if progress_callback is not None and (
                        updates == 1 or time.process_time() - last_progress_cpu >= 15. or
                        time.monotonic() - last_progress_wall >= 15.):
                    progress_callback(dict(epoch=epoch + 1, epochs=self.n_epochs,
                        minibatches_completed=updates, samples=n,
                        expected_minibatches=self.n_epochs * ((n + self.batch_size - 1) // self.batch_size)))
                    last_progress_cpu = time.process_time()
                    last_progress_wall = time.monotonic()
        self.memory.clear()
        # Measure the deployed distribution, rather than mistaking dropout noise
        # for parameter learning. Reuse the first logical batch from this update.
        deployment_metrics = self._deployment_change(probe)
        self.update_counter += 1
        self.policy_version += 1
        result = {k: v / updates for k, v in totals.items()}
        result.update(updated=True, num_updates=updates, num_samples=n,
                      expert_normalized_nll=supervised_totals['normalized_nll']/updates,
                      expert_weighted_loss=supervised_totals['weighted_loss']/updates,
                      expert_samples=supervised_totals['samples'],
                      expert_batches=supervised_totals['batches'],
                      initial_ratio_max_error=initial_ratio_error,
                      initial_train_ratio_max_error=initial_train_ratio_error,
                      initial_dropout_clip_fraction=initial_dropout_clip_fraction,
                      actor_update_mode='train',
                      advantage_std_before_normalization=raw_advantages.std(unbiased=False).item(),
                      update_step=self.update_counter, policy_version=self.policy_version)
        result.update(deployment_metrics)
        self.logger.info('PPO update: %s', result)
        return result
