"""Train-only expert CE regularizer; independent stateless minibatch sampling."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F


class ExpertSupervision:
    def __init__(self, config):
        from expert_data import OUTPUT, load_arrays, sha256
        self.weight = float(config['expert_supervision_weight'])
        self.batch_size = int(config['expert_supervision_batch_size'])
        self.seed = int(config['server_training_seed'])
        if not 0 < self.weight <= 1 or self.batch_size < 1:
            raise ValueError('Explicit positive expert coefficient and batch size required')
        if sha256(OUTPUT / 'expert.npz') != config['expert_dataset_sha256']:
            raise ValueError('Frozen expert data hash mismatch')
        self.data = load_arrays()
        train = np.flatnonzero(self.data['split'] == 'train')
        self.by_instance = {str(n): train[self.data['instances'][train] == n]
            for n in sorted(set(self.data['instances'][train]))}
        self.names = sorted(self.by_instance)
        if len(self.names) != 12 or len(train) != 1468:
            raise ValueError('Unexpected legal expert training split')
        lengths = np.diff(self.data['offsets'])
        if (lengths[train] < 2).any():
            raise ValueError('CE/log(K) requires at least two legal candidates')

    def sample_ids(self, update_counter, optimizer_minibatch):
        key = 'persistent-expert-v1:{}:{}:{}'.format(self.seed, update_counter, optimizer_minibatch)
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return np.asarray([int(rng.choice(self.by_instance[self.names[int(rng.randint(len(self.names)))]]))
            for _ in range(self.batch_size)], dtype=np.int64)

    def loss(self, actor, device, update_counter, optimizer_minibatch):
        from expert_data import batch
        ids = self.sample_ids(update_counter, optimizer_minibatch)
        if not np.all(self.data['split'][ids] == 'train'):
            raise ValueError('Validation states must never enter supervised gradients')
        x, node, mip, mask, labels = batch(self.data, ids, device)
        lengths = (~mask).sum(1)
        if (lengths < 2).any() or mask.gather(1, labels[:, None]).any():
            raise ValueError('Invalid legal expert label or candidate count')
        logits = actor.logits(x, node, mip, mask)
        normalized = (F.cross_entropy(logits, labels, reduction='none') / lengths.float().log()).mean()
        if not torch.isfinite(normalized):
            raise ValueError('Nonfinite supervised loss')
        return self.weight * normalized, dict(normalized_nll=float(normalized.detach()),
            weighted_loss=float((self.weight * normalized).detach()), samples=len(ids),
            batch_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
