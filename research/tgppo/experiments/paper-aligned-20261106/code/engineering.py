"""Small full-model update/recovery check. No SCIP tasks or performance claims."""
import copy
from pathlib import Path
import numpy as np
import torch
import server
from project import runtime as rt
from project.utils.protocol import write_json


def same_state(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def bridge(config, device, directory, actor_state=None):
    """CPU collection probabilities -> one real PPO minibatch -> exact replay.

    Uses four synthetic transitions at production model width/depth, batch4,
    one epoch, micro2. It is an interface check, not a training episode/pilot.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = dict(config, batch_size=4, n_epochs=1, learner_device=device)
    rt.seed_all(901)
    collector = rt.build_agent(cfg, training=False, device='cpu')
    if actor_state is not None:
        collector.actor_network.load_state_dict(actor_state)
    collector.memory.begin_episode('synthetic-bridge', 0)
    rng = np.random.RandomState(123)
    for i, length in enumerate([3, 5, 4, 7]):
        c = rng.randn(length, 25).astype('float32')
        n = rng.randn(8).astype('float32')
        m = rng.randn(53).astype('float32')
        action, value, logp = collector.choose_action(c, m, n)
        collector.remember(cands_state=c, mip_state=m, node_state=n,
            action=action, value=value, log_prob=logp,
            reward=[1., -.25, .75, -1.][i], done=i == 3)
    collector.seal_episode(task_terminal=True)
    rt.seed_all(901)
    agent = rt.build_agent(cfg, device=device)
    agent.actor_network.load_state_dict(collector.actor_network.state_dict())
    agent.critic_network.load_state_dict(collector.critic_network.state_dict())
    agent.memory.import_dict(collector.memory.export_dict())
    before = directory/'before.pt'
    rt.save_checkpoint(before, agent, cfg, {'scope': 'synthetic engineering check'})
    metrics = agent.learn(microbatch_size=2)
    expected = copy.deepcopy((agent.actor_network.state_dict(), agent.critic_network.state_dict(),
        agent.actor_optimizer.state_dict(), agent.critic_optimizer.state_dict()))
    rng_after = rt.rng_state()
    replay, _ = rt.load_agent(before, device=device)
    again = replay.learn(microbatch_size=2)
    actual = (replay.actor_network.state_dict(), replay.critic_network.state_dict(),
        replay.actor_optimizer.state_dict(), replay.critic_optimizer.state_dict())
    checks = dict(update_completed=metrics['updated'] and metrics['num_updates'] == 1,
        cpu_collection_ratio=metrics['initial_ratio_max_error'] < 1e-4,
        replay_ratio=again['initial_ratio_max_error'] < 1e-4,
        recovery_models_optimizers_exact=same_state(expected, actual),
        recovery_rng_exact=same_state(rng_after, rt.rng_state()))
    after = directory/'after.pt'
    rt.save_checkpoint(after, replay, cfg, {'scope': 'synthetic engineering check'})
    exported = directory/'actor.pt'
    server.export_actor(after, exported)
    original_critic = rt.Critic
    def forbidden(*args, **kwargs):
        raise AssertionError('Actor-only loader constructed Critic')
    rt.Critic = forbidden
    try:
        policy, _ = rt.load_actor(exported)
    finally:
        rt.Critic = original_critic
    checks['actor_only_cpu'] = all(p.device.type == 'cpu' for p in policy.actor_network.parameters())
    last = collector.memory.rows()[-1]
    expected_policy = rt.load_actor(after)[0]
    checks['actor_only_export_exact'] = policy.act(last['cands_state'], last['mip_state'],
        last['node_state'], deterministic=True) == expected_policy.act(last['cands_state'],
        last['mip_state'], last['node_state'], deterministic=True)
    result = dict(passed=all(checks.values()), checks=checks, metrics=metrics,
        device=device, synthetic_transitions=4, scip_solves=0,
        actual_optimizer_steps=4, committed_training_episodes=0,
        scope='synthetic full-model update/recovery; not method performance')
    write_json(directory/'result.json', result)
    if not result['passed']:
        raise RuntimeError('Update/recovery bridge failed: ' + str(checks))
    return result
