"""Synthetic target checks. No SCIP optimization, real episodes or expert solves."""
import copy
from pathlib import Path
import numpy as np
import torch
from project import runtime as rt
from engineering import bridge, same_state
from paper import configuration, state_digest, write, schedule, fp


def target_checks(c, seed, directory, device):
    directory = Path(directory)
    results = {}
    traces = []
    for variant in ('b0', 'mlp', 'mlp_bc'):
        rt.seed_all(seed)
        cfg = configuration(c, variant, seed, device)
        agent = rt.build_agent(cfg, device=device)
        traces.append(dict(critic=state_digest(agent.critic_network.state_dict()),
                           actor=state_digest(agent.actor_network.state_dict()), rng=rt.rng_state()))
        assert isinstance(agent.actor_optimizer, torch.optim.AdamW)
        assert isinstance(agent.critic_optimizer, torch.optim.AdamW)
        for opt, lr in [(agent.actor_optimizer, .00024), (agent.critic_optimizer, .00012)]:
            group = opt.param_groups[0]
            assert group['lr'] == lr and group['weight_decay'] == .01
            assert group['betas'] == (.9, .999) and group['eps'] == 1e-8
        del agent
        results[variant] = bridge(cfg, device, directory/variant)
    assert len({x['critic'] for x in traces}) == 1
    assert traces[1]['actor'] == traces[2]['actor']
    assert same_state(traces[0]['rng'], traces[1]['rng']) and same_state(traces[1]['rng'], traces[2]['rng'])
    plans = schedule(seed, c)
    expected = {(n, s) for n in c['train'] for s in range(5)}
    assert len(plans) == 500
    assert all(set(plans[i:i+125]) == expected for i in range(0, 500, 125))
    result = dict(passed=True, device=device, seed=seed, independent_streams_paired=True,
                  optimizer='AdamW', exact_resume=True, all125_pairs_in_each_cycle=True,
                  schedule_sha256=fp(plans), bridges=results, scip_solves=0, real_episodes=0)
    write(directory/'result.json', result)
    return result


def short_episode_check(c, directory):
    """Production batch256/3epochs must update even a four-decision episode."""
    cfg = configuration(c, 'mlp', 11, 'cpu')
    rt.seed_all(11)
    agent = rt.build_agent(cfg, device='cpu')
    rng = np.random.RandomState(1)
    def fill(name, n):
        agent.memory.begin_episode(name, agent.policy_version)
        for i in range(n):
            x = rng.normal(size=(3+i, 25)).astype('float32')
            node = rng.normal(size=8).astype('float32')
            mip = rng.normal(size=53).astype('float32')
            a, v, lp = agent.choose_action(x, mip, node)
            agent.remember(cands_state=x, mip_state=mip, node_state=node, action=a,
                           value=v, log_prob=lp, reward=float(i%2), done=i==n-1)
        agent.seal_episode(True)
    fill('short', 4)
    before = Path(directory)/'short-before.pt'
    rt.save_checkpoint(before, agent, cfg)
    result = agent.learn(microbatch_size=2)
    assert result['updated'] and result['num_samples'] == 4 and result['num_updates'] == 3
    expected = copy.deepcopy((agent.actor_network.state_dict(), agent.critic_network.state_dict(),
                              agent.actor_optimizer.state_dict(), agent.critic_optimizer.state_dict(), rt.rng_state()))
    agent, _ = rt.load_agent(before, device='cpu')
    agent.learn(microbatch_size=2)
    assert same_state(expected, (agent.actor_network.state_dict(), agent.critic_network.state_dict(),
                                 agent.actor_optimizer.state_dict(), agent.critic_optimizer.state_dict(), rt.rng_state()))
    fill('empty', 0)
    no_update = agent.learn()
    assert not no_update['updated'] and len(agent.memory.episodes) == 0
    write(Path(directory)/'short-episode.json', dict(passed=True, short_episode_updates=3,
          batch_size=256, empty_episode_no_fake_update=True, exact_recovery=True, real_episodes=0))
