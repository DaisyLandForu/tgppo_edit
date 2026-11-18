"""Bounded CPU forward/backward timing; no optimizer steps, solves or weight saves."""
import json
import signal
import time
from pathlib import Path

START = time.monotonic()
OUT = Path(__file__).resolve().parent
RUN = OUT.parents[3] / 'tgppo/runs/paper-aligned-20261106/training'
report = dict(scope='CPU forward/backward only; no optimizer steps or solves',
              trials=[], complete=False)


def save():
    report['elapsed_seconds'] = time.monotonic() - START
    (OUT / 'CPU_TIMING.json').write_text(json.dumps(report, indent=2) + '\n')


def stop(signum, frame):
    raise TimeoutError('55-second benchmark deadline')


signal.signal(signal.SIGALRM, stop)
signal.alarm(55)
try:
    import numpy as np
    import torch
    from project.runtime import read_checkpoint, build_agent

    torch.set_num_threads(1)
    torch.manual_seed(17)
    payload = read_checkpoint(RUN / 'mlp-s22/pending.pt')
    states = [r for e in payload['memory']['episodes'] for r in e['transitions']]
    order = sorted(range(len(states)), key=lambda i: len(states[i]['cands_state']))
    lengths = [len(states[i]['cands_state']) for i in order]
    report['candidate_count'] = dict(min=min(lengths), median=int(np.median(lengths)),
                                     max=max(lengths), states=len(states))
    report['note'] = ('One retained rmatr200-p5 trajectory only; B0 Actor initialized '
                      'for architecture timing, shared Critic uses retained weights. '
                      'Forward/backward loss is a timing proxy, not a PPO update.')
    for kind in ('mlp', 'b0'):
        cfg = dict(payload['config'], actor_kind=kind)
        agent = build_agent(cfg, training=False, device='cpu')
        agent.critic_network.load_state_dict(payload['critic'])
        if kind == 'mlp':
            agent.actor_network.load_state_dict(payload['actor'])
        agent.actor_network.eval()
        agent.critic_network.train()
        for bucket, pos in [('median', len(order)//2), ('maximum', len(order)-1)]:
            width = lengths[pos]
            count = min(8, max(1, 1048576 // (width * width)))
            ids = order[max(0, pos-count+1):pos+1]
            chosen = [states[i] for i in ids]
            cands = torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(r['cands_state']) for r in chosen], batch_first=True)
            lens = torch.tensor([len(r['cands_state']) for r in chosen])
            mask = torch.arange(cands.shape[1])[None, :] >= lens[:, None]
            node = torch.as_tensor(np.asarray([r['node_state'] for r in chosen]))
            mip = torch.as_tensor(np.asarray([r['mip_state'] for r in chosen]))
            action = torch.tensor([r['action'] for r in chosen])
            agent.actor_network.zero_grad(set_to_none=True)
            agent.critic_network.zero_grad(set_to_none=True)
            start = time.monotonic()
            probs = agent.actor_network(cands, node, mip, mask)
            loss = -torch.distributions.Categorical(probs).log_prob(action).mean()
            loss.backward()
            actor_s = time.monotonic()-start
            start = time.monotonic()
            values = agent.critic_network(cands, node, mip, mask).reshape(-1)
            values.square().mean().backward()
            critic_s = time.monotonic()-start
            report['trials'].append(dict(actor_kind=kind, bucket=bucket, width=width,
                                         batch=len(ids), actor_seconds=actor_s,
                                         critic_seconds=critic_s,
                                         per_transition_seconds=(actor_s+critic_s)/len(ids)))
            save()
        del agent
    report['complete'] = True
    report['cuda_initialized'] = torch.cuda.is_initialized()
except BaseException as exc:
    report['error'] = repr(exc)
finally:
    signal.alarm(0)
    save()
    print(json.dumps(report, ensure_ascii=False), flush=True)
