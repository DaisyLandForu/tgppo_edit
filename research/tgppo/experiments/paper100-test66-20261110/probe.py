"""Real frozen Actor inference/fork check; no optimization or training."""
import hashlib
import os
import struct


def signatures(paths):
    import numpy as np
    import torch
    from project import runtime as rt
    result = []
    for path in paths:
        policy, payload = rt.load_actor(path)
        assert not hasattr(policy, 'critic_network')
        assert payload['provenance']['progress']['next_episode'] == 100
        policy.actor_network.eval()
        digest = hashlib.sha256()
        for length in (7, 31):
            rng = np.random.RandomState(913 + length)
            c = rng.normal(size=(length, 25)).astype('float32')
            m = rng.normal(size=53).astype('float32')
            n = rng.normal(size=8).astype('float32')
            mask = np.zeros(length, dtype=bool)
            mask[-2:] = True
            inputs = policy._inputs(c, m, n, mask)
            torch.manual_seed(200 + length)
            with torch.no_grad():
                probs = policy.actor_network(inputs[0], inputs[2], inputs[1], inputs[3])
            assert torch.isfinite(probs).all()
            digest.update(probs.numpy().tobytes())
            action, logp = policy.act(c, m, n, mask)
            digest.update(struct.pack('id', action, logp))
        result.append(dict(probability_action_sha256=digest.hexdigest(),
                           parameters=sum(p.numel() for p in policy.actor_network.parameters())))
    return result


def child_probe(connection, paths):
    import torch
    try:
        from pyscipopt import Model
        m = Model()
        version = m.version()
        m.freeProb()
        connection.send(dict(values=signatures(paths), scip_version=version,
            threads=len(os.listdir('/proc/self/task')), cuda_initialized=torch.cuda.is_initialized()))
    except BaseException as exc:
        connection.send(dict(error=repr(exc)))
    finally:
        connection.close()


def check(paths):
    import torch
    from project import runtime as rt
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    expected = signatures(paths)
    assert not torch.cuda.is_initialized() and len(os.listdir('/proc/self/task')) == 1
    context = rt.mp.get_context('fork')
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=child_probe, args=(send, paths))
    child.start()
    send.close()
    try:
        if not receive.poll(60):
            raise RuntimeError('CPU fork verification timed out; no solve started')
        observed = receive.recv()
        child.join(timeout=5)
        assert child.exitcode == 0 and observed.get('values') == expected, observed
        assert observed['threads'] == 1 and not observed['cuda_initialized']
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
        receive.close()
    return dict(passed=True, observed=observed, probabilities_actions_bitwise_equal=True,
                solver_optimizations=0, training_steps=0)
