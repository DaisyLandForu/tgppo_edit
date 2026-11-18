"""Read stopped trusted checkpoints only; no model construction or optimizer step."""
import datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    started = time.time()
    import torch
    torch.set_num_threads(1)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    targets = [('b0_new', 'paper-trainmode-20261108', 'b0', s) for s in (11, 22)]
    targets += [(v, 'paper-recovery-20261109/cpu-aligned', v, s)
                for v in ('mlp', 'mlp_bc') for s in (11, 22)]
    rows = []
    def same(a, b):
        if torch.is_tensor(a):
            return torch.is_tensor(b) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    def inspect(p):
        assert p['schema_version'] == 2 and fp(p['config']) == p['config_sha256']
        assert p['config']['optimizer'] == 'AdamW' and p['config']['n_epochs'] == 3
        assert p['config']['task_time_limit'] == 3600
        for key in ('actor', 'critic'):
            assert p[key] and all(bool(torch.isfinite(v).all()) for v in p[key].values())
        for key in ('actor_optimizer', 'critic_optimizer'):
            opt = p[key]
            assert opt['param_groups'] and opt['state']
            assert all({'step', 'exp_avg', 'exp_avg_sq'} <= set(v) for v in opt['state'].values())
            assert all(bool(torch.isfinite(t).all()) for v in opt['state'].values()
                       for t in v.values() if torch.is_tensor(t))
        assert {'python', 'numpy', 'torch'} <= set(p['rng'])
    for label, run, variant, seed in targets:
        folder = ROOT / 'tgppo/runs' / run / 'training' / ('{}-s{}'.format(variant, seed))
        with (folder / 'execution.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            path = folder / 'latest.pt'
            digest = sha(path)
            p = torch.load(path, map_location='cpu')
            inspect(p)
            assert p['progress']['phase'] == 'ready' and p['memory']['num_transitions'] == 0
            assert p['config']['variant'] == variant and p['config']['server_training_seed'] == seed
            device = 'cuda' if label == 'b0_new' else 'cpu'
            assert p['config']['learner_device'] == device
            row = dict(label=label, seed=seed, folder=str(folder), checkpoint_sha256=digest,
                config_sha256=p['config_sha256'], committed_episodes=p['progress']['next_episode'],
                policy_version=p['policy_version'], update_counter=p['update_counter'],
                learner_device=device, model_parameters_finite=True, adamw_states_present=True,
                rng_present=True, pending=None)
            record = folder / 'updates' / ('episode-{:04d}.json'.format(p['progress']['next_episode']))
            update = json.loads(record.read_text())
            actor_digest = hashlib.sha256()
            for key, value in sorted(p['actor'].items()):
                actor_digest.update(key.encode())
                actor_digest.update(value.detach().cpu().numpy().tobytes())
            assert actor_digest.hexdigest() == update['actor_sha256']
            row['committed_update_record'] = str(record)
            row['actor_state_matches_committed_update'] = True
            row['committed_update_mtime_epoch'] = record.stat().st_mtime
            row['last_update_used_gradient'] = update['updated']
            pending_path = folder / 'pending.pt'
            if pending_path.exists():
                pending_digest = sha(pending_path)
                pending = torch.load(pending_path, map_location='cpu')
                inspect(pending)
                assert pending['config'] == p['config']
                assert pending['progress']['phase'] == 'update_pending'
                assert pending['progress']['next_episode'] == p['progress']['next_episode']
                assert pending['memory']['num_transitions'] > 0
                for key in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer', 'policy_version', 'update_counter'):
                    assert same(pending[key], p[key]), key
                collection = Path(pending['progress']['collection_directory'])
                assert collection.is_dir()
                row['pending'] = dict(sha256=pending_digest, episode=p['progress']['next_episode']+1,
                    samples=pending['memory']['num_transitions'], collection_directory=str(collection),
                    pre_update_models_and_optimizers_equal_latest=True,
                    recovery='Replay entire PPO update from pre-update pending; never continue partial in-memory steps')
                assert sha(pending_path) == pending_digest
                del pending
            assert sha(path) == digest
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del p
            gc.collect()
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, captured_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        rows=rows, elapsed_seconds=time.time()-started, training_steps=0, solver_optimizations=0,
        constructed_agents=0, cuda_initialized=False, source_checkpoints_unchanged=True)
    tmp = HERE / 'CHECKPOINT_AUDIT.json.tmp'
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(HERE / 'CHECKPOINT_AUDIT.json')


if __name__ == '__main__':
    main()
