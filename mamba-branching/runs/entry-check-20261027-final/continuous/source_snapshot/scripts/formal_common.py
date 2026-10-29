"""Shared persistence, locking and provenance for Custom jobs."""
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import tempfile
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def atomic_file(path, mode='wb'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with open(temporary, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another job holds {path}; do not run duplicate shards/trainers')
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({'host': socket.gethostname(), 'pid': os.getpid(), 'started': stamp()}))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def hardware():
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor())
    limits = {name: (Path('/sys/fs/cgroup') / name).read_text().strip() for name in
              ['cpu.max', 'memory.max'] if (Path('/sys/fs/cgroup') / name).exists()}
    return {'host': socket.gethostname(), 'cpu': cpu, 'cgroup': limits, 'python': sys.executable}


def source_files():
    return [*sorted((ROOT / 'code').glob('*.py')),
            *[ROOT / 'scripts' / name for name in ['formal_common.py', 'formal_jobs.py', 'formal_train.py',
                                                  'formal_data.py', 'runtime.sh']],
            ROOT / 'custom_job.sh', ROOT / 'configs/paper-milp-s.json',
            ROOT / 'data/milp-s/manifest.json', ROOT / 'data/milp-s/cutoffs.json']


def versions():
    return {name: importlib.metadata.version(name) for name in
            ['numpy', 'torch', 'pyscipopt', 'mamba-ssm', 'causal-conv1d', 'h5py']}


def prepare(directory, config):
    directory = Path(directory).resolve()
    cfg = read_json(config)
    known = read_json(ROOT / 'configs/milp-s-instances.json')
    assert cfg['profile'] in ['formal', 'diagnostic']
    assert set(cfg['instances']) <= set(known['train'])
    assert cfg['solver_seeds'] and 4 in cfg['solver_seeds'] and any(s < 4 for s in cfg['solver_seeds'])
    assert set(cfg['solver_seeds']) <= set(range(5))
    assert set(cfg['prefixes']) <= {0, 1, 5, 10, 15}
    assert cfg['training']['batch_size'] == 32 and cfg['training']['sequence_states'] == 100
    assert 1 <= cfg['training']['il_microbatch'] <= 32
    assert 1 <= cfg['training']['cl_microbatch'] <= 32
    assert cfg['training']['checkpoint_every_updates'] > 0
    assert cfg['training']['gradient_clip'] > 0 and cfg['training']['cl_sample_cap'] is None
    assert cfg['training']['cl_epochs'] > 0 and cfg['training']['il_epochs'] > 0
    if cfg['profile'] == 'formal':
        assert cfg['instances'] == known['train']
        assert cfg['solver_seeds'] == list(range(5)) and cfg['prefixes'] == [0, 1, 5, 10, 15]
        assert cfg['collection'] == {'time_limit_seconds': 3600, 'node_limit': -1}
        assert cfg['evaluation']['instances'] == known['test']
        assert cfg['evaluation']['solver_seeds'] == list(range(5))
        assert cfg['evaluation']['time_limit_seconds'] == 3600 and cfg['evaluation']['node_limit'] == -1
        assert cfg['evaluation']['cutoff'] is True
    else:
        assert set(cfg['evaluation']['instances']) <= set(known['train'])
        assert not cfg['evaluation']['cutoff']
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory / '.prepare.lock'):
        if (directory / 'run.json').exists():
            assert read_json(directory / 'config.json') == cfg, 'Existing run has different config'
            load_run(directory)
            return directory
        import pyscipopt
        assert Path(sys.prefix).resolve() == Path(os.environ['MAMBA_ENV']).resolve()
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files()}
        binding = Path(pyscipopt.scip.__file__)
        plan = [{'id': f'{name}-s{seed}-r{prefix}', 'instance': name, 'solver_seed': seed,
                 'random_prefix_steps': prefix, 'split': 'valid' if seed == 4 else 'train'}
                for seed in cfg['solver_seeds'] for prefix in cfg['prefixes'] for name in cfg['instances']]
        for relative in hashes:
            target = directory / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        write_json(directory / 'config.json', cfg)
        write_json(directory / 'plan.json', plan)
        write_json(directory / 'run.json', {'schema': 1, 'created_utc': stamp(), 'profile': cfg['profile'],
                   'config_sha256': sha(directory / 'config.json'), 'plan_sha256': sha(directory / 'plan.json'),
                   'sources': hashes, 'binding_path': str(binding), 'binding_sha256': sha(binding),
                   'versions': versions(),
                   'environment': hardware()})
    return directory


def load_run(directory):
    directory = Path(directory).resolve()
    record = read_json(directory / 'run.json')
    assert sha(directory / 'config.json') == record['config_sha256'], 'Config changed'
    assert sha(directory / 'plan.json') == record['plan_sha256'], 'Plan changed'
    for relative, digest in record['sources'].items():
        assert sha(ROOT / relative) == digest, f'Source changed: {relative}; use a new run or audited migration'
    assert sha(record['binding_path']) == record['binding_sha256'], 'SCIP binding changed'
    assert versions() == record['versions'], 'Environment package versions changed'
    return directory, read_json(directory / 'config.json'), record


def task_done(directory, task, kind='collection'):
    path = Path(directory) / kind / task['id'] / 'done.json'
    if not path.exists():
        return None
    result = read_json(path)
    assert result['task'] == task and result['status'] == 'passed'
    for relative, digest in result['artifacts'].items():
        assert sha(Path(directory) / relative) == digest, f'Corrupt committed artifact: {relative}'
    return result
