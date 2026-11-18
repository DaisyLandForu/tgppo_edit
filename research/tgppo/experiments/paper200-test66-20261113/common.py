"""Independent fixed-200 evaluation. Importing this module never starts SCIP."""
import hashlib
import json
import math
from pathlib import Path
import uuid

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
BASE = ROOT / 'tgppo/experiments/paper-aligned-20261106'
NEW = ROOT / 'tgppo/experiments/paper-trainmode-20261108'
RUN = ROOT / 'tgppo/runs/paper200-test66-20261113'
DOCS = ROOT / 'docs/methods/tgppo_paper200_test66_20261113'
TRAINING_SEEDS = {'native': 0, 'b0_new_s22': 22, 'mlp_s11': 11, 'mlp_bc_s11': 11, 'mlp_bc_s22': 22}
LABELS = tuple(TRAINING_SEEDS)
SHARDS = tuple(range(4))
TIME_LIMIT = 3600
WATCHDOG = 4200


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def plan():
    return read(PACKAGE / 'plan.json')


def selected(shard):
    if shard not in SHARDS:
        raise ValueError('shard must be 0, 1, 2 or 3')
    return [i for i in plan()['instances'] if i['shard'] == shard]


def task_resources(name):
    large = name == 'mspp16'
    return dict(workers=1 if large else 4, address_space_limit_bytes=(24 if large else 6)*1024**3,
                watchdog_seconds=WATCHDOG, phase='large_serial' if large else 'ordinary')


def specs(shard):
    tasks = []
    for index, item in enumerate(selected(shard)):
        # Rotate order before looking at results; all methods share the paper action stream.
        offset = index % len(LABELS)
        for label in LABELS[offset:] + LABELS[:offset]:
            tasks.append(dict(instance=item['name'], label=label, solver_seed=0,
                job_id='e{:03d}-{}-s0'.format(len(tasks), label),
                canonical_sampling_job_id='paper-test-{}-s0'.format(item['name'])))
    return tasks


def verify(release=True):
    p = plan()
    assert p['authorized'] and p['episodes'] == 200 and p['solver_seeds'] == [0]
    assert p['labels'] == list(LABELS) and p['total_tasks'] == 330
    canonical = ROOT / 'migration/2026-10-22/server-run/test66.json'
    assert sha(canonical) == p['canonical_test66_sha256']
    names = [i['name'] for i in p['instances']]
    assert len(names) == len(set(names)) == 66 and set(names) == set(read(canonical))
    assert not set(names) & set(read(BASE / 'configs/protocol.json')['train'])
    assert [len(selected(s)) for s in SHARDS] == [17, 17, 17, 15]
    assert next(i for i in p['instances'] if i['name'] == 'mspp16')['shard'] == 3
    assert sum(len(specs(s)) for s in SHARDS) == 330
    for package in (BASE, NEW):
        r = read(package / 'docs/release.json')
        assert r['passed']
        for name, digest in r['files'].items():
            if sha(package / name) != digest:
                raise ValueError('Frozen training package changed: ' + str(package / name))
        for name, digest in r['inputs'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Frozen input changed: ' + name)
        for name, digest in read(package / 'docs/protected_sources.json').items():
            if sha(ROOT / name) != digest:
                raise ValueError('Protected baseline changed: ' + name)
    assert {m['label']: m['training_seed'] for m in p['models']} == {k: v for k, v in TRAINING_SEEDS.items() if k != 'native'}
    assert sha(BASE / 'docs/author_subset_audit.json') == p['author_subset_audit_sha256']
    for m in p['models']:
        assert m['episode'] == 200 and m['progress']['next_episode'] == 200
        assert sha(ROOT / m['frozen_actor']) == m['actor_sha256']
        assert sha(ROOT / m['source_actor']) == m['actor_sha256']
    for i in p['instances']:
        assert sha(ROOT / i['path']) == i['sha256']
    if release:
        r = read(PACKAGE / 'release.json')
        assert r['passed']
        for name, digest in r['files'].items():
            if sha(PACKAGE / name) != digest:
                raise ValueError('Evaluation package changed: ' + name)
    return dict(models=4, instances=66, tasks=330, training_packages_unchanged=True)


def validate_row(row, spec):
    """Reject incomplete/corrupt rows before they can enter a paired panel."""
    if not row.get('valid'):
        return False
    expected_mode = 'native' if spec['label'] == 'native' else 'eval'
    assert row['job_id'] == spec['job_id'] and row['mode'] == expected_mode
    assert row['instance_name'] == spec['instance'] and row['solver_seed'] == 0
    action_key = str(TRAINING_SEEDS[spec['label']]) + ':' + spec['canonical_sampling_job_id']
    assert row['sampling_seed'] == int(hashlib.sha256(action_key.encode()).hexdigest()[:8], 16)
    assert row['time_limit_seconds'] == TIME_LIMIT
    assert row['measurement_version'] == 'native-pdi-snapshot-horizon-v2'
    assert row['pdi_horizon_seconds'] == TIME_LIMIT
    assert row['address_space_limit_bytes'] == task_resources(spec['instance'])['address_space_limit_bytes']
    assert row['worker_cpu_complete'] and not row['cuda_initialized']
    for key in ('pdi_horizon_lower', 'pdi_horizon_midpoint', 'pdi_horizon_upper',
                'scip_solve_time', 'optimize_wall_seconds', 'worker_wall_seconds', 'nnodes'):
        assert math.isfinite(row[key]) and row[key] >= 0, (key, row.get(key))
    assert row['pdi_horizon_lower'] <= row['pdi_horizon_midpoint'] <= row['pdi_horizon_upper']
    assert row.get('fallback_action_count', 0) == 0
    if expected_mode == 'eval':
        assert row['callback_seconds']['critic'] == 0
    return True


def scan(shard, folder=None):
    """Select earliest valid rows within one environment; never select by outcome.

    Completed five-method instance panels survive migration. Incomplete panels
    from different environments are never combined. JSON is on shared storage,
    so a killed launcher can be recovered without its /tmp cache.
    """
    folder = Path(folder) if folder is not None else RUN / ('shard-' + str(shard))
    expected = {s['job_id']: s for s in specs(shard)}
    cohorts = {}
    for attempt in sorted((folder / 'attempts').glob('*')):
        manifest = attempt / 'manifest.json'
        if not manifest.exists():
            continue
        m = read(manifest)
        assert m['plan_sha256'] == sha(PACKAGE / 'plan.json')
        cid = m['cohort_id']
        group = cohorts.setdefault(cid, dict(rows={}, first_attempt=attempt.name, environment=m['environment']))
        assert group['environment'] == m['environment']
        assert fingerprint(m['environment'])[:24] == cid
        for job in m['tasks']:
            spec = expected[job['job_id']]
            assert job['spec'] == spec
            path = attempt / 'raw' / (job['job_id'] + '.json')
            if not path.exists():
                continue
            row = read(path)
            if validate_row(row, spec) and spec['job_id'] not in group['rows']:
                wanted = m['environment']['cores'][:task_resources(spec['instance'])['workers']]
                assert row['cpu_affinity'] == wanted
                group['rows'][spec['job_id']] = dict(row, label=spec['label'], cohort_id=cid,
                    result_path=str(path), result_sha256=sha(path))
    panels = {}
    for cid, group in sorted(cohorts.items(), key=lambda x: x[1]['first_attempt']):
        for i in selected(shard):
            name = i['name']
            subset = [s for s in expected.values() if s['instance'] == name]
            if name not in panels and all(s['job_id'] in group['rows'] for s in subset):
                panels[name] = [group['rows'][s['job_id']] for s in subset]
    return panels, cohorts
