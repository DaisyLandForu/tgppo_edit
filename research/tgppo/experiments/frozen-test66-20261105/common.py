"""Frozen100episode weights, complete66 test instances, four CPU Custom jobs."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
RUN = ROOT / 'tgppo/runs/light-actor-test66-20261105'
DOCS = ROOT / 'docs/methods/tgppo_light_actor_test66_20261105'
ORIGINAL = ROOT / 'tgppo/experiments/light-actor-20261105'
PLAN = json.loads((PACKAGE / 'selection.json').read_text())
VARIANTS = ('b0', 'b0_bc', 'mlp', 'mlp_bc')
SEEDS = (11, 22)
BUDGET = 14400
SHARDS = tuple(range(4))
TOTAL_TASKS = 1188


def resources(shard):
    if shard not in SHARDS:
        raise ValueError('Shard must be0..3')
    return PLAN['shards'][shard]


def task_count(shard):
    return resources(shard)['tasks']


def task_resources(name):
    large = name == 'mspp16'
    return dict(phase='large_serial' if large else 'ordinary', workers=1 if large else 4,
        address_space_limit_bytes=(24 if large else 8) * 1024**3,
        watchdog_seconds=300 if large else 180)


def apply_resource_policy(job, shard):
    """mspp16 runs alone after all ordinary workers exit within shard3."""
    if job['instance_name'] not in {i['instance'] for i in selected(shard)}:
        raise ValueError('Instance assigned to another shard')
    policy = task_resources(job['instance_name'])
    job['address_space_limit_bytes'] = policy['address_space_limit_bytes']
    return job


def execution_batches(tasks):
    """Sequential phases; the24GiB process can never overlap the4-worker pool."""
    batches = []
    for name in ('ordinary', 'large_serial'):
        jobs = [t['job'] for t in tasks if task_resources(t['instance'])['phase'] == name]
        if jobs:
            batches.append(dict(task_resources(jobs[0]['instance_name']), jobs=jobs))
    return batches


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def output(shard):
    resources(shard)
    return RUN / ('shard-' + str(shard))


def selected(shard):
    return [i for i in PLAN['selected'] if i['shard'] == shard]


def specs(shard):
    labels = ['native'] + ['{}-s{}'.format(v, s) for s in SEEDS for v in VARIANTS]
    models = {m['label']: m for m in PLAN['frozen_models']}
    result = []
    for i, item in enumerate(selected(shard)):
        name = item['instance']
        for ss in (0, 1):
            offset = (2 * i + ss) % len(labels)
            for label in labels[offset:] + labels[:offset]:
                model = models.get(label)
                result.append(dict(shard=shard, label=label, variant=model['variant'] if model else 'native',
                    training_seed=model['training_seed'] if model else None, instance=name, solver_seed=ss,
                    job_id='t{:03d}-{}-{}-s{}'.format(len(result), label, name, ss),
                    canonical_sampling_job_id='test-{}-s{}'.format(name, ss)))
    assert len(result) == task_count(shard) and len({x['job_id'] for x in result}) == len(result)
    return result


def verify(shard=None):
    authorization = read(PACKAGE / 'authorization.json')
    assert authorization['authorized'] and authorization['total_solver_tasks'] == TOTAL_TASKS
    assert authorization['total_instances'] == 66 and authorization['shards'] == len(SHARDS)
    assert sha(PACKAGE / 'selection.json') == authorization['selection_sha256']
    release = read(PACKAGE / 'release.json')
    if not release['passed']:
        raise ValueError('Unverified CPU package')
    for name, digest in release['files'].items():
        if sha(PACKAGE / name) != digest:
            raise ValueError('CPU package changed: ' + name)
    for name, digest in release['external_inputs'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen code/input changed: ' + name)
    prior = read(ORIGINAL / 'docs/release.json')
    for name, digest in prior['files'].items():
        if sha(ORIGINAL / name) != digest:
            raise ValueError('Original Actor/runtime changed: ' + name)
    for name, digest in prior['external_inputs'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Original external input changed: ' + name)
    protected = read(ROOT / 'tgppo/experiments/candidate-residual-20261104/docs/protected_sources.json')
    for name, digest in protected.items():
        if sha(ROOT.parent / name) != digest:
            raise ValueError('Protected baseline changed: ' + name)
    models = {(m['variant'], m['training_seed']) for m in PLAN['frozen_models']}
    assert models == {(v, s) for v in VARIANTS for s in SEEDS}
    canonical = ROOT / 'migration/2026-10-22/server-run/test66.json'
    assert sha(canonical) == PLAN['canonical_test66_sha256']
    assert len(PLAN['selected']) == 66 and {i['instance'] for i in PLAN['selected']} == set(read(canonical))
    assert {f.name[:-7] for f in (ROOT / 'tgppo/datasets/test_instances').glob('*.mps.gz')} == set(read(canonical))
    assert sum(task_count(s) for s in SHARDS) == TOTAL_TASKS
    assert [i['shard'] for i in PLAN['selected'] if i['instance'] == 'mspp16'] == [3]
    assert [len(selected(s)) for s in SHARDS] == [18, 17, 17, 14]
    for s in SHARDS:
        policy = resources(s)
        assert policy['budget_seconds'] == BUDGET and policy['ram_gib'] == 32
        assert policy['workers'] == 4 and policy['cpu_request'] == 4
        assert policy['watchdog_seconds'] == 180
        assert policy['address_space_limit_bytes'] == 8 * 1024**3
        assert policy['has_serial_mspp16'] == (s == 3)
        assert task_count(s) == len(selected(s)) * 18
    for model in PLAN['frozen_models']:
        if sha(model['actor']) != model['actor_sha256']:
            raise ValueError('Frozen100episode Actor changed: ' + model['label'])
    instances = PLAN['selected'] if shard is None else selected(shard)
    for item in instances:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Test instance changed: ' + item['instance'])
    return dict(package_files=len(release['files']), original_files=len(prior['files']),
        external_inputs=len(release['external_inputs']), protected_sources=len(protected),
        frozen_actors=8, instance_hashes=len(instances), solver_tasks_executed=0, torch_imported=False)
