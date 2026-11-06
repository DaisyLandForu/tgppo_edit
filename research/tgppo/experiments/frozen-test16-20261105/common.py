"""Immutable inputs and deterministic task matrix for four CPU-only Custom jobs."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
RUN = ROOT / 'tgppo/runs/light-actor-test16-20261105'
DOCS = ROOT / 'docs/methods/tgppo_light_actor_test16_20261105'
ORIGINAL = ROOT / 'tgppo/experiments/light-actor-20261105'
PLAN = json.loads((PACKAGE / 'selection.json').read_text())
VARIANTS = ('b0', 'b0_bc', 'mlp', 'mlp_bc')
SEEDS = (11, 22)
BUDGET = 3000


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
    if shard not in range(4):
        raise ValueError('Shard must be0..3')
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
    assert len(result) == 72 and len({x['job_id'] for x in result}) == 72
    return result


def verify(shard=None):
    authorization = read(PACKAGE / 'authorization.json')
    assert authorization['authorized'] and authorization['total_solver_tasks'] == 288
    assert sha(PACKAGE / 'selection.json') == authorization['source_proposal_sha256']
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
    assert len(PLAN['selected']) == 16 and len({i['instance'] for i in PLAN['selected']}) == 16
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
