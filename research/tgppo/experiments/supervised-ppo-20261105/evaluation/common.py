"""One CPU allocation for all conditions:110 paired development tasks."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
TOP = PACKAGE.parent
ROOT = TOP.parents[2]
BASE = ROOT / 'tgppo/runs/supervised-ppo-20261105'
OUT = BASE / 'development-110'
DOCS = ROOT / 'docs/methods/tgppo_supervised_ppo_20261105/results'
SEEDS = (11, 22)
EPISODES = (0, 20, 100)
NAMES = ('vpm2', 'dcmulti', 'eil33-2', 'neos-476283', 'roi5alpha10n8')
BUDGET = 3000


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def model_specs():
    return [dict(label='{}-e{}-s{}'.format(v, e, s), variant=v, episode=e, training_seed=s)
        for s in SEEDS for v, e in [('bc', 0), ('control', 20), ('control', 100), ('expert', 20), ('expert', 100)]]


def specs():
    labels = [dict(label='native', variant='native', episode=None, training_seed=None)] + model_specs()
    result = []
    for i, name in enumerate(NAMES):
        for ss in (0, 1):
            offset = (2 * i + ss) % len(labels)
            for model in labels[offset:] + labels[:offset]:
                result.append(dict(model, job_id='t{:03d}-{}-{}-s{}'.format(len(result), model['label'], name, ss),
                    instance=name, solver_seed=ss,
                    canonical_sampling_job_id=('native-' if model['variant'] == 'native' else 'dev-') + name + '-s' + str(ss)))
    assert len(result) == 110 and len({r['job_id'] for r in result}) == 110
    return result


def verify():
    release = read(TOP / 'docs/release.json')
    if not release['passed']:
        raise ValueError('Unverified package')
    for name, digest in release['files'].items():
        if sha(TOP / name) != digest:
            raise ValueError('Package changed: ' + name)
    for name, digest in release['external_inputs'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    protected = read(ROOT / 'tgppo/experiments/candidate-residual-20261104/docs/protected_sources.json')
    for name, digest in protected.items():
        if sha(ROOT.parent / name) != digest:
            raise ValueError('Protected baseline changed: ' + name)
    return dict(package_files=len(release['files']), external_inputs=len(release['external_inputs']),
        protected_sources=len(protected), requested_tasks=110, solver_tasks_executed=0, torch_imported=False)
