"""CPU-only development curve: immutable inputs and one bounded Custom job."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
BASE = ROOT / 'tgppo/runs/light-actor-20261105'
OUT = BASE / 'development-curve-70'
DOCS = ROOT / 'docs/methods/tgppo_light_actor_20261105/dev_curve_20261105'
ORIGINAL = ROOT / 'tgppo/experiments/light-actor-20261105'
NAMES = ('vpm2', 'dcmulti', 'eil33-2', 'neos-476283', 'roi5alpha10n8')
SEEDS = (11, 22)
EPISODES = (0, 20, 100)
BUDGET = 2400


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


def specs():
    labels = ['native'] + ['ep{}-s{}'.format(e, s) for s in SEEDS for e in EPISODES]
    result = []
    for i, name in enumerate(NAMES):
        for ss in (0, 1):
            offset = (2 * i + ss) % len(labels)
            for label in labels[offset:] + labels[:offset]:
                native = label == 'native'
                episode, seed = (None, None) if native else tuple(map(int, label.replace('ep', '').split('-s')))
                result.append(dict(job_id='t{:03d}-{}-{}-s{}'.format(len(result), label, name, ss),
                    label=label, episode=episode, training_seed=seed, instance=name, solver_seed=ss,
                    canonical_sampling_job_id=('native-' if native else 'dev-') + name + '-s' + str(ss)))
    assert len(result) == 70 and len({r['job_id'] for r in result}) == 70
    return result


def verify():
    """Read-only: never imports Torch or starts a solver."""
    release = read(PACKAGE / 'release.json')
    for relative, digest in release['files'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Frozen delivery/input changed: ' + relative)
    original = read(ORIGINAL / 'docs/release.json')
    for relative, digest in original['files'].items():
        if sha(ORIGINAL / relative) != digest:
            raise ValueError('Original release changed: ' + relative)
    for relative, digest in original['external_inputs'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Original external input changed: ' + relative)
    protected = read(ROOT / 'tgppo/experiments/candidate-residual-20261104/docs/protected_sources.json')
    for relative, digest in protected.items():
        if sha(ROOT.parent / relative) != digest:
            raise ValueError('Protected baseline changed: ' + relative)
    return dict(delivery_and_inputs=len(release['files']), original_files=len(original['files']),
        external_inputs=len(original['external_inputs']), protected_sources=len(protected),
        requested_tasks=len(specs()), solver_tasks_executed=0, torch_imported=False)
