"""Freeze already-produced actor-0100.pt files; no training or solver optimize."""
from pathlib import Path
import shutil

from common import BASE, NEW, PACKAGE, ROOT, RUN, sha, write, read, verify


def main():
    from project import runtime as rt
    sources = [('b0', 'paper-aligned-20261106', 'b0'), ('b0_new', 'paper-trainmode-20261108', 'b0'),
               ('mlp', 'paper-aligned-20261106', 'mlp'), ('mlp_bc', 'paper-aligned-20261106', 'mlp_bc')]
    target = RUN / 'frozen'
    target.mkdir(parents=True, exist_ok=True)
    models = []
    for label, experiment, variant in sources:
        folder = ROOT / 'tgppo/runs' / experiment / 'training' / (variant + '-s11')
        source = folder / 'actor-0100.pt'
        checkpoint = folder / 'episode-0100.pt'
        p = rt.read_checkpoint(source, allow_actor_only=True)
        c, provenance = p['config'], p['provenance']
        assert p['schema_version'] == 'actor-only-v1'
        assert provenance['progress']['next_episode'] == 100 and provenance['progress']['phase'] == 'ready'
        assert c['server_training_seed'] == 11 and c['experiment'] == experiment
        assert c['variant'] == variant and c['task_time_limit'] == 3600 and c['optimizer'] == 'AdamW'
        assert c['actor_kind'] == ('b0' if label.startswith('b0') else 'mlp')
        assert c['expert_pretraining'] == ('bc' if label == 'mlp_bc' else 'none')
        assert sha(checkpoint) == provenance['checkpoint_sha256']
        frozen = target / (label + '-s11-actor-0100.pt')
        if not frozen.exists():
            shutil.copyfile(source, frozen)
            frozen.chmod(0o444)
        assert sha(frozen) == sha(source)
        models.append(dict(label=label, variant=variant, training_seed=11, episode=100,
            source_actor=str(source.relative_to(ROOT)), checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=provenance['checkpoint_sha256'], frozen_actor=str(frozen.relative_to(ROOT)),
            actor_sha256=sha(source), configuration_sha256=p['config_sha256'], configuration=c,
            progress=provenance['progress'],
            actor_update_mode='train' if label == 'b0_new' else 'eval',
            effective_update_dropout=0.05 if label == 'b0_new' else 0.0))
    # Old and new deployment paths are byte-identical; update logic is not used in evaluation.
    for rel in ('project/policy/actor.py', 'project/policy/modules.py', 'project/policy/light_actor.py',
                'project/environment/environment.py', 'project/brancher/brancher.py'):
        assert sha(BASE / 'code' / rel) == sha(NEW / 'code' / rel), rel
    names = read(ROOT / 'migration/2026-10-22/server-run/test66.json')
    ordinary = sorted([n for n in names if n != 'mspp16'], key=lambda n: __import__('hashlib').sha256(n.encode()).hexdigest())
    instances = []
    for index, name in enumerate(ordinary):
        shard = 0 if index < 17 else 1 if index < 34 else 2 if index < 51 else 3
        path = ROOT / 'tgppo/datasets/test_instances' / (name + '.mps.gz')
        instances.append(dict(name=name, shard=shard, path=str(path.relative_to(ROOT)), sha256=sha(path)))
    path = ROOT / 'tgppo/datasets/test_instances/mspp16.mps.gz'
    instances.append(dict(name='mspp16', shard=3, path=str(path.relative_to(ROOT)), sha256=sha(path)))
    p = dict(authorized=True, experiment='paper100-test66-20261110', episodes=100, training_seed=11,
        solver_seeds=[0], labels=['native', 'b0', 'b0_new', 'mlp', 'mlp_bc'], total_tasks=330,
        time_limit_seconds=3600, watchdog_seconds=4200, models=models, instances=instances,
        canonical_test66_sha256=sha(ROOT / 'migration/2026-10-22/server-run/test66.json'),
        selection='All four fixed100 checkpoints chosen before evaluation; no test-based checkpoint selection',
        authorization='User explicitly selected full66 and solver0; independent100 round, original500 entry unchanged',
        total_wall_limit_seconds=None, historical_test_exposure=True,
        partition='SHA256(name) order, ordinary counts17/17/17/14; mspp16 serial on shard3')
    if (PACKAGE / 'plan.json').exists():
        assert read(PACKAGE / 'plan.json') == p, 'Do not silently change an already-frozen plan'
    else:
        write(PACKAGE / 'plan.json', p)
    print(verify(release=False), flush=True)


if __name__ == '__main__':
    main()
