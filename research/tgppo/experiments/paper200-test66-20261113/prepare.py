"""Freeze the four already-available fixed200 exports; never read latest.pt."""
import gc
import shutil

from common import BASE, NEW, PACKAGE, ROOT, RUN, read, sha, write, verify


def main():
    import torch
    from project import runtime as rt
    sources = [
        ('b0_new_s22', 'paper-trainmode-20261108', 'b0', 22),
        ('mlp_s11', 'paper-recovery-20261109/cpu-aligned', 'mlp', 11),
        ('mlp_bc_s11', 'paper-recovery-20261109/cpu-aligned', 'mlp_bc', 11),
        ('mlp_bc_s22', 'paper-recovery-20261109/cpu-aligned', 'mlp_bc', 22),
    ]
    target = RUN / 'frozen'
    target.mkdir(parents=True, exist_ok=True)
    models = []
    for label, run, variant, seed in sources:
        folder = ROOT / 'tgppo/runs' / run / 'training' / ('{}-s{}'.format(variant, seed))
        source, checkpoint = folder / 'actor-0200.pt', folder / 'episode-0200.pt'
        actor_sha, checkpoint_sha = sha(source), sha(checkpoint)
        payload = rt.read_checkpoint(source, allow_actor_only=True)
        config, provenance = payload['config'], payload['provenance']
        expected_experiment = 'paper-trainmode-20261108' if variant == 'b0' else 'paper-aligned-20261106'
        assert payload['schema_version'] == 'actor-only-v1'
        assert provenance['progress']['next_episode'] == 200 and provenance['progress']['phase'] == 'ready'
        assert config['experiment'] == expected_experiment
        assert config['variant'] == variant and config['server_training_seed'] == seed
        assert config['task_time_limit'] == 3600 and config['optimizer'] == 'AdamW'
        assert config['actor_kind'] == ('b0' if variant == 'b0' else 'mlp')
        assert config['expert_pretraining'] == ('bc' if variant == 'mlp_bc' else 'none')
        assert config['learner_device'] == ('cuda' if variant == 'b0' else 'cpu')
        assert checkpoint_sha == provenance['checkpoint_sha256']
        full = rt.read_checkpoint(checkpoint)
        assert full['config'] == config and full['progress'] == provenance['progress']
        assert full['actor'].keys() == payload['actor'].keys()
        assert all(torch.equal(value, full['actor'][key]) for key, value in payload['actor'].items())
        frozen = target / (label + '-actor-0200.pt')
        if not frozen.exists():
            shutil.copyfile(source, frozen)
            frozen.chmod(0o444)
        assert sha(frozen) == actor_sha == sha(source) and sha(checkpoint) == checkpoint_sha
        continuation = folder / 'cpu_continuation.json'
        models.append(dict(label=label, variant=variant, training_seed=seed, episode=200,
            source_actor=str(source.relative_to(ROOT)), checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=checkpoint_sha, frozen_actor=str(frozen.relative_to(ROOT)),
            actor_sha256=actor_sha, configuration_sha256=payload['config_sha256'], configuration=config,
            progress=provenance['progress'], actor_update_mode='train' if variant == 'b0' else 'eval',
            effective_update_dropout=0.05 if variant == 'b0' else 0.0,
            cpu_continuation=read(continuation) if continuation.exists() else None))
        del payload, full
        gc.collect()
    for rel in ('project/policy/actor.py', 'project/policy/modules.py', 'project/policy/light_actor.py',
                'project/environment/environment.py', 'project/brancher/brancher.py'):
        assert sha(BASE / 'code' / rel) == sha(NEW / 'code' / rel)
    old_package = PACKAGE.with_name('paper100-test66-20261110')
    previous = read(old_package / 'plan.json')
    p = dict(authorized=True, experiment=PACKAGE.name, episodes=200, solver_seeds=[0],
        labels=['native']+[m['label'] for m in models], total_tasks=330,
        time_limit_seconds=3600, watchdog_seconds=4200, models=models, instances=previous['instances'],
        canonical_test66_sha256=previous['canonical_test66_sha256'],
        author_subset_audit_sha256=sha(BASE / 'docs/author_subset_audit.json'),
        selection='All available fixed200 checkpoints at preparation; no performance-based selection',
        authorization='User requested fixed200 parallel CPU evaluation; stop training manually, no automatic solver start',
        unavailable200=['b0_new_s11', 'mlp_s22', 'old_b0_s11', 'old_b0_s22'],
        unbalanced_training_seed_coverage=True,
        eligible_increment_comparisons=[['mlp_bc_s11', 'mlp_s11', 'BC_seed11'],
            ['mlp_bc_s22', 'b0_new_s22', 'combined_seed22']],
        structure_increment_not_identifiable_with_same_training_seed=True,
        total_wall_limit_seconds=None, historical_test_exposure=True,
        fixed500_gate_unchanged=True, reuse_old_evaluation_results=False,
        partition=previous['partition'], preparation_uses_no_latest=True)
    if (PACKAGE / 'plan.json').exists():
        assert read(PACKAGE / 'plan.json') == p, 'Do not change an already-frozen evaluation'
    else:
        write(PACKAGE / 'plan.json', p)
    print(verify(release=False), flush=True)


if __name__ == '__main__':
    main()
