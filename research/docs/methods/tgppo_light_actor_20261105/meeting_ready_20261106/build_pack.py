"""Prepare Friday deliverables from committed evidence only; never starts jobs."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIOR = HERE.parent / 'continuation'
RUN = ROOT / 'tgppo/runs/light-actor-20261105'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rows(name):
    with (PRIOR / name).open() as stream:
        return list(csv.DictReader(stream))


def csv_save(name, data):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main():
    params = dict(b0=4738895, b0_bc=4738895, mlp=24493, mlp_bc=24493,
                  set=32749, set_bc=32749, isab=92205, isab_bc=92205)
    main_table = []
    for split, file in [('development', 'development_panels.csv'), ('cross8_exploratory', 'cross_panels.csv')]:
        source = [r for r in rows(file) if r['panel'] == 'all']
        for variant in dict.fromkeys(r['variant'] for r in source):
            rs = [r for r in source if r['variant'] == variant]
            assert len(rs) == 2
            calls = sum(int(r['actor_calls']) for r in rs)
            main_table.append(dict(split=split, variant=variant, parameters=params[variant],
                pdi_seed11=float(next(r['pdi'] for r in rs if r['training_seed'] == '11')),
                pdi_seed22=float(next(r['pdi'] for r in rs if r['training_seed'] == '22')),
                pooled_pdi=st.mean(float(r['pdi']) for r in rs),
                proof_rate=st.mean(float(r['proof_rate']) for r in rs),
                mean_optimize_seconds=st.mean(float(r['wall']) for r in rs),
                actor_ms_per_call=sum(float(r['actor_ms']) * int(r['actor_calls']) for r in rs) / calls))
    dev_native = read(PRIOR / 'decision.json')['native']
    cross_rows = read(RUN / 'cross-instance-8/evaluation_rows.json')
    native = [r for r in cross_rows if r['mode'] == 'native']
    for split, pdi, proof, wall in [('development', dev_native['pdi'], dev_native['proof_rate'], dev_native['wall']),
        ('cross8_exploratory', st.mean(r['pdi_horizon_midpoint'] for r in native),
         st.mean(bool(r['proof_completed'] and r['scip_solve_time'] <= 120) for r in native),
         st.mean(r['optimize_wall_seconds'] for r in native))]:
        main_table.append(dict(split=split, variant='relpscost', parameters=None, pdi_seed11=None, pdi_seed22=None,
            pooled_pdi=pdi, proof_rate=proof, mean_optimize_seconds=wall, actor_ms_per_call=None))
    csv_save('main_results.csv', main_table)
    effects = []
    cross = rows('cross_panels.csv')
    for label, candidate, reference in [('structure_without_BC', 'mlp', 'b0'),
        ('BC_within_MLP', 'mlp_bc', 'mlp'), ('combined', 'mlp_bc', 'b0')]:
        for panel in ('all', 'common_actor'):
            for seed in ('11', '22', 'pooled'):
                choose = lambda v: [r for r in cross if r['variant'] == v and r['panel'] == panel
                    and (seed == 'pooled' or r['training_seed'] == seed)]
                a, b = choose(candidate), choose(reference)
                effects.append(dict(effect=label, candidate=candidate, reference=reference, panel=panel, training_seed=seed,
                    pdi_percent=100 * (st.mean(float(r['pdi']) for r in a) / st.mean(float(r['pdi']) for r in b) - 1),
                    proof_delta=st.mean(float(r['proof_rate']) for r in a) - st.mean(float(r['proof_rate']) for r in b)))
    csv_save('independent_effects.csv', effects)
    # Concrete OPTIONAL CPU expansion; prepare identities only, never launch it.
    manifest = read(RUN / 'cross-instance-8/manifest.json')
    previously_evaluated = {i['instance'] for i in manifest['selected']}
    test_dir = ROOT / 'tgppo/datasets/test_instances'
    names = sorted(p.name[:-7] for p in test_dir.glob('*.mps.gz'))
    assert len(names) == 66
    protocol = read(ROOT / 'docs/stage1/protocol_draft.json')
    assert not set(names) & set(protocol['data']['train'] + protocol['data']['development'])
    candidates = set(names) - previously_evaluated - {'mspp16'}
    selected = sorted(candidates, key=lambda n: hashlib.sha256(('light-actor-cpu-expand-v1:' + n).encode()).hexdigest())[:16]
    models = []
    for label, model in manifest['models'].items():
        assert sha(model['frozen_actor']) == model['actor_sha256']
        models.append(dict(label=label, variant=model['variant'], training_seed=model['seed'],
            actor=model['frozen_actor'], actor_sha256=model['actor_sha256']))
    tasks, selected_records = [], []
    for index, name in enumerate(selected):
        shard = index % 4
        selected_records.append(dict(instance=name, shard=shard, path=str(test_dir / (name + '.mps.gz')),
            sha256=sha(test_dir / (name + '.mps.gz'))))
        for ss in (0, 1):
            tasks.append(dict(shard=shard, instance=name, solver_seed=ss, variant='native', training_seed=None))
            for model in models:
                tasks.append(dict(shard=shard, instance=name, solver_seed=ss,
                    variant=model['variant'], training_seed=model['training_seed']))
    assert len(tasks) == 288 and all(sum(t['shard'] == i for t in tasks) == 72 for i in range(4))
    save('CPU_EVAL_PROPOSAL.json', dict(status='proposal_only_not_authorized_or_executed', selection_rule='SHA256(light-actor-cpu-expand-v1:name), first16 excluding current8 and mspp16',
        known_large_instance_exception='mspp16 excluded before any new result due existing single-worker24GiB protocol',
        selected=selected_records, frozen_models=models, tasks=288, model_tasks=256, fresh_native_tasks=32,
        shards=4, tasks_per_shard=72, requested_resources_per_shard=dict(cpu=4, ram_gib=32, gpu=0),
        proposed_wall_cap_seconds_per_shard=3000, time_limit_seconds=120, watchdog_seconds=180,
        planned_epochs=100, new_training=False, sampling='same dev/test action seed convention as frozen previous cross-instance evaluation',
        caveat='These16 are outside current8, but project historical test results were viewed; not an untouched holdout. Not a66-instance evaluation.',
        data_scope='old frozen MLP/B0 x noBC/BC x training seeds11/22, all methods for each instance on one CPU host'))
    csv_save('proposed_cpu_tasks.csv', tasks)
    status = {}
    for variant in ('control', 'expert'):
        for seed in (11, 22):
            p = ROOT / 'tgppo/runs/supervised-ppo-20261105/jobs' / (variant + '-s' + str(seed) + '.json')
            status[variant + '-s' + str(seed)] = read(p).get('status') if p.exists() else 'no_shared_start_record'
    save('EVIDENCE.json', dict(prepared_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        gpu_training_observed_status=status, scheduler_api_queried=False,
        real_training_started_by_this_script=False, solver_tasks_started_by_this_script=0,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (PRIOR / 'development_panels.csv',
            PRIOR / 'cross_panels.csv', PRIOR / 'decision.json', RUN / 'cross-instance-8/evaluation_rows.json',
            HERE.parent / 'dev_curve_20261105/analysis.json', Path(__file__).resolve())}))
    print('Meeting tables and OPTIONAL288-task CPU proposal prepared. No training/solver jobs started.')
    print('Proposed16 instances: ' + ', '.join(selected))


if __name__ == '__main__':
    main()
