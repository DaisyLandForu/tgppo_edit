"""Verify completed110 records and reproduce supervision comparisons without solving."""
import csv
import datetime
import hashlib
import math
from pathlib import Path
import statistics as st
import sys
import tarfile
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'tgppo/experiments/supervised-ppo-20261105/evaluation'))
import common as c


def panel(rows):
    calls = sum(r.get('branch_count', 0) for r in rows)
    proof = [bool(r['proof_completed'] and r['scip_solve_time'] <= 120) for r in rows]
    return dict(pdi=st.mean(r['pdi_horizon_midpoint'] for r in rows),
        lower=st.mean(r['pdi_horizon_lower'] for r in rows), upper=st.mean(r['pdi_horizon_upper'] for r in rows),
        proof_rate=st.mean(proof), proof_count=sum(proof), tasks=len(rows),
        wall=st.mean(r['optimize_wall_seconds'] for r in rows), actor_calls=calls,
        actor_ms=1000 * sum(r.get('callback_seconds', {}).get('actor', 0) for r in rows) / calls if calls else None)


def diff(a, b):
    return dict(pdi_delta=a['pdi'] - b['pdi'], pdi_percent=100*(a['pdi']/b['pdi']-1),
        lower=a['lower'] - b['upper'], upper=a['upper'] - b['lower'], proof_delta=a['proof_rate']-b['proof_rate'])


def main():
    verification = c.verify()
    analysis = c.read(HERE / 'analysis.json')
    for path, h in analysis['source_sha256'].items():
        assert c.sha(path) == h, path
    job, engine = c.read(c.OUT / 'job_state.json'), c.read(c.OUT / 'status.json')
    assert job['status'] == engine['status'] == 'complete' and job['charged_wall_seconds'] <= c.BUDGET
    assert not job['error'] and job['engine_complete'] and job['final_input_verification'] and engine['weights_unchanged']
    manifest = c.read(c.OUT / 'manifest.json')
    rows = c.read(c.OUT / 'evaluation_rows.json')
    found, tasks = {r['job_id']: r for r in rows}, {t['job_id']: t for t in manifest['tasks']}
    assert len(found) == len(rows) == len(tasks) == 110
    with tarfile.open(c.OUT / 'raw_results.tar', 'r') as archive:
        archived = {}
        for member in archive.getmembers():
            assert member.isfile() and member.name.startswith('t') and member.name.endswith('.json')
            row = json.load(archive.extractfile(member))
            assert row['job_id'] not in archived
            archived[row['job_id']] = row
        assert archived == found
    records = {}
    for spec in c.specs():
        t, r = tasks[spec['job_id']], found[spec['job_id']]
        assert all(t[k] == v for k, v in spec.items())
        assert r['valid'] and r['task_terminal'] and r['worker_cpu_complete'] and not r['cuda_initialized']
        assert r['time_limit_seconds'] == r['pdi_horizon_seconds'] == 120
        assert r['cpu_affinity'] == manifest['cpu_affinity'] and r['address_space_limit_bytes'] == 8*1024**3
        seed = spec['training_seed'] or 11
        sampling = int(hashlib.sha256((str(seed)+':'+spec['canonical_sampling_job_id']).encode()).hexdigest()[:8], 16)
        assert r['sampling_seed'] == t['job']['sampling_seed'] == sampling
        if r['mode'] == 'eval':
            assert r['callback_seconds']['critic'] == r['fallback_action_count'] == 0
        records.setdefault((spec['variant'], spec['episode'], spec['training_seed']), []).append(r)
    for m in manifest['models'].values():
        assert c.sha(m['source']) == m['source_sha256'] == c.sha(m['frozen_actor']) == m['actor_sha256']
        assert c.sha(m['checkpoint']) == m['checkpoint_sha256']
    probe = c.read(c.OUT / 'cpu_fork_verification.json')
    assert probe['passed'] and probe['parent_threads'] == probe['child']['threads'] == 1 and not probe['child']['cuda_initialized']
    for pair in manifest['pairings'].values():
        assert pair['control']['initialization'] == pair['expert']['initialization']
    active = [n for n in c.NAMES if all(r.get('branch_count', 0)>0 for k, rs in records.items()
              if k[0]!='native' for r in rs if r['instance_name']==n)]
    assert active == analysis['common_actor_instances']
    summaries = {}
    for k, rs in records.items():
        for tag, names in [('all', c.NAMES), ('common_actor', active)]:
            summaries[(k, tag)] = panel([r for r in rs if r['instance_name'] in names])
    with (HERE / 'summary.csv').open() as stream:
        for r in csv.DictReader(stream):
            own = summaries[((r['variant'], int(r['episode']), int(r['training_seed'])), r['panel'])]
            for key in ('pdi', 'lower', 'upper', 'proof_rate', 'wall', 'actor_ms'):
                assert math.isclose(own[key], float(r[key]), rel_tol=1e-11, abs_tol=1e-9)
    selected, chosen = {}, {}
    for v in ('control', 'expert'):
        for s in c.SEEDS:
            e = min(c.EPISODES, key=lambda ep:(summaries[(('bc',0,s) if ep==0 else (v,ep,s),'all')]['pdi'],ep))
            selected[v+'-s'+str(s)] = e
            chosen[(v,s)] = records[('bc',0,s) if e==0 else (v,e,s)]
    assert selected == {v['variant']+'-s'+str(v['training_seed']):v['selected_episode'] for v in analysis['selected']}
    tables = []
    for v,e in [('bc',0),('control',20),('control',100),('expert',20),('expert',100),('control','selected'),('expert','selected')]:
        for tag, names in [('all',c.NAMES),('common_actor',active)]:
            rs = [r for s in c.SEEDS for r in (chosen[(v,s)] if e=='selected' else records[(v,e,s)]) if r['instance_name'] in names]
            tables.append(dict(variant=v,episode=e,panel=tag,**panel(rs)))
    decompositions = []
    for label in (20,100,'selected'):
        for s in c.SEEDS:
            a,b = (chosen[('expert',s)],chosen[('control',s)]) if label=='selected' else (records[('expert',label,s)],records[('control',label,s)])
            for name in c.NAMES:
                d = diff(panel([r for r in a if r['instance_name']==name]),panel([r for r in b if r['instance_name']==name]))
                decompositions.append(dict(comparison=label,training_seed=s,instance=name,no_actor=name not in active,
                    contribution_to_full_mean_delta=d['pdi_delta']/5,**d))
    # Reproduce the fixed pre-run gate independently, including BC as a reference.
    passed, joint_wins = True, []
    for s in c.SEEDS:
        for ref in (chosen[('control',s)],records[('bc',0,s)]):
            for names in (c.NAMES,active):
                a=panel([r for r in chosen[('expert',s)] if r['instance_name'] in names]);b=panel([r for r in ref if r['instance_name'] in names])
                passed &= a['upper']<b['lower'] and a['proof_rate']>=b['proof_rate']
    for name in active:
        if all(panel([r for r in chosen[('expert',s)] if r['instance_name']==name])['upper'] <
               panel([r for r in ref if r['instance_name']==name])['lower'] for s in c.SEEDS
               for ref in (chosen[('control',s)],records[('bc',0,s)])):
            joint_wins.append(name)
    passed &= len(joint_wins)>=2
    assert bool(passed)==analysis['gate']['passed'] and joint_wins==analysis['gate']['common_instance_wins']
    training=[]
    paths=[c.OUT/n for n in ('manifest.json','job_state.json','status.json','evaluation_rows.json','raw_results.tar','cpu_fork_verification.json')]
    paths += [HERE/n for n in ('REPORT.md','analysis.json','summary.csv','fixed_budget_effects.csv','selected_effects.csv','selection.csv','training.csv')]
    for v in ('control','expert'):
        for s in c.SEEDS:
            source=c.BASE/v/'training'/('B0-s'+str(s));updates=c.read(source/'updates.json');actual=[u for u in updates if u['updated']]
            steps=sum(u['num_updates'] for u in actual);batches=sum(u['expert_batches'] for u in actual);samples=sum(u['expert_samples'] for u in actual)
            assert batches==(steps if v=='expert' else 0) and samples==16*batches
            assert all(u['expert_weighted_loss']>0 for u in actual) if v=='expert' else all(u['expert_weighted_loss']==0 for u in actual)
            training.append(dict(variant=v,seed=s,updates=len(actual),optimizer_steps=steps,expert_batches=batches,expert_samples=samples,
                last_training_expert_nll=actual[-1]['expert_normalized_nll'],last_weighted_loss=actual[-1]['expert_weighted_loss']))
            paths += [source/'updates.json',source/'summary.json',c.BASE/'jobs'/(v+'-s'+str(s)+'.json')]
    for name, values in [('verified_summary.csv',tables),('mean_decomposition.csv',decompositions)]:
        with (HERE/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    assert 'torch' not in sys.modules
    c.write(HERE/'REVIEW_DATA.json',dict(verified_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        valid_tasks=110,raw_tar_records_equal=True,source_and_frozen_weights_unchanged=True,full_checkpoints_unchanged=True,
        input_verification=verification,charged_wall_seconds=job['charged_wall_seconds'],selected=selected,
        gate_passed=bool(passed),cross_seed_joint_wins=joint_wins,summary=tables,decomposition=decompositions,training=training,
        training_pairings=manifest['pairings'],evaluation_cpu=manifest['stack']['cpu_model'],
        new_solver_tasks=0,new_training_episodes=0,torch_imported=False,
        source_sha256={str(p.relative_to(ROOT)):c.sha(p) for p in paths}))
    print(json.dumps(dict(valid_tasks=110,tar_equal=True,gate_passed=bool(passed),selected=selected,
        evaluation_minutes=job['charged_wall_seconds']/60,new_solver_tasks=0),indent=2))


if __name__=='__main__':
    main()
