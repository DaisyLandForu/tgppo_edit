"""Fixed 20-second task pilot. No test-set selection or automatic expansion."""
from probe_utils import *


def check_code():
    entries=json.loads((OUT/'online_code_manifest.json').read_text())
    for name,h in entries.items():assert sha(PACKAGE/name)==h,name
    assert json.loads((OUT/'offline/verification.json').read_text())['status']=='passed'
    verify_inputs()


def make_job(key,name,solver,mode,source=0,checkpoint=None,baseline=None,sampling_key=None):
    cutoffs=json.loads((ROOT/'tgppo/datasets/cutoffs.json').read_text())
    job=rt.make_job(key,ROOT/'tgppo/datasets/train_instances',name,plan()['solve_seconds'],solver,source,
        cutoffs,mode,checkpoint=checkpoint,baseline=baseline)
    job['sampling_seed']=rt.action_seed(source,sampling_key or key)
    return job


def execute(tag,jobs):
    d=OUT/tag;d.mkdir(parents=True,exist_ok=True)
    assert len(jobs)<=plan()['maximum_solver_tasks']
    save(d/'jobs.json',jobs)
    result=rt.run_jobs(jobs,d/'tasks',max_workers=1,watchdog_seconds=plan()['watchdog_seconds'])
    assert len(result)==len(jobs)
    save(d/'summary.json',dict(jobs=len(jobs),valid=sum(bool(x.get('valid')) for x in result),results=result))
    if not all(x.get('valid') for x in result):raise RuntimeError('Invalid solver task preserved in '+str(d))
    assert all(not x.get('cuda_initialized') for x in result)
    return result


def native():
    check_code();jobs=[]
    for name in plan()['online_training_instances']+plan()['development_instances']:
        for solver in plan()['solver_seeds']:
            jobs.append(make_job(f'native-{name}-s{solver}',name,solver,'native'))
    assert len(jobs)==14
    results=execute('native',jobs)
    save(OUT/'native/baselines.json',{f"{x['instance_name']}--s{x['solver_seed']}":dict(identity=x['identity'],result=x) for x in results})
    print('Native baselines complete:',len(results),flush=True)
    check_code()


def pilot():
    check_code();baselines=json.loads((OUT/'native/baselines.json').read_text());summary=[]
    for source in [11,22]:
        variants=['original','tail8'] if source==11 else ['tail8','original']
        for variant in variants:
            label=f'{variant}-s{source}';d=OUT/'pilot'/label;d.mkdir(parents=True)
            agent,payload=rt.load_agent(OUT/'offline'/label/'initial.pt',training=True,restore_random_state=False,device='cpu')
            cfg=payload['config'];rt.save_checkpoint(d/'initial.pt',agent,cfg,progress={'pilot_initial':True})
            jobs=[]
            for i,(name,solver) in enumerate((n,s) for s in [0,1] for n in plan()['online_training_instances']):
                jobs.append(make_job(f'collect-{i:02d}-{name}-s{solver}',name,solver,'collect',source,
                    d/'initial.pt',baselines[f'{name}--s{solver}'],sampling_key=f'collect-{i:02d}-{name}-s{solver}'))
            print('COLLECT BEGIN',label,flush=True)
            collected=execute(f'pilot/{label}/collection',jobs)
            rt.import_trajectories(agent,collected,d/'collection/tasks')
            rows=agent.memory.rows();n=len(rows)
            if n>=256:
                raw,_=agent.compute_targets();adv=(raw-raw.mean())/raw.std(unbiased=False).clamp_min(1e-8)
                rng=np.random.RandomState(9000+source)
                batches=[rng.choice(n,256,replace=False) for _ in range(plan()['online_steps_each_network_per_group'])]
                probe=rows[:min(n,256)]
                before=responses(agent.actor_network,probe,adv[:len(probe)].numpy())
                rt.seed_all(12000+source)
                update,grads=bounded_learn(agent,batches,d/'update')
                after=responses(agent.actor_network,probe,adv[:len(probe)].numpy())
                roundtrip_result=roundtrip(agent,cfg,d,probe)
                record=dict(model=label,source_seed=source,variant=variant,transitions=n,
                    actor_steps=8,critic_steps=8,full_buffer_advantage_std=float(raw.std(unbiased=False)),
                    before=before,after=after,update=update,roundtrip=roundtrip_result)
            else:
                # No extra episodes or smaller batch fallback when sampling is insufficient.
                rt.save_checkpoint(d/'latest.pt',agent,cfg,progress={'insufficient_samples':True})
                export_actor(d/'actor.pt',agent,cfg)
                record=dict(model=label,source_seed=source,variant=variant,transitions=n,actor_steps=0,critic_steps=0,
                    status='insufficient_samples_no_update')
            summary.append(record);save(OUT/'pilot/summary.json',dict(complete=len(summary)==4,results=summary))
            print('TRAIN END '+json.dumps(record),flush=True)
            del agent,rows
    check_code()


def evaluate():
    check_code();training=json.loads((OUT/'pilot/summary.json').read_text());assert training['complete']
    jobs=[]
    for name in plan()['development_instances']:
        for solver in [0,1]:
            for source in [11,22]:
                for variant in (['original','tail8'] if solver==0 else ['tail8','original']):
                    label=f'{variant}-s{source}';i=len(jobs)
                    job=make_job(f'eval-{i:03d}-{name}-q{solver}-{label}',name,solver,'eval',source,
                        OUT/'pilot'/label/'actor.pt',sampling_key=f'eval-{name}-q{solver}')
                    job.update(diagnostic_variant=variant,diagnostic_source_seed=source)
                    jobs.append(job)
    assert len(jobs)==40
    result=execute('evaluation',jobs)
    assert all(x.get('callback_seconds',{}).get('critic',0.)==0. for x in result)
    save(OUT/'evaluation/verification.json',dict(status='passed',tasks=40,actor_only=True,
        critic_seconds_zero=True,test_instances_used=0,solver_seeds=[0,1],development_instances=plan()['development_instances']))
    check_code()
