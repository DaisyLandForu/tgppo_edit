"""Read-only audit of the four completed formal runs; writes only review artifacts."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json,math,statistics,datetime,csv
import torch
import ablation,server
from project.runtime import read_checkpoint
from project.utils.protocol import fingerprint,sha256,write_json

HERE=Path(__file__).resolve().parent
RUN=ablation.ROOT/'tgppo/runs/transformer-ablation-20261027'
SEEDS=(11,22)
torch.set_num_threads(1)

def finite_tree(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite_tree(v) for v in value)
    if isinstance(value,float):return math.isfinite(value)
    return True

def audit_rows(pair):
    variant,seed=pair;c,run=ablation.protocol(variant);out=server.group_dir(run,'B0',seed)
    checks={};summary=server.read(out/'summary.json');configuration=server.read(out/'config.json')
    info=server.read(out/'preflight.json');server.require_target_evidence(ablation.ROOT,c,run,info)
    checks['current_config_matches']=configuration==server.params(ablation.ROOT,c,'B0',seed,'cuda')
    checks['protocol_matches']=summary['protocol_sha256']==fingerprint(c)
    checks['formal_budget_complete']=summary['status']=='training_complete' and summary['episodes_seen']==summary['trained_episodes']==500 and summary['pending_transitions']==0
    checks['task_schedule_matches']=server.read(out/'task_schedule.json')==server.schedule(seed,c)
    updates=server.read(out/'updates.json');actual=[u for u in updates if u.get('updated')]
    checks['all125_collection_batches']=len(updates)==125 and [u['episodes_seen'] for u in updates]==list(range(4,501,4))
    checks['finite_update_metrics']=all(all(math.isfinite(u[k]) for k in ('actor_loss','critic_loss','entropy','clip_fraction','initial_ratio_max_error')) for u in actual)
    checks['probability_recompute_consistent']=all(u['initial_ratio_max_error']<1e-4 for u in actual)
    collected=[];pending=0;version=0;backlog=[]
    coverage=[];learner_rows=[];reward_rows=[]
    schedule=server.read(out/'task_schedule.json')
    for begin,u in zip(range(0,500,4),updates):
        base=out/'collection'/('e{:05d}'.format(begin));rows=server.read(base/'selected_results.json')
        expected=['e{:05d}-{}-s{}'.format(i,schedule[i],c['solver_seeds'][(i//len(c['train']))%len(c['solver_seeds'])]) for i in range(begin,begin+4)]
        coverage.append(len(rows)==4 and [r['job_id'] for r in rows]==expected and all(r['policy_version']==version for r in rows))
        pending+=sum(r['num_transitions'] for r in rows)
        coverage.append(u['num_samples']==pending)
        if u['updated']:
            version+=1;coverage.append(u['update_step']==version and u['policy_version']==version);pending=0
        else:
            coverage.append(u['num_samples']<configuration['batch_size'] and u['policy_version']==version)
            backlog.append(dict(episodes_seen=u['episodes_seen'],pending_transitions=u['num_samples']))
        collected+=rows
        learner_rows.append(dict(variant=variant,seed=seed,**u))
        reward_rows.extend(dict(variant=variant,seed=seed,episode_index=i,job_id=r['job_id'],instance=r['instance_name'],solver_seed=r['solver_seed'],total_reward=r['total_reward'],transitions=r['num_transitions'],proof_completed=r['proof_completed'],status=r['status']) for i,r in zip(range(begin,begin+4),rows))
    checks['complete_on_policy_coverage_and_backlog']=all(coverage) and pending==0
    checks['all500_valid_cpu_workers']=len(collected)==500 and all(r['valid'] and r['task_terminal'] and r['worker_cpu_complete'] and not r['cuda_initialized'] and r['mode']=='collect' and r['fallback_action_count']==0 for r in collected)
    checks['all20_instances_25_times']=Counter(r['instance_name'] for r in collected)=={n:25 for n in c['train']}
    checks['same120s_reward_horizon']=all(r['time_limit_seconds']==120. and r['reward_version']=='upstream_h3_t120' for r in collected)
    checks['transition_counts_accounted']=sum(u['num_samples'] for u in actual)==sum(r['num_transitions'] for r in collected)
    checks['terminal_transition_counts']=all(r['num_terminal_transitions']==(1 if r['num_transitions'] else 0) and r['reward_count']==r['num_transitions'] for r in collected)
    raw=[r for path in out.glob('collection/*/attempt-*/results.json') for r in server.read(path)]
    checks['no_invalid_solver_attempts']=bool(raw) and all(r.get('valid') for r in raw)
    attempts=[server.read(p) for p in out.glob('attempts/*.json')]
    checks['normal_training_exit']=bool(attempts) and all(a.get('attempt_finished') and a.get('training_finished') and not a.get('error') for a in attempts)
    checks['summary_update_counts_match']=summary['complete_ppo_updates']==len(actual)==version and summary['optimizer_steps']==sum(u['num_updates'] for u in actual)
    finite_rewards=[r['total_reward'] for r in collected];checks['finite_rewards']=all(math.isfinite(v) for v in finite_rewards)
    row=dict(variant=variant,seed=seed,checks=checks,stack=info['stack'],gpu=info['gpu'],summary=summary,
        formal_protocol=fingerprint(c),config_sha256=fingerprint(configuration),collection_batches=125,episodes=500,
        transitions=sum(r['num_transitions'] for r in collected),effective_updates=len(actual),optimizer_steps=summary['optimizer_steps'],
        skipped_backlog=backlog,training_wall_hours=sum(a['attempt_wall_seconds'] for a in attempts)/3600,
        learning_wall_hours=sum(u['learning_wall_seconds'] for u in updates)/3600,
        worker_cpu_hours=sum(r['worker_cpu_seconds'] for r in raw)/3600,
        peak_gpu_allocated_gib=max(u['peak_cuda_allocated_bytes'] for u in updates)/1024**3,
        peak_gpu_reserved_gib=max(u['peak_cuda_reserved_bytes'] for u in updates)/1024**3,
        max_probability_recompute_error=max(u['initial_ratio_max_error'] for u in actual),
        clipping_nonzero_updates=sum(u['clip_fraction']>0 for u in actual),
        zero_decision_episodes=sum(r['num_transitions']==0 for r in collected),
        per_instance_coverage=dict(Counter(r['instance_name'] for r in collected)),
        reward_first100_mean=statistics.mean(finite_rewards[:100]),reward_last100_mean=statistics.mean(finite_rewards[-100:]),
        loss_stats={k:dict(min=min(u[k] for u in actual),max=max(u[k] for u in actual),first20_mean=statistics.mean(u[k] for u in actual[:20]),last20_mean=statistics.mean(u[k] for u in actual[-20:])) for k in ('actor_loss','critic_loss','entropy')})
    return row,learner_rows,reward_rows


def main():
    ablation.check_pair();pairs=[(v,s) for v in ablation.VARIANTS for s in SEEDS]
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(audit_rows,pairs))
    rows=[r for r,_,_ in results]
    for row in rows:
        variant,seed=row['variant'],row['seed'];_,run=ablation.protocol(variant);out=server.group_dir(run,'B0',seed)
        checks=row['checks'];summary=row['summary']
        checks['latest_hash_matches']=sha256(out/'latest.pt')==summary['final_checkpoint_sha256']
        final=read_checkpoint(out/'latest.pt')
        checks['latest_config_and_progress']=(fingerprint(final['config'])==row['config_sha256'] and final['policy_version']==final['update_counter']==row['effective_updates'] and final['progress']['next_episode']==final['progress']['trained_episodes']==500 and final['progress']['phase']=='ready' and final['memory']['num_transitions']==0)
        checks['latest_complete_finite_state']=all(finite_tree(final[k]) for k in ('actor','critic','actor_optimizer','critic_optimizer')) and bool(final['actor_optimizer']['state']) and bool(final['critic_optimizer']['state']) and 'cuda' in final['rng']
        checks['correct_transformer_architecture']=all(any(k.startswith('transformer.') for k in final[net])==(variant=='transformer') for net in ('actor','critic'))
        milestones=[]
        for episode in (100,200,300,400,500):
            path=out/('episode-{:04d}.pt'.format(episode));digest=sha256(path);payload=read_checkpoint(path)
            valid=(fingerprint(payload['config'])==row['config_sha256'] and payload['progress']['phase']=='ready' and payload['progress']['next_episode']==episode and payload['policy_version']==payload['update_counter'])
            if episode==500:valid=valid and all(torch.equal(value,payload[net][k]) for net in ('actor','critic') for k,value in final[net].items())
            milestones.append(dict(episode=episode,sha256=digest,valid=valid,policy_version=payload['policy_version']))
            del payload
        checks['five_readable_stable_milestones']=all(m['valid'] for m in milestones)
        row['milestones']=milestones;row['verified']=all(checks.values())
        print(json.dumps({k:v for k,v in row.items() if k not in ('summary','milestones','loss_stats','per_instance_coverage')},ensure_ascii=False),flush=True)
        del final
    paired={}
    for key in ('training_wall_hours','learning_wall_hours','transitions','peak_gpu_allocated_gib'):
        a=[r[key] for r in rows if r['variant']=='transformer'];b=[r[key] for r in rows if r['variant']=='bimatching']
        paired[key]=dict(transformer_mean=statistics.mean(a),transformer_sample_sd=statistics.stdev(a),bimatching_mean=statistics.mean(b),bimatching_sample_sd=statistics.stdev(b),relative_percent=100*(statistics.mean(b)/statistics.mean(a)-1))
    receipt=dict(audited_at_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        complete_models=4,all_four_training_verified=all(r['verified'] for r in rows),expected_original_models=6,
        completed_training_seeds=list(SEEDS),missing_training_seeds=[33],selection_started=any(RUN.glob('*/training/*/selection.json')),
        scope='Training integrity and costs only; no development selection or held-out quality evaluation executed',
        same_target_stack=all(r['stack']==rows[0]['stack'] and r['gpu']==rows[0]['gpu'] for r in rows),rows=rows,comparison=paired)
    write_json(HERE/'training-audit.json',receipt)
    for filename,items in [('updates.csv',[r for _,u,_ in results for r in u]),('episodes.csv',[r for _,_,e in results for r in e])]:
        fields=list(dict.fromkeys(k for r in items for k in r))
        with (HERE/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fields,lineterminator="\n");writer.writeheader();writer.writerows(items)
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},indent=2),flush=True)
    if not receipt['all_four_training_verified'] or not receipt['same_target_stack']:raise RuntimeError('Training audit found failed checks')


if __name__=='__main__':main()
