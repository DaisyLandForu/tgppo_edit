"""Stage5 end-of-experiment audit; never repairs or filters experimental outcomes."""
import json
from pathlib import Path
import subprocess
import torch
from project.runtime import load_agent,load_actor,read_checkpoint,cpu_seconds
from project.utils.protocol import sha256,write_json

ROOT=Path('/root/research');DOC=ROOT/'docs/stage5';RUN=ROOT/'tgppo/runs/stage5'
torch.set_num_threads(1)
c=json.loads((DOC/'execution_config.json').read_text());checks={};restoration=[]
checks['bridge_passed']=json.loads((DOC/'bridge_verification.json').read_text())['passed']
locked=json.loads((DOC/'locked_training_sources.json').read_text())
checks['training_sources_unchanged_since_launch']=all(Path(p).exists() and sha256(p)==h for p,h in locked.items())
upstream=json.loads((ROOT/'docs/stage1/upstream_file_hashes.json').read_text())
for relative in ('code/project/policy/actor.py','code/project/policy/critic.py','code/project/policy/modules.py','code/project/reward/reward.py','code/project/utils/settings.py'):
    checks['upstream_identical_'+relative]=sha256(ROOT/'tgppo/experiments/stage5'/relative)==upstream[relative]
checks['pristine_upstream_clean']=not subprocess.check_output(['git','-C',str(ROOT/'tgppo/repo'),'status','--porcelain']).strip()
checks['train_statistics_unchanged']=sha256(c['target_statistics'])==c['target_statistics_sha256']
checks['native_baselines_unchanged']=sha256(c['native_baselines'])==c['native_baselines_sha256']
for label in ('stage4','stage4_calibration'):
    ledger=json.loads((ROOT/'docs'/label/'artifact_hashes.json').read_text())['files']
    bad=[p for p,v in ledger.items() if not (ROOT/p).exists() or sha256(ROOT/p)!=v['sha256']]
    checks['old_'+label+'_artifacts_preserved']=not bad
    write_json(DOC/('preservation_'+label+'.json'),dict(total_files=len(ledger),changed_files=bad,passed=not bad))
reference_path=ROOT/'tgppo/runs/stage3/pilot-repaired/latest.pt'
checks['reference_baseline_checkpoint_preserved']=sha256(reference_path)=='1b912eec46bdf7ec3eaff04570c4aed0c3a57389b6d290bc14f3cc213442d4d1'
reference_config=read_checkpoint(reference_path)['config']
checks['frozen_config_copy_unchanged']=json.loads((RUN/'execution_config.json').read_text())==c
summaries=[];inits={};training_cpu=0.
for seed in c['training_seeds']:
    schedules=[]
    for group in c['groups']:
        name='{}-s{}'.format(group,seed);d=RUN/'training'/name
        summary=json.loads((d/'summary.json').read_text());summaries.append(summary);training_cpu+=summary['total_cpu_seconds']
        checks[name+'_valid_complete_update']=summary['within_budget'] and summary['complete_ppo_iterations']>=1 and summary['valid_episodes']==summary['episodes_attempted']
        checks[name+'_episode_and_cpu_caps']=summary['episodes_attempted']<=50 and summary['total_cpu_seconds']<=7200.
        inits[name]=json.loads((d/'initialization.json').read_text());schedules.append(json.loads((d/'task_schedule.json').read_text()))
        agent,p=load_agent(d/'latest.pt',restore_random_state=False)
        actor,_=load_actor(d/'latest.pt')
        expected_keys=('hidden_dim','num_layers','num_heads','dropout','actor_lr','critic_lr',
            'policy_clip','entropy_weight','gamma','gae_lambda','batch_size','n_epochs','reward_function','state_dims')
        checks[name+'_frozen_public_hyperparameters']=all(p['config'][k]==reference_config[k] for k in expected_keys)
        checks[name+'_same_frozen_online_protocol']=p['config']['online_protocol_sha256']==sha256(DOC/'execution_config.json')
        checks[name+'_mse_only']=agent.critic_network.loss=='mse'
        checks[name+'_worker_affinity_and_limits']=all(r['cpu_affinity']==[0,1,2,3] and r['time_limit_seconds']==120. for r in json.loads((d/'collection_results.json').read_text()))
        checks[name+'_full_critic_trainable_cpu']=all(t.requires_grad and t.device.type=='cpu' for t in agent.critic_network.parameters())
        checks[name+'_actor_only']=not hasattr(actor,'critic_network') and not hasattr(actor,'memory')
        checks[name+'_optimizers_restored']=bool(agent.actor_optimizer.state) and bool(agent.critic_optimizer.state)
        checks[name+'_raw_units']=float(agent.critic_network.target_mean)==torch.tensor(p['config']['target_mean'],dtype=torch.float32).item() and float(agent.critic_network.target_std)==torch.tensor(p['config']['target_std'],dtype=torch.float32).item()
        checks[name+'_empty_after_complete_update']=len(agent.memory)==0 and agent.memory.active is None
        selected=json.loads((d/'selected_checkpoints.json').read_text());all_points=json.loads((d/'checkpoints.json').read_text())
        for budget in (1800,3600,7200):
            record=selected[str(budget)];last=[r for r in all_points if r['total_cpu_seconds']<=budget][-1]
            checks[name+'_budget_selection_'+str(budget)]=all(record[k]==last[k] for k in last) and read_checkpoint(record['path'])['update_counter']==record['updates']
        checks[name+'_latest_final_hash']=sha256(d/'latest.pt')==sha256(selected['7200']['path'])==summary['latest_checkpoint_sha256']
        restoration.append(dict(group=group,training_seed=seed,policy_version=agent.policy_version,
            actor_optimizer_states=len(agent.actor_optimizer.state),critic_optimizer_states=len(agent.critic_optimizer.state),
            target_mean=float(agent.critic_network.target_mean),target_std=float(agent.critic_network.target_std),
            final_checkpoint_sha256=summary['latest_checkpoint_sha256']))
        del agent,actor,p
    checks['s{}_paired_schedule'.format(seed)]=schedules[0]==schedules[1]==schedules[2]
    for key in ('actor_sha256','common_critic_sha256','rng_torch_sha256'):
        checks['s{}_public_pairing_{}'.format(seed,key)]=len({inits['{}-s{}'.format(g,seed)][key] for g in c['groups']})==1
    checks['s{}_matched_capacity_Rprod_Rjoint'.format(seed)]=inits['Rprod-s'+str(seed)]['critic_parameters']==inits['Rjoint-s'+str(seed)]['critic_parameters']
checks['nine_training_runs']=len(summaries)==9
checks['main_cpu_cap']=training_cpu<=18*3600.
jobs=json.loads((RUN/'evaluation/jobs.json').read_text());rows=json.loads((RUN/'evaluation/jobs/results.json').read_text())
byid={r['job_id']:r for r in rows};checks['evaluation_complete_225']=len(jobs)==len(rows)==len(byid)==225
checks['final_matrix_135']=sum(j['nominal_budget_seconds']==7200 for j in jobs)==135
checks['early_matrix_90']=sum(j['nominal_budget_seconds']!=7200 for j in jobs)==90
checks['evaluation_all_valid']=all(r.get('valid') for r in rows)
checks['no_deployment_critic_cost']=all(r.get('callback_seconds',{}).get('critic',0.)==0. for r in rows)
checks['no_deployment_frontier_extraction']=all('search_extraction_wall_seconds' not in r for r in rows)
checks['same_120s_metric']=all(r.get('pdi_horizon_seconds')==120. and r.get('measurement_version')=='native-pdi-snapshot-horizon-v2' for r in rows)
checks['finite_pdi_intervals']=all(0<=r['pdi_horizon_lower']<=r['pdi_horizon_midpoint']<=r['pdi_horizon_upper']<=12000.+1e-6 for r in rows)
checks['paired_deployment_random_seeds']=all(len({j['sampling_seed'] for j in jobs if j['instance_name']==name and j['solver_seed']==ss and j['training_seed']==ts})==1
    for name in c['development'] for ss in c['solver_seeds'] for ts in c['training_seeds'])
manifest=json.loads((RUN/'evaluation/manifest.json').read_text())
checks['evaluation_checkpoints_preserved']=all(sha256(r['path'])==r['sha256'] for r in manifest)
write_json(DOC/'restoration.json',restoration)
write_json(DOC/'verification.json',dict(passed=all(checks.values()),checks=checks,
    training_cpu_seconds=training_cpu,audit_parent_cpu_seconds=cpu_seconds(),stage=5,stage6_started=False))
print(json.dumps(dict(passed=all(checks.values()),checks=len(checks),failed=[k for k,v in checks.items() if not v]),ensure_ascii=False))
if not all(checks.values()):raise RuntimeError('Stage5 audit failed; no result is promoted to accepted')
