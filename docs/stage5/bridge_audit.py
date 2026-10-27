"""Audit real online updates before authorizing the frozen matrix scheduler."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from project.runtime import load_agent,load_actor
from project.utils.protocol import write_json,sha256

ROOT=Path('/root/research');RUN=ROOT/'tgppo/runs/stage5';DOC=ROOT/'docs/stage5'
checks={};rows=[];inits=[];total_cpu=0.
for group in ('R0','Rprod','Rjoint'):
    d=RUN/'bridge'/(group+'-s11')
    s=json.loads((d/'summary.json').read_text());u=json.loads((d/'updates.json').read_text())[-1]
    r=json.loads((d/'restoration.json').read_text());init=json.loads((d/'initialization.json').read_text());inits.append(init)
    total_cpu+=s['total_cpu_seconds']
    checks[group+'_one_complete_real_update']=s['complete_ppo_iterations']==1 and s['checkpoint_episodes']==4 and s['valid_episodes']==4
    checks[group+'_parameters_changed']=u['actor_l2_change_from_initial']>0 and u['critic_encoder_l2_change_from_initial']>0
    checks[group+'_initial_ratio']=u['initial_ratio_max_error']<1e-4
    checks[group+'_optimizer_recovery']=r['optimizer_states_restored'] and r['checkpoint_schema_version']==3
    checks[group+'_full_encoder']=r['full_critic_trainable'] and init['all_critic_parameters_trainable']
    checks[group+'_budget']=s['within_budget']
    agent,payload=load_agent(d/'collected.pt',restore_random_state=False)
    adv,ret=agent.compute_targets();expected=[]
    for e in agent.memory.episodes:
        gae=0.;rs=e['transitions'];out=[]
        for i in reversed(range(len(rs))):
            row=rs[i];nonterminal=1.-row['done'];nv=rs[i+1]['value'] if i+1<len(rs) else e['bootstrap_value']
            delta=row['reward']+.97*nonterminal*nv-row['value']
            gae=delta+.97*.92*nonterminal*gae;out.append(gae+row['value'])
        expected+=out[::-1]
    checks[group+'_raw_gae_numpy']=np.allclose(ret.numpy(),expected,rtol=1e-5,atol=1e-4)
    checks[group+'_initial_raw_units']=np.allclose([r['value'] for r in agent.memory.rows()],float(agent.critic_network.target_mean),rtol=0,atol=1e-5)
    checks[group+'_terminal_tasks']=all(e['task_terminal'] and not e['collector_truncated'] and e['bootstrap_value']==0 for e in agent.memory.episodes)
    first=agent.memory.rows()[0]
    state={k:first[k] for k in ('cands_state','mip_state','node_state','remaining_time','frontier')}
    restored,_=load_agent(d/'latest.pt',restore_random_state=False)
    v1=restored.choose_action(**state,deterministic=True)[1]
    restored2,_=load_agent(d/'latest.pt',restore_random_state=False)
    v2=restored2.choose_action(**state,deterministic=True)[1]
    checks[group+'_raw_value_recovery']=v1==v2==r['restored_raw_value_first_real_state']
    checks[group+'_actor_only']=not hasattr(load_actor(d/'latest.pt')[0],'critic_network')
    sidecar_ok=True
    for path in sorted((d/'batch-000').glob('*.pkl')):
        p=pickle.load(path.open('rb'));rr=[r for e in p['episodes'] for r in e['transitions']]
        if group=='R0':sidecar_ok &= 'search_features' not in p and all(np.count_nonzero(r['frontier'])==0 for r in rr)
        else:
            ss=p['search_features']['rows'];sidecar_ok &= len(rr)==len(ss)
            for a,b in zip(rr,ss):
                sidecar_ok &= np.array_equal(a['frontier'],b['frontier']) and a['remaining_time']==b['remaining_time'] and a['action']==b['action']
    checks[group+'_extra_features_survive_real_chain']=bool(sidecar_ok)
    rows.append(dict(group=group,episodes=s['checkpoint_episodes'],samples=u['num_samples'],logical_optimizer_steps=u['num_updates'],
        cpu_seconds=s['total_cpu_seconds'],actor_l2_change=u['actor_l2_change_from_initial'],encoder_l2_change=u['critic_encoder_l2_change_from_initial'],
        initial_ratio_error=u['initial_ratio_max_error'],initial_raw_value=first['value'],restored_raw_value=v1,checkpoint_sha256=sha256(d/'latest.pt')))
for field in ('actor_sha256','common_critic_sha256','rng_torch_sha256'):
    checks['public_pairing_'+field]=len({i[field] for i in inits})==1
checks['parameter_count_Rprod_Rjoint']=inits[1]['critic_parameters']==inits[2]['critic_parameters']
audit=dict(passed=all(checks.values()),checks=checks,rows=rows,combined_bridge_cpu_seconds=total_cpu,
    scope='real finite-task rollout -> context export/import -> raw GAE -> full calibrated PPO update -> recovery; no solver-performance claim')
write_json(DOC/'bridge_verification.json',audit)
print(json.dumps(audit,ensure_ascii=False,indent=2))
if not audit['passed']:raise RuntimeError('Bridge failed; main matrix forbidden')
