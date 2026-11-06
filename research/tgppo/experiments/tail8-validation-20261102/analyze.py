"""Audit and summarize all bounded validation; no optimizer or solve calls."""
from probe_utils import *
from collections import defaultdict


def main():
    protected=verify_inputs();p=plan()
    offline=json.loads((OUT/'offline/summary.json').read_text());assert offline['complete']
    pilot=json.loads((OUT/'pilot/summary.json').read_text());assert pilot['complete']
    jobs=json.loads((OUT/'evaluation/jobs.json').read_text())
    evaluated=json.loads((OUT/'evaluation/tasks/results.json').read_text())
    native=json.loads((OUT/'native/tasks/results.json').read_text())
    native_index={(x['instance_name'],x['solver_seed']):x for x in native}
    tasks={j['job_id']:j for j in jobs};rows=[];index={};model_groups=defaultdict(list)
    for x in evaluated:
        assert x['valid'] and x['task_terminal'] and not x['cuda_initialized']
        assert x.get('callback_seconds',{}).get('critic',0.)==0
        j=tasks[x['job_id']];variant=j['diagnostic_variant'];source=j['diagnostic_source_seed']
        row=dict(model=f'{variant}-s{source}',variant=variant,source_seed=source,instance=x['instance_name'],solver_seed=x['solver_seed'],
            pdi=x['pdi_horizon_midpoint'],pdi_lower=x['pdi_horizon_lower'],pdi_upper=x['pdi_horizon_upper'],
            proof_completed=x['proof_completed'],status=x['status'],nodes=x['nnodes'],branches=x.get('branch_count',0),
            optimize_seconds=x['optimize_wall_seconds'],worker_seconds=x['worker_wall_seconds'],
            actor_seconds=x.get('callback_seconds',{}).get('actor',0.),
            callback_seconds=x.get('callback_seconds',{}).get('callback',0.),
            feature_seconds=x.get('callback_seconds',{}).get('features',0.),
            pdi_uncertainty=x['pdi_horizon_uncertainty'])
        assert all(np.isfinite(row[k]) for k in ['pdi','pdi_lower','pdi_upper','optimize_seconds','actor_seconds'])
        rows.append(row);index[(source,variant,row['instance'],row['solver_seed'])]=row;model_groups[row['model']].append(row)
    assert len(rows)==40
    pairs=[];instances=[]
    for name in p['development_instances']:
        for source in p['source_seeds']:
            a=[index[(source,'original',name,s)] for s in [0,1]]
            b=[index[(source,'tail8',name,s)] for s in [0,1]]
            original=float(np.mean([x['pdi'] for x in a]));tail=float(np.mean([x['pdi'] for x in b]))
            pairs.append(dict(instance=name,source_seed=source,original_pdi=original,tail8_pdi=tail,delta_pdi=tail-original,
                delta_lower=float(np.mean([x['pdi_lower'] for x in b]))-float(np.mean([x['pdi_upper'] for x in a])),
                delta_upper=float(np.mean([x['pdi_upper'] for x in b]))-float(np.mean([x['pdi_lower'] for x in a])),
                original_proofs=sum(x['proof_completed'] for x in a),tail8_proofs=sum(x['proof_completed'] for x in b),
                original_nodes=float(np.mean([x['nodes'] for x in a])),tail8_nodes=float(np.mean([x['nodes'] for x in b])),
                original_branches=sum(x['branches'] for x in a),tail8_branches=sum(x['branches'] for x in b),
                paired_tasks_both_no_actor=sum(x['branches']==0 and y['branches']==0 for x,y in zip(a,b))))
        pp=[x for x in pairs if x['instance']==name]
        baseline=[native_index[(name,s)] for s in [0,1]]
        instances.append(dict(instance=name,native_pdi=float(np.mean([x['pdi_horizon_midpoint'] for x in baseline])),
            original_pdi=float(np.mean([x['original_pdi'] for x in pp])),tail8_pdi=float(np.mean([x['tail8_pdi'] for x in pp])),
            tail8_minus_original=float(np.mean([x['delta_pdi'] for x in pp])),
            sources_favoring_tail8=sum(x['delta_pdi']<0 for x in pp),
            sources_resolved_favoring_tail8=sum(x['delta_upper']<0 for x in pp),
            original_branches=sum(x['original_branches'] for x in pp),tail8_branches=sum(x['tail8_branches'] for x in pp),
            paired_tasks_both_no_actor=sum(x['paired_tasks_both_no_actor'] for x in pp)))
    cost=[]
    for model,rr in sorted(model_groups.items()):
        branches=sum(x['branches'] for x in rr);optimize=sum(x['optimize_seconds'] for x in rr)
        cost.append(dict(model=model,tasks=len(rr),pdi_mean=float(np.mean([x['pdi'] for x in rr])),
            proof_rate=float(np.mean([x['proof_completed'] for x in rr])),branches=branches,
            no_actor_tasks=sum(x['branches']==0 for x in rr),
            actor_ms_per_branch=1000*sum(x['actor_seconds'] for x in rr)/branches if branches else None,
            callback_ms_per_branch=1000*sum(x['callback_seconds'] for x in rr)/branches if branches else None,
            actor_fraction=sum(x['actor_seconds'] for x in rr)/optimize,
            callback_fraction=sum(x['callback_seconds'] for x in rr)/optimize,
            nodes_per_wall_second=sum(x['nodes'] for x in rr)/optimize))
    rewards=[];reward_rows=[];reward_instances=[];collection_tasks=0
    for group in pilot['results']:
        folder=OUT/'pilot'/group['model']/'collection/tasks'
        collected=json.loads((folder/'results.json').read_text());collection_tasks+=len(collected)
        records=[]
        for x in collected:
            assert x['valid']
            rr=x['reward_diagnostics'];assert len(rr)==x['reward_count']
            assert abs(sum(t['reward'] for t in rr)-x['total_reward'])<1e-8
            records.extend(dict(t,instance=x['instance_name']) for t in rr)
            rewards.append(dict(model=group['model'],instance=x['instance_name'],solver_seed=x['solver_seed'],
                states=x['num_transitions'],total_reward=x['total_reward'],pdi=x['pdi_horizon_midpoint'],
                baseline_gap=(rr[0]['baseline_gap'] if rr else None),proof_completed=x['proof_completed']))
        steps=[x for x in records if not x['done']]
        reward_rows.append(dict(model=group['model'],rewards=len(records),nonterminal=len(steps),
            baseline_gap_zero_fraction=float(np.mean([x['baseline_gap']==0 for x in steps])) if steps else None,
            gap_level_saturated_fraction=float(np.mean([x['gap_level_saturated'] for x in steps])) if steps else None,
            reward_clipped_fraction=float(np.mean([abs(x['unclipped'])>1 for x in steps])) if steps else None,
            pdi_progress_positive=sum(x['progress_pdi']>0 for x in steps),
            weighted_nodes_mean=float(np.mean([x['weighted_terms']['nodes'] for x in steps])) if steps else None,
            weighted_gap_level_mean=float(np.mean([x['weighted_terms']['gap_level'] for x in steps])) if steps else None,
            weighted_pdi_level_mean=float(np.mean([x['weighted_terms']['pdi_level'] for x in steps])) if steps else None,
            weighted_progress_mean=float(np.mean([x['weighted_terms']['progress'] for x in steps])) if steps else None))
        for name in p['online_training_instances']:
            subset=[x for x in steps if x['instance']==name]
            reward_instances.append(dict(model=group['model'],instance=name,nonterminal_rewards=len(subset),
                baseline_gap_zero_fraction=float(np.mean([x['baseline_gap']==0 for x in subset])) if subset else None,
                gap_saturated_fraction=float(np.mean([x['gap_level_saturated'] for x in subset])) if subset else None,
                weighted_nodes_mean=float(np.mean([x['weighted_terms']['nodes'] for x in subset])) if subset else None,
                weighted_gap_mean=float(np.mean([x['weighted_terms']['gap_level'] for x in subset])) if subset else None,
                weighted_pdi_mean=float(np.mean([x['weighted_terms']['pdi_level'] for x in subset])) if subset else None,
                weighted_progress_mean=float(np.mean([x['weighted_terms']['progress'] for x in subset])) if subset else None))
    # Input symmetry check after actual coupled training, no optimization.
    observations=pickle.loads((OUT.parent/'tgppo_20261102_signal_localization/observations.pkl').read_bytes())
    observation=next(r for key,name,r in observations if 8<=len(r['cands_state'])<=64)
    invariance=[];microbench=[];anchors={}
    for key,name,rr in observations:
        if name not in anchors:anchors[name]=rr
    for group in pilot['results']:
        actor,_=rt.load_actor(OUT/'pilot'/group['model']/'actor.pt');net=actor.actor_network.eval()
        b=collate([observation],[0]);L=b['cands'].shape[1];perm=torch.arange(L-1,-1,-1)
        with torch.no_grad():
            base=net(b['cands'],b['node_state'],b['mip_state'],b['mask'])
            moved=net(b['cands'][:,perm],b['node_state'],b['mip_state'],b['mask'][:,perm])[:,perm]
            padded=torch.cat([b['cands'],torch.zeros(1,7,b['cands'].shape[-1])],1)
            mask=torch.cat([b['mask'],torch.ones(1,7,dtype=torch.bool)],1)
            result=net(padded,b['node_state'],b['mip_state'],mask)
        err=float((moved-base).abs().max());pad=float((result[:,:L]-base).abs().max())
        assert err<1e-6 and pad<1e-6 and float(result[:,L:].abs().sum())==0
        invariance.append(dict(model=group['model'],permutation_max_error=err,padding_max_error=pad,pad_probability=0.))
        for name,rr in anchors.items():
            sample=collate([rr],[0]);durations=[]
            with torch.no_grad():
                net(sample['cands'],sample['node_state'],sample['mip_state'],sample['mask'])
                for repeat in range(3):
                    begin=time.perf_counter()
                    net(sample['cands'],sample['node_state'],sample['mip_state'],sample['mask'])
                    durations.append(1000*(time.perf_counter()-begin))
            microbench.append(dict(model=group['model'],instance=name,candidates=len(rr['cands_state']),
                repeats=3,forward_ms_median=float(np.median(durations)),forward_ms_min=min(durations),forward_ms_max=max(durations)))
        next(x for x in cost if x['model']==group['model'])['fixed_state_forward_median_ms']=float(np.median(
            [x['forward_ms_median'] for x in microbench if x['model']==group['model']]))
    assert collection_tasks==16 and len(native)==14 and len(evaluated)==40
    online_actor_steps=sum(g['actor_steps'] for g in pilot['results']);online_critic_steps=sum(g['critic_steps'] for g in pilot['results'])
    assert online_actor_steps<=32 and online_critic_steps<=32
    files={'evaluation_rows.csv':rows,'paired_instances.csv':pairs,'instance_summary.csv':instances,
           'model_cost.csv':cost,'collection_rewards.csv':rewards,'reward_components.csv':reward_rows,'invariance.csv':invariance,
           'fixed_state_microbench.csv':microbench,'reward_components_by_instance.csv':reward_instances}
    for name,data in files.items():write_csv(OUT/name,data)
    result=dict(complete=True,solver_tasks=70,offline_optimizer_steps=16,online_optimizer_steps=online_actor_steps+online_critic_steps,
        total_optimizer_steps=16+online_actor_steps+online_critic_steps,models=cost,instances=instances,reward_components=reward_rows,
        source_seed_directions={str(seed):dict(mean_delta_pdi=float(np.mean([x['delta_pdi'] for x in pairs if x['source_seed']==seed])),
            instances_favoring_tail8=sum(x['delta_pdi']<0 for x in pairs if x['source_seed']==seed)) for seed in [11,22]},
        source_seed_directions_actor_participated={str(seed):dict(
            mean_delta_pdi=float(np.mean([x['delta_pdi'] for x in pairs if x['source_seed']==seed and (x['original_branches'] or x['tail8_branches'])])),
            instances=sum(bool(x['original_branches'] or x['tail8_branches']) for x in pairs if x['source_seed']==seed)) for seed in [11,22]},
        pdi_interval_max_width=max(x['pdi_uncertainty'] for x in rows),
        evaluation_tasks_without_actor=sum(x['branches']==0 for x in rows),
        measurement_resolved_pairs_favoring_tail8=sum(x['delta_upper']<0 for x in pairs),
        measurement_resolved_pairs_favoring_original=sum(x['delta_lower']>0 for x in pairs),
        protected_input_files=protected,task_level_quality_only=True,candidate_oracle_labels=False,test_set_tasks=0)
    assert verify_inputs()==protected and not torch.cuda.is_initialized()
    stack=json.loads((OUT/'runtime_stack.json').read_text())
    assert all(sha(Path(name))==digest for name,digest in stack['files'].items())
    save(OUT/'summary.json',result)
    save(OUT/'hardware.json',dict(cpu_model=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
        torch_version=torch.__version__,numpy_version=np.__version__,device='cpu'))
    save(OUT/'verification.json',dict(status='passed',protected_files_unchanged=protected,
        tasks_native=14,tasks_collection=16,tasks_evaluation=40,all_tasks_valid=True,
        actor_only=True,critic_evaluation_seconds_zero=True,roundtrip_and_symmetry_checked=True,
        total_optimizer_steps=result['total_optimizer_steps'],maximum_optimizer_steps=80,cuda_initialized=False,
        recorded_runtime_binary_hashes_unchanged=True))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
