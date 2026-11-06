"""Attribute observed training costs, without treating them as test performance."""
import json
import statistics
import numpy as np
import server
from project.utils.protocol import write_json,sha256,fingerprint
from diagnose_cause import HERE,RUN


def main():
    cause=server.read(HERE/'causal-results.json')
    if len(cause['summaries'])!=4 or not all(cause['checks'].values()):raise RuntimeError('All four causal verifications must pass first')
    rows=[];checks={};sources={}
    for variant in ('transformer','bimatching'):
        for seed in (11,22):
            out=RUN/variant/'training'/('B0-s'+str(seed));batches=[]
            for path in sorted(out.glob('collection/*/selected_results.json')):
                sources[str(path)]=sha256(path);batches.append(server.read(path))
            episodes=[r for batch in batches for r in batch]
            updates=server.read(out/'updates.json');attempts=[server.read(p) for p in out.glob('attempts/*.json')]
            wall=sum(a['attempt_wall_seconds'] for a in attempts)
            learning=sum(u['learning_wall_seconds'] for u in updates)
            critical=sum(max(r['worker_wall_seconds'] for r in b) for b in batches)
            inference=np.array([v for r in episodes for v in r['inference_seconds']])
            calls=sum(r['branch_count'] for r in episodes);transitions=sum(r['num_transitions'] for r in episodes)
            callback={k:sum(r['callback_seconds'][k] for r in episodes) for k in ('features','actor','critic','branch','callback')}
            item=dict(variant=variant,seed=seed,episodes=len(episodes),decisions=calls,transitions=transitions,
                training_wall_seconds=wall,learning_wall_seconds=learning,
                collection_critical_path_lower_bound_seconds=critical,
                scheduling_checkpoint_and_other_residual_seconds=wall-learning-critical,
                worker_summed_wall_seconds=sum(r['worker_wall_seconds'] for r in episodes),
                training_episode_summed_e2e_seconds=sum(r['end_to_end_wall_seconds'] for r in episodes),
                actor_forward_summed_seconds=float(inference.sum()),
                actor_forward_mean_ms=float(inference.mean()*1000),
                actor_forward_p50_ms=float(np.percentile(inference,50)*1000),
                actor_forward_p95_ms=float(np.percentile(inference,95)*1000),
                learning_seconds_per_1000_transitions=learning/transitions*1000,
                optimizer_steps=sum(u['num_updates'] for u in updates if u['updated']),
                learning_seconds_per_optimizer_step=learning/sum(u['num_updates'] for u in updates if u['updated']),
                callback_summed_seconds=callback,
                callback_mean_ms_per_decision={k:v/calls*1000 for k,v in callback.items()},
                time_limited_training_episodes=sum(r['status']=='timelimit' for r in episodes),
                proof_completed_training_episodes=sum(r['proof_completed'] for r in episodes),
                peak_gpu_allocated_bytes=max(u['peak_cuda_allocated_bytes'] for u in updates))
            key=variant+'/'+str(seed)
            checks[key+'_complete_valid_observations']=len(episodes)==500 and all(r['valid'] for r in episodes)
            checks[key+'_one_inference_per_transition']=len(inference)==calls==transitions
            checks[key+'_nonnegative_parent_residual']=item['scheduling_checkpoint_and_other_residual_seconds']>=0
            rows.append(item);print(json.dumps(item),flush=True)
    paired={}
    keys=('training_wall_seconds','learning_wall_seconds','decisions','actor_forward_mean_ms',
          'learning_seconds_per_1000_transitions','learning_seconds_per_optimizer_step',
          'collection_critical_path_lower_bound_seconds','scheduling_checkpoint_and_other_residual_seconds')
    for key in keys:
        a=[r[key] for r in rows if r['variant']=='transformer'];b=[r[key] for r in rows if r['variant']=='bimatching']
        paired[key]=dict(transformer_mean=statistics.mean(a),bimatching_mean=statistics.mean(b),
            relative_percent=100*(statistics.mean(b)/statistics.mean(a)-1))
    write_json(HERE/'cost-attribution.json',dict(checks=checks,rows=rows,comparison=paired,
        source_selected_result_hashes_sha256=fingerprint(sources),source_selected_results=sources,
        scope='Observed costs on 2000 training episodes; trajectories and numbers of decisions differ. Collection includes Critic inference and reward/memory work. No held-out solving-quality or matched-input speedup claim.'))
    if not all(checks.values()):raise RuntimeError('Cost accounting check failed')


if __name__=='__main__':main()
