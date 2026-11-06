"""Twenty real training tasks per group on V100; engineering cost evidence only."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time
import torch
import server
from project.utils.protocol import fingerprint, write_json, sha256


def pilot(root,c,run):
    info=server.preflight(root,c,'cuda')
    bases=server.baseline_bundle(root,c,run)
    server.require_target_evidence(root,c,run,info,require_pilot=False)
    directory=run/'throughput';directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'.pilot.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    probe=copy.deepcopy(c)
    probe.update(engineering_mode=True,max_episodes=len(c['train']),
                 milestones=[len(c['train'])],training_seeds=[11])
    probe.pop('model_overrides',None)
    identity=dict(parent_protocol_sha256=fingerprint(c),stack=info['stack'],gpu=info['gpu'])
    if (directory/'identity.json').exists() and server.read(directory/'identity.json')!=identity:
        raise ValueError('Pilot environment/protocol changed; retain existing evidence and use a new run root')
    write_json(directory/'identity.json',identity)
    write_json(directory/'probe_protocol.json',probe)
    # Reuse measured native rows: task limits, solver seeds, cutoffs and SCIP settings are unchanged.
    write_json(directory/'normalization/baselines.json',dict(protocol_sha256=fingerprint(probe),
        stack=info['stack'],baselines=bases,source=str(run/'normalization/baselines.json'),
        source_sha256=sha256(run/'normalization/baselines.json')))
    rows=[]
    for group in c['groups']:
        state=server.train(root,probe,directory,group,11,True,'cuda')
        out=server.group_dir(directory,group,11)
        selected=[]
        for path in sorted(out.glob('collection/e*/selected_results.json')):selected+=server.read(path)
        attempts=[server.read(path) for path in out.glob('attempts/*.json')]
        observed=sum(a['attempt_wall_seconds'] for a in attempts)
        covered={row['instance_name'] for row in selected}
        passed=(len(selected)==len(c['train']) and covered==set(c['train']) and
                all(row.get('valid') and not row['cuda_initialized'] for row in selected) and
                state['next_episode']==len(c['train']) and state['phase']=='ready' and
                len(state['history'])==len(c['train'])//c['batch_episodes'] and
                any(m['updated'] for m in state['history']))
        rows.append(dict(group=group,passed=passed,episodes_seen=state['next_episode'],
            trained_episodes=state['trained_episodes'],instances=sorted(covered),
            complete_ppo_updates=sum(bool(m['updated']) for m in state['history']),
            optimizer_steps=state['optimizer_steps'],
            observed_attempt_wall_seconds=observed,
            worker_cpu_seconds=sum(row['worker_cpu_seconds'] for row in selected),
            learner_update_wall_seconds=sum(m['learning_wall_seconds'] for m in state['history']),
            peak_cuda_allocated_bytes=max(m['peak_cuda_allocated_bytes'] for m in state['history']),
            peak_cuda_reserved_bytes=max(m['peak_cuda_reserved_bytes'] for m in state['history']),
            peak_worker_rss_kib=max(row['peak_rss_kib'] for row in selected),
            estimated_100_episode_seconds=observed*100/len(c['train']),
            estimated_500_episode_seconds=observed*500/len(c['train']),source=str(out)))
        torch.cuda.empty_cache()
    evidence=dict(identity,passed=all(row['passed'] for row in rows),rows=rows,
        scope='separate engineering pilot, not any part of the six formal 500-episode budgets',
        estimate_limit='initial-policy twenty-instance cycle, includes restart overhead; later decision counts and queue delays can differ')
    write_json(directory/'verification.json',evidence)
    print(json.dumps(evidence,indent=2),flush=True)
    if not evidence['passed']:raise RuntimeError('Representative cost pilot failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    args=parser.parse_args();pilot(args.root,server.plan(args.root,args.config),args.run_root)
