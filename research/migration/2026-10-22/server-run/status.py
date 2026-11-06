#!/usr/bin/env python3
"""Report committed checkpoints, never infer completion from heartbeat/running flags."""
import argparse
import json
from pathlib import Path
import sys


def status(root,run):
    sys.path[:0]=[str(root/'tgppo/experiments/stage4/metrics_binding/src'),
                  str(root/'tgppo/experiments/server-20261022/code')]
    from project.runtime import read_checkpoint
    from project.utils.protocol import sha256
    result=[]
    for group in ('B0','R0','Rjoint'):
        for seed in (11,22):
            directory=run/'training'/('{}-s{}'.format(group,seed))
            row=dict(group=group,seed=seed,committed_episodes=0,trained_episodes=0,
                     status='not_started',milestones=[],process_state='not inferred; check Crater job state')
            checkpoint=directory/'latest.pt'
            if checkpoint.exists():
                payload=read_checkpoint(checkpoint);progress=payload['progress']
                row.update(status='checkpoint_available',committed_episodes=progress['next_episode'],
                    trained_episodes=progress['trained_episodes'],phase=progress['phase'],
                    policy_version=payload['policy_version'],optimizer_steps=progress['optimizer_steps'],
                    pending_transitions=payload['memory']['num_transitions'],checkpoint=str(checkpoint))
                row['milestones']=[int(path.stem.split('-')[1]) for path in sorted(directory.glob('episode-*.pt'))]
                if (directory/'preflight.json').exists():
                    execution=json.loads((directory/'preflight.json').read_text())
                    row.update(last_host=execution['host'],gpu=execution['gpu'],
                               platform_job_metadata=execution.get('platform_job_metadata',{}))
                summary_path=directory/'summary.json'
                if summary_path.exists():
                    summary=json.loads(summary_path.read_text())
                    complete=(summary['episodes_seen']==500 and progress['next_episode']==500 and
                              progress['phase']=='ready' and summary['complete_ppo_updates']>0 and
                              row['milestones']==[100,200,300,400,500] and
                              summary['final_checkpoint_sha256']==sha256(checkpoint))
                    row['status']='complete_500' if complete else 'summary_needs_inspection'
            result.append(row)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--run-root',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    print(json.dumps(status(root,args.run_root or root/'tgppo/runs/server6-20261022'),indent=2))
