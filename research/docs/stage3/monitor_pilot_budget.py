"""Stop a pilot update that cannot finish inside the remaining authorized CPU budget.

This only tightens the live process soft CPU limit. The existing SIGXCPU handler
discards an unfinished update and retains the last complete checkpoint. It never
changes the 1800-second authorized maximum, models, trajectories or optimizers.
"""
import json
from pathlib import Path
import os
import resource
import time
import argparse

ROOT=Path('/root/research')
PILOT=ROOT/'tgppo/runs/stage3/pilot'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair',action='store_true')
    parser.add_argument('--run-root',type=Path,default=ROOT/'tgppo/runs/stage3')
    args=parser.parse_args()
    pilot=args.run_root/'pilot'
    directory=args.run_root/'pilot-repaired' if args.repair else pilot
    progress=json.loads((directory/'learning-progress.json').read_text())
    if progress.get('state','updating')!='updating' or progress['minibatches_completed']<3:
        raise RuntimeError('No measured minibatch update cost yet')
    pids=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try: args=(path/'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        is_pilot=len(args)>2 and args[1]==b'/root/research/tgppo/experiments/stage3/code/stage3.py' and args[2]==b'pilot'
        is_repair=len(args)>1 and args[1]==b'/root/research/tgppo/experiments/stage3/code/repair_pilot.py'
        if (is_repair if directory.name=='pilot-repaired' else is_pilot):
            pids.append(int(path.name))
    if len(pids)!=1:
        raise RuntimeError('Expected exactly one live pilot process')
    pid=pids[0]
    fields=Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')',1)[1].split()
    actual_cpu=(float(fields[11])+float(fields[12]))/os.sysconf('SC_CLK_TCK')
    per_step=progress['learning_cpu_seconds']/progress['minibatches_completed']
    projected_remaining=(progress['expected_minibatches']-progress['minibatches_completed'])*per_step
    prior=(json.loads((pilot/'summary.json').read_text())['total_cpu_seconds']
           if directory.name=='pilot-repaired' else progress['worker_cpu_seconds'])
    remaining=1800.-prior-actual_cpu
    if projected_remaining <= 1.25*remaining:
        raise RuntimeError('No clear projected budget overrun; do not stop')
    limits=resource.prlimit(pid,resource.RLIMIT_CPU)
    revised_soft=int(actual_cpu)+3
    record=dict(reason='measured_large_batch_cost_cannot_finish_current_3_epoch_update_within_remaining_budget',
        pid=pid, progress=progress, actual_learner_cpu_seconds_at_decision=actual_cpu,
        projected_remaining_learning_cpu_seconds=projected_remaining,
        remaining_authorized_cpu_seconds=remaining, authorized_total_cpu_maximum=1800.,
        previous_process_soft_cpu_limit=limits[0], revised_process_soft_cpu_limit=revised_soft,
        action='tighten soft CPU limit; existing handler retains last complete checkpoint',
        decision_unix_time=time.time())
    (directory/'forecast_budget_guard.json').write_text(json.dumps(record,indent=2)+'\n')
    resource.prlimit(pid,resource.RLIMIT_CPU,(revised_soft,limits[1]))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
