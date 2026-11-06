"""Single-CPU subprocess launcher with a cumulative hard wall-time budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

OUT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    assert command and args.max_seconds>0
    (OUT/'logs').mkdir(exist_ok=True)
    p=OUT/'compute_budget.json'
    ledger=json.loads(p.read_text()) if p.exists() else dict(limit_seconds=600.,preparation_reserve_seconds=10.,runs=[])
    assert ledger.get('status')!='closed','This accepted scope is closed; a new stage requires a new authorized ledger'
    assert not any(r.get('status')=='running' for r in ledger['runs']),'An existing run requires audit'
    used=sum(r['elapsed_seconds'] for r in ledger['runs'])
    remaining=ledger['limit_seconds']-ledger['preparation_reserve_seconds']-used
    assert remaining>0,'No remaining authorized compute budget'
    timeout=min(args.max_seconds,remaining)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
        env[key]='1'
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    entry=dict(tag=args.tag,command=command,cpu_affinity=[cpu],timeout_seconds=timeout,
               started_at_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),status='running')
    ledger['runs'].append(entry);p.write_text(json.dumps(ledger,indent=2)+'\n')
    start=time.perf_counter()
    code=1
    try:
        with (OUT/'logs'/f'{args.tag}.log').open('w') as f:
            completed=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        code=completed.returncode;entry.update(status='completed' if code==0 else 'failed',exit_code=code)
    except subprocess.TimeoutExpired:
        entry.update(status='budget_timeout',exit_code=124);code=124
    finally:
        entry['elapsed_seconds']=time.perf_counter()-start
        ledger['used_seconds']=sum(r['elapsed_seconds'] for r in ledger['runs'])
        ledger['remaining_seconds']=max(0.,ledger['limit_seconds']-ledger['preparation_reserve_seconds']-ledger['used_seconds'])
        p.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(dict(tag=args.tag,status=entry['status'],elapsed_seconds=entry['elapsed_seconds'],
                         cumulative_seconds=ledger['used_seconds'],remaining_seconds=ledger['remaining_seconds'])))
    raise SystemExit(code)


if __name__=='__main__':main()
