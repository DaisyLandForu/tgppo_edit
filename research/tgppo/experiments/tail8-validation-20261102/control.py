"""Cumulative one-CPU wall budget; no silent retries or detached workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

P=Path(__file__).resolve().parent
OUT=P.parents[2]/'docs/diagnostics/tgppo_20261102_end2end'


def save(path,x):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('tag');parser.add_argument('seconds',type=float);parser.add_argument('script')
    a=parser.parse_args();path=OUT/'budget.json'
    with (OUT/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads(path.read_text()) if path.exists() else dict(limit_seconds=3600,reserve_seconds=30,runs=[])
        assert ledger.get('status')!='closed'
        assert not any(r['tag']==a.tag or r['status']=='running' for r in ledger['runs']),'Inspect prior attempt; do not retry/reset'
        left=ledger['limit_seconds']-ledger['reserve_seconds']-sum(r['elapsed_seconds'] for r in ledger['runs'])
        cap=min(a.seconds,left);assert cap>5
        row=dict(tag=a.tag,status='running',cap_seconds=cap,script=a.script)
        ledger['runs'].append(row);save(path,ledger)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
            CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env.pop('PYTHONPATH',None)
        started=time.monotonic();child=None;code=1
        def interrupted(signum,frame):raise InterruptedError(signum)
        old={s:signal.signal(s,interrupted) for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]}
        try:
            with (OUT/f'{a.tag}.log').open('x') as log:
                child=subprocess.Popen([sys.executable,str(P/a.script)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=cap-5)
                except (subprocess.TimeoutExpired,InterruptedError):
                    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    code=124
        finally:
            for s,h in old.items():signal.signal(s,h)
            if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
            row.update(status='completed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-started)
            ledger['used_seconds']=sum(r['elapsed_seconds'] for r in ledger['runs']);save(path,ledger)
        print(json.dumps(row));raise SystemExit(code)


if __name__=='__main__':main()
