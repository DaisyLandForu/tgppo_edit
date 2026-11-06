"""Transfer an existing worker to a revised sub-deadline; never rerun its steps."""
import fcntl
import json
import os
from pathlib import Path
import signal
import time

P=Path(__file__).resolve().parent
OUT=P.parents[2]/'docs/diagnostics/tgppo_20261102_end2end'


def save(path,x):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)


def main():
    budget=OUT/'budget.json';ledger=json.loads(budget.read_text())
    row=ledger['runs'][-1];assert row['tag']=='pilot' and row['status']=='running'
    elapsed_before=time.time()-budget.stat().st_mtime
    parents=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=path.read_bytes().split(b'\0')
        except OSError:continue
        if any(x.endswith(b'/control.py') for x in args) and b'pilot' in args:
            parents.append(int(path.parent.name))
    assert len(parents)==1,parents
    parent=parents[0]
    children=[int(x) for x in Path(f'/proc/{parent}/task/{parent}/children').read_text().split()]
    assert len(children)==1,children
    child=children[0]
    assert b'/pilot.py' in Path(f'/proc/{child}/cmdline').read_bytes()
    os.killpg(child,signal.SIGSTOP);os.kill(parent,signal.SIGSTOP)
    os.kill(parent,signal.SIGKILL)
    with (OUT/'.lock').open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        completed=sum(x.get('elapsed_seconds',0.) for x in ledger['runs'][:-1])
        cap=min(2400.,ledger['limit_seconds']-ledger['reserve_seconds']-completed)
        assert cap>elapsed_before
        event=dict(reason='Measured full-Actor throughput requires reallocating stage time; total 3600 s cap unchanged',
            original_parent_pid=parent,worker_pid=child,old_subcap_seconds=row['cap_seconds'],new_subcap_seconds=cap,
            elapsed_before_handoff=elapsed_before,optimizer_steps_restarted=0,solver_tasks_restarted=0)
        row.update(cap_seconds=cap,handoff=event)
        save(OUT/'pilot_supervisor_handoff.json',event);save(budget,ledger)
        started=time.monotonic();os.killpg(child,signal.SIGCONT)
        code=1
        try:
            while True:
                try:state=Path(f'/proc/{child}/stat').read_text().rsplit(')',1)[1].split()[0]
                except FileNotFoundError:state='gone'
                if state in ['Z','gone']:
                    summary=OUT/'pilot/summary.json'
                    log=(OUT/'pilot.log').read_text()
                    code=0 if summary.exists() and json.loads(summary.read_text()).get('complete') and 'Traceback (most recent call last)' not in log else 1
                    break
                if elapsed_before+time.monotonic()-started>cap-5:
                    os.killpg(child,signal.SIGTERM);time.sleep(2)
                    try:os.killpg(child,signal.SIGKILL)
                    except ProcessLookupError:pass
                    code=124;break
                time.sleep(1)
        finally:
            if code:
                try:os.killpg(child,signal.SIGKILL)
                except ProcessLookupError:pass
            row.update(status='completed' if code==0 else 'failed',exit_code=code,
                elapsed_seconds=elapsed_before+time.monotonic()-started)
            ledger['used_seconds']=sum(x['elapsed_seconds'] for x in ledger['runs']);save(budget,ledger)
        print(json.dumps(row));raise SystemExit(code)


if __name__=='__main__':main()
