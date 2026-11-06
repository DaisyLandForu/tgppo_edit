"""Monitor the authorized evaluation, apply the rule, and conditionally confirm.

Never launches training, a GPU job, a full66 sweep, or an evaluation retry.
"""
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import tarfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=ROOT/'tgppo/runs/light-actor-20261105'
EVAL=RUN/'cpu-recovery-20261105'
VARIANTS=('b0','b0_bc','mlp','mlp_bc','set','set_bc','isab','isab_bc')


def read(p):return json.loads(Path(p).read_text())


def save(value):
    p=HERE/'progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def archive_raw():
    if not (EVAL/'LOCAL_RAW_OUTPUT.json').exists():return
    location=read(EVAL/'LOCAL_RAW_OUTPUT.json');folder=Path(location['local_directory'])
    files=sorted(folder.glob('t*.json'));destination=EVAL/'cached_raw_results.tar'
    if not files or destination.exists():return
    temporary=destination.with_suffix('.tmp')
    with tarfile.open(temporary,'w') as archive:
        for path in files:archive.add(path,arcname=path.name,recursive=False)
    temporary.replace(destination)


def main():
    lock=(HERE/'continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(pid=os.getpid(),host=os.uname().nodename,phase='waiting_for_cpu_evaluation',
        training=False,gpu=False,cross_instance_started=False,results_directory=str(EVAL/'results'))
    last=None;seen={}
    try:
        while True:
            state=read(EVAL/'status.json') if (EVAL/'status.json').exists() else {}
            budget=read(EVAL/'evaluation/budget.json');latest=budget['attempts'][-1]
            elapsed=sum(a.get('elapsed_seconds',min(a['allocated_seconds'],max(0,time.time()-a['started_epoch']))) for a in budget['attempts'])
            fresh_attempt='elapsed_seconds' not in latest and state.get('started_epoch',0)<latest['started_epoch']
            phase='bootstrapping_remaining_cpu_budget' if fresh_attempt else state.get('status','starting')
            cohorts=list((EVAL/'evaluation/episode-0100').glob('*'))
            committed=sum(10 for c in cohorts for v in VARIANTS for s in (11,22)
                if (c/('{}-s{}'.format(v,s))/'selected_results.json').exists())
            native=10 if any((c/'native.json').exists() or (c/'fresh-native/selected_results.json').exists() for c in cohorts) else 0
            for c in cohorts:
                for path in (c/'cached-global-000').glob('t*.json'):
                    if path.name in seen:continue
                    with path.open('rb') as stream:
                        stream.seek(0,2);stream.seek(max(0,stream.tell()-4096));tail=stream.read()
                    flags=re.findall(rb'"valid"\s*:\s*(true|false)',tail)
                    if flags:seen[path.name]=(flags[-1]==b'true','-native-' in path.name)
            status.update(evaluation_status=phase,committed_model_tasks=committed,
                valid_model_tasks_raw=sum(ok and not nat for ok,nat in seen.values()),
                valid_native_tasks_raw=sum(ok and nat for ok,nat in seen.values()),
                invalid_raw_tasks=sum(not ok for ok,nat in seen.values()),
                native_tasks=native,charged_seconds=elapsed,remaining_seconds=max(0,5400-elapsed),updated_epoch=time.time())
            save(status)
            key=(phase,committed,native,len(seen))
            if key!=last:print(json.dumps(status,ensure_ascii=False),flush=True);last=key
            if state.get('status')=='complete' and (EVAL/'results/complete.json').exists():break
            overrun='elapsed_seconds' not in latest and time.time()-latest['started_epoch']>latest['allocated_seconds']+60
            if (state.get('status')=='failed' and not fresh_attempt) or overrun:
                archive_raw()
                status.update(phase='evaluation_incomplete_no_cross_instance',stopped=True);save(status);return
            time.sleep(30)
        archive_raw();status['phase']='development_analysis';save(status)
        with (HERE/'decision.log').open('w') as log:
            result=subprocess.run(['/usr/bin/python3',str(HERE/'decide.py'),'--results',str(EVAL/'results')],
                stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise RuntimeError('Read-only analysis failed; see decision.log')
        decision=read(HERE/'decision.json')
        status.update(phase='development_review_complete',decision_status=decision['status'],
            chosen=decision.get('chosen'),updated_epoch=time.time());save(status)
        if not decision['cross_instance_allowed']:
            status.update(phase='complete_no_candidate_passed',stopped=True);save(status);return
        cross=RUN/'cross-instance-8'
        if (cross/'status.json').exists():raise RuntimeError('Cross attempt already exists; inspect rather than duplicate')
        status.update(phase='cross_instance_evaluation',cross_instance_started=True);save(status)
        command=('source /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/env.sh\n'
                 'export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=""\ncd /tmp\n'
                 'exec "$TGPPO_PYTHON" -u -c \'import torch,numpy,scipy.signal,h5py,sys,runpy; '
                 'sys.path[:0]=["/home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/code",'
                 '"/home/duweiyue25/tgppo-transfer/research/tgppo/experiments/stage4/metrics_binding/src",'
                 '"/home/duweiyue25/tgppo-transfer/research/docs/methods/tgppo_light_actor_20261105/continuation"]; '
                 'runpy.run_path("/home/duweiyue25/tgppo-transfer/research/docs/methods/tgppo_light_actor_20261105/continuation/cross_run.py",run_name="__main__")\'')
        (HERE/'cross_bootstrap.json').write_text(json.dumps(dict(started_epoch=time.time(),budget_seconds=5400))+'\n')
        with (HERE/'cross_launcher.log').open('w') as log:
            result=subprocess.run(['/usr/bin/timeout','--signal=INT','--kill-after=20s','5380','/bin/bash','-c',command],
                stdout=log,stderr=subprocess.STDOUT)
        status.update(phase='cross_instance_analysis',cross_returncode=result.returncode);save(status)
        with (HERE/'cross_analysis.log').open('w') as log:
            result=subprocess.run(['/usr/bin/python3',str(HERE/'analyze_cross.py')],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise RuntimeError('Cross analysis failed; see cross_analysis.log')
        status.update(phase='complete_with_cross_instance_report',stopped=True,updated_epoch=time.time());save(status)
    except BaseException as exc:
        status.update(phase='stopped_with_error',error=repr(exc),stopped=True,updated_epoch=time.time());save(status);raise


if __name__=='__main__':main()
