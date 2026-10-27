"""Run only accepted stage5 work. Stop on any failure; never enter stage6."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/root/research');DOC=ROOT/'docs/stage5';RUN=ROOT/'tgppo/runs/stage5'

def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    audit=json.loads((DOC/'bridge_verification.json').read_text())
    if not audit['passed']:raise RuntimeError('Real bridge not passed')
    if (RUN/'scheduler.json').exists():raise FileExistsError('Refuse automatic rerun')
    config=json.loads((DOC/'execution_config.json').read_text())
    pending=[(g,s) for s in config['training_seeds'] for g in config['groups']]
    active=[];finished=[];handles=[]
    os.sched_setaffinity(0,config['affinity'])
    start=time.time()
    try:
        while pending or active:
            while pending and len(active)<config['concurrent_training_runs']:
                g,s=pending.pop(0);log=(DOC/('train-{}-s{}.log'.format(g,s))).open('x');handles.append(log)
                process=subprocess.Popen([str(DOC/'run_env.sh'),'train','--group',g,'--seed',str(s)],
                    cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.append(dict(group=g,seed=s,process=process,started=time.time()))
            for a in list(active):
                code=a['process'].poll()
                if code is not None:
                    row=dict(group=a['group'],seed=a['seed'],exitcode=code,wall_seconds=time.time()-a['started'])
                    finished.append(row);active.remove(a)
                    print(json.dumps(row),flush=True)
                    if code:raise RuntimeError('Training failure; all records retained: '+str(row))
            write(RUN/'scheduler.json',dict(stage=5,state='training',pending=pending,
                active=[dict(group=a['group'],seed=a['seed'],pid=a['process'].pid) for a in active],finished=finished,
                wall_seconds=time.time()-start,stop_before_stage6=True))
            time.sleep(2.)
        p=ROOT/'docs/stage1/protocol_draft.json';protocol=json.loads(p.read_text())
        protocol['online_main_matrix_started']=True;protocol['stage5_progress']='nine_training_runs_complete_evaluation_running';write(p,protocol)
        write(RUN/'scheduler.json',dict(stage=5,state='evaluating',finished=finished,wall_seconds=time.time()-start,stop_before_stage6=True))
        with (DOC/'evaluation.log').open('x') as log:
            subprocess.run([str(DOC/'run_env.sh'),'evaluate'],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
        write(RUN/'scheduler.json',dict(stage=5,state='experiments_finished_report_pending',finished=finished,wall_seconds=time.time()-start,stop_before_stage6=True))
        print('Stage5 experiments finished; reporting/audit still required. Stage6 forbidden.',flush=True)
    except BaseException as e:
        for a in active:
            try:os.killpg(a['process'].pid,15)
            except ProcessLookupError:pass
        write(RUN/'scheduler_error.json',dict(error=repr(e),finished=finished,stop_before_stage6=True))
        raise
    finally:
        for h in handles:h.close()

if __name__=='__main__':main()
