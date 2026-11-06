"""Keep a live status page; finalize only stage5 after its frozen queue finishes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path('/root/research');DOC=ROOT/'docs/stage5';RUN=ROOT/'tgppo/runs/stage5'
STATUS=ROOT/'docs/阶段5_在线PPO实时状态_TGPPO.md'


def read(path):return json.loads(Path(path).read_text())

def write(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def live(state):
    text=['# 阶段 5 在线 PPO 实时状态','',
        '本页由阶段 5 监控进程更新。阶段 4 已验收；本阶段完成后停止，等待阶段 5 验收，不进入阶段 6。','',
        '当前调度状态：`{}`。'.format(state.get('state','starting')),
        '工程桥接：三组真实更新完成，40 项审计通过；46 项自动检查通过。','',
        '| 训练运行 | 进入完整更新的任务 | 完整 PPO 轮 | 已完成逻辑优化步 | 最近记录 CPU 核时 | 当前状态 |',
        '|---|---:|---:|---:|---:|---|']
    states=[]
    for seed in (11,22,33):
        for group in ('R0','Rprod','Rjoint'):
            d=RUN/'training'/('{}-s{}'.format(group,seed));name='{} / {}'.format(group,seed)
            summary=d/'summary.json';updates=d/'updates.json';progress=d/'learning-progress.json'
            episodes=iterations=steps=0;spent=0.;status='等待队列'
            if updates.exists():
                us=read(updates);completed=[u for u in us if u.get('updated')]
                if completed:episodes=completed[-1]['episodes_completed'];iterations=len(completed);steps=sum(u['num_updates'] for u in completed);spent=completed[-1]['total_cpu_seconds']/3600.
            if d.exists():status='采样 / 更新中'
            if progress.exists():
                p=read(progress);spent=max(spent,p['total_cpu_seconds']/3600.)
                status='更新批 {}：{}/{} 逻辑步（本轮完成后才计入模型）'.format(p['batch'],p['minibatches_completed'],p['expected_minibatches'])
            if summary.exists():
                s=read(summary);spent=s['total_cpu_seconds']/3600.;episodes=s['checkpoint_episodes'];iterations=s['complete_ppo_iterations'];steps=s['complete_optimizer_steps'];status='已停止：'+s['stop_reason']
            states.append(dict(group=group,seed=seed,checkpoint_episodes=episodes,complete_ppo_iterations=iterations,complete_optimizer_steps=steps,recorded_cpu_core_hours=spent,status=status))
            text.append('| {} | {} | {} | {} | {:.3f} | {} |'.format(name,episodes,iterations,steps,spent,status))
    evaluation=RUN/'evaluation/jobs';files=[f for f in evaluation.glob('*.json') if f.name!='results.json'] if evaluation.exists() else []
    text+=['','开发评估已落盘：{} / 225 条（最终 135＋早期预算点 90）。'.format(len(files)),'',
        '最近记录消耗未包含仍在运行、尚未落盘的工作进程。当前结果不是最终验收结论，不据此选择模型。','',
        '[冻结范围与工程证据](./阶段5_在线PPO执行记录_TGPPO_2026-10-21.md) · [调度JSON](../tgppo/runs/stage5/scheduler.json)']
    if (RUN/'scheduler_error.json').exists():text+=['','**队列发生异常，已停止推进；失败信息保留。** [错误记录](../tgppo/runs/stage5/scheduler_error.json)。需要检查后返修，不能称为阶段完成。']
    if (RUN/'stage5_completion.json').exists():text+=['','**阶段 5 已完成，等待验收。** [阶段 5 验收报告](./阶段5_在线PPO对照与验收报告_TGPPO_2026-10-21.md)。阶段 6 未开始。']
    temp=STATUS.with_suffix('.tmp');temp.write_text('\n'.join(text)+'\n');temp.replace(STATUS)
    write(RUN/'live_status.json',dict(stage=5,scheduler_state=state.get('state'),training=states,evaluation_rows_on_disk=len(files),stage6_started=False))


def package():
    work=ROOT/'tgppo/experiments/stage5'
    subprocess.run(['git','-C',str(work),'add','-N','.'],check=True)
    with (DOC/'stage5.patch').open('wb') as f:subprocess.run(['git','-C',str(work),'diff','--binary'],stdout=f,check=True)
    with (DOC/'audit.log').open('w') as log:subprocess.run([str(DOC/'python_env.sh'),str(DOC/'audit_results.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (DOC/'report.log').open('w') as log:subprocess.run(['/root/anaconda3/bin/python',str(DOC/'report_results.py')],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONPATH=''))
    # Snapshot immutable evidence; exclude live monitors and this own active log.
    paths=list((RUN/'training').rglob('*'))+list((RUN/'bridge').rglob('*'))+list((RUN/'evaluation').rglob('*'))
    paths+=list((work/'code').rglob('*.py'))+list((work/'tests').rglob('*.py'))+list(DOC.iterdir())
    paths+=[RUN/'stage5_completion.json',ROOT/'docs/阶段5_在线PPO对照与验收报告_TGPPO_2026-10-21.md']
    excluded={'artifact_hashes.json','finalizer.log','scheduler.log','scheduler.pid'}
    ledger={}
    for p in sorted(set(paths)):
        if not p.is_file() or p.name in excluded or p.suffix in ('.tmp','.pyc') or '__pycache__' in p.parts:continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        ledger[str(p.relative_to(ROOT))]=dict(sha256=h.hexdigest(),bytes=p.stat().st_size)
    write(DOC/'artifact_hashes.json',dict(schema=1,stage=5,files=ledger))
    print('Stage5 report and evidence packaged; stop before stage6.',flush=True)


def main():
    if (RUN/'finalizer_started.json').exists():raise FileExistsError('Finalizer already started')
    write(RUN/'finalizer_started.json',dict(pid=os.getpid(),stage=5,stop_before_stage6=True))
    while True:
        state=read(RUN/'scheduler.json') if (RUN/'scheduler.json').exists() else {}
        live(state)
        if (RUN/'scheduler_error.json').exists():raise RuntimeError('Queue failed; retained all evidence')
        if state.get('state')=='experiments_finished_report_pending':break
        time.sleep(15.)
    package();live(dict(state='stage5_delivered_pending_acceptance'))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(RUN/'finalizer_error.json',dict(error=traceback.format_exc(),stage=5,stage6_started=False))
        raise
