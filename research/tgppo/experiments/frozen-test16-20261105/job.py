"""One Custom allocation; shared lock, fixed total cap, archive and report on exit."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import tarfile
import time

from common import BUDGET, DOCS as REPORT_ROOT, RUN, PACKAGE, output, read, sha, specs, verify, write

SHARD = None
OUT = None
DOCS = None


def archive_results():
    # /tmp can disappear with the Custom pod. Persist JSON and tar on shared storage.
    rows = {}
    if (OUT / 'evaluation_rows.json').exists():
        rows.update((r['job_id'], r) for r in read(OUT / 'evaluation_rows.json'))
    files = sorted((OUT / 'raw').glob('t*.json'))
    if files:
        temporary = OUT / 'raw_results.tar.tmp'
        with tarfile.open(temporary, 'w') as archive:
            for path in files:
                archive.add(path, arcname=path.name, recursive=False)
                try:
                    row = read(path)
                    rows[row['job_id']] = row
                except (ValueError, KeyError):
                    pass  # Preserve the invalid raw file; analysis reports missing/invalid.
        temporary.replace(OUT / 'raw_results.tar')
    write(OUT / 'evaluation_rows.json', sorted(rows.values(), key=lambda r: r['job_id']))
    return len(rows)


def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Also remove any descendants if the engine already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def analyze():
    return subprocess.call([sys.executable, str(PACKAGE / 'analyze.py'), '--shard', str(SHARD)])


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    with (OUT / 'custom.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another Custom already owns this evaluation; no duplicate tasks launched.', flush=True)
            return 2
        if (OUT / 'bootstrap.json').exists():
            prior = read(OUT / 'job_state.json') if (OUT / 'job_state.json').exists() else {'status': 'interrupted'}
            print(json.dumps(prior, ensure_ascii=False, indent=2))
            print('Existing attempt preserved. No automatic retry or budget reset. Use status/summarize only.')
            return 0 if prior.get('status') == 'complete' else 2
        started_epoch, started_clock = time.time(), time.monotonic()
        bootstrap = dict(started_epoch=started_epoch, budget_seconds=BUDGET, workers=4,
            training_steps=0, gpu=False, automatic_retry=False, requested_tasks=72, shard=SHARD,
            authorization='User approved four CPU shards for frozen16-instance test subset',
            queue_time_excluded=True, scope='new independent evaluation budget; no old training deadline reused')
        write(OUT / 'bootstrap.json', bootstrap)
        state = dict(bootstrap, status='preparing', supervisor_pid=os.getpid(), host=os.uname().nodename)
        write(OUT / 'job_state.json', state)
        process = None
        error = None

        def interrupted(*args):
            raise RuntimeError('Custom interrupted or bounded execution time reached')

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGALRM, interrupted)
        signal.setitimer(signal.ITIMER_REAL, BUDGET - 25)
        try:
            state['input_verification'] = verify(SHARD)
            write(OUT / 'job_state.json', state)
            print('Starting frozen test shard:64 model + 8 fresh native tasks, cap50min.', flush=True)
            print('Persistent log: ' + str(OUT / 'runner.log'), flush=True)
            with (OUT / 'runner.log').open('w') as log:
                process = subprocess.Popen(['bash', str(PACKAGE / 'worker.sh')],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ,TGPPO_TEST16_SHARD=str(SHARD)))
                state.update(status='running', engine_pid=process.pid)
                write(OUT / 'job_state.json', state)
                process.wait(timeout=max(.1, BUDGET - 45 - (time.monotonic() - started_clock)))
            if process.returncode:
                raise RuntimeError('Evaluation engine exit code {}'.format(process.returncode))
            engine = read(OUT / 'status.json')
            if engine['status'] != 'complete' or engine.get('valid_tasks') != 72:
                raise RuntimeError('Evaluation not complete; preserve partial results')
            state['final_input_verification'] = verify(SHARD)
            state['engine_complete'] = True
        except BaseException as exc:
            error = repr(exc)
            state.update(status='failed_or_interrupted', error=error)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            # Finalization is bounded too; leave enough of the 3000s for persistence.
            signal.setitimer(signal.ITIMER_REAL, max(.1, BUDGET - (time.monotonic() - started_clock)))
            try:
                stop(process)
                state['persistent_rows'] = archive_results()
                state.update(status='analyzing', error=error,
                    charged_wall_seconds=time.monotonic() - started_clock)
                write(OUT / 'job_state.json', state)
                remaining = max(.1, BUDGET - (time.monotonic() - started_clock) - 2)
                result = subprocess.run([sys.executable, str(PACKAGE / 'analyze.py'), '--shard', str(SHARD)], timeout=remaining)
                complete = (read(DOCS / 'analysis.json').get('status') == 'complete') if (DOCS / 'analysis.json').exists() else False
                state.update(status='complete' if error is None and result.returncode == 0 and complete else 'incomplete',
                    analysis_exit_code=result.returncode)
            except BaseException as exc:
                state.update(status='incomplete', finalization_error=repr(exc))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                state['charged_wall_seconds'] = time.monotonic() - started_clock
                write(OUT / 'job_state.json', state)
                if not (DOCS / 'REPORT.md').exists():
                    (DOCS / 'REPORT.md').write_text('# 冻结测试分片：执行未完整\n\n'
                        'Custom已停止，没有自动重试或扩展训练。持久状态：{}；日志：{}。\n\n'
                        '错误：{}。保留共享盘记录，修复汇总无需新增求解。\n'.format(
                            OUT / 'job_state.json', OUT / 'runner.log',
                            state.get('finalization_error', state.get('error'))))
                remaining_global=max(.1,BUDGET-(time.monotonic()-started_clock)-2)
                try:
                    subprocess.run([sys.executable,str(PACKAGE/'analyze.py')],timeout=remaining_global,check=True)
                except BaseException as exc:
                    state['global_summary_note']=repr(exc)
                state['charged_wall_seconds']=time.monotonic()-started_clock
                write(OUT/'job_state.json',state)
                print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
                print('Report: ' + str(DOCS / 'REPORT.md'), flush=True)
        return 0 if state['status'] == 'complete' else 1


def main():
    global SHARD, OUT, DOCS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','status','summarize','check'),nargs='?',default='run')
    parser.add_argument('--shard',type=int,choices=range(4))
    args=parser.parse_args()
    if args.command=='run' and args.shard is None:
        parser.error('run requires --shard0/1/2/3')
    if args.shard is not None:
        SHARD=args.shard;OUT=output(SHARD);DOCS=REPORT_ROOT/('shard-'+str(SHARD))
    if args.command=='check':
        print(json.dumps(verify(args.shard),ensure_ascii=False,indent=2));return 0
    if args.command=='status':
        status={}
        for shard in (range(4) if args.shard is None else [args.shard]):
            folder=output(shard);path=folder/'job_state.json'
            item=read(path) if path.exists() else dict(status='not_started')
            if (folder/'evaluation_rows.json').exists():item['persistent_rows']=len(read(folder/'evaluation_rows.json'))
            item['raw_files_currently_visible']=len(list((folder/'raw').glob('t*.json')))
            status[str(shard)]=item
        print(json.dumps(status,ensure_ascii=False,indent=2));return 0
    if args.command=='run':return run()
    command=[sys.executable,str(PACKAGE/'analyze.py')]
    if args.shard is not None:
        if not OUT.exists():print('No attempt yet; no solves started.');return 2
        with (OUT/'custom.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:print('Shard is active; automatic report will run at exit.');return 2
            archive_results();command+=['--shard',str(args.shard)]
            return subprocess.call(command)
    return subprocess.call(command)


if __name__=='__main__':sys.exit(main())
