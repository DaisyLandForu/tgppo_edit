"""Resume the saved pre-update batch after the padded-batch memory failure.

No instance is resampled; prior failed computation counts toward the same cap.
Microbatches retain the original outer batch, targets and optimizer step count.
"""
import json
from pathlib import Path
import resource
import signal
import shutil
import traceback
import argparse
from project.runtime import load_agent,save_checkpoint,cpu_seconds
from project.utils.protocol import write_json
from stage3 import PilotCPULimit,_cpu_limit

ROOT=Path('/root/research/tgppo/runs/stage3')


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=ROOT)
    args=parser.parse_args()
    ROOT=args.run_root.resolve()
    source=ROOT/'pilot'
    out=ROOT/'pilot-repaired'
    if out.exists(): raise FileExistsError(str(out))
    out.mkdir()
    previous=json.loads((source/'summary.json').read_text())
    prior_cpu=previous['total_cpu_seconds']
    resource.setrlimit(resource.RLIMIT_AS,(32*1024**3,resource.getrlimit(resource.RLIMIT_AS)[1]))
    agent,payload=load_agent(source/'collected-batch-01.pt')
    config=dict(payload['config'],learner_microbatch_size=16,
                microbatch_semantics='sort inside fixed logical batch; sample-weighted gradient accumulation; one optimizer step')
    shutil.copyfile(source/'latest.pt',out/'latest.pt')
    write_json(out/'config.json',config)
    write_json(out/'provenance.json',dict(source_checkpoint=str(source/'collected-batch-01.pt'),
        source_failure=str(source/'learning_error.json'),prior_total_cpu_seconds=prior_cpu,
        trajectories_reused_without_resampling=True,logical_batch_size=256,microbatch_size=16,
        dropout_note='Critic training dropout stays .05; gradient equivalence tested with dropout off; stochastic masks need not be bitwise identical'))
    signal.signal(signal.SIGXCPU,_cpu_limit)
    original=resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU,(int(1800.-prior_cpu-5.),original[1]))
    begin=cpu_seconds()
    history=[]
    reason='repaired_saved_batch_complete_no_new_episodes'
    def progress(data):
        write_json(out/'learning-progress.json',dict(data,learning_cpu_seconds=cpu_seconds()-begin,
                    total_cpu_seconds=prior_cpu+cpu_seconds()))
    try:
        metrics=agent.learn(progress_callback=progress,microbatch_size=16)
        metrics.update(batch=1,episodes_completed=8,learning_cpu_seconds=cpu_seconds()-begin,
            worker_cpu_seconds=previous['worker_cpu_seconds'],
            learner_cpu_seconds=previous['learner_cpu_seconds']+cpu_seconds())
        history.append(metrics)
        save_checkpoint(out/'latest.pt',agent,config,dict(next_episode=8,
            worker_cpu_seconds=previous['worker_cpu_seconds'],
            learner_cpu_seconds=previous['learner_cpu_seconds']+cpu_seconds(),stage=3))
        print(json.dumps(metrics,sort_keys=True),flush=True)
    except PilotCPULimit:
        reason='cpu_budget_partial_update_discarded_last_complete_checkpoint_retained'
        write_json(out/'interrupted_update.json',dict(error=traceback.format_exc(),
                  total_cpu_seconds=prior_cpu+cpu_seconds()))
    except Exception:
        reason='repair_error_last_complete_checkpoint_retained'
        write_json(out/'learning_error.json',dict(error=traceback.format_exc()))
    finally:
        resource.setrlimit(resource.RLIMIT_CPU,original)
    write_json(out/'updates.json',history)
    valid,_=load_agent(out/'latest.pt')
    write_json(out/'summary.json',dict(stop_reason=reason,episodes_attempted=8,
        checkpoint_episodes=8 if history else 4,completed_updates=valid.update_counter,
        policy_version=valid.policy_version,cpu_cap_seconds=1800.,
        total_cpu_seconds=prior_cpu+cpu_seconds(),within_budget=prior_cpu+cpu_seconds()<=1800.,
        worker_cpu_seconds=previous['worker_cpu_seconds'],
        learner_cpu_seconds=previous['learner_cpu_seconds']+cpu_seconds(),
        prior_failed_run_cpu_seconds=prior_cpu,repair_cpu_seconds=cpu_seconds(),
        peak_learner_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
