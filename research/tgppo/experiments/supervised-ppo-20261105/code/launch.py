"""Frozen MLP+BC, original PPO versus persistent expert supervision, two seeds."""
import argparse
import contextlib
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time
import uuid
import torch
import server
from project import runtime as rt
from project.utils.protocol import fingerprint,sha256,write_json,validate_baseline
from expert_data import OUTPUT

HERE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('RESEARCH_ROOT',server.DEFAULT_ROOT)).resolve()
MATRIX=ROOT/'tgppo/runs/supervised-ppo-20261105'
VARIANTS=('control','expert')
SEEDS=(11,22)
TERMINAL=('training_complete','budget_exhausted','failed')
_CACHED=server.cached_jobs
CURRENT_INFO=None


def protocol(variant):
    c=server.plan(ROOT,HERE/'configs'/(variant+'.json'))
    return c,ROOT/c['run_directory']


def package_hashes():
    paths=list((HERE/'code').rglob('*.py'))+list((HERE/'configs').glob('*.json'))
    paths+=list(HERE.glob('*.sh'))+list((HERE/'tests').glob('*.py'))
    paths+=list((HERE/'evaluation').glob('*.py'))+list((HERE/'evaluation').glob('*.sh'))+[HERE/'docs/PROTOCOL.md']
    return {str(p.relative_to(HERE)):sha256(p) for p in sorted(paths)}


def verify():
    release=server.read(HERE/'docs/release.json')
    if not release['passed'] or release['files']!=package_hashes():raise ValueError('Package differs from tested release')
    checks=server.read(MATRIX/'preparation/checks.json')
    if not checks['passed']:raise ValueError('Local structural/PPO checks failed')
    audit=server.read(OUTPUT/'data_audit.json')
    if sha256(OUTPUT/'expert.npz')!=audit['dataset_sha256']:raise ValueError('Expert data changed')
    for path,digest in release['external_inputs'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Frozen external input changed: '+path)
    base,_=protocol('control')
    if base['actor_kind']!='mlp' or base['expert_pretraining']!='bc' or base['expert_supervision_weight']!=0.:
        raise ValueError('Control must be fixed MLP+BC with no auxiliary loss')
    if protocol('expert')[0]['expert_supervision_weight']!=.01:
        raise ValueError('Only the frozen0.01 coefficient is authorized')
    allowed={'expert_supervision_weight','variant','run_directory'}
    for variant in VARIANTS:
        c,_=protocol(variant)
        if any(c.get(k)!=base.get(k) for k in set(c)|set(base) if k not in allowed):
            raise ValueError('Uncontrolled protocol difference: '+variant)
        if c['expert_dataset_sha256']!=audit['dataset_sha256']:raise ValueError('Dataset/config mismatch')
    if len(VARIANTS)*len(SEEDS)*base['gpu_job_budget_seconds']+base['finalization_budget_seconds']+600>base['matrix_wall_budget_seconds']:
        raise ValueError('Declared job ceilings exceed matrix budget')


@contextlib.contextmanager
def lock(path,blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


class BudgetExhausted(RuntimeError):pass


@contextlib.contextmanager
def deadline(seconds):
    if seconds<=0:raise BudgetExhausted('No remaining wall/deadline budget')
    def expired(*_):raise BudgetExhausted('Declared cumulative wall limit reached')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def charged_seconds(attempts):
    # A killed process with no final accounting cannot silently reset its budget.
    return sum(a.get('elapsed_seconds',a['allocated_seconds']) for a in attempts)


def job_path(variant,seed):return MATRIX/'jobs'/('{}-s{}.json'.format(variant,seed))


def statuses(seeds=SEEDS):
    return {v+'-s'+str(s):(server.read(job_path(v,s)) if job_path(v,s).exists() else {'status':'not_started'})
            for v in VARIANTS for s in seeds}


def software(stack):return {k:v for k,v in stack.items() if k!='cpu_model'}


def check_software(info):
    with lock(MATRIX/'software.lock'):
        path=MATRIX/'software_reference.json'
        if not path.exists():write_json(path,info)
        if software(server.read(path)['stack'])!=software(info['stack']):
            raise ValueError('Software/SCIP/binding mismatch; CPU and GPU models are unrestricted')


def adopt_native(source,c,target,signature):
    bundle=server.read(source)
    if bundle['stack']!=signature:return False
    cutoffs=server.read(ROOT/'tgppo/datasets/cutoffs.json')
    values={}
    for name in c['train']+c['development']:
        for seed in c['solver_seeds']:
            ident=rt.job_identity(server.bound_job(ROOT,c,cutoffs,name,seed,11,'native'))
            record=bundle['baselines'][fingerprint(ident)];validate_baseline(record,ident)
            values[fingerprint(ident)]=record
    write_json(Path(target)/'normalization/baselines.json',dict(protocol_sha256=fingerprint(c),stack=signature,
        baselines=values,reused_from=str(source),source_sha256=sha256(source)))
    return True


def normalize(c,context,info):
    canonical,_=protocol('control');key=fingerprint(info['stack'])[:24]
    shared=MATRIX/'normalization'/key
    with lock(shared.with_suffix('.lock')):
        path=shared/'normalization/baselines.json'
        if not path.exists():
            sources=[ROOT/'tgppo/runs/server6-20261022/normalization/baselines.json',
                ROOT/'tgppo/runs/candidate-residual-20261104/shared/normalization/baselines.json']
            sources+=sorted((ROOT/'tgppo/runs/candidate-residual-20261104/cpu-flex/normalization').glob('*/normalization/baselines.json'))
            sources+=sorted((ROOT/'tgppo/runs/light-actor-20261105/normalization').glob('*/normalization/baselines.json'))
            reused=False
            for source in sources:
                if not source.exists():continue
                try:reused=adopt_native(source,canonical,shared,info['stack'])
                except (KeyError,ValueError):reused=False
                if reused:break
            if not reused:server.baseline(ROOT,canonical,shared)
        server.baseline_bundle(ROOT,canonical,shared)
        if not adopt_native(path,c,context,info['stack']):raise ValueError('Normalization stack changed')
        server.baseline_bundle(ROOT,c,context)


def training_context(c,run,seed,info):
    key=fingerprint(dict(stack=info['stack'],gpu=info['gpu']))[:24]
    context=MATRIX/'contexts'/('{}-s{}'.format(c['variant'],seed))/key
    context.mkdir(parents=True,exist_ok=True);(run/'training').mkdir(parents=True,exist_ok=True)
    link=context/'training'
    if not link.exists():link.symlink_to(run/'training',target_is_directory=True)
    if link.resolve()!=(run/'training').resolve():raise ValueError('Wrong training context link')
    write_json(context/'environment.json',info)
    return context


def cached_training(jobs,folder,identity,workers,watchdog):
    if CURRENT_INFO is not None and jobs and all(j['mode']=='collect' for j in jobs):
        return _CACHED(jobs,Path(folder)/('stack-'+fingerprint(CURRENT_INFO['stack'])[:24]),
            dict(identity,training_stack=CURRENT_INFO['stack']),workers,watchdog)
    return _CACHED(jobs,folder,identity,workers,watchdog)


def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def ensure_export(checkpoint,actor,payload=None):
    if not actor.exists():server.export_actor(checkpoint,actor);return
    saved=rt.read_checkpoint(actor,allow_actor_only=True)
    payload=payload or rt.read_checkpoint(checkpoint)
    if (saved.get('schema_version')!='actor-only-v1' or
        saved['provenance']['checkpoint_sha256']!=sha256(checkpoint) or
        saved['config_sha256']!=payload['config_sha256'] or
        state_digest(saved['actor'])!=state_digest(payload['actor'])):
        raise ValueError('Existing Actor export differs from its checkpoint')


def prepare_model(c,context,seed,info,resume):
    import shutil
    from pretrain import signature
    from engineering import bridge
    from target_checks import check_rng_pairing
    cfg=server.params(ROOT,c,'B0',seed,'cuda');out=server.group_dir(context,'B0',seed)
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'tgppo/runs/light-actor-20261105/mlp_bc/training'/('B0-s'+str(seed))
    record=server.read(source/'pretrain/result.json')
    if record['selected_step']!=100 or record['completed_steps']!=1000 or record['signature']!=signature(cfg):
        raise ValueError('Existing BC signature or selection differs')
    if sha256(source/'pretrain/actor.pt')!=record['actor_sha256']:
        raise ValueError('Existing BC changed')
    initial=server.read(source/'initialization.json')
    if state_digest(torch.load(source/'pretrain/actor.pt',map_location='cpu')['actor'])!=initial['actor_fingerprint']:
        raise ValueError('BC weights differ from the actual original PPO initializer')
    (out/'pretrain').mkdir(exist_ok=True)
    for name in ('actor.pt','result.json'):
        src=source/'pretrain'/name;dst=out/'pretrain'/name
        if not dst.exists():shutil.copyfile(src,dst)
        if sha256(src)!=sha256(dst):raise ValueError('BC reuse differs')
    write_json(out/'reused_initialization.json',dict(source=str(source/'pretrain/actor.pt'),
        source_sha256=record['actor_sha256'],selected_bc_step=100,new_bc_training_steps=0))
    actor_state=torch.load(out/'pretrain/actor.pt',map_location='cpu')['actor']
    if (out/'latest.pt').exists():
        if not resume:raise FileExistsError('Use --resume for existing PPO checkpoint')
        if rt.read_checkpoint(out/'latest.pt')['config']!=cfg:raise ValueError('Resume configuration changed')
    check_rng_pairing(c,seed,context/'target_rng.json')
    gate_path=context/'gpu-bridge/verification.json';digest=state_digest(actor_state)
    gate=server.read(gate_path) if gate_path.exists() else {}
    if not (gate.get('passed') and gate.get('actor_digest')==digest and gate.get('parent_protocol_sha256')==fingerprint(c)
            and gate.get('stack')==info['stack'] and gate.get('gpu')==info['gpu']):
        result=bridge(cfg,'cuda',context/'gpu-bridge'/digest[:16],actor_state=actor_state)
        if cfg['expert_supervision_weight']>0 and result['metrics']['expert_batches']!=1:
            raise ValueError('Target supervised update was not exercised')
        write_json(gate_path,dict(result,actor_digest=digest,parent_protocol_sha256=fingerprint(c),stack=info['stack'],gpu=info['gpu']))
    server.require_target_evidence(ROOT,c,context,info)


def run(variant,seed,resume=False):
    global CURRENT_INFO
    c,run_dir=protocol(variant);path=job_path(variant,seed)
    with lock(path.with_suffix('.lock'),blocking=False):
        previous=server.read(path) if path.exists() else dict(attempts=[])
        if previous.get('status')=='training_complete':
            print('Already complete; no training repeated.',flush=True);return
        if previous['attempts'] and not resume:raise FileExistsError('Previous attempt retained; inspect then use --resume')
        launch_epoch=float(os.environ.get('LIGHT_LAUNCH_EPOCH',time.time()))
        import_elapsed=max(0.,time.time()-launch_epoch)
        remaining=c['gpu_job_budget_seconds']-charged_seconds(previous['attempts'])-import_elapsed
        allocation=remaining-20  # cleanup reserve inside the new cumulative140min cap
        if allocation<=0:raise BudgetExhausted('Job/deadline budget exhausted')
        attempt=dict(id=uuid.uuid4().hex,allocated_seconds=allocation+import_elapsed,
            started_epoch=launch_epoch,host=socket.gethostname())
        record=dict(previous,status='running',variant=variant,seed=seed,protocol_sha256=fingerprint(c))
        record['attempts'].append(attempt);write_json(path,record)
        error=None
        try:
            with deadline(allocation):
                CURRENT_INFO=server.preflight(ROOT,c,'cuda');check_software(CURRENT_INFO)
                attempt['environment']=CURRENT_INFO;write_json(path,record)
                context=training_context(c,run_dir,seed,CURRENT_INFO)
                record['phase']='normalization';write_json(path,record);normalize(c,context,CURRENT_INFO)
                record['phase']='bc_and_gpu_bridge';write_json(path,record);prepare_model(c,context,seed,CURRENT_INFO,resume)
                record['phase']='ppo';write_json(path,record);server.cached_jobs=cached_training
                state=server.train(ROOT,c,context,'B0',seed,resume,'cuda')
                record['status']='training_complete';record['next_episode']=state['next_episode']
                out=server.group_dir(run_dir,'B0',seed)
                for episode in c['milestones']:
                    cp=out/('episode-{:04d}.pt'.format(episode))
                    if not cp.exists():raise ValueError('Missing required0/20/100 checkpoint: '+str(cp))
                    ensure_export(cp,out/('actor-{:04d}.pt'.format(episode)))
        except BudgetExhausted as exc:
            record['status']='budget_exhausted';record['error']=repr(exc);error=exc
        except BaseException as exc:
            record['status']='failed';record['error']=repr(exc);error=exc
        finally:
            attempt['elapsed_seconds']=time.time()-launch_epoch;attempt['finished_epoch']=time.time()
            record['charged_seconds']=charged_seconds(record['attempts']);record['phase']='stopped'
            write_json(path,record);server.cached_jobs=_CACHED
        print(json.dumps({k:record[k] for k in ('status','variant','seed','charged_seconds')},indent=2),flush=True)
        write_matrix_status()
        if error is not None:raise error


def write_matrix_status():
    with lock(MATRIX/'matrix-status.lock'):
        records=statuses(SEEDS)
        complete=all(r.get('status')=='training_complete' for r in records.values())
        write_json(MATRIX/'matrix_status.json',dict(training_complete=complete,jobs=records,
            cpu_evaluation='Submit separate CPU Custom after all4 complete; GPU allocation does not run evaluation'))
    if complete:print('All4 training jobs complete. Ready for separate CPU-only110-task evaluation.',flush=True)


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','status','check'])
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=11)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.command=='status':
        print(json.dumps(statuses(),ensure_ascii=False,indent=2));return
    verify()
    if args.command=='check':print('Frozen supervised-PPO package and legal data passed');return
    if not args.variant:parser.error('--variant control/expert required')
    run(args.variant,args.seed,args.resume)


if __name__=='__main__':main()
