"""CPU-model amendment, outside the training code used by the live B0 job.

The original protocol/config hashes, training code, checkpoints and budgets stay
valid. Normalization and GPU checks are scoped to each actual training machine.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import experiment as base
from project.utils.protocol import fingerprint, sha256, write_json

HERE = Path(__file__).resolve().parent
_VERIFY = base.verify_package
_TRAIN = base.server.train
_CACHED = base.server.cached_jobs
CURRENT_JOB = None
CONTEXT = None
CURRENT_INFO = None


def software_stack(stack):
    return {k: v for k, v in stack.items() if k != 'cpu_model'}


def check_platform(info, reference):
    if software_stack(info['stack']) != software_stack(reference['stack']):
        actual,expected=software_stack(info['stack']),software_stack(reference['stack'])
        changed={k:dict(expected=expected.get(k),actual=actual.get(k))
            for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k)}
        raise ValueError('Software mismatch (CPU model unrestricted): '+str(changed))
    if 'V100' not in info['gpu']:
        raise ValueError('This amendment keeps the V100 GPU requirement')


def own_hashes():
    paths = [HERE/'launcher.py', HERE/'custom_job.sh']+list((HERE/'tests').glob('*.py'))
    return {str(p.relative_to(HERE)): sha256(p) for p in sorted(paths)}


def verify():
    _VERIFY()
    release = base.server.read(HERE/'verification.json')
    if not release['passed'] or release['files'] != own_hashes():
        raise ValueError('CPU-flex amendment differs from its verified release')
    if release['parent_manifest_sha256'] != sha256(base.HERE/'docs/bundle_manifest.json'):
        raise ValueError('Frozen parent package changed')


def context_for(variant, seed, info):
    key = fingerprint(dict(stack=info['stack'], gpu=info['gpu']))[:24]
    return base.MATRIX/'cpu-flex/contexts'/('{}-s{}'.format(variant, seed))/key


def record_environment(variant, seed, info, context):
    folder = base.MATRIX/'cpu-flex'
    with base.file_lock(folder/'environment.lock'):
        path = folder/'training_environments.json'
        data = base.server.read(path) if path.exists() else dict(
            policy='CPU models may differ; software must match; normalization is CPU-specific', jobs={})
        # Include the already-running original job without modifying its state.
        for v in base.VARIANTS:
            for s in base.SEEDS:
                key = '{}-s{}'.format(v,s)
                old = base.server.group_dir(base.protocol(v)[1],'B0',s)/'preflight.json'
                if key not in data['jobs'] and old.exists():
                    measured=base.server.read(old)
                    data['jobs'][key]=[dict(stack=measured['stack'],gpu=measured['gpu'],
                        host=measured['host'],source=str(old),kind='original job')]
        key='{}-s{}'.format(variant,seed)
        data['jobs'].setdefault(key,[]).append(dict(stack=info['stack'],gpu=info['gpu'],
            host=info['host'],context=str(context),recorded_epoch=time.time(),kind='CPU-flex job'))
        write_json(path,data)
        lines=['# 训练硬件记录与CPU型号放宽', '',
            '用户授权不同CPU型号的训练节点。每个CPU/软件指纹使用匹配的原生归一化；'
            '最终全部模型在同一评估机器上比较。异构训练的轨迹、吞吐和时间截断会受硬件影响，'
            '本轮不宣称严格控制训练CPU的消融。', '',
            '| 作业 | CPU | GPU |', '|---|---|---|']
        for name, history in sorted(data['jobs'].items()):
            latest=history[-1]
            lines.append('| {} | {} | {} |'.format(name,latest['stack']['cpu_model'],latest['gpu']))
        result=base.MATRIX/'results/TRAINING_ENVIRONMENTS.md'
        result.parent.mkdir(parents=True,exist_ok=True)
        result.write_text('\n'.join(lines)+'\n')


def prepare(variant,c,run,info):
    global CONTEXT,CURRENT_INFO
    if CURRENT_JOB is None or CURRENT_JOB[0] != variant:
        raise ValueError('CPU-flex job identity missing')
    reference_path=base.MATRIX/'matrix_environment.json'
    if reference_path.exists():
        check_platform(info,base.server.read(reference_path))
    else:
        with base.file_lock(base.MATRIX/'cpu-flex/platform.lock'):
            marker=base.MATRIX/'cpu-flex/software_reference.json'
            if not marker.exists():write_json(marker,dict(stack=info['stack'],gpu=info['gpu']))
            check_platform(info,base.server.read(marker))
    CONTEXT=context_for(variant,CURRENT_JOB[1],info)
    CURRENT_INFO=info
    CONTEXT.mkdir(parents=True,exist_ok=True)
    training=run/'training'
    training.mkdir(parents=True,exist_ok=True)
    link=CONTEXT/'training'
    if not link.is_symlink():link.symlink_to(training,target_is_directory=True)
    if link.resolve()!=training.resolve():raise ValueError('Wrong training-context link')
    write_json(CONTEXT/'environment.json',info)
    record_environment(variant,CURRENT_JOB[1],info,CONTEXT)
    # Different CPU models prepare in parallel; equal fingerprints reuse one sweep.
    key=fingerprint(info['stack'])[:24]
    shared=base.MATRIX/'cpu-flex/normalization'/key
    canonical,_=base.protocol('b0')
    with base.file_lock(shared.with_suffix('.lock')):
        destination=shared/'normalization/baselines.json'
        if not destination.exists():
            sources=[base.MATRIX/'shared/normalization/baselines.json',
                base.ROOT/'tgppo/runs/server6-20261022/normalization/baselines.json',
                base.ROOT/'tgppo/runs/transformer-ablation-20261027/transformer/normalization/baselines.json']
            reused=False
            for source in sources:
                if not source.exists():continue
                try:reused=base.adopt_native(source,canonical,shared,info['stack'])
                except (KeyError,ValueError):reused=False
                if reused:break
            if not reused:base.server.baseline(base.ROOT,canonical,shared)
        base.server.baseline_bundle(base.ROOT,canonical,shared)
        attached=CONTEXT/'normalization/baselines.json'
        if not attached.exists():base.adopt_native(destination,c,CONTEXT,info['stack'])
        base.server.baseline_bundle(base.ROOT,c,CONTEXT)
    # GPU bridge is shared only by identical CPU/software/GPU and architecture.
    bridge_key=fingerprint(dict(stack=info['stack'],gpu=info['gpu']))[:24]
    bridge_root=base.MATRIX/'cpu-flex/bridges'/bridge_key/variant
    with base.file_lock(bridge_root/'bridge.lock'):
        gate=bridge_root/'verification.json'
        if not gate.exists():
            original=run/'gpu-bridge/verification.json'
            candidate=base.server.read(original) if original.exists() else {}
            if (candidate.get('passed') and candidate.get('parent_protocol_sha256')==fingerprint(c)
                    and candidate.get('stack')==info['stack'] and candidate.get('gpu')==info['gpu']):
                write_json(gate,candidate)
            else:
                from engineering import bridge
                result=bridge(base.server.params(base.ROOT,c,'B0',11,'cuda'),'cuda',bridge_root/'synthetic')
                write_json(gate,dict(result,parent_protocol_sha256=fingerprint(c),
                    stack=info['stack'],gpu=info['gpu'],variant=variant))
        target=CONTEXT/'gpu-bridge/verification.json'
        write_json(target,base.server.read(gate))
        base.server.require_target_evidence(base.ROOT,c,CONTEXT,info)


def train(root,c,run,group,seed,resume,device,**kwargs):
    if CONTEXT is None or CURRENT_JOB!=(c['variant'],seed):
        raise ValueError('Training context does not match current job')
    return _TRAIN(root,c,CONTEXT,group,seed,resume,device,**kwargs)


def cached(jobs,folder,identity,workers,watchdog):
    if CURRENT_INFO is None or not jobs or any(j['mode']!='collect' for j in jobs):
        return _CACHED(jobs,folder,identity,workers,watchdog)
    # A resumed job on a new CPU must not reuse an incomplete batch normalized
    # for its former CPU. Full trajectories already committed in the checkpoint
    # remain intact, including their recorded rewards/values.
    selected=Path(folder)/('environment-'+fingerprint(CURRENT_INFO['stack'])[:24])
    pairs=_CACHED(jobs,selected,dict(identity,training_stack=CURRENT_INFO['stack']),workers,watchdog)
    write_json(Path(folder)/'selected_results.json',[r for r,_ in pairs])
    return pairs


def automatic_resume(variant,seed):
    p=base.status_path(variant,seed)
    if not p.exists():return False
    old=base.server.read(p)
    out=base.server.group_dir(base.protocol(variant)[1],'B0',seed)
    okay=(old.get('status')=='failed' and 'Matrix CPU/software/GPU differs' in old.get('error','')
          and not (out/'latest.pt').exists())
    if okay:
        history=base.MATRIX/'cpu-flex/prior-failures'/('{}-s{}'.format(variant,seed))
        archive=history/(old['attempts'][-1]['id']+'.json')
        if not archive.exists():write_json(archive,old)
    return okay


def main():
    global CURRENT_JOB
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','status','finalize','report','check'])
    parser.add_argument('--variant',choices=base.VARIANTS)
    parser.add_argument('--seed',type=int,choices=base.SEEDS)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    verify()
    if args.command=='check':print('CPU-flex and unchanged parent verified; no training or solve.');return
    if args.command=='status':print(json.dumps(base.statuses(),indent=2));return
    base.prepare=prepare
    base.server.train=train
    base.server.cached_jobs=cached
    if args.command=='run':
        if args.variant is None or args.seed is None:parser.error('run requires --variant and --seed')
        CURRENT_JOB=(args.variant,args.seed)
        base.run(args.variant,args.seed,args.resume or automatic_resume(args.variant,args.seed))
    elif args.command=='finalize':base.maybe_finalize()
    else:
        from reporting import report
        report(base.ROOT,base.MATRIX,base.choose_endpoint(base.statuses()) or 0,base.statuses())


if __name__=='__main__':main()
