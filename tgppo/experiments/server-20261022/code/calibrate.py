"""Train-only calibration selection, then a frozen 24-fit diagnostic matrix."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback
import numpy as np
import torch
from project.runtime import load_agent,seed_all,cpu_seconds
from project.policy.calibrated_critic import CalibratedCritic,make_calibration
from project.utils.protocol import sha256,write_json
from stage4 import metrics

ROOT=Path('/root/research')
RUN=ROOT/'tgppo/runs/stage4-calibration'
PROTOCOL=ROOT/'docs/stage4_calibration/protocol.json'
KEYS=('aggregated','tree','tree_state','remaining_time','frontier')


def setup():
    config=json.loads(PROTOCOL.read_text())
    for field,digest in [('source_cache','cache_sha256'),('base_checkpoint','checkpoint_sha256')]:
        if sha256(config[field])!=config[digest]:raise ValueError('Frozen source changed: '+field)
    RUN.mkdir(parents=True,exist_ok=True)
    path=RUN/'protocol.json'
    if path.exists() and path.read_bytes()!=PROTOCOL.read_bytes():raise ValueError('Protocol changed after start')
    path.write_bytes(PROTOCOL.read_bytes())
    prior=config['prior_stage4_offline_cpu_seconds']+sum(json.loads(p.read_text())['cpu_seconds'] for p in RUN.glob('*_cost.json'))
    remaining=config['global_offline_cpu_cap_seconds']-prior
    if remaining<=10:raise RuntimeError('Shared offline CPU cap exhausted')
    def stop(signum,frame):raise RuntimeError('Shared offline CPU cap reached')
    signal.signal(signal.SIGXCPU,stop)
    resource.setrlimit(resource.RLIMIT_CPU,(int(remaining-5),resource.getrlimit(resource.RLIMIT_CPU)[1]))
    os.sched_setaffinity(0,[0,1,2,3]);torch.set_num_threads(1)
    return config,prior


def load_sources(config):
    data=torch.load(config['source_cache'],map_location='cpu')['arrays']
    split=json.loads((ROOT/'docs/stage1/protocol_draft.json').read_text())['data']
    agent,_=load_agent(config['base_checkpoint'],training=False,restore_random_state=False)
    return data,split,agent.critic_network


def tensor_data(data,device):
    return {k:torch.as_tensor(data[k],device=device) for k in KEYS+('target',)}


def batch_schedule(instances,eligible,seed,count):
    names=sorted(set(instances[eligible]))
    buckets=[np.flatnonzero((instances==name)&eligible) for name in names]
    rng=np.random.RandomState(seed);result=[]
    for _ in range(count):
        chosen=rng.randint(len(buckets),size=256)
        result.append(np.asarray([buckets[k][rng.randint(len(buckets[k]))] for k in chosen],dtype=np.int64))
    return np.stack(result)


def predict(model,features):
    model.eval();values=[]
    with torch.no_grad():
        for begin in range(0,len(features['target']),2048):
            values.append(model.forward_cached(*[features[k][begin:begin+2048] for k in KEYS]).cpu().numpy())
    result=np.concatenate(values)
    if not np.isfinite(result).all():raise ValueError('Nonfinite raw prediction')
    return result


def instance_mse(predicted,target,instances,mask):
    return float(np.mean([np.mean((predicted[ids]-target[ids])**2)
        for name in sorted(set(instances[mask])) for ids in [(instances==name)&mask]]))


def model_for(base,group,loss,statistics,prior,seed,device):
    seed_all(seed)
    model=CalibratedCritic(base,group,loss,statistics['mean'],statistics['std'],prior)
    model.freeze_encoder();model.to(device)
    return model


def train(model,features,schedule,lr,checkpoints,callback):
    model.train()
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
    for step,ids in enumerate(schedule,1):
        idx=torch.as_tensor(ids,device=features['target'].device)
        optimizer.zero_grad(set_to_none=True)
        output=model.logits_cached(*[features[k][idx] for k in KEYS])
        loss=model.training_loss(output,features['target'][idx])
        if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5)
        optimizer.step()
        if step%250==0 or step in checkpoints:
            callback(step,float(loss.detach().cpu()),optimizer,step in checkpoints)
            model.train()
    return optimizer


def save_adapter(path,model,optimizer,metadata):
    names={k for k,p in model.named_parameters() if p.requires_grad}
    names.update(('target_mean','target_std','return_edges','return_centers'))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in names}
    torch.save(dict(metadata,trainable_state_dict=state,optimizer=optimizer.state_dict()),path)


def select(config,prior_cpu):
    out=RUN/'pilot'
    if out.exists():raise FileExistsError(str(out))
    out.mkdir()
    all_data,split,base=load_sources(config)
    global_ids=np.flatnonzero(np.isin(all_data['instance'],split['train']))
    # No development labels or features are passed to a pilot fit or metric.
    data={k:v[global_ids] for k,v in all_data.items()}
    fitting=np.char.endswith(data['job_id'],'-s0')
    validation=np.char.endswith(data['job_id'],'-s1')
    if not np.all(fitting|validation) or np.any(fitting&validation):raise ValueError('Episode split error')
    if set(data['instance'][fitting])!=set(data['instance'][validation]):raise ValueError('Inner coverage mismatch')
    stats,probabilities=make_calibration(base,data['target'][fitting],data['instance'][fitting])
    write_json(out/'train_only_statistics.json',dict(stats,hl_prior=probabilities.tolist()))
    schedule=batch_schedule(data['instance'],fitting,11,5000)
    np.save(out/'sampling-global-indices-s11.npy',global_ids[schedule])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features=tensor_data(data,device)
    constant=np.full(len(data['target']),stats['mean'])
    constant_fit=instance_mse(constant,data['target'],data['instance'],fitting)
    constant_val=instance_mse(constant,data['target'],data['instance'],validation)
    candidates=[];curves=[]
    for loss in config['losses']:
        for lr in config['candidate_learning_rates']:
            model=model_for(base,'R0',loss,stats,probabilities,11,device)
            def checkpoint(step,objective,optimizer,is_candidate):
                curves.append(dict(loss=loss,lr=lr,update=step,objective=objective))
                if not is_candidate:return
                values=predict(model,features)
                row=dict(loss=loss,lr=lr,updates=step,
                    train_mse=instance_mse(values,data['target'],data['instance'],fitting),
                    inner_validation_mse=instance_mse(values,data['target'],data['instance'],validation),
                    constant_train_mse=constant_fit,constant_inner_validation_mse=constant_val)
                candidates.append(row)
                save_adapter(out/('{}-lr{}-u{}.pt'.format(loss,lr,step)),model,optimizer,
                    dict(row,group='R0',seed=11,calibration_statistics=stats,
                        statistics_scope='training instances solver0 only',hl_prior=probabilities.tolist()))
                write_json(out/'candidates.json',candidates)
                print(json.dumps(dict(event='pilot_candidate',**row)),flush=True)
            train(model,features,schedule,lr,(2000,5000),checkpoint)
    chosen={loss:min((r for r in candidates if r['loss']==loss),
                    key=lambda r:(r['inner_validation_mse'],r['updates'],r['lr'])) for loss in config['losses']}
    selection=dict(chosen=chosen,criterion=config['selection'],development_used=False,
        allowed_instance_names=sorted(set(data['instance'])),cache_sha256=config['cache_sha256'],
        protocol_sha256=sha256(PROTOCOL),pilot_data_global_indices=global_ids.tolist(),
        selection_frozen_before_final_fits=True)
    write_json(RUN/'selection.json',selection)
    write_json(out/'curves.json',curves)


def fit(config,prior_cpu):
    out=RUN/'fits'
    if out.exists():raise FileExistsError(str(out))
    out.mkdir()
    selection=json.loads((RUN/'selection.json').read_text())
    selection_sha=sha256(RUN/'selection.json')
    if selection['protocol_sha256']!=sha256(PROTOCOL) or selection['development_used']:
        raise ValueError('Training-only selection mismatch')
    data,split,base=load_sources(config)
    train_mask=np.isin(data['instance'],split['train'])
    dev_mask=np.isin(data['instance'],split['development'])
    stats,probabilities=make_calibration(base,data['target'][train_mask],data['instance'][train_mask])
    write_json(RUN/'final_train_only_statistics.json',dict(stats,hl_prior=probabilities.tolist()))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features=tensor_data(data,device)
    max_updates=max(r['updates'] for r in selection['chosen'].values())
    schedules={seed:batch_schedule(data['instance'],train_mask,seed,max_updates) for seed in config['fit_seeds']}
    for seed,schedule in schedules.items():np.save(RUN/('sampling-global-indices-s{}.npy'.format(seed)),schedule)
    per_instance=[];fit_rows=[];parameters=[];curves=[];gpu_ms=0.
    for loss in config['losses']:
        setting=selection['chosen'][loss]
        for seed in config['fit_seeds']:
            for group in config['groups']:
                model=model_for(base,group,loss,stats,probabilities,seed,device)
                initial={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
                parameters.append(dict(group=group,loss=loss,seed=seed,
                    total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
                initial_values=predict(model,features)
                initial_train=instance_mse(initial_values,data['target'],data['instance'],train_mask)
                start_cpu,start_wall=cpu_seconds(),time.perf_counter()
                if device.type=='cuda':
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                def progress(step,objective,optimizer,is_candidate):
                    curves.append(dict(group=group,loss=loss,seed=seed,update=step,objective=objective))
                    if step%1000==0:
                        write_json(RUN/'progress.json',dict(completed=len(fit_rows),total=24,group=group,
                            loss=loss,seed=seed,updates=step,total_updates=setting['updates'],
                            combined_offline_cpu_seconds=prior_cpu+cpu_seconds()))
                optimizer=train(model,features,schedules[seed][:setting['updates']],setting['lr'],(),progress)
                if device.type=='cuda':end.record();torch.cuda.synchronize();gpu_ms+=begin.elapsed_time(end)
                prediction=predict(model,features)
                name='{}-{}-s{}'.format(group,loss,seed)
                np.save(out/(name+'-predictions.npy'),prediction)
                metadata=dict(group=group,loss=loss,seed=seed,updates=setting['updates'],lr=setting['lr'],
                    checkpoint_sha256=config['checkpoint_sha256'],cache_sha256=config['cache_sha256'],
                    selection_sha256=selection_sha,protocol_sha256=sha256(PROTOCOL),
                    calibration_statistics=stats,hl_prior=probabilities.tolist(),
                    value_units='raw empirical discounted return after decoding',
                    training_instance_names=sorted(set(data['instance'][train_mask])))
                save_adapter(out/(name+'.pt'),model,optimizer,metadata)
                for name in sorted(set(data['instance'])):
                    ids=data['instance']==name
                    per_instance.append(dict(group=group,loss=loss,seed=seed,instance=name,
                        split='train' if name in split['train'] else 'development',
                        **metrics(prediction[ids],data['target'][ids],stats['variance'])))
                changed=sum((p.detach().cpu()-initial[k]).double().square().sum().item()
                    for k,p in model.named_parameters() if p.requires_grad)**.5
                row=dict(group=group,loss=loss,seed=seed,updates=setting['updates'],lr=setting['lr'],
                    initial_train_mse=initial_train,
                    train_mse=instance_mse(prediction,data['target'],data['instance'],train_mask),
                    development_mse=instance_mse(prediction,data['target'],data['instance'],dev_mask),
                    parameter_l2_change=changed,cpu_seconds=cpu_seconds()-start_cpu,
                    wall_seconds=time.perf_counter()-start_wall)
                fit_rows.append(row)
                write_json(RUN/'fit_runs.json',fit_rows)
                print(json.dumps(row),flush=True)
                del model,optimizer
    for name,values in [('per_instance.json',per_instance),('parameters.json',parameters),('curves.json',curves)]:
        write_json(RUN/name,values)
    constant=np.full(len(data['target']),stats['mean'])
    summary=dict(completed_fits=len(fit_rows),selection_sha256=selection_sha,
        selection_unchanged=sha256(RUN/'selection.json')==selection_sha,
        constant_train_mse=instance_mse(constant,data['target'],data['instance'],train_mask),
        constant_development_mse=instance_mse(constant,data['target'],data['instance'],dev_mask),
        gpu_fit_event_seconds=gpu_ms/1000.,device=str(device),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
        checkpoint_unchanged=sha256(config['base_checkpoint'])==config['checkpoint_sha256'],
        cache_unchanged=sha256(config['source_cache'])==config['cache_sha256'],online_started=False)
    write_json(RUN/'summary.json',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('select','fit'))
    parser.add_argument('--run-root',type=Path)
    args=parser.parse_args()
    global RUN
    if args.run_root:RUN=args.run_root.resolve()
    config,prior=setup();start_wall=time.perf_counter()
    try:
        globals()[args.command](config,prior)
    except Exception:
        write_json(RUN/('failed-{}-{}_cost.json'.format(args.command,os.getpid())),
            dict(cpu_seconds=cpu_seconds(),error=traceback.format_exc()))
        raise
    write_json(RUN/(args.command+'_cost.json'),dict(cpu_seconds=cpu_seconds(),
        wall_seconds=time.perf_counter()-start_wall,prior_shared_cpu_seconds=prior,
        combined_offline_cpu_seconds=prior+cpu_seconds(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
