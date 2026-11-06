"""Summarize the already completed, strictly limited PPO steps."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT=Path(__file__).resolve().parents[1]


def read(name):
    return list(csv.DictReader((OUT/name).open()))


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def group(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    return groups


def q(values):
    v=[float(x) for x in values if x not in [None,'']]
    return dict(zip(['min','p50','p95','max'],map(float,np.quantile(v,[0,.5,.95,1])))) if v else {}


def main():
    responses=read('policy_responses.csv')
    seeds=json.loads((OUT/'seed_summaries.json').read_text())
    assert len(seeds)==2 and len(responses)==2*(256+164)
    all_summary=[];per_instance=[]
    for (model,dataset),rs in group(responses,['model','dataset']).items():
        im=[]
        for (instance,),vs in group(rs,['instance']).items():
            value=dict(model=model,dataset=dataset,instance=instance,states=len(vs),
                       median_pre_post_tv=q(r['pre_post_tv'] for r in vs)['p50'],
                       median_post_tv_uniform=q(r['post_tv_uniform'] for r in vs)['p50'],
                       above_float32_screen=sum(r['above_conservative_float32_screen']=='True' for r in vs))
            per_instance.append(value);im.append(value['median_pre_post_tv'])
        all_summary.append(dict(model=model,dataset=dataset,states=len(rs),instances=len(im),
                                L1_states=sum(int(r['L'])==1 for r in rs),
                                tv_pre_post=q(r['pre_post_tv'] for r in rs),
                                tv_pre_repeat=q(r['pre_repeat_tv'] for r in rs),
                                pre_tv_uniform=q(r['pre_tv_uniform'] for r in rs),
                                post_tv_uniform=q(r['post_tv_uniform'] for r in rs),
                                pre_post_kl=q(r['pre_post_kl'] for r in rs),
                                instance_median_tv_pre_post=q(im),
                                above_repeat_noise=sum(r['above_repeat_noise']=='True' for r in rs),
                                above_float32_screen=sum(r['above_conservative_float32_screen']=='True' for r in rs),
                                post_tv_uniform_ge_001=sum(float(r['post_tv_uniform'])>=.01 for r in rs),
                                abs_common_logit_shift=q(abs(float(r['mean_logit_shift'])) for r in rs),
                                centered_logit_delta_rms=q(r['centered_logit_delta_rms'] for r in rs),
                                post_old_ratio=q(r['post_old_ratio'] for r in rs),
                                post_pre_action_ratio=q(r['post_pre_action_ratio'] for r in rs)))
    write('response_by_instance.csv',per_instance)
    write('response_summary.csv',[dict(model=r['model'],dataset=r['dataset'],states=r['states'],instances=r['instances'],
                                       tv_change_p50=r['tv_pre_post']['p50'],tv_change_max=r['tv_pre_post']['max'],
                                       post_tv_uniform_p50=r['post_tv_uniform']['p50'],post_tv_uniform_max=r['post_tv_uniform']['max'],
                                       above_float32_screen=r['above_float32_screen'],
                                       common_logit_shift_abs_p50=r['abs_common_logit_shift']['p50'],
                                       centered_logit_delta_rms_p50=r['centered_logit_delta_rms']['p50']) for r in all_summary])
    params=read('parameter_adam_changes.csv')
    parameter_summary=[]
    for (model,kind),rs in group(params,['model','network']).items():
        parameter_summary.append(dict(model=model,network=kind,tensors=len(rs),
                                      parameter_delta_l2=sum(float(r['parameter_delta_l2'])**2 for r in rs)**.5,
                                      parameter_l2_before=sum(float(r['parameter_l2_before'])**2 for r in rs)**.5,
                                      max_abs_parameter_delta=max(float(r['parameter_delta_max_abs']) for r in rs),
                                      changed_elements=sum(int(r['changed_elements']) for r in rs),
                                      total_elements=sum(int(r['numel']) for r in rs),
                                      step_before=q(r['adam_step_before'] for r in rs),
                                      step_after=q(r['adam_step_after'] for r in rs),
                                      each_adam_step_increased_once=all(float(r['adam_step_after'])-float(r['adam_step_before'])==1 for r in rs)))
    result=dict(seed_summaries=seeds,response=all_summary,parameter_updates=parameter_summary,
                clipping=read('gradient_clipping.csv'),
                optimizer_steps=json.loads((OUT/'optimizer_steps.json').read_text()),
                no_solver_or_generalization_claim=True,
                note='One actual logical minibatch per seed. Raw TV above zero-repeat noise is distinguished from a conservative float32 screen; neither is policy-quality evidence.')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='seed_summaries'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
