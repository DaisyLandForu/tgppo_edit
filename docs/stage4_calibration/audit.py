"""Read-only calibrated Critic audit. All values/metrics are in raw return units."""
import argparse
import csv
import json
from pathlib import Path
import pickle
import subprocess
import numpy as np
import torch
from project.runtime import load_agent,seed_all
from project.policy.calibrated_critic import CalibratedCritic,make_calibration
from project.utils.protocol import sha256,write_json

ROOT=Path('/root/research');RUN=ROOT/'tgppo/runs/stage4-calibration';DOC=ROOT/'docs/stage4_calibration'


def csv_write(name,rows):
    with (DOC/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    global RUN,DOC
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=RUN)
    p.add_argument('--report-dir',type=Path,default=DOC);args=p.parse_args()
    RUN=args.run_root.resolve();DOC=args.report_dir.resolve();DOC.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    cfg=json.loads((RUN/'protocol.json').read_text());selection=json.loads((RUN/'selection.json').read_text())
    old=json.loads((ROOT/'docs/stage4/summary.json').read_text())
    per=json.loads((RUN/'per_instance.json').read_text());fits=json.loads((RUN/'fit_runs.json').read_text())
    parameters=json.loads((RUN/'parameters.json').read_text());summary=json.loads((RUN/'summary.json').read_text())
    data=torch.load(cfg['source_cache'],map_location='cpu')['arrays']
    split=json.loads((ROOT/'docs/stage1/protocol_draft.json').read_text())['data']
    train=np.isin(data['instance'],split['train']);development=np.isin(data['instance'],split['development'])
    checks=dict(source_cache_unchanged=sha256(cfg['source_cache'])==cfg['cache_sha256'],
        actor_checkpoint_unchanged=sha256(cfg['base_checkpoint'])==cfg['checkpoint_sha256'],
        all_24_final_fits=len(fits)==24,
        train_only_selection=selection['development_used'] is False,
        original_upstream_clean=subprocess.check_output(['git','-C',str(ROOT/'tgppo/repo'),'status','--porcelain']).decode().strip()=='')
    checks['complete_matched_matrix']={(r['group'],r['loss'],r['seed']) for r in fits}=={
        (g,l,s) for g in cfg['groups'] for l in cfg['losses'] for s in cfg['fit_seeds']}
    fitting=np.char.endswith(data['job_id'],'-s0')
    pilot_indices=np.load(RUN/'pilot/sampling-global-indices-s11.npy')
    checks['pilot_sampling_training_solver0_only']=bool(np.all(train[pilot_indices]) and np.all(fitting[pilot_indices]))
    checks['pilot_instance_provenance']=set(selection['allowed_instance_names'])<=set(split['train'])
    expected_selection={}
    candidates=json.loads((RUN/'pilot/candidates.json').read_text())
    for loss in cfg['losses']:
        expected_selection[loss]=min((r for r in candidates if r['loss']==loss),
            key=lambda r:(r['inner_validation_mse'],r['updates'],r['lr']))
    checks['selection_matches_declared_rule']=selection['chosen']==expected_selection
    for seed in cfg['fit_seeds']:
        schedule=np.load(RUN/('sampling-global-indices-s{}.npy'.format(seed)))
        checks['train_only_final_schedule_s{}'.format(seed)]=bool(np.all(train[schedule]))
    for loss in cfg['losses']:
        relevant=[r for r in parameters if r['loss']==loss and r['group']!='R0']
        checks['matched_extra_parameter_counts_'+loss]=len({r['trainable'] for r in relevant})==1
        checks['same_update_lr_'+loss]=all(r['updates']==selection['chosen'][loss]['updates'] and
            r['lr']==selection['chosen'][loss]['lr'] for r in fits if r['loss']==loss)
    stats=json.loads((RUN/'final_train_only_statistics.json').read_text())
    agent,_=load_agent(cfg['base_checkpoint'],training=False,restore_random_state=False)
    recomputed,prior=make_calibration(agent.critic_network,data['target'][train],data['instance'][train])
    checks['normalization_from_train_only']=abs(stats['mean']-recomputed['mean'])<1e-10 and abs(stats['std']-recomputed['std'])<1e-10
    checks['prior_from_train_only']=bool(np.allclose(stats['hl_prior'],prior,atol=1e-8))
    # Restore each adapter on CPU and compare to the saved GPU predictions.
    probe_indices=np.linspace(0,len(data['target'])-1,17,dtype=int)
    features=[torch.from_numpy(data[k][probe_indices]) for k in ('aggregated','tree','tree_state','remaining_time','frontier')]
    path=next((ROOT/'tgppo/runs/stage4/collection/jobs').glob('offline-07-lseu-s0.pkl'))
    with path.open('rb') as f:raw=pickle.load(f)
    raw_row=raw['episodes'][0]['transitions'][0];extra=raw['search_features']['rows'][0]
    raw_index=np.flatnonzero(data['job_id']==path.stem)[0]
    restoration=[]
    for entry in fits:
        name='{}-{}-s{}'.format(entry['group'],entry['loss'],entry['seed'])
        adapter=torch.load(RUN/'fits'/(name+'.pt'),map_location='cpu')
        seed_all(entry['seed'])
        model=CalibratedCritic(agent.critic_network,entry['group'],entry['loss'])
        model.freeze_encoder()
        missing,unexpected=model.load_state_dict(adapter['trainable_state_dict'],strict=False)
        required={n for n,p in model.named_parameters() if p.requires_grad}|{'target_mean','target_std','return_edges','return_centers'}
        if required&set(missing) or unexpected:raise AssertionError('Incomplete calibrated adapter')
        model.eval()
        with torch.no_grad():
            predicted=model.forward_cached(*features).numpy()
            full=model(torch.from_numpy(raw_row['cands_state'][None]),torch.from_numpy(raw_row['node_state'][None]),
                torch.from_numpy(raw_row['mip_state'][None]),torch.tensor([[extra['remaining_time']]]),
                torch.from_numpy(extra['frontier'][None])).item()
        saved=np.load(RUN/'fits'/(name+'-predictions.npy'))
        np.testing.assert_allclose(predicted,saved[probe_indices],atol=1e-3,rtol=1e-4)
        if abs(full-saved[raw_index])>1e-3+1e-4*abs(saved[raw_index]):raise AssertionError('Raw feature-to-value mismatch')
        checks['selection_hash_'+name]=adapter['selection_sha256']==sha256(RUN/'selection.json')
        restoration.append(dict(group=entry['group'],loss=entry['loss'],seed=entry['seed'],
            max_cached_restore_error=float(np.max(np.abs(predicted-saved[probe_indices]))),
            raw_to_saved_error=float(abs(full-saved[raw_index]))))
    checks['all_24_adapters_restore']=len(restoration)==24
    unchanged=('policy/actor.py','policy/critic.py','policy/modules.py','reward/reward.py','utils/settings.py')
    checks['original_network_reward_settings_unchanged']=all(
        sha256(ROOT/'tgppo/experiments/stage4-calibration/code/project'/name)==sha256(ROOT/'tgppo/repo/code/project'/name) for name in unchanged)
    new_costs=[json.loads(p.read_text()) for p in RUN.glob('*_cost.json')]
    total_cpu=cfg['prior_stage4_offline_cpu_seconds']+sum(c['cpu_seconds'] for c in new_costs)
    checks['shared_offline_cpu_cap']=total_cpu<=cfg['global_offline_cpu_cap_seconds']
    aggregated=[];paired=[]
    for loss in cfg['losses']:
        for group in cfg['groups']:
            for which in ('train','development'):
                chosen=[r for r in per if r['loss']==loss and r['group']==group and r['split']==which]
                seed_values=[np.mean([r['mse'] for r in chosen if r['seed']==seed]) for seed in cfg['fit_seeds']]
                aggregated.append(dict(group=group,loss=loss,split=which,
                    instances=len({r['instance'] for r in chosen}),
                    mse=float(np.mean(seed_values)),mae=float(np.mean([r['mae'] for r in chosen])),
                    nmse=float(np.mean([r['nmse'] for r in chosen])),
                    mse_seed_min=float(min(seed_values)),mse_seed_max=float(max(seed_values)),
                    constant_mse=summary['constant_train_mse'] if which=='train' else summary['constant_development_mse']))
        names=sorted({r['instance'] for r in per if r['split']=='development'})
        for control in ('R0','Rcap','Rprod'):
            differences=[];seed_diff=[]
            for name in names:
                averages={g:np.mean([r['mse'] for r in per if r['loss']==loss and r['group']==g and r['instance']==name]) for g in ('Rjoint',control)}
                differences.append(averages['Rjoint']-averages[control])
            for seed in cfg['fit_seeds']:
                averages={g:np.mean([r['mse'] for r in per if r['loss']==loss and r['group']==g and r['seed']==seed and r['split']=='development']) for g in ('Rjoint',control)}
                seed_diff.append(averages['Rjoint']-averages[control])
            rng=np.random.RandomState(20261021)
            boot=np.asarray(differences)[rng.randint(len(names),size=(10000,len(names)))].mean(1)
            reference=next(r['mse'] for r in aggregated if r['group']==control and r['loss']==loss and r['split']=='development')
            paired.append(dict(loss=loss,control=control,joint_minus_control_mse=float(np.mean(differences)),
                relative_mse_percent=float(np.mean(differences)/reference*100),
                lower_mse_instances=int(sum(d<0 for d in differences)),lower_mse_fit_seeds=int(sum(d<0 for d in seed_diff)),
                instance_bootstrap_low=float(np.quantile(boot,.025)),instance_bootstrap_high=float(np.quantile(boot,.975))))
    mse_train=[r for r in aggregated if r['loss']=='mse' and r['split']=='train']
    gate=all(r['mse']<=.8*summary['constant_train_mse'] for r in mse_train) and all(
        r['train_mse']<summary['constant_train_mse'] for r in fits if r['loss']=='mse')
    csv_write('prediction_summary.csv',aggregated);csv_write('per_instance_prediction.csv',per)
    csv_write('paired_comparison.csv',paired);csv_write('parameters.csv',parameters)
    csv_write('fit_runs.csv',fits);csv_write('restoration.csv',restoration)
    csv_write('pilot_candidates.csv',candidates)
    write_json(DOC/'summary.json',dict(prediction_summary=aggregated,paired_comparison=paired,
        engineering_passed=all(checks.values()),mse_calibration_quality_gate_passed=gate,
        constants=dict(train=summary['constant_train_mse'],development=summary['constant_development_mse']),
        total_shared_offline_cpu_seconds=total_cpu,new_costs=new_costs,
        selected_settings=selection['chosen'],gpu_fit_event_seconds=summary['gpu_fit_event_seconds'],
        actual_value_scale=dict(mean=stats['mean'],std=stats['std']),
        fit_metadata=summary,old_stage4_development=old['prediction_summary'],
        conclusion_scope='exploratory fixed weak-Actor trajectory prediction; no online PPO or solver benefit claim',
        stage5_started=False))
    write_json(DOC/'verification.json',dict(passed=all(checks.values()),checks=checks,
        passed_scope='calibration implementation, train-only selection, data matching and checkpoint recovery',
        mse_calibration_quality_gate_passed=gate,online_benefit_validated=False,stage5_started=False))
    if not all(checks.values()):raise AssertionError(checks)
    print(json.dumps(dict(passed=True,mse_quality_gate=gate,restored_models=24,total_shared_cpu_seconds=total_cpu)))


if __name__=='__main__':main()
