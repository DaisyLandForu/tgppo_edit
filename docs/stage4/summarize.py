"""Audit stage-4 paired offline comparisons. Never alter the fitting protocol."""
import csv
import argparse
import math
import json
from pathlib import Path
import pickle
import subprocess
import numpy as np
import torch
from project.runtime import load_agent, seed_all
from project.policy.search_critic import SearchCritic
from project.utils.protocol import sha256, write_json

ROOT = Path('/root/research')
RUN = ROOT/'tgppo/runs/stage4'
DOC = ROOT/'docs/stage4'
BASE = ROOT/'tgppo/runs/stage3/pilot-repaired/latest.pt'


def write_csv(name, rows):
    with (DOC/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    global RUN, DOC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=RUN)
    parser.add_argument('--report-dir',type=Path,default=DOC)
    args=parser.parse_args()
    RUN=args.run_root.resolve();DOC=args.report_dir.resolve();DOC.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    config = json.loads((RUN/'execution_config.json').read_text())
    cached = torch.load(RUN/'offline/cache.pt', map_location='cpu')
    data = cached['arrays']
    results = [json.loads(p.read_text()) for p in sorted((RUN/'collection/jobs').glob('offline-*.json'))]
    fit_rows = json.loads((RUN/'offline/fit_runs.json').read_text())
    per_instance = json.loads((RUN/'offline/per_instance.json').read_text())
    parameters = json.loads((RUN/'offline/parameters.json').read_text())
    cache_cost = json.loads((RUN/'offline/cache_cost.json').read_text())
    fit_cost = json.loads((RUN/'offline/fit_cost.json').read_text())
    constants = json.loads((RUN/'offline/constant_predictor.json').read_text())
    offline_total_cpu=sum(json.loads(p.read_text()).get('cpu_seconds',0.) for p in (RUN/'offline').glob('*_cost.json'))
    checks = dict(all_50_collections_valid=len(results)==50 and all(r.get('valid') for r in results),
        all_24_fits_complete=len(fit_rows)==24 and all(r['updates']==500 for r in fit_rows),
        all_heads_updated=all(r['parameter_l2_change']>0 for r in fit_rows),
        fixed_checkpoint_unchanged=sha256(BASE)==config['checkpoint_sha256'],
        no_instance_split_overlap=not set(config['train']) & set(config['development']),
        offline_cpu_within_cap=offline_total_cpu<=14400,
        finite_returns_in_support=bool(np.isfinite(data['target']).all() and data['target'].min()>=-36 and data['target'].max()<=52))
    for loss in config['losses']:
        counts = [r['trainable'] for r in parameters if r['loss']==loss and r['group']!='R0']
        checks['matched_extra_parameters_'+loss] = len(set(counts))==1
    train_only=True
    for seed in config['fit_seeds']:
        schedule=np.load(RUN/'offline'/('sampling-indices-s{}.npy'.format(seed)))
        train_only=train_only and schedule.shape==(500,256) and set(data['instance'][schedule.ravel()])<=set(config['train'])
    checks['all_saved_fitting_indices_are_train_only']=bool(train_only)
    baseline_hashes = json.loads((ROOT/'docs/stage1/upstream_file_hashes.json').read_text())
    checks['pristine_upstream_files'] = all(sha256(ROOT/'tgppo/repo'/name)==digest for name,digest in baseline_hashes.items())
    unchanged = ('policy/actor.py','policy/critic.py','policy/modules.py','reward/reward.py','utils/settings.py')
    checks['upstream_network_reward_settings_unchanged'] = all(
        sha256(ROOT/'tgppo/experiments/stage4/code/project'/name)==sha256(ROOT/'tgppo/repo/code/project'/name)
        for name in unchanged)
    checks['original_git_clean'] = subprocess.check_output(['git','-C',str(ROOT/'tgppo/repo'),'status','--porcelain']).decode().strip()==''
    trajectories, feature_rows, counts = [], [], []
    raw_example = None
    for path in sorted((RUN/'collection/jobs').glob('offline-*.pkl')):
        result = json.loads(path.with_suffix('.json').read_text())
        with path.open('rb') as f: payload = pickle.load(f)
        rows = [r for ep in payload['episodes'] for r in ep['transitions']]
        extras = payload['search_features']['rows']
        if len(rows) != len(extras): raise AssertionError('Alignment')
        for row,extra in zip(rows,extras):
            if row['action'] != extra['action']: raise AssertionError('Action alignment')
            d=extra['diagnostics']
            feature_rows.append(dict(instance=result['instance_name'], solver_seed=result['solver_seed'],
                remaining_time=float(extra['remaining_time']), **d))
        if result['instance_name']=='lseu' and rows and raw_example is None:
            raw_example = (rows[0], extras[0], path.stem)
        trajectories.append(dict(job_id=path.stem, instance=result['instance_name'],
            solver_seed=result['solver_seed'], status=result['status'], transitions=len(rows),
            terminal_transitions=sum(r['done'] for r in rows),
            worker_cpu_seconds=result['worker_cpu_seconds'], worker_wall_seconds=result['worker_wall_seconds'],
            search_extraction_wall_seconds=result['search_extraction_wall_seconds'],
            callback_wall_seconds=result['callback_seconds']['callback'],
            max_open_nodes=max((x['diagnostics']['n_open'] for x in extras),default=0),
            peak_rss_kib=result['peak_rss_kib']))
    checks['trajectory_counts_and_terminal_flags'] = sum(r['transitions'] for r in trajectories)==len(data['target']) and all(
        r['terminal_transitions']==int(r['transitions']>0) for r in trajectories)
    # Read final trained adapters, verify CPU checkpoint restoration and raw-state→value chain.
    agent,_=load_agent(BASE,training=False,restore_random_state=False)
    row,extra,job_id=raw_example
    matching=np.flatnonzero(data['job_id']==job_id)[0]
    restoration=[]
    for entry in fit_rows:
        group,loss,seed=entry['group'],entry['loss'],entry['seed']
        path=RUN/'offline/fits'/('{}-{}-s{}.pt'.format(group,loss,seed))
        adapter=torch.load(path,map_location='cpu')
        seed_all(seed)
        model=SearchCritic(agent.critic_network,group,loss)
        model.freeze_encoder()
        missing,unexpected=model.load_state_dict(adapter['trainable_state_dict'],strict=False)
        trainable={name for name,p in model.named_parameters() if p.requires_grad}
        if unexpected or trainable & set(missing): raise AssertionError('Adapter missing trainable parameters')
        model.eval()
        with torch.no_grad():
            full=model(torch.from_numpy(row['cands_state'][None]),torch.from_numpy(row['node_state'][None]),
                torch.from_numpy(row['mip_state'][None]),torch.tensor([[extra['remaining_time']]]),
                torch.from_numpy(extra['frontier'][None])).item()
            args=[torch.from_numpy(data[k][matching:matching+1]) for k in
                ('aggregated','tree','tree_state','remaining_time','frontier')]
            restored=model.forward_cached(*args).item()
        saved=np.load(path.with_name(path.stem+'-predictions.npy'))[matching].item()
        error=max(abs(full-restored),abs(restored-saved))
        restoration.append(dict(group=group,loss=loss,seed=seed,max_absolute_restore_error=error))
        if error>1e-3: raise AssertionError('Trained checkpoint restoration mismatch')
        if adapter['training_instance_names']!=fit_cost['train_names']: raise AssertionError('Train split provenance mismatch')
    checks['all_24_adapters_restore_and_raw_chain_matches']=len(restoration)==24
    write_json(DOC/'restoration.json',restoration)
    # Per-instance coverage and information degeneracy, rather than transition-only averages.
    for name in config['train']+config['development']:
        f=[x for x in feature_rows if x['instance']==name]
        t=[x for x in trajectories if x['instance']==name]
        n=len(f)
        counts.append(dict(instance=name, split='train' if name in config['train'] else 'development',
            episodes=len(t), episodes_with_decisions=sum(x['transitions']>0 for x in t),transitions=n,
            nonempty_frontier_fraction=sum(x['n_open']>0 for x in f)/n if n else None,
            finite_frontier_fraction=sum(x['n_finite']>0 for x in f)/n if n else None,
            both_axes_nondegenerate_fraction=sum(x['both_axes_nondegenerate'] for x in f)/n if n else None,
            association_l1_mean=float(np.mean([x['association_l1'] for x in f])) if n else None,
            association_nonzero_fraction=sum(x['association_l1']>1e-6 for x in f)/n if n else None,
            max_open_nodes=max((x['n_open'] for x in f),default=0),
            extraction_ms_per_decision=1000*sum(x['search_extraction_wall_seconds'] for x in t)/n if n else None))
    summary=[]
    for loss in config['losses']:
        for group in config['groups']:
            for split in ('train','development'):
                selected=[r for r in per_instance if r['loss']==loss and r['group']==group and r['split']==split]
                seeds=[]
                for seed in config['fit_seeds']:
                    rr=[r for r in selected if r['seed']==seed]
                    seeds.append({key:float(np.mean([r[key] for r in rr if r[key] is not None]))
                        for key in ('mae','mse','nmse','explained_variance')})
                summary.append(dict(group=group,loss=loss,split=split,
                    covered_instances=len({r['instance'] for r in selected}),
                    **{key:float(np.mean([r[key] for r in seeds])) for key in seeds[0]},
                    mse_seed_min=min(r['mse'] for r in seeds),mse_seed_max=max(r['mse'] for r in seeds)))
    paired=[]
    for loss in config['losses']:
        names=sorted({r['instance'] for r in per_instance if r['split']=='development'})
        for control in ('R0','Rcap','Rprod'):
            differences=[]; relative=[]; seed_differences=[]
            for name in names:
                groups={g:[r['mse'] for r in per_instance if r['loss']==loss and r['group']==g and r['instance']==name]
                        for g in ('Rjoint',control)}
                differences.append(float(np.mean(groups['Rjoint'])-np.mean(groups[control])))
                relative.append(float((np.mean(groups['Rjoint'])/np.mean(groups[control])-1)*100))
            for seed in config['fit_seeds']:
                g={group:np.mean([r['mse'] for r in per_instance if r['split']=='development' and
                    r['group']==group and r['loss']==loss and r['seed']==seed]) for group in ('Rjoint',control)}
                seed_differences.append(float(g['Rjoint']-g[control]))
            rng=np.random.RandomState(20261021)
            boot=np.asarray(differences)[rng.randint(len(names),size=(10000,len(names)))].mean(1)
            joint=next(r for r in summary if r['loss']==loss and r['group']=='Rjoint' and r['split']=='development')
            reference=next(r for r in summary if r['loss']==loss and r['group']==control and r['split']=='development')
            paired.append(dict(loss=loss,control=control,covered_instances=len(names),
                joint_minus_control_mse=float(np.mean(differences)),
                joint_relative_mse_percent=(joint['mse']/reference['mse']-1)*100,
                lower_mse_instances=sum(x<0 for x in differences),
                lower_mse_fit_seeds=sum(x<0 for x in seed_differences),
                instance_bootstrap_low=float(np.quantile(boot,.025)),instance_bootstrap_high=float(np.quantile(boot,.975))))
    write_csv('collection.csv',trajectories);write_csv('coverage_and_information.csv',counts)
    write_csv('per_instance_prediction.csv',per_instance);write_csv('prediction_summary.csv',summary)
    write_csv('paired_comparison.csv',paired);write_csv('parameters.csv',parameters)
    write_csv('learning_curves.csv',json.loads((RUN/'offline/curves.json').read_text()))
    total_cpu=sum(r['worker_cpu_seconds'] for r in trajectories)
    extract=sum(r['search_extraction_wall_seconds'] for r in trajectories)
    callbacks=sum(r['callback_wall_seconds'] for r in trajectories)
    aggregate=dict(collection_worker_cpu_seconds=total_cpu,
        collection_parent_cpu_seconds=json.loads((RUN/'collection/summary.json').read_text())['parent_cpu_seconds'],
        transitions=len(data['target']),episodes=50,
        train_instances_with_decisions=len(fit_cost['train_names']),
        development_instances_with_decisions=len(fit_cost['development_names']),
        zero_decision_instances=[r['instance'] for r in counts if r['transitions']==0],
        extraction_total_wall_seconds=extract,extraction_ms_per_decision=extract/len(data['target'])*1000,
        extraction_fraction_of_callback_wall=extract/callbacks,
        offline_cpu_seconds=offline_total_cpu,
        offline_failed_attempts=[json.loads(p.read_text()) for p in (RUN/'offline').glob('failed-*_cost.json')],
        cache_cost=cache_cost, fit_cost=fit_cost, prediction_summary=summary,paired_comparison=paired,
        constant_predictor_development_mse=float(np.mean([r['mse'] for r in constants['per_instance']
            if r['split']=='development'])),
        target_outside_hl_support_fraction=float(np.mean((data['target'] < -36)|(data['target'] > 52))),
        hl_gaussian_outside_support_mass_max=max(.5*math.erfc((float(y)+36)/((88/64*.75)*2**.5))+
            .5*math.erfc((52-float(y))/((88/64*.75)*2**.5)) for y in data['target']),
        statistical_unit='instance; fit seeds are repeated fits, not independent MILP samples',
        conclusion_scope='fixed low-training Actor distribution, frozen B0 encoder, held-out development instances; no online benefit claim',
        stage5_started=False)
    write_json(DOC/'summary.json',aggregate)
    write_json(DOC/'verification.json',dict(passed=all(checks.values()),
        passed_scope='engineering correctness, provenance, comparability and recovery only',
        information_benefit_validated=False,
        scientific_conclusion='inconclusive due to common value-scale underfitting',
        any_model_beats_train_mean_on_development_mse=any(r['mse']<aggregate['constant_predictor_development_mse']
            for r in summary if r['split']=='development'),
        checks=checks,stage5_started=False))
    if not all(checks.values()):raise AssertionError(checks)
    print(json.dumps(dict(passed=True,episodes=50,transitions=len(data['target']),fits=24,summary=str(DOC/'summary.json'))))


if __name__=='__main__':main()
