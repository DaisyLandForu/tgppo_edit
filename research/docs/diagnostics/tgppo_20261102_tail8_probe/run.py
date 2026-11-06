"""Paired replacement of the narrow tail; accepted optimizer loop stays unchanged."""
import os
from pathlib import Path
import json
import sys
import time

START = time.monotonic()
HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2]
PACKAGE = RESEARCH / 'tgppo/experiments/offline-head-probe-20261102'
SOURCE = RESEARCH / 'tgppo/runs/offline-head-probe-20261102'
PREVIOUS = HERE.parent / 'tgppo_20261102_head_reset'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def worker():
    sys.path.insert(0, str(PACKAGE / 'code'))
    import torch
    import common as c
    import runner as accepted
    from unittest.mock import patch
    c.set_threads()
    p, prepared = c.load_prepared(SOURCE, check_sources=True)
    plan = json.loads((HERE / 'protocol.json').read_text())
    assert plan['source_seeds']==[11,22] and plan['head_initialization_seeds']==[101,202]
    assert plan['max_steps_per_trial']==p['max_actor_steps']==20
    assert plan['maximum_total_steps']==80 and plan['wall_seconds']==180
    assert c.sha(HERE/'run.py')==plan['runner_sha256']
    assert c.sha(PACKAGE/'protocol.json')==plan['accepted_protocol_file_sha256']
    before = dict(json.loads((PREVIOUS/'frozen_inputs.json').read_text()))
    seal = json.loads((PREVIOUS/'delivery_manifest.json').read_text())['files']
    before[str((PREVIOUS/'delivery_manifest.json').relative_to(RESEARCH))] = c.sha(PREVIOUS/'delivery_manifest.json')
    for name,digest in seal.items():
        before[str((PREVIOUS/name).relative_to(RESEARCH))] = digest
    for name in ['run.py','protocol.json','AUTHORIZATION.md']:
        before[str((HERE/name).relative_to(RESEARCH))] = c.sha(HERE/name)
    for name,digest in before.items():
        assert c.sha(RESEARCH/name)==digest,name
    save(HERE/'frozen_inputs.json',before)

    def tail_head(cache, variant):
        assert variant=='original'
        head = c.make_head(cache,variant)
        assert len(head.BranchingNet)==8 and head.n_units_dict[5]==8
        last = torch.nn.Linear(8,1)
        with torch.no_grad():
            last.weight.zero_();last.bias.zero_()
        head.BranchingNet = torch.nn.ModuleList([*list(head.BranchingNet[:5]),torch.nn.Sequential(last)])
        head.n_units_dict = {i:head.n_units_dict[i] for i in range(6)}
        head.n_layers = 6
        # Keep the whole original 510-output gate. Its last 6 outputs are unused;
        # trimming/reinitializing it would introduce another intervention.
        assert head.n_attentional_units==510 and sum(head.n_units_dict.values())==504
        assert sum(q.numel() for q in head.parameters())==241263
        return head.eval()

    reference = json.loads((PREVIOUS/'summary.json').read_text())
    results,curves,total_steps = [],[],0
    for source_seed in plan['source_seeds']:
        cache = torch.load(SOURCE/prepared['caches'][str(source_seed)]['path'],map_location='cpu')
        for init_seed in plan['head_initialization_seeds']:
            trial = f's{source_seed}-init{init_seed}'
            d = HERE/'trials'/trial;d.mkdir(parents=True,exist_ok=False)
            original_initial = torch.load(PREVIOUS/'trials'/trial/'initial.pt',map_location='cpu')
            paired_cache = dict(cache,head_state=original_initial)
            c.seed_all(init_seed)
            head = tail_head(paired_cache,'original')
            initial = {k:v.detach().clone() for k,v in head.state_dict().items()}
            retained = [k for k in initial if k.startswith('GatingNet.') or any(k.startswith(f'BranchingNet.{i}.') for i in range(5))]
            assert len(retained)==24 and all(torch.equal(initial[k],original_initial[k]) for k in retained)
            # Verify the exact gated 8-D boundary, including gate slice 496:504.
            old = c.make_head(paired_cache,'original');seen={}
            def capture(name):
                def hook(module,args):seen[name]=args[0].detach().clone()
                return hook
            hooks=[old.BranchingNet[5].register_forward_pre_hook(capture('old')),
                   head.BranchingNet[5].register_forward_pre_hook(capture('new'))]
            b=c.collate(cache['train'],next(c.chunks(cache['train'],16)))
            with torch.no_grad():old(b['features'],b['tree']);head(b['features'],b['tree'])
            for hook in hooks:hook.remove()
            assert seen['old'].shape[-1]==8 and torch.equal(seen['old'],seen['new'])
            check=c.evaluate(head,cache['train'],p)
            assert max(r['tv_uniform'] for r in check)==0
            ratio_error=max(abs(r['old_action_ratio']-1) for r in check)
            assert ratio_error<=p['initial_behavior_ratio_max_error']
            torch.save(initial,d/'initial.pt')
            save(d/'initialization.json',dict(source_seed=source_seed,initialization_seed=init_seed,
                retained_parameter_tensors_identical=24,gated_8d_boundary_exact=True,
                original_gate_outputs=510,used_gate_outputs=504,unused_gate_outputs=6,
                total_head_parameters=241263,unused_gate_output_parameters=1542,
                uniform_max_tv=0.0,behavior_ratio_max_error=ratio_error,
                source_initial_sha256=c.sha(PREVIOUS/'trials'/trial/'initial.pt'),initial_sha256=c.sha(d/'initial.pt')))
            provenance=dict(experiment='tgppo-tail8-probe-20261102',source_seed=source_seed,
                head_initialization_seed=init_seed,tail='gated 8 -> Linear(8,1)',
                protocol_sha256=c.sha(HERE/'protocol.json'),runner_sha256=c.sha(HERE/'run.py'),
                cache_sha256=prepared['caches'][str(source_seed)]['sha256'],
                paired_original_initial_sha256=c.sha(PREVIOUS/'trials'/trial/'initial.pt'),initial_sha256=c.sha(d/'initial.pt'))
            del head,old,b,seen,check
            print(f'BEGIN {trial}; completed global steps={total_steps}',flush=True)
            # Dependency injection affects only this process/call, no file changes.
            with patch.object(accepted,'make_head',tail_head):
                result=accepted.run_head_fit(paired_cache,'original',p,d,provenance)
            total_steps+=result['completed_steps']
            save(HERE/'steps.json',dict(committed_steps=total_steps,maximum_steps=80,last_trial=trial))
            assert result['status']=='completed' and result['completed_steps']==20
            history=json.loads((d/'history.json').read_text())
            assert [x['step'] for x in history]==list(range(21))
            assert result['first']['train']['tv_uniform_max']==result['first']['monitor']['tv_uniform_max']==0
            cp=torch.load(d/'latest.pt',map_location='cpu')
            assert cp['step']==20 and len(cp['optimizer']['state'])==26
            assert {int(v['step']) for v in cp['optimizer']['state'].values()}=={20}
            assert not (d/'inflight.json').exists()
            ref=next(x for x in reference['results'] if x['trial']==trial)
            change=result['last']['surrogate_loss']-result['first']['surrogate_loss']
            tv=result['last']['train']['tv_initial_p50']
            row=dict(trial=trial,source_seed=source_seed,initialization_seed=init_seed,steps=20,
                surrogate_change=change,train_tv_p50=tv,train_tv_max=result['last']['train']['tv_initial_max'],
                monitor_tv_p50=result['last']['monitor']['tv_initial_p50'],monitor_tv_max=result['last']['monitor']['tv_initial_max'],
                monitor_response_states=result['last']['monitor']['numerical_response_states'],
                mean_training_kl=result['last']['train']['mean_kl_initial'],
                centered_logit_rms_p50=result['last']['monitor']['centered_logit_delta_rms_p50'],
                common_logit_shift_p50=result['last']['monitor']['abs_common_logit_shift_p50'],
                measurable_fit=change<=-plan['surrogate_decrease_screen'] and tv>plan['tv_screen'],
                changed_parameter_elements=sum(int((cp['head'][k]!=v).sum()) for k,v in initial.items()),
                original_reference_surrogate_change=ref['surrogate_change'],original_reference_train_tv=ref['train_tv_p50'],
                original_reference_monitor_tv=ref['monitor_tv_p50'])
            results.append(row)
            for x in history:
                curves.append(dict(trial=trial,step=x['step'],surrogate_loss=x['surrogate_loss'],
                    train_tv=x['train']['tv_initial_p50'],monitor_tv=x['monitor']['tv_initial_p50'],train_kl=x['train']['mean_kl_initial']))
            print('END '+json.dumps(row),flush=True)
            del paired_cache,initial,original_initial,cp,result,history
        del cache
    assert total_steps==80
    for name,digest in before.items():assert c.sha(RESEARCH/name)==digest,name
    assert not torch.cuda.is_initialized() and 'pyscipopt' not in sys.modules
    c.write_csv(HERE/'results.csv',results);c.write_csv(HERE/'curves.csv',curves)
    save(HERE/'summary.json',dict(complete=True,results=results,passed_trials=sum(r['measurable_fit'] for r in results),
        completed_head_steps=80,new_solver_tasks=0,critic_updates=0,full_actor_updates=0))
    save(HERE/'verification.json',dict(status='passed',protected_files_unchanged=len(before),
        source_data_and_reference_results_unchanged=True,retained_parameters_identical=True,gated_8d_boundary_exact=True,
        paired_uniform_initialization_checked=True,trials=4,steps_per_trial=20,adam_parameter_tensors_per_trial=26,
        head_parameters=241263,unused_gate_output_parameters=1542,cuda_initialized=False,pyscipopt_imported=False,
        worker_seconds=time.monotonic()-START))

def supervise():
    import fcntl
    import signal
    import subprocess
    with (HERE / '.lock').open('a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE / 'budget.json').exists():
            raise RuntimeError('Budget already exists. No automatic retry/reset; preserve for review.')
        save(HERE / 'budget.json', dict(status='running', wall_limit_seconds=180, maximum_head_steps=80))
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        env.pop('PYTHONPATH', None)
        child = None
        code, status = 1, 'failed'
        def interrupted(signum, frame):
            raise InterruptedError(signum)
        old = {s: signal.signal(s, interrupted) for s in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]}
        try:
            with (HERE / 'run.log').open('x') as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=max(1, 175-(time.monotonic()-START)))
                    status = 'completed' if code == 0 else 'failed'
                except (subprocess.TimeoutExpired, InterruptedError) as exc:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code, status = (124, 'wall_timeout') if isinstance(exc, subprocess.TimeoutExpired) else (130, 'interrupted')
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            steps = json.loads((HERE / 'steps.json').read_text()) if (HERE / 'steps.json').exists() else {}
            save(HERE / 'budget.json', dict(status=status, exit_code=code, elapsed_seconds=time.monotonic()-START,
                wall_limit_seconds=180, reserved_cleanup_seconds=5, maximum_head_steps=80,
                committed_steps_record=steps.get('committed_steps',0),
                note='Per-trial steps.json/latest.pt/inflight.json remain authoritative if interrupted.'))
        print(json.dumps(json.loads((HERE / 'budget.json').read_text()), ensure_ascii=False))
        raise SystemExit(code)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        worker()
    elif not sys.argv[1:]:
        supervise()
    else:
        raise SystemExit('Usage: run.py')
