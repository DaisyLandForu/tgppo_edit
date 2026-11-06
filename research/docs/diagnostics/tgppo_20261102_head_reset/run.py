"""One local, globally bounded head-reset experiment; reuse the accepted fitting code."""
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
REVIEW = HERE.parent / 'tgppo_20261102_head_probe_review'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def worker():
    sys.path.insert(0, str(PACKAGE / 'code'))
    import torch
    import common as c
    import runner as accepted
    from frozen_model.modules import TreeGateBranchingNet
    c.set_threads()
    p, prepared = c.load_prepared(SOURCE, check_sources=True)
    plan = json.loads((HERE / 'protocol.json').read_text())
    assert plan['source_seeds'] == [11, 22] and plan['head_initialization_seeds'] == [101, 202]
    assert plan['max_steps_per_trial'] == p['max_actor_steps'] == 20
    assert plan['maximum_total_steps'] == 80 and plan['wall_seconds'] == 180
    assert c.sha(PACKAGE / 'protocol.json') == plan['accepted_protocol_file_sha256']
    assert c.sha(HERE / 'run.py') == plan['runner_sha256']
    before = dict(json.loads((REVIEW / 'input_hashes.json').read_text()))
    seal = json.loads((REVIEW / 'delivery_manifest.json').read_text())['files']
    before[str((REVIEW / 'delivery_manifest.json').relative_to(RESEARCH))] = c.sha(REVIEW / 'delivery_manifest.json')
    for name, digest in seal.items():
        before[str((REVIEW / name).relative_to(RESEARCH))] = digest
    for name in ['run.py', 'protocol.json', 'AUTHORIZATION.md']:
        before[str((HERE / name).relative_to(RESEARCH))] = c.sha(HERE / name)
    for name, digest in before.items():
        assert c.sha(RESEARCH / name) == digest, name
    save(HERE / 'frozen_inputs.json', before)
    initial_by_seed, results, curves = {}, [], []
    total_steps = 0
    for source_seed in plan['source_seeds']:
        cache = torch.load(SOURCE / prepared['caches'][str(source_seed)]['path'], map_location='cpu')
        references = {v: accepted.verify_job(SOURCE, f'{v}-s{source_seed}')[0] for v in ['original', 'linear']}
        for init_seed in plan['head_initialization_seeds']:
            trial = f's{source_seed}-init{init_seed}'
            d = HERE / 'trials' / trial
            d.mkdir(parents=True, exist_ok=False)
            c.seed_all(init_seed)
            h = cache['hidden_dim']
            fresh = TreeGateBranchingNet(h, h, 2, 1, norm='layer', depth=2, hidden_size=h).eval()
            assert sum(q.numel() for q in fresh.parameters()) == 241315
            assert {k: v.shape for k, v in fresh.state_dict().items()} == {k: v.shape for k, v in cache['head_state'].items()}
            for layer in fresh.modules():
                if isinstance(layer, torch.nn.LayerNorm):
                    assert torch.equal(layer.weight, torch.ones_like(layer.weight))
                    assert torch.equal(layer.bias, torch.zeros_like(layer.bias))
            with torch.no_grad():
                fresh.BranchingNet[-1][0].weight.zero_()
                fresh.BranchingNet[-1][0].bias.zero_()
            initial = {k: v.detach().clone() for k, v in fresh.state_dict().items()}
            if init_seed in initial_by_seed:
                assert all(torch.equal(initial[k], v) for k, v in initial_by_seed[init_seed].items())
            else:
                initial_by_seed[init_seed] = initial
            final_prefix = f'BranchingNet.{len(fresh.BranchingNet)-1}.0.'
            assert any(not torch.equal(v, cache['head_state'][k]) for k, v in initial.items() if not k.startswith(final_prefix))
            torch.save(initial, d / 'initial.pt')
            new_cache = dict(cache, head_state=initial)
            check_head = c.make_head(new_cache, 'original')
            check = c.evaluate(check_head, cache['train'], p)
            assert max(r['tv_uniform'] for r in check) == 0
            assert max(abs(r['old_action_ratio']-1) for r in check) <= p['initial_behavior_ratio_max_error']
            save(d / 'initialization.json', dict(source_seed=source_seed, initialization_seed=init_seed,
                uniform_max_tv=0.0, behavior_ratio_max_error=max(abs(r['old_action_ratio']-1) for r in check),
                architecture_unchanged=True, layernorm_reset_checked=True, final_readout_zero=True,
                initial_sha256=c.sha(d / 'initial.pt'), initialization_shared_across_source_seeds=True))
            del fresh, check_head, check
            provenance = dict(experiment='tgppo-original-head-reset-20261102', source_seed=source_seed,
                head_initialization_seed=init_seed, original_head_fully_reinitialized=True,
                protocol_sha256=c.sha(HERE / 'protocol.json'), runner_sha256=c.sha(HERE / 'run.py'),
                cache_sha256=prepared['caches'][str(source_seed)]['sha256'], initial_sha256=c.sha(d / 'initial.pt'))
            print(f'BEGIN {trial}; completed global steps={total_steps}', flush=True)
            result = accepted.run_head_fit(new_cache, 'original', p, d, provenance)
            total_steps += result['completed_steps']
            save(HERE / 'steps.json', dict(committed_steps=total_steps, maximum_steps=80, last_trial=trial))
            assert result['status'] == 'completed' and result['completed_steps'] == 20
            assert result['first']['train']['tv_uniform_max'] == 0 and result['first']['monitor']['tv_uniform_max'] == 0
            history = json.loads((d / 'history.json').read_text())
            assert [r['step'] for r in history] == list(range(21))
            cp = torch.load(d / 'latest.pt', map_location='cpu')
            assert cp['step'] == 20 and len(cp['optimizer']['state']) == 34
            assert {int(v['step']) for v in cp['optimizer']['state'].values()} == {20}
            assert not (d / 'inflight.json').exists()
            change = result['last']['surrogate_loss'] - result['first']['surrogate_loss']
            tv = result['last']['train']['tv_initial_p50']
            results.append(dict(trial=trial, source_seed=source_seed, initialization_seed=init_seed, steps=20,
                surrogate_change=change, train_tv_p50=tv, monitor_tv_p50=result['last']['monitor']['tv_initial_p50'],
                monitor_tv_max=result['last']['monitor']['tv_initial_max'],
                monitor_response_states=result['last']['monitor']['numerical_response_states'],
                mean_training_kl=result['last']['train']['mean_kl_initial'],
                centered_logit_rms_p50=result['last']['monitor']['centered_logit_delta_rms_p50'],
                common_logit_shift_p50=result['last']['monitor']['abs_common_logit_shift_p50'],
                measurable_fit=change <= -plan['surrogate_decrease_screen'] and tv > plan['tv_screen'],
                changed_parameter_elements=sum(int((cp['head'][k] != v).sum()) for k, v in initial.items()),
                original_reference_surrogate_change=references['original']['last']['surrogate_loss']-references['original']['first']['surrogate_loss'],
                linear_reference_surrogate_change=references['linear']['last']['surrogate_loss']-references['linear']['first']['surrogate_loss']))
            for row in history:
                curves.append(dict(trial=trial, step=row['step'], surrogate_loss=row['surrogate_loss'],
                    train_tv=row['train']['tv_initial_p50'], monitor_tv=row['monitor']['tv_initial_p50'],
                    train_kl=row['train']['mean_kl_initial']))
            print('END ' + json.dumps(results[-1]), flush=True)
            del new_cache, result, cp, history
        del cache
    assert total_steps == 80
    for name, digest in before.items():
        assert c.sha(RESEARCH / name) == digest, name
    assert not torch.cuda.is_initialized() and 'pyscipopt' not in sys.modules
    c.write_csv(HERE / 'results.csv', results)
    c.write_csv(HERE / 'curves.csv', curves)
    save(HERE / 'summary.json', dict(complete=True, results=results, passed_trials=sum(r['measurable_fit'] for r in results),
        completed_head_steps=total_steps, new_solver_tasks=0, critic_updates=0, full_actor_updates=0))
    save(HERE / 'verification.json', dict(status='passed', protected_files_unchanged=len(before),
        source_data_and_reference_results_unchanged=True, architecture_and_layernorm_reset_checked=True,
        paired_uniform_initialization_checked=True, same_initial_tensors_across_source_seeds=True,
        trials=4, steps_per_trial=20, adam_parameter_tensors_per_trial=34, cuda_initialized=False,
        pyscipopt_imported=False, source_solver_tasks=0, worker_seconds=time.monotonic()-START))


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
