#!/usr/bin/env python3
"""Read-only aviation-model checks; optional tiny smoke solves are separate."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time

import pyscipopt as scip

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('airwiring_compare', HERE / 'compare.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'validation.json')
    args = parser.parse_args()
    started = time.monotonic()
    entries = list((HERE / 'artifacts').glob('model-*/manifest.json'))
    assert len(entries) == 1, entries
    directory = entries[0].parent
    manifest = json.loads(entries[0].read_text())
    assert c.sha(c.SCIP_DIR / 'sssp_0923.cpp') == manifest['identity']['source_sha256']
    for name, checksum in manifest['files'].items():
        assert c.sha(directory / name) == checksum, name
    source = (directory / 'sssp_0923.original.cpp').read_text()
    generated = (directory / 'export_model.cpp').read_text()
    original_build = source[source.index('void buildModel('):source.index('struct SolverResult {')]
    assert original_build in generated, 'Model formulas must remain verbatim'
    model = scip.Model()
    model.readProblem(str(directory / 'model.cip'))
    assert model.getNVars() == 258164
    assert model.getNConss() == 162800
    assert sum(var.vtype() == 'BINARY' for var in model.getVars()) == 5736
    resolved = {}
    for key, value in c.solver_params(0).items():
        model.setParam(key, value)
        resolved[key] = model.getParam(key)
        assert resolved[key] == value, (key, resolved[key], value)
    assert resolved['limits/time'] == model.infinity() == 1e20
    model.freeProb()  # Deliberately no optimize on the aviation instance.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        counts = c.normalize_inputs(c.SCIP_DIR / 'dataset/edges.csv', c.SCIP_DIR / 'dataset/pairs-4-246.csv', out)
        assert counts == manifest['input_counts']
        assert c.sha(out / 'pairs.normalized.csv') == manifest['files']['pairs.normalized.csv']
        # The compact layout must be validated, rather than silently accepted.
        bad = out / 'bad.csv'
        bad.write_text('header\n1,H,C,missing,missing,-1,\n')
        try:
            c.normalize_inputs(c.SCIP_DIR / 'dataset/edges.csv', bad, out)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid pair was accepted')
        with c.lock(out / 'duplicate.lock'):
            try:
                with c.lock(out / 'duplicate.lock', blocking=False):
                    raise AssertionError('Duplicate run lock was acquired')
            except BlockingIOError:
                pass
        result_root = out / 'results'
        for method in c.METHODS:
            c.write_json(result_root / (method + '-s0') / 'attempt-test' / 'result.json',
                         dict(status='running', method=method, model_sha256='same-model',
                              scip_version=8.04, binding_sha256='binding', libscip_sha256='library',
                              parameters=c.solver_params(0), hardware={'cpu_model': 'same-cpu'}))
        summary_args = argparse.Namespace(results=result_root, seed=0)
        with contextlib.redirect_stdout(io.StringIO()):
            c.summarize(summary_args)
        summary = json.loads((result_root / 'summary-s0.json').read_text())
        assert summary['same_model'] and summary['same_solver_settings']
        assert all(row['status'].startswith('unfinished') for row in summary['rows'])
        bad_result = result_root / 'tgppo-s0/attempt-test/result.json'
        value = json.loads(bad_result.read_text())
        value['parameters']['limits/time'] = 1800.0
        c.write_json(bad_result, value)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.summarize(summary_args)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Mismatched solver settings were accepted')
    smoke_results = []
    for path in sorted((HERE / 'runs').glob('**/smoke-*/result.json')):
        value = json.loads(path.read_text())
        if value.get('smoke') and value.get('solver_status') == 'nodelimit' and not value.get('policy_error'):
            if value['method'] == 'tgppo' and value.get('branches', 0) < 2:
                continue
            if value['status'] != 'failed':
                smoke_results.append(dict(path=str(path.relative_to(HERE)), method=value['method'],
                                          branches=value.get('branches'), solutions=value['solutions'],
                                          solution_checked=value['solution_checked']))
    assert any(v['method'] == 'sssp' and v['solution_checked'] for v in smoke_results)
    assert any(v['method'] == 'tgppo' and v['solutions'] == 0 for v in smoke_results)
    report = dict(status='passed', checked_utc=c.stamp(), elapsed_seconds=time.monotonic()-started,
                  model_directory=str(directory), manifest_sha256=c.sha(entries[0]),
                  original_sources_unchanged=True, original_build_model_verbatim=True,
                  model_readback=dict(variables=258164, binary_variables=5736, constraints=162800),
                  production_parameters=resolved, compact_csv_normalization='passed',
                  invalid_pair_rejected=True, smoke_results=smoke_results,
                  duplicate_lock_rejected=True, summary_rejects_mismatched_settings=True,
                  aviation_optimization_performed=False,
                  mamba_cuda_check='pending: current node has no CUDA; auto-check in GPU Custom job')
    original_path = HERE / 'checks/actor-forward-original.json'
    deployment_path = HERE / 'checks/actor-forward-deployment.json'
    if original_path.exists() and deployment_path.exists():
        original, deployment = [json.loads(p.read_text()) for p in (original_path, deployment_path)]
        differences = []
        action_comparison = []
        for old, new in zip(original['cases'], deployment['cases']):
            assert old['candidates'] == new['candidates']
            differences.extend(abs(a-b) for a,b in zip(old['probabilities'], new['probabilities']))
            ranked = sorted(old['probabilities'], reverse=True)
            action_comparison.append(dict(candidates=old['candidates'], original_action=old['action'],
                                          deployment_action=new['action'],
                                          original_top2_probability_margin=None if len(ranked)<2 else ranked[0]-ranked[1]))
        assert max(differences) < 1e-5
        report['actor_torch_port'] = dict(original=original['torch_version'],
                                         deployment=deployment['torch_version'],
                                         candidate_counts=[r['candidates'] for r in original['cases']],
                                         actions_identical=all(r['original_action']==r['deployment_action'] for r in action_comparison),
                                         action_comparison=action_comparison,
                                         max_absolute_probability_error=max(differences),
                                         interpretation='FP32-close probabilities do not guarantee the same argmax near ties; '
                                                        'this is a common-stack deployment comparison, not an exact replay.')
    c.write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
