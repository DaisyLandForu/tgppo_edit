"""Fail-closed baseline identity and truthful solver result semantics."""
import hashlib
import json
import math
from pathlib import Path

PROTOCOL_VERSION = 'stage2-correctness-v1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def baseline_identity(instance, solver_seed, limits, params, cutoff, cutoff_provenance):
    # Include implementation hashes: normalization must use exactly the same settings.
    return dict(instance_sha256=sha256(instance), solver_seed=int(solver_seed),
                limits=limits, params=params, cutoff=cutoff,
                cutoff_provenance=cutoff_provenance,
                settings_sha256=sha256(Path(__file__).with_name('settings.py')),
                solver_stack='SCIP6.0.1-SoPlex4.0.1-customPySCIPOpt1.4.9',
                protocol_version=PROTOCOL_VERSION)


def validate_baseline(baseline, identity):
    if baseline is None or baseline.get('identity') != identity:
        raise ValueError('Missing/mismatched baseline identity (instance/seed/time/settings/cutoff)')
    result = baseline['result']
    if not result.get('valid') or not result.get('task_terminal'):
        raise ValueError('Normalization baseline did not finish a valid finite task')
    for key in ('nnodes', 'gap', 'primalDualIntegral'):
        if not isinstance(result[key], (int, float)) or not math.isfinite(result[key]):
            raise ValueError('Missing/nonfinite baseline metric: ' + key)
    return result


def solver_stats(model, cutoff_applied, cutoff_verified):
    status = str(model.getStatus())
    nsols = len(model.getSols())  # Old binding has no getNSols().
    original_optimal = status == 'optimal' and nsols > 0
    cutoff_proof = (status in ('optimal', 'infeasible') and cutoff_applied
                    and cutoff_verified)
    proof_completed = original_optimal or cutoff_proof or (
        status in ('infeasible', 'unbounded') and not cutoff_applied)
    terminal = proof_completed or status == 'timelimit'
    infinity = float(model.infinity())
    def finite(value):
        return math.isfinite(value) and abs(value) < infinity
    raw = dict(gap=float(model.getGap()), primal_bound=float(model.getPrimalbound()),
               dual_bound=float(model.getDualbound()),
               primalDualIntegral=float(model.getPrimalDualIntegral()))
    # Preserve finite SCIP infinity sentinels such as 1e20 verbatim. JSON has no IEEE inf.
    values = {k: v if math.isfinite(v) else None for k, v in raw.items()}
    values.update({k + '_finite': finite(v) for k, v in raw.items()})
    values.update({k + '_ieee_nonfinite': str(v) for k, v in raw.items() if not math.isfinite(v)})
    return dict(values, status=status, episode_ended=True, task_terminal=terminal,
                proof_completed=proof_completed, original_optimal=original_optimal,
                cutoff_proof=cutoff_proof, cutoff_applied=cutoff_applied,
                cutoff_verified=cutoff_verified, has_feasible_solution=nsols > 0,
                num_stored_solutions=nsols,
                valid=terminal and all(math.isfinite(v) for v in raw.values()),
                objective=float(model.getObjVal()) if nsols > 0 else None,
                nnodes=int(model.getNNodes()), scip_solve_time=float(model.getSolvingTime()),
                pdi_semantics='native_raw_until_actual_stop', scip_infinity=infinity)
