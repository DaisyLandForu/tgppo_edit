#!/usr/bin/env python3
"""Construct and check one tree solution against the unchanged original CIP.

No optimize(), presolve(), training, or injection into existing jobs. All
incidence, business demands, and costs are read from the actual CIP constraints.
"""
from collections import Counter, defaultdict, deque
import datetime as dt
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import time

import pyscipopt as scip

HERE = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    zone = dt.timezone(dt.timedelta(hours=8))
    out = HERE / 'reports' / ('diagnosis_' + dt.datetime.now(zone).strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    manifest_path = next((HERE / 'artifacts').glob('model-*/manifest.json'))
    manifest = json.loads(manifest_path.read_text())
    model_path = manifest_path.parent / 'model.cip'
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_sha == manifest['files']['model.cip']
    model = scip.Model()
    model.readProblem(str(model_path))
    model.setRealParam('numerics/feastol', 1e-6)
    variables = {v.name: v for v in model.getVars()}
    constraints = model.getConss()
    print('Read original CIP; constructing incidence and demands', flush=True)
    arc_ends = defaultdict(dict)
    z_demands = {}
    costs, flow_demands = {}, defaultdict(dict)
    row_counts, variable_counts = Counter(), Counter()
    for name in variables:
        variable_counts['X' if name.startswith('X_') else 'Z' if name.startswith('Z_') else
                        'f' if name.startswith('f_') else 'cost'] += 1
    for c in constraints:
        name = c.name
        for prefix in ('CostDef_', 'Total_Cost_Calc', 'TotalSrcFlow_', 'FlowBalance_', 'f_cap_sum_',
                       'FlowBal_', 'InDegree_', 'FlowX_', 'X_Mutex_'):
            if name.startswith(prefix):
                row_counts[prefix] += 1
                break
        if name.startswith('FlowBal_'):
            node = name[len('FlowBal_'):]
            lhs, rhs = model.getLhs(c), model.getRhs(c)
            assert lhs == rhs
            z_demands[node] = rhs
            for var, coeff in model.getValsLinear(c).items():
                assert coeff in (-1, 1)
                arc_ends[var]['head' if coeff == 1 else 'tail'] = node
        elif name.startswith('CostDef_'):
            group = name[len('CostDef_'):]
            terms = model.getValsLinear(c)
            assert terms.pop('Cost_' + group) == 1
            costs[group] = {var: -coef for var, coef in terms.items()}
    assert all(set(ends) == {'tail', 'head'} for ends in arc_ends.values())
    roots = [n for n, demand in z_demands.items() if demand < 0]
    assert len(roots) == 1
    root = roots[0]
    targets = {n: v for n, v in z_demands.items() if v > 0}
    assert root.startswith('N') and root.endswith('_L0')
    assert all(n.startswith('E') and n.endswith('_L0') for n in targets)
    assert math.fsum(z_demands.values()) == 0
    first_group = sorted(costs)[0]
    weights = {var[len('f_src' + first_group + '_'):]: value for var, value in costs[first_group].items()}
    adjacency = defaultdict(list)
    directed_arcs = {}
    for var, ends in arc_ends.items():
        u, v = ends['tail'], ends['head']
        if u.endswith('_L0') and v.endswith('_L0'):
            adjacency[u].append((v, weights[var[2:]], var))
            directed_arcs[u, v] = var
    distance, parent = {root: 0.}, {}
    queue = [(0., root)]
    while queue:
        dist, u = heapq.heappop(queue)
        if dist != distance[u]:
            continue
        for v, weight, arc in sorted(adjacency[u]):
            if not v.startswith('N'):
                continue
            new_dist = dist + weight
            if new_dist < distance.get(v, math.inf):
                distance[v] = new_dist
                parent[v] = (u, arc)
                heapq.heappush(queue, (new_dist, v))
    for v in targets:
        options = [(distance[u]+weight, u, directed_arcs[u, v])
                   for u, weight, _ in adjacency[v] if u in distance and u.startswith('N')]
        assert options, ('Unreachable terminal', v)
        _, u, arc = min(options)
        parent[v] = (u, arc)
    tree_demand = defaultdict(float)
    for terminal, amount in targets.items():
        v = terminal
        seen = set()
        while v != root:
            assert v not in seen
            seen.add(v)
            tree_demand[v] += amount
            v = parent[v][0]
    values = defaultdict(float)
    tree_adj = defaultdict(list)
    for v, amount in tree_demand.items():
        u, arc = parent[v]
        values[arc] = amount
        values['X_' + arc[2:]] = 1.
        tree_adj[u].append(v)
        tree_adj[v].append(u)
    for c in constraints:
        if not c.name.startswith(('TotalSrcFlow_', 'FlowBalance_')):
            continue
        rhs = model.getRhs(c)
        assert rhs == model.getLhs(c)
        if rhs == 0:
            continue
        first_var, coeff = next(iter(model.getValsLinear(c).items()))
        match = re.match(r'f_src(.+?_\d+_\d+)_', first_var)
        group = match[1]
        assert group in costs
        arc = 'Z_' + first_var[len('f_src' + group + '_'):]
        node = arc_ends[arc]['head' if coeff == 1 else 'tail']
        assert node not in flow_demands[group]
        flow_demands[group][node] = rhs
    for group, demands in flow_demands.items():
        sources = [n for n, v in demands.items() if v < 0]
        assert len(sources) == 1 and math.fsum(demands.values()) == 0
        source = sources[0]
        pred = {source: None}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in sorted(tree_adj[u]):
                if v not in pred:
                    pred[v] = u
                    queue.append(v)
        for dest, amount in demands.items():
            if amount <= 0:
                continue
            v = dest
            while v != source:
                u = pred[v]
                arc = directed_arcs[u, v]
                values['f_src' + group + '_' + arc[2:]] += amount
                v = u
        values['Cost_' + group] = math.fsum(coef*values.get(var, 0.) for var, coef in costs[group].items()) + 1e-4
    values['Total_Cost'] = math.fsum(values['Cost_' + group] for group in costs) + 1e-4
    assert set(values).issubset(variables)
    solution = model.createSol()
    for name, value in values.items():
        if value:
            model.setSolVal(solution, variables[name], value)
    print('Constructed tree; checking every original row, bound, and integrality', flush=True)
    checked = model.checkSol(solution, printreason=True, completely=True, checkbounds=True,
                             checkintegrality=True, checklprows=True, original=True)
    violations, max_violation = [], 0.
    for c in constraints:
        terms = model.getValsLinear(c)
        activity = math.fsum(coef*values.get(var, 0.) for var, coef in terms.items())
        violation = max(0., model.getLhs(c)-activity, activity-model.getRhs(c))
        max_violation = max(max_violation, violation)
        if violation > 1e-6:
            violations.append(dict(name=c.name, violation=violation, activity=activity,
                                   lhs=model.getLhs(c), rhs=model.getRhs(c)))
    assert checked and not violations, (checked, violations[:10])
    model.writeSol(solution, str(out/'tree_candidate.sol'), write_zeros=False)
    report = dict(status='passed', checked_beijing=dt.datetime.now(zone).isoformat(),
                  elapsed_seconds=time.monotonic()-started, model_sha256=model_sha,
                  objective_model=values['Total_Cost'], objective_real=values['Total_Cost']/1e6,
                  reference_lower_bound_real=30.762624190748827,
                  relative_gap_vs_reference_lower_bound=(values['Total_Cost']/1e6-30.762624190748827)/30.762624190748827,
                  original_constraints_checked=len(constraints), original_variables=len(variables),
                  scip_original_solution_check=bool(checked), independent_row_check_passed=not violations,
                  max_absolute_row_violation=max_violation, variable_counts=dict(variable_counts),
                  row_counts=dict(row_counts), business_groups=len(costs), connectivity_root=root,
                  connectivity_targets=len(targets), selected_binary_arcs=sum(v for k,v in values.items() if k.startswith('X_')),
                  business_demand_sum=sum(-v for d in flow_demands.values() for v in d.values() if v < 0),
                  nonzero_variables=sum(v != 0 for v in values.values()),
                  optimization_called=False, original_model_modified=False, live_jobs_modified=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  solution_sha256=hashlib.sha256((out/'tree_candidate.sol').read_bytes()).hexdigest())
    (out/'feasibility.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print('DIAGNOSIS_OUTPUT', out, flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    model.freeProb()


if __name__ == '__main__':
    main()
