"""Fixed-node search comparison across installed and read-only measurement bindings."""
import argparse
import json
from pathlib import Path
import pyscipopt
from project.utils.settings import init_params, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observed', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/root/research')
    cutoffs = json.loads((root / 'tgppo/datasets/cutoffs.json').read_text())
    rows = []
    for name in ('lseu', 'stein27'):
        model = pyscipopt.Model()
        init_params(model, dict(node_limit=150, time_limit=10.), settings['sandbox'])
        model.setBoolParam('randomization/permutevars', True)
        model.setIntParam('randomization/permutationseed', 0)
        model.readProblem(str(root / 'tgppo/datasets/train_instances' / (name + '.mps.gz')))
        model.setObjlimit(cutoffs[name]['cutoff'])
        if args.observed:
            from project.measurement import BoundTrace
            trace = BoundTrace()
            model.includeEventhdlr(trace, 'EquivalenceTrace', 'read-only measurement')
        model.optimize()
        row = dict(name=name, status=str(model.getStatus()), nodes=model.getNNodes(),
                   primal=model.getPrimalbound(), dual=model.getDualbound(), gap=model.getGap())
        if args.observed:
            first = model.getIntegralSnapshot()
            second = model.getIntegralSnapshot()
            assert first == second, 'Snapshot mutated the stopped solver state'
            assert first['integral'] == model.getPrimalDualIntegral()
            row['snapshot_is_readonly'] = True
        rows.append(row)
        model.freeProb()
    args.output.write_text(json.dumps(dict(binding=pyscipopt.__file__, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
