"""Freeze this new package after local checks, without training or solving."""
from pathlib import Path
import ast
import difflib
import json
from paper import PACKAGE, ROOT, protocol, read, sha, write


def main():
    c = protocol()
    checks = read(ROOT/'docs/methods/tgppo_paper_aligned_20261106/CPU_CHECKS.json')
    if not checks['passed']:
        raise ValueError('CPU preparation checks failed')
    protected = read(PACKAGE/'docs/protected_sources.json')
    for relative, digest in protected.items():
        if sha(ROOT/relative) != digest:
            raise ValueError('Historical file changed: ' + relative)
    source = ROOT/'tgppo/experiments/supervised-ppo-20261105/code'
    patch = []
    for relative in ['project/runtime.py', 'project/agent/agent.py', 'project/memory/memory.py']:
        patch.extend(difflib.unified_diff((source/relative).read_text().splitlines(True),
             (PACKAGE/'code'/relative).read_text().splitlines(True),
             fromfile='accepted/'+relative, tofile='paper-aligned/'+relative))
    (PACKAGE/'docs/core_changes.diff').write_text(''.join(patch))
    files = {}
    for p in sorted(PACKAGE.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name != 'release.json':
            if p.suffix == '.py':
                ast.parse(p.read_text())
            files[str(p.relative_to(PACKAGE))] = sha(p)
    inputs = {}
    external = [ROOT/'tgppo/configs/paper_hparams.json', ROOT/'tgppo/datasets/cutoffs.json',
                ROOT/'tgppo/runs/light-actor-20261105/offline/expert.npz',
                ROOT/'tgppo/runs/light-actor-20261105/offline/data_audit.json',
                ROOT/'tgppo/repo/plots_crc/finished_instances.xlsx',
                ROOT/'tgppo/repo/plots_crc/timelimit_instances.xlsx']
    external += [ROOT/'tgppo/datasets/train_instances'/(n+'.mps.gz') for n in c['train']]
    external += [ROOT/'tgppo/datasets/test_instances'/(n+'.mps.gz') for n in c['test']]
    for p in external:
        inputs[str(p.relative_to(ROOT))] = sha(p)
    release = dict(passed=True, status='prepared_not_submitted', files=files, inputs=inputs,
                   protected_files=len(protected), real_training_episodes=0, real_scip_solves=0,
                   gpu_started=False, target_gpu_checks='required on each actual Custom allocation',
                   cpu_check_evidence=checks)
    write(PACKAGE/'docs/release.json', release)
    print(json.dumps(dict(package_files=len(files), external_inputs=len(inputs),
                         protected_sources=len(protected), passed=True)))


if __name__ == '__main__':
    main()
