"""Audit the saved stage-2 short loop. No training is performed here."""
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import torch
import pyscipopt
from project.runtime import load_actor, read_checkpoint
from project.utils.protocol import sha256, write_json

ROOT = Path('/root/research')
DOC = ROOT / 'docs/stage2'
RUN = ROOT / 'tgppo/runs/stage2-verification/confirmed'
WORK = ROOT / 'tgppo/experiments/stage2'


def read(path):
    return json.loads(path.read_text())


def main():
    tests = (DOC / 'unit-tests.log').read_text()
    assert re.search(r'Ran 20 tests .*\n\nOK', tests), tests[-600:]
    native = read(RUN / 'native-short/jobs/results.json')
    collect = read(RUN / 'train-short/batch-000/results.json')
    evaluation = read(RUN / 'eval-short/jobs/results.json')
    upstream = read(RUN / 'upstream-bridge.json')
    updates = read(RUN / 'train-short/updates.json')[0]
    assert (len(native), len(collect), len(evaluation), len(upstream)) == (4, 4, 2, 4)
    assert all(r['valid'] for r in native + collect + evaluation)
    assert all(r['num_transitions'] == r['branch_count'] for r in collect)
    assert all(r['num_terminal_transitions'] == int(r['branch_count'] > 0) for r in collect)
    assert sum(r['num_transitions'] for r in collect) == updates['num_samples']
    assert any(r['num_transitions'] == 0 for r in collect)  # No invented presolve actions.
    for r in collect + evaluation:
        assert r['didnotrun_count'] == 0 and r['fallback_action_count'] == 0
        assert r['reward_count'] == (r['branch_count'] if r['mode'] == 'collect' else 0)
        if r['status'] == 'timelimit':
            assert r['gap'] > 0 and not r['proof_completed']
    assert all(r['callback_seconds']['critic'] == 0. for r in evaluation)
    assert all(r['total_reward'] == 0. for r in evaluation)
    assert updates['updated'] and updates['num_updates'] > 0
    assert updates['actor_parameter_l2_change'] > 0
    assert updates['initial_ratio_max_error'] < 1e-4
    assert all(r['gap'] == 0 for r in upstream if r['status'] == 'timelimit')
    assert any(r['observed_raw_gap'] > 0 for r in upstream)
    assert all(r['reward_time_limit'] == 3600 for r in upstream)
    checkpoint = RUN / 'train-short/latest.pt'
    before = read_checkpoint(RUN / 'train-short/before-batch-000.pt')
    payload = read_checkpoint(checkpoint)
    assert payload['policy_version'] == 1
    assert payload['progress']['next_batch'] == 1
    assert payload['memory']['num_transitions'] == 0
    for key in ('actor', 'critic'):
        assert all(torch.isfinite(v).all() for v in payload[key].values())
        assert any(not torch.equal(v, before[key][k]) for k, v in payload[key].items())
    frozen_hash = sha256(checkpoint)
    actor, _ = load_actor(checkpoint)
    assert not hasattr(actor, 'critic_network') and not hasattr(actor, 'memory')
    assert sha256(checkpoint) == frozen_hash
    hashes = read(ROOT / 'docs/stage1/upstream_file_hashes.json')
    assert all(sha256(ROOT / 'tgppo/repo' / k) == v for k, v in hashes.items())
    clean = subprocess.check_output(['git', '-C', str(ROOT / 'tgppo/repo'), 'status', '--porcelain'], text=True)
    assert not clean
    unchanged = ['code/project/policy/actor.py', 'code/project/policy/critic.py',
                 'code/project/policy/modules.py', 'code/project/reward/reward.py',
                 'code/project/utils/settings.py']
    assert all(sha256(WORK / k) == hashes[k] for k in unchanged)
    architecture = dict(actor_parameters=sum(v.numel() for v in payload['actor'].values()),
                        critic_parameters=sum(v.numel() for v in payload['critic'].values()),
                        shared_parameter_objects=0)
    manifest = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(WORK.rglob('*.py'))}
    manifest.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted(RUN.rglob('*'))
                     if p.is_file()})
    write_json(DOC / 'artifact_hashes.json', manifest)
    raw_line = next(line for line in tests.splitlines() if 'REAL_AIR04_TIMEOUT_EVIDENCE=' in line)
    real_timeout = json.loads(raw_line.split('REAL_AIR04_TIMEOUT_EVIDENCE=', 1)[1])
    assert real_timeout['status'] == 'timelimit' and real_timeout['gap'] > 0
    report = dict(passed=True, purpose='stage2_correctness_only_not_efficacy_or_full_reproduction',
        reward_version='upstream_h3_t2', reward_sha256=sha256(WORK / 'code/project/reward/reward.py'),
        formal_training_started=False, stage3_started=False,
        unit_and_integration_tests=20, native_jobs=len(native), collection_jobs=len(collect),
        development_evaluation_jobs=len(evaluation), upstream_bridge_jobs=len(upstream),
        architecture=architecture, updates=updates,
        training_summary=read(RUN / 'train-short/summary.json'),
        real_timeout=real_timeout, actor_only_critic_seconds=0.,
        checkpoint=str(checkpoint), checkpoint_sha256=frozen_hash,
        training_instances=[r['instance_name'] for r in collect],
        development_instances=[r['instance_name'] for r in evaluation],
        python=sys.version, torch=torch.__version__, pyscipopt=pyscipopt.__version__,
        platform=platform.platform(), code_manifest='artifact_hashes.json',
        upstream_clean=True, original_architecture_reward_settings_unchanged=unchanged)
    write_json(DOC / 'verification.json', report)
    print(json.dumps(dict(passed=True, samples=updates['num_samples'],
                         minibatch_updates=updates['num_updates'], tests=20)))


if __name__ == '__main__':
    main()
