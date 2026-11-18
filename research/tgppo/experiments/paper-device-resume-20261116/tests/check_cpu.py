"""CPU-only handoff audit + synthetic learner bridge; never production training."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'code'))
import resume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['aligned','trainmode'], required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    paper = resume.recover.boot(args.experiment)
    c = resume.recover.verify(paper)
    import torch
    from project import runtime as rt
    from engineering import same_state, bridge
    results = []
    variants = ['b0'] if args.experiment == 'trainmode' else ['mlp','mlp_bc']
    for variant in variants:
        for seed in [11,22]:
            spec = resume.condition(variant,seed)
            source, dest = spec['source'], spec['output']
            info = paper.read(dest/'device_continuation.json')
            assert paper.sha(source/'latest.pt') == info['source_latest_sha256']
            assert paper.sha(source/'budget.json') == info['source_budget_sha256']
            original, actual = rt.read_checkpoint(source/'latest.pt'), rt.read_checkpoint(dest/'latest.pt')
            assert same_state(resume.convert(original,spec['device']),actual)
            assert actual['config'] == paper.configuration(c,variant,seed,spec['device'])
            assert paper.read(source/'budget.json') == paper.read(dest/'budget.json')
            pending_ok = None
            if info['pending']:
                old_pending = rt.read_checkpoint(source/'pending.pt')
                new_pending = rt.read_checkpoint(dest/'pending.pt')
                cleanup = Path(new_pending['progress']['collection_directory'])
                assert dest in cleanup.parents and cleanup.exists() and not list(cleanup.iterdir())
                assert cleanup != Path(old_pending['progress']['collection_directory'])
                assert same_state(resume.convert(old_pending,spec['device'],cleanup),new_pending)
                assert paper.sha(source/'pending.pt') == info['pending']['source_sha256']
                pending_ok = dict(episode=info['pending']['episode'], transitions=info['pending']['transitions'],
                    all_memory_optimizers_rng_exact=True, cleanup_isolated=True)
            # Idempotent re-import retains this branch's checkpoint, not its parent.
            before = paper.sha(dest/'latest.pt')
            with resume.source_locks(paper,spec):
                resume.fork_locked(paper,c,spec,variant,seed)
                blocked = False
                try:
                    with paper.lock(source/'execution.lock',blocking=False):
                        pass
                except BlockingIOError:
                    blocked = True
                assert blocked
            assert paper.sha(dest/'latest.pt') == before
            results.append(dict(variant=variant,seed=seed,episode=actual['progress']['next_episode'],
                source_sha256=info['source_latest_sha256'],destination_sha256=before,
                full_payload_exact_except_device_and_cleanup=True,pending=pending_ok,
                original_budget_and_source_unchanged=True,idempotent=True))
    # Verify resource adapter CPU routing despite aligned train()'s hardcoded
    # cuda arguments. Then exercise one bounded synthetic update/replay bridge.
    spec = resume.condition(variants[0],22)
    threads = 4
    adapter_args = argparse.Namespace(cpus=4,ram_gib=32,learner_threads=threads)
    resume.install(paper,spec,adapter_args)
    assert paper.configuration(c,variants[0],22)['learner_device'] == 'cpu'
    import server
    environment = server.preflight(paper.ROOT,c,'cuda')
    assert environment['device'] == 'cpu' and environment['gpu'] is None
    cfg = dict(paper.configuration(c,variants[0],22), learner_device='cpu')
    with tempfile.TemporaryDirectory(prefix='tgppo-device-bridge-') as temporary:
        result = bridge(cfg,'cpu',Path(temporary))
        assert result['passed'] and torch.get_num_threads() == 1
    from project.agent.agent import Agent
    from types import SimpleNamespace
    broken = Agent.__new__(Agent)
    broken.memory = SimpleNamespace(active=True)
    try:
        broken.learn()
        raise AssertionError('An unfinished episode must be rejected')
    except RuntimeError:
        assert torch.get_num_threads() == 1
    output = resume.ROOT/'docs/methods/tgppo_device_resume_20261116'/('CHECKS_'+args.experiment+'.json')
    paper.write(output,dict(passed=True,experiment=args.experiment,conditions=results,
        synthetic_bridge=result, learner_threads=threads,source_solves=0,new_real_episodes=0,
        cuda_executed=False,gpu_bridge_requires_user_custom_job=True,
        duplicate_source_lock_rejected=True,cpu_preflight_cuda_argument_overridden=True,
        thread_restore_on_success_and_failure=True,environment=environment))
    print(json.dumps(dict(passed=True,experiment=args.experiment,conditions=len(results),report=str(output))))


if __name__ == '__main__':
    main()
