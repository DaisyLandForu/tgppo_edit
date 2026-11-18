"""Bounded CPU preparation checks; no real rollout, solver optimization or CUDA."""
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import unittest

START = time.monotonic()
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('RESEARCH_ROOT', str(PACKAGE.parents[2])))
DOCS = ROOT/'docs/methods/tgppo_paper_trainmode_20261108'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Local preparation must explicitly hide CUDA')
    DOCS.mkdir(parents=True, exist_ok=True)
    ledger_path = DOCS/'PREPARATION_BUDGET.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[]}
    spent = sum(a.get('elapsed_seconds',a['allocated_seconds']) for a in ledger['attempts'])
    remaining = 600.-spent
    if remaining <= 1:
        raise RuntimeError('600-second cumulative local preparation budget exhausted')
    attempt = dict(allocated_seconds=remaining, started_epoch=time.time())
    ledger['attempts'].append(attempt)
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    def stop(signum, frame):
        raise TimeoutError('Local preparation deadline reached')
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.1,remaining-(time.monotonic()-START)))
    passed = False
    try:
        import torch
        torch.set_num_threads(1)
        log = DOCS/('cpu-checks-attempt-{:02d}.log'.format(len(ledger['attempts'])))
        with log.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            suite = unittest.defaultTestLoader.discover(str(PACKAGE/'tests'))
            result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        passed = result.wasSuccessful() and not torch.cuda.is_initialized()
        checks = dict(passed=passed, tests_run=result.testsRun, failures=len(result.failures),
            errors=len(result.errors), cuda_initialized=torch.cuda.is_initialized(),
            real_training_episodes=0, real_scip_solves=0, log=str(log),
            scope='single-thread CPU synthetic updates, GAE, dropout modes, provenance and exact resume')
        (DOCS/'CPU_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps(checks,indent=2))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        attempt.update(elapsed_seconds=time.monotonic()-START,passed=passed)
        ledger['charged_seconds']=sum(a.get('elapsed_seconds',a['allocated_seconds']) for a in ledger['attempts'])
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
