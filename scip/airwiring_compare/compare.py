#!/usr/bin/env python3
"""One model exported from sssp_0923.cpp; three unlimited-time SCIP jobs.

Preparation/summary use the standard library. Solves use the isolated SCIP-8
feature binding. No training, cutoff, subprocess timeout, or solver watchdog.
"""
import argparse
import contextlib
import csv
import datetime as dt
import difflib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCIP_DIR = HERE.parent
RESEARCH = ROOT / 'research'
MAMBA = RESEARCH / 'mamba-branching'
TGPPO_CODE = RESEARCH / 'tgppo/experiments/server-20261022/code'
MAMBA_RUN = MAMBA / 'runs/milp-s-seed0-20261027'
TGPPO_EXPORT = RESEARCH / 'tgppo/runs/server6-20261022/training/B0-s11/actor_best.pt'
TGPPO_SHA = '8b605db5be6bc4a70d8f8ab7c7fa84552f8542b0676a4365a19041a101875397'
MAMBA_SHA = 'c1d1761e1085f8e7c92947a40d765d0d05e5a471bca618aa3fdc38e725ac2a74'
DIMS = dict(var_dim=25, node_dim=8, mip_dim=53)
METHODS = ('sssp', 'mamba', 'tgppo')


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def checked_replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source changed; expected one occurrence of: ' + old[:100])
    return text.replace(old, new, 1)


def exporter_source(source):
    """Keep the original graph reduction and model construction verbatim.

    Remove only the unused HiGHS adapter, XLS output, and optimization path.
    Reuse the original SCIP variable/row import, then write an original CIP.
    """
    prefix = source[:source.index('struct SolverResult {')]
    prefix = checked_replace(prefix, '#include <Highs.h>',
                             'constexpr double kHighsInf = std::numeric_limits<double>::infinity();')
    prefix = checked_replace(prefix, '#include <xlslib.h>\n', '')
    begin = prefix.index('    HighsStatus passToHighs(')
    end = prefix.index('    size_t numRows()', begin)
    prefix = prefix[:begin] + prefix[end:]
    check_begin = source.index('static void checkScip(')
    check_end = source.index('static std::string scipStatusToString', check_begin)
    checker = source[check_begin:check_end]
    begin = source.index('SolverResult solveWithScip(')
    end = source.index('        // 与原 HiGHS 设置对应', begin)
    exporter = source[begin:end]
    exporter = checked_replace(exporter,
        'SolverResult solveWithScip(const DirectModelBuilder& modelBuilder)',
        'void exportModel(const DirectModelBuilder& modelBuilder, const std::string& output)')
    exporter += '''        checkScip(SCIPwriteOrigProblem(scip, output.c_str(), "cip", FALSE), "write CIP");
        for (auto& var : scipVars) checkScip(SCIPreleaseVar(scip, &var), "release var");
        checkScip(SCIPfree(&scip), "free SCIP");
    } catch (...) {
        if (scip != nullptr) {
            for (auto& var : scipVars) if (var != nullptr) SCIPreleaseVar(scip, &var);
            SCIPfree(&scip);
        }
        throw;
    }
}
'''
    main = source[source.index('int main(int argc, char* argv[]) {'):]
    main = main[:main.index('    SolverResult solveResult;')]
    # This adapter receives a validated normalized 10-column CSV.
    main = checked_replace(main, 'if (c.size() >= 6) pairs.push_back(',
                           'if (c.size() >= 10) pairs.push_back(')
    main += '    exportModel(modelBuilder, outputFile);\n    return 0;\n}\n'
    return prefix + checker + exporter + main


def normalize_inputs(edges, pairs, out):
    with edges.open(newline='') as stream:
        edge_rows = list(csv.reader(stream))[1:]
    nodes = set()
    for row in edge_rows:
        if len(row) != 5 or not row[2] or not row[3]:
            raise ValueError('Invalid edge row: ' + repr(row))
        if not math.isfinite(float(row[4])) or float(row[4]) < 0:
            raise ValueError('Invalid edge cost: ' + repr(row))
        nodes.update(row[2:4])
    with pairs.open(newline='') as stream:
        pair_rows = list(csv.reader(stream))[1:]
    normalized = []
    layouts = set()
    for row in pair_rows:
        if len(row) in (6, 7):
            layouts.add('compact: u=3,v=4,density=5')
            if len(row) == 7 and row[6].strip():
                raise ValueError('Expected empty trailing column: ' + repr(row))
            normalized_row = row[:4] + ['', row[4], '', '', '', row[5]]
        elif len(row) >= 10:
            layouts.add('legacy: u=3,v=5,density=9')
            normalized_row = row[:10]
        else:
            raise ValueError('Unrecognized pair layout: ' + repr(row))
        if normalized_row[3] not in nodes or normalized_row[5] not in nodes:
            raise ValueError('Pair endpoint absent from edges: ' + repr(row))
        density = float(normalized_row[9])
        if not math.isfinite(density) or density <= 0:
            raise ValueError('Invalid density: ' + repr(row))
        normalized.append(normalized_row)
    if not edge_rows or not normalized:
        raise ValueError('Empty instance')
    # The original C++ edge reader needs no adaptation.
    shutil.copyfile(edges, out / 'edges.csv')
    with (out / 'pairs.normalized.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['id', 'harness', 'circuit', 'u', 'unused', 'v',
                         'unused', 'unused', 'unused', 'density'])
        writer.writerows(normalized)
    return dict(edges=len(edge_rows), nodes=len(nodes), pairs=len(normalized),
                pair_layouts=sorted(layouts))


def prepare(args):
    prefix = Path(os.environ.get('AIRWIRING_SCIP_PREFIX', '/usr/local/scip-8.0.4'))
    compiler = shutil.which(os.environ.get('CXX', 'g++'))
    if not compiler or not (prefix / 'include/scip/scip.h').exists():
        raise RuntimeError('Need g++ and SCIP 8.0.4 development headers: ' + str(prefix))
    source_path = SCIP_DIR / 'sssp_0923.cpp'
    generated = exporter_source(source_path.read_text())
    identity = dict(source_sha256=sha(source_path), edges_sha256=sha(args.edges),
                    pairs_sha256=sha(args.pairs), k=args.k,
                    exporter_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                    libscip_sha256=sha(prefix / 'lib/libscip.so'),
                    compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                    flags=['-O2', '-std=c++17'])
    directory = args.artifacts / ('model-' + digest(identity)[:20])
    directory.mkdir(parents=True, exist_ok=True)
    print('Preparing shared model (locked):', directory, flush=True)
    with lock(directory / '.prepare.lock'):
        manifest_path = directory / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['identity'] != identity:
                raise RuntimeError('Model cache identity mismatch')
            for name, checksum in manifest['files'].items():
                if sha(directory / name) != checksum:
                    raise RuntimeError('Model cache corrupted: ' + name)
            return directory, manifest
        counts = normalize_inputs(args.edges, args.pairs, directory)
        shutil.copyfile(source_path, directory / 'sssp_0923.original.cpp')
        generated_path = directory / 'export_model.cpp'
        generated_path.write_text(generated)
        (directory / 'export_model.diff').write_text(''.join(difflib.unified_diff(
            source_path.read_text().splitlines(True), generated.splitlines(True),
            fromfile='sssp_0923.original.cpp', tofile='export_model.cpp')))
        executable = directory / 'export_model'
        command = [compiler, '-O2', '-std=c++17', str(generated_path),
                   '-I' + str(prefix / 'include'), '-L' + str(prefix / 'lib'),
                   '-Wl,-rpath,' + str(prefix / 'lib'), '-lscip', '-o', str(executable)]
        with (directory / 'compile.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        temporary = directory / 'model.pending.cip'
        with (directory / 'export.log').open('w') as log:
            subprocess.run([str(executable), str(directory / 'edges.csv'),
                            str(directory / 'pairs.normalized.csv'), str(args.k),
                            str(temporary), '--solver', 'scip'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        os.replace(temporary, directory / 'model.cip')
        files = ['sssp_0923.original.cpp', 'export_model.cpp', 'edges.csv',
                 'pairs.normalized.csv', 'model.cip']
        manifest = dict(created_utc=stamp(), identity=identity, input_counts=counts,
                        files={name: sha(directory / name) for name in files},
                        objective_scale=1e6, optimization_performed=False,
                        adaptations=['normalize compact pair columns to legacy layout',
                                     'export original SCIP model before solving',
                                     'omit unused HiGHS adapter and XLS reporting'])
        write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        return directory, manifest


def solver_params(seed):
    return {'numerics/feastol': 1e-6, 'numerics/dualfeastol': 1e-6,
            'limits/time': 1e20, 'limits/softtime': -1, 'limits/nodes': -1, 'limits/totalnodes': -1,
            'limits/stallnodes': -1, 'limits/solutions': -1, 'limits/bestsol': -1,
            'limits/restarts': -1, 'limits/memory': 8796093022207.0,
            'limits/gap': 0.0, 'limits/absgap': 0.0,
            'parallel/maxnthreads': 1, 'lp/threads': 1,
            'randomization/randomseedshift': seed,
            'randomization/permutationseed': seed,
            'randomization/lpseed': seed,
            'display/verblevel': 4, 'display/freq': 100,
            'timing/clocktype': 2}


def finite_bound(model, value):
    value = float(value)
    return value if math.isfinite(value) and abs(value) < model.infinity() * .5 else None


def stats(model):
    primal = finite_bound(model, model.getPrimalbound())
    dual = finite_bound(model, model.getDualbound())
    return dict(solver_status=str(model.getStatus()), scip_seconds=model.getSolvingTime(),
                nodes=model.getNNodes(), nodes_left=model.getNNodesLeft(),
                solutions=model.getNSols(), gap=finite_bound(model, model.getGap()),
                primal_bound=primal, dual_bound=dual,
                primal_bound_real=None if primal is None else primal / 1e6,
                dual_bound_real=None if dual is None else dual / 1e6,
                primal_dual_integral=model.getPrimalDualIntegral())


def hardware():
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor())
    return dict(host=platform.node(), cpu_model=cpu, python=sys.version,
                executable=sys.executable, affinity=sorted(os.sched_getaffinity(0)),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'))


def load_policy(method, model, device):
    """Use frozen original networks/branchers, without creating a Critic."""
    if method == 'sssp':
        return None, {}, contextlib.nullcontext()
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use a CPU Custom job with --device cpu')
    sys.path.insert(0, str(TGPPO_CODE))
    if method == 'tgppo':
        if sha(TGPPO_EXPORT) != TGPPO_SHA:
            raise RuntimeError('Frozen TGPPO B0 checkpoint SHA mismatch')
        from project.policy.actor import Actor
        from project.brancher.brancher import Brancher
        from project.agent.agent import ActorPolicy
        payload = torch.load(TGPPO_EXPORT, map_location='cpu')
        if payload['schema_version'] != 'actor-only-v1' or digest(payload['config']) != payload['config_sha256']:
            raise RuntimeError('TGPPO actor schema/config mismatch')
        config = payload['config']
        if config['state_dims'] != DIMS or config['server_group'] != 'B0':
            raise RuntimeError('Expected original B0 with 25/8/53 features')
        actor = Actor(**DIMS, **{key: config[key] for key in ('hidden_dim', 'num_layers', 'num_heads', 'dropout')})
        actor.load_state_dict(payload['actor'])
        policy = ActorPolicy(actor, device=device)

        class PortedTGPPO(Brancher):
            def branchexeclp(self, allowaddcons):
                try:
                    return super().branchexeclp(allowaddcons)
                except Exception:
                    # The legacy exception handler calls Model.interrupt();
                    # PySCIPOpt 4 spells this interruptSolve().
                    self.error = self.error or traceback.format_exc()
                    self.model.interruptSolve()
                    return {'result': scip.SCIP_RESULT.DIDNOTRUN}

        import pyscipopt as scip
        brancher = PortedTGPPO(model, DIMS, policy, 'eval', deterministic=True)
        metadata = dict(checkpoint=str(TGPPO_EXPORT), checkpoint_sha256=TGPPO_SHA,
                        training_seed=11, architecture='original TGPPO B0',
                        training_scip_version='6.0.1', deployment_scip_version='8.0.4',
                        torch_training_version='1.13.1', torch_deployment_version=torch.__version__,
                        config=payload['config'], deterministic_argmax=True)
    else:
        sys.path.insert(0, str(MAMBA / 'code'))
        from network import TreeGateNet, GroupedMamba
        from env import MambaEvalBrancher
        selected = json.loads((MAMBA_RUN / 'training/selected.json').read_text())
        path = MAMBA_RUN / selected['checkpoint']
        if not selected['frozen'] or selected['checkpoint_sha256'] != MAMBA_SHA or sha(path) != MAMBA_SHA:
            raise RuntimeError('Frozen Mamba checkpoint SHA mismatch')
        checkpoint = torch.load(path, map_location='cpu')
        if (checkpoint['config_sha256'] != selected['config_sha256'] or
                checkpoint['dataset_sha256'] != selected['dataset_sha256']):
            raise RuntimeError('Mamba checkpoint provenance mismatch')
        embedding, sequence = TreeGateNet(infimum=8).to(device), GroupedMamba(8, 100).to(device)
        embedding.load_state_dict(checkpoint['embedding'])
        sequence.load_state_dict(checkpoint['sequence'])
        if device == 'cpu':
            from mamba_cpu import install_cpu_backend
            install_cpu_backend(sequence)
        embedding.eval()
        sequence.eval()

        def finite_inputs(_module, inputs):
            if any(not torch.isfinite(value).all() for value in inputs if isinstance(value, torch.Tensor)):
                raise ValueError('Nonfinite Mamba feature input')

        def finite_output(_module, _inputs, output):
            values = output if isinstance(output, tuple) else (output,)
            if any(not torch.isfinite(value).all() for value in values if isinstance(value, torch.Tensor)):
                raise ValueError('Nonfinite Mamba network output')

        embedding.register_forward_pre_hook(finite_inputs)
        embedding.register_forward_hook(finite_output)
        sequence.register_forward_hook(finite_output)

        class StrictMamba(MambaEvalBrancher):
            error = None
            callback_seconds = 0.0

            def branchexeclp(self, allowaddcons):
                started = time.perf_counter()
                try:
                    if not self.model.getLPBranchCands()[0]:
                        return {'result': scip.SCIP_RESULT.DIDNOTRUN}
                    return super().branchexeclp(allowaddcons)
                except Exception:
                    self.error = traceback.format_exc()
                    self.model.interruptSolve()
                    return {'result': scip.SCIP_RESULT.DIDNOTRUN}
                finally:
                    self.callback_seconds += time.perf_counter() - started

        import pyscipopt as scip
        brancher = StrictMamba(model, device, embedding, sequence, DIMS, False, 49, True, 1)
        metadata = dict(checkpoint=str(path), checkpoint_sha256=MAMBA_SHA, training_seed=0,
                        selection=selected, history_steps=24, max_seq_len=49,
                        relpscost_initial_branches=1, deterministic_argmax=True,
                        inference_backend='pytorch-cpu-mamba1' if device == 'cpu' else 'cuda-fused-mamba1')
        if device == 'cpu':
            metadata['cpu_backend_sha256'] = sha(HERE / 'mamba_cpu.py')
    model.includeBranchrule(brancher, 'AirwiringPolicy', method, 999999, -1, 1)
    metadata.update(torch_version=torch.__version__, device=device,
                    gpu_model=torch.cuda.get_device_name(device) if device.startswith('cuda') else None)
    source_root = TGPPO_CODE / 'project' if method == 'tgppo' else MAMBA / 'code'
    metadata['source_sha256'] = {str(path.relative_to(RESEARCH)): sha(path)
                                for path in sorted(source_root.rglob('*.py'))}
    return brancher, metadata, torch.inference_mode()


def monitor_class(scip):
    class Progress(scip.Eventhdlr):
        def __init__(self, directory):
            self.directory = directory
            self.last = -math.inf
            self.first_solution_seconds = None
            self.error = None
            self.seconds = 0.0

        def eventinit(self):
            self.model.catchEvent(scip.SCIP_EVENTTYPE.BESTSOLFOUND | scip.SCIP_EVENTTYPE.NODESOLVED, self)

        def eventexit(self):
            self.model.dropEvent(scip.SCIP_EVENTTYPE.BESTSOLFOUND | scip.SCIP_EVENTTYPE.NODESOLVED, self)

        def snapshot(self, reason, final=False):
            started = time.perf_counter()
            current = stats(self.model)
            current.update(reason=reason, observed_utc=stamp())
            if self.first_solution_seconds is None and current['solutions']:
                self.first_solution_seconds = current['scip_seconds']
            with (self.directory / 'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(current, allow_nan=False) + '\n')
            write_json(self.directory / 'progress.json', current)
            if current['solutions'] and (reason == 'best_solution' or final):
                temporary = self.directory / 'incumbent.pending.sol'
                self.model.writeBestSol(str(temporary), write_zeros=False)
                os.replace(temporary, self.directory / 'incumbent.sol')
            self.last = current['scip_seconds']
            self.seconds += time.perf_counter() - started
            return current

        def eventexec(self, event):
            try:
                best = bool(event.getType() & scip.SCIP_EVENTTYPE.BESTSOLFOUND)
                if best or self.model.getSolvingTime() - self.last >= 60:
                    self.snapshot('best_solution' if best else 'node_progress')
            except Exception:
                self.error = traceback.format_exc()
                self.model.interruptSolve()
    return Progress


def solve(args, method, directory, manifest, instance=None):
    setup_started = time.monotonic()
    import pyscipopt as scip
    import pyscipopt.scip as binding
    if abs(scip.Model().version() - 8.04) > .0001:
        raise RuntimeError('This comparison requires the shared SCIP 8.0.4 feature binding')
    instance = instance or directory / 'model.cip'
    model = scip.Model()
    libraries = {Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/libscip.so.' in line and line.split()[-1].startswith('/')}
    if len(libraries) != 1:
        raise RuntimeError('Expected exactly one loaded libscip: ' + repr(libraries))
    library = libraries.pop()
    library_sha = sha(library)
    if 'libscip_sha256' in manifest['identity'] and library_sha != manifest['identity']['libscip_sha256']:
        raise RuntimeError('Exporter and solve use different libscip binaries')
    model.setLogfile(str(args.attempt / 'scip.log'))
    model.readProblem(str(instance))
    parameters = solver_params(args.seed)
    if getattr(args, 'smoke', False):
        # Only synthetic validation: force branching to exercise callbacks.
        model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        model.setSeparating(scip.SCIP_PARAMSETTING.OFF)
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
        parameters['limits/nodes'] = 30
    for key, value in parameters.items():
        model.setParam(key, value)
    resolved = {key: model.getParam(key) for key in parameters}
    if not getattr(args, 'smoke', False):
        if resolved['limits/time'] < 1e19:
            raise RuntimeError('Unexpected finite time limit')
        if any(resolved[k] != 0 for k in ('limits/gap', 'limits/absgap')):
            raise RuntimeError('Nonzero gap limit')
    model.writeParams(str(args.attempt / 'settings.set'), comments=True, onlychanged=False)
    brancher, policy, inference = load_policy(method, model, args.device)
    monitor = monitor_class(scip)(args.attempt)
    model.includeEventhdlr(monitor, 'AirwiringProgress', 'save bounds and incumbents')
    result = dict(status='running', method=method, started_utc=stamp(), hardware=hardware(),
                  model_sha256=sha(instance), input_identity=manifest['identity'],
                  model_directory=str(directory), solver_seed=args.seed,
                  scip_version=model.version(), binding_sha256=sha(Path(binding.__file__)),
                  libscip_path=str(library), libscip_sha256=library_sha,
                  parameters=resolved, policy=policy, cutoff=None, primal_heuristics=True,
                  original_variables=model.getNVars(), original_constraints=model.getNConss(),
                  smoke=bool(getattr(args, 'smoke', False)))
    result['setup_wall_seconds'] = time.monotonic() - setup_started
    write_json(args.attempt / 'result.json', result)
    previous = {}
    interruptions = []

    def stop(signum, _frame):
        interruptions.append(signum)
        model.interruptSolve()

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, stop)
    started = time.monotonic()
    try:
        write_json(args.attempt / 'progress.json', dict(reason='start', solver_status='not_started',
                   observed_utc=stamp(), scip_seconds=0.0))
        with inference:
            model.optimize()
        optimize_elapsed = time.monotonic() - started
        result.update(monitor.snapshot('final', final=True))
        result.update(optimize_wall_seconds=optimize_elapsed,
                      first_solution_seconds=monitor.first_solution_seconds,
                      monitoring_seconds=monitor.seconds,
                      finished_utc=stamp(), interruptions=interruptions)
        if brancher is not None:
            result['branch_calls'] = getattr(brancher, 'callback_count', getattr(brancher, 'branchexec_count', 0))
            result['branches'] = brancher.branch_count
            result['branch_times'] = getattr(brancher, 'times', {'callback': getattr(brancher, 'callback_seconds', 0.)})
            result['policy_error'] = getattr(brancher, 'error', None)
        if monitor.error or result.get('policy_error'):
            raise RuntimeError(monitor.error or result['policy_error'])
        # PySCIPOpt 4.3 returns a Solution wrapper even for a NULL best-sol
        # pointer. Passing that wrapper to checkSol can segfault.
        has_solution = model.getNSols() > 0
        result['solution_checked'] = bool(has_solution and
            model.checkSol(model.getBestSol(), printreason=True, completely=True,
                           checkbounds=True, checkintegrality=True, checklprows=True, original=True))
        if has_solution and not result['solution_checked']:
            raise RuntimeError('Final incumbent failed original-model feasibility validation')
        result['proven_optimal'] = result['solver_status'] == 'optimal' and result['solution_checked']
        result['status'] = ('complete' if result['solver_status'] in ('optimal', 'infeasible', 'unbounded', 'inforunbd')
                            else 'interrupted')
        result['solve_pipeline_wall_seconds'] = time.monotonic() - setup_started
        model.printStatistics()
        write_json(args.attempt / 'result.json', result)
        return result
    except BaseException:
        result.update(status='failed', error=traceback.format_exc(), finished_utc=stamp(),
                      optimize_wall_seconds=time.monotonic() - started)
        write_json(args.attempt / 'result.json', result)
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        model.freeProb()


def run(args):
    directory, manifest = prepare(args)
    method_dir = args.results / f'{args.command}-s{args.seed}'
    execution_identity = dict(model_sha256=manifest['files']['model.cip'],
                              driver_sha256=sha(Path(__file__)), parameters=solver_params(args.seed),
                              method=args.command,
                              device=None if args.command == 'sssp' else args.device,
                              checkpoint_sha256={'sssp': None, 'mamba': MAMBA_SHA, 'tgppo': TGPPO_SHA}[args.command])
    if args.command == 'mamba' and args.device == 'cpu':
        execution_identity['cpu_backend_sha256'] = sha(HERE / 'mamba_cpu.py')
    # Concurrent duplicate submissions fail clearly; interrupted runs retain every attempt.
    with lock(method_dir / '.run.lock', blocking=False):
        done = method_dir / 'complete.json'
        if done.exists():
            previous = json.loads(done.read_text())
            if previous['input_identity'] != manifest['identity']:
                raise RuntimeError('Result directory belongs to a different model; use --results')
            if previous.get('execution_identity') != execution_identity:
                raise RuntimeError('Completed run used different code/settings/device; use --results')
            if sha(Path(previous['attempt']) / 'result.json') != previous['result_sha256']:
                raise RuntimeError('Committed result checksum mismatch')
            print('Already complete:', previous['attempt'], flush=True)
            return 0
        args.attempt = method_dir / ('attempt-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        args.attempt.mkdir(parents=True)
        write_json(args.attempt / 'launch.json', dict(started_utc=stamp(), hardware=hardware(),
                   command=sys.argv, source_sha256=sha(Path(__file__)), pid=os.getpid()))
        print('RESULT DIRECTORY:', args.attempt, flush=True)
        try:
            if args.command != 'sssp':
                probe = argparse.Namespace(**vars(args))
                probe.results = args.attempt / 'preflight'
                probe.method = args.command
                smoke(probe)
            result = solve(args, args.command, directory, manifest)
        except BaseException:
            if not (args.attempt / 'result.json').exists():
                write_json(args.attempt / 'result.json', dict(status='failed', method=args.command,
                           error=traceback.format_exc(), finished_utc=stamp(),
                           model_sha256=manifest['files']['model.cip']))
            raise
        if result['status'] == 'complete':
            write_json(done, dict(attempt=str(args.attempt), input_identity=manifest['identity'],
                       execution_identity=execution_identity,
                       result_sha256=sha(args.attempt / 'result.json')))
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result['status'] == 'complete' else 3


def summarize(args):
    rows = []
    identities = []
    columns = ['method', 'solver_seed', 'status', 'solver_status', 'proven_optimal',
               'primal_bound_real', 'dual_bound_real', 'gap', 'nodes', 'solutions',
               'scip_seconds', 'optimize_wall_seconds', 'first_solution_seconds',
               'primal_dual_integral', 'branches', 'model_sha256', 'cpu_model', 'gpu_model',
               'device', 'inference_backend', 'attempt']
    for method in METHODS:
        method_dir = args.results / f'{method}-s{args.seed}'
        attempts = sorted(method_dir.glob('attempt-*'))
        committed = method_dir / 'complete.json'
        if committed.exists():
            chosen = Path(json.loads(committed.read_text())['attempt'])
        elif attempts:
            chosen = attempts[-1]
        else:
            rows.append(dict(method=method, solver_seed=args.seed, status='not_started'))
            continue
        path = chosen / 'result.json'
        value = json.loads(path.read_text()) if path.exists() else {'status': 'starting'}
        if value.get('parameters'):
            identities.append({key: value.get(key) for key in
                               ('model_sha256', 'parameters', 'scip_version',
                                'binding_sha256', 'libscip_sha256')})
        if value.get('status') == 'running' and (chosen / 'progress.json').exists():
            value.update(json.loads((chosen / 'progress.json').read_text()))
        row = {key: value.get(key) for key in columns}
        row.update(method=method, solver_seed=args.seed, attempt=str(chosen),
                   cpu_model=value.get('hardware', {}).get('cpu_model'),
                   gpu_model=value.get('policy', {}).get('gpu_model'),
                   device=value.get('policy', {}).get('device'),
                   inference_backend=value.get('policy', {}).get('inference_backend'))
        if row['status'] == 'running':
            # Remote PID/liveness is deliberately not inferred from a persisted flag.
            row['status'] = 'unfinished (inspect job/log)'
        rows.append(row)
    observed = [r for r in rows if r.get('model_sha256')]
    compatible = len({r['model_sha256'] for r in observed}) <= 1
    same_cpu = len({r.get('cpu_model') for r in observed if r.get('cpu_model')}) <= 1
    same_settings = len({digest(value) for value in identities}) <= 1
    same_gpu = len({r.get('gpu_model') for r in observed if r.get('gpu_model')}) <= 1
    report = dict(updated_utc=stamp(), same_model=compatible, same_solver_settings=same_settings,
                  same_cpu_model=same_cpu, same_gpu_model=same_gpu, rows=rows,
                  note='PDI is over each actual solve duration; compare at common elapsed horizons separately. '
                       'Unfinished records do not establish live jobs. No cross-method incumbent sharing.')
    args.results.mkdir(parents=True, exist_ok=True)
    write_json(args.results / f'summary-s{args.seed}.json', report)
    with (args.results / f'summary-s{args.seed}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print('method | status | best objective (real) | lower bound (real) | gap | nodes | SCIP seconds')
    for row in rows:
        print(' | '.join(str(row.get(k, '')) for k in ('method', 'solver_status', 'primal_bound_real',
              'dual_bound_real', 'gap', 'nodes', 'scip_seconds')) + '  [' + str(row['status']) + ']')
    if not compatible:
        raise RuntimeError('Different models found: comparison invalid')
    if not same_settings:
        raise RuntimeError('Different solver settings/binding/library found: comparison invalid')
    if not same_cpu:
        print('CPU models differ: absolute solve times are not a matched-hardware comparison.')
    return 0


def smoke(args):
    """Tiny synthetic MILP only; never optimize the aviation instance locally."""
    import pyscipopt as scip
    import random
    args.attempt = args.results / ('smoke-' + uuid.uuid4().hex[:8])
    args.attempt.mkdir(parents=True)
    model = scip.Model()
    rng = random.Random(11)
    variables = [model.addVar(vtype='B', name=f'x{i}', obj=-rng.randrange(1, 100)) for i in range(40)]
    for j in range(5):
        weights = [rng.randrange(1, 60) for _ in variables]
        model.addCons(scip.quicksum(w*x for w,x in zip(weights, variables)) <= sum(weights)*.37)
    instance = args.attempt / 'tiny.cip'
    model.writeProblem(str(instance))
    model.freeProb()
    args.smoke = True
    manifest = {'identity': {'smoke': True}}
    result = solve(args, args.method, args.attempt, manifest, instance)
    if args.method != 'sssp' and result.get('branches', 0) < 2:
        raise RuntimeError('Smoke did not exercise neural branching')
    print('SMOKE PASSED:', args.attempt, 'branches=', result.get('branches'), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', *METHODS, 'summarize', 'smoke', 'help'])
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0, help='shared SCIP random seed')
    parser.add_argument('--edges', type=Path, default=SCIP_DIR / 'dataset/edges.csv')
    parser.add_argument('--pairs', type=Path, default=SCIP_DIR / 'dataset/pairs-4-246.csv')
    parser.add_argument('--artifacts', type=Path, default=HERE / 'artifacts')
    parser.add_argument('--results', type=Path, default=HERE / 'runs/aviation-k4-seed0')
    parser.add_argument('--device', default='cpu', choices=['cuda:0', 'cpu'],
                        help='CPU by default to support long CPU-bound solves; CUDA is opt-in')
    parser.add_argument('--method', choices=METHODS, default='sssp', help='synthetic smoke method only')
    args = parser.parse_args()
    if args.k < 1 or args.seed < 0:
        parser.error('k must be positive; seed must be nonnegative')
    for key in ('edges', 'pairs', 'artifacts', 'results'):
        setattr(args, key, getattr(args, key).resolve())
    if args.command == 'help':
        parser.print_help()
        return 0
    if args.command == 'prepare':
        prepare(args)
        return 0
    if args.command == 'summarize':
        return summarize(args)
    if args.command == 'smoke':
        return smoke(args)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
