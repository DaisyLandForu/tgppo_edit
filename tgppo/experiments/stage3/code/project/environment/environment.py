"""One finite-horizon SCIP task. Collection and Actor-only evaluation are separate."""
import logging
import time
import traceback
import pyscipopt as scip
from project.utils.settings import init_params
from project.utils.protocol import baseline_identity, validate_baseline, solver_stats
from project.brancher import Brancher
from project.measurement import BoundTrace, finish_measurement


class Environment:
    def __init__(self, device, agent, state_dims, scip_limits, scip_params, scip_seed,
                 reward_func=None, logger=None, mode='collect', deterministic=False):
        if mode not in ('collect', 'eval', 'native'):
            raise ValueError('Unknown environment mode')
        if mode == 'collect' and (agent is None or reward_func is None):
            raise ValueError('Collection requires an Agent and reward')
        if mode == 'eval' and agent is None:
            raise ValueError('Evaluation requires an Actor policy')
        if scip_limits['time_limit'] < 1:
            raise ValueError('H3 clamps subsecond limits; use time_limit >= 1 consistently')
        self.device, self.agent, self.state_dims = device, agent, state_dims
        self.scip_limits, self.scip_params, self.scip_seed = scip_limits, scip_params, scip_seed
        self.reward_func, self.mode = reward_func, mode
        self.logger = logger or logging.getLogger(__name__)
        self.deterministic = deterministic
        self.model = self.brancher = None
        self.episode_count = 0

    def reset(self, instance, cutoff=None, baseline=None, cutoff_provenance=None,
              episode_id=None):
        if self.model is not None:
            raise RuntimeError('Previous task not finalized')
        start = time.perf_counter()
        self.identity = baseline_identity(instance, self.scip_seed, self.scip_limits,
                                          self.scip_params, cutoff, cutoff_provenance)
        self.cutoff_verified = bool(cutoff_provenance and cutoff_provenance.get('verified'))
        self.cutoff_applied = bool(self.scip_params['cutoff'] and cutoff is not None)
        self.episode_id = episode_id or str(instance) + ':' + str(self.episode_count)
        if self.mode == 'collect':
            base = validate_baseline(baseline, self.identity)
            self.reward_func.reset(baseline_nodes=base['nnodes'], baseline_gap=base['gap'],
                baseline_pdi=base['primalDualIntegral'], solver_status=base['status'],
                time_limit=self.scip_limits['time_limit'], logger=self.logger)
        try:
            self.model = scip.Model()
            init_params(self.model, self.scip_limits, self.scip_params)
            self.model.setBoolParam('randomization/permutevars', True)
            self.model.setIntParam('randomization/permutationseed', int(self.scip_seed))
            self.model.readProblem(str(instance))
            if self.cutoff_applied:
                self.model.setObjlimit(float(cutoff))
            self.load_wall_seconds = time.perf_counter() - start
        except Exception:
            if self.model is not None:
                self.model.freeProb()
                self.model = None
            raise
        return True

    def run_episode(self):
        if self.model is None:
            raise RuntimeError('reset must precede run_episode')
        self.episode_count += 1
        info = {}
        start = time.perf_counter()
        try:
            trace = BoundTrace()
            self.model.includeEventhdlr(trace, 'Stage3BoundTrace', 'Read-only common metric trace')
            if self.mode == 'collect':
                self.agent.memory.begin_episode(self.episode_id, self.agent.policy_version)
            if self.mode != 'native':
                self.brancher = Brancher(self.model, self.state_dims, self.agent, self.mode,
                                         self.reward_func, self.deterministic)
                self.model.includeBranchrule(self.brancher, 'TreeGatePPO_Brancher',
                    'Stage2 original features', priority=999999, maxdepth=-1, maxbounddist=1)
            self.model.optimize()
            info = solver_stats(self.model, self.cutoff_applied, self.cutoff_verified)
            info.update(optimize_wall_seconds=time.perf_counter() - start,
                        load_wall_seconds=self.load_wall_seconds, episode_id=self.episode_id,
                        time_limit_seconds=self.scip_limits['time_limit'], mode=self.mode)
            info['scip_time_overshoot_seconds'] = max(
                0., info['scip_solve_time'] - self.scip_limits['time_limit'])
            if self.brancher and self.brancher.error:
                info.update(valid=False, task_terminal=False, error=self.brancher.error)
            if info['valid'] and self.brancher:
                finalize_start = time.perf_counter()
                self.brancher.finalize_episode(info['status'])
                info['finalize_wall_seconds'] = time.perf_counter() - finalize_start
            elif self.mode == 'collect':
                self.agent.memory.abort_episode()
            if self.brancher:
                info.update(self.brancher.stats())
            if info['valid']:
                info.update(finish_measurement(self.model, info, self.scip_limits['time_limit'], trace))
            info['environment_wall_seconds_before_free'] = (
                time.perf_counter() - start + self.load_wall_seconds)
        except Exception:
            if self.mode == 'collect':
                self.agent.memory.abort_episode()
            info.update(valid=False, task_terminal=False, episode_ended=True,
                        status='error', error=traceback.format_exc(), episode_id=self.episode_id,
                        mode=self.mode, optimize_wall_seconds=time.perf_counter() - start)
        finally:
            try:
                self.model.freeProb()
            finally:
                self.model = self.brancher = None
        return info['valid'], info, info.get('total_reward', 0.)
