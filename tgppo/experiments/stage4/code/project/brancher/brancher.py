"""Explicit collection/evaluation lifecycle around the unmodified feature getters."""
import time
import traceback
import pyscipopt as scip
import numpy as np
from project.memory.memory import state_arrays


class Brancher(scip.Branchrule):
    def __init__(self, model, state_dims, agent, mode, reward_func=None,
                 deterministic=False, search_recorder=None):
        super().__init__()
        self.model, self.state_dims, self.agent = model, state_dims, agent
        self.mode, self.reward_func = mode, reward_func
        self.deterministic = deterministic
        self.search_recorder = search_recorder
        self.prev = None
        self.finalized = False
        self.error = None
        self.branch_count = self.callback_count = self.didnotrun_count = 0
        self.episode_rewards = []
        self.times = {k: 0. for k in ('features', 'actor', 'critic', 'branch', 'callback')}

    def _store_previous(self, terminal):
        if self.prev is None:
            return
        reward = float(self.reward_func.compute(self.model, terminal))
        state, action, value, log_prob = self.prev
        self.agent.remember(**state, action=action, value=value, log_prob=log_prob,
                            reward=reward, done=terminal)
        self.episode_rewards.append(reward)
        self.prev = None

    def branchexeclp(self, allowaddcons):
        start = time.perf_counter()
        self.callback_count += 1
        try:
            begin = time.perf_counter()
            cands, _, states = self.model.getCandsState(
                self.state_dims['var_dim'], self.branch_count)
            if not cands:
                self.didnotrun_count += 1
                return {'result': scip.SCIP_RESULT.DIDNOTRUN}
            if states is None:
                raise ValueError('Candidate features missing')
            states = np.asarray(states, dtype=np.float32).reshape(
                len(cands), self.state_dims['var_dim'])
            state = state_arrays(states, self.model.getMIPState(self.state_dims['mip_dim']),
                                 self.model.getNodeState(self.state_dims['node_dim']),
                                 self.state_dims)
            self.times['features'] += time.perf_counter() - begin
            if self.search_recorder is not None:
                self.search_recorder.capture(self.model, state)
            if self.mode == 'collect':
                self._store_previous(False)
                action, value, log_prob = self.agent.choose_action(**state,
                    deterministic=self.deterministic)
            else:
                action, log_prob = self.agent.act(**state, deterministic=self.deterministic)
            self.times['actor'] += self.agent.last_actor_seconds
            self.times['critic'] += self.agent.last_critic_seconds
            if not 0 <= action < len(cands):
                raise ValueError('Policy action out of range; no action-0 fallback')
            begin = time.perf_counter()
            self.model.branchVar(cands[action])
            if self.search_recorder is not None:
                self.search_recorder.commit(action)
            self.times['branch'] += time.perf_counter() - begin
            self.branch_count += 1
            if self.mode == 'collect':
                self.prev = (state, action, value, log_prob)
            return {'result': scip.SCIP_RESULT.BRANCHED}
        except Exception:
            self.error = traceback.format_exc()
            self.didnotrun_count += 1
            self.model.interrupt()  # Available in the installed custom 1.4.9 binding.
            return {'result': scip.SCIP_RESULT.DIDNOTRUN}
        finally:
            self.times['callback'] += time.perf_counter() - start

    def finalize_episode(self, end_reason):
        if self.finalized:
            return
        if self.mode == 'collect':
            self._store_previous(True)
            self.agent.seal_episode(task_terminal=True, end_reason=end_reason)
        self.finalized = True

    def branchfree(self):
        # SCIP may invoke this after optimize/freeProb. Never inspect a freed model.
        self.prev = None

    def stats(self):
        return dict(branch_count=self.branch_count, callback_count=self.callback_count,
                    didnotrun_count=self.didnotrun_count, fallback_action_count=0,
                    native_fallback_possible=self.didnotrun_count > 0,
                    reward_count=len(self.episode_rewards),
                    total_reward=sum(self.episode_rewards), callback_seconds=self.times.copy())
