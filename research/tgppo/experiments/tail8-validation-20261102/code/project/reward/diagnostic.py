"""Observe the original H3 calls; never substitute its returned reward."""
import math
import time
import numpy as np
from .reward import RewardH3, _safe_tanh


class DiagnosticRewardH3(RewardH3):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.records = []
        self.diagnostic_seconds = 0.0

    def compute(self, model, done):
        started = time.perf_counter()
        seen = {}
        names = {'getNNodes', 'getSolvingTime', 'getGap', 'getPrimalDualIntegral',
                 'getPrimalbound', 'getDualbound', 'getStatus'}

        class Observe:
            def __getattr__(self, name):
                member = getattr(model, name)
                if name not in names:
                    return member
                def call(*args, **kwargs):
                    value = member(*args, **kwargs)
                    seen[name] = value
                    return value
                return call

        prev_gap, prev_pdi, prev_pb, prev_db = self.prev_gap, self.prev_pdi, self.prev_pb, self.prev_db
        reward = super().compute(Observe(), done)
        nodes, pdi = seen['getNNodes'], seen['getPrimalDualIntegral']
        gap = seen['getGap']
        if math.isinf(gap):
            gap = 1e6
        fraction = min(max(seen['getSolvingTime']/self.time_limit, 0.), 1.)
        row = dict(done=bool(done), nodes=int(nodes), time_fraction=fraction, gap=float(gap),
                   pdi=float(pdi), baseline_gap=self.baseline_gap, baseline_pdi=self.baseline_pdi,
                   reward=float(reward))
        if not done:
            progress_gap = _safe_tanh((prev_gap-gap)/max(abs(prev_gap),1e-9),s=self.scale) if prev_gap < float('inf') else 0.
            pb, db = seen['getPrimalbound'], seen['getDualbound']
            progress_pb = _safe_tanh((prev_pb-pb)/(abs(prev_pb)+1e-9),s=self.scale) if prev_pb is not None and pb < prev_pb else 0.
            progress_db = _safe_tanh((db-prev_db)/(abs(prev_db)+1e-9),s=self.scale) if prev_db is not None and db > prev_db else 0.
            progress_pdi = _safe_tanh((prev_pdi-pdi)/self.baseline_pdi,s=self.scale) if prev_pdi > 0 else 0.
            terms = dict(nodes=self.w_nodes*self._node_eff(nodes), pace=self.w_pace*self._pace(nodes,fraction),
                gap_level=self.w_gap*(-_safe_tanh(gap/(self.baseline_gap+1e-9),s=.5)),
                pdi_level=self.w_pdi*_safe_tanh((self.baseline_pdi-pdi)/self.baseline_pdi),
                progress=self.w_progress*(progress_gap+progress_pb+progress_db+progress_pdi))
            unclipped = terms['nodes']+terms['pace']+terms['gap_level']+terms['pdi_level']+terms['progress']
            assert abs(float(np.clip(unclipped,-1.,1.))-reward)<1e-12
            row.update(weighted_terms=terms, unclipped=unclipped, progress_pdi=progress_pdi,
                       gap_level_saturated=abs(_safe_tanh(gap/(self.baseline_gap+1e-9),s=.5))>.999)
        self.records.append(row)
        self.diagnostic_seconds += time.perf_counter()-started
        return reward
