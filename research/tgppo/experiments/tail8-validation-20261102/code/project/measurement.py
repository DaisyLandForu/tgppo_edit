"""Read-only bounds trace and bounded native PDI over a common horizon.

SCIP 6's integral is in percent-seconds and has integrand in [0, 100].
The custom Python getter reads the accumulator without flushing its tail.
A separate read-only snapshot exposes the native last time and previous gap.
We keep the original raw value for H3, and compute the complete tail separately.
No policy input, reward formula or solver setting is changed here.
"""
import math
import time
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE

MEASUREMENT_VERSION = 'native-pdi-snapshot-horizon-v2'


def percent_gap(primal, dual, infinity=1e20, epsilon=1e-9):
    if not all(math.isfinite(x) for x in (primal, dual)):
        raise ValueError('IEEE nonfinite bound')
    if abs(primal - dual) <= epsilon:
        return 0.
    if primal * dual <= 0 or max(abs(primal), abs(dual)) >= infinity:
        return 100.
    return min(100., 100. * abs(primal - dual) / max(abs(primal), abs(dual)))


def horizon_integral(integral, stop_time, horizon, final_gap, proof_completed):
    """Exact extension before T; rigorous enclosure when a solve overshoots T.

    Never infer the last segment from getGap(), whose definition is different.
    For overshoot, all possible integrals satisfy I(s)-100(s-T) <= I(T) <= I(s).
    Midpoint is an explicitly labelled estimate, not an exact clipped integral.
    """
    if not all(math.isfinite(x) for x in (integral, stop_time, horizon, final_gap)):
        raise ValueError('Invalid integral inputs')
    if integral < 0 or stop_time < 0 or horizon <= 0 or not 0 <= final_gap <= 100:
        raise ValueError('Invalid integral range')
    if stop_time <= horizon:
        tail = 0. if proof_completed else final_gap * (horizon - stop_time)
        lower = upper = integral + tail
    else:
        lower = max(0., integral - 100. * (stop_time - horizon))
        upper = min(integral, 100. * horizon)
    if upper < lower - 1e-6:
        raise ValueError('Native integral outside physical range')
    return dict(pdi_horizon_lower=lower, pdi_horizon_upper=upper,
                pdi_horizon_midpoint=(lower + upper) / 2.,
                pdi_horizon_uncertainty=upper - lower,
                pdi_horizon_exact=abs(upper - lower) < 1e-9,
                pdi_horizon_normalized_midpoint=(lower + upper) / (2. * 100. * horizon),
                pdi_horizon_seconds=horizon)


class BoundTrace(Eventhdlr):
    def __init__(self):
        self.points = []
        self.seconds = 0.
        self.calls = 0
        self.error = None
        self.event_mask = (SCIP_EVENTTYPE.PRESOLVEROUND | SCIP_EVENTTYPE.NODEFOCUSED |
            SCIP_EVENTTYPE.NODESOLVED | SCIP_EVENTTYPE.LPSOLVED | SCIP_EVENTTYPE.BESTSOLFOUND)

    def eventinit(self):
        self.model.catchEvent(self.event_mask, self)

    def eventexit(self):
        self.model.dropEvent(self.event_mask, self)

    def eventexec(self, event):
        self.sample('event')

    def sample(self, reason):
        start = time.perf_counter()
        self.calls += 1
        try:
            p, d = float(self.model.getPrimalbound()), float(self.model.getDualbound())
            if not all(math.isfinite(x) for x in (p, d)):
                raise ValueError('Nonfinite bound trace')
            # Log every bound change, not every event. Initial unknown interval is 100%.
            if not self.points or (p, d) != (self.points[-1]['primal'], self.points[-1]['dual']) or reason == 'stop':
                snapshot = self.model.getIntegralSnapshot()
                self.points.append(dict(time=float(self.model.getSolvingTime()), primal=p, dual=d,
                    native_raw=snapshot['integral'], native_last_update_time=snapshot['last_update_time'],
                    native_previous_percent_gap=snapshot['previous_percent_gap'], reason=reason))
        except Exception as exc:
            self.error = repr(exc)
            self.model.interrupt()
        finally:
            self.seconds += time.perf_counter() - start


def finish_measurement(model, info, horizon, trace):
    trace.sample('stop')
    if trace.error:
        raise ValueError(trace.error)
    snapshot = model.getIntegralSnapshot()
    before, stop = snapshot['integral'], snapshot['solving_time']
    if abs(before - model.getPrimalDualIntegral()) > 1e-9:
        raise ValueError('Read-only snapshot disagrees with original getter')
    if snapshot['last_update_time'] > stop + 1e-6:
        raise ValueError('Native integral update time exceeds stopped clock')
    gap = snapshot['previous_percent_gap']
    complete = before + max(0., stop - snapshot['last_update_time']) * gap
    # Presolve can certify cutoff infeasibility while getDualbound() stays -infinity.
    # Preserve that raw bound. A verified proof closes the *cutoff search task*;
    # its zero extension is based on the certificate, not an invented raw bound.
    metrics = horizon_integral(complete, stop, horizon, gap, info['proof_completed'])
    metrics.update(pdi_native_raw_before_flush=before, pdi_native_complete_at_stop=complete,
        pdi_tail_completed_readonly=complete - before, pdi_native_final_percent_gap=gap,
        integral_snapshot=snapshot,
        pdi_extension_basis=('verified_proof_certificate' if info['proof_completed']
                             else 'hold_final_native_bound_gap'),
        measurement_version=MEASUREMENT_VERSION, bounds_trace=trace.points,
        measurement_event_calls=trace.calls, measurement_callback_seconds=trace.seconds,
        pdi_measurement_semantics='native_percent_seconds; exact_extension_or_rigorous_overshoot_interval')
    return metrics
