"""Action-before-branch frontier summary and separately versioned trajectory sidecar."""
import time
import numpy as np

SCHEMA = 'stage4-frontier-v1'


def frontier_features(snapshot):
    bounds = np.asarray(snapshot['bounds'], dtype=np.float64)
    depths = np.asarray(snapshot['depths'], dtype=np.int64)
    n = len(bounds)
    if bounds.ndim != 1 or depths.shape != bounds.shape or (depths < 0).any():
        raise ValueError('Malformed open-node snapshot')
    if n != sum(snapshot[k] for k in ('nleaves', 'nchildren', 'nsiblings')):
        raise ValueError('Missing open-node category')
    if n != snapshot['solver_open_nodes']:
        raise ValueError('Frontier count differs from SCIPgetNNodesLeft')
    if np.isnan(bounds).any():
        raise ValueError('NaN frontier bound')
    infinity = float(snapshot['infinity'])
    pos = bounds >= infinity
    neg = bounds <= -infinity
    finite = ~(pos | neg)
    h = np.zeros((8, 8), dtype=np.float64)
    nf = int(finite.sum())
    bound_span = float(np.ptp(bounds[finite])) if nf else 0.
    max_depth = int(depths.max()) if n else 0
    finite_depth_span = int(np.ptp(depths[finite])) if nf else 0
    if nf:
        b = bounds[finite]
        bx = (b - b.min()) / bound_span if bound_span > 0 else np.zeros(nf)
        dx = depths[finite] / float(max_depth) if max_depth else np.zeros(nf)
        bi = np.minimum((bx * 8).astype(np.int64), 7)
        di = np.minimum((dx * 8).astype(np.int64), 7)
        np.add.at(h, (bi, di), 1.)
        h /= nf
    prod = np.outer(h.sum(1), h.sum(0))
    scalars = np.asarray([np.log1p(n), float(n == 0), nf / n if n else 0.,
                          float(pos.sum()) / n if n else 0.,
                          float(neg.sum()) / n if n else 0.], dtype=np.float32)
    diag = dict(n_open=n, n_finite=nf, nleaves=int(snapshot['nleaves']),
        nchildren=int(snapshot['nchildren']), nsiblings=int(snapshot['nsiblings']),
        bound_span=bound_span, finite_depth_span=finite_depth_span, max_depth=max_depth,
        both_axes_nondegenerate=bool(bound_span > 0 and finite_depth_span > 0),
        association_l1=float(np.abs(h-prod).sum()))
    return np.r_[h.ravel(), scalars].astype(np.float32), diag


def discounted_returns(rewards, gamma=.97):
    result = np.empty(len(rewards), dtype=np.float32)
    total = 0.
    for i in range(len(rewards)-1, -1, -1):
        total = float(rewards[i]) + gamma * total
        result[i] = total
    return result


class SearchRecorder:
    def __init__(self, time_limit):
        self.time_limit = float(time_limit)
        self.rows = []
        self.pending = None
        self.seconds = 0.

    def capture(self, model, state):
        if self.pending is not None:
            raise RuntimeError('Previous feature snapshot not committed')
        start = time.perf_counter()
        snapshot = model.getOpenNodeSnapshot()
        vector, diag = frontier_features(snapshot)
        t = float(snapshot['solving_time'])
        self.pending = dict(frontier=vector, remaining_time=np.float32(
            np.clip(1.-t/self.time_limit, 0., 1.)), solving_time=t,
            num_candidates=len(state['cands_state']), diagnostics=diag)
        self.seconds += time.perf_counter() - start

    def commit(self, action):
        if self.pending is None:
            raise RuntimeError('No action-before-branch snapshot')
        self.pending.update(action=int(action), decision_index=len(self.rows))
        self.rows.append(self.pending)
        self.pending = None

    def export(self, trajectory):
        rows = [r for e in trajectory['episodes'] for r in e['transitions']]
        if self.pending is not None or len(rows) != len(self.rows):
            raise ValueError('Search sidecar / trajectory length mismatch')
        for i, (row, extra) in enumerate(zip(rows, self.rows)):
            if row['action'] != extra['action'] or len(row['cands_state']) != extra['num_candidates']:
                raise ValueError('Search sidecar alignment failure at {}'.format(i))
        return dict(schema=SCHEMA, collection_timing='before action and branchVar',
                    rows=self.rows, extraction_wall_seconds=self.seconds)
