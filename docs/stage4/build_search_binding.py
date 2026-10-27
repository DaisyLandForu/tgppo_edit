"""Build a separate stage-4 binding; retain the stage-3 read-only metric patch."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/root/research')
SOURCE = ROOT / 'tgppo/experiments/stage3/metrics_binding'
DEST = ROOT / 'tgppo/experiments/stage4/metrics_binding'

ADDITION = '''    def getOpenNodeSnapshot(self):
        """Copy all open leaves/children/siblings in internal minimization space.

        No node pointers escape this call. The current focus node is excluded.
        Intended only for action-before-branch collection, not deployment.
        """
        cdef SCIP_NODE** leaves
        cdef SCIP_NODE** children
        cdef SCIP_NODE** siblings
        cdef int nl, nc, ns, i, j
        PY_SCIP_CALL(SCIPgetOpenNodesData(self._scip, &leaves, &children, &siblings,
                                        &nl, &nc, &ns))
        bounds = np.empty(nl + nc + ns, dtype=np.float64)
        depths = np.empty(nl + nc + ns, dtype=np.int32)
        cdef double[::1] bv = bounds
        cdef int[::1] dv = depths
        j = 0
        for i in range(nl):
            bv[j] = SCIPnodeGetLowerbound(leaves[i])
            dv[j] = SCIPnodeGetDepth(leaves[i])
            j += 1
        for i in range(nc):
            bv[j] = SCIPnodeGetLowerbound(children[i])
            dv[j] = SCIPnodeGetDepth(children[i])
            j += 1
        for i in range(ns):
            bv[j] = SCIPnodeGetLowerbound(siblings[i])
            dv[j] = SCIPnodeGetDepth(siblings[i])
            j += 1
        return dict(bounds=bounds, depths=depths, nleaves=nl, nchildren=nc,
                    nsiblings=ns, solver_open_nodes=SCIPgetNNodesLeft(self._scip),
                    infinity=SCIPinfinity(self._scip),
                    solving_time=SCIPgetSolvingTime(self._scip))

'''


def main():
    if DEST.exists():
        raise FileExistsError(str(DEST))
    shutil.copytree(SOURCE, DEST, ignore=shutil.ignore_patterns(
        '.git', 'build', 'dist', '*.egg-info', '__pycache__', '*.so', 'scip.c'))
    path = DEST / 'src/pyscipopt/scip.pyx'
    source = path.read_text()
    needle = '    def getMIPState(self, mip_dim):\n'
    # Place before its decorators, rather than moving decorators onto the new method.
    marker = '    @cython.boundscheck(False)\n    @cython.wraparound(False)\n' + needle
    assert source.count(marker) == 1
    path.write_text(source.replace(marker, ADDITION + marker))
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=DEST, check=True)


if __name__ == '__main__':
    main()
