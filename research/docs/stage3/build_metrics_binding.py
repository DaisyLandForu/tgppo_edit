"""Build an isolated, read-only native integral snapshot; installed binding is untouched."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/root/research')
SOURCE = ROOT / 'tgppo/setup/PySCIPOpt'
DEST = ROOT / 'tgppo/experiments/stage3/metrics_binding'


def main():
    if DEST.exists():
        raise FileExistsError(str(DEST))
    shutil.copytree(SOURCE, DEST, ignore=shutil.ignore_patterns(
        '.git', 'build', 'dist', '*.egg-info', '__pycache__', '*.so', 'scip.c'))
    pxd = DEST / 'src/pyscipopt/scip.pxd'
    text = pxd.read_text()
    needle = '        SCIP_Real previousgap\n'
    assert text.count(needle) == 1
    pxd.write_text(text.replace(needle, needle + '        SCIP_Real previntegralevaltime\n'))
    pyx = DEST / 'src/pyscipopt/scip.pyx'
    text = pyx.read_text()
    needle = '    def getPrimalDualIntegral(self):\n'
    assert text.count(needle) == 1
    addition = '''    def getIntegralSnapshot(self):
        """Read-only native integration state, with no update or flush."""
        return dict(integral=self._scip.stat.primaldualintegral,
                    last_update_time=self._scip.stat.previntegralevaltime,
                    previous_percent_gap=self._scip.stat.previousgap,
                    last_primal_bound=self._scip.stat.lastprimalbound,
                    last_dual_bound=self._scip.stat.lastdualbound,
                    solving_time=SCIPgetSolvingTime(self._scip))

'''
    pyx.write_text(text.replace(needle, addition + needle))
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=DEST, check=True)
    print('Isolated binding:', DEST / 'src')


if __name__ == '__main__':
    main()
