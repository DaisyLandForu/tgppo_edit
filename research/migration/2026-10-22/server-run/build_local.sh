#!/usr/bin/env bash
# Rebuild from packaged sources; keep original sources and all build attempts.
set -euo pipefail
launch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$launch_dir/env.sh"
case "${1:-}" in ''|--binding-only) ;; *) echo 'Usage: build_local.sh [--binding-only]'; exit 2;; esac
build_root="$SERVER_RUN_ROOT/preparation/build"
attempt_dir="$SERVER_RUN_ROOT/preparation/build-attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$build_root" "$attempt_dir" "$SCIPOPTDIR/lib"
exec > >(tee "$attempt_dir/driver.log") 2>&1
"$TGPPO_PYTHON" -c 'import sys,numpy,Cython,scipy; assert sys.version_info[:2] == (3,8), "Python 3.8 required"; print("Build Python:",sys.version); print("NumPy/Cython/SciPy:",numpy.__version__,Cython.__version__,scipy.__version__)'
build_jobs=${TGPPO_BUILD_JOBS:-8}
if [[ ${1:-} != --binding-only ]]; then
bliss_prefix="$build_root/bliss-root/usr"
if [[ ! -f "$bliss_prefix/include/bliss/graph.hh" ]]; then
    (cd "$build_root" && apt-get download libbliss-dev libbliss-dev-common libbliss2)
    for package in "$build_root"/libbliss*.deb; do
        dpkg-deb -x "$package" "$build_root/bliss-root"
    done
fi
# libbliss is the Ubuntu dependency, not a source-machine solver binary.
cp -a "$bliss_prefix/lib/x86_64-linux-gnu/"libbliss.so* "$SCIPOPTDIR/lib/"
cmake -S "$RESEARCH_ROOT/tgppo/setup/soplex-4.0.1" -B "$build_root/soplex" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SCIPOPTDIR" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON > "$attempt_dir/soplex-configure.log" 2>&1
cmake --build "$build_root/soplex" --parallel "$build_jobs" > "$attempt_dir/soplex-build.log" 2>&1
cmake --install "$build_root/soplex" > "$attempt_dir/soplex-install.log" 2>&1
cmake -S "$RESEARCH_ROOT/tgppo/setup/scip-6.0.1" -B "$build_root/scip" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SCIPOPTDIR" \
    -DSOPLEX_DIR="$SCIPOPTDIR/lib/cmake/soplex" -DZIMPL=OFF -DIPOPT=OFF \
    -DBLISS_INCLUDE_DIR="$bliss_prefix/include" \
    -DBLISS_LIBRARY="$SCIPOPTDIR/lib/libbliss.so" \
    -DCMAKE_INSTALL_RPATH="$SCIPOPTDIR/lib" > "$attempt_dir/scip-configure.log" 2>&1
cmake --build "$build_root/scip" --target libscip scip --parallel "$build_jobs" > "$attempt_dir/scip-build.log" 2>&1
cmake --install "$build_root/scip" > "$attempt_dir/scip-install.log" 2>&1
fi
[[ -f "$SCIPOPTDIR/lib/libscip.so" ]] || { echo 'SCIP build missing'; exit 1; }
# Cython may regenerate scip.c. Compile a copy, leaving archived sources intact.
binding_src="$RESEARCH_ROOT/tgppo/experiments/stage4/metrics_binding"
binding_build="$attempt_dir/metrics_binding"
mkdir -p "$binding_build"
cp -a "$binding_src/setup.py" "$binding_src/setup.cfg" "$binding_src/README.rst" "$binding_src/src" "$binding_build/"
(cd "$binding_build" && "$TGPPO_PYTHON" setup.py build_ext --inplace > "$attempt_dir/binding-build.log" 2>&1)
cp "$binding_build/src/pyscipopt/"scip*.so "$binding_src/src/pyscipopt/"
"$TGPPO_PYTHON" -c 'from pyscipopt import Model; m=Model(); print("SCIP",m.version()); print({n:hasattr(Model,n) for n in ("getCandsState","getMIPState","getNodeState","getIntegralSnapshot","getOpenNodeSnapshot")}); m.freeProb()'
"$SCIPOPTDIR/bin/scip" --version
printf 'Build evidence: %s\n' "$attempt_dir"
