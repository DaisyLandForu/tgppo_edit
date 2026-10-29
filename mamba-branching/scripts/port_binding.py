#!/usr/bin/env python3
"""Build a separate SCIP-8 binding with the published TreeGate feature formulas.

The source templates in upstream/ and the historical TGPPO tree are read only.
Generated C declarations are compiled against actual SCIP-8 headers, never a
SCIP-6 binary or a hand-written approximation of C struct offsets.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent


def model_methods(source):
    matches = list(re.finditer(r"^    (?:@[^\n]+\n    )*def (\w+)\([^\n]*", source, re.M))
    result = {}
    for match in matches:
        end = re.search(r"\n(?:    (?:@|def |cdef )|[^ \n#])", source[match.end():])
        stop = match.end() + end.start() if end else len(source)
        result[match.group(1)] = source[match.start():stop].rstrip() + "\n"
    return result


def main():
    legacy = RESEARCH / "tgppo/setup/TBranT/PySCIPOpt/src/pyscipopt"
    upstream = ROOT / "upstream/PySCIPOpt-4.3.0"
    target = ROOT / "setup/PySCIPOpt-4.3.0-features"
    if target.exists():
        raise SystemExit(f"Already exists: {target}; use the saved source/patch, do not overwrite a build.")
    shutil.copytree(upstream, target, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    pyx_path = target / "src/pyscipopt/scip.pyx"
    pxd_path = target / "src/pyscipopt/scip.pxd"
    legacy_pyx = (legacy / "scip.pyx").read_text()
    legacy_pxd = (legacy / "scip.pxd").read_text()
    pyx, pxd = pyx_path.read_text(), pxd_path.read_text()
    methods = model_methods(legacy_pyx)
    names = ["getNDiscreteVars", "getChildren", "getMaxDepth", "getNNodesLeft", "getNObjlimLeaves", "getNCreatedNodes", "getNCreatedNodesRun",
             "getPrimalDualIntegral", "getNumLPBranchCands", "relPosition", "relDistance", "gNormMax",
             "getVarScore", "getFairNNodes", "getNodeState", "getMIPState", "getCandsState",
             "executeBranchRule"]
    added = "\n".join(methods[name] for name in names)
    # PySCIPOpt-4 returns a tuple; the legacy version returned only candidates.
    added = added.replace("len(self.getLPBranchCands())", "self.getNumLPBranchCands()")
    # Use accessors for opaque nodes rather than expose SCIP-6 struct layouts.
    added = re.sub(r"(leaves|children|siblings)\[([^]]+)\]\.lowerbound", r"SCIPnodeGetLowerbound(\1[\2])", added)
    added = re.sub(r"(leaves|children|siblings)\[([^]]+)\]\.depth", r"SCIPnodeGetDepth(\1[\2])", added)
    added = added.replace("        brancher = SCIPfindBranchrule(self._scip, brancher_name)",
                          "        if isinstance(brancher_name, str):\n            brancher_name = brancher_name.encode('utf-8')\n        brancher = SCIPfindBranchrule(self._scip, brancher_name)")
    # Validate sizes before entering unchecked Cython feature writes.
    for method, dim, expected in [("getNodeState", "node_dim", 8), ("getMIPState", "mip_dim", 53), ("getCandsState", "var_dim", 25)]:
        pattern = rf'(def {method}\([^\n]*\n\s+"""[\s\S]*?""")'
        added, count = re.subn(pattern, rf'\1\n        if {dim} != {expected}:\n            raise ValueError("{method} requires dimension {expected}")', added, count=1)
        assert count == 1
    # Explicit division can be zero when primal and dual bounds coincide.
    # Preserve finite values; define an already-closed interval as position zero.
    added = added.replace("if SCIPisInfinity(self._scip, ub):", "if SCIPisInfinity(self._scip, ub) or ub == lb:", 1)
    pyx = pyx.replace("import weakref\n", "import weakref\nimport numpy as np\n", 1)
    # The end of Model is followed by module-level memory debugging functions.
    insertion = pyx.index("\n# debugging memory management")
    pyx = pyx[:insertion] + "\n" + added + "\n" + pyx[insertion:]
    node_method = methods["getBranchInfos"].replace("self.node", "self.scip_node")
    insertion = pyx.index("\ncdef class Variable(")
    pyx = pyx[:insertion] + "\n" + node_method + "\n" + pyx[insertion:]
    # Public functions missing from PySCIPOpt-4's own declarations.
    used = sorted(set(re.findall(r"\b(SCIP\w+)\(", added + node_method)))
    declarations = []
    for name in used:
        if re.search(r"\b" + name + r"\s*\(", pxd):
            continue
        match = re.search(r"^    [\w *]+\b" + name + r"\([^;]*?\)(?:\s*#[^\n]*)?\s*$", legacy_pxd, re.M)
        if not match:
            raise RuntimeError(f"No source declaration for {name}")
        declarations.append(match.group(0).rstrip())
    pxd += '\n\ncdef extern from "scip/scip.h":\n' + "\n".join(declarations) + "\n"
    stats = sorted(set(re.findall(r"\.stat\.(\w+)", added)))
    # Verify fields against SCIP-8 headers before generating partial declarations.
    stat_header = Path("/usr/local/scip-8.0.4/include/scip/struct_stat.h").read_text()
    for name in stats:
        if not re.search(r"\b" + name + r"\b", stat_header):
            raise RuntimeError(f"SCIP-8 is missing statistical field {name}")
    pxd += '\ncdef extern from "scip/struct_stat.h":\n    ctypedef struct SCIP_STAT:\n'
    pxd += "".join(f"        {'SCIP_Real' if 'gap' in name or name == 'primaldualintegral' else 'SCIP_Longint'} {name}\n" for name in stats)
    pxd = pxd.replace("ctypedef struct SCIP:\n        pass", "ctypedef struct SCIP:\n        SCIP_STAT* stat", 1)
    pxd = pxd.replace("ctypedef struct SCIP_BRANCHRULE:\n        pass", "ctypedef struct SCIP_BRANCHRULE:\n        SCIP_RETCODE (*branchexeclp)(SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result)", 1)
    pxd += '\ncdef extern from "scip/struct_scip.h":\n    pass\ncdef extern from "scip/struct_branch.h":\n    pass\n'
    pyx_path.write_text(pyx)
    pxd_path.write_text(pxd)
    record = {"legacy_pyx_sha256": hashlib.sha256(legacy_pyx.encode()).hexdigest(),
              "legacy_pxd_sha256": hashlib.sha256(legacy_pxd.encode()).hexdigest(),
              "added_model_methods": names, "added_node_methods": ["getBranchInfos"],
              "missing_public_declarations": len(declarations), "scip_stat_fields": stats,
              "status": "generated; compilation and real callback checks required"}
    (ROOT / "setup/binding-port.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
