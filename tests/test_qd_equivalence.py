"""The config-driven QD driver must reproduce each legacy runner cell-for-cell.

This is the gate for phase 2. For a fixed seed and eval budget, `python -m nslens.qd <name>`
must produce the same archive as the script it replaces: same cells, same genotypes, same
fitness, same behaviour coordinates, same invalid count, same history.

Only then is deleting the legacy runner a tidying rather than a result change.

    python tests/test_qd_equivalence.py            # default 600 evals
    python tests/test_qd_equivalence.py --evals 2000

Note the driver writes a SUPERSET of the legacy `info` fields for some experiments
(multifield especially), so comparison is over the fields the legacy runner wrote plus
the cell identity. Extra keys are additive and never change search behaviour.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_DESC = "ns_probe/run_fold_qd_descriptors.py"

# name -> (legacy argv, nslens experiment name)
CASES = {
    "fold_qd": (["ns_probe/run_fold_qd.py"], "fold_qd"),
    "fold_qd_notilt": (["ns_probe/run_fold_qd.py", "--lock-tilt"], "fold_qd_notilt"),
    "orderly_qd": (["ns_probe/run_orderly_qd.py"], "orderly_qd"),
    "prism_fold_qd": (["ns_probe/run_prism_fold_qd.py"], "prism_fold_qd"),
    "multifield_qd": (["ns_probe/run_multifield_qd.py"], "multifield_qd"),
    # doc 12 descriptor ablation: one case per BC1 axis, plus a crossed grid.
    # `robust` is the expensive one -- it re-evaluates each design 3x under jitter,
    # so it also exercises the separate perturbation RNG stream (seed + 9999).
    "desc_baseline": ([_DESC, "--bc", "baseline"], "desc_baseline"),
    "desc_nsfrac": ([_DESC, "--bc", "nsfrac"], "desc_nsfrac"),
    "desc_tilt": ([_DESC, "--bc", "tilt"], "desc_tilt"),
    "desc_critmargin": ([_DESC, "--bc", "critmargin"], "desc_critmargin"),
    "desc_locus": ([_DESC, "--bc", "locus"], "desc_locus"),
    "desc_robust": ([_DESC, "--bc", "robust"], "desc_robust"),
    "desc_cross": ([_DESC, "--bc", "baseline", "--bc2", "locus"],
                   "desc_baseline_x_locus"),
    "arvr_qd": (["ns_probe/run_arvr_qd.py"], "arvr_qd"),
}

# fields every legacy runner writes per cell (multifield writes its own extras)
CORE_FIELDS = ("fitness", "genotype", "tir_count", "T")


def _load(p: Path) -> dict:
    return json.loads((p / "archive.json").read_text())


def _compare(legacy: dict, new: dict) -> list:
    problems = []
    A = {(c["i0"], c["i1"]): c for c in legacy["cells"]}
    B = {(c["i0"], c["i1"]): c for c in new["cells"]}
    if set(A) != set(B):
        problems.append(f"cell keys differ: only-legacy={sorted(set(A)-set(B))[:4]} "
                        f"only-new={sorted(set(B)-set(A))[:4]}")
        return problems
    for k in sorted(A):
        for f in CORE_FIELDS:
            if A[k].get(f) != B[k].get(f):
                problems.append(f"cell {k} field {f!r}: {A[k].get(f)!r} != {B[k].get(f)!r}")
        # every field the legacy runner wrote must be reproduced
        for f in A[k]:
            if f in ("i0", "i1"):
                continue
            if f in B[k] and A[k][f] != B[k][f]:
                problems.append(f"cell {k} field {f!r}: {A[k][f]!r} != {B[k][f]!r}")
            elif f not in B[k]:
                problems.append(f"cell {k} missing field {f!r}")
    for f in ("evals", "invalid", "seed"):
        if legacy["meta"].get(f) != new["meta"].get(f):
            problems.append(f"meta {f}: {legacy['meta'].get(f)} != {new['meta'].get(f)}")
    if legacy["history"] != new["history"]:
        problems.append("history differs")
    return problems


def run_case(name: str, evals: int, seed: int, tmp: Path) -> bool:
    legacy_argv, exp = CASES[name]
    leg_out, new_out = tmp / f"{name}_legacy", tmp / f"{name}_new"

    r1 = subprocess.run([sys.executable, *legacy_argv, "--evals", str(evals),
                         "--seed", str(seed), "--out", str(leg_out)],
                        cwd=REPO, capture_output=True, text=True, timeout=1800)
    r2 = subprocess.run([sys.executable, "-m", "nslens.qd", exp, "--evals", str(evals),
                         "--seed", str(seed), "--out", str(new_out)],
                        cwd=REPO, capture_output=True, text=True, timeout=1800)
    for label, r in (("legacy", r1), ("nslens", r2)):
        if r.returncode != 0:
            print(f"  FAIL {name}: {label} runner exited {r.returncode}")
            for line in (r.stdout + r.stderr).strip().splitlines()[-6:]:
                print("        |", line)
            return False

    problems = _compare(_load(leg_out), _load(new_out))
    n_cells = len(_load(leg_out)["cells"])
    if problems:
        print(f"  FAIL {name}: {len(problems)} difference(s), {n_cells} cells")
        for p in problems[:5]:
            print("        -", p)
        return False
    print(f"  PASS {name}: {n_cells} cells identical")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="run a single case")
    args = ap.parse_args()

    cases = [args.only] if args.only else list(CASES)
    print(f"QD driver vs legacy runners ({args.evals} evals, seed {args.seed})")
    with tempfile.TemporaryDirectory() as td:
        results = [run_case(c, args.evals, args.seed, Path(td)) for c in cases]
    if all(results):
        print("ALL QD EQUIVALENCE CHECKS PASS")
        return 0
    print("QD EQUIVALENCE FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
