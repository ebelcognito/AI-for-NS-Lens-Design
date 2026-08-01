"""nslens.analysis must reproduce the legacy analysers' NUMBERS.

The invariant here differs from the QD driver's on purpose. These are reports: what must
not move is the machine-readable summary each one writes -- every metric, every seed, per
arm. The prose around it is deliberately unified, because the two legacy twins had
drifted into seven cosmetic differences and one latent bug (ci95's empty case).

    python tests/test_analysis_equivalence.py
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# arena -> (legacy script, results dir)
CASES = {
    "collinear": ("ns_probe/analyze_disc_e1.py", "ns_probe/results_disc"),
    "prism": ("ns_probe/analyze_disc_prism.py", "ns_probe/results_disc_prism"),
}


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def run_case(arena: str, script: str, results: str) -> bool:
    summary_path = REPO / results / "e1_summary.json"
    if not (REPO / results).exists():
        print(f"  SKIP {arena}: {results} not present")
        return True

    saved = None
    if summary_path.exists():
        saved = summary_path.read_bytes()
    try:
        # legacy
        r = subprocess.run([sys.executable, script], cwd=REPO,
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"  FAIL {arena}: legacy exited {r.returncode}")
            for line in (r.stdout + r.stderr).strip().splitlines()[-6:]:
                print("        |", line)
            return False
        legacy = json.loads(summary_path.read_text())

        # nslens
        r = subprocess.run([sys.executable, "-m", "nslens.analysis.discovery_control",
                            arena, "--quiet"], cwd=REPO,
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"  FAIL {arena}: nslens exited {r.returncode}")
            for line in (r.stdout + r.stderr).strip().splitlines()[-6:]:
                print("        |", line)
            return False
        new = json.loads(summary_path.read_text())
    finally:
        if saved is not None:
            summary_path.write_bytes(saved)

    problems = []
    if set(legacy) != set(new):
        problems.append(f"arms differ: {sorted(legacy)} != {sorted(new)}")
    for arm in sorted(set(legacy) & set(new)):
        if set(legacy[arm]) != set(new[arm]):
            problems.append(f"{arm}: metric keys differ")
            continue
        for k in sorted(legacy[arm]):
            va, vb = legacy[arm][k], new[arm][k]
            if len(va) != len(vb):
                problems.append(f"{arm}.{k}: {len(va)} seeds != {len(vb)}")
                continue
            for i, (x, y) in enumerate(zip(va, vb)):
                if not _same(x, y):
                    problems.append(f"{arm}.{k}[seed {i}]: {x!r} != {y!r}")

    if problems:
        print(f"  FAIL {arena}: {len(problems)} difference(s)")
        for p in problems[:5]:
            print("        -", p)
        return False
    n = sum(len(v) for arm in new.values() for v in arm.values())
    print(f"  PASS {arena}: {n} metric values identical across {len(new)} arms")
    return True


def main() -> int:
    print("nslens.analysis vs legacy discovery-control analysers")
    results = [run_case(a, s, r) for a, (s, r) in CASES.items()]
    if all(results):
        print("ALL ANALYSIS EQUIVALENCE CHECKS PASS")
        return 0
    print("ANALYSIS EQUIVALENCE FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
