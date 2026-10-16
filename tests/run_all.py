"""The gate. Run this before and after every refactor commit.

    python tests/run_all.py

Exit 0 means the refactor was a pure tidying: structure changed, behaviour did not.
Any non-zero exit means numbers moved, which invalidates archives, benchmarks and the
CODE V export -- revert rather than re-baseline, unless the change was intended.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FOLD1_REF = {
    "ray_y": -18.65635660138779, "ang": 37.483939982719576,
    "cx": 63.993227745392275, "thick": 41.69415522629459,
    "fsag": 9.34758117693124, "bsag": 0.7439142577369253,
    "tilt_a": 20.4858703051664, "tilt_b": -14.223046072754633,
    "n": 1.6571399294573437,
}


def run(label: str, args: list, expect: str | None = None) -> bool:
    proc = subprocess.run([sys.executable, *args], cwd=REPO,
                          capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    ok = proc.returncode == 0 and (expect is None or expect in out)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print("        exit", proc.returncode)
        for line in out.strip().splitlines()[-8:]:
            print("        |", line)
    return ok


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "fold1_ref.json"
        ref.write_text(json.dumps(FOLD1_REF))

        print("nslens verification gate")
        results = [
            run("characterization pins (11 probes, 3 geometries)",
                ["tests/characterization/pin_geometry.py"]),
            run("Element == legacy element builders",
                ["tests/test_element_equivalence.py"], "ALL EQUIVALENCE CHECKS PASS"),
            run("Layout == legacy scene builders",
                ["tests/test_scene_equivalence.py"], "ALL SCENE EQUIVALENCE CHECKS PASS"),
            run("Problem constants == legacy modules",
                ["tests/test_problem_constants.py"], "ALL CONSTANTS AGREE"),
            run("fold1 benchmark (13/13)",
                ["ns_probe/benchmarks/fold1_env/verify_fold1.py", str(ref)],
                "13/13 checks passed"),
            run("CODE V Tier-A v2 (5 fields close)",
                ["codev_handoff/verify_v2.py"], "ALL FIELDS CLOSE"),
            run("analysis == legacy analysers (numbers)",
                ["tests/test_analysis_equivalence.py"],
                "ALL ANALYSIS EQUIVALENCE CHECKS PASS"),
            run("typed-interaction physics checks",
                ["ns_probe/test_typed_interactions.py"], "ALL CHECKS PASSED"),
            run("canonical interaction fingerprint v1",
                ["tests/test_interaction_fingerprint.py"],
                "ALL FINGERPRINT CHECKS PASS"),
            run("fixed source/material environment contract",
                ["tests/test_fixed_environments.py"],
                "ALL FIXED-ENVIRONMENT CHECKS PASS"),
            run("fixed interaction-aware search objective",
                ["tests/test_fixed_interaction_objective.py"],
                "ALL FIXED-OBJECTIVE CHECKS PASS"),
            run("fixed baseline dense-audit admission contract",
                ["tests/test_fixed_baseline_audit.py"],
                "ALL FIXED-BASELINE AUDIT CHECKS PASS"),
            # the RL env deliberately DIVERGES from the frozen defaults below; this pair
            # asserts the divergence is still there and that the reward still agrees with
            # the driver wherever both produce a number
            run("RL env stays hardened vs the frozen defaults",
                ["tests/test_rl_env_hardened.py"], "ALL RL ENV CHECKS PASS"),
            run("RL reward == QD driver (fitness + bc)",
                ["tests/test_rl_reward.py"], "ALL RL REWARD CHECKS PASS"),
            run("RL/SFT dataset contracts and leakage-safe splits",
                ["tests/test_rl_data.py"], "ALL RL DATA CHECKS PASS"),
            run("Stage-1 freeform body (continuous boundary, no faces)",
                ["tests/test_freeform_body.py"], "ALL FREEFORM-BODY CHECKS PASS"),
            run("spherical-face element with sampled corners",
                ["tests/test_arc_polygon.py"], "ALL ARC-POLYGON CHECKS PASS"),
            run("Stage-1 AR/VR relay objective and feasibility ladder",
                ["tests/test_freeform_relay.py"], "ALL FREEFORM-RELAY CHECKS PASS"),
            run("official EvoTF emitter adapter and driver hook",
                ["tests/test_evotf_emitter.py"], "ALL EVOTF EMITTER CHECKS PASS"),
            run("lossless optimizer trajectory recorder and splits",
                ["tests/test_optimizer_trajectory.py"],
                "ALL OPTIMIZER TRAJECTORY CHECKS PASS"),
            run("Optical-EvoTF adapters, symmetry, gradients, and checkpoints",
                ["tests/test_optical_evotf.py"],
                "ALL OPTICAL EVOTF CHECKS PASS"),
            run("paired EvoTF ablation summaries",
                ["tests/test_fold_evotf_ablation.py"],
                "ALL FOLD EVOTF ABLATION CHECKS PASS"),
            # slowest check by far (~8 min): reruns every legacy runner AND its nslens
            # config, then diffs the archives cell-for-cell
            run("QD driver == legacy runners (13 cases)",
                ["tests/test_qd_equivalence.py", "--evals", "600"],
                "ALL QD EQUIVALENCE CHECKS PASS"),
        ]

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
