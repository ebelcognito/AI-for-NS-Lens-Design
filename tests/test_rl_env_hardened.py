"""The RL env config must keep diverging from the frozen QD defaults.

Every permissive default in `ExperimentConfig` is there so `test_qd_equivalence.py` can
reproduce committed archives cell-for-cell, and every one of them is also an exploit that
CMA-ME already found. `nslens/rl/env_config.py` turns them all on. This test is the
tripwire for somebody "tidying" that file back toward the defaults -- it fails naming the
exploit that was just reopened.

It also checks the second half of doc 18's milestone-1 gate: the hardened config must
actually reject known-exploit designs pulled from the archives on disk. No engine needed --
every exploit checked here is decided by geometry before a ray is traced.

    python tests/test_rl_env_hardened.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.rl import env_config                                   # noqa: E402
from nslens.rl.reward import geometry_verdict                      # noqa: E402

#: milestone-1 gate: at least this many archive elites must be refused
MIN_EXPLOITS_REJECTED = 3


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'} {label}" + (f": {detail}" if detail else ""))
    return ok


def check_divergences() -> bool:
    """Every documented divergence is present, on every env."""
    ok = True
    expected = set(env_config.DIVERGENCES) | set(env_config.EXTRA_GATES)
    for name in env_config.names():
        env = env_config.get(name)
        found = set(env_config.divergences(env))
        missing = sorted(expected - found)
        ok &= _check(f"env {name!r}: {len(found)}/{len(expected)} gates hardened",
                     not missing,
                     "" if not missing else
                     "REVERTED TO THE FROZEN DEFAULT: " + ", ".join(
                         f"{k} ({env_config.DIVERGENCES.get(k, env_config.EXTRA_GATES.get(k))[2]})"
                         for k in missing))
    return ok


def check_not_gates() -> bool:
    """The two metrics measured-but-never-gated stay out of the gate set."""
    env = env_config.get("fold")
    hardened = set(env_config.divergences(env))
    leaked = sorted(set(env_config.NOT_GATES) & hardened)
    return _check("etendue/ratio tests stay out of the gate set", not leaked,
                  "" if not leaked else f"{leaked} became gates; both were MEASURED to "
                                        "reject the best designs in the program")


def check_variant_isolation() -> bool:
    """A variant changes the fields and the window and nothing else -- otherwise a spec
    silently redefines the genome box or a gate."""
    base = env_config.get("fold")
    var = env_config.get("fold", fov=12.0, focus_w=4.0)
    ok = _check("variant changes the field set",
                var.fields != base.fields and max(var.fields) == 12.0)
    ok &= _check("variant changes the scoring window",
                 var.problem.focus_w == 4.0 and base.problem.focus_w == 8.0)
    ok &= _check("variant keeps the genome box",
                 list(var.problem.lo) == list(base.problem.lo)
                 and list(var.problem.hi) == list(base.problem.hi))
    ok &= _check("variant keeps every gate",
                 set(env_config.divergences(var)) == set(env_config.divergences(base)))
    return ok


def check_explicit_clearances() -> bool:
    """Direct tripwires for the two physical-intersection exploits.

    Archive sampling alone is insufficient: today it happened to find bulb/source failures
    but no detector-intersection elite. These deterministic designs ensure both clearance
    implementations themselves remain live.
    """
    env = env_config.get("fold")
    # Flat, unsheared body ends at x=107.5 and clears the x=120 detector by 12.5 mm.
    clear = [0, 0, 85, 45, 0, 0, 0, 0, 1.7]
    # Face shear pushes the same body through the detector plane (xmax ~=121.5).
    detector_hit = [0, 0, 85, 45, 0, 0, -25, 25, 1.7]
    # The low-cx sheared body reaches left of the source line at x=0.
    source_hit = [0, 0, 35, 45, 0, 0, 25, 25, 1.7]
    v_clear = geometry_verdict(env, clear)
    v_det = geometry_verdict(env, detector_hit)
    v_src = geometry_verdict(env, source_hit)
    ok = _check("12 mm detector clearance accepts a 12.5 mm gap", v_clear is None)
    ok &= _check("glass intersecting detector is a hard pre-trace rejection",
                 v_det is not None and v_det.stage == "detector_intersect",
                 "" if v_det is None else f"stage={v_det.stage}: {v_det.reason}")
    ok &= _check("glass intersecting source is a hard pre-trace rejection",
                 v_src is not None and v_src.stage == "embedded_source",
                 "" if v_src is None else f"stage={v_src.stage}: {v_src.reason}")
    return ok


def check_rejects_known_exploits() -> bool:
    """The milestone-1 gate: the hardened config refuses real archive elites.

    Geometry only, deliberately. These elites come from runs whose own gates were already
    on, so what is being measured here is the tightening THIS config adds -- chiefly
    `max_sag_ratio` 0.55 -> 0.268 and the source-clearance test the fold env never had.
    """
    env = env_config.get("fold")
    cells = [c for f in sorted(glob.glob(str(REPO / "runs/ab40k_*/seed*/archive.json")))
             for c in json.loads(Path(f).read_text())["cells"]]
    if not cells:
        return _check("known-exploit rejection", False, "no ab40k archives on disk")

    stages = {}
    for c in cells:
        v = geometry_verdict(env, c["genotype"])
        if v is not None:
            stages[v.stage] = stages.get(v.stage, 0) + 1
    n = sum(stages.values())
    return _check(
        f"hardened geometry refuses {n} of {len(cells)} archive elites",
        n >= MIN_EXPLOITS_REJECTED,
        ", ".join(f"{k}={v}" for k, v in sorted(stages.items())) or "nothing rejected")


def main() -> int:
    print("RL hardened env config")
    ok = all([
        check_divergences(),
        check_not_gates(),
        check_variant_isolation(),
        check_explicit_clearances(),
        check_rejects_known_exploits(),
    ])
    print("ALL RL ENV CHECKS PASS" if ok else "RL ENV CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
