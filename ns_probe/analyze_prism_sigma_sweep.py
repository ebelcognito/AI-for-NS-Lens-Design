"""Analysis for the prism sigma-scaling sweep (run_prism_sigma_sweep.py).

Asks three questions, per environment, isotropic vs normalized, over matched seeds:

  1. Is the index actually being searched now?   -> frac of elites on an n rail,
     and the spread of n across the archive.
  2. Is the search better?                       -> QD score, coverage, best fitness,
     and (arvr) the number of eyebox-valid elites.
  3. Are the extra elites REAL?                  -> geometry audit: glass crossing the
     detector/pupil plane, self-crossing left/right arcs. Neither environment's
     is_valid catches these, so a coverage win that is all broken geometry is not a win.

    PYTHONPATH=.:ns_probe python ns_probe/analyze_prism_sigma_sweep.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from nslens.qd import experiments

REPO = Path(__file__).resolve().parents[1]
ARMS = ("isotropic", "normalized")

#: "strong design" threshold per env -- the two fitnesses are on different scales
#: (focused throughput saturates near 1.0; the 3-field relay tops out near 0.65)
STRONG = {"prism_fold_qd": 0.8, "arvr_qd": 0.30}
DEEP_TIR = 4          # a design counts as a deep fold at >= 4 TIR bounces


def audit(env: str, cells: List[dict]) -> Dict[str, float]:
    """Per-archive metrics, including the geometry defects is_valid misses."""
    cfg = experiments.get(env)
    problem = cfg.problem
    lo = np.asarray(problem.lo, float)
    hi = np.asarray(problem.hi, float)
    names = list(problem.param_names)
    i_n = names.index("n")
    plane = problem.x_sensor          # detector for prism_fold, eye pupil for arvr

    g = np.array([c["genotype"] for c in cells], float)
    tol = 1e-3 * (hi - lo)
    on_rail = (g <= lo + tol) | (g >= hi - tol)

    past_plane = cross = 0
    for c in cells:
        p = problem.vec_to_params(c["genotype"])
        if problem.element(p).x_extent()[1] > plane:
            past_plane += 1
        # left and right arc control points swapped -> the body is inside out
        if p["thick"] + p["fsag"] + p["bsag"] < 0:
            cross += 1

    fits = np.array([c["fitness"] for c in cells], float)
    clean = [c for c, bad in zip(cells, [
        problem.element(problem.vec_to_params(c["genotype"])).x_extent()[1] > plane
        for c in cells]) if not bad]
    # the arvr archive reports its bounce count as "tir", the throughput envs as
    # "tir_count"; a design with neither counts as unfolded
    tir = np.array([c.get("tir_count", c.get("tir", 0)) for c in cells], float)
    return dict(
        cells=len(cells),
        qd=float(fits.sum()),
        coverage=len(cells) / (cfg.bc_bins[0] * cfg.bc_bins[1]),
        best=float(fits.max()),
        best_clean=float(max((c["fitness"] for c in clean), default=float("nan"))),
        n_strong=int((fits >= STRONG.get(env, 0.8)).sum()),
        deep=float(fits[tir >= DEEP_TIR].max()) if (tir >= DEEP_TIR).any() else float("nan"),
        n_rail=float(on_rail[:, i_n].mean()),
        n_std=float(g[:, i_n].std()),
        rail_all=float(on_rail.mean()),
        past_plane=past_plane / len(cells),
        self_cross=cross / len(cells),
    )


KEYS = [("cells", "filled cells", "{:.0f}"),
        ("qd", "QD score (sum fitness)", "{:.1f}"),
        ("coverage", "coverage", "{:.3f}"),
        ("best", "best fitness", "{:.4f}"),
        ("best_clean", "best fitness, glass clear of plane", "{:.4f}"),
        ("n_strong", "# strong designs", "{:.0f}"),
        ("deep", f"deep-fold reach (best fit @ tir>={DEEP_TIR})", "{:.4f}"),
        ("n_rail", "frac of elites with n on a rail", "{:.3f}"),
        ("n_std", "std(n) across elites", "{:.4f}"),
        ("rail_all", "frac of ALL genes on a rail", "{:.3f}"),
        ("past_plane", "frac with glass past the plane", "{:.3f}"),
        ("self_cross", "frac with self-crossing arcs", "{:.3f}")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "runs" / "prism_sigma_sweep"))
    args = ap.parse_args()
    root = Path(args.root)

    # directories are <env>__<arm-label>__seed<k>; the arm label carries its sigma0
    # when it is not the driver default (e.g. "normalized-sig0.05")
    found: Dict[str, Dict[str, List[Path]]] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "archive.json").exists():
            continue
        parts = d.name.split("__")
        if len(parts) != 3:
            continue
        found.setdefault(parts[0], {}).setdefault(parts[1], []).append(d)

    summary: Dict[str, Dict[str, List[dict]]] = {}
    for env, per_label in sorted(found.items()):
        print(f"\n{'='*100}\n{env}   (median over seeds; delta vs isotropic)\n{'='*100}")
        arms = sorted(per_label, key=lambda a: (a != "isotropic", a))
        per_arm = {a: [audit(env, json.loads((d / 'archive.json').read_text())["cells"])
                       for d in sorted(per_label[a])] for a in arms}
        summary[env] = per_arm

        med = {a: {k: (np.nanmedian([r[k] for r in per_arm[a]])
                       if per_arm[a] else float("nan")) for k, _, _ in KEYS}
               for a in arms}
        w = 20
        print(f"{'metric':<40}" + "".join(f"{a:>{w}}" for a in arms))
        print(f"{'':<40}" + "".join(f"{'n=' + str(len(per_arm[a])):>{w}}" for a in arms))
        base = "isotropic" if "isotropic" in arms else arms[0]
        for key, label, fmt in KEYS:
            row = f"{label:<40}"
            for a in arms:
                v, b = med[a][key], med[base][key]
                cell = fmt.format(v)
                if a != base and np.isfinite(v) and np.isfinite(b) and b:
                    cell += f" ({(v/b - 1)*100:+.0f}%)"
                row += f"{cell:>{w}}"
            print(row)

        # the median hides a one-seed breakthrough, which is exactly what a QD run is
        # for, so report the pooled best too
        for key, label in (("best", "BEST over all seeds"),
                           ("best_clean", "BEST over all seeds, glass clear")):
            row = f"{label:<40}"
            for a in arms:
                v = [r[key] for r in per_arm[a] if np.isfinite(r[key])]
                row += f"{(max(v) if v else float('nan')):>{w}.4f}"
            print(row)

    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {root/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
