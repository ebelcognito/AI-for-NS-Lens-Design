"""Search a fold env with the index gene PINNED to a real glass, at a chosen field.

The archives all prefer n ~ 1.85-1.95, but that preference was never tested against a
manufacturability constraint: nothing in the search knows that n > 1.8 means S-LAH/N-LASF
rather than a crown you can mould. This pins n to a narrow band around a catalogue nd and
lets the other eight genes re-optimise around it, which is the question the offline
index-clamp cannot answer -- clamping n on a design tuned for n=1.81 breaks TIR at the
edge fields, but a design SEARCHED at n=1.51 can pay for the larger critical angle with
steeper face tilts.

    python ns_probe/run_pinned_index.py --nd 1.51 --half-fov 10 --seed 0 \
        --out runs/fold24_nd151/seed0

NOTE Vd / dispersion is NOT modelled. The engine carries one scalar index per design and
never splits a ray by wavelength, so an Abbe number cannot influence the search. Pinning
nd is meaningful; pinning Vd would be decoration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nslens.qd import experiments
from nslens.qd.driver import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="fold24_wide", help="experiment to pin")
    ap.add_argument("--nd", type=float, required=True, help="catalogue index to pin to")
    ap.add_argument("--band", type=float, default=0.002,
                    help="+-band around nd; a hard pin would make the normalized "
                         "emitter divide by a zero range")
    ap.add_argument("--half-fov", type=float, default=8.0,
                    help="half-angle in deg; 9 fields are spread over +-this")
    ap.add_argument("--evals", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--det-clearance", type=float, default=None,
                    help="mm of clear space the body must leave in front of the "
                         "detector. FoldProblem has never enforced this, so designs "
                         "whose glass passes THROUGH the sensor are archived and their "
                         "thinness_T is measured against a plane they intersect. "
                         "12 matches the gap PrismFoldProblem has always required.")
    ap.add_argument("--constrained", action="store_true",
                    help="relative_sag + max_sag_ratio, i.e. fold24_clean's geometry: "
                         "a bulb becomes unreachable by construction instead of being "
                         "rejected after the engine has already traced it")
    ap.add_argument("--clean-gate", action="store_true",
                    help="require clean_pack (converging bundle, no internal focus) "
                         "during the SEARCH rather than as an offline re-score")
    ap.add_argument("--gate-all-fields", action="store_true",
                    help="demand clean_pack at every field. Known to be too harsh from "
                         "a cold start -- pair with --resume-from.")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = experiments.get(a.base)
    p = cfg.problem
    # copy: experiments.get() hands back a fresh config but the arrays are still shared
    p.lo, p.hi = p.lo.copy(), p.hi.copy()
    p.lo[-1], p.hi[-1] = a.nd - a.band, a.nd + a.band      # n is the last gene

    p.det_clearance = a.det_clearance
    if a.constrained:
        # the sag genes become fractions of their own face's half-chord, so the box has
        # to change units too -- +-0.5 of a half-chord, not +-18 mm
        p.relative_sag = True
        p.max_sag_ratio = 0.55
        p.lo[4], p.hi[4] = -0.5, 0.5      # fsag
        p.lo[5], p.hi[5] = -0.5, 0.5      # bsag
    cfg.require_monotone_convergence = a.clean_gate
    cfg.convergence_gate = "clean_pack"
    cfg.gate_all_fields = a.gate_all_fields
    # fold24_clean carries a resume_from pointing at high-index elites; those are outside
    # a pinned band and would seed every emitter from genotypes the box cannot represent
    cfg.resume_from = a.resume_from

    # 9 fields over the requested cone, so the cost per evaluation is unchanged and the
    # worst-field fitness is measured at the edge the user actually cares about
    fields = np.linspace(-a.half_fov, a.half_fov, 9)
    cfg.fields = tuple(round(float(f), 4) for f in fields)
    # field_accept is a HALF-angle, so it cannot exceed half_fov; bin over what is
    # attainable rather than over the descriptor's registered 0-20
    cfg.bc_bounds = (cfg.bc_bounds[0], (0.0, a.half_fov))
    cfg.bc_bins = (cfg.bc_bins[0], 5)
    cfg.seed = a.seed
    cfg.evals = a.evals
    cfg.trace = True
    cfg.label = f"{a.base}_nd{a.nd:.3f}_fov{a.half_fov:g}"
    cfg.meta_extra = dict(pinned_nd=a.nd, n_band=[p.lo[-1], p.hi[-1]],
                          half_fov_deg=a.half_fov, dispersion_modelled=False,
                          det_clearance=a.det_clearance, constrained_sag=a.constrained,
                          searched_clean_gate=a.clean_gate,
                          gate_all_fields=a.gate_all_fields)
    run(cfg, Path(a.out))


if __name__ == "__main__":
    main()
