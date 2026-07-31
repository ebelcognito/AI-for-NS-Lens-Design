"""
Run the SINGLE-ELEMENT NS folding-emergence probe (CMA-ME) and save the archive.

Same descriptors / fitness / archive as run.py, but one element with per-face reflectivity.
Output goes to a SEPARATE directory so the 2-element results are not overwritten.

    python3 ns_probe/run_single.py --evals 15000 --seed 0 --out ns_probe/results_single
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ns_probe import probe, probe_single
from ns_probe.cma_me import CMA_ME, GridArchive
from ns_probe.probe import Evaluator, calibrate_b_in
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

BC_BINS = (24, 12)
BC_BOUNDS = ((10.0, 120.0), (0.0, 6.0))   # (device depth T; mean axial reversals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=5)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--flux-floor", type=float, default=0.08)
    ap.add_argument("--out", type=str, default=str(REPO / "ns_probe" / "results_single"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    with PersistentRayOpticsRunner(timeout_s=12) as runner:
        b_in = calibrate_b_in(runner)
        print(f"[probe-single] B_in = {b_in:.3f}", flush=True)
        ev = Evaluator(runner, b_in, flux_floor=args.flux_floor,
                       scene_builder=probe_single.build_scene)

        archive = GridArchive(BC_BINS, BC_BOUNDS)
        lo, hi = probe_single.bounds_lo_hi()
        algo = CMA_ME(archive, ev, sample_genotype=probe_single.sample_genotype,
                      bounds=(lo, hi), n_emitters=args.emitters, sigma0=args.sigma0,
                      popsize=args.popsize, seed=args.seed)

        def logger(rec):
            el = time.time() - t_start
            print(f"[probe-single] evals={rec['evals']:6d} filled={rec['filled']:4d} "
                  f"cov={rec['coverage']*100:5.1f}% best_fit={rec['best']:8.3f} "
                  f"({el:6.1f}s, {ev.n_errors} sim-errs)", flush=True)

        algo.run(args.evals, log_every=1000, logger=logger)

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), T=float(c.bc[0]), mean_reversals=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(x) for x in c.genotype])
        for k, val in c.info.items():
            if k not in rec:
                rec[k] = float(val) if isinstance(val, (int, float, np.floating)) else val
        cells.append(rec)

    payload = dict(
        meta=dict(variant="single_element", evals=int(algo.n_evals), seed=int(args.seed),
                  b_in=float(b_in), flux_floor=float(args.flux_floor), bc_bins=BC_BINS,
                  bc_bounds=BC_BOUNDS,
                  bc_axes=["device_depth_T (thin=small)", "mean_axial_reversals (folded>=2)"],
                  dim=int(probe_single.DIM), n_elem=1,
                  sim_errors=int(ev.n_errors), sim_calls=int(ev.n_calls),
                  wall_seconds=float(time.time() - t_start)),
        history=algo.history, cells=cells,
    )
    (out / "archive.json").write_text(json.dumps(payload, indent=2))
    print(f"[probe-single] wrote {out/'archive.json'} ({len(cells)} cells, "
          f"{algo.n_evals} evals, {payload['meta']['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
