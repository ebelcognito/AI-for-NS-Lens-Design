"""
ARCHIVE-WIDE REFINABILITY MAP. Run the FD-Jacobian + Levenberg-Marquardt gate
(refine_lm.py) on EVERY cell of a QD archive, with the honest no-erosion margin
guard, and summarize: per-cell spot headroom, throughput/margin change, whether the
topology stayed in-cell, and how refinability tracks dominant-energy share.

One persistent Node process for the whole sweep (not one per cell).

    python3 ns_probe/refine_sweep.py
    python3 ns_probe/refine_sweep.py --rays 81 --iters 40 --archive ns_probe/results_fold_qd/archive.json
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
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
import refine_lm as R

BEAM_W = 8.0


def refine_one(runner, cell, *, rays, iters, fd_step, opt_idx):
    p0 = dict(zip(R.PARAM_NAMES, [float(v) for v in cell["genotype"]]))
    try:
        res0 = R.trace(runner, p0, n_rays=rays, beam_w=BEAM_W)
    except Exception:
        return None
    if res0.get("error"):
        return None
    dom_sig, dom_e = R.dominant_signature(res0, p0)
    if dom_sig is None:
        return None
    recs0 = R.dom_records_by_dy(res0, p0, dom_sig)
    if not recs0:
        return None
    tracked = sorted(recs0.keys())
    w0 = np.array([recs0[dy][1] for dy in tracked])
    emitted = float(res0["energyLedger"]["emitted"])
    if emitted <= 0:
        return None

    obj = R.Objective(runner, p0, dom_sig, tracked, w0, n_rays=rays, beam_w=BEAM_W,
                      opt_idx=opt_idx, thru0=dom_e / emitted,
                      thru_slack=0.02, thru_weight=50.0)
    xn0 = (np.array([p0[n] for n in R.PARAM_NAMES]) - R.LO) / (R.HI - R.LO)
    info0 = obj.evaluate(xn0)
    if info0 is None:
        return None
    margin_floor = info0["min_margin"] - 0.25                 # no-erosion
    try:
        xnF, infoF, _hist, _J, _r = R.lm_optimize(
            obj, xn0, iters=iters, h=fd_step,
            margin_floor=margin_floor, max_step=0.1, verbose=False)
    except Exception:
        return None
    pF = obj.unpack(xnF)

    R._last_cx[0] = p0["cx"]; rA = R.trace(runner, p0, n_rays=21, beam_w=BEAM_W)
    sA = R.archive_style_spot(rA)
    R._last_cx[0] = pF["cx"]; rB = R.trace(runner, pF, n_rays=21, beam_w=BEAM_W)
    sB = R.archive_style_spot(rB)
    domF, _ = R.dominant_signature(rB, pF)
    margin, _crit, ntir = R.critical_angle_margins(rB, pF)

    headroom = (1.0 - sB / sA) if (sA and sA > 0 and np.isfinite(sB)) else 0.0
    return dict(
        cell=[int(cell["i0"]), int(cell["i1"])], tir=int(cell["tir_count"]),
        T=float(cell["T"]), dom_share=float(cell["dom_energy_share"]),
        dom_sig=R.format_signature(dom_sig), n_tracked=len(tracked),
        spot21_before=float(sA), spot21_after=float(sB), headroom=float(headroom),
        thru_before=float(info0["thru"]), thru_after=float(infoF["thru"]),
        margin_before=float(info0["min_margin"]), margin_after=float(margin),
        topo_preserved=bool(domF == dom_sig), n_tir=int(ntir))


def make_map(path, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[map] skipped ({e})"); return False
    ok = [r for r in rows if r and np.isfinite(r["spot21_before"])]
    tir = np.array([r["tir"] for r in ok])
    share = np.array([r["dom_share"] for r in ok])
    head = np.array([100 * r["headroom"] for r in ok])
    sb = np.array([r["spot21_before"] for r in ok])
    sa = np.array([r["spot21_after"] for r in ok])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sc = ax[0].scatter(share, head, c=tir, cmap="viridis", s=60, edgecolor="k", lw=.4)
    ax[0].set(title="refinability vs dominant-energy share",
              xlabel="dominant-signature share (QD champion)",
              ylabel="spot headroom from refinement (%)")
    ax[0].grid(alpha=.3); fig.colorbar(sc, ax=ax[0], label="# TIR bounces")

    lim = [min(sa.min(), sb.min()) * 0.7 + 1e-3, max(sa.max(), sb.max()) * 1.3]
    ax[1].plot(lim, lim, "k--", alpha=.5, label="no change")
    sc2 = ax[1].scatter(sb, sa, c=tir, cmap="viridis", s=60, edgecolor="k", lw=.4)
    ax[1].set(xscale="log", yscale="log", xlim=lim, ylim=lim,
              title="spot: before vs after refinement (21-ray, honest)",
              xlabel="spot BEFORE (QD champion)", ylabel="spot AFTER (refined)")
    ax[1].grid(alpha=.3, which="both"); ax[1].legend()
    fig.colorbar(sc2, ax=ax[1], label="# TIR bounces")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(REPO / "ns_probe" / "results_fold_qd" / "archive.json"))
    ap.add_argument("--rays", type=int, default=81)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--fd-step", type=float, default=1e-3)
    ap.add_argument("--params", default=",".join(R.DEFAULT_PARAMS))
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_refine_lm"))
    args = ap.parse_args()

    arc = json.loads(Path(args.archive).read_text())
    cells = sorted(arc["cells"], key=lambda c: (c["i0"], c["i1"]))
    opt_idx = np.array([R.PARAM_NAMES.index(n) for n in args.params.split(",")])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"sweeping {len(cells)} cells from {Path(args.archive).name} "
          f"({args.rays} rays, {args.iters} iters, no-erosion guard)")
    t0 = time.time()
    rows = []
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for k, c in enumerate(cells):
            r = refine_one(runner, c, rays=args.rays, iters=args.iters,
                           fd_step=args.fd_step, opt_idx=opt_idx)
            rows.append(r)
            if r:
                print(f"[{k+1:3d}/{len(cells)}] cell({r['cell'][0]},{r['cell'][1]}) "
                      f"tir={r['tir']} share={r['dom_share']:.2f}  "
                      f"spot {r['spot21_before']:7.3f} -> {r['spot21_after']:7.3f} "
                      f"({100*r['headroom']:+5.0f}%) thru {r['thru_after']:.2f} "
                      f"marg {r['margin_after']:4.1f} topo={'ok' if r['topo_preserved'] else 'BROKE'}",
                      flush=True)
            else:
                print(f"[{k+1:3d}/{len(cells)}] cell({c['i0']},{c['i1']}) -- skipped (no dominant topology)",
                      flush=True)

    good = [r for r in rows if r]
    dt = time.time() - t0
    hs = np.array([r["headroom"] for r in good if np.isfinite(r["spot21_before"])])
    print("-" * 70)
    print(f"refined {len(good)}/{len(cells)} cells in {dt:.0f}s")
    print(f"median spot headroom: {100*np.median(hs):.0f}%  | "
          f"cells >50% improved: {int((hs>0.5).sum())}/{len(hs)}  | "
          f"topology preserved: {sum(r['topo_preserved'] for r in good)}/{len(good)}")
    # refinability vs share, bucketed
    for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        b = [r for r in good if lo <= r["dom_share"] < hi]
        if b:
            mh = np.median([r["headroom"] for r in b])
            print(f"  dom-share [{lo:.1f},{hi:.1f}): {len(b):2d} cells, median headroom {100*mh:+.0f}%")

    summary = out / "refinability_map.json"
    summary.write_text(json.dumps(dict(
        archive=str(args.archive), config=vars(args),
        wall_seconds=dt, n_cells=len(cells), n_refined=len(good),
        rows=rows), indent=2, default=float))
    fig = out / "refinability_map.png"
    made = make_map(fig, rows)
    print(f"[summary] {summary}")
    if made:
        print(f"[map] {fig}")


if __name__ == "__main__":
    main()
