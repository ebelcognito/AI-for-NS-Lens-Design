"""Archive-wide autodiff refinement sweep -- the differentiable analogue of
ns_probe/refine_sweep.py.

For every QD archive cell: extract the frozen topology, refine with autodiff Levenberg-
Marquardt (analytic Jacobian, no-erosion guards), then RE-VALIDATE through the JS oracle at
QD's 21-ray density (the honest, apples-to-apples spot) and check the dominant topology is
preserved. Tabulates per-cell results + a summary broken down by fold depth and dominant
share, mirroring the refine_lm sweep so the two refiners can be compared directly.

One persistent Node process is shared across all cells. Never-overwrites a JSON map + a
2-panel refinability figure under ns_autodiff/results_refine_autodiff/.

Usage:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.refine_sweep --rays 81 --iters 30           # full archive
    $PY -m ns_autodiff.refine_sweep --limit 12                     # quick subset (first 12)
    $PY -m ns_autodiff.refine_sweep --cells 3,5 4,11 3,10 0,1      # specific cells
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner  # noqa: E402
import fold_demo as FD                                                                # noqa: E402

from . import topology as TOPO                                                        # noqa: E402
from .element_torch import PARAM_NAMES, PIDX                                          # noqa: E402
from .forward import trace                                                            # noqa: E402
from .refine_autodiff import (DEFAULT_PARAMS, TwinObjective, lm_refine,               # noqa: E402
                              oracle_archive_spot, oracle_dom_signature,
                              oracle_tir_margin)

ARCHIVE = REPO / "ns_probe" / "results_fold_qd" / "archive.json"
DTYPE = torch.float64


def refine_cell(runner, cell, opt_names, *, rays, beam_w, iters):
    """Refine one cell end-to-end; return a result row (status='ok' or a failure reason)."""
    row = dict(cell=list(cell))
    try:
        topo = TOPO.extract(runner, ARCHIVE, cell, n_rays=rays, beam_w=beam_w)
    except SystemExit as e:
        row.update(status="extract_fail", reason=str(e)); return row
    row.update(tir=topo.meta["tir_count"], dom_share=topo.dom_share,
               archive_spot=topo.archive_spot, topology=topo.signature_str(),
               n_tracked=topo.meta["n_tracked"])
    try:
        p0 = torch.tensor([topo.p0[n] for n in PARAM_NAMES], dtype=DTYPE)
        with torch.no_grad():
            thru0 = float(trace(p0, topo).energy.mean())
        obj = TwinObjective(topo, opt_names, thru0=thru0, slack=0.02, mu=50.0)
        diag0 = obj.diagnostics(obj.base_xn[obj.opt_idx])
        margin_floor = (diag0["margin"] - 0.25) if not np.isnan(diag0["margin"]) else -1e9
        xnF, history, _J, _r = lm_refine(obj, iters=iters, margin_floor=margin_floor,
                                         verbose=False)
        pF = obj.params_from(xnF).detach().numpy()
        p0_np = obj.params_from(obj.base_xn[obj.opt_idx]).detach().numpy()
        p0_d = dict(zip(PARAM_NAMES, [float(v) for v in p0_np]))
        pF_d = dict(zip(PARAM_NAMES, [float(v) for v in pF]))
        resA, _ = runner.simulate(FD.scene(p0_d, n_rays=21, beam_w=beam_w, record=True))
        resB, _ = runner.simulate(FD.scene(pF_d, n_rays=21, beam_w=beam_w, record=True))
        spotA = oracle_archive_spot(resA, p0_d["cx"])
        spotB = oracle_archive_spot(resB, pF_d["cx"])
        same = (oracle_dom_signature(resA, p0_d["cx"]) == oracle_dom_signature(resB, pF_d["cx"]))
        marginB, crit = oracle_tir_margin(resB, pF_d["n"])
        imp = (100.0 * (1 - spotB / spotA)) if (spotA and np.isfinite(spotA) and spotA > 0) else float("nan")
        kappas = [h["kappa"] for h in history if h.get("kappa")]
        row.update(status="ok",
                   twin_spot_start=history[0]["spot"], twin_spot_refined=history[-1]["spot"],
                   oracle_spot21_start=spotA, oracle_spot21_refined=spotB,
                   honest_improvement_pct=imp,
                   throughput_start=history[0]["thru"], throughput_refined=history[-1]["thru"],
                   topology_preserved=bool(same),
                   margin_start=diag0["margin"], margin_refined=marginB,
                   kappa_max=(max(kappas) if kappas else None), iters_run=len(history) - 1)
    except Exception as e:  # noqa: BLE001 -- a single bad cell must not kill the sweep
        row.update(status="error", reason=f"{type(e).__name__}: {e}")
    return row


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("honest_improvement_pct", np.nan))]
    print("\n" + "=" * 78)
    print(f"SUMMARY  ({len(ok)}/{len(rows)} cells refined & scored)")
    print("=" * 78)
    if not ok:
        print("  no scorable cells."); return {}
    imp = np.array([r["honest_improvement_pct"] for r in ok])
    pres = np.array([r["topology_preserved"] for r in ok])
    print(f"  median honest spot improvement : {np.median(imp):+.1f} %")
    print(f"  improved > 50%                 : {int((imp > 50).sum())}/{len(ok)}")
    print(f"  improved (any, >1%)            : {int((imp > 1).sum())}/{len(ok)}")
    print(f"  got WORSE (honest)             : {int((imp < -1).sum())}/{len(ok)}")
    print(f"  topology preserved             : {int(pres.sum())}/{len(ok)}")
    print("\n  by fold depth (tir):")
    print(f"    {'tir':>4} {'n':>4} {'median imp%':>12} {'topo kept':>10}")
    for tir in sorted({r["tir"] for r in ok}):
        sub = [r for r in ok if r["tir"] == tir]
        si = np.array([r["honest_improvement_pct"] for r in sub])
        sp = sum(r["topology_preserved"] for r in sub)
        print(f"    {tir:>4} {len(sub):>4} {np.median(si):>+12.1f} {sp:>6}/{len(sub)}")
    print("\n  by dominant share:")
    for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        sub = [r for r in ok if lo <= r["dom_share"] < hi]
        if sub:
            si = np.array([r["honest_improvement_pct"] for r in sub])
            print(f"    [{lo:.1f},{hi:.1f}) n={len(sub):>3}  median imp {np.median(si):+.1f} %")
    # standouts
    best = sorted(ok, key=lambda r: -r["honest_improvement_pct"])[:5]
    print("\n  top improvements (honest 21-ray spot):")
    for r in best:
        print(f"    cell {tuple(r['cell'])}: {r['oracle_spot21_start']:.3f} -> "
              f"{r['oracle_spot21_refined']:.3f} ({r['honest_improvement_pct']:+.0f}%) "
              f"tir={r['tir']} topo={'kept' if r['topology_preserved'] else 'BROKE'}")
    return dict(median_imp=float(np.median(imp)),
                n_improved_50=int((imp > 50).sum()), n_worse=int((imp < -1).sum()),
                n_topo_preserved=int(pres.sum()), n_scored=len(ok))


def make_map_figure(path, rows):
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("honest_improvement_pct", np.nan))]
    if not ok:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure] skipped ({e})"); return False
    tir = np.array([r["tir"] for r in ok])
    imp = np.array([r["honest_improvement_pct"] for r in ok]).clip(-100, 100)
    pres = np.array([r["topology_preserved"] for r in ok])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    jit = (np.random.default_rng(0).random(len(tir)) - 0.5) * 0.3 if False else 0.0  # noqa
    ax[0].scatter(tir[pres], imp[pres], c="tab:green", label="topology kept", alpha=.7)
    ax[0].scatter(tir[~pres], imp[~pres], c="tab:red", marker="x", label="topology broke", alpha=.7)
    ax[0].axhline(0, ls=":", color="grey")
    ax[0].set(title="autodiff refinement: honest spot improvement vs fold depth",
              xlabel="TIR count", ylabel="honest 21-ray spot improvement (%)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    buckets = sorted({int(t) for t in tir})
    med = [np.median(imp[tir == t]) for t in buckets]
    ax[1].bar([str(t) for t in buckets], med, color="tab:blue")
    ax[1].axhline(0, ls=":", color="grey")
    ax[1].set(title="median honest improvement by fold depth", xlabel="TIR count",
              ylabel="median improvement (%)"); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", type=int, default=81)
    ap.add_argument("--beam-w", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--params", default=",".join(DEFAULT_PARAMS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N cells")
    ap.add_argument("--cells", nargs="*", default=None, help="explicit i0,i1 cells")
    ap.add_argument("--out", default=str(REPO / "ns_autodiff" / "results_refine_autodiff"))
    args = ap.parse_args()
    opt_names = [s for s in args.params.split(",") if s]

    arc = json.loads(ARCHIVE.read_text())
    cells = [(int(c["i0"]), int(c["i1"])) for c in arc["cells"]]
    if args.cells:
        want = {tuple(int(v) for v in c.split(",")) for c in args.cells}
        cells = [c for c in cells if c in want]
    if args.limit:
        cells = cells[:args.limit]

    print(f"AUTODIFF REFINEMENT SWEEP over {len(cells)} cells "
          f"(rays={args.rays}, iters={args.iters}, dof={opt_names})")
    print("-" * 78)
    rows = []
    t0 = time.time()
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for k, cell in enumerate(cells):
            tc = time.time()
            row = refine_cell(runner, cell, opt_names, rays=args.rays,
                              beam_w=args.beam_w, iters=args.iters)
            rows.append(row)
            if row["status"] == "ok":
                print(f"[{k+1:>3}/{len(cells)}] cell {str(cell):<9} tir={row['tir']} "
                      f"honest {row['oracle_spot21_start']:.3f}->{row['oracle_spot21_refined']:.3f} "
                      f"({row['honest_improvement_pct']:+.0f}%) "
                      f"topo={'kept' if row['topology_preserved'] else 'BROKE'} "
                      f"({time.time()-tc:.1f}s)", flush=True)
            else:
                print(f"[{k+1:>3}/{len(cells)}] cell {str(cell):<9} {row['status']}: "
                      f"{row.get('reason','')[:50]}", flush=True)
    dt = time.time() - t0
    stats = summarize(rows)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = out / "refine_sweep_autodiff.json"
    p, j = base, 0
    while p.exists():
        j += 1; p = base.with_name(f"refine_sweep_autodiff_{j}.json")
    p.write_text(json.dumps(dict(config=vars(args), wall_seconds=dt, summary=stats,
                                 rows=rows), indent=2, default=float))
    fig = p.with_suffix(".png")
    made = make_map_figure(fig, rows)
    print(f"\n[map] {p}  ({dt:.0f}s total)")
    if made:
        print(f"[figure] {fig}")


if __name__ == "__main__":
    main()
