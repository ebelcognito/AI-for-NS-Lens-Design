"""Close the loop: take folds the freeform SEARCH discovered (coarse), FREEZE each one's
discovered topology, and POLISH the geometry in the differentiable twin -- then render a
gallery of the refined, high-quality folds.

This is the full pipeline end to end:
    SEARCH (discover, no freeze)  ->  FREEZE the discovered order  ->  REFINE (gradient)
The discovered topology is frozen straight from the validated discovering tracer (trace_discover),
so no JS oracle is needed -- the twin is the forward model throughout.

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.refine_discovered --depths 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import topology as TOPO
from .refine_freeform import TwinObjectiveFF, honest_dense, lm_refine_ff
from .render_search_gallery import draw
from .search_freeform import (BEAM_W, U_SCALE, evaluate, trace_discover)

DTYPE = torch.float64
BASE9 = ["ray_y", "ang", "cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b", "n"]
REFINE_BASE = ["cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b", "n"]   # field (ray_y,ang) fixed
POWERS = (2, 4)


def build_frozen_topology(x, *, n_rays=121, beam_w=BEAM_W):
    """Freeze the dominant interaction order of design x straight from the discovering tracer."""
    rays = trace_discover(x, n_rays=n_rays, beam_w=beam_w)
    dys = np.linspace(-beam_w / 2, beam_w / 2, n_rays)
    fwd = [(i, r) for i, r in enumerate(rays) if r["forward"] and r["b"] > 0]
    by = {}
    for i, r in fwd:
        sig = tuple((t, f) for (t, f, *_z) in r["seq"]); by[sig] = by.get(sig, 0.0) + r["b"]
    if not by:
        return None
    dom = max(by, key=by.get)
    keep = [(i, r) for i, r in fwd if tuple((t, f) for (t, f, *_z) in r["seq"]) == dom]
    if len(keep) < 5:
        return None
    tdys = np.array([dys[i] for i, _r in keep])
    oy = np.array([r["y"] for _i, r in keep]); ob = np.array([r["b"] for _i, r in keep])
    p0 = dict(zip(BASE9, [float(v) for v in x[:9]]))
    return TOPO.FrozenTopology(
        cell=(-1, -1), p0=p0, events=[(t, f) for (t, f) in dom],
        dys=tdys, oracle_y=oy, oracle_b=ob, w0=ob.copy(),
        dom_share=float(by[dom] / sum(by.values())), archive_spot=float("nan"),
        meta=dict(emitted=float(n_rays), dom_energy=float(by[dom]), n_tracked=len(keep),
                  tir_count=sum(1 for t, _f in dom if t == "tir")))


def refine_one(x, *, iters=25, reg=2e-2):
    """Freeze x's topology and jointly polish base+aspheric (order-preserving). Returns the
    refined 13-vector and (start_spot, refined_spot) honest dense spots."""
    topo = build_frozen_topology(x)
    if topo is None:
        return None
    u0 = np.asarray(x[9:], float)                         # discovered scaled coeffs
    thru0 = float(np.mean(np.array([oy for oy in topo.oracle_b])))  # placeholder; set below
    # twin throughput at start (consistent floor reference)
    obj0 = TwinObjectiveFF(topo, REFINE_BASE, POWERS, thru0=1.0, slack=0.02, mu=50.0, reg=reg, u0=u0)
    with torch.no_grad():
        from .forward_freeform import trace_ff
        thru0 = float(trace_ff(torch.tensor([topo.p0[n] for n in BASE9], dtype=DTYPE), topo,
                               coeffs={"A": torch.tensor(u0[:2] * U_SCALE),
                                       "B": torch.tensor(u0[2:] * U_SCALE)}, powers=POWERS).energy.mean())
    obj = TwinObjectiveFF(topo, REFINE_BASE, POWERS, thru0=thru0, slack=0.02, mu=50.0, reg=reg, u0=u0)
    start_spot, _, _ = honest_dense(obj, obj.x0())
    x_ref, _hist, best_honest = lm_refine_ff(obj, iters=iters, preserve_order=True,
                                             tir_eps=0.05, verbose=False, tag="ref")
    p_ref, coeffs_ref = obj.params_from(x_ref)
    u_ref = x_ref[obj.nb:].detach().numpy()              # scaled coeffs back to genotype form
    x13 = np.concatenate([p_ref.detach().numpy(), u_ref])
    return dict(x13=x13, start_spot=float(start_spot), refined_spot=float(best_honest))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="ns_autodiff/results_search_freeform/archive_freeform.json")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--out", default="ns_autodiff/results_search_freeform")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arch = json.loads(Path(args.archive).read_text())
    good = [c for c in arch["cells"]
            if c.get("efficiency", 0) >= 0.7 and c.get("n_forward", 0) >= 14 and c["spot"] < 8.0]
    picks = []
    for d in args.depths:
        cand = [c for c in good if int(c["tir_count"]) == d]
        if cand:
            picks.append(min(cand, key=lambda c: c["spot"]))

    results = []
    for c in picks:
        x = np.array(c["genotype"], float)
        print(f"[refine] {int(c['tir_count'])}-TIR  discovered spot {c['spot']:.3f} ...", flush=True)
        rr = refine_one(x, iters=args.iters)
        if rr is None:
            print("   (skipped: topology not freezable)"); continue
        rr["tir"] = int(c["tir_count"]); rr["sig"] = c["dom_signature"]; rr["x_disc"] = x
        gain = 100 * (1 - rr["refined_spot"] / rr["start_spot"]) if rr["start_spot"] > 0 else 0
        print(f"   honest spot {rr['start_spot']:.3f} -> {rr['refined_spot']:.3f}  ({gain:+.1f}%)", flush=True)
        if gain > 5.0:                                    # keep only folds that meaningfully polished
            results.append(rr)
        else:
            print("   (skipped from gallery: <5% gain)")

    if not results:
        raise SystemExit("nothing refined")
    nrow = len(results)
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 3.6 * nrow))
    axes = np.atleast_2d(axes)
    for row, rr in enumerate(results):
        draw(axes[row, 0], rr["x_disc"], f"{rr['tir']}-TIR  DISCOVERED  spot {rr['start_spot']:.3f} mm")
        draw(axes[row, 1], rr["x13"], f"{rr['tir']}-TIR  REFINED  spot {rr['refined_spot']:.3f} mm")
    fig.suptitle("Full pipeline: search-discovered freeform folds, then freeze→refine (polished)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out); base = out / "gallery_refined.png"; png, k = base, 0
    while png.exists():
        k += 1; png = base.with_name(f"gallery_refined_{k}.png")
    fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)
    png.with_suffix(".json").write_text(json.dumps(
        [dict(tir=r["tir"], start_spot=r["start_spot"], refined_spot=r["refined_spot"],
              signature=r["sig"], genotype=[float(v) for v in r["x13"]]) for r in results],
        indent=2))
    print(f"[gallery] {png}")


if __name__ == "__main__":
    main()
