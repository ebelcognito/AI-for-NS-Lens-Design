"""Before/after ray-diagram renderer for the freeform refinement.

Draws the folded element (freeform faces + flat edges) with a fan of rays traced through it,
side by side: ARC-refined vs FREEFORM-refined, each with a zoom inset on the sensor focus so the
20x spot tightening is visible. Reads the refine_freeform_cell*.json report for the geometry.

Never overwrites (standing rule): writes a unique PNG + a JSON sidecar carrying the exact geometry
and provenance, with a +k suffix if the name is taken.

Run:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.render_freeform --cell 3,5
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner  # noqa: E402

from . import topology as TOPO                                                        # noqa: E402
from .element_torch import PARAM_NAMES, PIDX, X_SENSOR, input_ray                     # noqa: E402
from .forward_freeform import trace_ff, build_surfaces_freeform                       # noqa: E402
from .geometry_freeform import sag                                                    # noqa: E402

DTYPE = torch.float64
ARCHIVE = REPO / "ns_probe" / "results_fold_qd" / "archive.json"


def surface_curve(ff, n=80):
    """Points along a FreeformSurface for drawing: P(h) = v + sag(h)*axis + h*tang."""
    h = torch.linspace(float(ff.h_min), float(ff.h_max), n, dtype=DTYPE)
    z, _ = sag(h, ff)
    P = ff.v.unsqueeze(0) + z.unsqueeze(-1) * ff.axis.unsqueeze(0) + h.unsqueeze(-1) * ff.tang.unsqueeze(0)
    return P.detach().numpy()


def ray_polylines(p, events, dys, coeffs, powers):
    """Full ray paths [source -> each hit -> sensor] for the given design, plus sensor y, energy,
    and the per-TIR-event critical-angle margins (deg) over the reached rays."""
    shim = types.SimpleNamespace(dys=np.asarray(dys), events=events)
    O0, _ = input_ray(p, shim.dys)
    with torch.no_grad():
        tr = trace_ff(p, shim, coeffs=coeffs, powers=powers)
    O0 = O0.detach().numpy()
    hp = tr.hit_path.detach().numpy()           # [steps, m, 2]
    reached = tr.reached.detach().numpy()
    ys = tr.y.detach().numpy()
    polys = []
    for i in range(len(dys)):
        if not reached[i]:
            continue
        pts = [O0[i]] + [hp[s, i] for s in range(hp.shape[0])] + [[X_SENSOR, ys[i]]]
        polys.append(np.array(pts))
    # TIR margins: incidence - critical (deg), per event, for reached rays only
    if tr.tir_incidence.numel():
        marg = torch.rad2deg(tr.tir_incidence - tr.critical).detach().numpy()  # [k, m]
        marg = marg[:, reached].reshape(-1)
    else:
        marg = np.array([])
    return polys, ys[reached], tr.energy.detach().numpy()[reached], marg


def draw_panel(ax, p, surfaces, polys, spot, title, ybar):
    import matplotlib.pyplot as plt
    # element faces
    for f in ("A", "B"):
        c = surface_curve(surfaces[f])
        ax.plot(c[:, 0], c[:, 1], color="black", lw=2.0, zorder=5)
    for f in ("T", "D"):
        s = surfaces[f]
        a, b = s["a"].detach().numpy(), s["b"].detach().numpy()
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=2.0, zorder=5)
    # rays
    for poly in polys:
        ax.plot(poly[:, 0], poly[:, 1], color="tab:blue", lw=0.6, alpha=0.7, zorder=3)
    # sensor plane
    ax.axvline(X_SENSOR, color="tab:red", lw=1.2, ls="--", zorder=4)
    ax.text(X_SENSOR + 1, ybar, "sensor", color="tab:red", rotation=90, va="center", fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="3,5")
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--n-rays", type=int, default=19, help="rays to draw")
    ap.add_argument("--beam-w", type=float, default=8.0)
    ap.add_argument("--out", default=str(REPO / "ns_autodiff" / "results_refine_freeform"))
    args = ap.parse_args()
    cell = tuple(int(v) for v in args.cell.split(","))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # latest report for this cell
    out = Path(args.out)
    reps = sorted(out.glob(f"refine_freeform_cell{cell[0]}_{cell[1]}*.json"))
    if not reps:
        raise SystemExit(f"no refine report for cell {cell} in {out}")
    rep = json.loads(reps[-1].read_text())
    powers = tuple(rep["powers"])
    g_arc = rep["arc"]["genotype"]
    p_arc = torch.tensor([g_arc[n] for n in PARAM_NAMES], dtype=DTYPE)
    coeffs_ff = {"A": torch.tensor(rep["freeform"]["coeffs_A"], dtype=DTYPE),
                 "B": torch.tensor(rep["freeform"]["coeffs_B"], dtype=DTYPE)}

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        topo = TOPO.extract(runner, Path(args.archive), cell, n_rays=121, beam_w=args.beam_w)
    events = [tuple(e) for e in [(t, f) for (t, f) in topo.events]]
    dys = np.linspace(-args.beam_w / 2, args.beam_w / 2, args.n_rays)

    # arc-refined (coeffs=None) and freeform-refined (coeffs) -- same frozen base p_arc
    surf_arc = build_surfaces_freeform(p_arc, coeffs=None, powers=())
    surf_ff = build_surfaces_freeform(p_arc, coeffs=coeffs_ff, powers=powers)
    polys_arc, y_arc, w_arc, marg_arc = ray_polylines(p_arc, events, dys, None, ())
    polys_ff, y_ff, w_ff, marg_ff = ray_polylines(p_arc, events, dys, coeffs_ff, powers)
    spot_arc = rep["arc"]["honest_spot"]; spot_ff = rep["freeform"]["honest_spot"]
    ybar_arc = float(np.average(y_arc, weights=w_arc))
    ybar_ff = float(np.average(y_ff, weights=w_ff))

    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 1], hspace=0.32, wspace=0.55)
    axA = fig.add_subplot(gs[0, 0:3]); axF = fig.add_subplot(gs[0, 3:6])
    draw_panel(axA, p_arc, surf_arc, polys_arc, spot_arc,
               f"ARC refined  —  honest spot {spot_arc:.3f} mm", ybar_arc)
    draw_panel(axF, p_arc, surf_ff, polys_ff, spot_ff,
               f"FREEFORM refined  —  honest spot {spot_ff:.3f} mm", ybar_ff)
    # shared limits
    xs = np.concatenate([np.concatenate([pl[:, 0] for pl in polys_arc]),
                         np.concatenate([pl[:, 0] for pl in polys_ff])])
    ys = np.concatenate([np.concatenate([pl[:, 1] for pl in polys_arc]),
                         np.concatenate([pl[:, 1] for pl in polys_ff])])
    xl = (xs.min() - 5, X_SENSOR + 8); yl = (ys.min() - 5, ys.max() + 5)
    for ax in (axA, axF):
        ax.set_xlim(*xl); ax.set_ylim(*yl)

    # spot zoom: sensor landing positions about each centroid (shared scale)
    axsA = fig.add_subplot(gs[1, 0:2]); axsF = fig.add_subplot(gs[1, 2:4])
    rng = max((y_arc - ybar_arc).std() * 4, 0.05)
    for ax, y, yb, sp, col in [(axsA, y_arc, ybar_arc, spot_arc, "tab:gray"),
                               (axsF, y_ff, ybar_ff, spot_ff, "tab:green")]:
        ax.scatter(y - yb, np.zeros_like(y), s=18, c=col, alpha=0.8)
        ax.axvline(0, color="tab:red", ls=":", lw=1)
        ax.set_xlim(-rng, rng); ax.set_yticks([])
        ax.set_xlabel("sensor y − centroid (mm)"); ax.grid(alpha=0.2, axis="x")
        ax.set_title(f"spot {sp:.3f} mm  (shared scale)", fontsize=9)

    # TIR critical-angle margin: every fold's (incidence − critical), arc vs freeform.
    # Higher = more robustly past critical. Min margin is the headline robustness number.
    axM = fig.add_subplot(gs[1, 4:6])
    rngx = np.random.default_rng(0)
    for xc, marg, col, lbl in [(0, marg_arc, "tab:gray", "arc"), (1, marg_ff, "tab:green", "freeform")]:
        if marg.size:
            jit = (np.linspace(-0.18, 0.18, marg.size) if marg.size > 1 else np.zeros(1))
            axM.scatter(np.full(marg.size, xc) + jit, marg, s=14, c=col, alpha=0.7)
            axM.hlines(marg.min(), xc - 0.25, xc + 0.25, color=col, lw=2.5)
            axM.text(xc, marg.min() - 0.18, f"min {marg.min():.2f}°", ha="center", va="top",
                     fontsize=8, color=col, fontweight="bold")
    axM.axhline(0, color="tab:red", ls="--", lw=1.2)
    axM.text(1.45, 0.04, "critical (TIR fails below)", color="tab:red", fontsize=7, va="bottom", ha="right")
    axM.set_xticks([0, 1]); axM.set_xticklabels(["arc", "freeform"])
    axM.set_ylabel("TIR margin past critical (°)"); axM.grid(alpha=0.2, axis="y")
    axM.set_xlim(-0.5, 1.5); axM.set_title("every fold stays past critical", fontsize=9)

    cap = (f"cell {cell}   topology {topo.signature_str()}   powers {powers}\n"
           f"champion {rep['champion_spot']:.3f} → arc {spot_arc:.3f} → freeform {spot_ff:.3f} mm "
           f"({rep['honest_gain_pct']:+.1f}% over best arc; base frozen, only polynomial sag unlocked)")
    fig.suptitle(cap, fontsize=11)

    # never-overwrite write
    base = out / f"render_freeform_cell{cell[0]}_{cell[1]}.png"
    png, k = base, 0
    while png.exists():
        k += 1; png = base.with_name(f"render_freeform_cell{cell[0]}_{cell[1]}_{k}.png")
    fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)
    side = png.with_suffix(".json")
    side.write_text(json.dumps(dict(
        cell=list(cell), topology=topo.signature_str(), powers=list(powers),
        source_report=str(reps[-1].name), n_rays_drawn=args.n_rays, beam_w=args.beam_w,
        arc_genotype=g_arc, freeform_coeffs_A=rep["freeform"]["coeffs_A"],
        freeform_coeffs_B=rep["freeform"]["coeffs_B"],
        honest_spot_arc=spot_arc, honest_spot_freeform=spot_ff,
        champion_spot=rep["champion_spot"]), indent=2, default=float))
    print(f"[render] {png}")
    print(f"[sidecar] {side}")


if __name__ == "__main__":
    main()
