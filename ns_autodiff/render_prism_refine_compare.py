"""Before/after gallery of the differentiable PRISM refinement (results_refine_prism_ff/).

Renders each refined cell as a START | REFINED pair in the OLD ray-optics theme (black ground,
translucent grey glass, glowing gold relay rays, red display squares, white dashed pupil) -- NOT
the plain-white matplotlib look of render_prism_gallery. The refined faces are freeform
(parametric-arc aspheric) which the JS engine can't trace, so we draw the exact curve in
matplotlib but match the old aesthetic. y-axis inverted so the display sits at the bottom, as in
the original engine renders.

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.render_prism_refine_compare           # -> prism_refine_gallery_LATEST.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .prism_branch import (PUPIL_X, SOURCE_PTS, SOURCE_AIM, FAN_HALFANGLE, EYEBOX_HALF,
                           MAX_DEPTH, build_arcs_ff, nearest_hit_ff, _unwrap, _delta)

REPODIR = Path(__file__).resolve().parents[1]
INDIR = REPODIR / "ns_autodiff" / "results_refine_prism_ff"

# old-theme palette (matched to ns_probe/results_arvr_qd/*.png)
BG = "black"
GLASS = "#9a9a9a"
RAY = "#ece24a"          # glowing gold
SRC = "#ff2d2d"          # red display squares
PUPIL = "#dddddd"


def face_curve(a, n=80):
    ce = a["center"]
    th1 = _unwrap(math.atan2(a["c1"][1] - ce[1], a["c1"][0] - ce[0]), a["th_apex"])
    th2 = _unwrap(math.atan2(a["c2"][1] - ce[1], a["c2"][0] - ce[0]), a["th_apex"])
    th = np.linspace(th1, th2, n)
    s = (th - a["th_apex"]) / a["half_span"]
    R = a["r"] + np.array([_delta(si, a["coeffs"])[0] for si in s])
    return np.column_stack([ce[0] + R * np.cos(th), ce[1] + R * np.sin(th)])


def glass_polygon(arcs, used=None):
    """Closed prism outline: face samples ordered by angle around the centroid (convex). Faces
    not in `used` are drawn as a FLAT CHORD (corner-to-corner) instead of their curved arc --
    optically inert (no ray touches them) but a cleaner slide shape."""
    parts = []
    for name, a in arcs.items():
        if used is None or name in used:
            parts.append(face_curve(a))
        else:
            parts.append(np.array([a["c1"], a["c2"]]))      # flat: just the two corners
    pts = np.vstack(parts)
    cen = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
    return pts[np.argsort(ang)]


def hit_faces(arcs, n, n_fan, threshold=0.15):
    """Faces struck by a MEANINGFUL fraction of the drawn fan rays -- the optically active faces.
    A face hit by < `threshold` of rays (e.g. a lone grazing edge-ray) carries no imaging bundle,
    so it's treated as dead and flattened. Returns the set of active face names."""
    cnt = {k: 0 for k in arcs}
    nray = 0
    for (sx, sy) in SOURCE_PTS:
        for ai in np.radians(SOURCE_AIM + np.linspace(-FAN_HALFANGLE, FAN_HALFANGLE, n_fan)):
            nray += 1
            O = np.array([sx, sy]); D = np.array([math.cos(ai), math.sin(ai)]); inside = False
            seen = set()
            for _ in range(MAX_DEPTH):
                h = nearest_hit_ff(O, D, arcs)
                if h is None:
                    break
                _t, P, normal, name = h; seen.add(name)
                n1 = (1.0 / n) if not inside else n
                cos1 = -(D @ normal); sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
                if sq1 < 0:
                    D = D + 2.0 * cos1 * normal
                else:
                    cos2 = math.sqrt(sq1); D = n1 * D + (n1 * cos1 - cos2) * normal; inside = not inside
                D = D / np.linalg.norm(D); O = P
            for nm in seen:
                cnt[nm] += 1
    return {nm for nm, c in cnt.items() if c >= threshold * max(nray, 1)}


def trace_draw(O, D, arcs, n, max_depth=16):
    pts = [O.copy()]; ntir = 0; inside = False
    O = O.copy(); D = D.copy()
    for _ in range(max_depth):
        hit = nearest_hit_ff(O, D, arcs)
        if hit is None:
            break
        _t, P, normal, _name = hit
        n1 = (1.0 / n) if not inside else n
        cos1 = -(D @ normal); sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        pts.append(P)
        if sq1 < 0:
            D = D + 2.0 * cos1 * normal; D = D / np.linalg.norm(D); ntir += 1
        else:
            cos2 = math.sqrt(sq1); D = n1 * D + (n1 * cos1 - cos2) * normal
            D = D / np.linalg.norm(D); inside = not inside
        O = P
    if D[0] > 1e-9:
        ts = (PUPIL_X - O[0]) / D[0]
        if ts > 0:
            pts.append(np.array([PUPIL_X, O[1] + ts * D[1]]))
    return np.array(pts), ntir


def draw(ax, x, title, n_fan=17, flatten_unused=True):
    ax.set_facecolor(BG)
    arcs, n = build_arcs_ff(np.asarray(x, float))
    used = hit_faces(arcs, n, n_fan) if flatten_unused else set(arcs)
    poly = glass_polygon(arcs, used)
    ax.fill(poly[:, 0], poly[:, 1], color=GLASS, alpha=0.28, zorder=1, lw=0)
    for name, a in arcs.items():
        if name in used:
            c = face_curve(a)
            ax.plot(c[:, 0], c[:, 1], color="#cccccc", lw=1.1, alpha=0.5, zorder=2)
        else:
            ax.plot([a["c1"][0], a["c2"][0]], [a["c1"][1], a["c2"][1]],
                    color="#cccccc", lw=1.1, alpha=0.5, zorder=2)            # flattened dead face
    for (sx, sy) in SOURCE_PTS:
        for ai in np.radians(SOURCE_AIM + np.linspace(-FAN_HALFANGLE, FAN_HALFANGLE, n_fan)):
            pts, _ = trace_draw(np.array([sx, sy]), np.array([math.cos(ai), math.sin(ai)]), arcs, n)
            ax.plot(pts[:, 0], pts[:, 1], color=RAY, lw=2.2, alpha=0.05, zorder=3)   # glow
            ax.plot(pts[:, 0], pts[:, 1], color=RAY, lw=0.55, alpha=0.55, zorder=4)  # core
    for (sx, sy) in SOURCE_PTS:
        ax.plot(sx, sy, marker="s", color=SRC, ms=7, zorder=6)
    ax.plot([PUPIL_X, PUPIL_X], [-45, 60], color=PUPIL, ls=(0, (6, 5)), lw=1.1, zorder=5)
    ax.plot([PUPIL_X, PUPIL_X], [-EYEBOX_HALF, EYEBOX_HALF], color=PUPIL, lw=3.0, zorder=6)
    ax.set_xlim(-6, 128); ax.set_ylim(-45, 60); ax.invert_yaxis()    # display at bottom (old look)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, color="#f0f0f0", fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(INDIR))
    ap.add_argument("--suffix", default="_LATEST")
    ap.add_argument("--min-folds", type=int, default=1)
    ap.add_argument("--max-folds", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=1e-4, help="min delta-fitness to include")
    ap.add_argument("--outname", default="prism_refine_gallery")
    ap.add_argument("--title", default="AR/VR 90° prism — differentiable freeform refinement (before → after)")
    ap.add_argument("--layout", choices=["vertical", "horizontal"], default="horizontal",
                    help="horizontal = 2 rows (START top / REFINED bottom), one column per design (slide-friendly)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = []
    for p in sorted(Path(args.indir).glob("refine_cell*.json")):
        r = json.loads(p.read_text())
        if "genotype_refined" not in r:
            continue
        if not (args.min_folds <= r.get("folds", 1) <= args.max_folds):
            continue
        if r.get("delta_fit", 0) < args.min_delta:
            continue
        reps.append(r)
    reps.sort(key=lambda r: -r["delta_fit"])           # best-improved first
    reps = reps[:args.max_rows]
    if not reps:
        print("no refined cells matching filter in", args.indir); return

    def labels(r):
        s, rf = r["start"], r["refined"]
        fr = r.get("folds_refined", r["folds"])
        fold_lbl = f"{r['folds']} fold" + (f"→{fr}" if fr != r["folds"] else "")
        top = (f"cell {r['i0']},{r['i1']}  ({fold_lbl})  START\n"
               f"$\\sigma$={s['sigma_mean']:.3f}°  fit={s['fitness']:.3f}")
        bot = (f"REFINED   $\\sigma$={rf['sigma_mean']:.3f}°  fit={rf['fitness']:.3f}\n"
               f"({r['sigma_improve_pct']:+.0f}% sharper)")
        return top, bot

    if args.layout == "horizontal":
        ncol = len(reps)
        fig, axes = plt.subplots(2, ncol, figsize=(3.5 * ncol, 7.4), squeeze=False)
        fig.patch.set_facecolor(BG)
        for j, r in enumerate(reps):
            top, bot = labels(r)
            draw(axes[0, j], r["genotype_start"], top)
            draw(axes[1, j], r["genotype_refined"], bot)
        axes[0, 0].set_ylabel("BEFORE", color="#bbbbbb", fontsize=12)
        axes[1, 0].set_ylabel("AFTER", color="#bbbbbb", fontsize=12)
        for ax in (axes[0, 0], axes[1, 0]):           # keep the row label visible (axis is off)
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    else:
        nrow = len(reps)
        fig, axes = plt.subplots(nrow, 2, figsize=(13, 3.4 * nrow), squeeze=False)
        fig.patch.set_facecolor(BG)
        for i, r in enumerate(reps):
            top, bot = labels(r)
            draw(axes[i, 0], r["genotype_start"], top.replace("\n", "   "))
            draw(axes[i, 1], r["genotype_refined"], bot.replace("\n", "   "))
    fig.suptitle(args.title + "\ngold = display→fold→eyebox relay rays   red = display sources   "
                 "dashed = eye pupil plane", color="#f5f5f5", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(args.indir) / f"{args.outname}{args.suffix}.png"
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"[refine-gallery] {out}  ({len(reps)} cells, {args.layout})")


if __name__ == "__main__":
    main()
