"""Top-9 gallery for an AR/VR PRISM archive (freeform or arc).

Draws the best collimating relays: the 4-face prism, the 3 display sources fanning in, and the
relay rays folding (TIR) to the eye pupil. Each panel is annotated with FOV, collimation fitness,
and fold count (post-hoc TIR-bounce count on the chief ray -- folds aren't an archive axis, so we
recover them here). Picks the top 9 by fitness; with --tir-set it restricts to a fold-count range.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .prism_branch import (PUPIL_X, SOURCE_PTS, SOURCE_AIM, FAN_HALFANGLE, EYEBOX_HALF,
                           build_arcs_ff, nearest_hit_ff, _unwrap, _delta)


def face_curve(a, n=70):
    ce = a["center"]
    th1 = _unwrap(math.atan2(a["c1"][1] - ce[1], a["c1"][0] - ce[0]), a["th_apex"])
    th2 = _unwrap(math.atan2(a["c2"][1] - ce[1], a["c2"][0] - ce[0]), a["th_apex"])
    th = np.linspace(th1, th2, n)
    s = (th - a["th_apex"]) / a["half_span"]
    R = a["r"] + np.array([_delta(si, a["coeffs"])[0] for si in s])
    return ce[0] + R * np.cos(th), ce[1] + R * np.sin(th)


def trace_draw(O, D, arcs, n, max_depth=16):
    """Single dominant path (refract-or-TIR by physics) for drawing + fold counting."""
    pts = [O.copy()]; ntir = 0; inside = False
    O = O.copy(); D = D.copy()
    for _ in range(max_depth):
        hit = nearest_hit_ff(O, D, arcs)
        if hit is None:
            break
        t, P, normal, name = hit
        n1 = (1.0 / n) if not inside else n
        cos1 = -(D @ normal); sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        pts.append(P)
        if sq1 < 0:
            D = D + 2.0 * cos1 * normal; D = D / np.linalg.norm(D); ntir += 1
        else:
            cos2 = math.sqrt(sq1)
            D = n1 * D + (n1 * cos1 - cos2) * normal; D = D / np.linalg.norm(D)
            inside = not inside
        O = P
    if D[0] > 1e-9:
        ts = (PUPIL_X - O[0]) / D[0]
        if ts > 0:
            pts.append(np.array([PUPIL_X, O[1] + ts * D[1]]))
    return np.array(pts), ntir


def draw(ax, x, title):
    arcs, n = build_arcs_ff(x)
    for name, a in arcs.items():
        cx, cy = face_curve(a); ax.plot(cx, cy, "k", lw=1.6, zorder=5)
    folds = 0
    for (sx, sy) in SOURCE_PTS:
        ax.plot(sx, sy, "o", color="tab:orange", ms=4, zorder=6)
        for k, ai in enumerate(np.radians(SOURCE_AIM + np.linspace(-FAN_HALFANGLE, FAN_HALFANGLE, 5))):
            O = np.array([sx, sy]); D = np.array([math.cos(ai), math.sin(ai)])
            pts, ntir = trace_draw(O, D, arcs, n)
            if k == 2:
                folds = max(folds, ntir)                  # chief ray fold count
            ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", lw=0.6, alpha=0.7, zorder=3)
    ax.axvline(PUPIL_X, color="tab:red", ls="--", lw=1.0, zorder=4)
    ax.plot([PUPIL_X, PUPIL_X], [-EYEBOX_HALF, EYEBOX_HALF], color="tab:red", lw=3, zorder=7)  # eyebox
    ax.set_xlim(-5, 128); ax.set_ylim(-45, 60)          # fixed window (display top -> eyebox); clip strays
    ax.set_aspect("auto"); ax.grid(alpha=0.2)
    ax.set_title(title + f"   ({folds} folds)", fontsize=9)
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", default="ns_autodiff/results_search_prism_freeform")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tir-set", type=int, nargs="+", default=None, help="restrict to these fold counts")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arch = json.loads(Path(args.archive).read_text())
    cells = sorted(arch["cells"], key=lambda c: -c["fitness"])
    # fold-count filter (post-hoc) if requested: rank then keep those with chief-ray folds in set
    picks = []
    for c in cells:
        x = np.array(c["genotype"], float)
        if args.tir_set is not None:
            arcs, n = build_arcs_ff(x)
            sx, sy = SOURCE_PTS[1]
            _, folds = trace_draw(np.array([sx, sy]),
                                  np.array([math.cos(math.radians(SOURCE_AIM)),
                                            math.sin(math.radians(SOURCE_AIM))]), arcs, n)
            if folds not in args.tir_set:
                continue
        picks.append(c)
        if len(picks) == 9:
            break

    fig, axes = plt.subplots(3, 3, figsize=(16, 11)); axes = axes.ravel()
    for ax in axes[len(picks):]:
        ax.axis("off")
    for ax, c in zip(axes, picks):
        draw(ax, np.array(c["genotype"], float),
             f"FOV {c['fov']:.1f}deg  fit {c['fitness']:.3f}  sig {c['sigma_mean']:.2f}deg")
    kind = arch["meta"]["experiment"]
    fig.suptitle(f"AR/VR prism — top 9 collimating relays  [{kind}]\n"
                 f"orange=display sources, red bar=eyebox, blue=relay rays (folding to the eye)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out); base = out / f"gallery_prism{args.tag}.png"; png, k = base, 0
    while png.exists():
        k += 1; png = base.with_name(f"gallery_prism{args.tag}_{k}.png")
    fig.savefig(png, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"[prism-gallery] {png}")
    for c in picks:
        print(f"  FOV {c['fov']:.1f}  fit {c['fitness']:.3f}  depth {c['depth']:.0f}  sig {c['sigma_mean']:.2f}")


if __name__ == "__main__":
    main()
