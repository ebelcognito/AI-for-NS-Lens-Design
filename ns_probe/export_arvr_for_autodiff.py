"""
Export selected AR/VR prism designs in a form the differentiable twin (ns_autodiff) can
consume. This is a READ-ONLY bridge: it reads the QD archive, traces each design through the
JS engine as a forward-truth ORACLE, and writes a self-contained JSON per design plus a
combined file. It does not edit ns_autodiff or ray-optics.

Each exported design is an AR-extended analogue of ns_autodiff.topology.FrozenTopology, with
the three things that differ from the spot-fold problem made explicit:

  1. ELEMENT: a 4-curved-face freeform prism (FOUR arcs), not 2 arcs + 2 flat edges. We emit
     each surface in ns_autodiff's surface form: arc -> {center, r, end1, end2, apex} (built
     from the same 3-point circumcircle ns_autodiff uses), or line if a face is ~flat.
  2. SOURCES: 3 point sources each firing a fan, not one collimated beam. A tracked ray's
     identity is (source_id, fan_angle_deg) -- the analogue of the spot problem's beam-offset
     dy. We emit each tracked ray's explicit origin + unit input direction.
  3. TARGET: COLLIMATED at the eye pupil -> the loss is ANGULAR. We export, per tracked
     (eyebox-reaching) ray, the oracle's exit DIRECTION at the pupil (and its angle), not just
     a sensor y. The differentiable refiner minimises per-source angular spread (collimation).

The frozen topology per ray is the ordered list of interaction TYPES (refract/tir) + a
best-effort nearest-surface face label (audit only; the twin re-derives geometry and does not
trust labels, exactly as its README warns).

    python3 ns_probe/export_arvr_for_autodiff.py            # top 12 by fitness
    python3 ns_probe/export_arvr_for_autodiff.py --n 8 --sort fov
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
import arvr_prism as AV


def circumcircle(p1, p2, p3):
    """Center + radius of the circle through 3 points (matches geometry_torch.circumcenter).
    Returns (None, None) if ~collinear (treat that face as a line)."""
    (ax, ay), (bx, by), (cx, cy) = p1, p2, p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None, None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    return [ux, uy], r


def build_surfaces_export(p):
    """The four freeform faces of the AR prism (see arvr_prism.prism_element), each emitted
    in ns_autodiff surface form. Face keys name the physical edge."""
    cx, thick = p["cx"], p["thick"]
    H, SAG = AV.ELEM_H, None
    lx, rx = cx - thick / 2.0, cx + thick / 2.0
    dA = math.tan(math.radians(p["tilt_a"])) * H
    dB = math.tan(math.radians(p["tilt_b"])) * H
    bl, br = [lx + dA, -H], [rx + dB, -H]
    tr, tl = [rx - dB, H], [lx - dA, H]
    faces = {
        "bottom": (bl, [(bl[0] + br[0]) / 2.0, -H - p["dsag"]], br),
        "right":  (br, [rx + p["bsag"], 0.0], tr),
        "top":    (tr, [(tr[0] + tl[0]) / 2.0, H + p["tsag"]], tl),
        "left":   (tl, [lx - p["fsag"], 0.0], bl),
    }
    out = {}
    for name, (e1, apex, e2) in faces.items():
        center, r = circumcircle(e1, apex, e2)
        if center is None:
            out[name] = {"kind": "line", "a": e1, "b": e2}
        else:
            out[name] = {"kind": "arc", "center": center, "r": r,
                         "end1": e1, "end2": e2, "apex": apex}
    return out


def nearest_face(surfaces, x, y):
    """Best-effort label of which physical face a hit (x,y) lies on (audit only)."""
    best, bestd = None, 1e18
    for name, s in surfaces.items():
        if s["kind"] == "arc":
            d = abs(math.hypot(x - s["center"][0], y - s["center"][1]) - s["r"])
        else:
            (ax, ay), (bx, by) = s["a"], s["b"]
            ex, ey = bx - ax, by - ay
            t = max(0.0, min(1.0, ((x - ax) * ex + (y - ay) * ey) / (ex * ex + ey * ey + 1e-12)))
            d = math.hypot(x - (ax + t * ex), y - (ay + t * ey))
        if d < bestd:
            bestd, best = d, name
    return best


def export_design(runner, cell, surfaces):
    """Trace the design via the oracle; return its tracked (eyebox-reaching) rays with frozen
    topology + oracle exit direction/energy, and per-source dominant signatures."""
    p = AV.prism_dict(cell["genotype"])
    x = np.array(cell["genotype"], dtype=float)
    res, _ = runner.simulate(AV.scene_arvr(x, record=True))
    m = AV.arvr_metrics(res)

    rays = []
    for r in res.get("raySequences", []):
        if r.get("dirx", 0) <= 0 or r.get("b", 0) <= 0:
            continue
        if abs(float(r["y"]) - AV.EYEBOX_Y) > AV.EYEBOX_HALF:
            continue
        seq = r["seq"]
        s0, s1 = seq[0], seq[1]
        in_ang = math.degrees(math.atan2(s1["y"] - s0["y"], s1["x"] - s0["x"]))
        src = int(np.argmin([abs(s0["x"] - sx) for sx in AV.SOURCE_XS]))
        events = [[v.get("t"), nearest_face(surfaces, v["x"], v["y"])]
                  for v in seq if v.get("t") not in ("source", "sensor")]
        exit_ang = math.degrees(math.atan2(r["diry"], r["dirx"]))
        rays.append(dict(
            source=src, fan_angle_deg=round(in_ang - AV.SOURCE_AIM, 4),
            origin=[round(s0["x"], 6), round(s0["y"], 6)],
            input_dir=[round(math.cos(math.radians(in_ang)), 8),
                       round(math.sin(math.radians(in_ang)), 8)],
            n_events=len(events), events=events,
            oracle_pupil_y=round(float(r["y"]), 6),
            oracle_exit_dir=[round(float(r["dirx"]), 8), round(float(r["diry"]), 8)],
            oracle_exit_angle_deg=round(exit_ang, 5),
            oracle_energy=round(float(r["b"]), 8)))

    # per-source dominant frozen topology (the shared event TYPE-sequence to unroll)
    per_src = {}
    for s in range(len(AV.SOURCE_XS)):
        sigs = {}
        for ry in rays:
            if ry["source"] != s:
                continue
            key = tuple(t for t, _f in ry["events"])
            sigs[key] = sigs.get(key, 0.0) + ry["oracle_energy"]
        if sigs:
            dom = max(sigs, key=sigs.get)
            per_src[s] = dict(n_events=len(dom), type_sequence=list(dom),
                              n_rays=sum(1 for ry in rays if ry["source"] == s
                                         and tuple(t for t, _f in ry["events"]) == dom))
    return rays, per_src, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(REPO / "ns_probe" / "results_arvr_qd" / "archive.json"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sort", default="fitness", choices=["fitness", "fov"])
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_arvr_qd" / "autodiff_export"))
    args = ap.parse_args()

    arc = json.loads(Path(args.archive).read_text())
    # Match the source/geometry config the archive was generated with (e.g. the multifold
    # run's tilted source_aim=-102). Tracing with the wrong source aim would give wrong paths.
    geom = arc.get("meta", {}).get("geometry", {})
    if "source_aim" in geom:
        AV.SOURCE_AIM = float(geom["source_aim"])
        print(f"[config] using source_aim={AV.SOURCE_AIM} deg from archive meta")
    cells = sorted(arc["cells"], key=lambda c: -c[args.sort])[:args.n]
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    common = dict(
        problem="off-axis AR near-eye display prism relay (2-D meridional)",
        element_type="4-curved-face freeform prism (FOUR arcs)",
        param_names=AV.PRISM_NAMES,
        imaging="collimated (virtual image at infinity); per-field target is a PARALLEL bundle",
        loss="per-source angular spread (collimation) of eyebox rays; maximise eyebox throughput",
        sources=dict(display_x=list(map(float, AV.SOURCE_XS)), display_y=AV.DISPLAY_Y,
                     source_aim_deg=AV.SOURCE_AIM, n_fan=AV.N_FAN,
                     fan_halfangle_deg=AV.FAN_HALFANGLE),
        detector=dict(plane_x=AV.PUPIL_X, eyebox_y=AV.EYEBOX_Y, eyebox_half=AV.EYEBOX_HALF,
                      measure="exit ray direction (angle) at plane_x, gated to the eyebox"),
        elem_h=AV.ELEM_H, sag_range=AV.SAG, sigma0_deg=AV.SIGMA0,
        units="engine units (~mm); same scale as ns_probe / ns_autodiff",
        oracle="ray-optics JS engine via PersistentRayOpticsRunner (forward truth)")

    designs = []
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for rank, c in enumerate(cells, 1):
            p = AV.prism_dict(c["genotype"])
            surfaces = build_surfaces_export(p)
            rays, per_src, m = export_design(runner, c, surfaces)
            d = dict(
                id=f"cell_{int(c['i0'])}_{int(c['i1'])}", rank=rank,
                cell=[int(c["i0"]), int(c["i1"])],
                genotype={k: float(v) for k, v in p.items()},
                genotype_vector=[float(v) for v in c["genotype"]],
                refractive_index=float(p["n"]),
                surfaces=surfaces,
                per_source_topology=per_src,
                tracked_rays=rays,
                reference_metrics=dict(
                    fitness=float(c["fitness"]), fov_deg=float(c["fov"]),
                    depth=float(c["depth"]), tir=int(c.get("tir", 0)),
                    fields=[dict(present=f["present"], throughput=f["T"],
                                 theta_deg=f["theta"], sigma_deg=f["sigma"])
                            for f in m["fields"]]))
            designs.append(d)
            (outdir / f"{d['id']}.json").write_text(
                json.dumps(dict(common=common, design=d), indent=2, default=float))
            print(f"[{rank:2d}/{len(cells)}] {d['id']}: fit={c['fitness']:.3f} "
                  f"FOV={c['fov']:.1f} tracked_rays={len(rays)} "
                  f"src_topos={[t['type_sequence'] for t in per_src.values()]}")

    combined = outdir / "arvr_autodiff_export.json"
    combined.write_text(json.dumps(dict(common=common, n_designs=len(designs),
                                        designs=designs), indent=2, default=float))
    print(f"\n[export] {combined}")
    print(f"[export] {len(designs)} per-design files in {outdir}/")


if __name__ == "__main__":
    main()
