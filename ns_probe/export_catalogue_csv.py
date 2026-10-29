"""Export the ray-path catalogue as a spreadsheet: one row per design, picture included.

    python ns_probe/export_catalogue_csv.py --catalogue runs/catalogue_gated.json \
                                            --out catalogue_gated

Writes into --out:

    png/                one PNG per design -- body outline + the dominant ray bundle
    designs.csv         GATE-PASSING designs only, best first. This is the deliverable.
    designs_all.csv     every candidate topology including the rejected ones, for coverage
    designs.xlsx        designs.csv with the PNG actually EMBEDDED in the image column

CSV cannot hold an image -- a cell is text. So the `image` column carries the relative
path (`png/001_....png`), which is what a script or a static-site build wants, and the
XLSX carries the real picture in the cell, which is what a person opening it wants. Both
are generated from the same rows so they cannot drift.

PNG here rather than the SVG that export_catalogue_dir.py writes: openpyxl can only embed
a raster, and .gitignore drops *.png -- which is correct for this file, since it is a
rendering of runs/catalogue_gated.json and is reproducible from it in seconds.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.path import Path as MPath            # noqa: E402
from matplotlib.patches import PathPatch             # noqa: E402

C_GLASS = "#3E9CAA"
C_GLASS_FILL = "#3E9CAA"
C_RAY = "#D98A1F"
C_PLANE = "#8A9A9C"
XS = 120.0          # detector plane, mm from the entrance face
PNG_W = 460         # thumbnail width in pixels; the XLSX column is sized to match


def slug(signature: str) -> str:
    """`T@A -> TIR@D -> TIR@B -> T@B` -> `A-Dt-Bt-B`  (lowercase t marks a TIR)."""
    parts = []
    for tok in signature.split(" -> "):
        if "@" not in tok:
            continue
        kind, face = tok.split("@")
        parts.append(face.strip() + {"TIR": "t", "PR": "p"}.get(kind.strip().upper(), ""))
    return re.sub(r"[^A-Za-z0-9-]", "", "-".join(parts))


def outline_mpath(o):
    """Body outline -> a matplotlib Path. `arc` vertices are quadratic control points."""
    if not o:
        return None
    verts, codes, prev, pending = [], [], None, None
    for v in list(o) + [o[0]]:
        if v.get("arc"):
            pending = v
            continue
        if prev is None:
            verts.append((v["x"], v["y"]))
            codes.append(MPath.MOVETO)
        elif pending:
            # same control-point reconstruction as the SVG exporter, so the two drawings
            # of one design are the same curve and not merely similar
            cx = 2 * pending["x"] - (prev["x"] + v["x"]) / 2
            cy = 2 * pending["y"] - (prev["y"] + v["y"]) / 2
            verts += [(cx, cy), (v["x"], v["y"])]
            codes += [MPath.CURVE3, MPath.CURVE3]
        else:
            verts.append((v["x"], v["y"]))
            codes.append(MPath.LINETO)
        prev, pending = v, None
    return MPath(verts, codes)


def render_png(e, path: Path) -> None:
    x0, x1, y0, y1 = 0.0, XS, -32.0, 32.0
    for v in e.get("outline") or []:
        x0, x1 = min(x0, v["x"] - 4), max(x1, v["x"] + 4)
        y0, y1 = min(y0, v["y"] - 4), max(y1, v["y"] + 4)
    for r in e.get("rays") or []:
        for p in r:
            y0, y1 = min(y0, p[1]), max(y1, p[1])
    x1 = max(x1, XS + 6)          # keep the detector plane inside the frame, not on it
    w, h = x1 - x0, y1 - y0
    dpi = 100.0
    fig = plt.figure(figsize=(PNG_W / dpi, PNG_W / dpi * h / w), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    ax.axvline(XS, color=C_PLANE, lw=0.9, ls=(0, (3, 3)))
    mp = outline_mpath(e.get("outline"))
    if mp is not None:
        ax.add_patch(PathPatch(mp, facecolor=C_GLASS_FILL, alpha=0.15, lw=0))
        ax.add_patch(PathPatch(mp, facecolor="none", edgecolor=C_GLASS, lw=1.1))
    for r in e.get("rays") or []:
        ax.plot([p[0] for p in r], [p[1] for p in r],
                color=C_RAY, lw=0.6, alpha=0.85, solid_joinstyle="round")
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)


BASE_COLS = ["image", "rank", "signature", "folds", "fitness", "spot_mm", "thinness_mm",
             "crit_margin_deg", "ns_frac", "field_accept", "topology_stable", "verdict",
             "times_seen", "trace_fitness", "source_run", "source_seed", "genome_source",
             "relative_sag", "n_rays_drawn"]


def row_of(e, img, rank, pkeys):
    p = e.get("params") or {}
    r = {
        "image": img, "rank": rank, "signature": e["signature"],
        "folds": e.get("tir_count"),
        "fitness": (None if e.get("hardened_fitness") is None
                    else round(e["hardened_fitness"], 4)),
        "spot_mm": _r(e.get("spot"), 4), "thinness_mm": _r(e.get("thinness"), 2),
        "crit_margin_deg": _r(e.get("crit_margin"), 3), "ns_frac": _r(e.get("ns_frac"), 3),
        "field_accept": _r(e.get("field_accept"), 3),
        "topology_stable": e.get("hardened_signature") == e["signature"],
        "verdict": e["verdict"], "times_seen": e.get("times_seen"),
        "trace_fitness": round(e["trace_fitness"], 4),
        "source_run": e.get("source_run"), "source_seed": e.get("source_seed"),
        "genome_source": e.get("genome_source"),
        "relative_sag": e.get("relative_sag"), "n_rays_drawn": len(e.get("rays") or []),
    }
    for k in pkeys:
        r["p_" + k] = _r(p.get(k), 5)
    r["genotype"] = json.dumps([round(v, 6) for v in e.get("genotype") or []])
    return r


def _r(v, n):
    return None if v is None else round(float(v), n)


def write_xlsx(rows, cols, out: Path, root: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "designs"
    ws.append(cols)
    for c in range(1, len(cols) + 1):
        ws.cell(row=1, column=c).font = Font(bold=True)
    ws.freeze_panes = "B2"

    icol = cols.index("image") + 1
    ws.column_dimensions[get_column_letter(icol)].width = PNG_W / 7.0
    for j, name in enumerate(cols, start=1):
        if name == "image":
            continue
        w = max(len(name) + 2, 11)
        ws.column_dimensions[get_column_letter(j)].width = min(w if name != "signature"
                                                               else 46, 46)
    for i, r in enumerate(rows, start=2):
        for j, name in enumerate(cols, start=1):
            if name == "image":
                continue
            ws.cell(row=i, column=j, value=r.get(name))
        img_path = root / r["image"]
        if img_path.exists():
            im = XLImage(str(img_path))
            scale = (PNG_W * 0.75) / im.width      # 0.75 px per point at 96 dpi
            im.width, im.height = im.width * scale, im.height * scale
            ws.row_dimensions[i].height = im.height * 0.75 + 6
            ws.add_image(im, f"{get_column_letter(icol)}{i}")
        ws.cell(row=i, column=cols.index("signature") + 1).alignment = Alignment(
            vertical="center", wrap_text=True)
    wb.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="runs/catalogue_gated.json")
    ap.add_argument("--out", default="catalogue_gated")
    ap.add_argument("--min-fitness", type=float, default=0.001,
                    help="floor for designs.csv. The default drops only the designs that "
                         "pass every gate yet are blind at some field (fitness 0).")
    a = ap.parse_args()

    doc = json.loads(Path(a.catalogue).read_text())
    entries = doc["entries"]
    entries.sort(key=lambda e: -(e.get("hardened_fitness") or -1))

    out = Path(a.out)
    (out / "png").mkdir(parents=True, exist_ok=True)
    pkeys = sorted({k for e in entries for k in (e.get("params") or {})})

    rows = []
    for rank, e in enumerate(entries, start=1):
        fit = e.get("hardened_fitness")
        name = "%03d_%s_%sf_%s.png" % (rank, "----" if fit is None else "%04d" % (fit * 1e4),
                                       e.get("tir_count"), slug(e["signature"]) or "none")
        render_png(e, out / "png" / name)
        rows.append(row_of(e, "png/" + name, rank, pkeys))

    cols = BASE_COLS + ["p_" + k for k in pkeys] + ["genotype"]
    keep = [r for r in rows
            if r["verdict"] == "PASS" and (r["fitness"] or 0) >= a.min_fitness]

    for fname, rs in (("designs.csv", keep), ("designs_all.csv", rows)):
        with open(out / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rs)
        print("wrote %s  (%d rows)" % (out / fname, len(rs)))

    write_xlsx(keep, cols, out / "designs.xlsx", out)
    print("wrote %s  (%d rows, images embedded)" % (out / "designs.xlsx", len(keep)))
    (out / "meta.json").write_text(json.dumps(doc["meta"], indent=2))
    print("gates: %s" % json.dumps(doc["meta"]["hardened"]))


if __name__ == "__main__":
    main()
