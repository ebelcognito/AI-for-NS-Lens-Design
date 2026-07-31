"""
High-quality gallery of ORDERLY non-sequential folds (the whole bundle folding together).

"Orderly" = one ray-surface-interaction sequence carries most of the bundle's energy
(high dominant-energy share) and it focuses -> a clean, coherent fold rather than a
chaotic spray. Two sources:

  --from-archive : render the most orderly designs ALREADY in a QD archive (existing data)
  --search       : freshly discover orderly folds by maximizing the share of energy taking
                   a *given* target sequence (fold_demo's objective) -> uniform folds

Renders are high resolution; PNGs are NEVER overwritten (suffix + geometry sidecar JSON).

    python3 ns_probe/fold_gallery.py --from-archive ns_probe/results_fold_qd/archive.json --n 12
    python3 ns_probe/fold_gallery.py --search
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import fold_demo as FD
from sequence_tools import (dominance_report, format_signature,
                            make_single_element_face_classifier, parse_target)
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

# High-quality render config
HQ_W = 1900
HQ_RAYS = 15
HQ_DEPTH = 11          # enough for the dominant fold; trims deep faint ghost chains -> cleaner
CROP_X = (-4.0, 145.0)  # extends PAST the sensor (x=120) so the converging focal cone is visible
CROP_Y = (-40.0, 40.0)

OUTDIR = REPO / "ns_probe" / "results_gallery"


def uniq(path: Path) -> Path:
    if not path.exists():
        return path
    k = 1
    while path.with_name(f"{path.stem}_{k}{path.suffix}").exists():
        k += 1
    return path.with_name(f"{path.stem}_{k}{path.suffix}")


def hq_scene(p, *, rays=HQ_RAYS, brightness=0.85):
    a = np.radians(p["ang"])
    objs = []
    for dy in np.linspace(-4.0, 4.0, rays):
        y = p["ray_y"] + dy
        objs.append({"type": "SingleRay", "p1": {"x": 0.0, "y": float(y)},
                     "p2": {"x": float(np.cos(a)), "y": float(y + np.sin(a))},
                     "brightness": float(brightness)})
    objs.append(FD.element(p))
    objs.append({"type": "Detector", "p1": {"x": FD.X_SENSOR, "y": -120},
                 "p2": {"x": FD.X_SENSOR, "y": 120}})
    objs.append({"type": "CropBox", "p1": {"x": CROP_X[0], "y": CROP_Y[0]},
                 "p4": {"x": CROP_X[1], "y": CROP_Y[1]}, "width": HQ_W})
    return {"version": 5, "maxRayDepth": HQ_DEPTH, "objs": objs}


def render_one(runner, p, name, meta, outdir):
    rres, _ = runner.simulate(hq_scene(p))
    imgs = rres.get("images") or []
    if not imgs or not imgs[0].get("dataUrl"):
        print(f"  ! render failed for {name}: {rres.get('error')}")
        return None
    png = uniq(outdir / f"{name}.png")
    png.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
    png.with_suffix(".json").write_text(json.dumps({"geometry": p, **meta}, indent=2))
    return png


def contact(items, title, out):
    n = len(items)
    if n == 0:
        return
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.7 * rows))
    axes = np.atleast_2d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (png, lab) in zip(axes, items):
        ax.imshow(mpimg.imread(png)); ax.set_title(lab, fontsize=8, loc="left"); ax.axis("off")
    fig.suptitle(title, fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    cs = uniq(out)
    fig.savefig(cs, dpi=120); plt.close(fig)
    print(f"[gallery] wrote {cs}")


def from_archive(runner, archive_paths, n, outdir, focus=False, min_eff=0.45):
    cells = []
    for ap in archive_paths:
        d = json.loads(Path(ap).read_text())
        for c in d["cells"]:
            c["_src"] = Path(ap).parent.name
            cells.append(c)
    if focus:
        # best-FOCUSING folds: real fold, enough delivered energy, sort by tightest spot
        folds = [c for c in cells if c.get("tir_count", 0) >= 2
                 and c.get("efficiency", 0) >= min_eff and 0 < c.get("spot", 99) < 99]
        folds.sort(key=lambda c: (c.get("spot", 99), -c.get("efficiency", 0)))
        title = "Folds with successful focusing (tightest spot first)  [high-res]"
        csname = "contact_focusing_folds.png"
    else:
        folds = [c for c in cells if c.get("tir_count", 0) >= 1]
        folds.sort(key=lambda c: (-c.get("dom_energy_share", 0), c.get("spot", 99), -c.get("fitness", 0)))
        title = "Orderly folds from the QD archive (existing data)  [high-res]"
        csname = "contact_archive_orderly.png"
    picks = folds[:n]
    print(f"[archive] rendering {len(picks)} {'best-focusing' if focus else 'most-orderly'} "
          f"designs from {len(archive_paths)} archive(s)")
    items = []
    for i, c in enumerate(picks, 1):
        p = dict(zip(FD.PARAM_NAMES, c["genotype"]))
        tir = int(round(c["tir_count"]))
        meta = dict(source=c.get("_src"), tir=tir, dom_share=c.get("dom_energy_share"),
                    focused_throughput=c.get("fitness"), efficiency=c.get("efficiency"),
                    spot=c.get("spot"), dom_signature=c.get("dom_signature"))
        tag = f"focus_{i:02d}_spot{c.get('spot',0):.2f}_tir{tir}" if focus else f"arch_{i:02d}_tir{tir}"
        png = render_one(runner, p, tag.replace(".", "p"), meta, outdir)
        if png:
            lab = (f"{tir} TIR | spot={c.get('spot',0):.2f} | eff={c.get('efficiency',0):.0%} | "
                   f"order={c.get('dom_energy_share',0):.0%}\n{c.get('dom_signature','')}")
            items.append((png, lab)); print("  " + lab.replace("\n", "  "))
    contact(items, title, outdir / csname)


# Curated orderly target sequences (uniform folds of increasing depth / different face routes)
TARGETS = [
    "T@A TIR@B TIR@A T@B",
    "T@A TIR@D TIR@B T@B",
    "T@A TIR@B TIR@A TIR@B T@B",
    "T@A TIR@D TIR@B TIR@B T@B",
    "T@A TIR@B TIR@A TIR@B TIR@A T@B",
    "T@A TIR@D TIR@B TIR@A TIR@T T@B",
]


def from_search(runner, outdir, seeds=(0, 1)):
    print(f"[search] discovering orderly folds for {len(TARGETS)} target sequences x {len(seeds)} seeds")
    items = []
    for ti, spec in enumerate(TARGETS):
        target = parse_target(spec)
        for sd in seeds:
            x, f = FD.search(runner, target, n_rays=21, beam_w=8.0, coating=None,
                             n_random=1800, cma_gens=40, seed=sd, verbose=False)
            if x is None or f < 0.4:
                print(f"  [skip] '{spec}' seed {sd}: best share {f:.2f}")
                continue
            p = FD.vec_to_params(x)
            face_of = make_single_element_face_classifier(p["cx"], FD.ELEM_H)
            res, _ = runner.simulate(FD.scene(p, n_rays=21, beam_w=8.0))
            rep = dominance_report(res, face_of, target)
            meta = dict(source="search", target=spec, target_share=rep["target_share_of_input"],
                        ledger=rep["ledger"], seed=sd, dominant=rep["dominant"])
            depth = spec.count("TIR")
            png = render_one(runner, p, f"seq_tir{depth}_s{sd}_{ti}", meta, outdir)
            if png:
                lab = (f"{depth} TIR | uniform fold, share={rep['target_share_of_input']:.0%}\n{spec}")
                items.append((png, lab)); print(f"  [ok] '{spec}' seed {sd}: share {f:.0%}")
    contact(items, "Orderly folds discovered by target-sequence search  [high-res]",
            outdir / "contact_search_orderly.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-archive", default=None, help="comma-separated archive.json paths")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--focus", action="store_true", help="rank by tightest focus (best-focusing folds)")
    ap.add_argument("--min-eff", type=float, default=0.45)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        if args.from_archive:
            paths = [s for s in args.from_archive.split(",") if s.strip()]
            from_archive(runner, paths, args.n, outdir, focus=args.focus, min_eff=args.min_eff)
        if args.search:
            from_search(runner, outdir)


if __name__ == "__main__":
    main()
