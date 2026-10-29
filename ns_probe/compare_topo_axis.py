"""Compare the tir_bounces archive against the tir_faces archive.

The two arms are the same experiment -- same genome box, fitness, field set, gates,
param_scaling and eval budget -- differing only in the first behaviour axis. So the
question this answers is narrow and fair: does indexing the archive by WHICH faces the
light bounces off, instead of HOW MANY times, retain more distinct ray paths, and what
does that cost in fitness?

Three metrics, in order of what they are for:

  distinct signatures retained   the headline. The baseline axis has 3 attainable fold
                                 counts, so it can separate at most 3 topology classes
                                 however many designs it stores.
  best / median fitness          did buying diversity cost quality?
  designs per occupied cell      did the finer axis starve the emitters? This is the
                                 real risk of the change -- an archive with one design
                                 per cell gives CMA-ME no "improved" signal at all.

    python ns_probe/compare_topo_axis.py --a runs/fold24_clean_ab --b runs/fold24_topo
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np


def load(run_dir: str):
    """All seeds of one arm -> (cells, meta)."""
    cells, meta = [], None
    for f in sorted(glob.glob(f"{run_dir}/seed*/archive.json")):
        d = json.loads(Path(f).read_text())
        meta = meta or d.get("meta", {})
        for c in d.get("cells", []):
            c["_seed"] = f.split("/")[-2]
            cells.append(c)
    return cells, (meta or {})


def summarise(name: str, cells, meta):
    if not cells:
        return dict(name=name, n=0)
    fits = np.array([c["fitness"] for c in cells], float)
    sigs = [c.get("dom_signature") for c in cells if c.get("dom_signature")]
    per_seed = collections.Counter(c["_seed"] for c in cells)
    # distinct topologies at three granularities
    faces = set()
    for s in sigs:
        f = tuple(sorted(collections.Counter(
            tok.split("@")[1] for tok in s.split(" -> ")
            if "@" in tok and tok.split("@")[0].upper() == "TIR").items()))
        faces.add(f)
    return dict(
        name=name, n=len(cells), seeds=len(per_seed),
        cells_per_seed=round(len(cells) / max(1, len(per_seed)), 1),
        distinct_signatures=len(set(sigs)),
        distinct_face_multisets=len(faces),
        distinct_fold_counts=len({c.get("tir_count") for c in cells}),
        best=float(fits.max()), median=float(np.median(fits)),
        above_05=int((fits > 0.5).sum()), above_07=int((fits > 0.7).sum()),
        grid=meta.get("bc_bins"), evals=meta.get("evals"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs/fold24_clean_ab", help="baseline arm (tir_bounces)")
    ap.add_argument("--b", default="runs/fold24_topo", help="topology arm (tir_faces)")
    args = ap.parse_args()

    rows = []
    for label, d in (("tir_bounces (baseline)", args.a), ("tir_faces (topology)", args.b)):
        cells, meta = load(d)
        rows.append(summarise(label, cells, meta))

    keys = [("n", "elites kept"), ("cells_per_seed", "cells / seed"),
            ("distinct_signatures", "DISTINCT SIGNATURES"),
            ("distinct_face_multisets", "distinct face multisets"),
            ("distinct_fold_counts", "distinct fold counts"),
            ("best", "best fitness"), ("median", "median fitness"),
            ("above_05", "elites > 0.5"), ("above_07", "elites > 0.7"),
            ("grid", "grid"), ("evals", "evals/seed")]
    w = max(len(lbl) for _, lbl in keys) + 2
    print(f"{'':{w}}" + "".join(f"{r['name']:>26s}" for r in rows))
    print("-" * (w + 26 * len(rows)))
    for k, lbl in keys:
        vals = []
        for r in rows:
            v = r.get(k)
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"{lbl:{w}}" + "".join(f"{v:>26s}" for v in vals))

    a, b = rows
    if a.get("n") and b.get("n"):
        print(f"\ntopology retention: {b['distinct_signatures']} vs "
              f"{a['distinct_signatures']} distinct ray paths "
              f"({b['distinct_signatures']/max(1,a['distinct_signatures']):.1f}x)")
        print(f"fitness cost:       best {b['best']:.3f} vs {a['best']:.3f} "
              f"({b['best']-a['best']:+.3f})")
        pressure = b["n"] / max(1, b["cells_per_seed"])
        print(f"selection pressure: {b['cells_per_seed']} cells/seed on a "
              f"{b['grid']} grid -- if this approaches the grid size the axis is too fine")


if __name__ == "__main__":
    main()
