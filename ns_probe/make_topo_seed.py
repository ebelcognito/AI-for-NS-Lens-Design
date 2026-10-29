"""Build the warm-start archive for a `tir_faces`-indexed run.

The topology axis cannot cold-start. fold24_clean's gate stack keeps ~15% of what the
emitters propose, and from an empty archive CMA-ME never accumulates the elites its
emitters need to sample around -- the first attempt at that gate filled 0-3 of 48 cells
in 4k evals. `resume_from` is the documented fix, but it reads each cell's behaviour
coordinates by NAME (`cell_bc_keys`), and no archive on disk carries a `tir_faces` key.

This writes one that does, from two sources:

  * the existing fold24_clean seed archive -- 7 cells, already in the relative-sag genome
  * the catalogue's hardened survivors from relative-sag runs -- topologically diverse
    designs that the (tir_count, thinness) archive had no cell to keep

Only relative-sag genotypes are eligible: the fold24_clean genome reads fsag/bsag as
FRACTIONS of the face's half-chord, so a genotype from an absolute-sag run means a
different element under it. Mixing them is the bug that flattened 17 catalogue entries.

    python ns_probe/make_topo_seed.py --out runs/fold24_topo_seed
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nslens.qd.descriptors import TIR_FACE_INDEX, TIR_FACE_VOCAB  # noqa: E402

# keep in step with build_catalogue.RELATIVE_SAG_RUNS
RELATIVE_SAG_RUNS = {"fold24_clean", "fold24_clean_v3"}


def tir_faces_of(signature: str):
    """Formatted signature string -> `tir_faces` coordinate, or None if out of vocab."""
    faces = []
    for tok in signature.split(" -> "):
        if "@" not in tok:
            continue
        kind, face = tok.split("@")
        if kind.strip().upper() == "TIR":
            faces.append(face.strip())
    idx = TIR_FACE_INDEX.get(tuple(sorted(Counter(faces).items())))
    return None if idx is None else float(idx) + 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-archive", default="runs/fold24_clean_seed/archive.json")
    ap.add_argument("--catalogue", default="runs/catalogue.json")
    ap.add_argument("--out", default="runs/fold24_topo_seed")
    a = ap.parse_args()

    cells, skipped = [], 0

    src = Path(a.seed_archive)
    if src.exists():
        prev = json.loads(src.read_text())
        for c in prev["cells"]:
            v = tir_faces_of(c.get("dom_signature") or "")
            if v is None:
                skipped += 1
                continue
            rec = dict(c)
            rec["tir_faces"] = v
            cells.append(rec)
        print(f"{len(cells)} cells from {src}")

    cat = Path(a.catalogue)
    if cat.exists():
        n0 = len(cells)
        for e in json.loads(cat.read_text())["entries"]:
            if e["source_run"] not in RELATIVE_SAG_RUNS:
                continue
            if e["verdict"] != "PASS" or (e.get("hardened_fitness") or 0) <= 0.001:
                continue
            sig = e.get("hardened_signature") or e["signature"]
            v = tir_faces_of(sig)
            if v is None:
                skipped += 1
                continue
            cells.append({
                "tir_faces": v,
                "thinness_T": float(e.get("thinness") or 0.0),
                "tir_count": e.get("tir_count"),
                "dom_signature": sig,
                "fitness": float(e["hardened_fitness"]),
                "genotype": [float(g) for g in e["genotype"]],
                "spot": e.get("spot"),
                "per_field": e.get("per_field"),
                "from_catalogue": True,
            })
        print(f"{len(cells)-n0} cells from {cat} (relative-sag, hardened survivors)")

    # Dedupe by FULL SIGNATURE, not by archive cell. This file is the COMMON seed for
    # both arms of the comparison, and the two arms bin it differently by construction --
    # collapsing on either arm's cell coordinates would hand that arm a seed already
    # shaped to its own grid, which is the confound the common seed exists to remove.
    # Each arm's GridArchive collapses whatever it cannot separate when it loads.
    best = {}
    for c in cells:
        key = c.get("dom_signature") or repr(c["genotype"])
        if key not in best or c["fitness"] > best[key]["fitness"]:
            best[key] = c
    out_cells = sorted(best.values(), key=lambda c: -c["fitness"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "archive.json").write_text(json.dumps({
        "meta": {"experiment": "fold24_topo_seed",
                 "note": "warm start for the tir_faces axis; relative-sag genomes only",
                 "sources": [str(src), str(cat)],
                 "bc_axes": ["tir_faces", "thinness_T"]},
        "history": [],
        "cells": out_cells,
    }))
    topos = {round(c["tir_faces"]) for c in out_cells}
    print(f"\nwrote {out/'archive.json'}: {len(out_cells)} cells, "
          f"{len(topos)} distinct topologies, {skipped} skipped (out of vocab)")
    print(f"best fitness {max(c['fitness'] for c in out_cells):.3f}")
    for c in out_cells[:8]:
        m = TIR_FACE_VOCAB[int(c["tir_faces"])]
        label = " ".join(f"{f}{n}" for f, n in m) or "unfolded"
        print(f"  {c['fitness']:.3f}  {label:12s}  T={c['thinness_T']:.1f}  "
              f"{c['dom_signature']}")


if __name__ == "__main__":
    main()
