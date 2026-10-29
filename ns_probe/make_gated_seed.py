"""Warm-start archives for the gated runs, in both the 2-curved and 3-curved env.

A 14.7% gate stack cannot cold-start (driver.py, gate_all_fields: 0-3 of 48 cells filled
in 4k evals from empty), and adding the exit-convergence gate makes it stricter still. So
both gated experiments resume, and this builds what they resume from.

Sources, pooled and re-scored under the run's OWN gated config so nothing enters on a
score it earned under looser rules:

  runs/fold24_common_seed   the 7-cell common seed doc 19 section 5.4 used, so the gated
                            re-run starts where the ungated one did
  catalogue/catalogue.json  the 13 hardened survivors -- already clear det_clearance=12,
                            max_sag_ratio=0.268 and clean_pack at all nine fields, and
                            all 13 pass the exit gate

ONE seed file feeds both arms of the current-env comparison, carrying tir_count AND
tir_faces on every cell so each arm reads the coordinate its own cell_bc_keys names. That
is the confound doc 19 section 5.3 records: the pilot gave the topology arm a seed built
from catalogue survivors -- stronger AND more diverse than the baseline's -- which made
its fitness column unquotable. Deduping is by FULL SIGNATURE rather than by either arm's
grid, for the same reason.

The 3-curved seed is the same genotypes LIFTED with dsag=0 -- a flat bottom face, i.e.
the identical optic -- and re-scored under the 3-curved problem. So that run starts from
exactly the 2-curved designs and can only win by using the new surface.

    python ns_probe/make_gated_seed.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nslens.engine import Runner                    # noqa: E402
from nslens.qd import experiments                   # noqa: E402
from nslens.qd.driver import _evaluate              # noqa: E402


def gather() -> list:
    """Genotypes to seed from, as 9-gene (2-curved) vectors."""
    out = []
    src = _REPO / "runs" / "fold24_common_seed" / "archive.json"
    if src.exists():
        for c in json.loads(src.read_text())["cells"]:
            out.append(("common_seed", [float(v) for v in c["genotype"]]))
    cat = _REPO / "catalogue" / "catalogue.json"
    if cat.exists():
        for e in json.loads(cat.read_text())["entries"]:
            # only the relative-sag entries share fold24's genome interpretation; an
            # absolute-sag genotype re-read as fractions is a different element (doc 19)
            if e["verdict"] == "PASS" and e.get("relative_sag"):
                out.append(("catalogue", [float(v) for v in e["genotype"]]))
    return out


def lift(g: list, index: int = 6, value: float = 0.0) -> list:
    """9-gene 2-curved genotype -> 10-gene 3-curved, with a FLAT new face."""
    return list(g[:index]) + [value] + list(g[index:])


def build(cfg, genotypes, out_dir: Path, note: str) -> None:
    cells, seen = [], {}
    with Runner(timeout_s=20) as runner:
        for src, g in genotypes:
            x = np.array(g, float)
            res = _evaluate(runner, cfg.problem, cfg, x)
            if res is None:
                continue
            info = res["info"]
            sig = info.get("dom_signature") or repr(g)
            if sig in seen and seen[sig]["fitness"] >= res["fitness"]:
                continue
            seen[sig] = dict(
                fitness=float(res["fitness"]),
                genotype=[float(v) for v in x],
                # both arms' coordinates on every cell -- see the module docstring
                tir_count=info.get("tir_count"),
                tir_faces=info.get("tir_faces"),
                thinness_T=info.get("thinness_T"),
                dom_signature=sig,
                spot=info.get("spot"),
                per_field=info.get("per_field"),
                exit_ok_all=info.get("exit_ok_all"),
                exit_width_sensor_worst=info.get("exit_width_sensor_worst"),
                source=src,
            )
    cells = sorted(seen.values(), key=lambda c: -c["fitness"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "archive.json").write_text(json.dumps({
        "meta": {"experiment": out_dir.name, "note": note,
                 "scored_under": cfg.name,
                 "bc_axes": ["tir_count", "tir_faces", "thinness_T"]},
        "history": [], "cells": cells}, indent=1))
    print(f"\nwrote {out_dir/'archive.json'}: {len(cells)} cells from "
          f"{len(genotypes)} candidates, scored under {cfg.name}")
    if cells:
        print(f"best {cells[0]['fitness']:.3f}, "
              f"{len({c['dom_signature'] for c in cells})} distinct signatures")
        for c in cells[:8]:
            print(f"  {c['fitness']:.3f}  T={c['thinness_T']:.1f}  "
                  f"{(c['source']):<10s} {c['dom_signature']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out2", default="runs/fold24_gated_seed")
    ap.add_argument("--out3", default="runs/fold24_gated3_seed")
    a = ap.parse_args()

    genos = gather()
    print(f"{len(genos)} candidate genotypes "
          f"({sum(1 for s, _ in genos if s == 'catalogue')} from the catalogue)")

    print("\n--- 2-curved (current env, gated) ---")
    build(experiments.get("fold24_topo_gated"), genos, _REPO / a.out2,
          "common seed + hardened catalogue survivors, re-scored with the exit gate "
          "and det_clearance on; feeds BOTH current-env arms")

    print("\n--- 3-curved (dsag lifted to 0 = flat bottom face) ---")
    build(experiments.get("fold24_topo3_gated"),
          [(s, lift(g)) for s, g in genos], _REPO / a.out3,
          "the 2-curved seed with a flat third face (dsag=0), re-scored under the "
          "3-curved genome; this run can only win by USING the new surface")


if __name__ == "__main__":
    main()
