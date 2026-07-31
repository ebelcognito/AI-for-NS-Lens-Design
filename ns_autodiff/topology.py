"""Extract a frozen interaction topology from a QD champion, using the JS engine as a
read-only oracle.

The differentiable forward (forward.py) needs, for one champion:
  * the ordered event list  [(face, interaction_type), ...]  of the dominant typed
    signature -- shared by every tracked ray (this is the "champion-recorded ray graph"
    the design doc says to unroll), and
  * the set of tracked input rays (by beam offset dy), each with the oracle's recorded
    sensor y and energy, and a frozen per-ray weight (its champion energy).

This mirrors ns_probe/refine_lm.py (dom_records_by_dy / dominant_signature) so the twin
refines exactly the same tracked sub-bundle, but it imports those tools read-only.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

import fold_demo as FD                                       # noqa: E402
from sequence_tools import (aggregate_by_signature,          # noqa: E402
                            make_single_element_face_classifier, signature_of)

PARAM_NAMES = FD.PARAM_NAMES


@dataclass
class FrozenTopology:
    cell: Tuple[int, int]
    p0: Dict[str, float]                       # champion genotype
    events: List[Tuple[str, str]]              # [(type, face), ...] dominant signature
    dys: np.ndarray                            # tracked beam offsets [m]
    oracle_y: np.ndarray                       # oracle sensor y per tracked ray [m]
    oracle_b: np.ndarray                       # oracle sensor energy per tracked ray [m]
    w0: np.ndarray                             # frozen per-ray weights [m]
    dom_share: float
    archive_spot: float
    meta: dict = field(default_factory=dict)

    def signature_str(self) -> str:
        pretty = {"refract": "T", "tir": "TIR", "partial_reflect": "PR",
                  "mirror": "M", "beam_split": "BS"}
        return " -> ".join(f"{pretty.get(t, t)}@{f}" for t, f in self.events)


def load_champion_genotype(archive_path: Path, cell: Tuple[int, int]):
    arc = json.loads(Path(archive_path).read_text())
    i0, i1 = cell
    hit = [c for c in arc["cells"] if int(c["i0"]) == i0 and int(c["i1"]) == i1]
    if not hit:
        raise SystemExit(f"no archive cell {cell}")
    c = hit[0]
    p = dict(zip(PARAM_NAMES, [float(v) for v in c["genotype"]]))
    return p, c


def _dom_records_by_dy(res, p, dom_sig):
    """Oracle (sensor_y, energy) of the strongest forward record whose typed signature
    equals dom_sig, keyed by beam offset dy. Same logic as refine_lm.dom_records_by_dy."""
    face_of = make_single_element_face_classifier(p["cx"], FD.ELEM_H)
    out = {}
    for r in res.get("raySequences", []):
        if r.get("dirx", 0) <= 0 or r.get("b", 0) <= 0:
            continue
        if signature_of(r, face_of) != dom_sig:
            continue
        dy = round(float(r["seq"][0]["y"]) - p["ray_y"], 6)
        if dy not in out or r["b"] > out[dy][1]:
            out[dy] = (float(r["y"]), float(r["b"]))
    return out


def extract(runner, archive_path: Path, cell: Tuple[int, int], *,
            n_rays: int = 121, beam_w: float = 8.0) -> FrozenTopology:
    p0, champ = load_champion_genotype(archive_path, cell)
    res, _ = runner.simulate(FD.scene(p0, n_rays=n_rays, beam_w=beam_w, record=True))
    if res.get("error"):
        raise SystemExit(f"oracle error tracing champion: {res['error']}")

    face_of = make_single_element_face_classifier(p0["cx"], FD.ELEM_H)
    by_sig = aggregate_by_signature(res.get("raySequences", []), face_of)
    if not by_sig:
        raise SystemExit("champion produced no classifiable sensor rays")
    dom_sig = max(by_sig, key=by_sig.get)

    recs = _dom_records_by_dy(res, p0, dom_sig)
    if not recs:
        raise SystemExit("no rays follow the dominant signature at refine density")
    dys = np.array(sorted(recs.keys()))
    oracle_y = np.array([recs[dy][0] for dy in dys])
    oracle_b = np.array([recs[dy][1] for dy in dys])

    emitted = float(res["energyLedger"]["emitted"])
    return FrozenTopology(
        cell=(int(champ["i0"]), int(champ["i1"])),
        p0=p0,
        events=[(t, f) for (t, f) in dom_sig],
        dys=dys, oracle_y=oracle_y, oracle_b=oracle_b, w0=oracle_b.copy(),
        dom_share=float(champ.get("dom_energy_share", float("nan"))),
        archive_spot=float(champ.get("spot", float("nan"))),
        meta=dict(n_rays=n_rays, beam_w=beam_w, emitted=emitted,
                  dom_energy=float(by_sig[dom_sig]),
                  tir_count=int(champ.get("tir_count", 0)),
                  n_tracked=len(dys)),
    )
