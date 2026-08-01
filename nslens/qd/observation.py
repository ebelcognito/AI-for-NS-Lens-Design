"""One traced scene, reduced to the quantities every fitness and descriptor is built on.

All six legacy QD runners open with the SAME ~20 lines: pull the energy ledger, keep the
forward sensor-arriving rays with a valid first-crossing signature, aggregate by
signature, take the dominant one, compute the energy centroid and the window energy.
They then differ in about six lines. This module is those twenty lines, once.

Counting rule (load-bearing, from run_fold_qd): a ray's energy is counted ONCE, by
keeping only records whose first-crossing signature is valid -- `signature_of` returns
None for re-crossings. Without that, focused throughput can exceed the input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..engine.signatures import aggregate_by_signature, signature_of


@dataclass
class Observation:
    """Reduced result of tracing one scene."""

    emitted: float
    records: List[dict]                 # forward sensor-arriving rays, counted once
    by_signature: Dict[tuple, float]
    dominant_signature: Optional[tuple]
    centroid: float
    spot: float
    delivered: float
    optic_x_min: Optional[float]

    @property
    def tir_count(self) -> int:
        """TIR bounces on the dominant-energy interaction sequence = fold depth."""
        return sum(1 for (t, _f) in (self.dominant_signature or ()) if t == "tir")

    @property
    def dominant_energy(self) -> float:
        if self.dominant_signature is None:
            return 0.0
        return float(self.by_signature[self.dominant_signature])

    @property
    def n_dominant(self) -> int:
        """Rays taking the dominant sequence (run_orderly_qd reports this as n_dom)."""
        return sum(1 for r in self.records if r.get("_sig") == self.dominant_signature)

    @property
    def is_empty(self) -> bool:
        """Nothing usable arrived: no forward rays, or no interaction signature."""
        return not self.records or self.dominant_signature is None

    def thinness(self, x_sensor: float) -> float:
        """T = sensor plane minus the frontmost optic surface actually used. Thin = small."""
        return x_sensor - (self.optic_x_min if self.optic_x_min is not None else 0.0)

    def window_energy(self, focus_w: float, *, dominant_only: bool = False) -> float:
        """Energy landing within +-focus_w of the centroid.

        dominant_only=True restricts to rays taking the dominant sequence, which is what
        turns 'focused throughput' into the COHERENT fitness used by run_orderly_qd.
        """
        recs = self.records
        if dominant_only:
            recs = [r for r in recs if r.get("_sig") == self.dominant_signature]
        if not recs:
            return 0.0
        ys = np.array([r["y"] for r in recs], dtype=float)
        bs = np.array([r["b"] for r in recs], dtype=float)
        cen = self.centroid
        if dominant_only:
            # the coherent variant re-centres on its own subset (see run_orderly_qd)
            cen = float(np.average(ys, weights=bs))
        return float(bs[np.abs(ys - cen) <= focus_w].sum())


def measure(result: dict, face_of: Callable, *, x_sensor: float,
            dominant_only_centroid: bool = False,
            allow_empty: bool = False) -> Optional[Observation]:
    """Reduce an engine result to an Observation.

    Three outcomes, matching the legacy runners exactly:

      None                  hard failure -- engine error. Always invalidates the
                            evaluation.
      empty Observation     no emitted energy, no forward records, or no signature.
                            Only produced when allow_empty=True, which is the
                            multifield case: a field that catches nothing contributes
                            zero rather than killing the whole evaluation
                            (run_multifield_qd.field_metrics returns focthru=0.0,
                            spot=99.0, dom_sig=None here). With allow_empty=False --
                            every single-field runner -- this is None instead.
      Observation           the normal case.
    """
    if result.get("error"):
        return None
    led = result.get("energyLedger") or {}
    emitted = float(led.get("emitted", 0.0))
    seqs = result.get("raySequences", []) or []

    records = []
    for r in seqs:
        if r.get("dirx", 0) > 0 and r.get("b", 0) > 0:
            sig = signature_of(r, face_of)
            if sig is not None:
                r = dict(r)
                r["_sig"] = sig
                records.append(r)
    if emitted <= 0 or not records:
        if not allow_empty:
            return None
        return Observation(emitted=emitted, records=[], by_signature={},
                           dominant_signature=None, centroid=0.0, spot=99.0,
                           delivered=0.0, optic_x_min=None)

    by_sig = aggregate_by_signature(seqs, face_of)
    dom_sig = max(by_sig, key=by_sig.get) if by_sig else None
    if dom_sig is None and not allow_empty:
        return None

    src = records
    if dominant_only_centroid:
        src = [r for r in records if r["_sig"] == dom_sig] or records
    ys = np.array([r["y"] for r in src], dtype=float)
    bs = np.array([r["b"] for r in src], dtype=float)
    centroid = float(np.average(ys, weights=bs))
    spot = float(np.sqrt(np.average((ys - centroid) ** 2, weights=bs)))

    optic_xs = [v["x"] for r in src for v in r["seq"] if 0.5 < v["x"] < x_sensor - 0.5]
    return Observation(
        emitted=emitted,
        records=records,
        by_signature=by_sig,
        dominant_signature=dom_sig,
        centroid=centroid,
        spot=spot,
        delivered=float(led.get("delivered", bs.sum())),
        optic_x_min=min(optic_xs) if optic_xs else None,
    )
