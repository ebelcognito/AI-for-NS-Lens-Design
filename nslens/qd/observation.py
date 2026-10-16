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

from dataclasses import dataclass, field
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
    #: (min_rays, n_glass) -> beam_legs result. See beam_legs; not part of the value.
    _legs_cache: Dict[tuple, List[dict]] = field(default_factory=dict, repr=False,
                                                 compare=False)

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

    # ------------------------------------------------------------------ beam packing
    def dominant_rays(self, *, min_rays: int = 3) -> List[dict]:
        """The records sharing the dominant-energy signature -- the bundle everything
        in this section measures. Empty below `min_rays`: a spread of two rays is not
        a bundle."""
        if self.dominant_signature is None:
            return []
        grp = [r for r in self.records if r.get("_sig") == self.dominant_signature]
        return grp if len(grp) >= min_rays else []

    def beam_legs(self, *, min_rays: int = 3, n_glass: Optional[float] = None
                  ) -> List[dict]:
        """Memoised wrapper -- see `_beam_legs` for what this measures.

        `convergence()` walks the legs and then calls `focus_crossings()`, which walks them
        again with the same arguments, so every consumer paid for the profile twice. The
        result is a pure function of an Observation that nothing mutates after `measure`
        builds it, so caching it changes no number. Measured on the RL reward path (nine
        fields, full beam diagnostics per field) this is most of the 63% of wall clock that
        was Python rather than engine.
        """
        key = (min_rays, None if n_glass is None else float(n_glass))
        if key not in self._legs_cache:
            self._legs_cache[key] = self._beam_legs(min_rays=min_rays, n_glass=n_glass)
        return self._legs_cache[key]

    def _beam_legs(self, *, min_rays: int = 3, n_glass: Optional[float] = None
                   ) -> List[dict]:
        """Full per-leg profile of the dominant bundle: where it is, how wide it is, and
        HOW FAST IT IS SPREADING, after every surface it touches.

        Leg k runs from interaction vertex k to vertex k+1, so leg k is what the bundle
        looks like immediately AFTER the surface event at vertex k. Rays travel in
        straight lines between surfaces, so each ray's direction on the leg is exactly
        `P[k+1] - P[k]` -- these angles are measured, not differenced approximations.

        Width alone cannot tell packing from a caustic: a bundle crossing through a line
        focus inside the glass is spatially narrow precisely BECAUSE it is angularly
        wide. The pair (width, spread) separates them, and the phase-space area they
        bound says whether the element is redirecting light or wrecking it:

            width      perpendicular peak-to-peak of the ray positions, in the leg's own
                       frame, so grazing incidence does not inflate it
            spread     full angular extent of the ray directions about the chief ray
            rms        energy-weighted RMS angular deviation -- the spread the BULK of
                       the light actually has, not its worst outlier
            emittance  the real invariant: sqrt(<s^2><u^2> - <s u>^2) over the bundle,
                       with s the transverse position and u = n sin(angle) the reduced
                       angle. This is the AREA the bundle occupies in phase space.
                       Free propagation shears it, a flat surface leaves u alone, an
                       ideal curved surface shears it the other way -- all determinant 1,
                       so a perfect element holds emittance FLAT. It grows only where
                       figure error tears rays of one bundle apart, and no downstream
                       surface can put that back.
            etendue    the naive width * n * sin(spread/2). Reported because it is the
                       familiar number, but it is a BOUNDING BOX on the phase-space area,
                       tight only where the bundle has no position-angle correlation. It
                       therefore grows on any converging leg even with zero aberration --
                       read `emittance` for aberration, not this.

        `emittance` and `etendue` need `n_glass`: the index belongs to the problem, not to
        the observation. Legs inside the glass are found by parity -- every `refract` on
        the signature crosses the boundary, `tir` and `mirror` do not.
        """
        grp = self.dominant_rays(min_rays=min_rays)
        if not grp:
            return []
        sig = self.dominant_signature
        n_v = min(len(r["seq"]) for r in grp)
        legs: List[dict] = []
        n_refract = 0
        for k in range(n_v - 1):
            P = np.array([[r["seq"][k]["x"], r["seq"][k]["y"]] for r in grp], float)
            D = np.array([[r["seq"][k + 1]["x"] - r["seq"][k]["x"],
                           r["seq"][k + 1]["y"] - r["seq"][k]["y"]] for r in grp], float)
            L = np.linalg.norm(D, axis=1)
            if np.any(L < 1e-12):
                return []
            U = D / L[:, None]
            # energy on THIS leg: what continued past the event that launched it
            b = np.array([float(r["seq"][k].get("bOut", r.get("b", 0.0))) for r in grp])
            wts = b if b.sum() > 0 else np.ones(len(grp))

            # chief direction from the mean DISPLACEMENT, not the mean unit vector: the
            # two differ whenever rays have unequal path length on a leg, and this is the
            # convention every width in the archives was measured with. RMS spread is
            # taken about the energy-weighted mean angle below, so it does not inherit
            # this choice -- only `width` and `spread_deg` are referred to the chief ray.
            d = D.mean(axis=0)
            nd = np.linalg.norm(d)
            if nd < 1e-12:      # rays anti-parallel: no meaningful chief ray
                return []
            d = d / nd
            perp = np.array([-d[1], d[0]])
            # signed angle to the chief ray via the cross product -- no atan2 wraparound
            ang = np.arcsin(np.clip(U @ perp, -1.0, 1.0))
            ang = np.where(U @ d < 0, np.pi - ang, ang)     # keep back-turned rays sane

            width = float(np.ptp(P @ perp))
            spread = float(np.ptp(ang))
            rms = float(np.sqrt(np.average((ang - np.average(ang, weights=wts)) ** 2,
                                           weights=wts)))
            inside = bool(n_refract % 2)
            leg = dict(
                k=k, width=width,
                spread_deg=float(np.degrees(spread)),
                rms_deg=float(np.degrees(rms)),
                chief_deg=float(np.degrees(np.arctan2(d[1], d[0]))),
                inside=inside, energy=float(b.sum()),
                event=("source" if k == 0 or sig is None or k > len(sig)
                       else f"{sig[k-1][0]}@{sig[k-1][1]}"),
            )
            if n_glass is not None:
                n_leg = float(n_glass) if inside else 1.0
                leg["etendue"] = width * n_leg * float(np.sin(spread / 2.0))
                s = P @ perp
                s = s - np.average(s, weights=wts)
                u = n_leg * np.sin(ang - np.average(ang, weights=wts))
                s2 = float(np.average(s * s, weights=wts))
                u2 = float(np.average(u * u, weights=wts))
                su = float(np.average(s * u, weights=wts))
                leg["emittance"] = float(np.sqrt(max(s2 * u2 - su * su, 0.0)))
            legs.append(leg)
            if k < len(sig or ()) and sig[k][0] == "refract":
                n_refract += 1
        return legs

    def beam_widths(self, *, min_rays: int = 3) -> List[float]:
        """Perpendicular width of the dominant bundle on each leg of its path.

        A fold exists to PACK light into a small volume. If the bundle leaves a surface
        wider than it arrived, that fold has cost volume and bought nothing -- so the
        width profile along the path is the thing that says whether a design is doing
        its job, and it is measurable from the sequences we already record.

        The width column of `beam_legs()`; see there for the angular columns that say
        whether a narrow bundle is packed or merely crossing through a focus.
        """
        return [leg["width"] for leg in self.beam_legs(min_rays=min_rays)]

    def focus_crossings(self, *, min_rays: int = 3, legs: Optional[List[dict]] = None
                        ) -> dict:
        """Where the dominant bundle passes through an internal focus.

        At a line focus the rays CROSS, so their transverse ordering inverts. That is a
        threshold-free signature of a caustic -- unlike "the width got small", it does not
        need a length scale to compare against.

        One correction makes it usable: the transverse axis is defined as the chief ray
        rotated a quarter turn, so every reflection (tir, mirror) flips the sign
        convention even though the physical bundle order is untouched. Reflection parity
        is known exactly from the signature, so it is divided back out. What survives is
        real ray crossing.

        Returns the leg indices AFTER which the order inverts, and the correlation of
        each leg's ordering with the entry leg's (parity-corrected, so +1 = same order).

        `legs` lets a caller that already walked the profile hand it over. Only `k` and
        `chief_deg` are read here and neither depends on `n_glass`, so the legs computed
        with an index are the same legs -- which is why `convergence()` passes its own
        rather than paying for a second identical walk.
        """
        legs = self.beam_legs(min_rays=min_rays) if legs is None else legs
        grp = self.dominant_rays(min_rays=min_rays)
        if len(legs) < 2:
            return dict(measured=False, crossings=[], n_crossings=0, order_corr=[])
        sig = self.dominant_signature or ()

        # transverse coordinate of each ray on each leg, in that leg's own frame
        coords = []
        parity = 1
        for leg in legs:
            k = leg["k"]
            th = np.radians(leg["chief_deg"])
            perp = np.array([-np.sin(th), np.cos(th)])
            P = np.array([[r["seq"][k]["x"], r["seq"][k]["y"]] for r in grp], float)
            coords.append(parity * (P @ perp))
            if k < len(sig) and sig[k][0] in ("tir", "mirror"):
                parity = -parity

        ref = np.argsort(np.argsort(coords[0])).astype(float)
        corr, crossings = [], []
        prev = 1.0
        for i, c in enumerate(coords):
            r = np.argsort(np.argsort(c)).astype(float)
            if np.std(r) < 1e-12 or np.std(ref) < 1e-12:
                corr.append(float("nan"))
                continue
            rho = float(np.corrcoef(ref, r)[0, 1])
            corr.append(rho)
            if i and prev > 0 and rho < 0:
                crossings.append(i)          # order inverted entering leg i
            if not np.isnan(rho):
                prev = rho
        return dict(measured=True, crossings=crossings, n_crossings=len(crossings),
                    order_corr=[round(float(v), 3) for v in corr])

    def exit_convergence(self, *, min_rays: int = 3) -> dict:
        """Where the dominant bundle focuses on the ONE leg `convergence()` cannot see:
        the exit face to the detector.

        `convergence()` is blind here twice over, and both are structural rather than
        oversights:

          * every leg's `width` is measured at the vertex that STARTS it (beam_legs takes
            `P` at index k), so the final leg contributes the bundle width at the exit
            FACE. The width at the detector plane never enters `packing` at all.
          * `focus_crossings` deliberately excludes an inversion on the final leg, because
            focusing onto the sensor is the objective rather than a defect.

        Between them, a bundle that leaves the glass narrow and then fans out over the gap
        to the sensor passes `clean_pack` untouched. Nothing rejects it; `focused_throughput`
        merely scores it near zero. That is fine for ranking WITHIN a run and wrong as a
        gate, because a weakly-occupied archive cell is then held by a design that never
        forms an image.

        Measured by propagating the exit bundle analytically, which needs no extra trace.
        With `a_i` the transverse position of ray i referred to the mean exit point and
        `m_i` its transverse slope, both in the chief-ray frame, the ray is
        `y_i(t) = a_i + m_i t` along the chief axis, so the energy-weighted variance

            var(t) = var(a) + 2 t cov(a, m) + t^2 var(m)

        is a parabola whose minimum is the geometric focus:

            focus_dist   t* = -cov(a, m) / var(m), the distance from the exit face to the
                         waist along the chief ray. NEGATIVE means the waist is behind the
                         exit face -- the bundle is already past its focus and diverging.
            sensor_dist  distance from the exit face to the detector, same axis
            focus_ratio  t* / sensor_dist. 1.0 = focuses exactly on the detector, > 1
                         undershoots (still converging when it arrives), < 0 diverging.
            rms_exit / rms_sensor / rms_focus   bundle RMS at those three planes
            sensor_ratio rms_sensor / rms_exit; < 1 means it did narrow on the way

        `converges` is the honest weak verdict -- a real waist ahead of the exit face AND
        a bundle narrower at the detector than it was at the glass. It deliberately does
        not require the focus to LAND on the detector: how close t* comes to sensor_dist
        is what fitness already prices, and duplicating that in a gate would just be spot
        size with extra steps.

        A perfectly collimated exit (var(m) ~ 0) has no waist at any finite distance; it
        reports `focus_dist=None` and does not converge, which is correct -- an element
        that hands the detector a parallel beam has not formed an image.
        """
        unmeasured = dict(measured=False, focus_dist=None, sensor_dist=None,
                          focus_ratio=None, rms_exit=None, rms_sensor=None,
                          rms_focus=None, sensor_ratio=None, width_exit=None,
                          width_sensor=None, converges=False)
        grp = self.dominant_rays(min_rays=min_rays)
        if len(grp) < min_rays:
            return unmeasured
        n_v = min(len(r["seq"]) for r in grp)
        if n_v < 2:
            return unmeasured

        P = np.array([[r["seq"][n_v - 2]["x"], r["seq"][n_v - 2]["y"]] for r in grp], float)
        Q = np.array([[r["seq"][n_v - 1]["x"], r["seq"][n_v - 1]["y"]] for r in grp], float)
        D = Q - P
        if np.any(np.linalg.norm(D, axis=1) < 1e-12):
            return unmeasured
        d = D.mean(axis=0)
        nd = np.linalg.norm(d)
        if nd < 1e-12:
            return unmeasured
        d = d / nd
        perp = np.array([-d[1], d[0]])
        # energy leaving the exit face into this leg, matching beam_legs' convention
        b = np.array([float(r["seq"][n_v - 2].get("bOut", r.get("b", 0.0))) for r in grp])
        wts = b if b.sum() > 0 else np.ones(len(grp))

        # longitudinal, so the exit face being curved or tilted does not smear the fit
        P0 = np.average(P, axis=0, weights=wts)
        t0 = (P - P0) @ d
        along = D @ d
        if np.any(np.abs(along) < 1e-12):        # a ray running along the exit plane
            return unmeasured
        m = (D @ perp) / along
        a = ((P - P0) @ perp) - m * t0           # transverse at chief-coordinate t = 0

        def rms_at(t: float) -> float:
            y = a + m * t
            return float(np.sqrt(np.average((y - np.average(y, weights=wts)) ** 2,
                                            weights=wts)))

        def ptp_at(t: float) -> float:
            return float(np.ptp(a + m * t))

        am = np.average(a, weights=wts)
        mm = np.average(m, weights=wts)
        var_m = float(np.average((m - mm) ** 2, weights=wts))
        cov_am = float(np.average((a - am) * (m - mm), weights=wts))
        sensor_dist = float(np.average((Q - P0) @ d, weights=wts))

        focus_dist = None if var_m <= 1e-18 else float(-cov_am / var_m)
        rms_exit, rms_sensor = rms_at(0.0), rms_at(sensor_dist)
        return dict(
            measured=True,
            focus_dist=focus_dist,
            sensor_dist=sensor_dist,
            focus_ratio=(None if focus_dist is None or abs(sensor_dist) < 1e-9
                         else float(focus_dist / sensor_dist)),
            rms_exit=rms_exit, rms_sensor=rms_sensor,
            rms_focus=(None if focus_dist is None else rms_at(focus_dist)),
            sensor_ratio=(float(rms_sensor / rms_exit) if rms_exit > 1e-12 else None),
            width_exit=ptp_at(0.0), width_sensor=ptp_at(sensor_dist),
            converges=bool(focus_dist is not None and focus_dist > 0.0
                           and rms_sensor < rms_exit),
        )

    def convergence(self, *, rise_tol: float = 0.02, envelope_tol: float = 0.20,
                    regrow_tol: float = 1.5, skip_source_leg: bool = True,
                    n_glass: Optional[float] = None) -> dict:
        """Is the bundle packing on its way to the exit?

        The source leg (collimated, in air, ahead of the element) is skipped: it says
        nothing about what the element does.

        `monotone` is the STRICT reading -- every leg at most (1 + rise_tol) times the
        one before. Measured against the archives it is too strict to be a gate: the best
        design in fold24_wide (worst-field 0.814) widens 5.4% between the entry surface
        and its first internal bounce, then converges 45%, and strict monotonicity throws
        it away along with everything better than 0.441.

        `packing` is the envelope reading, and is what the gate uses. Three conditions,
        each with its own physical job:

            ratio    < 1                the beam leaves narrower than it entered, so the
                                        fold paid for itself
            envelope <= envelope_tol    it never bulges appreciably wider than its entry
                                        width along the way
            regrow   <= regrow_tol      it does not re-expand after its narrowest point,
                                        which is what separates a packed beam from a
                                        CAUSTIC inside the glass -- an internal line
                                        focus carries the intensity and figure
                                        sensitivity that would not survive manufacture

        On fold24_wide's 45 designs this keeps 27 including the best, and rejects every
        design whose bundle expands (their regrow runs 3x to 15x).

        `packing` is a WIDTH-ONLY test, and width alone cannot tell a packed beam from a
        bundle crossing through a line focus -- the caustic is narrow precisely because
        it is angularly wide, and `regrow` only sees it if it re-expands before the exit.
        The angular columns close that:

            entry/exit/waist_spread   energy-weighted RMS angular spread (deg) of the
                                      bundle on the entry leg, the exit leg, and at the
                                      waist. A packed beam trades width for angle; a
                                      caustic spikes at the waist and stays wide.
            spread_growth             max RMS spread / entry RMS spread
            etendue_growth            same for width * n * sin(spread/2), which refraction
                                      and TIR both conserve. > 1 means figure error is
                                      shearing the bundle apart, and nothing downstream
                                      can undo it. Requires `n_glass`.
            n_crossings / crossed     internal focus crossings from focus_crossings(),
                                      counted threshold-free by ray-order inversion. A
                                      crossing on the final leg is excluded: focusing
                                      onto the sensor is the objective, not a defect.
            clean_pack                `packing` AND no internal crossing -- the verdict to
                                      gate on if you want packed beams and not caustics.
        """
        legs = self.beam_legs(n_glass=n_glass)
        w = [leg["width"] for leg in legs]
        start = 1 if (skip_source_leg and len(w) > 1) else 0
        seg = w[start:]
        if len(seg) < 2 or seg[0] <= 1e-9:
            return dict(measured=False, ratio=None, monotone=False, packing=False,
                        max_rise=None, envelope=None, regrow=None,
                        waist=None, waist_at=None, n_legs=len(seg),
                        entry_spread=None, exit_spread=None, waist_spread=None,
                        spread_growth=None, etendue_growth=None,
                        emittance_exit=None, emittance_max=None, order_corr=[],
                        n_crossings=0, crossed=False, clean_pack=False)
        ratio = float(seg[-1] / seg[0])
        max_rise = max((b / a - 1.0) if a > 1e-9 else float("inf")
                       for a, b in zip(seg, seg[1:]))
        envelope = float(max(seg) / seg[0] - 1.0)
        waist = float(min(seg))
        regrow = float(seg[-1] / waist) if waist > 1e-6 else float("inf")
        packing = bool(ratio < 1.0 and envelope <= envelope_tol and regrow <= regrow_tol)

        # --- angular: what width alone cannot see -------------------------------
        sl = legs[start:]
        rms = [leg["rms_deg"] for leg in sl]
        entry_spread, exit_spread = rms[0], rms[-1]
        waist_spread = rms[int(np.argmin(seg))]
        spread_growth = (float(max(rms) / entry_spread) if entry_spread > 1e-9
                         else float("inf"))
        et = [leg.get("etendue") for leg in sl]
        etendue_growth = None
        if n_glass is not None and et[0] and et[0] > 1e-12:
            etendue_growth = float(max(et) / et[0])
        em = [leg.get("emittance") for leg in sl]
        emittance_exit = float(em[-1]) if n_glass is not None else None
        emittance_max = float(max(em)) if n_glass is not None else None
        cr = self.focus_crossings(legs=legs)
        # a crossing before the LAST leg is a focus inside the element; one on the way to
        # the sensor is just the design focusing onto it, which is the whole point
        internal = [i for i in cr["crossings"] if i < len(legs) - 1]

        return dict(measured=True, ratio=ratio,
                    entry_spread=entry_spread, exit_spread=exit_spread,
                    waist_spread=waist_spread, spread_growth=spread_growth,
                    etendue_growth=etendue_growth,
                    emittance_exit=emittance_exit, emittance_max=emittance_max,
                    n_crossings=len(internal), crossed=bool(internal),
                    order_corr=cr["order_corr"],
                    clean_pack=bool(packing and not internal),
                    monotone=bool(max_rise <= rise_tol),
                    packing=bool(ratio < 1.0 and envelope <= envelope_tol
                                 and regrow <= regrow_tol),
                    max_rise=float(max_rise), envelope=envelope, regrow=regrow,
                    waist=waist, waist_at=int(np.argmin(seg)), n_legs=len(seg))

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
