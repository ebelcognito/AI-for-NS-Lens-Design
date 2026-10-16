"""Problems: genome + how a genotype becomes a traceable scene.

A Problem owns the three things that used to be copy-pasted module constants at the top
of every runner: the genome box, the source/detector placement, and the face classifier.
Constants are stated here rather than imported from ns_probe so that nslens does not
depend on the legacy tree; tests/test_qd_equivalence.py asserts they still match.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..engine.signatures import make_single_element_face_classifier
from ..geometry import Element
from ..scene import Detector, Layout, ParallelBundle


@dataclass
class Problem:
    """Base: a genome box plus a genotype -> Layout mapping."""

    param_names: Sequence[str]
    lo: np.ndarray
    hi: np.ndarray
    half_h: float
    x_sensor: float
    focus_w: float = 8.0
    n_rays: int = 21
    beam_w: float = 8.0
    #: reject any element with a face steeper than this |sag|/half-chord. 0.268 is
    #: radius >= 2x half-chord (a gentle cap); above 1.0 the arc exceeds 180 deg and the
    #: "face" is a bulb. None = the legacy behaviour, which permits q up to 5.0.
    max_sag_ratio: Optional[float] = None
    #: interpret the genome's sag genes as FRACTIONS of their own face's half-chord
    #: rather than absolute mm. A face's chord is not fixed -- top/bottom scale with
    #: `thick` -- so an absolute sag box means something different at every thickness,
    #: which is what lets a 6 mm-thick element carry a 15 mm bulb.
    relative_sag: bool = False
    #: required clear distance between the element's right-most point and the detector
    #: plane. None = the legacy behaviour, which is NO check: PrismFoldProblem has always
    #: enforced a 12 mm gap via its own is_valid, and FoldProblem has never enforced
    #: anything, so a fold env could hand back a design whose glass passes THROUGH the
    #: sensor. Measured on the archives that is not hypothetical -- 1 to 6 elites per
    #: fold24 run sit past x_sensor, and their thinness_T is meaningless because T is
    #: measured against a plane the body intersects.
    #:
    #: Left None by default because switching it on rejects genotypes the frozen
    #: archives contain, which would break tests/test_qd_equivalence.py's cell-for-cell
    #: comparison. Opt in per experiment, exactly as param_scaling does.
    det_clearance: Optional[float] = None

    #: genome sag key -> the half-chord it is relative to under `relative_sag`
    _SAG_CHORD = {"fsag": "half_h", "bsag": "half_h",
                  "tsag": "half_thick", "dsag": "half_thick"}

    def vec_to_params(self, x: Sequence[float]) -> Dict[str, float]:
        p = dict(zip(self.param_names, [float(v) for v in x]))
        if self.relative_sag:
            # nominal chords (shear ignored here; the exact ratio is re-checked by
            # max_sag_ratio in is_valid, which uses the sheared corners)
            chord = {"half_h": self.half_h, "half_thick": p.get("thick", 0.0) / 2.0}
            for key, which in self._SAG_CHORD.items():
                if key in p:
                    p[key] = p[key] * chord[which]
        return p

    def element(self, p: Dict[str, float]) -> Element:
        return Element.from_legacy_params(p, half_h=self.half_h)

    def face_of(self, p: Dict[str, float]):
        return make_single_element_face_classifier(p["cx"], self.half_h)

    def layout(self, p: Dict[str, float], **kw) -> Layout:
        raise NotImplementedError

    def is_valid(self, p: Dict[str, float]) -> bool:
        el = self.element(p)
        if not el.is_valid():
            return False
        return self.sag_ok(el) and self.detector_clear(el)

    def sag_ok(self, el) -> bool:
        """No face steeper than max_sag_ratio (no-op when it is None)."""
        return self.max_sag_ratio is None or el.max_sag_ratio() <= self.max_sag_ratio

    def detector_clear(self, el) -> bool:
        """Body stops short of the detector plane (no-op when det_clearance is None)."""
        return (self.det_clearance is None
                or el.x_extent()[1] < self.x_sensor - self.det_clearance)


@dataclass
class FoldProblem(Problem):
    """The collinear fold env: a parallel bundle in from the left, a detector on the
    right, and the 9-DOF tilt-enabled single element. Prismatic tilt is what lets a
    whole bundle exceed the critical angle, so it is the mechanism behind every real
    TIR fold here -- which is why experiment #7 ablates it.

    Backs experiments #5 (fold-QD), #6 (orderly), #7 (tilt ablation) and #8 (multifield).
    """

    param_names: Sequence[str] = field(default_factory=lambda: [
        "ray_y", "ang", "cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b", "n"])
    lo: np.ndarray = field(default_factory=lambda: np.array(
        [-20.0, -55.0, 35.0, 6.0, -18.0, -18.0, -25.0, -25.0, 1.50]))
    hi: np.ndarray = field(default_factory=lambda: np.array(
        [20.0, 55.0, 85.0, 45.0, 18.0, 18.0, 25.0, 25.0, 2.00]))
    half_h: float = 30.0
    x_sensor: float = 120.0

    def layout(self, p: Dict[str, float], *, field_deg: float = 0.0) -> Layout:
        return Layout(
            source=ParallelBundle(center=p["ray_y"], aim_deg=p["ang"] + field_deg,
                                  width=self.beam_w, n_rays=self.n_rays,
                                  axis="y", origin=0.0),
            element=self.element(p),
            detector=Detector.vertical(self.x_sensor),
        )


@dataclass
class PrismFoldProblem(Problem):
    """The perpendicular env: the bundle enters from BELOW travelling up, the detector
    sits on the right, so straight transmission is geometrically impossible and the
    bundle must be TIR-folded ~90 deg. Uses the 4-curved-face prism because a +-25 deg
    face cannot turn 90 deg.
    """

    param_names: Sequence[str] = field(default_factory=lambda: [
        "ray_x", "cx", "thick", "fsag", "bsag", "tsag", "dsag",
        "tilt_a", "tilt_b", "n"])
    lo: np.ndarray = field(default_factory=lambda: np.array(
        [15.0, 35.0, 6.0, -15.0, -15.0, -15.0, -15.0, -50.0, -50.0, 1.50]))
    hi: np.ndarray = field(default_factory=lambda: np.array(
        [105.0, 85.0, 45.0, 15.0, 15.0, 15.0, 15.0, 50.0, 50.0, 2.00]))
    half_h: float = 30.0
    x_sensor: float = 170.0
    y_source: float = -75.0
    det_gap: float = 12.0

    def layout(self, p: Dict[str, float], *, field_deg: float = 0.0) -> Layout:
        return Layout(
            source=ParallelBundle(center=p["ray_x"], aim_deg=90.0 + field_deg,
                                  width=self.beam_w, n_rays=self.n_rays,
                                  axis="x", origin=self.y_source),
            element=self.element(p),
            detector=Detector.vertical(self.x_sensor),
        )

    def is_valid(self, p: Dict[str, float]) -> bool:
        """Body must be non-degenerate AND stay clear of the detector plane, otherwise
        the detector cuts through the glass."""
        el = self.element(p)
        if not el.is_valid() or not self.sag_ok(el):
            return False
        return el.x_extent()[1] < self.x_sensor - self.det_gap


#: tilt parameter indices, zeroed by the experiment #7 ablation
TILT_INDICES = (6, 7)


def lock_tilt(x: np.ndarray, indices: Sequence[int] = TILT_INDICES) -> np.ndarray:
    """Experiment #7: force face tilts to 0 (a rotationally-symmetric-ish element)."""
    out = np.asarray(x, dtype=float).copy()
    for i in indices:
        out[i] = 0.0
    return out
