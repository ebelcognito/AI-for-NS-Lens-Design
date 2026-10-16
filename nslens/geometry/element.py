"""The parametric lens element.

ONE object replaces fold_demo.element, arvr_prism.prism_element, prism_fold (which had
no element of its own -- it imported arvr's) and ns_autodiff.element_torch.build_surfaces.
Those were never four geometries; they are four *configurations* of this one:

    fold_demo             arcs on left/right, flat top/bottom
    arvr_prism            arcs on all four faces
    search_prism_freeform freeform on all four faces

The element is a closed boundary path around a glass body: four corners, plus one arc
control point for each curved face. Corner positions come from the body parameters and
the per-face tilts; control points come from the per-face sag.

--------------------------------------------------------------------------------
FACE LABELS -- read this before touching anything downstream.

The runtime labels come from sequence_tools.make_single_element_face_classifier:

    y <= -half_h  ->  'T'      the BOTTOM edge
    y >= +half_h  ->  'D'      the TOP edge
    otherwise     ->  'A' (x < cx, left) or 'B' (x >= cx, right)

So 'T' is the bottom and 'D' is the top. arvr_prism's docstring claims the opposite
("T=top, D=bottom"); it is wrong, and element_torch's inline comment is right. Worse,
the legacy sag parameter names are crossed relative to the labels: `dsag` is applied to
the bottom edge (face 'T') and `tsag` to the top edge (face 'D').

nslens therefore uses unambiguous internal names -- left/right/bottom/top -- and pins
the legacy mapping in LEGACY_FACE_LABEL below, in one place, so the trap cannot spread.
Interaction signatures still read 'A'/'B'/'T'/'D' so existing archives stay comparable.

Note also that in a signature like `T@A -> TIR@D -> T@B` the token before '@' is the
EVENT (T = transmit, TIR, PR = partial reflection) and the token after it is the FACE.
The letter 'T' means transmit on the left of '@' and the bottom edge on the right.
--------------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional

import numpy as np

from .shapes import Arc, Flat, Freeform, Shape

#: internal face name -> legacy signature label produced by the face classifier
LEGACY_FACE_LABEL: Dict[str, str] = {
    "left": "A",
    "right": "B",
    "bottom": "T",   # yes, 'T' -- see the module docstring
    "top": "D",
}

#: legacy genotype sag key -> internal face name (the crossed pair is deliberate)
LEGACY_SAG_KEY: Dict[str, str] = {
    "fsag": "left",
    "bsag": "right",
    "dsag": "bottom",
    "tsag": "top",
}

FACE_ORDER = ("left", "right", "bottom", "top")


@dataclass(frozen=True)
class Face:
    """One boundary face: how much its corners are sheared, and its shape."""

    shape: Shape = field(default_factory=Flat)
    tilt_deg: float = 0.0

    @property
    def sag(self) -> float:
        return self.shape.sag_value()

    @property
    def is_curved(self) -> bool:
        return self.shape.emits_control_point


@dataclass(frozen=True)
class Element:
    """A single glass element, uncoated unless `reflectivity` is set.

    Uncoated means every surface event is honest Fresnel + TIR, so the only loss-free
    reflection available to a design is geometric TIR. Setting `reflectivity` forces a
    partial mirror -- the lossy route -- and is used only by the coating ablation.

    Only `left`/`right` tilts are physical in the legacy genotypes; `bottom`/`top` tilts
    exist for generality and default to 0, which reproduces the legacy corner positions
    exactly.
    """

    cx: float
    thick: float
    n: float
    half_h: float = 30.0
    faces: Mapping[str, Face] = field(default_factory=dict)
    reflectivity: Optional[float] = None

    def __post_init__(self) -> None:
        filled = {name: self.faces.get(name, Face()) for name in FACE_ORDER}
        unknown = set(self.faces) - set(FACE_ORDER)
        if unknown:
            raise ValueError(f"unknown face(s) {sorted(unknown)}; "
                             f"expected a subset of {FACE_ORDER}")
        object.__setattr__(self, "faces", filled)

    # ------------------------------------------------------------------ geometry
    @property
    def lx(self) -> float:
        return self.cx - self.thick / 2.0

    @property
    def rx(self) -> float:
        return self.cx + self.thick / 2.0

    def _shear(self, face: str) -> float:
        """Corner x-offset produced by a face tilt, = tan(tilt) * half_h."""
        return float(np.tan(np.radians(self.faces[face].tilt_deg)) * self.half_h)

    def corners(self) -> Dict[str, tuple]:
        """The four body corners. A left/right tilt shears that face's two corners in
        opposite directions, which is what lets a whole bundle exceed the critical
        angle -- the prismatic wedge is the mechanism behind every real TIR fold."""
        dA, dB = self._shear("left"), self._shear("right")
        H = self.half_h
        return {
            "bottom_left": (self.lx + dA, -H),
            "bottom_right": (self.rx + dB, -H),
            "top_right": (self.rx - dB, H),
            "top_left": (self.lx - dA, H),
        }

    def path(self) -> list:
        """The closed boundary path: corners interleaved with one arc control point per
        curved face. Vertex order matches the legacy element builders exactly, which is
        what makes this a drop-in replacement (see tests/test_element_equivalence.py).

        A Flat face simply contributes no control point, which is precisely how
        fold_demo's 6-vertex path relates to arvr_prism's 8-vertex path.
        """
        c = self.corners()
        bl, br, tr, tl = (c["bottom_left"], c["bottom_right"],
                          c["top_right"], c["top_left"])
        H = self.half_h

        out: list = [{"x": bl[0], "y": bl[1], "arc": False}]
        if self.faces["bottom"].is_curved:
            out.append({"x": (bl[0] + br[0]) / 2.0,
                        "y": -H - self.faces["bottom"].sag, "arc": True})
        out.append({"x": br[0], "y": br[1], "arc": False})
        if self.faces["right"].is_curved:
            out.append({"x": self.rx + self.faces["right"].sag, "y": 0.0, "arc": True})
        out.append({"x": tr[0], "y": tr[1], "arc": False})
        if self.faces["top"].is_curved:
            out.append({"x": (tr[0] + tl[0]) / 2.0,
                        "y": H + self.faces["top"].sag, "arc": True})
        out.append({"x": tl[0], "y": tl[1], "arc": False})
        if self.faces["left"].is_curved:
            out.append({"x": self.lx - self.faces["left"].sag, "y": 0.0, "arc": True})
        return out

    # ------------------------------------------------------------------ validity
    def is_valid(self) -> bool:
        """Non-degenerate body: both the bottom and the top edge must keep a positive
        width after shearing, otherwise the tilts have folded the element inside out."""
        c = self.corners()
        return ((c["bottom_right"][0] - c["bottom_left"][0]) > 1.0
                and (c["top_right"][0] - c["top_left"][0]) > 1.0)

    def sag_ratios(self) -> Dict[str, float]:
        """Per curved face: |sag| / half-chord, the scale-free curvature.

        This is the number the genome box SHOULD have been bounding. Sag is stored in
        absolute mm, but a face's chord is not fixed -- the top and bottom chords are
        `thick` wide (6..45 mm) while the left and right chords are set by half_h and the
        tilt shear. A +-15 mm sag bound is therefore a gentle cap on one face and an
        absurdity on another:

            q <= 0.268   radius >= 2 x half-chord, a gentle spherical cap
            q <= 1.0     at most a hemisphere
            q >  1.0     the arc exceeds 180 deg -- a BULB, not a face

        At thick=6 the legacy +-15 mm bound permits q = 5.0.
        """
        H = self.half_h
        dA, dB = self._shear("left"), self._shear("right")
        # left/right: the chord runs corner-to-corner and is tilted, so the sag measured
        # perpendicular to it is the horizontal sag foreshortened by cos(tilt)
        half_chord = {
            "left": float(np.hypot(dA, H)),
            "right": float(np.hypot(dB, H)),
            "bottom": abs(self.thick + dB - dA) / 2.0,
            "top": abs(self.thick - dB + dA) / 2.0,
        }
        cos_t = {"left": H / half_chord["left"] if half_chord["left"] else 1.0,
                 "right": H / half_chord["right"] if half_chord["right"] else 1.0,
                 "bottom": 1.0, "top": 1.0}
        out: Dict[str, float] = {}
        for name in FACE_ORDER:
            face = self.faces[name]
            if not face.is_curved:
                continue
            hc = half_chord[name]
            out[name] = abs(face.sag) * cos_t[name] / hc if hc > 1e-9 else float("inf")
        return out

    def max_sag_ratio(self) -> float:
        """The steepest face's curvature ratio; 0.0 when every face is flat."""
        return max(self.sag_ratios().values(), default=0.0)

    def x_extent(self) -> tuple:
        """(min, max) x over the whole boundary -- the horizontal footprint of the glass,
        control points included, so a bulging face is not missed."""
        xs = [v["x"] for v in self.path()]
        return min(xs), max(xs)

    # ------------------------------------------------------------------ adapters
    def to_scene_dict(self) -> dict:
        """The ray-optics JS engine representation (the 'port' for that backend)."""
        for name, face in self.faces.items():
            if isinstance(face.shape, Freeform) and not face.shape.is_pure_arc:
                raise ValueError(
                    f"face {name!r} carries an active freeform deviation, which the "
                    f"ray-optics scene format cannot express. Trace freeform designs "
                    f"with the torch backend instead.")
        obj = {"type": "Glass", "refIndex": float(self.n),
               "path": [{"x": float(v["x"]), "y": float(v["y"]), "arc": v["arc"]}
                        for v in self.path()]}
        if self.reflectivity is not None:
            obj["reflectivity"] = float(self.reflectivity)
        return obj

    def with_faces(self, **faces: Face) -> "Element":
        """Return a copy with some faces replaced."""
        return replace(self, faces={**self.faces, **faces})

    # ------------------------------------------------------------------ legacy IO
    @classmethod
    def from_legacy_params(cls, p: Mapping[str, float], *, half_h: float = 30.0,
                           reflectivity: Optional[float] = None) -> "Element":
        """Build from any of the legacy genotype dicts.

        Curvature is inferred from which sag keys are present, which is exactly what
        distinguished the old element builders: fold_demo supplies fsag/bsag only,
        arvr_prism supplies all four.
        """
        faces = {}
        for key, name in LEGACY_SAG_KEY.items():
            if key in p:
                faces[name] = Face(shape=Arc(float(p[key])))
        for name, tilt_key in (("left", "tilt_a"), ("right", "tilt_b")):
            tilt = float(p.get(tilt_key, 0.0))
            base = faces.get(name, Face())
            faces[name] = Face(shape=base.shape, tilt_deg=tilt)
        return cls(cx=float(p["cx"]), thick=float(p["thick"]), n=float(p["n"]),
                   half_h=half_h, faces=faces, reflectivity=reflectivity)

    def legacy_face_label(self, face: str) -> str:
        return LEGACY_FACE_LABEL[face]
