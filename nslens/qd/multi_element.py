"""Uncoated two-element spherical search space for structural path inception.

The mature fold experiments use one glass body.  Perturbing that body explores its
existing topology basins very effectively but cannot invent interactions with another
physical element.  This module adds exactly that degree of structural freedom while
retaining the same circular-arc surface representation and non-sequential tracer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..engine.signatures import make_single_element_face_classifier
from ..geometry import Element
from ..scene import Detector, ParallelBundle
from .problems import Problem


@dataclass(frozen=True)
class CompoundElement:
    """The small geometry interface used by the shared hardened evaluator."""

    elements: tuple[Element, ...]

    def is_valid(self) -> bool:
        return all(element.is_valid() for element in self.elements)

    def max_sag_ratio(self) -> float:
        return max((element.max_sag_ratio() for element in self.elements), default=0.0)

    def x_extent(self) -> tuple[float, float]:
        extents = [element.x_extent() for element in self.elements]
        return min(lo for lo, _ in extents), max(hi for _, hi in extents)

    def path(self) -> list[dict]:
        return [vertex for element in self.elements for vertex in element.path()]


@dataclass(frozen=True)
class MultiElementLayout:
    source: ParallelBundle
    elements: tuple[Element, ...]
    detector: Detector
    max_depth: int = 24
    record: bool = True
    crop: object | None = None

    def to_scene_dict(self) -> dict:
        objects = list(self.source.rays())
        objects.extend(element.to_scene_dict() for element in self.elements)
        objects.append(self.detector.to_obj())
        scene = {"version": 5, "maxRayDepth": self.max_depth, "objs": objects}
        if self.record:
            scene["recordInteractions"] = True
        if self.crop is not None:
            objects.append(self.crop.to_obj())
        return scene


@dataclass
class TwoElementSphericalProblem(Problem):
    """Two separated, uncoated glass bodies with three circular-arc faces each.

    Both bodies share a refractive index.  This keeps intended-path critical-margin
    checks exact while still exposing fourteen independent pose/shape variables.
    ``dsag`` curves the physical bottom face; the top face stays planar, matching the
    best-tested three-curved single-element representation.
    """

    param_names: Sequence[str] = field(default_factory=lambda: [
        "ray_y", "ang", "n",
        "cx1", "thick1", "fsag1", "bsag1", "dsag1", "tilt_a1", "tilt_b1",
        "cx2", "thick2", "fsag2", "bsag2", "dsag2", "tilt_a2", "tilt_b2",
    ])
    lo: np.ndarray = field(default_factory=lambda: np.array([
        -20.0, -55.0, 1.50,
        25.0, 6.0, -0.50, -0.50, -0.50, -25.0, -25.0,
        55.0, 4.0, -0.50, -0.50, -0.50, -25.0, -25.0,
    ]))
    hi: np.ndarray = field(default_factory=lambda: np.array([
        20.0, 55.0, 2.00,
        60.0, 35.0, 0.50, 0.50, 0.50, 25.0, 25.0,
        104.0, 25.0, 0.50, 0.50, 0.50, 25.0, 25.0,
    ]))
    half_h: float = 30.0
    second_half_h: float = 60.0
    x_sensor: float = 120.0
    focus_w: float = 8.0
    n_rays: int = 17
    beam_w: float = 8.0
    max_sag_ratio: float | None = 0.55
    relative_sag: bool = True
    det_clearance: float | None = 12.0
    source_clearance: float = 2.0
    inter_element_clearance: float = 2.0

    def vec_to_params(self, x: Sequence[float]) -> dict[str, float]:
        if len(x) != len(self.param_names):
            raise ValueError(f"expected {len(self.param_names)} values, got {len(x)}")
        params = dict(zip(self.param_names, map(float, x)))
        if self.relative_sag:
            for index in (1, 2):
                half_h = self.half_h if index == 1 else self.second_half_h
                params[f"fsag{index}"] *= half_h
                params[f"bsag{index}"] *= half_h
                params[f"dsag{index}"] *= params[f"thick{index}"] / 2.0
        return params

    def _one(self, params: Mapping[str, float], index: int) -> Element:
        local = {
            "cx": params[f"cx{index}"], "thick": params[f"thick{index}"],
            "fsag": params[f"fsag{index}"], "bsag": params[f"bsag{index}"],
            "dsag": params[f"dsag{index}"],
            "tilt_a": params[f"tilt_a{index}"],
            "tilt_b": params[f"tilt_b{index}"], "n": params["n"],
        }
        half_h = self.half_h if index == 1 else self.second_half_h
        return Element.from_legacy_params(local, half_h=half_h)

    def elements(self, params: Mapping[str, float]) -> tuple[Element, Element]:
        return self._one(params, 1), self._one(params, 2)

    def element(self, params: Mapping[str, float]) -> CompoundElement:
        return CompoundElement(self.elements(params))

    def face_of(self, params: Mapping[str, float]):
        classifiers = (
            make_single_element_face_classifier(params["cx1"], self.half_h),
            make_single_element_face_classifier(params["cx2"], self.second_half_h),
        )
        centers = (float(params["cx1"]), float(params["cx2"]))

        def classify(vertex: Mapping) -> str:
            # ParallelBundle emits one scene object per source ray, so the two glass
            # object indices immediately follow those n_rays objects.  Use that exact
            # identity when present: strongly curved neighbouring faces can be closer
            # to the other element's centre, making a coordinate-only classifier lie.
            object_index = vertex.get("i")
            indexed = (int(object_index) - self.n_rays
                       if object_index is not None else -1)
            which = indexed if indexed in (0, 1) else (
                0 if abs(float(vertex["x"]) - centers[0])
                <= abs(float(vertex["x"]) - centers[1]) else 1)
            return f"E{which + 1}{classifiers[which](vertex)}"
        return classify

    def layout(self, params: Mapping[str, float], *, field_deg: float = 0.0
               ) -> MultiElementLayout:
        return MultiElementLayout(
            source=ParallelBundle(center=params["ray_y"],
                                  aim_deg=params["ang"] + field_deg,
                                  width=self.beam_w, n_rays=self.n_rays,
                                  axis="y", origin=0.0),
            elements=self.elements(params),
            detector=Detector.vertical(self.x_sensor),
        )

    def is_valid(self, params: Mapping[str, float]) -> bool:
        first, second = self.elements(params)
        if not first.is_valid() or not second.is_valid():
            return False
        if not self.sag_ok(first) or not self.sag_ok(second):
            return False
        first_extent, second_extent = first.x_extent(), second.x_extent()
        if first_extent[0] <= self.source_clearance:
            return False
        if first_extent[1] + self.inter_element_clearance >= second_extent[0]:
            return False
        return (self.det_clearance is None
                or second_extent[1] < self.x_sensor - self.det_clearance)


__all__ = ["CompoundElement", "MultiElementLayout", "TwoElementSphericalProblem"]
