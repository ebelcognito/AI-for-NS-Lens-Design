"""Detector, crop box, and the Layout that assembles a complete engine scene.

Layout is the whole "scenario": source placement + element + detector placement. Those
were three of the five axes that varied across the old run_*/search_* scripts; the other
two (fitness, descriptors) live in nslens.fitness and nslens.descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from ..geometry import Element
from .sources import ParallelBundle, PointFan

Source = Union[ParallelBundle, PointFan]


@dataclass(frozen=True)
class Detector:
    """A detector segment. Every scenario in the project uses a plane perpendicular to
    an axis, but the engine takes a general segment, so this stays general."""

    p1: tuple
    p2: tuple

    @classmethod
    def vertical(cls, x: float, half_height: float = 120.0) -> "Detector":
        return cls((x, -half_height), (x, half_height))

    @classmethod
    def horizontal(cls, y: float, half_width: float = 120.0) -> "Detector":
        return cls((-half_width, y), (half_width, y))

    def to_obj(self) -> dict:
        return {"type": "Detector",
                "p1": {"x": self.p1[0], "y": self.p1[1]},
                "p2": {"x": self.p2[0], "y": self.p2[1]}}


@dataclass(frozen=True)
class CropBox:
    """Render framing only -- never affects the traced physics."""

    p1: tuple
    p4: tuple
    width: int = 1360

    def to_obj(self) -> dict:
        return {"type": "CropBox",
                "p1": {"x": self.p1[0], "y": self.p1[1]},
                "p4": {"x": self.p4[0], "y": self.p4[1]},
                "width": self.width}


@dataclass(frozen=True)
class Layout:
    """source + element + detector -> a ray-optics scene dict."""

    source: Source
    element: Element
    detector: Detector
    max_depth: int = 16
    record: bool = True
    crop: Optional[CropBox] = None

    def to_scene_dict(self) -> dict:
        objs = list(self.source.rays())
        objs.append(self.element.to_scene_dict())
        objs.append(self.detector.to_obj())
        scene = {"version": 5, "maxRayDepth": self.max_depth, "objs": objs}
        if self.record:
            scene["recordInteractions"] = True
        if self.crop is not None:
            # matches legacy ordering: the crop box is appended AFTER the scene dict
            # already references objs, so it lands at the end of the same list
            objs.append(self.crop.to_obj())
        return scene
