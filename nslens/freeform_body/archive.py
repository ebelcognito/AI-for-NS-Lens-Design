"""A joint shape/interaction archive for genuinely diverse freeform optics.

Unlike the legacy two-axis grid, this archive's primary partition is the dominant
continuous-boundary route.  Parent sampling is uniform over route families, so a large
high-performing family cannot monopolize every emitter restart.  Within a route, a
coarse invariant shape key prevents cosmetic near-duplicates from filling the store.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, log

import numpy as np

from .descriptors import Descriptor
from .problem import DiscoveryEvaluation


@dataclass(frozen=True)
class ShapeNicheResolution:
    log_aspect: float = 0.20
    compactness: float = 0.08
    concave_fraction: float = 0.05
    curvature_changes: float = 2.0
    spectral_strength: float = 0.02

    def __post_init__(self) -> None:
        if min(self.log_aspect, self.compactness, self.concave_fraction,
               self.curvature_changes, self.spectral_strength) <= 0:
            raise ValueError("all shape-niche resolutions must be positive")


def shape_niche_key(descriptor: Descriptor,
                    resolution: ShapeNicheResolution = ShapeNicheResolution()) -> tuple:
    """Coarse, interpretable morphology cell from an invariant shape descriptor."""
    d = descriptor.as_dict()
    spectral = [(int(name.rsplit("_", 1)[1]), value) for name, value in d.items()
                if name.startswith("radial_fourier_")]
    dominant_mode, dominant_strength = max(spectral, key=lambda pair: pair[1], default=(0, 0.0))
    return (
        int(floor(log(max(d["aspect_ratio"], 1e-12)) / resolution.log_aspect)),
        int(floor(d["isoperimetric_compactness"] / resolution.compactness)),
        int(floor(d["concave_fraction"] / resolution.concave_fraction)),
        int(floor(d["curvature_sign_changes"] / resolution.curvature_changes)),
        dominant_mode,
        int(floor(dominant_strength / resolution.spectral_strength)),
    )


def interaction_niche_key(evaluation: DiscoveryEvaluation) -> tuple:
    """Dominant relative-footprint route plus coarse bundle-level behavior."""
    if evaluation.interaction is None:
        return ("invalid",)
    d = evaluation.interaction.as_dict()
    dominant_route = evaluation.route_keys[0] if evaluation.route_keys else ("no_surface_route",)
    return (
        dominant_route,
        int(round(d["footprint_count"])),
        int(d["revisit_fraction"] >= 0.25),
        int(floor(d["dominant_route_share"] * 5.0)),
    )


@dataclass(frozen=True)
class ArchiveRecord:
    genes: np.ndarray
    fitness: float
    route_key: tuple
    shape_key: tuple
    evaluation: DiscoveryEvaluation


class DiverseDiscoveryArchive:
    """One elite per joint route/shape niche, sampled uniformly by route."""

    def __init__(self, resolution: ShapeNicheResolution = ShapeNicheResolution()):
        self.resolution = resolution
        self._records: dict[tuple, ArchiveRecord] = {}
        self._route_index: dict[tuple, set[tuple]] = {}

    @property
    def records(self) -> tuple[ArchiveRecord, ...]:
        return tuple(self._records.values())

    @property
    def joint_niche_count(self) -> int:
        return len(self._records)

    @property
    def route_family_count(self) -> int:
        return len(self._route_index)

    @property
    def shape_family_count(self) -> int:
        return len({record.shape_key for record in self._records.values()})

    @property
    def best_fitness(self) -> float:
        return max((record.fitness for record in self._records.values()), default=float("-inf"))

    def add(self, genes, evaluation: DiscoveryEvaluation) -> tuple[str, float | None]:
        """Return ``new``, ``improved``, or ``none`` and the previous fitness."""
        if not evaluation.feasible or evaluation.shape is None or evaluation.interaction is None:
            return "none", None
        route = interaction_niche_key(evaluation)
        shape = shape_niche_key(evaluation.shape, self.resolution)
        joint = (route, shape)
        current = self._records.get(joint)
        if current is not None and evaluation.fitness <= current.fitness:
            return "none", current.fitness
        status, old = ("new", None) if current is None else ("improved", current.fitness)
        record = ArchiveRecord(
            genes=np.asarray(genes, dtype=float).copy(),
            fitness=float(evaluation.fitness),
            route_key=route,
            shape_key=shape,
            evaluation=evaluation,
        )
        self._records[joint] = record
        self._route_index.setdefault(route, set()).add(joint)
        return status, old

    def random_elite(self, rng: np.random.Generator) -> np.ndarray | None:
        """Select a route family uniformly, then a shape niche within that route."""
        if not self._route_index:
            return None
        # Hash iteration order varies between interpreter processes because route
        # keys contain strings.  Sorting preserves reproducibility for a fixed seed.
        routes = tuple(sorted(self._route_index, key=repr))
        route = routes[int(rng.integers(0, len(routes)))]
        joints = tuple(sorted(self._route_index[route], key=repr))
        joint = joints[int(rng.integers(0, len(joints)))]
        return self._records[joint].genes.copy()

    def summary(self) -> dict[str, float | int]:
        return {
            "joint_niches": self.joint_niche_count,
            "route_families": self.route_family_count,
            "shape_families": self.shape_family_count,
            "best_fitness": self.best_fitness,
        }
