"""Descriptor (behaviour-characteristic) registry.

The archive's diversity axes. These are DESCRIPTIVE -- measured from the engine's typed
interaction events after the fact, never optimised for. Doc 12's ablation showed the
choice matters: axes that are not fold-defining let shallow/transmission designs evict
folds from their niches, so the archive collapses.

A descriptor is `fn(obs, problem) -> float`.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .observation import Observation

_REGISTRY: Dict[str, Callable] = {}


def register(name: str, *, axis_label: str):
    def deco(fn):
        _REGISTRY[name] = fn
        fn.descriptor_name = name
        fn.axis_label = axis_label
        return fn
    return deco


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown descriptor {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def axis_label(name: str) -> str:
    return get(name).axis_label


def names() -> List[str]:
    return sorted(_REGISTRY)


@register("tir_bounces", axis_label="dominant_TIR_bounce_count")
def tir_bounces(obs: Observation, problem) -> float:
    """Fold depth: TIR bounces on the dominant-energy sequence. 0 = straight
    transmission, 2 = double-TIR fold, 4 = quad."""
    return float(obs.tir_count)


@register("thinness", axis_label="thinness_T (thin=small)")
def thinness(obs: Observation, problem) -> float:
    """T = sensor plane minus the frontmost optic surface used. Thin = small."""
    return float(obs.thinness(problem.x_sensor))
