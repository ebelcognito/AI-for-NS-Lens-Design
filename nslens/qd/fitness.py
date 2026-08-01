"""Fitness registry.

This is the axis that separated run_fold_qd from run_orderly_qd: ~6 lines out of ~170.

Every fitness here is FOLD-BLIND -- none of them names folding, TIR or thinness. That is
the whole point of the experimental design: if a fold shows up in a folded niche, it
emerged from the physics, it was not prescribed by the objective. Do not add a fitness
that rewards TIR directly without renaming the experiment.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .observation import Observation

_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        fn.fitness_name = name
        return fn
    return deco


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown fitness {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names() -> List[str]:
    return sorted(_REGISTRY)


@register("focused_throughput")
def focused_throughput(obs: Observation, *, focus_w: float) -> float:
    """Fraction of INPUT energy delivered within a tight window around the energy
    centroid. Rewards routing a lot of energy AND concentrating it. (experiment #5)"""
    return obs.window_energy(focus_w) / obs.emitted


@register("coherent_focused_throughput")
def coherent_focused_throughput(obs: Observation, *, focus_w: float) -> float:
    """Same, but only the rays taking the DOMINANT interaction sequence count. Each
    archive cell then holds the most orderly (uniform-bundle) design of its fold
    topology. (experiment #6)"""
    return obs.window_energy(focus_w, dominant_only=True) / obs.emitted


#: human-readable strings written into archive.json meta, kept identical to the legacy
#: runners so old and new archives compare cleanly
META_DESCRIPTION = {
    "focused_throughput": "focused_throughput (window energy / emitted)",
    "coherent_focused_throughput":
        "coherent_focused_throughput (dominant-signature window energy / emitted)",
    "multifield_mean_focused_throughput":
        "mean focused_throughput over fields",
}
