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


@register("intended_focused_throughput")
def intended_focused_throughput(obs: Observation, *, focus_w: float,
                                target=None) -> float:
    """Focused input-energy fraction carried by an explicitly requested path.

    Unlike ``coherent_focused_throughput``, the requested path does not silently change
    when another signature becomes dominant.  The shared driver supplies ``target`` for
    fixed-condition experiments; calling this fitness without one is a configuration
    error rather than a fallback to the historically inferred topology.
    """
    if target is None:
        raise ValueError("intended_focused_throughput requires an interaction target")
    from .interaction_objective import target_focused_throughput
    return target_focused_throughput(obs, target, focus_w)


@register("arvr_relay_3field")
def arvr_relay_3field(obs: Observation, *, focus_w: float) -> float:
    """Placeholder: the AR/VR relay fitness needs the raw engine result (three sources
    attributed by emission x, angular spread at the eyebox, field ordering), not an
    Observation. ArvrProblem.evaluate computes it; this exists so the config can name it."""
    raise NotImplementedError(
        "arvr_relay_3field is computed by ArvrProblem.evaluate, not the shared evaluator")


#: human-readable strings written into archive.json meta, kept identical to the legacy
#: runners so old and new archives compare cleanly
META_DESCRIPTION = {
    "arvr_relay_3field": "well_collimated_throughput_3field",
    "focused_throughput": "focused_throughput (window energy / emitted)",
    "coherent_focused_throughput":
        "coherent_focused_throughput (dominant-signature window energy / emitted)",
    "intended_focused_throughput":
        "intended_focused_throughput (target-signature window energy / emitted)",
    "multifield_mean_focused_throughput":
        "mean focused_throughput over fields",
}
