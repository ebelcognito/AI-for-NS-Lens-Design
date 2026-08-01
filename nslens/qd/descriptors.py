"""Descriptor (behaviour-characteristic) registry.

The archive's diversity axes. These are DESCRIPTIVE -- measured from the engine's typed
interaction events after the fact, never optimised for. Doc 12's ablation showed the
choice matters: axes that are not fold-defining let shallow/transmission designs evict
folds from their niches, and the archive collapses.

Each descriptor carries its own grid spec (bins, bounds, label), so an experiment
selects an axis by NAME and inherits the binning that axis was calibrated with. That is
the thing run_fold_qd_descriptors.py had to keep in a parallel BC1 dict.

A descriptor is `fn(ctx: DescriptorContext) -> float`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .observation import Observation


@dataclass
class DescriptorContext:
    """Everything any descriptor needs. Most use only `obs`; a few need more --
    tilt_mag reads the genotype, crit_margin needs the refractive index and the ray
    geometry, robustness has to re-run the engine."""

    obs: Observation
    problem: Any
    params: Dict[str, float]
    genotype: np.ndarray
    fitness: float = 0.0
    runner: Any = None
    pert_rng: Any = None
    with_robust: bool = False


_REGISTRY: Dict[str, Callable] = {}

#: sentinel margin for designs with no TIR -- lands in the bottom critmargin bin
CRIT_NOTIR = -5.0

#: perturbation for `robustness`: manufacturing-like jitter on the physical surface DOFs
#: (cx, thick, fsag, bsag, tilt_a, tilt_b). NOT the beam (ray_y, ang) and NOT the index.
PERT_IDX = np.array([2, 3, 4, 5, 6, 7])
PERT_K = 3


def register(name: str, *, bins: int, bounds: tuple, label: str, aliases=()):
    def deco(fn):
        fn.descriptor_name = name
        fn.bins = bins
        fn.bounds = bounds
        fn.axis_label = label
        _REGISTRY[name] = fn
        for a in aliases:
            _REGISTRY[a] = fn
        return fn
    return deco


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown descriptor {name!r}; known: {names()}")
    return _REGISTRY[name]


def axis_label(name: str) -> str:
    return get(name).axis_label


def spec(name: str) -> tuple:
    """(bins, bounds, label) for this axis."""
    fn = get(name)
    return fn.bins, fn.bounds, fn.axis_label


def names() -> List[str]:
    return sorted({fn.descriptor_name for fn in _REGISTRY.values()})


# --------------------------------------------------------------------------- axes
@register("tir_bounces", bins=9, bounds=(0.0, 9.0),
          label="dominant_TIR_bounce_count", aliases=("baseline", "tir_count"))
def tir_bounces(ctx: DescriptorContext) -> float:
    """Fold depth: TIR bounces on the dominant-energy sequence. 0 = straight
    transmission, 2 = double-TIR fold, 4 = quad."""
    return float(ctx.obs.tir_count)


@register("thinness", bins=16, bounds=(10.0, 120.0),
          label="thinness_T (thin=small)", aliases=("thinness_T",))
def thinness(ctx: DescriptorContext) -> float:
    """T = sensor plane minus the frontmost optic surface used. Thin = small."""
    return float(ctx.obs.thinness(ctx.problem.x_sensor))


@register("ns_frac", bins=12, bounds=(0.0, 1.0),
          label="TIR_energy_fraction (continuous NS-ness)", aliases=("nsfrac",))
def ns_frac(ctx: DescriptorContext) -> float:
    """Continuous non-sequential-ness: forward sensor energy that underwent >=1 TIR,
    over all forward energy. Unlike tir_bounces this is not quantised, so a design that
    folds only part of its bundle is distinguishable."""
    obs = ctx.obs
    # Summation order is load-bearing to the last ULP, because this value is binned:
    # the denominator is numpy's PAIRWISE sum over the brightness array, the numerator
    # a SEQUENTIAL Python accumulation. Matching run_fold_qd_descriptors exactly.
    fwd_total = float(np.array([r["b"] for r in obs.records], dtype=float).sum())
    if fwd_total <= 0:
        return 0.0
    tir_energy = 0.0
    for r in obs.records:
        sig_r = r.get("_sig")
        if sig_r and any(t == "tir" for (t, _f) in sig_r):
            tir_energy += r["b"]
    return tir_energy / fwd_total


@register("tilt_mag", bins=12, bounds=(0.0, 36.0),
          label="prismatic_tilt_magnitude_deg", aliases=("tilt",))
def tilt_mag(ctx: DescriptorContext) -> float:
    """Genotype-level, not measured: the prismatic symmetry break itself. Doc 12 uses it
    to test whether a CAUSE makes a good axis (it does not -- it is not fold-defining)."""
    p = ctx.params
    return float(math.hypot(p.get("tilt_a", 0.0), p.get("tilt_b", 0.0)))


@register("fold_locus", bins=11, bounds=(0.0, 1.0),
          label="edge_TIR_fraction (locus: 0=curved-face, 1=edge-guide)",
          aliases=("locus",))
def fold_locus(ctx: DescriptorContext) -> float:
    """WHERE the fold happens: fraction of the dominant path's TIR events landing on the
    flat edges T/D rather than the curved faces A/B. 1.0 is edge light-guiding."""
    dom = ctx.obs.dominant_signature or ()
    tir_count = sum(1 for (t, _f) in dom if t == "tir")
    if tir_count == 0:
        return 0.0
    edge_tir = sum(1 for (t, f) in dom if t == "tir" and f in ("T", "D"))
    return float(edge_tir) / tir_count


def incidence_deg(seq, k) -> Optional[float]:
    """Angle of incidence (from the normal) at vertex k, inferred from the ray's turn."""
    a = np.array([seq[k]["x"] - seq[k - 1]["x"], seq[k]["y"] - seq[k - 1]["y"]])
    b = np.array([seq[k + 1]["x"] - seq[k]["x"], seq[k + 1]["y"] - seq[k]["y"]])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    c = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
    return 90.0 - math.degrees(math.acos(c)) / 2.0


@register("crit_margin", bins=12, bounds=(-6.0, 42.0),
          label="min_incidence_margin_past_critical_deg", aliases=("critmargin",))
def crit_margin(ctx: DescriptorContext) -> float:
    """How comfortably the dominant ray's TIR events clear the critical angle. Small
    margin = a fold living on a knife edge, which is what makes a design fragile."""
    obs, p = ctx.obs, ctx.params
    if not obs.records:
        return CRIT_NOTIR
    theta_c = math.degrees(math.asin(min(1.0 / p["n"], 1.0)))
    dom_ray = max(obs.records, key=lambda r: r["b"])
    seq = dom_ray["seq"]
    margins = []
    for k in range(1, len(seq) - 1):
        if seq[k].get("t") == "tir":
            ti = incidence_deg(seq, k)
            if ti is not None:
                margins.append(ti - theta_c)
    return float(min(margins)) if margins else CRIT_NOTIR


@register("robustness", bins=12, bounds=(0.0, 1.0),
          label="tolerance_retained_focused_throughput", aliases=("robust",))
def robustness(ctx: DescriptorContext) -> float:
    """Fraction of fitness retained under manufacturing-like jitter (PERT_K perturbed
    re-evaluations). EXPENSIVE -- it multiplies engine calls -- so it is computed only
    when it is actually one of the two axes; otherwise it reports 0.0, exactly as the
    legacy runner did."""
    if not ctx.with_robust:
        return 0.0
    problem, rng = ctx.problem, ctx.pert_rng
    lo, hi = np.asarray(problem.lo, float), np.asarray(problem.hi, float)
    sig_p = 0.02 * (hi - lo)
    keep = []
    for _ in range(PERT_K):
        xp = np.array(ctx.genotype, float).copy()
        xp[PERT_IDX] += rng.normal(0.0, 1.0, size=len(PERT_IDX)) * sig_p[PERT_IDX]
        xp = np.clip(xp, lo, hi)
        f = _focused_throughput_only(ctx, xp)
        keep.append(0.0 if f is None else f)
    return float(np.clip(np.mean(keep) / max(ctx.fitness, 1e-3), 0.0, 1.0))


# ------------------------------------------------------- AR/VR relay axes
# These two are computed inside ArvrProblem.evaluate (they need the raw engine result,
# not an Observation). They are registered so an experiment can name them and inherit
# their axis labels; calling them directly is a bug.
def _owned_by_problem(name: str):
    def fn(ctx: DescriptorContext) -> float:
        raise NotImplementedError(
            f"{name!r} is computed by ArvrProblem.evaluate, not the shared evaluator")
    return fn


register("fov", bins=20, bounds=(0.0, 40.0), label="FOV_deg")(_owned_by_problem("fov"))
register("prism_depth", bins=28, bounds=(4.0, 88.0),
         label="prism_depth (compact=small)")(_owned_by_problem("prism_depth"))


def _focused_throughput_only(ctx: DescriptorContext, x) -> Optional[float]:
    """Just the focused-throughput fitness of a perturbed genotype."""
    from .observation import measure

    problem = ctx.problem
    p = problem.vec_to_params(x)
    if not problem.is_valid(p):
        return None
    try:
        res, _ = ctx.runner.simulate(problem.layout(p).to_scene_dict())
    except Exception:
        return None
    obs = measure(res, problem.face_of(p), x_sensor=problem.x_sensor)
    if obs is None:
        return None
    return float(obs.window_energy(problem.focus_w) / obs.emitted)
