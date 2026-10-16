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

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

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
    #: multifield only: the field angles evaluated and the per-field fitness, in the same
    #: order. Empty for a single-field experiment, which is why field_accept refuses to
    #: run there rather than silently reporting 0.
    fields: Sequence[float] = ()
    per_field: Sequence[float] = ()
    #: Observation.convergence() -- measured once by the evaluator and shared
    convergence: Optional[Dict[str, Any]] = None


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


#: Faces, in a FIXED order. This order defines TIR_FACE_VOCAB's indices, and those
#: indices are written into every archive built on the `tir_faces` axis -- changing the
#: order silently renames every cell. Do not reorder. ('A' left, 'B' right, 'T' the
#: BOTTOM edge, 'D' the TOP edge -- see Element.legacy_face_label.)
TIR_FACES = ("A", "B", "D", "T")

#: Longest fold depth the axis resolves. Measured over all 58,980 scored trace rows the
#: observed depths are 2-5, so 5 covers 100% of evaluations; 4 would drop 1.8%.
TIR_FACE_KMAX = 5


def _tir_face_vocab():
    """Every multiset of TIR faces up to TIR_FACE_KMAX, in a fixed canonical order.

    A multiset, not a sequence: "which faces the light bounces off, and how many times
    each" without the order. Sequence would resolve 254 classes against the multiset's
    79, but at 4^k it cannot be enumerated into a fixed grid, and 63% of the multiset
    bins are actually realised against 38% at k<=6 -- this is the granularity that
    survives being an archive axis.
    """
    vocab = []
    for k in range(TIR_FACE_KMAX + 1):
        for combo in itertools.combinations_with_replacement(TIR_FACES, k):
            vocab.append(tuple(sorted(Counter(combo).items())))
    return vocab


TIR_FACE_VOCAB = _tir_face_vocab()
TIR_FACE_INDEX = {m: i for i, m in enumerate(TIR_FACE_VOCAB)}


@register("tir_faces", bins=len(TIR_FACE_VOCAB), bounds=(0.0, float(len(TIR_FACE_VOCAB))),
          label="TIR face multiset (topology)", aliases=("topology", "fold_faces"))
def tir_faces(ctx: DescriptorContext) -> float:
    """WHICH faces the light TIRs off, not merely HOW MANY times it bounces.

    `tir_bounces` is a 9-bin projection of an 804-value quantity, and in the fold24
    regime it resolves to FOUR values. Measured across every trace on disk, 41.7 distinct
    interaction signatures compete for a single (tir_bounces, thinness) cell -- 94% of
    occupied cells hold more than one topology, and the archive deletes all but the best.
    133 of the 171 catalogued topologies never win a cell anywhere.

    This axis separates them by the multiset of TIR faces, which is what makes two
    designs structurally different routes rather than the same route at a different
    depth: A->B->A and A->T->A are both "2 folds" and are not the same optic.

    Out-of-vocabulary (more than TIR_FACE_KMAX bounces) returns an index past the top
    bound, so `reject_out_of_bounds` discards it rather than clamping it into the last
    cell -- clamping is exactly the failure this axis exists to fix.
    """
    faces = [f for (t, f) in (ctx.obs.dominant_signature or ()) if t == "tir"]
    key = tuple(sorted(Counter(faces).items()))
    idx = TIR_FACE_INDEX.get(key)
    return float(len(TIR_FACE_VOCAB)) + 1.0 if idx is None else float(idx) + 0.5


def tir_face_label(idx: int) -> str:
    """Human-readable name of a `tir_faces` cell index, e.g. 'A2 B1' or 'unfolded'."""
    if not (0 <= idx < len(TIR_FACE_VOCAB)):
        return "out-of-vocab"
    m = TIR_FACE_VOCAB[idx]
    return " ".join(f"{f}{n}" for f, n in m) if m else "unfolded"


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


@register("beam_convergence", bins=10, bounds=(0.0, 2.0),
          label="beam_width_ratio_exit_over_entry (packing=small)",
          aliases=("packing", "beam_ratio"))
def beam_convergence(ctx: DescriptorContext) -> float:
    """How much the dominant bundle packs between the entry surface and the exit.

    < 1 the fold pays for itself: the beam leaves narrower than it entered.
    = 1 the folds are pure redirection, buying compactness but not concentration.
    > 1 the element is a diffuser with extra steps.

    A design with no measurable dominant bundle reports the top of the range rather than
    0, so "unmeasurable" cannot masquerade as "perfectly packed".
    """
    c = ctx.convergence or ctx.obs.convergence()
    return float(c["ratio"]) if c["measured"] else 2.0


#: a field counts as ACCEPTED when it keeps this fraction of the on-axis performance.
#: Relative, not absolute: the axis then measures angular ACCEPTANCE (how far the design
#: keeps working) independently of how bright it is on axis, so a dim-but-wide design and
#: a bright-but-wide design share a niche and compete on fitness.
ACCEPT_FRAC = 0.5


@register("field_accept", bins=10, bounds=(0.0, 20.0),
          label="field_acceptance_half_angle_deg", aliases=("accept",))
def field_accept(ctx: DescriptorContext) -> float:
    """Angular half-width over which the design keeps working.

    Walks outward from the on-axis field in both directions and stops at the first field
    that drops below ACCEPT_FRAC of the on-axis value, so it measures the CONTIGUOUS
    accepted cone -- a design that fails at +-4 deg and happens to revive at +-12 deg is
    not a wide-field design, it is two narrow ones.

    Reported as the smaller of the two half-widths, i.e. the symmetric cone the design
    actually supports.
    """
    fields, vals = list(ctx.fields), list(ctx.per_field)
    if not fields or len(fields) != len(vals):
        raise ValueError("field_accept needs a multifield experiment: set "
                         "ExperimentConfig.fields to the field angles to sample")
    i0 = fields.index(0.0)
    thresh = ACCEPT_FRAC * vals[i0]
    if vals[i0] <= 0:
        return 0.0

    def reach(step: int) -> float:
        """Half-width in degrees before the first field that falls below threshold."""
        last = 0.0
        i = i0 + step
        while 0 <= i < len(fields) and vals[i] >= thresh:
            last = abs(fields[i] - fields[i0])
            i += step
        return last

    return float(min(reach(+1), reach(-1)))


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
