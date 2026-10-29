"""The spec space, and how a probe's archive turns into specs.

A spec is a *variant* -- which environment, at what field of view, against what spot
tolerance -- plus a *cell* of that variant's descriptor grid: which TIR-face multiset the
light must route through, at roughly what thickness.

The descriptor grid alone is not a task distribution. `fold` has 121 topology bins x 4
thinness bins = 484 cells, of which measurement says ~79 topologies are ever realised, so
the grid tops out around 300 reachable specs per environment -- the same order as the 695
already on disk, and the reason doc 18 §0.3 rejects the archive as a task source. The FOV
and window axes are what widen it: 5 x 5 variants x ~300 reachable cells is O(10^4), which
is what §3.1 asks for, and every one of them is a genuinely different optical requirement
rather than a relabelling of the same one.

Nothing here decides what is *achievable*. Enumeration is free and wrong on its own; the
probe in `certify` is what makes a candidate into a spec.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ...qd.descriptors import TIR_FACE_VOCAB, tir_face_label
from ..env_config import FOCUS_W_CHOICES, FOV_CHOICES, RLEnv, get, names
from ..reward import Spec


@dataclass(frozen=True)
class Variant:
    """One environment configuration a batch of specs shares. Probes are run per variant,
    because the field set and the scoring window change what every score means."""

    env: str
    fov: float
    focus_w: float

    @property
    def key(self) -> Tuple[str, float, float]:
        return (self.env, self.fov, self.focus_w)

    @property
    def slug(self) -> str:
        return f"{self.env}_fov{self.fov:g}_w{self.focus_w:g}"

    def resolve(self) -> RLEnv:
        return get(self.env, fov=self.fov, focus_w=self.focus_w)


def variants(envs: Optional[Sequence[str]] = None,
             fovs: Sequence[float] = FOV_CHOICES,
             focus_ws: Sequence[float] = FOCUS_W_CHOICES) -> List[Variant]:
    """The full variant grid. Small and enumerable on purpose -- probes are run per
    variant, so this is the axis whose size sets the certification bill."""
    envs = list(envs if envs is not None else names())
    return [Variant(e, float(f), float(w))
            for e, f, w in itertools.product(envs, fovs, focus_ws)]


def cell_space(env: RLEnv) -> List[Tuple[int, int]]:
    """Every (topology index, thinness bin) the env's grid can represent.

    Vocabulary entries below the low bc bound are excluded, not merely unlikely: indices
    0-4 are the k=0 and k=1 multisets, and admitting them lets straight transmission --
    which scores ~0.92 because it pays no fold losses -- be a legitimate answer to a
    folding task. That is the doc 19 §5.1 failure, and it belongs in the enumeration
    rather than in a downstream filter.
    """
    (t_lo, t_hi), _ = env.cfg.bc_bounds
    n_thin = env.cfg.bc_bins[1]
    return [(t, b) for t in range(len(TIR_FACE_VOCAB))
            if t_lo <= t + 0.5 <= t_hi
            for b in range(n_thin)]


def spec_id(variant: Variant, topology: int, thinness_bin: int) -> str:
    return f"{variant.slug}__t{topology:03d}_h{thinness_bin}"


def spec_from_cell(variant: Variant, topology: int, thinness_bin: int, *,
                   achievable: float, difficulty: Optional[float] = None) -> Spec:
    return Spec(env=variant.env, topology=int(topology),
                thinness_bin=int(thinness_bin), achievable=float(achievable),
                spec_id=spec_id(variant, topology, thinness_bin),
                difficulty=difficulty, fov=variant.fov, focus_w=variant.focus_w)


def describe(spec: Spec) -> str:
    """The human-readable spec, and the basis of the prompt text."""
    return (f"env={spec.env} fov=+-{spec.fov:g} deg window=+-{spec.focus_w:g} mm | "
            f"route the light through TIR faces {spec.topology_label} "
            f"| thinness bin {spec.thinness_bin} "
            f"| certified achievable {spec.achievable:.3f}")


def summarise(specs: Sequence[Spec]) -> Dict[str, object]:
    """The numbers to check before trusting a generated task set."""
    if not specs:
        return dict(n=0)
    ach = np.array([s.achievable for s in specs], float)
    return dict(
        n=len(specs),
        envs=sorted({s.env for s in specs}),
        variants=len({(s.env, s.fov, s.focus_w) for s in specs}),
        topologies=len({s.topology for s in specs}),
        achievable_pct={q: round(float(np.percentile(ach, q)), 3)
                        for q in (5, 25, 50, 75, 95)},
        top_topologies=[tir_face_label(t) for t, _ in
                        sorted(((t, sum(1 for s in specs if s.topology == t))
                                for t in {s.topology for s in specs}),
                               key=lambda kv: -kv[1])[:8]],
    )
