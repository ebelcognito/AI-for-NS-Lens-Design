"""Train / test_cell / test_env splits.

Never split randomly. Elites in adjacent cells of the same run are near-duplicates -- they
descend from the same emitter mean, often differ in the fourth decimal, and a random split
puts one in train and its twin in test. The measured version of this: two ray paths account
for 60% of the fold-count archive, and the top six cells of that arm are the SAME signature
at six thicknesses. A random split over those cells measures memorisation and reports it as
generalisation.

So the two held-out sets each answer a specific question:

    test_cell   descriptor cells inside a trained environment that no training spec names.
                Can the policy hit a ray path it was never asked for? -- interpolation.
    test_env    an entire environment never trained on. `prism` is the perpendicular-fold
                geometry, where straight transmission is impossible and the element must
                turn the bundle ~90 deg; doc 16 already shows it behaves differently from
                the collinear env. -- transfer.

Held out by TOPOLOGY, not by (topology, thinness) cell: the thinness axis of the same
topology is the same optic at a different thickness, so holding out one thinness bin of a
trained topology leaks almost everything about it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...qd.descriptors import tir_face_label
from ..env_config import get as get_env
from .certify import Certified

#: fraction of each trained env's SPECS held out as test_cell. The cut is still made at
#: topology granularity -- never randomly -- but the target is measured in specs, because
#: topology classes are wildly uneven: measured on the 20k task set, 14 topologies hold
#: 1 to 43 specs each, so holding out "20% of topologies" held out 37% of the data and left
#: 96 specs to train on. Picking topologies until the SPEC budget is met keeps both
#: properties: no near-duplicate leaks across the split, and a split that is the size it says.
TEST_CELL_FRAC = 0.2


@dataclass
class Split:
    train: List[Certified]
    test_cell: List[Certified]
    test_env: List[Certified]
    held_out_topologies: Dict[str, List[int]]

    def summary(self) -> Dict[str, object]:
        def topo(rows):
            return len({r.topology for r in rows})
        return dict(
            train=len(self.train), test_cell=len(self.test_cell),
            test_env=len(self.test_env),
            train_topologies=topo(self.train),
            test_cell_topologies=topo(self.test_cell),
            test_env_topologies=topo(self.test_env),
            held_out={e: [tir_face_label(t) for t in ts]
                      for e, ts in self.held_out_topologies.items()},
        )


def split(certs: Sequence[Certified], *, seed: int = 0,
          test_cell_frac: float = TEST_CELL_FRAC,
          test_envs: Optional[Sequence[str]] = None) -> Split:
    """Partition certified specs. `test_envs` defaults to whichever envs are flagged
    `held_out` in `env_config`, so the transfer split is declared in one place."""
    if test_envs is None:
        test_envs = [e for e in {c.env for c in certs} if get_env(e).held_out]
    test_envs = set(test_envs)

    in_env = [c for c in certs if c.env not in test_envs]
    out_env = [c for c in certs if c.env in test_envs]

    rng = np.random.default_rng(seed)
    held: Dict[str, List[int]] = {}
    for env in sorted({c.env for c in in_env}):
        rows = [c for c in in_env if c.env == env]
        sizes: Dict[int, int] = {}
        for c in rows:
            sizes[c.topology] = sizes.get(c.topology, 0) + 1
        budget = test_cell_frac * len(rows)
        chosen: List[int] = []
        taken = 0
        # topologies in random order, taken while they fit the budget. A topology larger
        # than the whole budget is skipped rather than allowed to swallow the split --
        # otherwise the single dominant route decides the split size by itself.
        for t in rng.permutation(sorted(sizes)).tolist():
            if taken >= budget:
                break
            if taken + sizes[t] <= budget or not chosen:
                chosen.append(int(t))
                taken += sizes[t]
        held[env] = sorted(chosen)

    train, test_cell = [], []
    for c in in_env:
        (test_cell if c.topology in held.get(c.env, ()) else train).append(c)
    return Split(train=train, test_cell=test_cell, test_env=out_env,
                 held_out_topologies=held)


def curriculum_weights(certs: Sequence[Certified],
                       pass_rate: Optional[Dict[str, float]] = None,
                       *, band: Tuple[float, float] = (0.2, 0.8),
                       floor: float = 0.05) -> np.ndarray:
    """Sampling weights over training specs, preferring the 0.2-0.8 pass band.

    This is not a tuning nicety. A spec the policy solves every time and a spec it never
    solves both produce a group whose rewards are identical, hence zero advantage, hence no
    gradient -- doc 18 §4.3 names this the most likely way the run quietly wastes a week.
    Sampling into the band is the primary mitigation; raising the group size is the
    secondary one.

    Before a policy exists there is no pass rate, so the probe-derived `difficulty` stands
    in: a cell only 1 of 3 probe seeds reached is a hard cell. `floor` keeps every spec
    reachable so the distribution cannot collapse onto a handful of tasks.
    """
    if not len(certs):
        return np.zeros(0)
    lo, hi = band
    mid = 0.5 * (lo + hi)
    half = max(hi - mid, 1e-6)
    w = np.empty(len(certs), float)
    for i, c in enumerate(certs):
        p = (pass_rate or {}).get(c.spec_id)
        if p is None:
            # no policy yet: read the probe's own hit rate as a proxy pass rate
            p = 1.0 - c.difficulty
        w[i] = max(floor, 1.0 - abs(p - mid) / half)
    return w / w.sum()
