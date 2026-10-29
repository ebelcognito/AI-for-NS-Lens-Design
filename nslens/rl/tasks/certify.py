"""Certification: a spec is only a spec once a probe has actually reached it.

The rule this module exists to enforce is one this program learned the expensive way. The
Tier A V1 benchmark shipped a prism spec that turned out to be unsatisfiable -- only 2 of
its 5 fields could ever close -- and everything measured against it was measuring nothing.
A spec that names a score nobody has reached is a spec that might be impossible, and under
GRPO an impossible spec is worse than useless: every rollout in the group fails, the
within-group advantage is identically zero, and the step is spent (doc 18 §4.3).

So: run a short CMA-ME probe per variant, harvest the cells it actually filled, and let the
score it reached BE the spec's `achievable`. Cells no probe reached do not become specs.

Two things about this are worth stating plainly rather than burying.

**One probe certifies many specs.** Doc 18 §3.1 costs certification at one 2,000-eval probe
per spec. That is not how the driver works -- a single probe fills every cell it can reach,
so it certifies a dozen or more specs at once, and the certification bill is set by the
number of VARIANTS (25 per env), not by the number of specs.

**The probes warm-start, and that contaminates part of the answer.** A 15%-acceptance gate
stack cannot cold-start: measured, a fresh run fills 0-3 of 48 cells in 4k evals at best
fitness 0.03, which would certify nothing. Every archive in this program warm-starts for
this reason. The consequence is that for a cell the seed already occupies, `achievable` may
be the seed's own score rather than anything this probe found -- so each certified spec
carries `from_seed`, and a training curriculum that wants search-found difficulty should
filter on it. Cells the seed does not occupy are genuinely probe-found.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...qd import driver
from ..env_config import get as get_env
from ..reward import Spec, evaluate, hardened_pass
from .generate import Variant, spec_from_cell, variants

_REPO = Path(__file__).resolve().parents[3]

#: warm-start archive per env. `None` means the env cold-starts, which is honest but will
#: certify very few cells -- see the module docstring.
SEEDS: Dict[str, Optional[str]] = {
    "fold": str(_REPO / "runs" / "fold24_gated_seed"),
    "prism": None,
}


@dataclass
class Certified:
    """One certified spec: the cell, the score a probe reached, and how it got there."""

    env: str
    fov: float
    focus_w: float
    topology: int
    thinness_bin: int
    achievable: float
    #: fraction of independent probe seeds that failed to fill this cell, in [0, 1). The
    #: only difficulty signal available before a policy exists; the pass-rate curriculum of
    #: doc 18 §3.3 replaces it once one does.
    difficulty: float
    seeds_hit: int
    seeds_run: int
    from_seed: bool
    genotype: List[float]
    signature: str
    crit_margin: float
    spec_id: str

    def to_spec(self) -> Spec:
        return spec_from_cell(Variant(self.env, self.fov, self.focus_w),
                              self.topology, self.thinness_bin,
                              achievable=self.achievable, difficulty=self.difficulty)


def _probe(args) -> Tuple[Tuple[str, float, float], int, List[dict], List[list]]:
    """One CMA-ME probe. Returns the archive cells and the seed genotypes it started from,
    so the caller can tell a search-found cell from a carried-over one."""
    variant, seed, evals, out_root = args
    env = get_env(variant[0], fov=variant[1], focus_w=variant[2])
    resume = SEEDS.get(variant[0])
    cfg = replace(env.cfg, evals=int(evals), seed=int(seed), resume_from=resume,
                  name=f"certify_{variant[0]}_fov{variant[1]:g}_w{variant[2]:g}_s{seed}")
    out_dir = Path(out_root) / cfg.name
    seed_gts: List[list] = []
    if resume:
        prev = json.loads((Path(resume) / "archive.json").read_text())
        seed_gts = [[round(float(v), 6) for v in c["genotype"]] for c in prev["cells"]]
    payload = driver.run(cfg, out_dir, log=lambda *a, **k: None)
    return variant, seed, payload["cells"], seed_gts


def certify(variant_list: Optional[Sequence[Variant]] = None, *,
            seeds: int = 3, evals: int = 2000, workers: Optional[int] = None,
            out_root: Optional[str] = None, min_achievable: float = 0.05,
            log=print) -> List[Certified]:
    """Probe every variant `seeds` times and turn what the probes filled into specs.

    Embarrassingly parallel across (variant, seed), which is how the certification bill
    stays affordable: the driver is single-process, so the parallelism has to be here.
    """
    variant_list = list(variant_list if variant_list is not None else variants())
    workers = int(workers or min(20, os.cpu_count() or 4))
    tmp = out_root or tempfile.mkdtemp(prefix="certify_")
    jobs = [(v.key, s, evals, tmp) for v in variant_list for s in range(seeds)]
    log(f"[certify] {len(variant_list)} variants x {seeds} seeds x {evals} evals "
        f"= {len(jobs)} probes on {workers} workers")

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        results = pool.map(_probe, jobs, chunksize=1)
    log(f"[certify] probes finished in {time.time() - t0:.0f}s")

    # (variant, topology, thinness_bin) -> best cell across seeds, plus how many hit it
    best: Dict[tuple, dict] = {}
    hits: Dict[tuple, int] = {}
    seed_sets: Dict[str, set] = {}
    for variant, seed, cells, seed_gts in results:
        seed_sets.setdefault(variant[0], set()).update(tuple(g) for g in seed_gts)
        for c in cells:
            key = (variant, int(c["i0"]), int(c["i1"]))
            hits[key] = hits.get(key, 0) + 1
            if key not in best or c["fitness"] > best[key]["fitness"]:
                best[key] = c

    out: List[Certified] = []
    for (variant, topo_bin, thin_bin), c in sorted(best.items()):
        if c["fitness"] < min_achievable:
            continue
        # i0 indexes the tir_faces GRID, whose low bound is 5.0 with one vocabulary entry
        # per bin -- the vocabulary index is the grid index plus that offset
        env = get_env(variant[0], fov=variant[1], focus_w=variant[2])
        topology = int(env.cfg.bc_bounds[0][0]) + int(topo_bin)
        gt = tuple(round(float(v), 6) for v in c["genotype"])
        n_hit = hits[(variant, topo_bin, thin_bin)]
        out.append(Certified(
            env=variant[0], fov=variant[1], focus_w=variant[2],
            topology=topology, thinness_bin=int(thin_bin),
            achievable=float(c["fitness"]),
            difficulty=1.0 - n_hit / float(seeds),
            seeds_hit=n_hit, seeds_run=seeds,
            from_seed=gt in seed_sets.get(variant[0], set()),
            genotype=[float(v) for v in c["genotype"]],
            signature=str(c.get("dom_signature", "")),
            crit_margin=float(c.get("crit_margin", 0.0)),
            spec_id=spec_from_cell(Variant(*variant), topology, thin_bin,
                                   achievable=c["fitness"]).spec_id))
    log(f"[certify] {len(out)} cells certified from {len(best)} filled "
        f"({sum(1 for c in out if c.from_seed)} carried over from a seed)")
    return out


def harden(certs: Sequence[Certified], *, min_crit_margin: Optional[float] = None,
           workers: Optional[int] = None, log=print) -> List[Certified]:
    """Drop specs whose certifying design is a knife edge.

    This is where the crit-margin floor is ENFORCED. `reward.reward_of` deliberately grades
    it rather than gating on it (a cliff at a threshold 26 of 85 known-good elites sit below
    produces exactly the zero-variance groups the plan is trying to avoid), so the place to
    apply it is here: never ASK for a spec whose only known answer is a fold living on
    0.1 deg of margin. The policy is then free to be graded on robustness without the task
    set quietly requiring an unbuildable one.
    """
    from ..server import RewardPool

    if not certs:
        return []
    with RewardPool(workers=workers) as pool:
        vs = pool.verdicts([((c.env, c.fov, c.focus_w), c.genotype) for c in certs])
    kept: List[Certified] = []
    for c, v in zip(certs, vs):
        env = get_env(c.env, fov=c.fov, focus_w=c.focus_w)
        floor = env.min_crit_margin if min_crit_margin is None else min_crit_margin
        margin = float(v.gates.get("crit_margin", -5.0)) if v.gates else -5.0
        # ``hardened_pass`` covers geometry, class bounds, clean packing, exit imaging,
        # and the environment's normal margin floor. Keep the explicit comparison so a
        # caller may request a stricter certification margin than the environment default.
        if hardened_pass(v) and margin >= floor:
            kept.append(replace(c, crit_margin=margin))
    log(f"[certify] hardened: {len(kept)}/{len(certs)} specs keep a buildable "
        f"certifying design")
    return kept


def save(certs: Sequence[Certified], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        dict(built=time.strftime("%Y-%m-%d %H:%M:%S"), n=len(certs),
             specs=[asdict(c) for c in certs]), indent=2))


def load(path: str) -> List[Certified]:
    data = json.loads(Path(path).read_text())
    return [Certified(**c) for c in data["specs"]]


def main(argv=None) -> int:  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="certify RL task specs")
    ap.add_argument("--out", default="runs/rl_tasks/certified.json")
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--evals", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--probe-dir", default=None,
                    help="keep the probe archives here instead of a temp dir")
    ap.add_argument("--no-harden", action="store_true")
    a = ap.parse_args(argv)

    certs = certify(variants(a.envs), seeds=a.seeds, evals=a.evals,
                    workers=a.workers, out_root=a.probe_dir)
    if not a.no_harden:
        certs = harden(certs, workers=a.workers)
    save(certs, a.out)
    print(f"wrote {a.out} ({len(certs)} certified specs)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
