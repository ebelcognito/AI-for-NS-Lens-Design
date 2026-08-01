"""The CMA-ME driver: the loop that was copy-pasted into six runners.

The RNG protocol is reproduced EXACTLY as the legacy runners had it, because archives
are compared cell-for-cell against committed results:

  rng          = default_rng(seed)
  emitter i    seeded with seed + i at startup
  restarts     seeded from rng.integers(0, 1<<30)
  new emitter  starts from archive.random_elite(rng), else rng.uniform(lo, hi)

The ranking keys are also the legacy ones: 2e12 + fitness for a NEW cell, 1e12 + delta
for an IMPROVED cell, plain fitness otherwise, -1e9 for an invalid evaluation. Those
magic offsets impose new > improved > kept, which is what makes this CMA-ME rather than
plain CMA-ES. Changing any of it changes the search.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..engine import Runner
from ..engine.signatures import format_signature
from ..search import CMAES, GridArchive
from . import descriptors as descriptor_registry
from . import fitness as fitness_registry
from .observation import measure
from .problems import Problem, lock_tilt


@dataclass
class ExperimentConfig:
    """One QD experiment. This replaces a ~170-line runner script."""

    name: str
    problem: Problem
    fitness: str = "focused_throughput"
    descriptors: Sequence[str] = ("tir_bounces", "thinness")
    bc_bins: Sequence[int] = (9, 16)
    bc_bounds: Sequence[Sequence[float]] = ((0.0, 9.0), (10.0, 120.0))
    evals: int = 14000
    seed: int = 0
    emitters: int = 5
    popsize: int = 10
    sigma0: float = 0.12
    lock_tilt: bool = False
    fields: Optional[Sequence[float]] = None      # multifield: aggregate over these
    label: str = ""                                # meta.experiment string

    def __post_init__(self) -> None:
        fitness_registry.get(self.fitness)
        for d in self.descriptors:
            descriptor_registry.get(d)
        if not self.label:
            self.label = self.name


def _evaluate(runner, problem, cfg, x) -> Optional[dict]:
    """Genotype -> {fitness, bc, info}, or None if unusable."""
    p = problem.vec_to_params(x)
    if not problem.is_valid(p):
        return None

    fit_fn = fitness_registry.get(cfg.fitness)
    coherent = cfg.fitness == "coherent_focused_throughput"
    face_of = problem.face_of(p)

    def observe(field_deg: float, *, allow_empty: bool):
        try:
            res, _ = runner.simulate(
                problem.layout(p, field_deg=field_deg).to_scene_dict())
        except Exception:
            return None
        return measure(res, face_of, x_sensor=problem.x_sensor,
                       dominant_only_centroid=coherent, allow_empty=allow_empty)

    if cfg.fields:
        # a field that catches nothing scores 0; only an engine error is fatal, and
        # only the CENTRAL field must carry a dominant signature (it defines the bc)
        obs_list = [observe(f, allow_empty=True) for f in cfg.fields]
        if any(o is None for o in obs_list):
            return None
        central = obs_list[list(cfg.fields).index(0.0)]
        if central.is_empty:
            return None
        vals = [0.0 if o.is_empty else float(fit_fn(o, focus_w=problem.focus_w))
                for o in obs_list]
        fitness = float(np.mean(vals))
        obs = central
        extra = dict(mean_focthru=fitness, worst_focthru=float(min(vals)),
                     per_field=[round(float(v), 4) for v in vals],
                     central_spot=float(central.spot))
    else:
        obs = observe(0.0, allow_empty=False)
        if obs is None:
            return None
        fitness = float(fit_fn(obs, focus_w=problem.focus_w))
        extra = {}

    bc = tuple(float(descriptor_registry.get(d)(obs, problem)) for d in cfg.descriptors)
    info = dict(
        fitness=float(fitness),
        efficiency=float(obs.delivered) / obs.emitted,
        spot=float(obs.spot),
        tir_count=int(obs.tir_count),
        thinness_T=float(obs.thinness(problem.x_sensor)),
        dom_energy_share=float(obs.dominant_energy / obs.emitted),
        dom_signature=format_signature(obs.dominant_signature),
        centroid=float(obs.centroid),
        n_forward=len(obs.records),
        n_dom=obs.n_dominant,
    )
    info.update(extra)
    return dict(fitness=float(fitness), bc=bc, info=info)


def run(cfg: ExperimentConfig, out_dir: Path, *, log=print) -> dict:
    """Run one experiment and write archive.json. Returns the payload."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = cfg.problem
    lo, hi = np.asarray(problem.lo, float), np.asarray(problem.hi, float)

    archive = GridArchive(tuple(cfg.bc_bins), tuple(tuple(b) for b in cfg.bc_bounds))
    rng = np.random.default_rng(cfg.seed)
    scale = hi - lo
    t0 = time.time()

    def new_emitter(s):
        x0 = archive.random_elite(rng)
        if x0 is None:
            x0 = rng.uniform(lo, hi)
        return CMAES(np.asarray(x0, float), cfg.sigma0 * float(np.mean(scale)),
                     popsize=cfg.popsize, seed=s, bounds=(lo, hi))

    n = 0
    n_invalid = 0
    history: List[dict] = []
    with Runner(timeout_s=20) as runner:
        emitters = [new_emitter(cfg.seed + i) for i in range(cfg.emitters)]
        while n < cfg.evals:
            for ei in range(cfg.emitters):
                if n >= cfg.evals:
                    break
                em = emitters[ei]
                X = em.ask()
                keys = np.empty(len(X))
                added = False
                for k, xx in enumerate(X):
                    if cfg.lock_tilt:
                        xx = lock_tilt(xx)
                    ev = _evaluate(runner, problem, cfg, xx)
                    n += 1
                    if ev is None:
                        keys[k] = -1e9
                        n_invalid += 1
                        continue
                    st, old = archive.add(xx, ev["fitness"], ev["bc"], info=ev["info"])
                    if st == "new":
                        keys[k] = 2e12 + ev["fitness"]; added = True
                    elif st == "improved":
                        keys[k] = 1e12 + (ev["fitness"] - (old or 0.0)); added = True
                    else:
                        keys[k] = ev["fitness"]
                em.tell(np.argsort(-keys))
                if (not added) or em.restart_needed:
                    emitters[ei] = new_emitter(int(rng.integers(0, 1 << 30)))
            if n // 1000 > len(history):
                rec = dict(evals=n, filled=len(archive.cells),
                           coverage=archive.coverage, best=archive.best_fitness,
                           invalid=n_invalid)
                history.append(rec)
                log(f"[{cfg.name}] evals={n:6d} filled={rec['filled']:3d} "
                    f"cov={rec['coverage']*100:5.1f}% best={rec['best']:.3f} "
                    f"invalid={n_invalid} ({time.time()-t0:6.1f}s)")

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), tir_count=float(c.bc[0]), T=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info)
        cells.append(rec)

    payload = dict(
        meta=dict(experiment=cfg.label, evals=int(n), seed=int(cfg.seed),
                  invalid=int(n_invalid), bc_bins=list(cfg.bc_bins),
                  bc_bounds=[list(b) for b in cfg.bc_bounds],
                  bc_axes=[descriptor_registry.axis_label(d) for d in cfg.descriptors],
                  fitness=fitness_registry.META_DESCRIPTION.get(cfg.fitness, cfg.fitness),
                  focus_w=problem.focus_w,
                  param_names=list(problem.param_names),
                  wall_seconds=float(time.time() - t0)),
        history=history, cells=cells)
    (out_dir / "archive.json").write_text(json.dumps(payload, indent=2))
    log(f"[{cfg.name}] wrote {out_dir/'archive.json'} ({len(cells)} cells, {n} evals, "
        f"{payload['meta']['wall_seconds']:.0f}s)")
    return payload
