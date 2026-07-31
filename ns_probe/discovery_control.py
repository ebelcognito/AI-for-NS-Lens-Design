"""
Discovery-control experiment harness (plan 14_discovery_control_experiment_plan.md).

ONE script, three samplers, IDENTICAL fitness/engine/bounds/archive/insertion — only the
*sampler* differs, so the evolutionary contribution is isolated. All of fitness, genotype
bounds, archive grid and insertion rule are imported byte-identical from run_fold_qd; we do
NOT re-implement evaluate() or GridArchive.

Arms (the ablation ladder, §1):
  A  cmame       current run_fold_qd loop (5 emitters, restart-from-elite, archive-improvement
                 ranking)                                             -> full stack
  B0 random      x ~ Uniform(FD.LO, FD.HI) i.i.d.                     -> archive niche-protection only
  B1 restart_ls  uniform seed -> vanilla CMA-ES minimizing -fitness (single-objective, raw-fitness
                 ranking, NO archive-improvement), insert every point; restart from a FRESH UNIFORM
                 seed on convergence/stagnation                       -> covariance adaptation only

Every arm counts invalid geometries as evals (matched budget = matched evaluate() calls, §9),
inserts into the SAME GridArchive(BC_BINS, BC_BOUNDS) with the SAME archive.add() call, and logs
per-1000-evals: coverage, qd_score (Sum fitness over filled cells), best_focthru, plus the eval
index at which each cell was FIRST filled (first_fill), and a pool of all good folds seen
(fitness>=0.5 & tir>=2) for the E2a hit-rate / E3 basin / E4 novelty analyses.

    PYTHONPATH=.:ns_probe python ns_probe/discovery_control.py --mode cmame --evals 14000 \
        --seed 0 --out ns_probe/results_disc/cmame/seed0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_fold_qd import evaluate, BC_BINS, BC_BOUNDS   # identical fitness + BCs + genotype
import fold_demo as FD
from cma_me import CMAES, GridArchive

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

REPO = Path(__file__).resolve().parents[1]

GOOD, HIGH = 0.5, 0.9      # focused-throughput thresholds (match compare_descriptors)
NS_TIR = 2                 # non-sequential := >=2 TIR on the dominant path


# ----------------------------------------------------------------------------------
# Shared bookkeeping — identical across every arm
# ----------------------------------------------------------------------------------
class Book:
    """Accumulates matched, arm-agnostic logs so no arm can differ except in its sampler."""

    def __init__(self, archive: GridArchive, evals: int, dump_valid: bool):
        self.archive = archive
        self.evals = evals
        self.dump_valid = dump_valid
        self.n = 0
        self.n_invalid = 0
        self.n_valid = 0
        self.n_fold = 0            # tir >= 2
        self.n_good_fold = 0       # tir >= 2 & foc >= GOOD
        self.n_high_fold = 0       # tir >= 2 & foc >= HIGH
        self.first_fill: dict[str, int] = {}
        self.history: list[dict] = []
        self.good_pool: list[dict] = []      # every good fold seen (for E3/E4)
        self.valid_geno: list[list[float]] = []   # every valid genotype (E4, big runs only)
        self.t0 = time.time()

    def record(self, x, ev):
        """Insert one evaluated point. Returns (status, old_fitness) for the caller's ranking.
        Counts the eval, updates first_fill/counters/pools. Invalids count as evals."""
        self.n += 1
        if ev is None:
            self.n_invalid += 1
            return ("none", None)
        self.n_valid += 1
        if self.dump_valid:
            self.valid_geno.append([float(v) for v in x])
        st, old = self.archive.add(x, ev["fitness"], ev["bc"], info=ev["info"])
        if st == "new":
            i0, i1 = self.archive.index(ev["bc"])
            self.first_fill[f"{i0},{i1}"] = self.n
        info = ev["info"]
        tir = int(info["tir_count"]); foc = float(ev["fitness"])
        if tir >= NS_TIR:
            self.n_fold += 1
            if foc >= GOOD:
                self.n_good_fold += 1
                self.good_pool.append(dict(
                    genotype=[float(v) for v in x], fitness=foc, tir_count=tir,
                    spot=float(info["spot"]), dom_signature=info["dom_signature"],
                    efficiency=float(info["efficiency"]), thinness_T=float(info["thinness_T"]),
                    first_seen=self.n))
            if foc >= HIGH:
                self.n_high_fold += 1
        return (st, old)

    def qd_score(self) -> float:
        return float(sum(c.fitness for c in self.archive.cells.values()))

    def maybe_log(self, tag: str):
        if self.n // 1000 > len(self.history):
            rec = dict(evals=self.n, filled=len(self.archive.cells),
                       coverage=self.archive.coverage, qd_score=self.qd_score(),
                       best_focthru=self.archive.best_fitness, invalid=self.n_invalid,
                       n_fold=self.n_fold, n_good_fold=self.n_good_fold,
                       n_high_fold=self.n_high_fold)
            self.history.append(rec)
            print(f"[{tag}] evals={self.n:6d} filled={rec['filled']:3d} "
                  f"cov={rec['coverage']*100:5.1f}% qd={rec['qd_score']:7.2f} "
                  f"best={rec['best_focthru']:.3f} good_folds={self.n_good_fold} "
                  f"invalid={self.n_invalid} ({time.time()-self.t0:6.1f}s)", flush=True)


# ----------------------------------------------------------------------------------
# Arm A — CMA-ME (byte-identical to run_fold_qd's loop, routed through Book)
# ----------------------------------------------------------------------------------
def run_cmame(runner, book: Book, rng, args):
    scale = FD.HI - FD.LO
    sig = args.sigma0 * float(np.mean(scale))

    def new_emitter(s):
        # persistence = restart from a random elite; --no-persistence ablates it (E3 item 3):
        # reseed from a fresh uniform sample instead, so only the persistence lever changes.
        x0 = None if args.no_persistence else book.archive.random_elite(rng)
        if x0 is None:
            x0 = rng.uniform(FD.LO, FD.HI)
        return CMAES(np.asarray(x0, float), sig, popsize=args.popsize, seed=s,
                     bounds=(FD.LO, FD.HI))

    emitters = [new_emitter(args.seed + i) for i in range(args.emitters)]
    while book.n < book.evals:
        for ei in range(args.emitters):
            if book.n >= book.evals:
                break
            em = emitters[ei]
            X = em.ask()
            keys = np.empty(len(X)); added = False
            for k, xx in enumerate(X):
                ev = evaluate(runner, xx)
                st, old = book.record(xx, ev)
                if ev is None:
                    keys[k] = -1e9
                    continue
                if st == "new":
                    keys[k] = 2e12 + ev["fitness"]; added = True
                elif st == "improved":
                    keys[k] = 1e12 + (ev["fitness"] - (old or 0.0)); added = True
                else:
                    keys[k] = ev["fitness"]
            em.tell(np.argsort(-keys))
            if (not added) or em.restart_needed:
                emitters[ei] = new_emitter(int(rng.integers(0, 1 << 30)))
        book.maybe_log("cmame")


# ----------------------------------------------------------------------------------
# Arm B0 — pure random search
# ----------------------------------------------------------------------------------
def run_random(runner, book: Book, rng, args):
    while book.n < book.evals:
        x = rng.uniform(FD.LO, FD.HI)
        ev = evaluate(runner, x)
        book.record(x, ev)
        book.maybe_log("random")


# ----------------------------------------------------------------------------------
# Arm B1 — random-restart local search (single-objective vanilla CMA-ES, no archive ranking)
# ----------------------------------------------------------------------------------
def run_restart_ls(runner, book: Book, rng, args):
    """Vanilla CMA-ES minimizing -fitness. Rank by RAW fitness only (invalids last). Insert
    every evaluated point into the shared archive (but the archive does NOT influence search).
    Restart from a FRESH UNIFORM seed on covariance collapse or `patience` gens without a
    new best -- never from an elite, so persistence is genuinely removed (§9)."""
    scale = FD.HI - FD.LO
    sig = args.sigma0 * float(np.mean(scale))
    patience = args.patience

    def fresh():
        x0 = rng.uniform(FD.LO, FD.HI)
        return CMAES(np.asarray(x0, float), sig, popsize=args.popsize,
                     seed=int(rng.integers(0, 1 << 30)), bounds=(FD.LO, FD.HI))

    es = fresh(); best_f = -np.inf; stagnant = 0
    while book.n < book.evals:
        X = es.ask()
        fits = np.empty(len(X))
        for k, xx in enumerate(X):
            ev = evaluate(runner, xx)
            book.record(xx, ev)                 # inserted, but NOT used for ranking
            fits[k] = ev["fitness"] if ev is not None else -1e9
            if book.n >= book.evals:
                fits = fits[:k + 1]
                break
        es.tell(np.argsort(-fits[:len(X)]) if len(fits) == len(X) else np.argsort(-fits))
        gen_best = float(fits.max())
        if gen_best > best_f + 1e-6:
            best_f = gen_best; stagnant = 0
        else:
            stagnant += 1
        if es.restart_needed or stagnant >= patience:
            es = fresh(); best_f = -np.inf; stagnant = 0
        book.maybe_log("restart_ls")


RUNNERS = {"cmame": run_cmame, "random": run_random, "restart_ls": run_restart_ls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(RUNNERS), required=True)
    ap.add_argument("--evals", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=5)     # arm A only
    ap.add_argument("--popsize", type=int, default=10)     # arms A & B1
    ap.add_argument("--sigma0", type=float, default=0.12)  # arms A & B1 (fraction of mean range)
    ap.add_argument("--patience", type=int, default=25)    # arm B1 restart trigger (gens)
    ap.add_argument("--no-persistence", action="store_true",
                    help="arm A ablation (E3): reseed emitters from fresh uniform, not an elite")
    ap.add_argument("--dump-valid", action="store_true",
                    help="also save every valid genotype (for the 50k E4 novelty pool)")
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_disc" / "run"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    archive = GridArchive(BC_BINS, BC_BOUNDS)
    rng = np.random.default_rng(args.seed)
    book = Book(archive, args.evals, args.dump_valid)

    with PersistentRayOpticsRunner(timeout_s=20) as runner:
        RUNNERS[args.mode](runner, book, rng, args)

    # cells payload — same schema as run_fold_qd so compare_descriptors.score() reads it as-is
    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), tir_count=float(c.bc[0]), T=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info)
        cells.append(rec)

    N = float(book.n_valid) if book.n_valid else 1.0
    payload = dict(
        meta=dict(experiment="discovery_control", mode=args.mode, evals=int(book.n),
                  seed=int(args.seed), invalid=int(book.n_invalid),
                  bc_bins=BC_BINS, bc_bounds=BC_BOUNDS,
                  bc_axes=["dominant_TIR_bounce_count", "thinness_T (thin=small)"],
                  fitness="focused_throughput (window energy / emitted)",
                  param_names=FD.PARAM_NAMES, wall_seconds=float(time.time() - book.t0),
                  emitters=args.emitters, popsize=args.popsize, sigma0=args.sigma0,
                  no_persistence=bool(args.no_persistence)),
        hit_rate=dict(   # E2a — the manifold's measure (byproduct of every run; use the random arm)
            n_evals=int(book.n), n_valid=int(book.n_valid), n_invalid=int(book.n_invalid),
            n_fold=int(book.n_fold), n_good_fold=int(book.n_good_fold),
            n_high_fold=int(book.n_high_fold),
            frac_valid=book.n_valid / max(book.n, 1),
            frac_fold_of_valid=book.n_fold / N,
            frac_good_fold_of_valid=book.n_good_fold / N,
            frac_high_fold_of_valid=book.n_high_fold / N),
        history=book.history, first_fill=book.first_fill,
        good_pool=book.good_pool, cells=cells)
    (out / "archive.json").write_text(json.dumps(payload, indent=2))
    if book.dump_valid and book.valid_geno:
        np.save(out / "valid_genotypes.npy", np.asarray(book.valid_geno, dtype=float))
    print(f"[{args.mode}] wrote {out/'archive.json'} ({len(cells)} cells, {book.n} evals, "
          f"{payload['meta']['wall_seconds']:.0f}s, good_folds={book.n_good_fold})", flush=True)


if __name__ == "__main__":
    main()
