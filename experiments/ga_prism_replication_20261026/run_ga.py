"""Two-stage GA of Su & Romanova (2026), Sec. 3, plus the Sec. 4 batch study and Table 2 baseline.

  full      Stage 1 (GA, pop 150, 60 gen) -> Stage 2 (NSGA-II, pop 200, 300 gen); 50 runs  (Figs. 7-10)
  two50     Stage 1 -> Stage 2 for 50 generations; 50 runs                            (Table 2, row 2)
  only50    Stage 2 alone from a uniform random population, 50 generations; 50 runs   (Table 2, row 1)

    python run_ga.py full two50 only50 [--runs 50] [--workers 18]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402
from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.algorithms.soo.nonconvex.ga import GA  # noqa: E402
from pymoo.core.problem import Problem  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prism as M  # noqa: E402

OUT = Path(__file__).resolve().parent/'runs'
NG = len(M.GEOM)
MC_SAMPLES, MC_REL = 200, 5e-3


class Stage1(Problem):
    def __init__(self):
        super().__init__(n_var=NG, n_obj=1, xl=M.LO[:NG], xu=M.HI[:NG])

    def _evaluate(self, X, out, *a, **k):
        out['F'] = M.stage1_fitness(X)[0][:, None]


class Stage2(Problem):
    def __init__(self):
        super().__init__(n_var=len(M.NAMES), n_obj=3, n_ieq_constr=1, xl=M.LO, xu=M.HI)

    def _evaluate(self, X, out, *a, **k):
        F, G, _ = M.stage2_objectives(X)
        out['F'], out['G'] = F, G


def stage1(rng, seed):
    # Initial population: the Table 1 seed plus uniform samples inside the bounds (MATLAB ga's creation).
    X0 = np.vstack([M.SEED[:NG], M.LO[:NG] + rng.random((149, NG))*(M.HI[:NG] - M.LO[:NG])])
    res = minimize(Stage1(), GA(pop_size=150, sampling=X0), ('n_gen', 60), seed=seed, verbose=False)
    return np.asarray(res.X, float), float(np.atleast_1d(res.F)[0])


def stage2(rng, seed, p1, n_gen):
    rand = M.LO + rng.random((200, len(M.NAMES)))*(M.HI - M.LO)
    if p1 is None:
        X0 = rand
    else:
        # Sec. 3.B: p1 copied 50 times, copies 2-50 scaled by (1 + U[-0.01, 0.01]); the paper's population is
        # 200, so the other 150 come from uniform creation as in MATLAB's gamultiobj.
        p = np.r_[p1, np.zeros(len(M.FREEFORM))]
        cloud = p*(1. + rng.uniform(-.01, .01, (50, len(p))))
        cloud[0] = p
        X0 = np.vstack([np.clip(cloud, M.LO, M.HI), rand[:150]])
    res = minimize(Stage2(), NSGA2(pop_size=200, sampling=X0), ('n_gen', n_gen), seed=seed, verbose=False)
    return res


def select(res):
    """Sec. 3.B selection: zero penalty and no leakage, then smallest center RMS among those near the target width."""
    X = res.pop.get('X')
    F, G, r = M.stage2_objectives(X)
    ok = (r['frac'] == 1.) & (r['Pwedge'] == 0) & (G[:, 0] <= 0)
    if not ok.any():
        return None
    nd = _nondominated(F[ok])
    idx = np.flatnonzero(ok)[nd]
    near = idx[F[idx, 2] <= 1.] if (F[idx, 2] <= 1.).any() else idx
    best = near[np.argmin(r['rms'][near, M.CENTER_FIELD])]
    return X[best], len(idx)


def _nondominated(F):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
            keep[i] = not dom.any()
    return np.flatnonzero(keep)


def describe(x):
    r = M.trace(x[None])
    return dict(x=x.tolist(), center_rms=float(r['rms'][0, M.CENTER_FIELD]), worst_rms=float(np.nanmax(r['rms'][0])),
                mean_rms=float(np.nanmean(r['rms'][0])), field_rms=r['rms'][0].tolist(), W=float(r['W'][0]),
                frac=float(r['frac'][0]), Pwedge=float(r['Pwedge'][0]), sep=float(r['sep'][0]))


def sensitivity(x, rng):
    """Monte Carlo around p*: relative perturbations, correlation of each parameter's error with the center RMS."""
    eps = rng.uniform(-MC_REL, MC_REL, (MC_SAMPLES, len(x)))
    r = M.trace(np.clip(x*(1. + eps), M.LO, M.HI))
    rms = r['rms'][:, M.CENTER_FIELD]
    ok = (r['frac'] == 1.) & np.isfinite(rms)
    nominal = float(M.trace(x[None])['rms'][0, M.CENTER_FIELD])
    d = rms[ok] - nominal
    corr = np.zeros(len(x))
    for i in range(len(x)):
        e = eps[ok, i]*x[i]
        if np.std(e) > 0 and np.std(d) > 0:
            corr[i] = abs(np.corrcoef(e, d)[0, 1])
    return dict(score=float(np.mean(np.abs(d))) if len(d) else None, lost_fraction=float(1 - ok.mean()),
                top=M.NAMES[int(np.argmax(corr))], corr=dict(zip(M.NAMES, corr.round(4).tolist())))


def run(job):
    mode, i = job
    path = OUT/mode/f'run_{i:03d}.json'
    if path.exists():
        return path
    rng = np.random.default_rng(1000*('full', 'two50', 'only50').index(mode) + i)
    out = dict(mode=mode, run=i)
    t0 = time.time()
    p1 = None
    if mode != 'only50':
        p1, f1 = stage1(rng, i)
        out['stage1'] = dict(fitness=f1, seconds=time.time() - t0, **describe(np.r_[p1, np.zeros(len(M.FREEFORM))]))
    res = stage2(rng, i, p1, 300 if mode == 'full' else 50)
    sel = select(res)
    out['seconds'] = time.time() - t0
    if sel is None:
        out['final'] = None
    else:
        x, n_front = sel
        out['final'] = describe(x)
        out['final']['feasible_front'] = n_front
        if mode == 'full':
            out['sensitivity'] = sensitivity(x, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('modes', nargs='+', choices=['full', 'two50', 'only50'])
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--workers', type=int, default=18)
    a = ap.parse_args()
    jobs = [(m, i) for m in a.modes for i in range(a.runs)]
    with Pool(a.workers) as pool:
        for k, p in enumerate(pool.imap_unordered(run, jobs), 1):
            print(f'{k}/{len(jobs)} {p.parent.name}/{p.name}', flush=True)


if __name__ == '__main__':
    main()
