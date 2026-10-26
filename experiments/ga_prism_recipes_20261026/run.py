"""Our quad optimizer recipes versus the two-stage GA, on the GA paper's freeform AR prism.

Same tracer, variables, bounds and constraints as experiments/ga_prism_replication_20261026. Same budget as the GA's
two-stage arm (Stage 1 150 x 60 + Stage 2 200 x 50 = 19,000 evaluations), every evaluation counted.
Starts: the paper's Sec. 4 robustness variants of the Table 1 seed, R_B in {-56, -80, -90} x R_A in {132, 150, 200},
freeform terms zero. Genes are normalised to [-1, 1] over the Table 1 / freeform bounds.

Recipes (ported from quad_imaging_env / quad_focus_recipe / quad_cma_control, 2026-10-16):
  TRF_all        least squares (trf) on all 26 genes, forward-difference step 1e-3
  TRF_all_fd1e-2 same, step 1e-2
  TRF_ff         TRF on the 10 freeform terms + 4 image-plane genes only (body frozen)
  STAGE          CMA sigma .02 on the 16 geometric genes for 60% of the budget, then TRF_ff
  CMA_0.02       CMA, fixed sigma .02, population 8, all 26 genes
  CMA_0.1        CMA, fixed sigma .1
  CMA_portfolio  sigma .02 for half the budget from the start, then sigma .1 from the start
  CMA_IPOP       restarts from the incumbent, sigma .05, population doubling, on stall
  CMA_adaptive   restarts from the incumbent on stall, sigma .02 -> x3 (to .3) -> back to .02
  TRF_restarts   TRF to convergence, then TRF again from the incumbent + N(0, .05) until the budget is spent
  GA1_TRF        the paper's Stage 1 GA (9,000 evaluations), then TRF_all for the remaining 10,000

Objective (all recipes): residuals = 9 field RMS radii (mm, capped at 3) + 30 x lost-ray fraction + 10 x wedge
violation counted from 20 um inside each wall (mm) + 10 x shortfall below 1.2 mm C-image separation + image-width
3 x image-width error beyond 0.5 mm. CMA ranks r.r.
In spec = 100% throughput, zero wedge penalty, separation >= 1 mm, |W - 12| <= 1 mm (the GA's selection rule).
The incumbent is the lowest-r.r in-spec design seen.

    /playpen/hisham26/deeplens_env/bin/python experiments/ga_prism_recipes_20261026/run.py   (from repo root)
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/ga_prism_replication_20261026'))
import prism as M  # noqa: E402
from nslens.search.cma_me import CMAES  # noqa: E402

BUDGET = 19000
CHECKPOINTS = (500, 1000, 2000, 5000, 10000, 19000)
MARGIN = .02                    # mm
STALL = 30                      # CMA generations without a new incumbent (quads used 4 at a 300-eval budget)
NG, N = len(M.GEOM), len(M.NAMES)
MID, HALF = (M.HI + M.LO)/2, (M.HI - M.LO)/2
IMAGE = [M.IDX[k] for k in ('xI', 'yI', 'zI', 'tI')]
FF_MASK = np.zeros(N, bool); FF_MASK[NG:] = True; FF_MASK[IMAGE] = True
GEO_MASK = np.zeros(N, bool); GEO_MASK[:NG] = True
ALL_MASK = np.ones(N, bool)
RECIPES = ['TRF_all', 'TRF_all_fd1e-2', 'TRF_ff', 'STAGE', 'CMA_0.02', 'CMA_0.1', 'CMA_portfolio', 'CMA_IPOP',
           'CMA_adaptive', 'TRF_restarts', 'GA1_TRF']


def starts():
    out = []
    for rb in (-56., -80., -90.):
        for ra in (132., 150., 200.):
            p = M.SEED.copy(); p[M.IDX['RB']], p[M.IDX['RA']] = rb, ra
            out.append(dict(name=f'RB{int(rb)}_RA{int(ra)}', genes=(p - MID)/HALF))
    return out


def residuals(G):
    """(P, N) normalised genes -> residual matrix (P, 13) and the in-spec mask and trace."""
    r = M.trace(MID + np.clip(G, -1, 1)*HALF)
    rms = np.minimum(np.nan_to_num(r['rms'], nan=3.), 3.)
    sep = np.nan_to_num(r['sep'], nan=-10.)
    werr = np.nan_to_num(np.abs(r['W'] - M.W_TARGET), nan=50.)
    # Wedge and width penalties aim inside the walls (20 um, 0.5 mm): least squares otherwise parks on the boundary.
    wedge = np.maximum(0., np.nan_to_num(r['sAC'], neginf=-1.) + MARGIN) + np.maximum(0., np.nan_to_num(r['sBC'], neginf=-1.) + MARGIN)
    R = np.c_[rms, 30*(1 - r['frac']), 10*wedge, 10*np.maximum(0., M.MIN_SEP + .2 - sep), 3*np.maximum(0., werr - .5)]
    spec = (r['frac'] == 1) & (r['Pwedge'] == 0) & (sep >= M.MIN_SEP) & (werr <= 1.)
    return R, spec


class Done(Exception):
    pass


class Budget:
    def __init__(self, g0):
        self.calls, self.best, self.genes, self.log = 0, np.inf, None, {}
        self.start = np.clip(g0, -1, 1)

    def __call__(self, G):
        G = np.atleast_2d(np.clip(G, -1, 1))
        if self.calls + len(G) > BUDGET:
            G = G[:BUDGET - self.calls]
            if not len(G):
                raise Done
        R, spec = residuals(G)
        for k in range(len(G)):
            self.calls += 1
            v = float(R[k] @ R[k])
            if spec[k] and v < self.best:
                self.best, self.genes = v, G[k].copy()
            for c in CHECKPOINTS:
                if self.calls == c:
                    self.log[c] = self.best
        return R

    def incumbent(self):
        return self.genes if self.genes is not None else self.start


def trf(B, x_full, mask, fd, stop_at=None):
    """TRF on the masked genes, batched forward-difference Jacobian (N_masked evaluations per Jacobian)."""
    idx = np.flatnonzero(mask)
    stop_at = BUDGET if stop_at is None else stop_at

    def full(x):
        g = np.tile(x_full, (len(np.atleast_2d(x)), 1)); g[:, idx] = np.atleast_2d(x)
        return g

    def fun(x):
        if B.calls >= stop_at:
            raise Done
        return B(full(x))[0]

    def jac(x):
        h = fd*np.where(x >= 0, 1., -1.)*np.maximum(1., np.abs(x))
        h = np.where(np.abs(x + h) > 1, -h, h)
        X = np.tile(x, (len(x) + 1, 1)); X[1:] += np.diag(h)
        if B.calls + len(X) > stop_at:
            raise Done
        R = B(full(X))
        return (R[1:] - R[0]).T/h

    try:
        res = least_squares(fun, x_full[idx], jac=jac, bounds=(-np.ones(len(idx)), np.ones(len(idx))), method='trf',
                            x_scale=1., max_nfev=10**6, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        g = x_full.copy(); g[idx] = res.x
        return g, 'converged'
    except Done:
        return None, 'budget'


def cma(B, x0, sigma, pop, end, seed, mask=ALL_MASK, stall=None):
    idx = np.flatnonzero(mask)
    es = CMAES(x0[idx], sigma, popsize=pop, seed=seed, bounds=(-np.ones(len(idx)), np.ones(len(idx))),
               boundary_handling='reflect')
    since, last = 0, B.best
    while B.calls + pop <= end:
        X = np.tile(x0, (pop, 1)); X[:, idx] = es.ask()
        R = B(X)
        es.tell(np.argsort(np.sum(R*R, 1)))
        if B.best < last - 1e-12:
            since, last = 0, B.best
        else:
            since += 1
        if stall and (since >= stall or es.sigma < 1e-4):
            return 'stall'
    return 'budget'


def ga_stage1(B, g0, seed):
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=NG, n_obj=1, xl=M.LO[:NG], xu=M.HI[:NG])

        def _evaluate(self, X, out, *a, **k):
            B(np.c_[(X - MID[:NG])/HALF[:NG], np.tile(g0[NG:], (len(X), 1))])  # counts the evaluations
            out['F'] = M.stage1_fitness(X)[0][:, None]

    rng = np.random.default_rng(seed)
    p0 = MID[:NG] + g0[:NG]*HALF[:NG]
    X0 = np.vstack([p0, M.LO[:NG] + rng.random((149, NG))*(M.HI[:NG] - M.LO[:NG])])
    res = minimize(P(), GA(pop_size=150, sampling=X0), ('n_gen', 60), seed=seed, verbose=False)
    return np.r_[(np.asarray(res.X) - MID[:NG])/HALF[:NG], g0[NG:]]


def run(job):
    start, recipe = job
    path = HERE/'runs'/f"{start['name']}__{recipe}.json"
    if path.exists():
        return path
    g0 = start['genes']
    B = Budget(g0)
    seed = 7 + RECIPES.index(recipe)
    t = time.time()
    phases = []
    try:
        B(g0[None])
        if recipe == 'TRF_all':
            trf(B, g0, ALL_MASK, 1e-3)
        elif recipe == 'TRF_all_fd1e-2':
            trf(B, g0, ALL_MASK, 1e-2)
        elif recipe == 'TRF_ff':
            trf(B, g0, FF_MASK, 1e-3)
        elif recipe == 'STAGE':
            cma(B, g0, .02, 8, int(.6*BUDGET), seed, mask=GEO_MASK)
            trf(B, B.incumbent(), FF_MASK, 1e-3)
        elif recipe == 'CMA_0.02':
            cma(B, g0, .02, 8, BUDGET, seed)
        elif recipe == 'CMA_0.1':
            cma(B, g0, .1, 8, BUDGET, seed)
        elif recipe == 'CMA_portfolio':
            cma(B, g0, .02, 8, BUDGET//2, seed)
            cma(B, g0, .1, 8, BUDGET, seed + 1)
        elif recipe == 'CMA_IPOP':
            pop = 8
            while B.calls + pop <= BUDGET:
                phases.append(dict(at=B.calls, pop=pop))
                cma(B, B.incumbent(), .05, pop, BUDGET, seed + len(phases), stall=STALL)
                pop *= 2
        elif recipe == 'CMA_adaptive':
            sigma = .02
            while B.calls + 8 <= BUDGET:
                phases.append(dict(at=B.calls, sigma=sigma))
                cma(B, B.incumbent(), sigma, 8, BUDGET, seed + len(phases), stall=STALL)
                sigma = min(sigma*3, .3) if sigma < .1 else .02
        elif recipe == 'TRF_restarts':
            rng = np.random.default_rng(seed)
            x = g0
            while B.calls < BUDGET:
                phases.append(dict(at=B.calls))
                g, why = trf(B, x, ALL_MASK, 1e-3)
                if why == 'budget':
                    break
                x = np.clip(B.incumbent() + rng.normal(0, .05, N), -1, 1)
        elif recipe == 'GA1_TRF':
            g1 = ga_stage1(B, g0, seed)
            phases.append(dict(at=B.calls, stage='TRF'))
            trf(B, g1, ALL_MASK, 1e-3)
    except Done:
        pass
    out = dict(start=start['name'], recipe=recipe, calls=B.calls, seconds=time.time() - t, phases=phases,
               in_spec=B.genes is not None, best_score=B.best, log={str(k): v for k, v in B.log.items()})
    if B.genes is not None:
        x = MID + B.genes*HALF
        r = M.trace(x[None])
        out.update(x=x.tolist(), center_rms=float(r['rms'][0, M.CENTER_FIELD]), worst_rms=float(np.max(r['rms'][0])),
                   mean_rms=float(np.mean(r['rms'][0])), field_rms=r['rms'][0].tolist(), W=float(r['W'][0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{start['name']:14s} {recipe:15s} in_spec={out['in_spec']} "
          f"center={out.get('center_rms', np.nan)*1e3:6.1f} worst={out.get('worst_rms', np.nan)*1e3:6.1f} µm "
          f"{out['seconds']:.0f}s", flush=True)
    return path


def main():
    jobs = [(s, r) for s in starts() for r in RECIPES]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 18) as pool:
        list(pool.imap_unordered(run, jobs))


if __name__ == '__main__':
    main()
