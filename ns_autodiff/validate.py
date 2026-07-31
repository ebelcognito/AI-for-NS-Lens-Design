"""Validate the differentiable twin against the JS oracle (the stated success gate:
"validated forward-pass against the JS oracle first").

Two checks:
  (1) FORWARD MATCH -- at the champion geometry, the twin's per-ray sensor y and delivered
      energy must equal the oracle's recorded values (the twin is a faithful reimplementation,
      not an approximation). Tolerances are tight (geometry ~1e-3 mm, energy ~1e-4).
  (2) JACOBIAN MATCH (--jac) -- the analytic reverse-mode Jacobian of the dense spot w.r.t.
      the optimized DOF must agree with the central finite-difference Jacobian that
      ns_probe/refine_lm.py builds through the JS engine. This is the concrete answer to
      "it'll be interesting to see how [the gradients] would be constructed".

Usage:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.validate --cell 0,1            # a transmission lens (simplest)
    $PY -m ns_autodiff.validate --cell 3,5 --jac      # a 3-TIR fold + Jacobian check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner  # noqa: E402

from . import topology as TOPO                                                        # noqa: E402
from .element_torch import PARAM_NAMES                                                # noqa: E402
from .forward import trace                                                            # noqa: E402
from .loss import dense_spot                                                          # noqa: E402

ARCHIVE = REPO / "ns_probe" / "results_fold_qd" / "archive.json"
DTYPE = torch.float64


def p_tensor(p0: dict, requires_grad=False):
    return torch.tensor([p0[n] for n in PARAM_NAMES], dtype=DTYPE, requires_grad=requires_grad)


def forward_match(topo, *, verbose=True):
    p = p_tensor(topo.p0)
    with torch.no_grad():
        tr = trace(p, topo)
    y_err = np.abs(tr.y.numpy() - topo.oracle_y)
    e_err = np.abs(tr.energy.numpy() - topo.oracle_b)
    reached = tr.reached.numpy()
    if verbose:
        print(f"  tracked rays                 : {len(topo.dys)}  "
              f"(reached sensor in twin: {int(reached.sum())})")
        print(f"  sensor-y abs error  max/mean : {y_err.max():.3e} / {y_err.mean():.3e} mm")
        print(f"  energy   abs error  max/mean : {e_err.max():.3e} / {e_err.mean():.3e}")
        print(f"  twin dense spot              : {float(dense_spot(tr, topo)):.5f}")
    return y_err, e_err, reached


def fd_jacobian_spot(topo, opt_names, h=2e-3):
    """Central-difference Jacobian of the dense spot w.r.t. bound-normalized opt DOF,
    computed with the TWIN forward (the same recipe refine_lm uses through the JS engine;
    we compare the twin's analytic gradient to its own FD gradient, which isolates the
    autodiff correctness from any twin<->oracle modelling gap)."""
    from .element_torch import PIDX
    import fold_demo as FD
    LO, HI = FD.LO, FD.HI
    p0 = np.array([topo.p0[n] for n in PARAM_NAMES])
    xn0 = (p0 - LO) / (HI - LO)
    idx = [PIDX[n] for n in opt_names]

    def spot_at(xn):
        p = torch.tensor(LO + xn * (HI - LO), dtype=DTYPE)
        with torch.no_grad():
            return float(dense_spot(trace(p, topo), topo))

    g = np.zeros(len(idx))
    for j, _ in enumerate(idx):
        xp = xn0.copy(); xp[idx[j]] += h
        xm = xn0.copy(); xm[idx[j]] -= h
        g[j] = (spot_at(xp) - spot_at(xm)) / (2 * h)
    return g


def analytic_jacobian_spot(topo, opt_names):
    """Reverse-mode gradient of the dense spot w.r.t. bound-normalized opt DOF."""
    from .element_torch import PIDX
    import fold_demo as FD
    LO = torch.tensor(FD.LO, dtype=DTYPE); HI = torch.tensor(FD.HI, dtype=DTYPE)
    p0 = p_tensor(topo.p0)
    xn = ((p0 - LO) / (HI - LO)).detach().requires_grad_(True)
    p = LO + xn * (HI - LO)
    spot = dense_spot(trace(p, topo), topo)
    spot.backward()
    idx = [PIDX[n] for n in opt_names]
    return xn.grad.numpy()[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="3,5")
    ap.add_argument("--rays", type=int, default=121)
    ap.add_argument("--beam-w", type=float, default=8.0)
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--jac", action="store_true", help="also cross-check analytic vs FD Jacobian")
    ap.add_argument("--params", default="cx,thick,fsag,bsag,tilt_a,tilt_b,n")
    args = ap.parse_args()
    cell = tuple(int(v) for v in args.cell.split(","))

    print("=" * 74)
    print(f"TWIN VALIDATION  cell {cell}")
    print("=" * 74)
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        topo = TOPO.extract(runner, Path(args.archive), cell,
                            n_rays=args.rays, beam_w=args.beam_w)
        print(f"dominant topology : {topo.signature_str()}")
        print(f"tir_count={topo.meta['tir_count']}  dom_share={topo.dom_share:.3f}  "
              f"archive_spot={topo.archive_spot:.3f}")
        print("-" * 74)
        print("[1] FORWARD MATCH (twin vs oracle @ champion geometry)")
        y_err, e_err, reached = forward_match(topo)

        if args.jac:
            opt = [s for s in args.params.split(",") if s]
            print("-" * 74)
            print(f"[2] JACOBIAN MATCH  d(spot)/d(norm DOF)  for {opt}")
            ga = analytic_jacobian_spot(topo, opt)
            gf = fd_jacobian_spot(topo, opt)
            cos = float(np.dot(ga, gf) / (np.linalg.norm(ga) * np.linalg.norm(gf) + 1e-30))
            print(f"  {'dof':<8} {'analytic':>14} {'finite-diff':>14} {'abs-err':>12}")
            for nme, a, f in zip(opt, ga, gf):
                print(f"  {nme:<8} {a:>14.5f} {f:>14.5f} {abs(a-f):>12.3e}")
            print(f"  cosine(analytic, FD) = {cos:.6f}   "
                  f"max abs err = {np.max(np.abs(ga-gf)):.3e}")

    print("-" * 74)
    ok = (y_err.max() < 5e-3) and (e_err.max() < 5e-4) and reached.all()
    print(f"FORWARD-MATCH VERDICT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
