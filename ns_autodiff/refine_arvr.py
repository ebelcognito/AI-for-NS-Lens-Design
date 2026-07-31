"""Autodiff refinement of the AR/VR display-prism designs (collimation).

The imaging condition is COLLIMATED (virtual image at infinity): each display point must
leave the pupil as a PARALLEL bundle, so the objective is the per-source angular spread
(collimation error) sigma_s -- NOT a spatial spot. This refiner minimises sum_s sigma_s**2
of the eyebox bundle with the analytic Jacobian of the differentiable prism twin, under the
same family of guards as the spot-fold refiner:
  * throughput-floor penalty per source (don't collimate by dumping eyebox energy),
  * no-erosion TIR critical-angle margin floor (keep the fold a real TIR),
  * trust region + geometry validity.

After optimising on the twin we RE-VALIDATE through the JS oracle (arvr_prism.arvr_metrics):
report fitness, FOV, and per-source sigma/throughput before vs after. Never overwrites.

Usage:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.refine_arvr --cell 5,6
    $PY -m ns_autodiff.refine_arvr --cell 11,19 --iters 60
    $PY -m ns_autodiff.refine_arvr --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner  # noqa: E402
import arvr_prism as AV                                                               # noqa: E402

from .arvr_data import load_export, make_fans, source_n_events                        # noqa: E402
from .element_arvr import PARAM_NAMES, PIDX, geometry_valid_arvr                       # noqa: E402
from .forward_arvr import trace_arvr                                                   # noqa: E402

DTYPE = torch.float64
LO = torch.tensor(AV.LO, dtype=DTYPE)
HI = torch.tensor(AV.HI, dtype=DTYPE)
EXPORT = REPO / "ns_probe" / "results_arvr_qd" / "autodiff_export"


# --------------------------------------------------------------------------- #
#  collimation objective on the twin (frozen eyebox set + frozen weights)
# --------------------------------------------------------------------------- #
class CollimationObjective:
    def __init__(self, design, common, opt_names, *, slack, mu):
        self.common = common
        self.eyebox_half = common["detector"]["eyebox_half"]
        self.eyebox_y = common["detector"]["eyebox_y"]
        self.n_fan = common["sources"]["n_fan"]
        O, D, sid = make_fans(common)
        self.O = torch.tensor(O, dtype=DTYPE)
        self.D = torch.tensor(D, dtype=DTYPE)
        self.sid = sid
        ne = source_n_events(design, n_src=len(common["sources"]["display_x"]))
        self.ne_per_ray = np.array([ne[s] for s in sid])
        self.opt_idx = [PIDX[n] for n in opt_names]
        self.base_xn = ((torch.tensor(design["genotype_vector"], dtype=DTYPE) - LO) / (HI - LO)).detach()
        self.slack, self.mu = slack, mu
        # freeze the eyebox membership + weights at the champion
        with torch.no_grad():
            ang, en, py, rc = self._raw_trace(self.base_xn[self.opt_idx])
        self.groups = []          # list of (idx_tensor, w0_tensor, T0) per source
        for s in sorted(set(sid)):
            m = (sid == s) & rc.numpy() & (en.numpy() > 0) & \
                (np.abs(py.numpy() - self.eyebox_y) <= self.eyebox_half)
            if m.sum() >= 2:
                idx = np.where(m)[0]
                w0 = en.numpy()[idx]
                self.groups.append((torch.tensor(idx), torch.tensor(w0, dtype=DTYPE),
                                    float(w0.sum() / self.n_fan)))
        self.floor = [T0 * (1.0 - slack) for (_, _, T0) in self.groups]

    def params_from(self, xn_opt):
        xn = self.base_xn.clone()
        xn[self.opt_idx] = xn_opt
        return LO + xn.clamp(0.0, 1.0) * (HI - LO)

    def _raw_trace(self, xn_opt):
        p = self.params_from(xn_opt)
        ang = torch.zeros(len(self.sid), dtype=DTYPE)
        en = torch.zeros(len(self.sid), dtype=DTYPE)
        py = torch.zeros(len(self.sid), dtype=DTYPE)
        rc = torch.zeros(len(self.sid), dtype=torch.bool)
        margins = []
        for ne in sorted(set(int(x) for x in self.ne_per_ray)):
            ii = np.where(self.ne_per_ray == ne)[0]
            tr = trace_arvr(p, self.O[ii], self.D[ii], ne)
            t = torch.tensor(ii)
            ang = ang.index_copy(0, t, tr.exit_angle); en = en.index_copy(0, t, tr.energy)
            py = py.index_copy(0, t, tr.pupil_y); rc = rc.index_copy(0, t, tr.reached)
            mm = tr.min_tir_margin_deg()
            if torch.is_tensor(mm) and torch.isfinite(mm):
                margins.append(mm)
        self._last_margin = float(min(m.item() for m in margins)) if margins else float("nan")
        return ang, en, py, rc

    def residual(self, xn_opt):
        ang, en, py, rc = self._raw_trace(xn_opt)
        rs, thr = [], []
        for (idx, w0, T0), floor in zip(self.groups, self.floor):
            a = ang[idx]; w = w0
            W = w.sum().clamp_min(1e-30)
            theta = (w * a).sum() / W
            rs.append(torch.sqrt(w / W) * (a - theta))           # sum of squares -> sigma_s^2
            T = en[idx].sum() / self.n_fan
            thr.append((self.mu ** 0.5) * torch.clamp(floor - T, min=0.0).reshape(1))
        return torch.cat(rs + thr)

    def diagnostics(self, xn_opt):
        with torch.no_grad():
            ang, en, py, rc = self._raw_trace(xn_opt)
            sig, Ts = [], []
            for (idx, w0, T0) in self.groups:
                a = ang[idx]; W = w0.sum()
                theta = (w0 * a).sum() / W
                sig.append(float(torch.sqrt((w0 * (a - theta) ** 2).sum() / W)))
                Ts.append(float(en[idx].sum() / self.n_fan))
        return dict(sigma_mean=float(np.mean(sig)), sigma=sig, T=Ts,
                    margin=self._last_margin)


def jacobian(obj, xn):
    x = xn.detach().clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(lambda z: obj.residual(z), x,
                                              vectorize=True).detach()


def lm_refine(obj, *, iters, margin_floor, max_step=0.08, lam0=1e-2, verbose=True,
              oracle_fit=None):
    """Twin-gradient LM with an HONEST-METRIC ACCEPTANCE GUARD: the twin's collimation
    objective supplies the (cheap) descent direction, but if `oracle_fit` is given we
    evaluate the real JS-engine fitness at every accepted iterate and RETURN the best-by-
    oracle design. This makes refinement unable to regress the true metric -- essential here
    because the collimation Jacobian is near-rank-deficient (kappa ~ 1e20) and these designs
    start already well-collimated, so the twin proxy overfits without it."""
    xn = obj.base_xn[obj.opt_idx].detach().clone()
    r = obj.residual(xn).detach(); cost = float(r @ r); lam = lam0
    d0 = obj.diagnostics(xn)
    best_fit = oracle_fit(xn) if oracle_fit else None
    best_xn = xn.clone()
    hist = [dict(it=0, cost=cost, oracle_fit=best_fit, **d0)]
    last_J = jacobian(obj, xn)
    for it in range(1, iters + 1):
        J = jacobian(obj, xn).numpy(); last_J = torch.tensor(J)
        rn = r.numpy(); g = J.T @ rn; H = J.T @ J
        sv = np.linalg.svd(J, compute_uv=False)
        kappa = float(sv[0] / max(sv[-1], 1e-30))
        accepted, rej = False, 0; step = np.zeros(len(obj.opt_idx))
        for _ in range(14):
            try:
                step = np.linalg.solve(H + lam * np.eye(H.shape[0]), -g)
            except np.linalg.LinAlgError:
                lam *= 4; continue
            sn = float(np.linalg.norm(step))
            if sn > max_step:
                step *= max_step / sn
            xnt = (xn + torch.tensor(step, dtype=DTYPE)).clamp(0, 1)
            if not geometry_valid_arvr(obj.params_from(xnt)):
                lam = min(lam * 4, 1e6); continue
            rt = obj.residual(xnt).detach(); dt = obj.diagnostics(xnt)
            if not np.isnan(dt["margin"]) and dt["margin"] < margin_floor:
                rej += 1; lam = min(lam * 4, 1e6); continue
            ct = float(rt @ rt)
            if ct < cost:
                xn, r, cost = xnt, rt, ct; lam = max(lam * 0.5, 1e-9); accepted = True; break
            lam = min(lam * 4, 1e6)
        d = obj.diagnostics(xn)
        of = oracle_fit(xn) if (oracle_fit and accepted) else (best_fit if oracle_fit else None)
        if oracle_fit and accepted and of is not None and of > best_fit:
            best_fit, best_xn = of, xn.clone()
        hist.append(dict(it=it, cost=cost, kappa=kappa, oracle_fit=of, **d))
        if verbose:
            ostr = f" oracle_fit={of:.4f}(best {best_fit:.4f})" if oracle_fit else ""
            print(f"  [lm {it:2d}] sigma_mean={d['sigma_mean']:.4f} "
                  f"T={[round(t,3) for t in d['T']]} margin={d['margin']:.2f} "
                  f"kappa={kappa:.1e}{ostr} {'acc' if accepted else ('REJm' if rej else 'STALL')}",
                  flush=True)
        if not accepted and lam >= 1e6:
            break
    ret = best_xn if oracle_fit else xn
    return ret, hist, last_J, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="5,6")
    ap.add_argument("--all", action="store_true", help="refine every design in the export")
    ap.add_argument("--params", default="cx,thick,fsag,bsag,tsag,dsag,tilt_a,tilt_b,n")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--slack", type=float, default=0.03)
    ap.add_argument("--mu", type=float, default=50.0)
    ap.add_argument("--margin-floor", type=float, default=None)
    ap.add_argument("--max-step", type=float, default=0.08)
    ap.add_argument("--out", default=str(EXPORT / "refined"))
    args = ap.parse_args()
    opt_names = [s for s in args.params.split(",") if s]
    ex = load_export(); common = ex["common"]
    designs = ex["designs"] if args.all else \
        [d for d in ex["designs"] if tuple(d["cell"]) == tuple(int(v) for v in args.cell.split(","))]
    if not designs:
        raise SystemExit("design not found")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = []

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for design in designs:
            cell = tuple(design["cell"])
            obj = CollimationObjective(design, common, opt_names, slack=args.slack, mu=args.mu)
            d0 = obj.diagnostics(obj.base_xn[obj.opt_idx])
            mfloor = (d0["margin"] - 0.25) if (args.margin_floor is None and not np.isnan(d0["margin"])) \
                else (args.margin_floor if args.margin_floor is not None else -1e9)
            print("=" * 78)
            print(f"AR/VR COLLIMATION REFINE  cell {cell}  ({len(obj.groups)} sources in eyebox)")
            print(f"  start twin sigma_mean={d0['sigma_mean']:.4f} deg  T={[round(t,3) for t in d0['T']]}  "
                  f"margin={d0['margin']:.2f}")
            print("-" * 78)

            def oracle_fit(xn, _obj=obj):
                p = _obj.params_from(xn).detach().numpy()
                return AV.evaluate(runner, p)[0]

            xnF, hist, J, r = lm_refine(obj, iters=args.iters, margin_floor=mfloor,
                                        max_step=args.max_step, oracle_fit=oracle_fit)
            pF = obj.params_from(xnF).detach().numpy()
            p0 = obj.params_from(obj.base_xn[obj.opt_idx]).detach().numpy()

            # ---- honest oracle re-validation (the real QD metric) ----
            f0, m0 = AV.evaluate(runner, p0)
            fF, mF = AV.evaluate(runner, pF)
            print("-" * 78)
            print(f"RESULT cell {cell}")
            print(f"  twin sigma_mean:  {hist[0]['sigma_mean']:.4f} -> {hist[-1]['sigma_mean']:.4f} deg")
            print(f"  ORACLE fitness:   {f0:.4f} -> {fF:.4f}")
            if m0 and mF:
                s0 = [round(f['sigma'], 3) if f['present'] else None for f in m0['fields']]
                sF = [round(f['sigma'], 3) if f['present'] else None for f in mF['fields']]
                print(f"  ORACLE sigma/src: {s0} -> {sF} deg")
                print(f"  ORACLE FOV:       {m0['fov']:.2f} -> {mF['fov']:.2f} deg   "
                      f"(min throughput {m0['T_min']:.3f} -> {mF['T_min']:.3f})")
            sig_mean_0 = np.mean([f['sigma'] for f in m0['fields'] if f['present']]) if m0 else float('nan')
            sig_mean_F = np.mean([f['sigma'] for f in mF['fields'] if f['present']]) if mF else float('nan')
            summary.append(dict(cell=list(cell), fitness0=f0, fitnessF=fF,
                                oracle_sigma_mean0=float(sig_mean_0), oracle_sigma_meanF=float(sig_mean_F),
                                fov0=(m0['fov'] if m0 else None), fovF=(mF['fov'] if mF else None)))
            rep = dict(cell=list(cell), method="autodiff-collimation-LM",
                       genotype_start=dict(zip(PARAM_NAMES, [float(v) for v in p0])),
                       genotype_refined=dict(zip(PARAM_NAMES, [float(v) for v in pF])),
                       twin_sigma_start=hist[0]['sigma_mean'], twin_sigma_refined=hist[-1]['sigma_mean'],
                       oracle_fitness_start=f0, oracle_fitness_refined=fF,
                       oracle_metrics_start=m0, oracle_metrics_refined=mF, history=hist)
            base = out / f"refined_cell{cell[0]}_{cell[1]}.json"; rp, k = base, 0
            while rp.exists():
                k += 1; rp = base.with_name(f"refined_cell{cell[0]}_{cell[1]}_{k}.json")
            rp.write_text(json.dumps(rep, indent=2, default=float))
            print(f"  [report] {rp}")

    if summary:
        print("\n" + "=" * 78)
        print(f"SUMMARY ({len(summary)} designs)")
        print("=" * 78)
        print(f"  {'cell':>9} {'fit0':>7} {'fitF':>7} {'sig0':>7} {'sigF':>7} {'fov0':>6} {'fovF':>6}")
        for s in summary:
            print(f"  {str(tuple(s['cell'])):>9} {s['fitness0']:>7.3f} {s['fitnessF']:>7.3f} "
                  f"{s['oracle_sigma_mean0']:>7.3f} {s['oracle_sigma_meanF']:>7.3f} "
                  f"{(s['fov0'] or 0):>6.2f} {(s['fovF'] or 0):>6.2f}")
        imp = [s for s in summary if s['fitnessF'] > s['fitness0']]
        print(f"\n  fitness improved: {len(imp)}/{len(summary)}   "
              f"mean sigma {np.mean([s['oracle_sigma_mean0'] for s in summary]):.3f} -> "
              f"{np.mean([s['oracle_sigma_meanF'] for s in summary]):.3f} deg")
        (out / "refine_arvr_summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
