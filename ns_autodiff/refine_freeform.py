"""Guarded freeform refinement -- the payoff of the freeform surface type.

Causal demonstration that freeform DOF buy real headroom a circular arc cannot reach:

  Phase 1 (ARC)      : refine the base DOF (cx,thick,fsag,bsag,tilt_a,tilt_b,n) from the QD
                       champion -- this is exactly refine_autodiff, the circular-arc twin.
  Phase 2 (FREEFORM) : FREEZE the arc-refined base, unlock ONLY the polynomial sag coefficients
                       (started at 0, so phase 2 begins at the phase-1 design) and refine them.

If phase 2's spot beats phase 1's, the gain is purely what the freeform shape adds on top of the
best arc -- aberration correction the circle's single curvature cannot express.

Optimizer: Levenberg-Marquardt (analytic autodiff Jacobian through trace_ff), with the
refine_lm/refine_autodiff guards (throughput floor, no-erosion TIR margin floor, trust region)
PLUS Tikhonov regularization on the polynomial coefficients -- the freeform analog of the
"regularization is mandatory" rule (a high-order coefficient can sharpen the tracked sub-bundle
while rippling the surface between rays; the penalty + a DENSE twin-native honest re-score guard
against that). The coefficients are scaled to "sag bump in mm at the aperture edge" so the LM
trust region sees a well-conditioned vector.

Honest metric: the oracle CANNOT trace a freeform surface, so re-validation is twin-native -- a
DENSER bundle (same dominant topology) re-traced through the freeform twin. The twin is validated
to ~1e-9 against the oracle on arcs (validate_freeform*.py), so it is the trustworthy forward model
here. We report the dense honest spot and keep the best-by-honest iterate (no-regression guard).

Usage:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.refine_freeform --cell 3,5 --powers 2 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner  # noqa: E402
import fold_demo as FD                                                                # noqa: E402

from . import topology as TOPO                                                        # noqa: E402
from .element_torch import ELEM_H, PARAM_NAMES, PIDX, geometry_valid                  # noqa: E402
from .forward_freeform import trace_ff, dominant_signature_twin                       # noqa: E402

DTYPE = torch.float64
LO = torch.tensor(FD.LO, dtype=DTYPE)
HI = torch.tensor(FD.HI, dtype=DTYPE)
ARCHIVE = REPO / "ns_probe" / "results_fold_qd" / "archive.json"
H_CHAR = ELEM_H                # characteristic aperture for coefficient scaling


class TwinObjectiveFF:
    """Residual on the freeform twin over an opt vector x = [base_norm_opt | u_A | u_B], where
    u are coefficients scaled so a_power = u * H_CHAR**(-power) (=> u ~ edge sag bump in mm)."""

    def __init__(self, topo, base_names, powers, *, thru0, slack, mu, reg, u0=None):
        self.topo = topo
        self.u0_init = None if u0 is None else torch.as_tensor(u0, dtype=DTYPE)
        self.w = torch.tensor(topo.w0 / topo.w0.sum(), dtype=DTYPE)
        self.sqrt_w = torch.sqrt(self.w)
        self.base_idx = [PIDX[n] for n in base_names]
        self.powers = tuple(powers)
        self.d = len(self.powers)
        self.scale = torch.tensor([H_CHAR ** (-pw) for pw in self.powers], dtype=DTYPE)
        self.base_xn = ((torch.tensor([topo.p0[n] for n in PARAM_NAMES], dtype=DTYPE) - LO)
                        / (HI - LO)).detach()
        self.floor = float(thru0) * (1.0 - slack)
        self.mu = float(mu)
        self.reg = float(reg)
        self.nb = len(self.base_idx)

    def split(self, x):
        xb = x[:self.nb]
        u = x[self.nb:]
        uA, uB = u[:self.d], u[self.d:]
        return xb, uA, uB

    def params_from(self, x):
        xb, uA, uB = self.split(x)
        xn = self.base_xn.clone()
        if self.nb:
            xn[self.base_idx] = xb
        p = LO + xn.clamp(0.0, 1.0) * (HI - LO)
        if self.d:
            coeffs = {"A": uA * self.scale, "B": uB * self.scale}
        else:
            coeffs = None
        return p, coeffs

    def residual(self, x):
        p, coeffs = self.params_from(x)
        tr = trace_ff(p, self.topo, coeffs=coeffs, powers=self.powers)
        ybar = (self.w * tr.y).sum()
        r_spot = self.sqrt_w * (tr.y - ybar)
        thru = tr.energy.mean()
        r_thru = (self.mu ** 0.5) * torch.clamp(self.floor - thru, min=0.0)
        parts = [r_spot, r_thru.reshape(1)]
        if self.d and self.reg > 0:
            parts.append((self.reg ** 0.5) * x[self.nb:])     # Tikhonov on the scaled coeffs
        return torch.cat(parts), tr

    def diagnostics(self, x):
        with torch.no_grad():
            p, coeffs = self.params_from(x)
            tr = trace_ff(p, self.topo, coeffs=coeffs, powers=self.powers)
            ybar = (self.w * tr.y).sum()
            spot = float(torch.sqrt((self.w * (tr.y - ybar) ** 2).sum().clamp_min(0) + 1e-30))
            thru = float(tr.energy.mean())
            margin = float(tr.min_tir_margin_deg())
        return dict(spot=spot, thru=thru, margin=margin, reached=int(tr.reached.sum()))

    def x0(self):
        xb = self.base_xn[self.base_idx].detach().clone() if self.nb else torch.empty(0, dtype=DTYPE)
        u = (self.u0_init.clone() if (self.u0_init is not None and self.d)
             else torch.zeros(2 * self.d, dtype=DTYPE))
        return torch.cat([xb, u])


def jacobian(obj, x):
    xj = x.detach().clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(lambda z: obj.residual(z)[0], xj,
                                           vectorize=True, create_graph=False)
    return J.detach()


def honest_dense(obj, x, *, n_dense=201):
    """Twin-native honest spot: re-trace the SAME geometry on a DENSER bundle (oversampling the
    optimization bundle, so genuinely probing between the optimization rays) of the same dominant
    topology -- energy-weighted RMS over rays that reach the sensor (floating centroid). Catches a
    freeform shape that sharpens the coarse optimization bundle but ripples between its samples."""
    p, coeffs = obj.params_from(x)
    dys = np.linspace(float(obj.topo.dys.min()), float(obj.topo.dys.max()), n_dense)
    shim = types.SimpleNamespace(dys=dys, events=obj.topo.events)
    with torch.no_grad():
        tr = trace_ff(p, shim, coeffs=coeffs, powers=obj.powers)
        reached = tr.reached
        y = tr.y[reached]; w = tr.energy[reached]
        if w.numel() == 0 or float(w.sum()) <= 0:
            return float("nan"), 0.0, float(tr.min_tir_margin_deg())
        ybar = (w * y).sum() / w.sum()
        spot = float(torch.sqrt((w * (y - ybar) ** 2).sum() / w.sum()))
        margin = float(tr.min_tir_margin_deg())
    return spot, float(reached.float().mean()), margin


def realized_order(obj, x):
    """The realized typed interaction signature of the design at x, in the twin (the thing the
    differentiator's frozen graph assumes). Returns (signature, min_tir_margin_deg)."""
    p, coeffs = obj.params_from(x)
    with torch.no_grad():
        tr = trace_ff(p, obj.topo, coeffs=coeffs, powers=obj.powers)
    dom, _ = dominant_signature_twin(tr, obj.topo.p0["cx"])
    return dom, float(tr.min_tir_margin_deg())


def lm_refine_ff(obj, *, iters, preserve_order=True, tir_eps=0.05, max_step=0.15,
                 lam0=1e-2, verbose=True, tag=""):
    """Order-preserving LM. The only topology guard is that the realized interaction ORDER must
    not change (reject any step whose typed signature != the frozen one) -- a TIR->refract flip
    OR a nearest-surface swap both change the order and break the differentiator's frozen graph.
    The TIR margin is allowed to shrink freely (robustness is not guarded); tir_eps just keeps the
    batch uniformly TIR at frozen-TIR steps so the order is well-defined."""
    frozen = tuple((t, f) for (t, f) in obj.topo.events)
    x = obj.x0()
    r = obj.residual(x)[0].detach()
    cost = float(r @ r)
    lam = lam0
    d0 = obj.diagnostics(x)
    hsp0, _, _ = honest_dense(obj, x)
    history = [dict(it=0, **d0, honest=hsp0, cost=cost, lam=lam, order="frozen")]
    best_x = x.clone(); best_honest = hsp0
    for it in range(1, iters + 1):
        J = jacobian(obj, x).numpy()
        rn = r.numpy()
        g = J.T @ rn
        H = J.T @ J
        accepted, rej_order = False, 0
        step = np.zeros(len(x))
        for _inner in range(14):
            try:
                step = np.linalg.solve(H + lam * np.eye(H.shape[0]), -g)
            except np.linalg.LinAlgError:
                lam *= 4.0; continue
            sn = float(np.linalg.norm(step))
            if sn > max_step:
                step *= max_step / sn
            x_try = x + torch.tensor(step, dtype=DTYPE)
            x_try[:obj.nb] = x_try[:obj.nb].clamp(0.0, 1.0)
            if not geometry_valid(obj.params_from(x_try)[0]):
                lam = min(lam * 4.0, 1e6); continue
            # ORDER-PRESERVATION GUARD (the only topology guard): the realized signature must
            # equal the frozen one, and frozen-TIR steps must stay strictly past critical (eps).
            if preserve_order:
                dom_try, marg_try = realized_order(obj, x_try)
                if dom_try != frozen or (not np.isnan(marg_try) and marg_try < tir_eps):
                    rej_order += 1; lam = min(lam * 4.0, 1e6); continue
            r_try = obj.residual(x_try)[0].detach()
            c_try = float(r_try @ r_try)
            if c_try < cost:
                x, r, cost = x_try, r_try, c_try
                lam = max(lam * 0.5, 1e-9); accepted = True
                break
            lam = min(lam * 4.0, 1e6)
        d = obj.diagnostics(x)
        hsp, _, _ = honest_dense(obj, x)
        if accepted and not np.isnan(hsp) and hsp < best_honest:
            best_honest = hsp; best_x = x.clone()
        history.append(dict(it=it, **d, honest=hsp, cost=cost, lam=lam))
        if verbose:
            print(f"  [{tag} {it:2d}] spot={d['spot']:.4f} honest={hsp:.4f} "
                  f"thru={d['thru']:.3f} margin={d['margin']:6.2f} lam={lam:.1e} "
                  f"{'acc' if accepted else ('REJ-order' if rej_order else 'STALL')}", flush=True)
        if not accepted and lam >= 1e6:
            if verbose:
                print(f"  [{tag}] converged / stalled.")
            break
    return best_x, history, best_honest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="3,5")
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--base-params", default="cx,thick,fsag,bsag,tilt_a,tilt_b,n")
    ap.add_argument("--powers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--rays", type=int, default=121)
    ap.add_argument("--beam-w", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--thru-slack", type=float, default=0.02)
    ap.add_argument("--thru-weight", type=float, default=50.0)
    ap.add_argument("--reg", type=float, default=2e-2, help="Tikhonov weight on poly coeffs")
    ap.add_argument("--tir-eps", type=float, default=0.05,
                    help="keep frozen-TIR steps >= this far past critical (numerical; keeps the order well-defined). NOT a robustness floor.")
    ap.add_argument("--allow-order-change", action="store_true",
                    help="disable the order-preservation guard (lets the frozen-topology differentiator break)")
    ap.add_argument("--out", default=str(REPO / "ns_autodiff" / "results_refine_freeform"))
    args = ap.parse_args()
    cell = tuple(int(v) for v in args.cell.split(","))
    base_names = [s.strip() for s in args.base_params.split(",") if s.strip()]

    print("=" * 80)
    print("FREEFORM REFINEMENT (arc baseline -> freeze base -> unlock polynomial sag)")
    print("=" * 80)
    t0 = time.time()
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        topo = TOPO.extract(runner, Path(args.archive), cell, n_rays=args.rays, beam_w=args.beam_w)
        p0 = torch.tensor([topo.p0[n] for n in PARAM_NAMES], dtype=DTYPE)
        with torch.no_grad():
            thru0 = float(trace_ff(p0, topo, coeffs=None).energy.mean())
        print(f"cell {cell}  topology: {topo.signature_str()}")
        print(f"dom_share={topo.dom_share:.3f} archive_spot={topo.archive_spot:.3f} "
              f"tracked={topo.meta['n_tracked']}/{args.rays}  powers={tuple(args.powers)}")

        base0 = TwinObjectiveFF(topo, base_names, (), thru0=thru0, slack=args.thru_slack,
                                mu=args.thru_weight, reg=0.0)
        d0 = base0.diagnostics(base0.x0())
        preserve = not args.allow_order_change
        print(f"start (twin): spot={d0['spot']:.4f} thru={d0['thru']:.3f} margin={d0['margin']:.2f}")
        print(f"guard: order-preservation={'ON' if preserve else 'OFF'} "
              f"(TIR margin free to shrink to {args.tir_eps:.2f} deg; robustness NOT guarded)")
        print("-" * 80)

        # ----- Phase 1: ARC refine (base DOF only) -----
        print("[Phase 1] ARC refine (base DOF, circular faces)")
        arc_obj = TwinObjectiveFF(topo, base_names, (), thru0=thru0, slack=args.thru_slack,
                                  mu=args.thru_weight, reg=0.0)
        x_arc, hist_arc, honest_arc = lm_refine_ff(arc_obj, iters=args.iters,
                                                   preserve_order=preserve, tir_eps=args.tir_eps, tag="arc")
        spot_arc = arc_obj.diagnostics(x_arc)["spot"]
        # arc-refined base, as a frozen genotype for phase 2
        p_arc, _ = arc_obj.params_from(x_arc)
        topo_arc = TOPO.FrozenTopology(**{**topo.__dict__,
                                          "p0": dict(zip(PARAM_NAMES, [float(v) for v in p_arc]))})

        print("-" * 80)
        # ----- Phase 2: FREEFORM refine (freeze base, unlock poly) -----
        print("[Phase 2] FREEFORM refine (base FROZEN at arc optimum, unlock polynomial sag)")
        ff_obj = TwinObjectiveFF(topo_arc, [], tuple(args.powers), thru0=thru0,
                                 slack=args.thru_slack, mu=args.thru_weight, reg=args.reg)
        x_ff, hist_ff, honest_ff = lm_refine_ff(ff_obj, iters=args.iters,
                                                preserve_order=preserve, tir_eps=args.tir_eps, tag="ff")
        spot_ff = ff_obj.diagnostics(x_ff)["spot"]
        p_ff, coeffs_ff = ff_obj.params_from(x_ff)
        dom_ff, _ = dominant_signature_twin(
            trace_ff(p_ff, topo_arc, coeffs=coeffs_ff, powers=tuple(args.powers)), topo_arc.p0["cx"])
        topo_preserved = (dom_ff == tuple((t, f) for (t, f) in topo.events))

    dt = time.time() - t0
    print("=" * 80)
    print(f"RESULT  ({dt:.0f}s)  cell {cell}")
    print(f"  twin dense-spot  champion {d0['spot']:.4f}")
    print(f"  Phase 1 ARC      twin {spot_arc:.4f}  honest {honest_arc:.4f}")
    print(f"  Phase 2 FREEFORM twin {spot_ff:.4f}  honest {honest_ff:.4f}")
    gain = 100 * (1 - honest_ff / honest_arc) if honest_arc > 0 else float("nan")
    print(f"  --> freeform gain over best arc (honest): {gain:+.1f}%")
    print(f"  topology preserved: {topo_preserved}   coeffs(A)={[round(float(v),3) for v in (coeffs_ff['A']*1)]}"
          if coeffs_ff else "")
    if coeffs_ff:
        print(f"      a_powers={tuple(args.powers)}  A={[f'{float(v):.2e}' for v in coeffs_ff['A']]}"
              f"  B={[f'{float(v):.2e}' for v in coeffs_ff['B']]}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = out / f"refine_freeform_cell{cell[0]}_{cell[1]}.json"
    rep, k = base, 0
    while rep.exists():
        k += 1; rep = base.with_name(f"refine_freeform_cell{cell[0]}_{cell[1]}_{k}.json")
    report = dict(method="freeform-twin", cell=list(cell), topology=topo.signature_str(),
                  powers=list(args.powers),
                  champion_spot=d0["spot"],
                  arc=dict(twin_spot=spot_arc, honest_spot=honest_arc,
                           genotype=dict(zip(PARAM_NAMES, [float(v) for v in p_arc]))),
                  freeform=dict(twin_spot=spot_ff, honest_spot=honest_ff,
                                coeffs_A=[float(v) for v in coeffs_ff["A"]] if coeffs_ff else [],
                                coeffs_B=[float(v) for v in coeffs_ff["B"]] if coeffs_ff else [],
                                topology_preserved=bool(topo_preserved)),
                  honest_gain_pct=gain,
                  config=dict(base_params=base_names, rays=args.rays, iters=args.iters,
                              reg=args.reg, tir_eps=args.tir_eps, preserve_order=preserve),
                  history_arc=hist_arc, history_ff=hist_ff, wall_seconds=dt)
    rep.write_text(json.dumps(report, indent=2, default=float))
    print(f"[report] {rep}")


if __name__ == "__main__":
    main()
