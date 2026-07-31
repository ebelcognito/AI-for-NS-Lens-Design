"""Autodiff refinement of a QD-discovered fold -- the differentiable counterpart to
ns_probe/refine_lm.py.

Same problem, same objective, same guards, same outputs -- but the residual Jacobian is
EXACT reverse-mode autodiff through the differentiable twin (forward.py) instead of
2*n_opt central-difference traces through the JS engine. Two optimizers are provided:

  * 'lm'   (default) -- Levenberg-Marquardt / damped least squares on the per-ray residual,
            (J^T J + lambda I) d = -J^T r, with the analytic J. This is the apples-to-apples
            match to refine_lm, so the two can be compared head-to-head.
  * 'adam' -- plain first-order descent on the scalar objective (spot^2 + mu*penalty^2),
            to show the framework supports ordinary gradient methods too.

GUARDS (identical intent to refine_lm, the design doc's "regularization is mandatory"):
  - throughput-floor penalty in the residual (can't sharpen by dumping energy),
  - no-erosion TIR critical-angle margin floor (reject steps that drive folds toward
    critical / out of the archive cell),
  - trust-region cap on the step.

After optimizing on the twin we RE-VALIDATE against the JS oracle: re-trace start & refined
geometries at QD's 21-ray density, report the honest archive-comparable spot, confirm the
dominant topology is preserved, and re-check every TIR event is past critical. Never
overwrites: writes refine_autodiff_cell<i>_<j>[.json/.png] with a +k suffix if taken.

Usage:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.refine_autodiff --cell 3,5
    $PY -m ns_autodiff.refine_autodiff --cell 3,10 --iters 80
    $PY -m ns_autodiff.refine_autodiff --cell 3,5 --optimizer adam --iters 300
    $PY -m ns_autodiff.refine_autodiff --cell 3,5 --margin-floor 1.0   # tightness knee
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
import fold_demo as FD                                                                # noqa: E402
from sequence_tools import (aggregate_by_signature, format_signature,                 # noqa: E402
                            make_single_element_face_classifier, signature_of)

from . import topology as TOPO                                                        # noqa: E402
from .element_torch import ELEM_H, PARAM_NAMES, PIDX, geometry_valid                  # noqa: E402
from .forward import trace                                                            # noqa: E402
from .loss import dense_spot                                                          # noqa: E402

DTYPE = torch.float64
LO = torch.tensor(FD.LO, dtype=DTYPE)
HI = torch.tensor(FD.HI, dtype=DTYPE)
ARCHIVE = REPO / "ns_probe" / "results_fold_qd" / "archive.json"
DEFAULT_PARAMS = ["cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b", "n"]


# --------------------------------------------------------------------------- #
#  differentiable residual / objective on the twin
# --------------------------------------------------------------------------- #
class TwinObjective:
    """Builds the residual vector r(xn_opt) = [sqrt(w_i)(y_i - ybar) ; sqrt(mu)*relu(floor-thru)]
    on the twin, exactly mirroring refine_lm.Objective (frozen weights, floating centroid,
    throughput-floor penalty). Everything is a torch graph so J = dr/dxn is autodiff."""

    def __init__(self, topo, opt_names, *, thru0, slack, mu):
        self.topo = topo
        self.w = torch.tensor(topo.w0 / topo.w0.sum(), dtype=DTYPE)
        self.sqrt_w = torch.sqrt(self.w)
        self.opt_idx = [PIDX[n] for n in opt_names]
        self.base_xn = ((torch.tensor([topo.p0[n] for n in PARAM_NAMES], dtype=DTYPE) - LO)
                        / (HI - LO)).detach()
        self.thru0 = float(thru0)
        self.floor = float(thru0) * (1.0 - slack)
        self.mu = float(mu)

    def params_from(self, xn_opt):
        xn = self.base_xn.clone()
        xn[self.opt_idx] = xn_opt
        return LO + xn.clamp(0.0, 1.0) * (HI - LO)

    def residual(self, xn_opt):
        p = self.params_from(xn_opt)
        tr = trace(p, self.topo)
        ybar = (self.w * tr.y).sum()
        r_spot = self.sqrt_w * (tr.y - ybar)
        thru = tr.energy.mean()
        r_thru = (self.mu ** 0.5) * torch.clamp(self.floor - thru, min=0.0)
        r = torch.cat([r_spot, r_thru.reshape(1)])
        return r, tr

    def diagnostics(self, xn_opt):
        with torch.no_grad():
            p = self.params_from(xn_opt)
            tr = trace(p, self.topo)
            spot = float(dense_spot(tr, self.topo))
            thru = float(tr.energy.mean())
            margin = float(tr.min_tir_margin_deg())
        return dict(spot=spot, thru=thru, margin=margin, reached=int(tr.reached.sum()))


def jacobian(obj, xn_opt):
    """Analytic residual Jacobian J [m+1, n_opt] via reverse-mode autodiff."""
    xn = xn_opt.detach().clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(lambda x: obj.residual(x)[0], xn,
                                           vectorize=True, create_graph=False)
    return J.detach()


# --------------------------------------------------------------------------- #
#  Levenberg-Marquardt (analytic J) with the refine_lm guards
# --------------------------------------------------------------------------- #
def lm_refine(obj, *, iters, margin_floor, max_step=0.1, lam0=1e-2, verbose=True):
    xn = obj.base_xn[obj.opt_idx].detach().clone()
    r, _ = obj.residual(xn)
    r = r.detach()
    cost = float(r @ r)
    lam = lam0
    diag0 = obj.diagnostics(xn)
    history = [dict(it=0, **diag0, cost=cost, lam=lam, kappa=None, step=0.0)]
    last_J = jacobian(obj, xn)
    for it in range(1, iters + 1):
        J = jacobian(obj, xn).numpy()
        last_J = torch.tensor(J)
        rn = r.numpy()
        g = J.T @ rn
        H = J.T @ J
        sv = np.linalg.svd(J, compute_uv=False)
        kappa = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")
        accepted, rej = False, 0
        step = np.zeros(len(obj.opt_idx))
        for _inner in range(14):
            try:
                step = np.linalg.solve(H + lam * np.eye(H.shape[0]), -g)
            except np.linalg.LinAlgError:
                lam *= 4.0; continue
            sn = float(np.linalg.norm(step))
            if sn > max_step:
                step *= max_step / sn
            xn_try = (xn + torch.tensor(step, dtype=DTYPE)).clamp(0.0, 1.0)
            if not geometry_valid(obj.params_from(xn_try)):
                lam = min(lam * 4.0, 1e6); continue
            r_try, _ = obj.residual(xn_try)
            r_try = r_try.detach()
            dtry = obj.diagnostics(xn_try)
            if not np.isnan(dtry["margin"]) and dtry["margin"] < margin_floor:
                rej += 1; lam = min(lam * 4.0, 1e6); continue
            c_try = float(r_try @ r_try)
            if c_try < cost:
                xn, r, cost = xn_try, r_try, c_try
                lam = max(lam * 0.5, 1e-9); accepted = True
                break
            lam = min(lam * 4.0, 1e6)
        d = obj.diagnostics(xn)
        history.append(dict(it=it, **d, cost=cost, lam=lam, kappa=kappa,
                            step=float(np.linalg.norm(step)), rej_margin=rej))
        if verbose:
            print(f"  [lm {it:2d}] spot={d['spot']:.4f} thru={d['thru']:.3f} "
                  f"margin={d['margin']:6.2f} kappa(J)={kappa:10.1f} lam={lam:.1e} "
                  f"{'acc' if accepted else ('REJm' if rej else 'STALL')}", flush=True)
        if not accepted and lam >= 1e6:
            if verbose:
                print("  [lm] converged / stalled.");
            break
    return xn, history, last_J, r


def adam_refine(obj, *, iters, margin_floor, lr=5e-3, verbose=True):
    xn = obj.base_xn[obj.opt_idx].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([xn], lr=lr)
    best_xn = xn.detach().clone()
    best = obj.diagnostics(best_xn)
    history = [dict(it=0, **best, cost=float((obj.residual(best_xn)[0] ** 2).sum()),
                    lam=0.0, kappa=None, step=0.0)]
    for it in range(1, iters + 1):
        opt.zero_grad()
        r, _ = obj.residual(xn)
        cost = (r ** 2).sum()
        cost.backward()
        opt.step()
        with torch.no_grad():
            xn.clamp_(0.0, 1.0)
        d = obj.diagnostics(xn.detach())
        # keep the best in-guard iterate
        if (np.isnan(d["margin"]) or d["margin"] >= margin_floor) and d["spot"] < best["spot"]:
            best, best_xn = d, xn.detach().clone()
        history.append(dict(it=it, **d, cost=float(cost), lam=0.0, kappa=None, step=lr))
        if verbose and it % max(1, iters // 20) == 0:
            print(f"  [adam {it:3d}] spot={d['spot']:.4f} thru={d['thru']:.3f} "
                  f"margin={d['margin']:6.2f}", flush=True)
    return best_xn, history, jacobian(obj, best_xn), obj.residual(best_xn)[0].detach()


# --------------------------------------------------------------------------- #
#  honest re-validation through the JS oracle
# --------------------------------------------------------------------------- #
def oracle_archive_spot(res, cx):
    face_of = make_single_element_face_classifier(cx, ELEM_H)
    recs = [r for r in res.get("raySequences", [])
            if r.get("dirx", 0) > 0 and r.get("b", 0) > 0
            and signature_of(r, face_of) is not None]
    if not recs:
        return float("nan")
    ys = np.array([r["y"] for r in recs]); bs = np.array([r["b"] for r in recs])
    c = np.average(ys, weights=bs)
    return float(np.sqrt(np.average((ys - c) ** 2, weights=bs)))


def oracle_dom_signature(res, cx):
    face_of = make_single_element_face_classifier(cx, ELEM_H)
    by = aggregate_by_signature(res.get("raySequences", []), face_of)
    return (max(by, key=by.get) if by else None)


def oracle_tir_margin(res, n):
    crit = np.degrees(np.arcsin(1.0 / n)); margins = []
    for r in res.get("raySequences", []):
        if r.get("dirx", 0) <= 0:
            continue
        seq = r["seq"]
        for k, v in enumerate(seq):
            if v.get("t") != "tir" or not (0 < k < len(seq) - 1):
                continue
            pin = np.array([seq[k]["x"] - seq[k - 1]["x"], seq[k]["y"] - seq[k - 1]["y"]])
            pout = np.array([seq[k + 1]["x"] - seq[k]["x"], seq[k + 1]["y"] - seq[k]["y"]])
            if np.linalg.norm(pin) > 1e-9 and np.linalg.norm(pout) > 1e-9:
                pin /= np.linalg.norm(pin); pout /= np.linalg.norm(pout)
                ang = np.degrees(np.arccos(np.clip(np.dot(pin, pout), -1, 1)))
                margins.append((180.0 - ang) / 2.0 - crit)
    return (float(min(margins)) if margins else float("nan")), float(crit)


# --------------------------------------------------------------------------- #
#  reporting
# --------------------------------------------------------------------------- #
def make_figure(path, history, J_final, r_final, topo, opt_names, info0, infoF):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure] skipped ({e})"); return False
    its = [h["it"] for h in history]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].plot(its, [h["spot"] for h in history], "o-")
    ax[0, 0].set(title="twin dense spot", xlabel="iter", ylabel="weighted RMS spot"); ax[0, 0].grid(alpha=.3)
    ax[0, 1].plot(its, [h["thru"] for h in history], "s-", color="tab:green")
    ax[0, 1].axhline(history[0]["thru"], ls="--", color="grey", label=f"start {history[0]['thru']:.3f}")
    ax[0, 1].set(title="dominant-topology throughput", xlabel="iter"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
    ax[0, 2].plot(its, [h["margin"] for h in history], "d-", color="tab:orange")
    ax[0, 2].axhline(0, ls=":", color="red", label="critical")
    ax[0, 2].set(title="min TIR critical-angle margin (deg)", xlabel="iter"); ax[0, 2].legend(); ax[0, 2].grid(alpha=.3)
    Jn = J_final.numpy()
    sv = np.linalg.svd(Jn, compute_uv=False)
    ax[1, 0].semilogy(range(1, len(sv) + 1), sv, "o-")
    ax[1, 0].set(title="analytic-J singular spectrum", xlabel="index", ylabel="singular value"); ax[1, 0].grid(alpha=.3)
    rn = r_final.numpy()
    cancel = {}
    for j, nme in enumerate(opt_names):
        contrib = Jn[:, j] * rn
        gross = float(np.sum(np.abs(contrib))); net = abs(float(np.sum(contrib)))
        cancel[nme] = float(1 - net / gross) if gross > 1e-15 else 0.0
    ax[1, 1].barh(list(cancel.keys()), list(cancel.values()), color="tab:purple")
    ax[1, 1].set(title="per-DOF gradient cancellation\n(0=aligned, 1=rays fight)",
                 xlim=(0, 1), xlabel="1 - |sum|/sum|.|"); ax[1, 1].grid(alpha=.3, axis="x")
    ax[1, 2].hist(info0["y"] - info0["ybar"], bins=30, weights=info0["b"], alpha=.5,
                  label=f"start spot={history[0]['spot']:.3f}")
    ax[1, 2].hist(infoF["y"] - infoF["ybar"], bins=30, weights=infoF["b"], alpha=.5,
                  label=f"refined spot={history[-1]['spot']:.3f}")
    ax[1, 2].set(title="twin sensor energy vs centroid", xlabel="y - centroid"); ax[1, 2].legend(); ax[1, 2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return True


def _sensor_hist_info(topo, p_np):
    p = torch.tensor(p_np, dtype=DTYPE)
    with torch.no_grad():
        tr = trace(p, topo)
    w = topo.w0 / topo.w0.sum()
    ybar = float((w * tr.y.numpy()).sum())
    return dict(y=tr.y.numpy(), b=tr.energy.numpy(), ybar=ybar)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="3,5")
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--params", default=",".join(DEFAULT_PARAMS))
    ap.add_argument("--rays", type=int, default=121)
    ap.add_argument("--beam-w", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--optimizer", default="lm", choices=["lm", "adam"])
    ap.add_argument("--lr", type=float, default=5e-3, help="adam learning rate")
    ap.add_argument("--thru-slack", type=float, default=0.02)
    ap.add_argument("--thru-weight", type=float, default=50.0)
    ap.add_argument("--margin-floor", type=float, default=None,
                    help="abs TIR margin floor (deg); default = no-erosion (champion start - 0.25)")
    ap.add_argument("--max-step", type=float, default=0.1)
    ap.add_argument("--out", default=str(REPO / "ns_autodiff" / "results_refine_autodiff"))
    args = ap.parse_args()
    cell = tuple(int(v) for v in args.cell.split(","))
    opt_names = [s.strip() for s in args.params.split(",") if s.strip()]

    print("=" * 78)
    print("AUTODIFF REFINEMENT (differentiable twin; analytic Jacobian)")
    print("=" * 78)
    t0 = time.time()
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        topo = TOPO.extract(runner, Path(args.archive), cell,
                            n_rays=args.rays, beam_w=args.beam_w)
        thru0 = float(topo.meta["dom_energy"] / topo.meta["emitted"]) \
            if topo.meta["emitted"] else 0.0
        # twin's own dominant-throughput at champion (consistent reference for the floor)
        p0 = torch.tensor([topo.p0[n] for n in PARAM_NAMES], dtype=DTYPE)
        with torch.no_grad():
            thru0_twin = float(trace(p0, topo).energy.mean())
        obj = TwinObjective(topo, opt_names, thru0=thru0_twin,
                            slack=args.thru_slack, mu=args.thru_weight)

        diag0 = obj.diagnostics(obj.base_xn[obj.opt_idx])
        if args.margin_floor is None:
            margin_floor = diag0["margin"] - 0.25 if not np.isnan(diag0["margin"]) else -1e9
            floor_mode = f"no-erosion ({margin_floor:.2f} deg = start - 0.25)"
        else:
            margin_floor = args.margin_floor
            floor_mode = f"absolute {margin_floor:.2f} deg" + (" (disabled)" if margin_floor <= 0 else "")

        print(f"cell {cell}  topology: {topo.signature_str()}")
        print(f"tir={topo.meta['tir_count']} dom_share={topo.dom_share:.3f} "
              f"archive_spot={topo.archive_spot:.3f}  tracked={topo.meta['n_tracked']}/{args.rays}")
        print(f"refined DOF: {opt_names}")
        print(f"start (twin): spot={diag0['spot']:.4f} thru={diag0['thru']:.3f} "
              f"min-TIR-margin={diag0['margin']:.2f} deg")
        print(f"optimizer={args.optimizer}  guards: margin-floor={floor_mode}, trust={args.max_step}")
        print("-" * 78)

        if args.optimizer == "lm":
            xnF, history, J_final, r_final = lm_refine(
                obj, iters=args.iters, margin_floor=margin_floor, max_step=args.max_step)
        else:
            xnF, history, J_final, r_final = adam_refine(
                obj, iters=args.iters, margin_floor=margin_floor, lr=args.lr)

        pF = obj.params_from(xnF).detach().numpy()
        p0_np = obj.params_from(obj.base_xn[obj.opt_idx]).detach().numpy()
        info0 = _sensor_hist_info(topo, p0_np)
        infoF = _sensor_hist_info(topo, pF)

        # ---- honest re-validation through the JS oracle at 21-ray density ----
        p0_d = dict(zip(PARAM_NAMES, [float(v) for v in p0_np]))
        pF_d = dict(zip(PARAM_NAMES, [float(v) for v in pF]))
        resA, _ = runner.simulate(FD.scene(p0_d, n_rays=21, beam_w=args.beam_w, record=True))
        resB, _ = runner.simulate(FD.scene(pF_d, n_rays=21, beam_w=args.beam_w, record=True))
        spotA = oracle_archive_spot(resA, p0_d["cx"])
        spotB = oracle_archive_spot(resB, pF_d["cx"])
        domA = oracle_dom_signature(resA, p0_d["cx"])
        domB = oracle_dom_signature(resB, pF_d["cx"])
        same_topo = (domA == domB)
        marginB, crit = oracle_tir_margin(resB, pF_d["n"])

    dt = time.time() - t0
    kappas = [h["kappa"] for h in history if h.get("kappa")]
    print("-" * 78)
    print(f"RESULT  ({dt:.0f}s, optimizer={args.optimizer})")
    print(f"  twin dense spot:                 {history[0]['spot']:.4f} -> {history[-1]['spot']:.4f} "
          f"({100*(1-history[-1]['spot']/history[0]['spot']):+.1f}%)")
    print(f"  ORACLE 21-ray honest spot:       {spotA:.4f} -> {spotB:.4f} "
          f"({100*(1-spotB/spotA):+.1f}%)")
    print(f"  twin throughput:                 {history[0]['thru']:.4f} -> {history[-1]['thru']:.4f}")
    if kappas:
        print(f"  analytic kappa(J): start {kappas[0]:.1f}  median {np.median(kappas):.1f}  max {max(kappas):.1f}")
    print(f"  oracle topology preserved:       {same_topo}  ({format_signature(domB) if domB else None})")
    print(f"  oracle min TIR margin (refined): {marginB:.2f} deg (crit={crit:.2f})")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = out / f"refine_autodiff_cell{cell[0]}_{cell[1]}.json"
    rep_path, k = base, 0
    while rep_path.exists():
        k += 1; rep_path = base.with_name(f"refine_autodiff_cell{cell[0]}_{cell[1]}_{k}.json")
    fig_path = rep_path.with_suffix(".png")
    report = dict(
        method="autodiff-twin", optimizer=args.optimizer, cell=list(cell),
        topology=topo.signature_str(),
        genotype_start=p0_d, genotype_refined=pF_d,
        result=dict(
            twin_spot_start=history[0]["spot"], twin_spot_refined=history[-1]["spot"],
            oracle_spot21_start=spotA, oracle_spot21_refined=spotB,
            twin_throughput_start=history[0]["thru"], twin_throughput_refined=history[-1]["thru"],
            kappa_start=(kappas[0] if kappas else None),
            kappa_max=(max(kappas) if kappas else None),
            topology_preserved=bool(same_topo),
            dom_signature_refined=(format_signature(domB) if domB else None),
            oracle_tir_margin_refined_deg=marginB, critical_angle_deg=crit,
            margin_floor_deg=float(margin_floor)),
        config=dict(refined_dof=opt_names, rays=args.rays, iters=args.iters,
                    thru_slack=args.thru_slack, thru_weight=args.thru_weight,
                    tracked_rays=topo.meta["n_tracked"]),
        history=history, wall_seconds=dt)
    rep_path.write_text(json.dumps(report, indent=2, default=float))
    made = make_figure(fig_path, history, J_final, r_final, topo, opt_names, info0, infoF)
    print("-" * 78)
    print(f"[report] {rep_path}")
    if made:
        print(f"[figure] {fig_path}")


if __name__ == "__main__":
    main()
