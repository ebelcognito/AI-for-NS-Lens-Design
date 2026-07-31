"""
GATE EXPERIMENT: finite-difference Jacobian + Levenberg-Marquardt (damped least
squares) local refinement of a QD-discovered non-sequential fold.

WHY THIS EXISTS (the decision it informs). Before investing in a differentiable
NS tracer, we want to know three things about a QD champion, measured with the
*existing* JS engine as the forward oracle (no new tracer, no autodiff):

  (1) HEADROOM   - does local, gradient-style refinement actually shrink the spot,
                   or is the QD champion already at a local optimum?
  (2) CONDITIONING - is the per-ray sensitivity Jacobian ill-conditioned? This is
                   the quantitative version of the "do the gradients from a surface's
                   multiple (transmit-then-revisit) interactions FIGHT?" worry. We
                   report kappa(J)=sigma_max/sigma_min and a per-parameter
                   sign-cancellation ratio that *is* "how much do the rays fight".
  (3) TOPOLOGY STABILITY - as we step, do tracked rays keep following the champion's
                   dominant interaction sequence (stay in the archive cell / keep
                   their TIR events past critical), or does the topology break up?

METHOD. We freeze the topology by tracking only the input rays that reach the
sensor via the champion's DOMINANT typed interaction signature, identified by their
beam offset dy (invariant to the lens DOF). The objective is the energy-weighted RMS
spot of those tracked rays (frozen per-ray weights, floating best-focus centroid),
plus a one-sided throughput-floor penalty so refinement cannot cheat by dumping
energy out of the fold. The residual Jacobian is built by CENTRAL FINITE DIFFERENCES
in bound-normalized parameter coordinates; LM solves (J^T J + lambda I) d = -J^T r
with the standard lambda up/down acceptance rule. Everything is traced through the
honest Fresnel+TIR JS engine; nothing here is differentiated analytically.

We deliberately DENSIFY the bundle (refine density >> QD's 21 rays): refinement
traces only a handful of designs, so we can afford a much denser, sharper objective
than the coarse QD fitness. Before/after we ALSO report the archive-comparable spot
(all forward rays at QD's 21-ray density) for an apples-to-apples number, and we
re-validate that every TIR event in the final design still sits past the critical
angle.

USAGE
    python3 ns_probe/refine_lm.py                       # default champion (3-TIR, spot 2.42)
    python3 ns_probe/refine_lm.py --pick cleanest       # max dominant-signature share
    python3 ns_probe/refine_lm.py --cell 3,7            # a specific archive cell (i0,i1)
    python3 ns_probe/refine_lm.py --rays 161 --iters 60 # denser bundle / longer run
    python3 ns_probe/refine_lm.py --params cx,thick,fsag,bsag,tilt_a,tilt_b  # which DOF to move

Outputs a never-overwritten JSON report (the geometry sidecar) and a diagnostics
figure under ns_probe/results_refine_lm/.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
import fold_demo as FD
from sequence_tools import (aggregate_by_signature, format_signature,
                            make_single_element_face_classifier, signature_of)

ELEM_H = FD.ELEM_H
X_SENSOR = FD.X_SENSOR
PARAM_NAMES = FD.PARAM_NAMES          # [ray_y, ang, cx, thick, fsag, bsag, tilt_a, tilt_b, n]
LO, HI = FD.LO.copy(), FD.HI.copy()

# Element-shape DOF refined by default; the input beam (ray_y, ang) is the FIELD we
# design for and is frozen, so "better focus" means a better lens, not a moved field.
DEFAULT_PARAMS = ["cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b", "n"]


# --------------------------------------------------------------------------- #
#  champion selection
# --------------------------------------------------------------------------- #
def load_champion(archive_path: Path, *, pick: str, rank: int, cell):
    arc = json.loads(archive_path.read_text())
    cells = arc["cells"]
    if cell is not None:
        i0, i1 = cell
        hit = [c for c in cells if int(c["i0"]) == i0 and int(c["i1"]) == i1]
        if not hit:
            raise SystemExit(f"no archive cell ({i0},{i1})")
        c = hit[0]
    else:
        folds = [c for c in cells if c["tir_count"] >= 2]
        if pick == "cleanest":
            folds.sort(key=lambda c: -c["dom_energy_share"])
            c = folds[0]
        elif pick == "rank":
            folds.sort(key=lambda c: -c["fitness"])
            c = folds[min(rank, len(folds) - 1)]
        else:  # "headroom" (default): clean topology AND a spot worth shrinking
            cand = [c for c in folds if c["dom_energy_share"] >= 0.85]
            cand.sort(key=lambda c: -(c["spot"] if c["spot"] < 8.0 else 0.0))
            c = cand[0] if cand else sorted(folds, key=lambda c: -c["fitness"])[0]
    p = dict(zip(PARAM_NAMES, [float(v) for v in c["genotype"]]))
    return p, c


# --------------------------------------------------------------------------- #
#  tracing / topology tracking
# --------------------------------------------------------------------------- #
def trace(runner, p, *, n_rays, beam_w):
    res, _ = runner.simulate(FD.scene(p, n_rays=n_rays, beam_w=beam_w, record=True))
    return res


def dom_records_by_dy(res, p, dom_sig):
    """For each input ray (keyed by beam offset dy = source_y - ray_y, which is
    invariant to the lens DOF), the (sensor_y, energy) of the highest-energy
    forward record whose typed signature equals the champion's dominant one."""
    face_of = make_single_element_face_classifier(p["cx"], ELEM_H)
    out = {}
    for r in res.get("raySequences", []):
        if r.get("dirx", 0) <= 0 or r.get("b", 0) <= 0:
            continue
        if signature_of(r, face_of) != dom_sig:
            continue
        dy = round(float(r["seq"][0]["y"]) - p["ray_y"], 6)
        if dy not in out or r["b"] > out[dy][1]:
            out[dy] = (float(r["y"]), float(r["b"]))
    return out


def dominant_signature(res, p):
    face_of = make_single_element_face_classifier(p["cx"], ELEM_H)
    by_sig = aggregate_by_signature(res.get("raySequences", []), face_of)
    if not by_sig:
        return None, 0.0
    dom = max(by_sig, key=by_sig.get)
    return dom, by_sig[dom]


def archive_style_spot(res):
    """The exact spot metric run_fold_qd records: energy-weighted RMS of ALL
    forward, first-crossing sensor rays about their energy centroid (so before/after
    is comparable to the archive number)."""
    recs = [r for r in res.get("raySequences", [])
            if r.get("dirx", 0) > 0 and r.get("b", 0) > 0
            and signature_of(r, make_single_element_face_classifier(
                _last_cx[0], ELEM_H)) is not None]
    if not recs:
        return float("nan")
    ys = np.array([r["y"] for r in recs]); bs = np.array([r["b"] for r in recs])
    c = np.average(ys, weights=bs)
    return float(np.sqrt(np.average((ys - c) ** 2, weights=bs)))


_last_cx = [0.0]   # set per-call so archive_style_spot uses the right face split


def critical_angle_margins(res, p):
    """Min margin (deg) of any TIR event past the critical angle, over forward
    dominant-ish rays. >0 everywhere => every fold is real TIR."""
    n = p["n"]
    crit = np.degrees(np.arcsin(1.0 / n))
    margins = []
    for r in res.get("raySequences", []):
        if r.get("dirx", 0) <= 0:
            continue
        seq = r["seq"]
        for k, v in enumerate(seq):
            if v.get("t") != "tir":
                continue
            # incidence at this vertex: angle between incoming dir and local geometry
            # is not stored, but TIR labeling already guarantees incidence>crit; we
            # recover incidence from the turn if neighbors exist.
            if 0 < k < len(seq):
                pin = np.array([seq[k]["x"] - seq[k - 1]["x"], seq[k]["y"] - seq[k - 1]["y"]])
                if k + 1 < len(seq):
                    pout = np.array([seq[k + 1]["x"] - seq[k]["x"], seq[k + 1]["y"] - seq[k]["y"]])
                    if np.linalg.norm(pin) > 1e-9 and np.linalg.norm(pout) > 1e-9:
                        pin /= np.linalg.norm(pin); pout /= np.linalg.norm(pout)
                        # specular turn => incidence = (pi - angle(pin,pout))/2
                        ang = np.degrees(np.arccos(np.clip(np.dot(pin, pout), -1, 1)))
                        inc = (180.0 - ang) / 2.0
                        margins.append(inc - crit)
    return (float(min(margins)) if margins else float("nan"),
            float(crit), len(margins))


# --------------------------------------------------------------------------- #
#  residuals (frozen-topology, frozen-weight spot + throughput-floor penalty)
# --------------------------------------------------------------------------- #
class Objective:
    def __init__(self, runner, p0, dom_sig, tracked_dys, w0, *, n_rays, beam_w,
                 opt_idx, thru0, thru_slack, thru_weight):
        self.runner = runner
        self.p0 = p0
        self.dom_sig = dom_sig
        self.dys = tracked_dys
        self.m = len(tracked_dys)
        self.w0 = w0 / w0.sum()                  # frozen per-ray weights, sum 1
        self.n_rays, self.beam_w = n_rays, beam_w
        self.opt_idx = opt_idx
        self.thru0 = thru0
        self.floor = thru0 * (1.0 - thru_slack)
        self.mu = thru_weight
        self.dy_tol = (beam_w / max(n_rays - 1, 1)) * 0.5 + 1e-6

    def unpack(self, xn_full):
        return dict(zip(PARAM_NAMES, LO + np.clip(xn_full, 0, 1) * (HI - LO)))

    def evaluate(self, xn_full):
        """Return per-ray residuals (NaN for rays that dropped the dominant topology),
        plus spot/throughput/centroid diagnostics. None if geometry invalid/errors."""
        p = self.unpack(xn_full)
        if not FD.geometry_valid(p):
            return None
        try:
            res = trace(self.runner, p, n_rays=self.n_rays, beam_w=self.beam_w)
        except Exception:
            return None
        if res.get("error"):
            return None
        led = res.get("energyLedger") or {}
        emitted = float(led.get("emitted", 0.0))
        if emitted <= 0:
            return None
        recs = dom_records_by_dy(res, p, self.dom_sig)
        ys = np.full(self.m, np.nan)
        bs = np.zeros(self.m)
        for k, dy in enumerate(self.dys):
            best = None
            for d, (y, b) in recs.items():
                if abs(d - dy) <= self.dy_tol and (best is None or b > best[1]):
                    best = (y, b)
            if best is not None:
                ys[k], bs[k] = best
        present = ~np.isnan(ys)
        wpr = self.w0 * present
        if wpr.sum() <= 0:
            return None
        ybar = float(np.nansum(wpr * np.nan_to_num(ys)) / wpr.sum())
        r_spot = np.where(present, np.sqrt(self.w0) * (np.nan_to_num(ys) - ybar), np.nan)
        thru = float(bs.sum() / emitted)
        r_thru = np.sqrt(self.mu) * max(0.0, self.floor - thru)
        r = np.concatenate([r_spot, [r_thru]])
        spot = float(np.sqrt(np.sum(wpr * (np.nan_to_num(ys) - ybar) ** 2) / wpr.sum()))
        margin, _crit, _n = critical_angle_margins(res, p)
        return dict(r=r, ys=ys, bs=bs, present=present, ybar=ybar, thru=thru,
                    spot=spot, emitted=emitted, n_dropped=int((~present).sum()),
                    min_margin=margin, p=p, res=res)

    def fill(self, info, center_r):
        """Replace NaN residuals (dropped rays) with the current-iterate values so the
        residual vector keeps a fixed dimension for J and ||r||."""
        r = info["r"].copy()
        nan = np.isnan(r)
        r[nan] = center_r[nan]
        return r, int(nan.sum())

    def cost(self, r):
        return float(np.dot(r, r))


# --------------------------------------------------------------------------- #
#  finite-difference Jacobian + Levenberg-Marquardt
# --------------------------------------------------------------------------- #
def fd_jacobian(obj, xn, center_r, h):
    """Central-difference residual Jacobian over the optimized DOF only.
    Returns J (m+1 x n_opt) and the number of dropped-ray substitutions used."""
    nopt = len(obj.opt_idx)
    J = np.zeros((len(center_r), nopt))
    drops = 0
    for j, idx in enumerate(obj.opt_idx):
        xp = xn.copy(); xp[idx] += h
        xm = xn.copy(); xm[idx] -= h
        ip, im = obj.evaluate(xp), obj.evaluate(xm)
        if ip is None and im is None:
            continue                                  # column stays 0; DOF locally dead
        if ip is None:
            rp = center_r
        else:
            rp, d = obj.fill(ip, center_r); drops += d
        if im is None:
            rm = center_r
        else:
            rm, d = obj.fill(im, center_r); drops += d
        J[:, j] = (rp - rm) / (2.0 * h)
    return J, drops


def lm_optimize(obj, xn0, *, iters, h, lam0=1e-2, margin_floor=0.0, max_step=0.1,
                verbose=True):
    """Damped least squares with two guards that keep refinement physical and in-cell:
    (a) a TIR critical-angle MARGIN FLOOR (reject any step that pushes a fold below it),
    (b) a TRUST REGION cap on the normalized step norm (don't jump out of the cell)."""
    xn = xn0.copy()
    info = obj.evaluate(xn)
    if info is None:
        raise SystemExit("champion geometry evaluated as invalid -- cannot start")
    center_r, _ = obj.fill(info, np.zeros_like(info["r"]))
    cost = obj.cost(center_r)
    lam = lam0
    history = [dict(it=0, spot=info["spot"], thru=info["thru"], cost=cost, lam=lam,
                    dropped=info["n_dropped"], fd_subs=0, margin=info["min_margin"],
                    kappa=None, step=0.0)]
    last_J = None
    for it in range(1, iters + 1):
        J, jdrops = fd_jacobian(obj, xn, center_r, h)
        last_J = J
        g = J.T @ center_r                            # gradient of 1/2||r||^2
        H = J.T @ J
        sv = np.linalg.svd(J, compute_uv=False)
        kappa = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")
        accepted, rej_margin = False, 0
        step = np.zeros(len(obj.opt_idx))
        for _inner in range(14):
            try:
                step = np.linalg.solve(H + lam * np.eye(H.shape[0]), -g)
            except np.linalg.LinAlgError:
                lam *= 4.0; continue
            sn = float(np.linalg.norm(step))          # trust region
            if sn > max_step:
                step *= max_step / sn
            xn_try = xn.copy()
            xn_try[obj.opt_idx] = np.clip(xn[obj.opt_idx] + step, 0.0, 1.0)
            trial = obj.evaluate(xn_try)
            if trial is None:
                lam = min(lam * 4.0, 1e6); continue
            if trial["min_margin"] < margin_floor:    # physics guard
                rej_margin += 1
                lam = min(lam * 4.0, 1e6); continue
            r_try, _ = obj.fill(trial, center_r)
            c_try = obj.cost(r_try)
            if c_try < cost:
                xn, info, center_r, cost = xn_try, trial, r_try, c_try
                lam = max(lam * 0.5, 1e-9)
                accepted = True
                break
            lam = min(lam * 4.0, 1e6)
        history.append(dict(it=it, spot=info["spot"], thru=info["thru"], cost=cost,
                            lam=lam, dropped=info["n_dropped"], fd_subs=jdrops,
                            margin=info["min_margin"], kappa=kappa,
                            step=float(np.linalg.norm(step)), rej_margin=rej_margin))
        if verbose:
            print(f"  [lm {it:2d}] spot={info['spot']:.4f} thru={info['thru']:.3f} "
                  f"margin={info['min_margin']:5.2f} kappa(J)={kappa:10.1f} "
                  f"lam={lam:.1e} fd_subs={jdrops:3d} "
                  f"{'acc' if accepted else ('REJm' if rej_margin else 'STALL')}",
                  flush=True)
        if not accepted and lam >= 1e6:
            if verbose:
                print("  [lm] converged / stalled.")
            break
    return xn, info, history, last_J, center_r


def cancellation_ratio(J, r, opt_names):
    """Per-parameter sign-cancellation in the gradient g_j = sum_i J_ij r_i.
    1 - |sum| / sum|.| in [0,1]: 0 = all rays pull the same way, ~1 = the rays'
    contributions nearly cancel (the quantitative 'do the gradients fight?')."""
    out = {}
    for j, name in enumerate(opt_names):
        contrib = J[:, j] * r
        gross = float(np.sum(np.abs(contrib)))
        net = abs(float(np.sum(contrib)))
        out[name] = float(1.0 - net / gross) if gross > 1e-15 else 0.0
    return out


# --------------------------------------------------------------------------- #
#  reporting
# --------------------------------------------------------------------------- #
def make_figure(path, history, J_final, r_final, info0, infoF, opt_names, cancel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure] skipped ({e})")
        return False
    its = [h["it"] for h in history]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    ax[0, 0].plot(its, [h["spot"] for h in history], "o-")
    ax[0, 0].set(title="objective spot (dominant-topology, dense)",
                 xlabel="LM iter", ylabel="weighted RMS spot")
    ax[0, 0].grid(alpha=.3)

    ax[0, 1].plot(its, [h["thru"] for h in history], "s-", color="tab:green")
    ax[0, 1].axhline(info0["thru"], ls="--", color="grey",
                     label=f"start {info0['thru']:.3f}")
    ax[0, 1].set(title="dominant-topology throughput", xlabel="LM iter",
                 ylabel="energy fraction"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

    ax[0, 2].plot(its, [h["margin"] for h in history], "d-", color="tab:orange")
    ax[0, 2].axhline(0, ls=":", color="red", label="critical angle")
    ax[0, 2].set(title="min TIR critical-angle margin (physics guard)",
                 xlabel="LM iter", ylabel="margin past critical (deg)")
    ax[0, 2].legend(); ax[0, 2].grid(alpha=.3)

    sv = np.linalg.svd(J_final, compute_uv=False)
    ax[1, 0].semilogy(range(1, len(sv) + 1), sv, "o-")
    ax[1, 0].set(title="final-J singular spectrum", xlabel="index",
                 ylabel="singular value (log)"); ax[1, 0].grid(alpha=.3)

    names = list(cancel.keys()); vals = [cancel[n] for n in names]
    ax[1, 1].barh(names, vals, color="tab:purple")
    ax[1, 1].set(title="per-DOF gradient cancellation\n(0=aligned, 1=rays fight)",
                 xlabel="1 - |sum|/sum|.|", xlim=(0, 1)); ax[1, 1].grid(alpha=.3, axis="x")

    y0 = info0["ys"][info0["present"]]; b0 = info0["bs"][info0["present"]]
    yF = infoF["ys"][infoF["present"]]; bF = infoF["bs"][infoF["present"]]
    ax[1, 2].hist(y0 - info0["ybar"], bins=30, weights=b0, alpha=.5,
                  label=f"start spot={info0['spot']:.3f}")
    ax[1, 2].hist(yF - infoF["ybar"], bins=30, weights=bF, alpha=.5,
                  label=f"refined spot={infoF['spot']:.3f}")
    ax[1, 2].set(title="sensor energy vs centroid", xlabel="y - centroid",
                 ylabel="energy"); ax[1, 2].legend(); ax[1, 2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default=str(REPO / "ns_probe" / "results_fold_qd" / "archive.json"))
    ap.add_argument("--pick", default="headroom", choices=["headroom", "cleanest", "rank"])
    ap.add_argument("--rank", type=int, default=0, help="for --pick rank: i-th fold by fitness")
    ap.add_argument("--cell", default=None, help="archive cell as i0,i1 (overrides --pick)")
    ap.add_argument("--params", default=",".join(DEFAULT_PARAMS),
                    help="comma list of DOF to refine (subset of %s)" % PARAM_NAMES)
    ap.add_argument("--rays", type=int, default=121, help="refinement bundle density")
    ap.add_argument("--beam-w", type=float, default=FD.__dict__.get("BEAM_W", 8.0))
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--fd-step", type=float, default=2e-3, help="central-diff step (normalized)")
    ap.add_argument("--thru-slack", type=float, default=0.02,
                    help="allowed fractional throughput drop before penalty bites")
    ap.add_argument("--thru-weight", type=float, default=50.0)
    ap.add_argument("--margin-floor", type=float, default=None,
                    help="absolute TIR critical-angle margin floor in deg (0 disables the "
                         "guard). Default: 'no-erosion' = the champion's own start margin, "
                         "so the refined design is at least as robust as the champion.")
    ap.add_argument("--max-step", type=float, default=0.1,
                    help="trust-region cap on the normalized LM step norm")
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_refine_lm"))
    args = ap.parse_args()

    cell = tuple(int(v) for v in args.cell.split(",")) if args.cell else None
    p0, champ = load_champion(Path(args.archive), pick=args.pick, rank=args.rank, cell=cell)
    opt_names = [s.strip() for s in args.params.split(",") if s.strip()]
    bad = [n for n in opt_names if n not in PARAM_NAMES]
    if bad:
        raise SystemExit(f"unknown DOF {bad}; choose from {PARAM_NAMES}")
    opt_idx = np.array([PARAM_NAMES.index(n) for n in opt_names])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    beam_w = 8.0

    print("=" * 78)
    print("FD-JACOBIAN + LEVENBERG-MARQUARDT REFINEMENT GATE")
    print("=" * 78)
    print(f"champion cell ({champ['i0']},{champ['i1']}): "
          f"tir={int(champ['tir_count'])} T={champ['T']:.1f} "
          f"fit={champ['fitness']:.3f} eff={champ['efficiency']:.3f} "
          f"archive-spot={champ['spot']:.3f} dom-share={champ['dom_energy_share']:.3f}")
    print(f"dominant signature: {champ['dom_signature']}")
    print(f"refined DOF: {opt_names}   (frozen: {[n for n in PARAM_NAMES if n not in opt_names]})")
    print(f"bundle: {args.rays} rays (QD used 21)   LM iters: {args.iters}")
    print("-" * 78)

    t0 = time.time()
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        # establish the frozen topology + tracked rays at the champion
        res0 = trace(runner, p0, n_rays=args.rays, beam_w=beam_w)
        dom_sig, dom_e = dominant_signature(res0, p0)
        recs0 = dom_records_by_dy(res0, p0, dom_sig)
        if not recs0:
            raise SystemExit("no rays follow a dominant signature at refine density")
        tracked_dys = sorted(recs0.keys())
        w0 = np.array([recs0[dy][1] for dy in tracked_dys])
        print(f"tracked {len(tracked_dys)}/{args.rays} rays on dominant topology "
              f"{format_signature(dom_sig)}  ({100*len(tracked_dys)/args.rays:.0f}% of bundle)")

        obj = Objective(runner, p0, dom_sig, tracked_dys, w0,
                        n_rays=args.rays, beam_w=beam_w, opt_idx=opt_idx,
                        thru0=float(dom_e / res0["energyLedger"]["emitted"]),
                        thru_slack=args.thru_slack, thru_weight=args.thru_weight)

        xn0 = (np.array([p0[n] for n in PARAM_NAMES]) - LO) / (HI - LO)
        info0 = obj.evaluate(xn0)
        # "no-erosion" default: never let the refined design be less robust than the
        # champion. A small numerical slack avoids stalling on the start point itself.
        if args.margin_floor is None:
            margin_floor = info0["min_margin"] - 0.25
            floor_mode = f"no-erosion ({margin_floor:.2f} deg = start - 0.25)"
        else:
            margin_floor = args.margin_floor
            floor_mode = f"absolute {margin_floor:.2f} deg" + (" (disabled)" if margin_floor <= 0 else "")
        print(f"start: spot={info0['spot']:.4f} thru={info0['thru']:.3f} "
              f"min-TIR-margin={info0['min_margin']:.2f} deg")
        print(f"guards: margin-floor={floor_mode}, trust-step={args.max_step}")
        print("-" * 78)

        xnF, infoF, history, J_final, r_final = lm_optimize(
            obj, xn0, iters=args.iters, h=args.fd_step,
            margin_floor=margin_floor, max_step=args.max_step)

        cancel = cancellation_ratio(J_final, r_final, opt_names)
        pF = obj.unpack(xnF)

        # ---- validation: re-trace start & end at QD's 21-ray density ----
        _last_cx[0] = p0["cx"]
        resA = trace(runner, p0, n_rays=21, beam_w=beam_w)
        spotA = archive_style_spot(resA)
        _last_cx[0] = pF["cx"]
        resB = trace(runner, pF, n_rays=21, beam_w=beam_w)
        spotB = archive_style_spot(resB)
        domF, domFe = dominant_signature(resB, pF)
        same_topo = (domF == dom_sig)
        margin, crit, n_tir = critical_angle_margins(resB, pF)

    dt = time.time() - t0
    print("-" * 78)
    print(f"RESULT  ({dt:.0f}s)")
    print(f"  objective spot (dense, dominant-topology): {info0['spot']:.4f} -> "
          f"{infoF['spot']:.4f}   ({100*(1-infoF['spot']/info0['spot']):+.1f}%)")
    print(f"  archive-comparable spot (21 rays, all fwd): {spotA:.4f} -> {spotB:.4f}")
    print(f"  dominant-topology throughput:               {info0['thru']:.4f} -> {infoF['thru']:.4f}")
    kappas = [h['kappa'] for h in history if h['kappa']]
    print(f"  Jacobian conditioning kappa(J):             "
          f"start {kappas[0]:.1f}  median {np.median(kappas):.1f}  max {max(kappas):.1f}")
    print(f"  topology preserved after refine:            {same_topo} "
          f"({format_signature(domF)})")
    print(f"  min TIR critical-angle margin:              {info0['min_margin']:.2f} -> "
          f"{margin:.2f} deg (crit={crit:.2f}, {n_tir} TIR events)")
    print(f"  worst gradient cancellation:                "
          + ", ".join(f"{k}={v:.2f}" for k, v in
                      sorted(cancel.items(), key=lambda kv: -kv[1])[:3]))

    # ---- write report (never overwrite; JSON is the geometry sidecar) ----
    tag = f"cell{champ['i0']}_{champ['i1']}"
    base = out / f"refine_{tag}.json"
    k = 0
    rep_path = base
    while rep_path.exists():
        k += 1; rep_path = base.with_name(f"refine_{tag}_{k}.json")
    fig_path = rep_path.with_suffix(".png")

    report = dict(
        champion=dict(cell=[int(champ["i0"]), int(champ["i1"])],
                      genotype_start={n: p0[n] for n in PARAM_NAMES},
                      genotype_refined={n: float(pF[n]) for n in PARAM_NAMES},
                      dom_signature=format_signature(dom_sig),
                      archive_spot=champ["spot"], dom_share=champ["dom_energy_share"]),
        config=dict(refined_dof=opt_names, refine_rays=args.rays, beam_w=beam_w,
                    iters=args.iters, fd_step=args.fd_step,
                    thru_slack=args.thru_slack, thru_weight=args.thru_weight,
                    tracked_rays=len(tracked_dys)),
        result=dict(
            spot_dense_start=info0["spot"], spot_dense_refined=infoF["spot"],
            spot_dense_improvement_pct=100 * (1 - infoF["spot"] / info0["spot"]),
            spot_21ray_start=spotA, spot_21ray_refined=spotB,
            throughput_start=info0["thru"], throughput_refined=infoF["thru"],
            kappa_start=kappas[0], kappa_median=float(np.median(kappas)),
            kappa_max=float(max(kappas)),
            topology_preserved=bool(same_topo),
            dom_signature_refined=format_signature(domF),
            tir_margin_start_deg=info0["min_margin"], tir_margin_refined_deg=margin,
            critical_angle_deg=crit, n_tir_events=n_tir, margin_floor_deg=margin_floor,
            gradient_cancellation=cancel),
        history=history, wall_seconds=dt)
    rep_path.write_text(json.dumps(report, indent=2, default=float))

    made = make_figure(fig_path, history, J_final, r_final, info0, infoF, opt_names, cancel)
    print("-" * 78)
    print(f"[report] {rep_path}")
    if made:
        print(f"[figure] {fig_path}")


if __name__ == "__main__":
    main()
