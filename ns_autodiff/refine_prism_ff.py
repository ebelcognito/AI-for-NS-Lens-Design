"""Differentiable refinement of a QD-discovered AR/VR 90-degree prism (freeform faces).

Closes doc 03 open step #1 ("Prism refinement -- the missing freeform win"). Same engine class
as the fold's 20x refiner: analytic gradients (IFT through geometry_prism_ff), Levenberg-Marquardt
/ damped least squares, the guard family, honest re-score. The DOF are the 8 aspheric coefficients
(a2,a4 per face A/B/T/D); base geometry and the per-source bright-path topology are FROZEN.

Pipeline (discover -> freeze -> refine):
  1. load a cell of results_search_prism_freeform (genotype = 9 base + 8 aspheric).
  2. freeze the bright-path topology (per-ray n_events) from prism_branch's single-path follow.
  3. LM on the aspheric coeffs, minimizing per-source collimation sigma (+ throughput-floor), with
     the analytic Jacobian of the differentiable bright-path twin (forward_prism_ff).
  4. GUARDS (mandatory, doc 00 "regularization or it cheats"): throughput-floor no-erosion,
     TIR-margin no-erosion, ordering preserved, and ORDER PRESERVATION = the per-ray bright
     interaction (face,kind) signature is unchanged (the only thing doc 03's guard must protect).
  5. every accepted step is re-scored through the FAITHFUL branching tracer (prism_branch) -- the
     twin proposes, the branching tracer disposes. Report before/after at search density.

Never overwrites: writes results_refine_prism_ff/refine_cell<i>_<j>[.json] (+k suffix if taken).

Usage:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.refine_prism_ff --cell 9,18
    $PY -m ns_autodiff.refine_prism_ff --top 6 --iters 40
    $PY -m ns_autodiff.refine_prism_ff --top 8 --max-folds 3      # restrict to <=3-fold relays
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ns_probe"))

from . import prism_branch as PB
from . import forward_prism_ff as FF

torch.set_default_dtype(torch.float64)

ARCHIVE = REPO / "ns_autodiff" / "results_search_prism_freeform" / "archive_prism_freeform.json"
OUTDIR = REPO / "ns_autodiff" / "results_refine_prism_ff"

# residual weights / guards
W_T = 3.0            # throughput-shortfall weight in the residual
T_FLOOR_SLACK = 1e-3
MARGIN_SLACK = 0.05  # deg of TIR-margin erosion tolerated
TIKHONOV = 0.02      # keep coeffs near the discovered design (regularization)
COEFF_LO, COEFF_HI = -3.0, 3.0
SIGMA_ABSENT = 5.0


# --------------------------------------------------------------------------- #
#  faithful branching re-score + bright-path order signature
# --------------------------------------------------------------------------- #
def branching_score(x17):
    """Honest score through the faithful branching tracer (what search used)."""
    ex = PB.trace_branching_ff(np.asarray(x17, float))
    if ex is None:
        return None
    return PB.metrics(ex)


def bright_signature(x17, O, D):
    """Per-ray bright-path (face,kind) signature -- the frozen interaction ORDER to preserve."""
    arcs, n = PB.build_arcs_ff(np.asarray(x17, float))
    sigs = []
    for i in range(len(O)):
        Oi = O[i].copy(); Di = D[i].copy(); inside = False; seq = []
        for _ in range(PB.MAX_DEPTH):
            hit = PB.nearest_hit_ff(Oi, Di, arcs)
            if hit is None:
                break
            _t, P, normal, name = hit
            n1 = (1.0 / n) if not inside else n
            cos1 = -(Di @ normal); sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
            if sq1 < 0:
                Di = Di + 2.0 * cos1 * normal; seq.append((name, "tir"))
            else:
                cos2 = math.sqrt(sq1); Di = n1 * Di + (n1 * cos1 - cos2) * normal
                inside = not inside; seq.append((name, "refract"))
            Di = Di / np.linalg.norm(Di); Oi = P
        sigs.append(tuple(seq))
    return tuple(sigs)


def fold_count(x17):
    """TIR-bounce count of the central source's chief ray (the fold-count descriptor)."""
    arcs, n = PB.build_arcs_ff(np.asarray(x17, float))
    sx, sy = PB.SOURCE_PTS[1]
    O = np.array([sx, sy]); D = np.array([math.cos(math.radians(PB.SOURCE_AIM)),
                                          math.sin(math.radians(PB.SOURCE_AIM))])
    inside = False; ntir = 0
    for _ in range(PB.MAX_DEPTH):
        hit = PB.nearest_hit_ff(O, D, arcs)
        if hit is None:
            break
        _t, P, normal, _name = hit
        n1 = (1.0 / n) if not inside else n
        cos1 = -(D @ normal); sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        if sq1 < 0:
            D = D + 2.0 * cos1 * normal; ntir += 1
        else:
            cos2 = math.sqrt(sq1); D = n1 * D + (n1 * cos1 - cos2) * normal; inside = not inside
        D = D / np.linalg.norm(D); O = P
    return ntir


# --------------------------------------------------------------------------- #
#  LM refinement of one cell
# --------------------------------------------------------------------------- #
def refine_cell(cell, *, iters=40, n_fan=33, tr=0.4, allow_order_change=False, verbose=True):
    x0 = np.asarray(cell["genotype"], float)
    base9 = x0[:9]
    c0_np = x0[9:].copy()

    # freeze topology at search density for the order signature; refine at (denser) n_fan
    O_s, D_s, sid_s, _ = FF.freeze_topology(x0, n_fan=PB.N_FAN)
    sig0 = bright_signature(x0, O_s, D_s)
    O, D, sid, nev = FF.freeze_topology(x0, n_fan=n_fan)

    m_start = branching_score(x0)
    if m_start is None or not m_start["ordering_ok"]:
        return dict(skipped="start invalid / no ordering")
    champ_Tmin = m_start["T_min"]
    # champion bright-path TIR margin (for the no-erosion guard)
    faces0, n_idx, _ = FF.build_faces(base9, torch.tensor(c0_np))
    _, _, _, _, champ_margin = FF.trace_grouped(faces0, n_idx, O, D, nev)

    def resid(coeffs):
        faces, n_idx, _ = FF.build_faces(base9, coeffs)
        ang, en, py, rc, _margin = FF.trace_grouped(faces, n_idx, O, D, nev)
        m = FF.collimation(ang, en, py, sid, rc, n_fan=n_fan)
        parts = []
        for si in range(3):
            if m["present"][si]:
                parts.append(m["sigmas"][si])
                parts.append(W_T * torch.relu(torch.as_tensor(champ_Tmin) - m["Ts"][si]))
            else:
                parts.append(torch.as_tensor(float(SIGMA_ABSENT)))
                parts.append(torch.as_tensor(float(W_T * champ_Tmin)))
        parts.append(TIKHONOV * coeffs)                    # 8 Tikhonov terms (vector)
        return torch.cat([p.reshape(-1) for p in parts])

    def guards_and_score(coeffs_np):
        """Return (ok, branching_metrics, margin) for a candidate coeff vector."""
        x = np.concatenate([base9, coeffs_np])
        mb = branching_score(x)
        if mb is None or not mb["ordering_ok"]:
            return False, mb, None
        if mb["T_min"] < champ_Tmin - T_FLOOR_SLACK:
            return False, mb, None
        if (not allow_order_change) and bright_signature(x, O_s, D_s) != sig0:   # ORDER changed -> reject
            return False, mb, None
        faces, n_idx, _ = FF.build_faces(base9, torch.tensor(coeffs_np))
        _, _, _, _, margin = FF.trace_grouped(faces, n_idx, O, D, nev)
        if np.isfinite(champ_margin) and np.isfinite(margin) and margin < champ_margin - MARGIN_SLACK:
            return False, mb, margin
        return True, mb, margin

    coeffs = torch.tensor(c0_np, dtype=torch.float64)
    best_fit = m_start["fitness"]
    best_coeffs = c0_np.copy()
    lam = 1e-2
    n_accept = 0
    for it in range(iters):
        r0 = resid(coeffs).detach()
        J = torch.autograd.functional.jacobian(resid, coeffs, vectorize=True).detach()  # [R,8]
        H = J.T @ J
        g = J.T @ r0
        diag = torch.diag(torch.clamp(torch.diagonal(H), min=1e-9))
        accepted = False
        for _inner in range(8):
            try:
                dp = torch.linalg.solve(H + lam * diag, -g)
            except Exception:
                lam *= 4; continue
            nrm = float(dp.norm())
            if nrm > tr:
                dp = dp * (tr / nrm)
            cand = torch.clamp(coeffs + dp, COEFF_LO, COEFF_HI).numpy()
            ok, mb, _margin = guards_and_score(cand)
            if ok and mb["fitness"] > best_fit + 1e-5:
                coeffs = torch.tensor(cand, dtype=torch.float64)
                best_fit = mb["fitness"]; best_coeffs = cand
                lam = max(lam * 0.5, 1e-4); accepted = True; n_accept += 1
                if allow_order_change:           # topology may have shifted -> re-freeze n_events
                    nev = FF.freeze_topology(np.concatenate([base9, cand]), n_fan=n_fan)[3]
                break
            lam *= 3.0
        if verbose:
            print(f"    it {it:02d}  fit {best_fit:.4f}  lam {lam:.1e}  {'acc' if accepted else '---'}")
        if not accepted and lam > 1e6:
            break

    x_ref = np.concatenate([base9, best_coeffs])
    m_ref = branching_score(x_ref)
    return dict(
        i0=int(cell["i0"]), i1=int(cell["i1"]), folds=int(fold_count(x0)),
        folds_refined=int(fold_count(x_ref)), allow_order_change=bool(allow_order_change),
        n_events=sorted(set(int(v) for v in nev)), n_accept=n_accept,
        start=dict(fitness=m_start["fitness"], fov=m_start["fov"],
                   sigma_mean=m_start["sigma_mean"], T_min=m_start["T_min"]),
        refined=dict(fitness=m_ref["fitness"], fov=m_ref["fov"],
                     sigma_mean=m_ref["sigma_mean"], T_min=m_ref["T_min"]),
        delta_fit=m_ref["fitness"] - m_start["fitness"],
        sigma_improve_pct=(100.0 * (1.0 - (m_ref["sigma_mean"] or 0) / (m_start["sigma_mean"] or 1))),
        genotype_start=x0.tolist(), genotype_refined=x_ref.tolist(),
    )


def _save(report, tag):
    OUTDIR.mkdir(exist_ok=True)
    base = OUTDIR / f"refine_cell{report['i0']}_{report['i1']}{tag}.json"
    p, k = base, 1
    while p.exists():
        p = OUTDIR / f"refine_cell{report['i0']}_{report['i1']}{tag}+{k}.json"; k += 1
    p.write_text(json.dumps(report, indent=1))
    return p


def main():
    global OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=str, default=None, help="i,j of a specific archive cell")
    ap.add_argument("--top", type=int, default=0, help="refine the top-K cells by stored fitness")
    ap.add_argument("--max-folds", type=int, default=99)
    ap.add_argument("--min-folds", type=int, default=0)
    ap.add_argument("--min-fit", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--n-fan", type=int, default=33)
    ap.add_argument("--allow-order-change", action="store_true",
                    help="accept order-changing steps (re-freeze topology) -- needed for multi-fold")
    ap.add_argument("--archive", default=str(ARCHIVE), help="search archive JSON to refine from")
    ap.add_argument("--outdir", default=str(OUTDIR), help="where to write refine_cell*.json")
    args = ap.parse_args()

    OUTDIR = Path(args.outdir)
    cells = json.loads(Path(args.archive).read_text())["cells"]
    by_key = {(c["i0"], c["i1"]): c for c in cells}

    if args.cell:
        i, j = (int(v) for v in args.cell.split(","))
        targets = [by_key[(i, j)]]
    else:
        ranked = sorted([c for c in cells if c["fitness"] >= args.min_fit],
                        key=lambda c: -c["fitness"])
        targets = []
        for c in ranked:
            f = fold_count(np.asarray(c["genotype"], float))
            if args.min_folds <= f <= args.max_folds:
                targets.append(c)
            if len(targets) >= max(args.top, 1):
                break

    print(f"Refining {len(targets)} cell(s)  (n_fan={args.n_fan}, iters={args.iters})")
    print("=" * 92)
    rows = []
    for c in targets:
        f = fold_count(np.asarray(c["genotype"], float))
        print(f"cell {c['i0']},{c['i1']}  folds={f}  stored_fit={c['fitness']:.4f}")
        rep = refine_cell(c, iters=args.iters, n_fan=args.n_fan,
                          allow_order_change=args.allow_order_change)
        if "skipped" in rep:
            print(f"  SKIP: {rep['skipped']}"); continue
        path = _save(rep, "")
        s, r = rep["start"], rep["refined"]
        print(f"  fit {s['fitness']:.4f} -> {r['fitness']:.4f}  (d{rep['delta_fit']:+.4f})   "
              f"sigma {s['sigma_mean']:.4f} -> {r['sigma_mean']:.4f}  "
              f"({rep['sigma_improve_pct']:+.1f}%)   Tmin {s['T_min']:.3f} -> {r['T_min']:.3f}   "
              f"[{rep['n_accept']} steps]  -> {path.name}")
        rows.append(rep)

    if rows:
        print("=" * 92)
        avg_sig = np.mean([r["sigma_improve_pct"] for r in rows])
        avg_df = np.mean([r["delta_fit"] for r in rows])
        print(f"SUMMARY  {len(rows)} cells   mean sigma improvement {avg_sig:+.1f}%   "
              f"mean delta-fitness {avg_df:+.4f}")


if __name__ == "__main__":
    main()
