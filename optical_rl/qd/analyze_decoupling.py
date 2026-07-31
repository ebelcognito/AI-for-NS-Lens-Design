#!/usr/bin/env python3
"""analyze_decoupling.py

Quantify descriptor <-> geometry decoupling in a QD archive.

This is a format-agnostic diagnostic: point it at *any* QD run directory
produced by the optical_rl pipeline (singlet, doublet, triplet, ef50 zernike,
...) and it measures how much geometric variation among performant elites
lives in directions the behavioral descriptor cannot see.

The central QD claim you usually want to support is:

    "Among designs that meet a performance floor, tiny differences in the
     behavioral descriptor can correspond to large differences in the
     underlying geometric parameters."

The script supports that claim with three independent tests, then prints one
headline number per test so you can paste them into a paper.

------------------------------------------------------------------------------
HOW IT WORKS
------------------------------------------------------------------------------

Inputs it expects in the run directory:
  * One archive file, either:
      - archive_cells.json   (dict with keys "archive_bounds", "cells", ...)
      - archive.json         (flat list of elite dicts)
    Each elite must provide the fields:
        theta:    list[float]   (geometric parameters, dim D)
        measures: list[float]   (behavioral descriptor, dim M)
        fitness:  float         (convention: fitness = -rms_spot_mm)
  * Exactly one *.yaml config carrying:
        qd.theta_bounds:    list of [lo, hi] per theta dim
        qd.archive_bounds:  list of [lo, hi] per measure dim, or null
    If archive_bounds is null/missing we fall back to the archive file's own
    "archive_bounds" field, and failing that to the empirical min/max of the
    observed measures.

Pipeline:
  1. Load the archive, normalize its two different on-disk layouts into a
     uniform (theta, measures, rms) triple.
  2. Filter to elites that meet a performance floor (default: rms <= 2x best;
     override with --rms-max or --top-frac).
  3. Rescale theta and measures into unit boxes using theta_bounds and
     measure_bounds, so distances across the two spaces are comparable and
     reproducible across runs.
  4. Run three tests on the filtered, normalized elites:

     (1) Pairwise d_theta vs d_measure scatter.
         For every pair (i, j) of surviving elites compute normalized
         Euclidean distances in descriptor space and in theta space. Scatter
         them. Report Pearson / Spearman correlation and, as the headline,
         an "epsilon-ball" statistic: among pairs whose d_measure lies in the
         tightest 1% of the distribution, how large does d_theta get? If
         d_theta in that ball reaches a large fraction of the global d_theta
         range, then two designs with essentially the same descriptor can be
         nearly as geometrically different as any two designs in the archive.

     (2) k-NN Jaccard mismatch.
         For each elite, take its k nearest neighbors under d_measure and
         under d_theta. Compute Jaccard overlap of the two sets, per elite,
         and report the mean / fraction with zero overlap. Low overlap means
         "descriptor-nearby" and "geometry-nearby" are nearly independent
         neighborhood structures.

     (3) Linear null-space variance.
         Fit OLS  measures_centered = theta_centered @ W  (W is D x M). The
         map theta -> measures has a (D - rank(W))-dimensional null space;
         project the centered theta samples onto that null space and measure
         the fraction of theta variance captured there. A high fraction means
         the archive has populated many directions in theta that any linear
         readout of the descriptor cannot see - i.e. geometric diversity that
         is literally invisible to the chosen descriptor.

Outputs (in <run-dir>/decoupling/ unless --out-dir is given):
  decoupling_scatter.png       - pairwise distance scatter
  decoupling_knn.png           - Jaccard overlap histogram
  decoupling_summary.json      - all numbers + top-20 most decoupled pairs
  A single headline block is also printed to stdout.

Usage examples:
    python -m optical_rl.qd.analyze_decoupling \
        --run-dir results/models/ef50_zernike_20260410_180300 --rms-max 1.0
    python -m optical_rl.qd.analyze_decoupling \
        --run-dir results/models/triplet_cmamega_20260328_192052 --top-frac 0.2
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Loading - format-agnostic
# ---------------------------------------------------------------------------

def _find_archive_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "archive_cells.json",
        run_dir / "archive.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # last-chance glob
    hits = sorted(run_dir.glob("archive*.json"))
    if hits:
        return hits[0]
    raise SystemExit(
        f"No archive JSON found in {run_dir}. Expected archive_cells.json or archive.json."
    )


def _find_config_file(run_dir: Path) -> Path:
    hits = sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
    if not hits:
        raise SystemExit(f"No *.yaml config in {run_dir}; needed for theta_bounds.")
    return hits[0]


def load_archive(run_dir: Path):
    """Return (elites_list, archive_bounds_or_None).

    elites_list: list[dict] where each dict has at least theta, measures, fitness.
    archive_bounds: (M, 2) array or None if the archive file did not store them.
    """
    path = _find_archive_file(run_dir)
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        # doublet / triplet style: flat list of elites
        return data, None

    if isinstance(data, dict):
        if "cells" in data:
            bounds = data.get("archive_bounds")
            bounds = np.asarray(bounds, dtype=np.float64) if bounds else None
            return data["cells"], bounds
        if "elites" in data:
            bounds = data.get("archive_bounds")
            bounds = np.asarray(bounds, dtype=np.float64) if bounds else None
            return data["elites"], bounds
        # sometimes a dict keyed by cell index -> elite
        values = list(data.values())
        if values and isinstance(values[0], dict) and "theta" in values[0]:
            return values, None

    raise SystemExit(f"Unrecognized archive format in {path}")


def load_config(run_dir: Path) -> dict:
    with open(_find_config_file(run_dir)) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("qd", cfg)


def stack_elites(elites: list):
    theta = np.asarray([e["theta"] for e in elites], dtype=np.float64)
    meas = np.asarray([e["measures"] for e in elites], dtype=np.float64)
    fit = np.asarray([e["fitness"] for e in elites], dtype=np.float64)
    return theta, meas, fit


def resolve_bounds(theta: np.ndarray,
                   meas: np.ndarray,
                   qd: dict,
                   archive_bounds_from_file):
    """Return (theta_bounds, measure_bounds) as (D,2) and (M,2) arrays.

    Priority for theta_bounds: qd.theta_bounds -> empirical min/max.
    Priority for measure_bounds: qd.archive_bounds -> archive file -> empirical.
    """
    tb = qd.get("theta_bounds")
    if tb is None:
        theta_bounds = np.stack([theta.min(0), theta.max(0)], axis=1)
    else:
        theta_bounds = np.asarray(tb, dtype=np.float64)
        if theta_bounds.shape != (theta.shape[1], 2):
            raise SystemExit(
                f"theta_bounds shape {theta_bounds.shape} != ({theta.shape[1]}, 2)"
            )

    mb = qd.get("archive_bounds")
    if mb is not None:
        measure_bounds = np.asarray(mb, dtype=np.float64)
    elif archive_bounds_from_file is not None:
        measure_bounds = archive_bounds_from_file
    else:
        pad = 1e-9
        measure_bounds = np.stack(
            [meas.min(0) - pad, meas.max(0) + pad], axis=1
        )

    if measure_bounds.shape != (meas.shape[1], 2):
        raise SystemExit(
            f"measure_bounds shape {measure_bounds.shape} != ({meas.shape[1]}, 2)"
        )
    return theta_bounds, measure_bounds


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def _pdist_normed(x: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance, divided by sqrt(dim) so distances across
    spaces of different dimensionality are comparable on a [0, 1]-ish scale."""
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt((diff * diff).sum(-1))
    return d / np.sqrt(x.shape[1])


def pairwise_decoupling(meas_n: np.ndarray, theta_n: np.ndarray):
    """Test 1: pairwise d_theta vs d_measure + epsilon-ball punchline."""
    n = meas_n.shape[0]
    dM = _pdist_normed(meas_n)
    dT = _pdist_normed(theta_n)
    iu = np.triu_indices(n, k=1)
    dM_p = dM[iu]
    dT_p = dT[iu]

    r_p = float(pearsonr(dM_p, dT_p)[0])
    r_s = float(spearmanr(dM_p, dT_p)[0])

    eps = float(np.quantile(dM_p, 0.01))
    in_ball = dM_p <= eps
    stats = {
        "eps_quantile": 0.01,
        "eps_value_normalized": eps,
        "n_pairs_in_ball": int(in_ball.sum()),
        "d_theta_global_median": float(np.median(dT_p)),
        "d_theta_global_p99": float(np.quantile(dT_p, 0.99)),
    }
    if in_ball.any():
        b = dT_p[in_ball]
        stats.update({
            "d_theta_in_ball_median": float(np.median(b)),
            "d_theta_in_ball_p99": float(np.quantile(b, 0.99)),
            "d_theta_in_ball_max": float(b.max()),
        })
    return dM, dT, dM_p, dT_p, iu, r_p, r_s, stats


def knn_mismatch(dM: np.ndarray, dT: np.ndarray, k: int):
    """Test 2: k-NN Jaccard overlap between descriptor and theta neighborhoods."""
    n = dM.shape[0]
    k = min(k, n - 1)
    nn_m = np.argsort(dM, axis=1)[:, 1:k + 1]
    nn_t = np.argsort(dT, axis=1)[:, 1:k + 1]
    j = np.empty(n, dtype=np.float64)
    for i in range(n):
        a = set(nn_m[i].tolist())
        b = set(nn_t[i].tolist())
        u = a | b
        j[i] = len(a & b) / len(u) if u else 0.0
    return k, j


def linear_nullspace(theta_n: np.ndarray, meas_n: np.ndarray):
    """Test 3: R^2 of theta -> measures and fraction of theta variance
    lying in the null space of that linear map."""
    Xc = theta_n - theta_n.mean(0)
    Yc = meas_n - meas_n.mean(0)
    W, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)           # (D, M)
    Yhat = Xc @ W
    ss_res = ((Yc - Yhat) ** 2).sum(0)
    ss_tot = (Yc * Yc).sum(0) + 1e-30
    r2_per_dim = 1.0 - ss_res / ss_tot
    r2_joint = float(1.0 - ss_res.sum() / ss_tot.sum())

    # Map v -> v @ W has null space = {v : W^T v = 0} = left null space of W.
    U, sv, _ = np.linalg.svd(W, full_matrices=True)
    rank = int((sv > 1e-10).sum())
    null_basis = U[:, rank:]
    range_basis = U[:, :rank]
    var_null = float((Xc @ null_basis).var(axis=0, ddof=0).sum())
    var_range = float((Xc @ range_basis).var(axis=0, ddof=0).sum())
    var_total = var_null + var_range + 1e-30
    return {
        "r2_per_dim": r2_per_dim.tolist(),
        "r2_joint": r2_joint,
        "W_rank": rank,
        "W_singular_values": sv.tolist(),
        "theta_variance_in_nullspace": var_null,
        "theta_variance_in_range": var_range,
        "theta_variance_in_nullspace_frac": float(var_null / var_total),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scatter(dM_p, dT_p, eps, n, floor_mode, r_p, r_s, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(dM_p, dT_p, s=2, alpha=0.15, rasterized=True)
    ax.axvline(eps, color="red", lw=0.8, ls="--",
               label=f"1% quantile d_measure = {eps:.3g}")
    ax.set_xlabel("normalized d_measure")
    ax.set_ylabel("normalized d_theta")
    ax.set_title(f"Pairwise distances  N={n}  floor={floor_mode}\n"
                 f"Pearson r={r_p:+.3f}   Spearman r={r_s:+.3f}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_knn(jaccards, k, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    n = len(jaccards)
    ax.hist(jaccards, bins=min(40, max(5, n // 4)),
            color="#4c72b0", edgecolor="white")
    ax.axvline(float(jaccards.mean()), color="red", ls="--",
               label=f"mean = {jaccards.mean():.3f}")
    ax.set_xlabel(f"Jaccard overlap of k={k} nearest neighbors")
    ax.set_ylabel("count")
    ax.set_title("Descriptor-NN vs Geometry-NN agreement\n"
                 "(0 = independent neighborhoods, 1 = identical)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level analysis function (callable from other scripts, not just CLI)
# ---------------------------------------------------------------------------

def analyze_run(run_dir: Path,
                rms_max: float = None,
                top_frac: float = None,
                k: int = 5,
                out_dir: Path = None) -> dict:
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else run_dir / "decoupling"
    out_dir.mkdir(parents=True, exist_ok=True)

    elites, archive_bounds_file = load_archive(run_dir)
    qd = load_config(run_dir)
    theta, meas, fitness = stack_elites(elites)
    rms = -fitness

    theta_bounds, measure_bounds = resolve_bounds(theta, meas, qd, archive_bounds_file)

    # ---- performance floor -----------------------------------------------
    if top_frac is not None:
        keep = max(2, int(round(len(elites) * top_frac)))
        idx = np.argsort(rms)[:keep]
        mask = np.zeros(len(elites), dtype=bool)
        mask[idx] = True
        rms_max_effective = float(rms[mask].max())
        floor_mode = f"top_frac={top_frac}"
    else:
        best = float(rms.min())
        rms_max_effective = rms_max if rms_max is not None else 2.0 * best
        mask = rms <= rms_max_effective
        floor_mode = f"rms_max={rms_max_effective:.4f}mm"

    N = int(mask.sum())
    if N < 10:
        raise SystemExit(f"Too few elites after filter (N={N}); relax the floor.")

    theta_f = theta[mask]
    meas_f = meas[mask]
    rms_f = rms[mask]
    print(f"[filter] {floor_mode}: kept {N}/{len(elites)} elites  "
          f"(best={rms_f.min():.4f}, worst={rms_f.max():.4f})")

    # ---- normalize --------------------------------------------------------
    theta_range = theta_bounds[:, 1] - theta_bounds[:, 0]
    measure_range = measure_bounds[:, 1] - measure_bounds[:, 0]
    theta_n = (theta_f - theta_bounds[:, 0]) / theta_range
    meas_n = (meas_f - measure_bounds[:, 0]) / measure_range

    # ---- three tests ------------------------------------------------------
    dM, dT, dM_p, dT_p, iu, r_p, r_s, eps_stats = pairwise_decoupling(meas_n, theta_n)
    k_eff, jaccards = knn_mismatch(dM, dT, k)
    linear_stats = linear_nullspace(theta_n, meas_n)

    # ---- plots ------------------------------------------------------------
    plot_scatter(dM_p, dT_p, eps_stats["eps_value_normalized"], N, floor_mode,
                 r_p, r_s, out_dir / "decoupling_scatter.png")
    plot_knn(jaccards, k_eff, out_dir / "decoupling_knn.png")

    # ---- top decoupled pairs ---------------------------------------------
    score = dT_p / (dM_p + 1e-6)
    top_idx = np.argsort(score)[::-1][:20]
    filt_global = np.where(mask)[0]
    top_pairs = []
    for t in top_idx:
        a = iu[0][t]
        b = iu[1][t]
        top_pairs.append({
            "elite_i": int(filt_global[a]),
            "elite_j": int(filt_global[b]),
            "measures_i": meas_f[a].tolist(),
            "measures_j": meas_f[b].tolist(),
            "d_measure_normalized": float(dM[a, b]),
            "d_theta_normalized": float(dT[a, b]),
            "rms_i_mm": float(rms_f[a]),
            "rms_j_mm": float(rms_f[b]),
            "theta_i": theta_f[a].tolist(),
            "theta_j": theta_f[b].tolist(),
        })

    # ---- summary ----------------------------------------------------------
    summary = {
        "run_dir": str(run_dir),
        "n_total": int(len(elites)),
        "n_filtered": N,
        "floor_mode": floor_mode,
        "rms_max_mm": float(rms_max_effective),
        "best_rms_mm": float(rms.min()),
        "theta_dim": int(theta.shape[1]),
        "descriptor_dim": int(meas.shape[1]),
        "theta_bounds": theta_bounds.tolist(),
        "measure_bounds": measure_bounds.tolist(),
        "pearson_r_d_measure_vs_d_theta": r_p,
        "spearman_r_d_measure_vs_d_theta": r_s,
        "epsilon_ball": eps_stats,
        "knn": {
            "k": k_eff,
            "jaccard_mean": float(jaccards.mean()),
            "jaccard_median": float(np.median(jaccards)),
            "jaccard_min": float(jaccards.min()),
            "jaccard_max": float(jaccards.max()),
            "fraction_zero_overlap": float((jaccards == 0).mean()),
        },
        "linear_decoupling": linear_stats,
        "top_decoupled_pairs": top_pairs,
    }
    with open(out_dir / "decoupling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_headline(summary)
    print(f"\n  outputs in: {out_dir}")
    return summary


def _print_headline(s: dict):
    print()
    print("=" * 68)
    print("  DECOUPLING SUMMARY")
    print("=" * 68)
    print(f"  theta_dim x descriptor_dim  : {s['theta_dim']} x {s['descriptor_dim']}")
    print(f"  elites under floor          : {s['n_filtered']} / {s['n_total']}")
    print(f"  rms_spot floor              : {s['rms_max_mm']:.4f} mm  "
          f"(best={s['best_rms_mm']:.4f})")
    print(f"  Pearson  r(d_m, d_theta)    : {s['pearson_r_d_measure_vs_d_theta']:+.3f}")
    print(f"  Spearman r(d_m, d_theta)    : {s['spearman_r_d_measure_vs_d_theta']:+.3f}")
    knn = s["knn"]
    print(f"  k-NN Jaccard (k={knn['k']})         : mean={knn['jaccard_mean']:.3f}  "
          f"median={knn['jaccard_median']:.3f}  "
          f"zero-overlap={knn['fraction_zero_overlap']:.3f}")
    lin = s["linear_decoupling"]
    print(f"  linear R^2 theta->measures  : joint={lin['r2_joint']:+.3f}  "
          f"per-dim={[round(x, 3) for x in lin['r2_per_dim']]}")
    print(f"  theta variance in NULL(W)   : "
          f"{lin['theta_variance_in_nullspace_frac'] * 100:5.1f}%")
    eps = s["epsilon_ball"]
    if eps.get("n_pairs_in_ball", 0) > 0:
        print()
        print(f"  epsilon-ball (d_m <= {eps['eps_value_normalized']:.3g}, 1% quantile):")
        print(f"    pairs in ball             : {eps['n_pairs_in_ball']}")
        print(f"    max    d_theta in ball    : {eps['d_theta_in_ball_max']:.4f}")
        print(f"    p99    d_theta in ball    : {eps['d_theta_in_ball_p99']:.4f}")
        print(f"    median d_theta in ball    : {eps['d_theta_in_ball_median']:.4f}")
        print(f"    global median d_theta     : {eps['d_theta_global_median']:.4f}")
        print(f"    global p99    d_theta     : {eps['d_theta_global_p99']:.4f}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", required=True,
                   help="Directory containing an archive JSON and run yaml.")
    p.add_argument("--rms-max", type=float, default=None,
                   help="Performance floor: keep elites with rms_spot <= this "
                        "value (mm). Default: 2x the best rms in the archive.")
    p.add_argument("--top-frac", type=float, default=None,
                   help="Alternative to --rms-max: keep top fraction by fitness.")
    p.add_argument("--k", type=int, default=5,
                   help="k for k-NN Jaccard overlap (default 5).")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <run-dir>/decoupling).")
    return p.parse_args()


def main():
    args = parse_args()
    analyze_run(
        run_dir=Path(args.run_dir),
        rms_max=args.rms_max,
        top_frac=args.top_frac,
        k=args.k,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
