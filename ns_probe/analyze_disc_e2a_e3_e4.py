"""
E2a / E3 / E4 analysis (§3, §4, §5) — manifold measure, basin structure, novelty reachability.

Uses the 50k random run (results_disc/random50k) as the ground-truth map of the good manifold,
and the E1 archives (cmame/random seed*) for the matched-budget comparisons.

  E2a  random-hit rate            : fraction of uniform draws that are valid / folded / good / high
  E3   basin census + reach       : DBSCAN basins of high-eff folds; does A@14k reach basins B0@14k
                                     misses (esp. deep tir>=4)?
  E4   novelty reachability audit : distance from each A high-eff elite to nearest random draw;
                                     fraction with NO random neighbor within eps=0.2

    PYTHONPATH=.:ns_probe python ns_probe/analyze_disc_e2a_e3_e4.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import fold_demo as FD

REPO = Path(__file__).resolve().parents[1]
DISC = REPO / "ns_probe" / "results_disc"
NS_TIR = 2
HIGH = 0.9
EPS = 0.2                      # normalized-param radius (the HyperLens metric)
SCALE = (FD.HI - FD.LO)


def norm(g):
    """Normalize genotype params to [0,1] by the genotype bounds."""
    return (np.asarray(g, float) - FD.LO) / SCALE


def load(p):
    return json.loads(Path(p).read_text())


def high_folds(good_pool):
    """High-eff NS folds (foc>=0.9, tir>=2) from a good_pool, as normalized genotypes + tir."""
    G, tirs, sigs = [], [], []
    for r in good_pool:
        if r["tir_count"] >= NS_TIR and r["fitness"] >= HIGH:
            G.append(norm(r["genotype"])); tirs.append(int(r["tir_count"]))
            sigs.append(r["dom_signature"])
    return (np.array(G) if G else np.empty((0, len(FD.LO)))), np.array(tirs), sigs


# ----------------------------------------------------------------------------------
def e2a(rand50k):
    hr = rand50k["hit_rate"]
    print("=== E2a — random-hit rate (the manifold's measure), N =", hr["n_evals"], "===")
    print(f"  valid geometries      : {hr['n_valid']:6d}  ({hr['frac_valid']*100:5.2f}% of draws)")
    print(f"  folded (tir>=2)        : {hr['n_fold']:6d}  ({hr['frac_fold_of_valid']*100:5.2f}% of valid)")
    print(f"  GOOD folds (foc>=0.5)  : {hr['n_good_fold']:6d}  ({hr['frac_good_fold_of_valid']*100:5.3f}% of valid)")
    print(f"  HIGH folds (foc>=0.9)  : {hr['n_high_fold']:6d}  ({hr['frac_high_fold_of_valid']*100:5.3f}% of valid)")
    frac_good_all = hr["n_good_fold"] / max(hr["n_evals"], 1)
    print(f"  => good folds are {frac_good_all*100:.4f}% of ALL draws "
          f"({'FAT target -> multi-start should win' if frac_good_all > 0.01 else 'rare needle -> efficient sampler can earn its keep'})")
    return hr


# ----------------------------------------------------------------------------------
def e3(rand50k):
    print("\n=== E3 — basin census (DBSCAN eps=%.2f on normalized params) ===" % EPS)
    Gr, _, _ = high_folds(rand50k["good_pool"])
    print(f"  high-eff NS folds in 50k RANDOM pool: {len(Gr)}  "
          f"{'(random cannot even reach the high-eff manifold)' if len(Gr) == 0 else ''}")
    # The high-eff manifold is populated ~only by the optimizer, so build the ground-truth
    # basin map from ALL high-eff folds seen by ANY arm (A seeds + B0 seeds + 50k random).
    pool = list(rand50k["good_pool"])
    for sub in ("cmame", "random", "restart_ls"):
        for sd in sorted((DISC / sub).glob("seed*/archive.json")):
            pool += load(sd)["good_pool"]
    G, tirs, _ = high_folds(pool)
    print(f"  high-eff NS folds pooled over ALL arms: {len(G)}")
    if len(G) == 0:
        print("  (none anywhere — no high-eff folds to census)")
        return None
    labels = DBSCAN(eps=EPS, min_samples=2).fit_predict(G)
    basins = {}          # basin_id -> dict(centroid, tir_mode, size)
    for lab in sorted(set(labels)):
        mask = labels == lab
        if lab == -1:
            # each noise point is its own singleton basin
            for gi in np.where(mask)[0]:
                bid = f"s{gi}"
                basins[bid] = dict(centroid=G[gi], tir=int(tirs[gi]), size=1)
            continue
        basins[lab] = dict(centroid=G[mask].mean(0),
                           tir=int(np.bincount(tirs[mask]).argmax()), size=int(mask.sum()))
    per_tir = {}
    for b in basins.values():
        per_tir.setdefault(b["tir"], 0)
        per_tir[b["tir"]] += 1
    print(f"  distinct basins: {len(basins)} total")
    for t in sorted(per_tir):
        print(f"    tir={t}: {per_tir[t]} basins")

    # reach: A@14k vs B0@14k basins (assign each matched-budget elite to nearest census basin)
    cents = np.array([b["centroid"] for b in basins.values()])
    bt = np.array([b["tir"] for b in basins.values()])
    nn = NearestNeighbors(n_neighbors=1).fit(cents)

    def reached_basins(good_pool):
        G2, _, _ = high_folds(good_pool)
        if len(G2) == 0:
            return set()
        dist, idx = nn.kneighbors(G2)
        return {int(i) for d, i in zip(dist[:, 0], idx[:, 0]) if d <= EPS}

    a_all, b0_all = set(), set()
    for seed_dir in sorted((DISC / "cmame").glob("seed*")):
        a_all |= reached_basins(load(seed_dir / "archive.json")["good_pool"])
    for seed_dir in sorted((DISC / "random").glob("seed*")):
        b0_all |= reached_basins(load(seed_dir / "archive.json")["good_pool"])
    a_only = a_all - b0_all
    print(f"\n  basins reached at matched 14k budget (union over seeds):")
    print(f"    A (CMA-ME) : {len(a_all)}   B0 (random) : {len(b0_all)}")
    print(f"    A-only basins (A reaches, B0@14k never enters): {len(a_only)}")
    deep_a_only = [bid for bid in a_only if bt[bid] >= 4]
    print(f"    of which DEEP (tir>=4): {len(deep_a_only)}  "
          f"{'<- persistence evidence (A reaches deep basins B0 misses)' if deep_a_only else '(none — no deep-basin advantage)'}")

    # persistence ablation, if the no-persistence run exists
    np_archives = sorted((DISC / "cmame_noPersist").glob("seed*/archive.json"))
    if np_archives:
        npr = set()
        for ap in np_archives:
            npr |= reached_basins(load(ap)["good_pool"])
        deep_np = [bid for bid in npr if bt[bid] >= 4]
        deep_a = [bid for bid in a_all if bt[bid] >= 4]
        print(f"\n  persistence ablation (A vs A-no-persistence), deep-basin (tir>=4) coverage:")
        print(f"    A deep basins: {len(deep_a)}   A-no-persist deep basins: {len(deep_np)}   "
              f"{'-> persistence IS the active ingredient' if len(deep_a) > len(deep_np) else '-> persistence NOT load-bearing'}")
    else:
        print("\n  (persistence ablation run cmame_noPersist not present — skip E3 item 3)")
    return basins


# ----------------------------------------------------------------------------------
def e4(rand50k):
    print("\n=== E4 — novelty reachability audit (eps=%.2f) ===" % EPS)
    # reachable set = ALL valid random draws (strongest test), fallback to good_pool
    vg_path = DISC / "random50k" / "valid_genotypes.npy"
    if vg_path.exists():
        R = (np.load(vg_path) - FD.LO) / SCALE
        ref_name = f"all {len(R)} valid random draws"
    else:
        R = np.array([norm(r["genotype"]) for r in rand50k["good_pool"]])
        ref_name = f"{len(R)} good random folds"
    nn = NearestNeighbors(n_neighbors=1).fit(R)
    print(f"  reachable reference set: {ref_name}")

    # A elites = high-eff folds pooled over cmame seeds
    A = []
    for seed_dir in sorted((DISC / "cmame").glob("seed*")):
        G, _, _ = high_folds(load(seed_dir / "archive.json")["good_pool"])
        A.extend(G.tolist())
    A = np.array(A)
    if len(A) == 0:
        print("  (no A high-eff folds — nothing to audit)")
        return
    dist, _ = nn.kneighbors(A)
    d = dist[:, 0]
    far = d > EPS
    print(f"  A high-eff folds audited: {len(A)}")
    print(f"  distance to nearest random draw: min={d.min():.3f} median={np.median(d):.3f} "
          f"max={d.max():.3f}")
    print(f"  fraction with NO random neighbor within eps={EPS}: {far.mean()*100:.1f}%  "
          f"({far.sum()}/{len(A)})")
    if far.mean() > 0.1:
        print("  => a meaningful fraction of A's folds are eps-far from every random draw: "
              "the search reaches genuinely hard-to-sample regions (Q4: novel).")
    else:
        print("  => ~every A fold has a close random neighbor: A polishes points random sampling "
              "already lands near (Q4: not novel).")


def main():
    r50 = load(DISC / "random50k" / "archive.json")
    e2a(r50)
    e3(r50)
    e4(r50)


if __name__ == "__main__":
    main()
