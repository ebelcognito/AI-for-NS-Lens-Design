"""Fairness check: the GA's best worst-field in-spec design anywhere in its final population (not its center-RMS pick)."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
sys.path.insert(0, str(R.ROOT/'experiments/ga_prism_replication_20261026'))
import run_ga as G  # noqa: E402


def job(args):
    mode, i = args
    rng = np.random.default_rng(1000*('full', 'two50').index(mode) + i)
    p1, _ = G.stage1(rng, i)
    res = G.stage2(rng, i, p1, 300 if mode == 'full' else 50)
    X = res.pop.get('X')
    _, spec = R.residuals((X - R.MID)/R.HALF)
    rms = R.M.trace(X)['rms']
    if not spec.any():
        return dict(mode=mode, run=i, in_spec=False)
    k = np.flatnonzero(spec)[np.argmin(rms[spec].max(1))]
    return dict(mode=mode, run=i, in_spec=True, worst_um=float(rms[k].max()*1e3), center_um=float(rms[k, 4]*1e3))


if __name__ == '__main__':
    with Pool(18) as pool:
        rows = pool.map(job, [(m, i) for m in ('two50', 'full') for i in range(18)])
    (HERE/'ga_best_worst.json').write_text(json.dumps(rows, indent=1))
    for m in ('two50', 'full'):
        ok = [r for r in rows if r['mode'] == m and r['in_spec']]
        print(m, f"{len(ok)}/18 in spec; best-worst-field member: median worst {np.median([r['worst_um'] for r in ok]):.1f} µm, "
              f"min {min(r['worst_um'] for r in ok):.1f}, its center {np.median([r['center_um'] for r in ok]):.1f} µm")
