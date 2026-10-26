"""Recipe table + figure versus the GA runs (re-scored with the same in-spec rule and objective) -> RESULTS.json, PNG."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

SURF, INK, INK2, MUTED, GRID, BLUE = '#fcfcfb', '#0b0b0b', '#52514e', '#8f8d86', '#e7e6e2', '#2a78d6'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
GA = R.ROOT/'experiments/ga_prism_replication_20261026/runs'


def ga_rows(mode):
    rows = []
    for p in sorted((GA/mode).glob('run_*.json')):
        d = json.loads(p.read_text())
        x = np.asarray(d['final']['x'])
        Rm, spec = R.residuals(((x - R.MID)/R.HALF)[None])
        rms = R.M.trace(x[None])['rms'][0]
        rows.append(dict(in_spec=bool(spec[0]), score=float(Rm[0] @ Rm[0]), center_rms=float(rms[R.M.CENTER_FIELD]),
                         worst_rms=float(rms.max()), mean_rms=float(rms.mean()), seconds=d['seconds']))
    return rows


def summarize(rows, budget):
    ok = [r for r in rows if r['in_spec']]
    med = lambda k: float(np.median([r[k] for r in ok]))*1e3 if ok else None
    return dict(budget=budget, runs=len(rows), in_spec=len(ok), median_center_um=med('center_rms'),
                median_worst_um=med('worst_rms'), median_mean_um=med('mean_rms'),
                best_worst_um=float(min(r['worst_rms'] for r in ok))*1e3 if ok else None,
                median_score=float(np.median([r['score'] if r['in_spec'] else r['best_score'] for r in rows]))
                if 'best_score' in rows[0] else float(np.median([r['score'] for r in ok])),
                median_seconds=float(np.median([r['seconds'] for r in rows])))


def main():
    runs = [json.loads(p.read_text()) for p in sorted((HERE/'runs').glob('*.json'))]
    for r in runs:
        r['score'] = r['best_score']
    table = {rec: summarize([r for r in runs if r['recipe'] == rec], R.BUDGET) for rec in R.RECIPES}
    table['GA two-stage (paper, 50 gen)'] = summarize(ga_rows('two50'), 19000)
    table['GA Stage-2 only (50 gen)'] = summarize(ga_rows('only50'), 10000)
    table['GA two-stage (paper, 300 gen)'] = summarize(ga_rows('full'), 69000)
    curves = {rec: {c: float(np.median([r['log'].get(str(c), np.inf) for r in runs if r['recipe'] == rec]))
                    for c in R.CHECKPOINTS} for rec in R.RECIPES}
    (HERE/'RESULTS.json').write_text(json.dumps(dict(table=table, median_best_score_at=curves), indent=1))
    order = sorted(table, key=lambda k: (table[k]['median_worst_um'] is None, table[k]['median_worst_um'] or 0))
    print(f"{'method':32s} {'budget':>7s} {'in spec':>8s} {'center':>7s} {'worst':>7s} {'mean':>7s} {'best worst':>10s} {'sec':>5s}")
    for k in order:
        t = table[k]
        f = lambda v: f'{v:7.1f}' if v is not None else '      -'
        print(f"{k:32s} {t['budget']:7d} {t['in_spec']:3d}/{t['runs']:<4d} {f(t['median_center_um'])} {f(t['median_worst_um'])} "
              f"{f(t['median_mean_um'])} {f(t['best_worst_um']):>10s} {t['median_seconds']:5.0f}")
    fig(runs, order)


def fig(runs, order):
    data = {}
    for k in order:
        if k in R.RECIPES:
            data[k] = [r['worst_rms']*1e3 for r in runs if r['recipe'] == k and r['in_spec']]
    for mode, k in (('two50', 'GA two-stage (paper, 50 gen)'), ('only50', 'GA Stage-2 only (50 gen)'),
                    ('full', 'GA two-stage (paper, 300 gen)')):
        data[k] = [r['worst_rms']*1e3 for r in ga_rows(mode) if r['in_spec']]
    keys = [k for k in order if data.get(k)]
    fig, ax = plt.subplots(figsize=(7.5, .38*len(keys) + 1.2))
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        v = np.asarray(data[k])
        ax.plot(v, i + rng.uniform(-.15, .15, len(v)), 'o', ms=4, color=BLUE, alpha=.55, mec=SURF, mew=.6)
        ax.plot([np.median(v)]*2, [i - .3, i + .3], color=INK, lw=2)
    ax.set_yticks(range(len(keys)), [f"{k}  ({len(data[k])})" for k in keys]); ax.invert_yaxis()
    ax.set_xscale('log'); ax.grid(axis='y', visible=False)
    ax.set_xlabel('worst-field RMS spot radius, in-spec designs (µm, log scale)  ·  bar = median  ·  (n) = runs in spec')
    ax.set_title('Optimizer recipes versus the two-stage GA, same prism and tracer', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(HERE/'recipes_vs_ga.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
