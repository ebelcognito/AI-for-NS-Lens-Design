"""Tables and figures for the replication (paper's Table 2, Figs. 7-10) -> RESULTS.json and PNGs."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prism as M  # noqa: E402

SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8f8d86', '#e7e6e2'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})


def load(mode):
    return [json.loads(p.read_text()) for p in sorted((HERE/'runs'/mode).glob('run_*.json'))]


def iqr_keep(v):
    q1, q3 = np.percentile(v, [25, 75])
    return (v >= q1 - 1.5*(q3 - q1)) & (v <= q3 + 1.5*(q3 - q1))


def table2():
    rows = {}
    for mode, label in (('only50', 'Stage-2 only'), ('two50', 'Two stages')):
        runs = load(mode)
        ok = [r for r in runs if r['final']]
        c = np.array([r['final']['center_rms'] for r in ok])
        w = np.array([r['final']['worst_rms'] for r in ok])
        keep = iqr_keep(c)
        rows[label] = dict(runs=len(runs), no_feasible_design=len(runs) - len(ok), kept=int(keep.sum()),
                           removed=int((~keep).sum()), median_center=float(np.median(c[keep])),
                           median_worst=float(np.median(w[keep])),
                           median_seconds=float(np.median([r['seconds'] for r in runs])))
    return rows


def batch():
    runs = [r for r in load('full') if r['final']]
    c = np.array([r['final']['center_rms'] for r in runs])
    keep = iqr_keep(c)
    kept = [r for r, k in zip(runs, keep) if k]
    ck = c[keep]
    w = np.array([r['final']['worst_rms'] for r in kept])
    q1, med, q3 = np.percentile(ck, [25, 50, 75])
    return kept, dict(runs=len(load('full')), feasible=len(runs), kept=len(kept), removed=int((~keep).sum()),
                      min_center=float(ck.min()), Q1=float(q1), median_center=float(med), Q3=float(q3),
                      max_center_kept=float(ck.max()), max_center_removed=float(c[~keep].max()) if (~keep).any() else None,
                      min_worst=float(w.min()), median_worst=float(np.median(w)),
                      median_W=float(np.median([r['final']['W'] for r in kept])),
                      min_seconds=float(min(r['seconds'] for r in kept)),
                      median_seconds=float(np.median([r['seconds'] for r in kept])))


def fig8(kept, s):
    c = np.sort([r['final']['center_rms'] for r in kept])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(np.arange(1, len(c) + 1), c*1e3, 'o', ms=5, color=BLUE, mec=SURF, mew=1)
    for q, name in ((s['Q1'], 'Q1'), (s['median_center'], 'median'), (s['Q3'], 'Q3')):
        ax.axhline(q*1e3, color=MUTED, lw=1, ls='--')
        ax.text(len(c) + .8, q*1e3 + .6, f'{name} {q*1e3:.1f}', va='bottom', fontsize=8, color=INK2)
    ax.set(xlabel='run (sorted)', ylabel='center-field RMS spot radius (µm)', xlim=(0, len(c) + 7))
    ax.set_title(f'Center-field RMS, IQR-filtered runs (n = {len(c)})', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(HERE/'fig8_center_rms.png', dpi=160); plt.close(fig)


def fig9(kept):
    cnt = Counter(r['sensitivity']['top'] for r in kept)
    names, vals = zip(*cnt.most_common())
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(range(len(vals)), vals, color=BLUE, height=.6)
    ax.set_yticks(range(len(vals)), names); ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + .2, i, str(v), va='center', fontsize=8, color=INK2)
    ax.set(xlabel=f'runs where it is the most sensitive parameter (of {len(kept)})')
    ax.grid(axis='y', visible=False)
    ax.set_title('Most-sensitive parameter frequency', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(HERE/'fig9_sensitivity_frequency.png', dpi=160); plt.close(fig)
    return dict(cnt)


def pareto2(a, b):
    return np.array([not np.any((a <= a[i]) & (b <= b[i]) & ((a < a[i]) | (b < b[i]))) for i in range(len(a))])


def fig10(kept):
    c = np.array([r['final']['center_rms'] for r in kept])*1e3
    s = np.array([r['sensitivity']['score'] for r in kept])*1e3
    pe = pareto2(c, s)
    # Balanced pick: the Pareto run closest to the utopia point after min-max normalisation.
    cn, sn = (c - c.min())/np.ptp(c), (s - s.min())/np.ptp(s)
    pick = np.flatnonzero(pe)[np.argmin(np.hypot(cn, sn)[pe])]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(s[~pe], c[~pe], 'o', ms=5, color=BLUE, alpha=.55, mec=SURF, mew=1, label='run')
    ax.plot(s[pe], c[pe], 's', ms=7, color=BLUE, mec=SURF, mew=1, label='Pareto-efficient')
    ax.plot(s[pick], c[pick], 's', ms=11, mfc='none', mec=INK, mew=1.2)
    ax.annotate('selected', (s[pick], c[pick]), xytext=(8, 8), textcoords='offset points', fontsize=8, color=INK2)
    ax.set(xlabel='sensitivity score: mean |Δ center RMS| under ±0.5% perturbation (µm)',
           ylabel='center-field RMS (µm)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Center RMS versus sensitivity', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(HERE/'fig10_rms_vs_sensitivity.png', dpi=160); plt.close(fig)
    return kept[pick]


def surface_curve(surf, pts):
    """Meridional (local y = 0) section of a surface over the local-z span its rays use, padded 4 mm."""
    pl = M.to_local(pts[None], surf.v, surf.R)[0]
    z = pl[:, 2][np.isfinite(pl[:, 2])]
    zz = np.linspace(z.min() - 4, z.max() + 4, 200)[None]
    s, *_ = M.sag(surf.c, surf.K, surf.a, 0*zz, zz, clamp=True)
    loc = np.stack([s, 0*zz, zz], -1)
    return surf.v[:, None, :] + M.dir_to_global(loc, surf.R)


def fig7(run):
    x = np.array(run['final']['x'])
    X, g, A, B, C, img = M.build(x[None])
    r = M.trace(X)
    P = r['points']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    meridional = [k for k in range(81) if abs(M.RAY_D[k, 1]) < 1e-12 and abs(M.RAY_O[k, 1]) < 1e-12]
    for k in meridional:
        seg = np.array([M.RAY_O[k], P['A1'][0, k], P['B'][0, k], P['A2'][0, k], P['C'][0, k], P['I'][0, k]])
        ax.plot(seg[:, 0], seg[:, 2], color=MUTED, lw=.6, alpha=.8)
    allA = np.vstack([P['A1'][0], P['A2'][0]])
    for surf, pts, col, name in ((A, allA, BLUE, 'A'), (B, P['B'][0], ORANGE, 'B'), (C, P['C'][0], AQUA, 'C')):
        cv = surface_curve(surf, pts)[0]
        ax.plot(cv[:, 0], cv[:, 2], color=col, lw=2)
        ax.text(cv[-1, 0], cv[-1, 2], f' {name}', color=INK, fontsize=10, fontweight='bold')
    pI = P['I'][0]
    vv = (pI - img['p0'][0]) @ img['v'][0]
    line = img['p0'][0] + np.outer(np.linspace(np.nanmin(vv) - 3, np.nanmax(vv) + 3, 2), img['v'][0])
    ax.plot(line[:, 0], line[:, 2], color=INK2, lw=3)
    ax.annotate('display', line[0, [0, 2]], xytext=(0, -14), textcoords='offset points', color=INK2, fontsize=9)
    ax.plot([M.PUPIL_X]*2, [-M.EYEBOX/2, M.EYEBOX/2], color=INK, lw=3)
    ax.text(M.PUPIL_X + 1, -5, 'eye box', color=INK2, fontsize=9)
    ax.set_aspect('equal'); ax.set(xlabel='X (mm)', ylabel='Z (mm)')
    ax.set_title('Layout, meridional section (fold plane)', loc='left', fontsize=10)
    u, v = r['u'][0].reshape(9, 9), r['v'][0].reshape(9, 9)
    for k in range(9):
        ax2.plot(v[k], u[k], 'o', ms=3, mfc='none', mec=BLUE, mew=.8)
        ax2.text(np.nanmean(v[k]), np.nanmean(u[k]) + .9, f'RMS {r["rms"][0, k]*1e3:.1f} µm', ha='center',
                 fontsize=7.5, color=INK2)
    ax2.margins(y=.12)
    ax2.set_aspect('equal'); ax2.set(xlabel='image v, in fold plane (mm)', ylabel='image u, horizontal (mm)')
    ax2.set_title('Spot diagram, 9 fields', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(HERE/'fig7_selected_design.png', dpi=160); plt.close(fig)


def main():
    t2 = table2()
    kept, s = batch()
    fig8(kept, s)
    freq = fig9(kept)
    pick = fig10(kept)
    fig7(pick)
    out = dict(table2=t2, batch=s, top_sensitive=freq,
               selected=dict(run=pick['run'], **{k: pick['final'][k] for k in ('center_rms', 'worst_rms', 'W', 'frac')},
                             params=dict(zip(M.NAMES, np.round(pick['final']['x'], 6).tolist()))))
    (HERE/'RESULTS.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'selected'} | {'selected': {k: v for k, v in out['selected'].items()
                                                                                    if k != 'params'}}, indent=1))


if __name__ == '__main__':
    main()
