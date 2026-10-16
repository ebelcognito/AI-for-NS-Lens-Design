"""Matched-scale optical layouts, detector footprints and nine-field spot maps."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .plot import _backend, _surface_profile


def render(before, after, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for label, path in [('CMA freeform', before), ('After least-squares', after)]:
        data = json.loads(Path(path).read_text())
        optical = _backend(data, data['genes'])
        rays = optical.trace(data['genes'], fields=3, waves=1, pupils=17)
        runs.append((label, optical, rays))
    colors = plt.cm.tab10(np.arange(9))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    for col, (label, optical, rays) in enumerate(runs):
        ax = axs[0, col]
        m = rays['model']; npup = rays['good'].shape[-1]
        for fi in (3, 4, 5):
            ids = np.flatnonzero(rays['good'][fi, 0])
            ids = ids[np.linspace(0, len(ids)-1, min(12, len(ids)), dtype=int)]
            for idx in ids:
                p = rays['paths'][:, fi*npup+idx]
                ax.plot(p[:, 0], p[:, 2], color=colors[fi], alpha=.4, lw=.7)
        _surface_profile(ax,optical,m['plate'],m['primary_radius'],'tab:green','Corrector')
        ax.plot([-m['primary_radius'],m['primary_radius']], [0,0], color='tab:green', lw=1)
        _surface_profile(ax,optical,m['primary'],m['primary_radius'],'black','Primary',m['hole_radius'])
        _surface_profile(ax,optical,m['secondary'],m['secondary_radius'],'tab:orange','Secondary')
        ax.plot([-.96,.96],[m['detector_z']]*2,color='magenta',lw=3,label='Detector')
        ax.set(title=label+' — folded ray paths',xlabel='x (mm)',ylabel='z (mm)',
               xlim=(-5.5,5.5),ylim=(-1.2,8.3),aspect='equal')
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='lower left')
        ax = axs[1,col]
        for fi in range(9):
            xy = rays['xy'][fi,0,rays['good'][fi,0]]
            ax.scatter(xy[:,0],xy[:,1],s=2,alpha=.35,color=colors[fi])
            if len(xy):
                center = xy.mean(axis=0)
                ax.annotate(str(fi+1),center,xytext=(4,4),textcoords='offset points',fontsize=9)
        ax.add_patch(Rectangle((-.96,-.72),1.92,1.44,fill=False,ls='--',color='black'))
        ax.set(title='Detector footprints — nine fields',xlabel='Sensor x (mm)',ylabel='Sensor y (mm)',
               xlim=(-1.05,1.05),ylim=(-.82,.82),aspect='equal')
        ax.grid(alpha=.2)
    fig.suptitle('Camera reconstruction: one channel, before and after polishing\n'
                 'x–z ray projections; surface sections at y=0. Steering prisms omitted here.',fontsize=13)
    fig.savefig(out/'design_comparison.png',dpi=180); plt.close(fig)
    fig = plt.figure(figsize=(14,10),layout='constrained')
    subfigs = fig.subfigures(1,2)
    all_points = []
    for _,_,r in runs:
        for fi in range(9):
            p = r['xy'][fi,0,r['good'][fi,0]]*1000
            all_points.append(p-p.mean(axis=0))
    limit = max(15.,5*np.ceil(max(float(np.max(abs(p))) for p in all_points)/5))
    for sf,(label,optical,rays) in zip(subfigs,runs):
        axs = sf.subplots(3,3,sharex=True,sharey=True)
        sf.suptitle(label,fontsize=14)
        # Positive vertical field is at the top, matching the sensor footprint.
        for fi in range(9):
            row,col = divmod(fi,3); ax = axs[2-row,col]
            p = rays['xy'][fi,0,rays['good'][fi,0]]*1000
            p -= p.mean(axis=0)
            rms = np.sqrt(np.mean(np.sum(p*p,axis=1)))
            ax.scatter(p[:,0],p[:,1],s=2,alpha=.5,color=colors[fi])
            ax.add_patch(Rectangle((-6,-6),12,12,fill=False,ls='--',lw=1,color='black'))
            fx,fy = rays['fields_deg'][fi]
            ax.set(title=f'{fi+1}: ({fx:g}°, {fy:g}°)\nRMS {rms:.2f} µm\nPupil {rays["good"][fi,0].mean():.1%}',
                   xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal')
            ax.title.set_fontsize(8)
            ax.grid(alpha=.2); ax.tick_params(labelsize=8)
            if row==0: ax.set_xlabel('Δx (µm)')
            if col==0: ax.set_ylabel('Δy (µm)')
    fig.suptitle('Centroided geometric spot maps — identical scales in every panel\n'
                 'Dashed square = one assumed 12 µm pixel. 11 µm wavelength; no diffraction PSF.',fontsize=13)
    fig.savefig(out/'spot_comparison.png',dpi=180); plt.close(fig)
    print(out/'design_comparison.png'); print(out/'spot_comparison.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('before');p.add_argument('after');p.add_argument('--out',required=True)
    a=p.parse_args();render(a.before,a.after,a.out)

