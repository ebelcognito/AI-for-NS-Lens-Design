"""Render a saved spectrometer candidate: python -m nslens.spectrometer.plot result.json."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import trace,sag,evaluate

def render(path):
    path=Path(path); data=json.loads(path.read_text()); g=data['genes']; apertures=data.get('configuration',{}).get('aperture_radii'); r=trace(g,fields=3,waves=3,pupils=7,aperture_radii=apertures); m=r['model']
    fig,axs=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    ax=axs[0,0]
    colors=['tab:blue','tab:green','tab:red']
    good=r['good'].ravel()
    for k in np.flatnonzero(good)[::max(1,np.sum(good)//50)]:
        p=r['paths'][:,k]; ax.plot(p[:,0],p[:,2],alpha=.25,lw=.7)
    for j,s in enumerate(m['surfaces']):
        x=np.linspace(-s['aperture'],s['aperture'],101); z,_,_=sag(x,np.zeros_like(x),s['coeff']); p=s['vertex']+x[:,None]*s['basis'][:,0]+z[:,None]*s['basis'][:,2]
        ax.plot(p[:,0],p[:,2],'k',lw=2); ax.text(*s['vertex'][[0,2]],['Primary','Grating','Tertiary'][j])
    p=m['image']+np.array([-8,8])[:,None]*m['image_x']; ax.plot(p[:,0],p[:,2],'m',lw=2,label='Detector')
    ax.set(title='Meridional layout (accepted sampled rays)',xlabel='Global x (mm)',ylabel='Global z (mm)',aspect='equal'); ax.grid(alpha=.2)
    ax=axs[0,1]
    for f in range(3):
        for w in range(3):
            p=r['xy'][f,w][r['good'][f,w]]
            if len(p): ax.scatter(p[:,0],p[:,1],s=5,alpha=.5,color=colors[w])
    ax.set(title='Common detector: slit × wavelength footprints',xlabel='Dispersion coordinate (mm)',ylabel='Slit coordinate (mm)',aspect='equal'); ax.grid(alpha=.2)
    ax=axs[1,0]
    for f in range(3):
        for w in range(3):
            p=r['xy'][f,w][r['good'][f,w]]
            if len(p):
                p=(p-p.mean(axis=0))*1000; ax.scatter(p[:,0],p[:,1],s=5,alpha=.4,color=colors[w])
    ax.set(title='Centred spot footprints (all slit/band samples)',xlabel='Local x (µm)',ylabel='Local y (µm)',aspect='equal'); ax.grid(alpha=.2)
    ax=axs[1,1]; ax.axis('off'); stats=evaluate(g,fields=8,waves=8,pupils=10,aperture_radii=apertures)
    text='DENSE SAME-ENGINE AUDIT\n\n'+ '\n'.join(f'{key}: {stats[key]:.5g}' for key in ['throughput','min_throughput','rms_spot_mm','max_spot_mm','rms_wfe_waves','max_wfe_waves','dispersion_mm_per_um','slit_magnification','obstruction_fraction'])
    text+='\n\n500–1100 nm; 10 mm slit; F/3.8 input\n300 lines/mm uniform projected grating\nBlue / green / red = 500 / 800 / 1100 nm\nPrototype model; sampled clearance checks only.\nNot a fabrication-ready or paper-equivalent design.'
    ax.text(0,1,text,va='top',fontfamily='monospace',fontsize=10)
    target=path.with_name('candidate.png'); fig.savefig(target,dpi=160); plt.close(fig); print(target)
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('result'); render(p.parse_args().result)
