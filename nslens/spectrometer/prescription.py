"""Export a saved candidate's physical prescription and directional blur metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import model
from .search import write_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result');a=p.parse_args(argv)
    path=Path(a.result);r=json.loads(path.read_text());g=r['genes'];apertures=r.get('configuration',{}).get('aperture_radii')
    d=model.decode(g,aperture_radii=apertures);tr=model.trace(g,fields=8,waves=8,pupils=10,aperture_radii=apertures)
    good=tr['good'];xy=tr['xy'];n=good.sum(2);cent=np.sum(np.where(good[...,None],xy,0),axis=2)/np.maximum(n[...,None],1)
    rms=np.sqrt(np.sum(np.where(good[...,None],(xy-cent[:,:,None])**2,0),axis=2)/np.maximum(n[...,None],1))
    metrics=model.distortion_metrics(g,aperture_radii=apertures)
    metrics.update(dispersion_direction_rms_um=float(np.sqrt(np.mean(rms[:,:,0]**2))*1000),worst_dispersion_direction_rms_um=float(rms[:,:,0].max()*1000),slit_direction_rms_um=float(np.sqrt(np.mean(rms[:,:,1]**2))*1000),min_throughput=float(np.min(n/good.shape[2])))
    write_json(path.parent/'distortion.json',metrics)
    lines=['# Discovered spectrometer prescription','',f'Seed {r["seed"]}; source result `{path}`. This is a research candidate, not a manufacturing-ready or paper-equivalent design. It descends from randomized CMA discovery, without a published prescription seed. All coordinates and distances below are mm.','',f'Normalized genes: `{g}`','', 'Slit spans y = −5 to +5 mm at x=z=0. Input cone F/3.8. Grating density300 lines/mm, first diffracted order, wavelength500–1100nm.', '',f'Chief scaffold segment lengths: `{d["lengths"].tolist()}`',f'Chief scaffold turns (degrees): `{np.degrees(d["turns"]).tolist()}`','']
    for i,s in enumerate(d['surfaces']):
        lines.extend([f'## Surface {i+1}: '+(['primary mirror','reflective grating','tertiary mirror'][i]),'',f'Vertex: `{s["vertex"].tolist()}`',f'Local basis (columns are local x,y,z in global coordinates): `{s["basis"].tolist()}`',f'Actual circular optical aperture radius: {s["aperture"]:.12g} mm.','', '| Sag monomial | Coefficient (mm^(1−degree)) |','|---|---:|'])
        for (x,y),c in zip(model.TERMS,s['coeff']):lines.append(f'| x^{x} y^{y} | {c:.14g} |')
        lines.append('')
    lines.extend(['## Common image plane','',f'Point: `{d["image"].tolist()}`',f'Normal: `{d["image_normal"].tolist()}`',f'Spectral-coordinate axis: `{d["image_x"].tolist()}`',f'Slit-coordinate axis: `{d["image_y"].tolist()}`','', 'The JSON and independent audit retain the achieved support, signed principal-curvature checks and imaging metrics. A positive quadratic grating base alone does not guarantee convexity after freeform departures. Substrates and mounts are not modeled.'])
    (path.parent/'PRESCRIPTION.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
