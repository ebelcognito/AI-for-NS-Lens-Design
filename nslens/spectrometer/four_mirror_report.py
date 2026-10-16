"""Create a portable HTML report for the folded spectrometer experiment."""
from pathlib import Path
import argparse
import html
import json
import numpy as np
from . import four_mirror as fm
from .four_mirror_plot import render
from .four_mirror_search import package_metrics
from .search import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    args=parser.parse_args();out=Path(args.directory)
    result=json.loads((out/'result.json').read_text())
    b=fm.load_baseline(result['configuration']['source'],fields=5,waves=7,pupils=9)
    layout=fm.build_layout(result['best']['params'],b)
    base=dict(b['model'],detector_bounds=b['detector_bounds'])
    render(layout,base,out/'comparison.png',paths=layout['nominal_paths'])
    # Denser disk sampling at frozen selected box orientations; no reselection.
    bg=package_metrics(base,result['baseline']['geometry']['rotation_xyz_deg'],samples=101)
    fg=package_metrics(layout,result['best']['geometry']['rotation_xyz_deg'],samples=101)
    write_json(out/'dense_envelope.json',dict(baseline=bg,folded=fg,reduction_fraction=1-fg['volume_cm3']/bg['volume_cm3'],limitation='Dense sampled envelope at frozen orientations; no mechanical components.'))
    prescription=[]
    for name,s in zip(['F1 flat','M1 powered','Curved grating','M2 powered','F2 flat'],layout['surfaces']):
        prescription.append(dict(name=name,vertex_mm=s['vertex'],normal=s['basis'][:,2],basis=s['basis'],aperture_radius_mm=s['aperture'],sag_coefficients=s['coeff']))
    write_json(out/'prescription.json',dict(params=result['best']['params'],surfaces=prescription,detector={k:layout[k] for k in ['image','image_x','image_y','image_normal','detector_bounds']},baseline_source=result['configuration']['source']))
    def dims(g):return ' × '.join(f'{x:.2f}' for x in g['bbox_mm'])+' mm'
    auditpath=out/'audit.json';audit=json.loads(auditpath.read_text()) if auditpath.exists() else {}
    status='PASS' if audit.get('passed') else ('REJECTED — lost rays in independent audit' if audit else 'Pending independent audit')
    params=result['best']['params'];ap=[s['aperture']*2 for s in layout['surfaces']]
    rows=''.join('<tr><td>'+html.escape(str(k))+'</td><td>'+html.escape(str(v))+'</td></tr>' for k,v in [
        ('Baseline optical envelope',f"{bg['volume_cm3']:.2f} cm³ ({dims(bg)})"),
        ('Four-mirror optical envelope',f"{fg['volume_cm3']:.2f} cm³ ({dims(fg)})"),
        ('Envelope reduction',f"{100*(1-fg['volume_cm3']/bg['volume_cm3']):.1f}%"),
        ('Search and local-refinement objective calls',result['calls']),
        ('Independent ray audit',status),
        ('F1: leg fraction / turn / azimuth',f'{params[0]:.6f} / {params[1]:.3f}° / {params[2]:.3f}°'),
        ('F2: leg fraction / turn / azimuth',f'{params[3]:.6f} / {params[4]:.3f}° / {params[5]:.3f}°'),
        ('Physical optical diameters F1 / M1 / G / M2 / F2',' / '.join(f'{x:.3f}' for x in ap)+' mm')])
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Four-mirror spectrometer exploration</title>
<style>body{{font:17px/1.5 system-ui;margin:40px auto;max-width:1150px;padding:0 20px;color:#182331}}table{{border-collapse:collapse;width:100%}}td{{padding:10px;border-bottom:1px solid #ddd}}td:first-child{{font-weight:600}}img{{width:100%}}code{{background:#eef2f5}}a{{color:#1858a8}}</style></head><body>
<h1>Four-mirror geometry feasibility probe</h1><p>Two added flat mirrors fold the existing two-powered-mirror spectrometer. The optical prescription is inherited from the accepted baseline. Discovery and bounded Powell refinement change packaging; they do not improve the inherited optical aberrations.</p>
<table>{rows}</table><img src="comparison.png" alt="Equal-scale optical layouts">
<p>Configuration: 10 mm slit, 500–1100 nm, F/3.8 input, 300 lines/mm curved reflective grating. Both envelopes include the full sampled physical optical disks, slit endpoints, and the same detector rectangle with a 1 mm edge margin. Each layout receives the same box-orientation search procedure; quoted dimensions are dense resamples at those fixed orientations, not certified minimum-volume boxes.</p>
<p>Flat apertures are sized on the training rays and held fixed for validation. A passing ray audit establishes sampled geometric survival and image preservation; reflectivity, diffraction efficiency, tolerances, mounts, and substrates remain unmodeled. Two extra real mirrors will incur coating losses. Existing powered-surface aperture margins are inherited, including the tight grating margin.</p>
<p>Azimuth is defined by <code>four_mirror._flat</code> about each incoming chief direction; it is not an Euler mirror tilt. The prescription JSON contains unambiguous physical normals and all local bases.</p>
<p><a href="result.json">Search and refinement results</a> · <a href="audit.json">Independent ray audit</a> · <a href="prescription.json">Physical prescription</a> · <a href="dense_envelope.json">Dense envelope comparison</a> · <a href="configuration.json">Run configuration</a></p>
</body></html>'''
    (out/'report.html').write_text(text)
    print(out/'report.html')

if __name__=='__main__':main()
