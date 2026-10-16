"""Produce a compact local HTML handoff from measured experiment artifacts."""
import argparse
import html
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.out)
    audit=json.loads((root/'audit.json').read_text()); chosen=json.loads((root/'selected.json').read_text())
    gate_rows=''.join(f'<tr><td>{html.escape(k.replace("_"," "))}</td><td>{"PASS" if v else "FAIL"}</td></tr>' for k,v in audit['gates'].items())
    rms=audit['max_supported_cell_rms_mm']; quality='Passed the sampled pilot gates' if audit['passes_all_gates'] else 'Improved pilot — chosen pilot image-quality target not yet achieved'
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Monolithic IR experiment</title>
<style>body{{max-width:1120px;margin:48px auto;padding:0 24px;font:17px/1.55 system-ui;color:#172635;background:white}}h1{{font-size:32px}}h2{{font-size:22px;margin-top:38px}}a{{color:#2475c2}}img,video{{width:100%;border:1px solid #e4e9ef;border-radius:8px}}table{{border-collapse:collapse;width:100%}}td{{padding:8px;border-bottom:1px solid #e4e9ef}}small{{color:#586576}}</style>
<h1>Monolithic infrared objective</h1><p>{quality}.</p>
<p>Dense sampled ray transmission: {audit['support']:.1%}; worst field/wavelength: {audit['minimum_cell_support']:.1%}.
Worst supported monochromatic RMS spot radius: {1000*rms:.1f} µm (17 µm pilot target).</p>
<p><a href="selected.json">Selected prescription</a> · <a href="audit.json">Full independent audit</a> · <a href="trajectory.json">Saved evolution states</a> · <a href="status.json">Run status</a></p>
<img src="visualizations/final_design.png" alt="Traced monolithic optical layout in white spectrometer theme">
<h2>Actual design evolution</h2><p>Recorded CMA incumbents and accepted least-squares iterates; no interpolated prescriptions.</p>
<video controls preload="metadata" src="visualizations/evolution.mp4"></video>
<h2>All-field spot maps</h2><img src="visualizations/spot_maps.png" alt="Field-by-field polychromatic geometric spot diagrams">
<h2>Audit gates</h2><table>{gate_rows}</table>
<h2>What this reconstructs</h2>
<p>A Fraunhofer-inspired four-zone monolithic architecture: refractive entrance, two coated reflective surfaces, real intermediate image, and refractive exit. This is not the unpublished numerical prescription.</p>
<p>Published target: 37° × 25°, F/1, 17 µm pixels, 8–14 µm and a 20 × 20 × 10 mm envelope. Our 6 mm focal length and 6 mm entrance pupil are explicit assumptions. Four XY-polynomial patches provide more freeform freedom than the reference's stated configuration. The solid uses interpolated non-optical faces; their machinability is not certified.</p>
<p>IG4 dispersion is used as a disclosed material choice. Values beyond 12 µm are extrapolated; transmission, coatings, MTF, tolerances, stray light, and thermal robustness remain unvalidated. Geometric spot size alone is not a diffraction-limited imaging claim.</p>
<p>Sources: <a href="https://www.iof.fraunhofer.de/en/business-fields/optical-components-and-systems/freeform-based-monolithic-night-vision-objective.html">Fraunhofer IOF target</a> · <a href="https://www.vitron.de/upload/vitron-ig-4-datenblatt-okt-2020-1_299.pdf">VITRON IG4 data</a> · <a href="https://www.nature.com/articles/s41377-021-00538-1">Related freeform design method (different Ge objective)</a></p>
<small>Each search and polishing folder contains its model snapshot, configuration, trajectory, and result. The audit separates supported and extrapolated spectral bands.</small></html>'''
    (root/'index.html').write_text(content)

if __name__=='__main__':main()

