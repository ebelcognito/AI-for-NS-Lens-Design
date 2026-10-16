"""Local handoff page for the Ge specification-matched experiment."""
import html
import json
from pathlib import Path

def write_report(root):
    root=Path(root)
    if not (root/'configuration.json').exists() or not (root/'audit.json').exists():return
    a=json.loads((root/'audit.json').read_text());c=json.loads((root/'configuration.json').read_text())
    rows=''.join(f'<tr><td>{html.escape(k.replace("_"," "))}</td><td>{"PASS" if v else "FAIL"}</td></tr>' for k,v in a['gates'].items())
    caveats=''.join(f'<li>{html.escape(x)}</li>' for x in c['limitations'])
    def um(key):return f'{a[key]*1000:.2f}' if a.get(key) is not None else 'unavailable'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ge F/1.4 matched specifications</title>
<style>body{{max-width:1100px;margin:48px auto;padding:0 24px;font:17px/1.5 system-ui;color:#172635;background:white}}h1{{font-size:32px}}h2{{font-size:22px}}img,video{{width:100%;border:1px solid #e4e9ef}}a{{color:#2475c2}}td{{padding:8px;border-bottom:1px solid #e4e9ef}}</style>
<h1>Germanium monolithic objective — matched specifications</h1>
<p>F/1.4 · 8.4 mm entrance pupil · 11.76 mm inferred focal length · 37° × 25° field · 8–12 µm · 58 mm internal chief-ray path.</p>
<p>Three rotational aspheres plus one XY freeform on mirror 1. CMA 17 → 35 variables followed by bounded least-squares. This is not the paper's original prescription or Fermat-based solver.</p>
<p>Dense audit: average supported monochromatic RMS <b>{um('average_supported_cell_rms_mm')} µm</b>; worst monochromatic RMS <b>{um('max_supported_cell_rms_mm')} µm</b>; worst polychromatic RMS <b>{um('max_polychromatic_field_rms_mm')} µm</b>.
Geometric ray transmission {a['support']:.2%}; minimum field/wavelength transmission {a['minimum_cell_support']:.2%}.</p>
<p>Traced internal chief path: {a['traced_chief_path_mm']:.6f} mm. Maximum relative field distortion: {a['max_relative_distortion']:.2%}.</p>
<p><a href="configuration.json">Exact inputs and limitations</a> · <a href="audit.json">Full audit</a> · <a href="selected.json">Selected prescription</a> · <a href="trajectory.json">Saved states</a></p>
<img src="visualizations/final_design.png" alt="Optical layout">
<h2>Actual saved evolution</h2><video controls preload="metadata" src="visualizations/evolution.mp4"></video>
<h2>All-field spots</h2><img src="visualizations/spot_maps.png" alt="Spot maps">
<h2>Audit screens</h2><table>{rows}</table>
<p>Comparing geometric RMS to an Airy radius is only a screening test, not proof of diffraction-limited performance. No MTF, wavefront, absorption, coatings, manufacturing, or thermal validation is included.</p>
<h2>Remaining comparison limitations</h2><ul>{caveats}</ul>
<p>Reference: <a href="https://www.nature.com/articles/s41377-021-00538-1">Duerr and Thienpont (2021), Figure 6</a>. Different parameterizations, initialization, material approximations, and computational implementations prevent an optimizer-superiority claim from this run alone.</p></html>'''
    (root/'index.html').write_text(page)
