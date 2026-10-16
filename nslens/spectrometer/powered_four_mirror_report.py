"""Render and report saved population-CMA powered spectrometer candidates."""
import argparse
import html
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import powered_four_mirror as pm, model
from .four_mirror_render import render_layout
from .four_mirror_search import optimized_package, package_metrics, trace_summary
from .search import serial


def convergence(root,out):
    runs=[]
    for path in sorted(root.glob('seed_*/result.json')):
        row=json.loads(path.read_text())
        if row.get('history'):runs.append(row)
    if not runs and (root/'result.json').exists():
        row=json.loads((root/'result.json').read_text())
        if row.get('history'):runs.append(row)
    if not runs:return None
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained');colors=['#2475c2','#2b9a63','#d6534d']
    stages=[]
    for run,color in zip(runs,colors*10):
        h=run['history'];xs=[r['evaluations'] for r in h];ys=[r['best']['loss'] for r in h]
        seed=run['seed'];seedfile=root/f'seed_{seed}'/'inherited_seed.json'
        if not seedfile.exists():seedfile=root/'inherited_seed.json'
        initial=json.loads(seedfile.read_text())['metrics']['loss'] if seedfile.exists() else ys[0]
        ax.plot(xs,ys,color=color,lw=2,label=f'Seed {seed} · retained best')
        ax.axhline(initial,color=color,ls=':',lw=1.2,alpha=.7)
        split=next((r['evaluations']-r.get('stage_evaluations',0) for r in h if r['stage']=='freeform'),None)
        if split is not None:stages.append(split)
    if stages:
        split=stages[0];ax.axvline(split,color='#586576',ls='--',lw=1)
        ax.text(split*.5,.98,'Coarse: geometry + quadratic power',transform=ax.get_xaxis_transform(),ha='center',va='top',color='#586576')
        ax.text((split+max(xs))*.5,.98,'Fine: freeform terms released',transform=ax.get_xaxis_transform(),ha='center',va='top',color='#586576')
    ax.set(xlabel='Optical objective evaluations per seed',ylabel='Retained objective value',title='Population CMA-ES · coarse-to-fine convergence')
    ax.grid(alpha=.15);ax.legend(frameon=False,loc='best');ax.margins(y=.18)
    fig.text(.5,.005,'Dotted lines: inherited seed objective. Packaging orientation fitting below is reporting only.',ha='center',fontsize=9,color='#586576')
    fig.savefig(out/'cma_convergence.png',dpi=190);plt.close(fig)
    return dict(seeds=[r['seed'] for r in runs],stage_boundaries=stages)


def report(source,out=None):
    source=Path(source);data=json.loads(source.read_text());root=source.parent
    run_root=root.parent if root.name.startswith('seed_') else root
    output=Path(out) if out else root;visual=output/'visualizations';visual.mkdir(parents=True,exist_ok=True)
    genes=np.asarray(data.get('genes',data.get('best',{}).get('genes')),float)
    if genes.ndim!=1 or len(genes)!=pm.N_TOTAL:raise ValueError('Result must contain powered-model genes')
    tr=pm.trace(genes,fields=13,waves=17,pupils=11);layout=tr['model']
    hybrid='residual_calls' in data
    title=('CMA + terminal least-squares refinement' if data.get('accepted') else 'Provisional hybrid · terminal iterate rejected') if hybrid else 'CMA coarse-to-fine · four powered mirrors'
    rendered=render_layout(layout,tr,visual,title)
    parent_root=Path(data['source']).parent if hybrid and data.get('source') else run_root
    powers=[]
    for index in [0,1,3,4]:
        c=layout['surfaces'][index]['coeff'][:2]
        powers.append(dict(mirror=len(powers)+1,surface_index=index,quadratic_x_per_mm=float(c[0]),quadratic_y_per_mm=float(c[1]),vertex_curvature_x_per_mm=float(2*c[0]),vertex_curvature_y_per_mm=float(2*c[1]),active=bool(np.any(abs(c)>1e-12))))
    baseline=model.decode(pm.REFERENCE,aperture_radii=pm.APERTURES[1:4]);baseline['detector_bounds']=layout['detector_bounds'].copy()
    # Identical objective, dense sampling, multistart orientation budgets for both.
    packages={}
    for label,m in [('original_two_mirror',baseline),('four_powered_mirror',layout)]:
        candidates=[optimized_package(m,seed=s,iterations=60) for s in [406,907]]
        angle=min(candidates,key=lambda r:r['volume_cm3'])['rotation_xyz_deg']
        packages[label]=package_metrics(m,angle,samples=81)
    baseline_trace=model.trace(pm.REFERENCE,fields=13,waves=17,pupils=11,aperture_radii=pm.APERTURES[1:4])
    conv=convergence(parent_root,visual)
    summary=dict(source=str(source),sampling=rendered['sampling'],render_metrics=rendered,quadratic_powers=powers,
                 optical_envelopes=packages,baseline_imaging=trace_summary(baseline_trace),candidate_imaging=trace_summary(tr),convergence=conv,
                 package_comparison='Same fixed detector rectangle and original powered-surface apertures. Both receive identical two-start orientation fitting; dense81 surface samples plus rims. Orientation fitting is a reporting operation, not optical refinement.',
                 limitations='Sampled optical envelopes exclude mounts/substrates; orientation minima not proven global. Geometric survival excludes coating efficiency. WFE improvement is not inferred from a changed reference surface.')
    (output/'report_metrics.json').write_text(json.dumps(serial(summary),indent=2)+'\n')
    old=packages['original_two_mirror']['volume_cm3'];new=packages['four_powered_mirror']['volume_cm3']
    rows=''.join(f'<tr><td>M{p["mirror"]}</td><td>{p["quadratic_x_per_mm"]:.7g}</td><td>{p["quadratic_y_per_mm"]:.7g}</td><td>{"yes" if p["active"] else "no"}</td></tr>' for p in powers)
    dimensions=lambda row:' × '.join(f'{x:.2f}' for x in row['bbox_mm'])
    package_rows=''.join(f'<tr><td>{name}</td><td>{dimensions(packages[key])}</td><td>{packages[key]["volume_cm3"]:.3f}</td></tr>' for name,key in [('Original: two mirrors + grating','original_two_mirror'),('Four powered mirrors + grating','four_powered_mirror')])
    links=' · '.join(f'<a href="{html.escape(os.path.relpath(path,output))}">{label}</a>' for path,label in [(parent_root/'bounds.json','Search bounds'),(source,'Saved result'),(run_root/'independent_audit.json','Independent audit'),(output/'report_metrics.json','Report data'),(visual/'gallery.html','Render gallery')])
    if (visual/'seed_gallery.html').exists():links += ' · <a href="visualizations/seed_gallery.html">All three seed setups</a>'
    audit_note='Independent audit has not yet been generated; its link will become available when complete.'
    if (run_root/'independent_audit.json').exists():
        audit=json.loads((run_root/'independent_audit.json').read_text())
        audit_note=('Independent geometric audit passed for this selected prescription.' if audit.get('passed') and np.allclose(audit.get('genes',[]),genes,rtol=0,atol=0) else 'Independent audit available; consult its status and limitations.')+' This is not an independent optical engine or mechanical certification.'
    imaging_rows=''.join(f'<tr><td>{name}</td><td>{100*row["throughput"]:.4f}%</td><td>{1000*row["rms_spot_mm"]:.3f}</td><td>{1000*row["max_spot_mm"]:.3f}</td></tr>' for name,row in [('Original',summary['baseline_imaging']),('Candidate',summary['candidate_imaging'])])
    access_note=''
    access_path=run_root/'slit_access_audit.json'
    if access_path.exists():
        access=json.loads(access_path.read_text())
        blocked=access['blocked_fraction_within_distance_mm']['20']
        nearest=min(row['nearest_positive_distance_mm'] for row in access['upstream_obstructions'] if row['nearest_positive_distance_mm'] is not None)
        access_note=f'<aside style="border-left:4px solid #d8ad52;padding:12px 18px;background:#fff9ec"><strong>Entrance-access finding: optical candidate, not build-ready.</strong><p>The original audit starts at the slit. Backward continuation of the required input bundle first intersects optics {nearest:.3f} mm upstream; {100*blocked:.2f}% of sampled backward rays intersect optics within 20 mm. This is not a measured throughput loss: a nearby feed relay could redirect the bundle. Illumination access, slit holder, substrates and mounts still need design and validation.</p><a href="{html.escape(os.path.relpath(access_path,output))}">Entrance-access audit</a> · <a href="{html.escape(os.path.relpath(run_root / 'visualizations' / 'slit_access.html',output))}">Access visualization</a></aside>'
    text=f'''<!doctype html><meta charset="utf-8"><title>Four-powered-mirror CMA result</title><style>body{{font:16px/1.6 "DejaVu Sans",Arial;color:#172635;background:white;max-width:1200px;margin:35px auto;padding:0 25px}}h1,h2{{line-height:1.25}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;margin:15px 0}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.muted{{color:#586576}}a{{color:#2475c2}}</style>
<h1>Four powered mirrors + curved grating</h1><p class="muted">Run: {html.escape(run_root.name)}</p><p>{html.escape(title)}. Initialized from the inherited optical scaffold. Saved-result rendering and matched package comparison.</p>{'<p>Terminal local-refinement acceptance: '+str(data.get('accepted'))+'.</p>' if hybrid else ''}
<p><strong>{100*tr['good'].mean():.4f}% geometric ray survival</strong> across {tr['good'].size:,} launched rays. {sum(p['active'] for p in powers)} of four mirrors have nonzero quadratic power.</p>
{access_note}<img src="visualizations/design_3d.png"><h2>Matched optical envelopes</h2><table><tr><th>Layout</th><th>Bounding box (mm)</th><th>Volume (cm³)</th></tr>{package_rows}</table><p>Reported volume change: {(1-new/old)*100:.1f}% smaller. Both orientations fitted with the same budget; no optical parameters changed during reporting. The detector rectangle is fixed and identical. These are sampled optical envelopes, excluding substrates and mounts.</p>
<h2>Matched geometric imaging grid</h2><table><tr><th>Layout</th><th>Survival</th><th>Aggregate RMS spot (µm)</th><th>Worst RMS spot (µm)</th></tr>{imaging_rows}</table><p>Both use 13 slit positions × 17 wavelengths × 81 pupil samples. Imaging statistics use received rays; survival is listed separately.</p><h2>Active quadratic coefficients</h2><table><tr><th>Mirror</th><th>x² coefficient (mm⁻¹)</th><th>y² coefficient (mm⁻¹)</th><th>Powered</th></tr>{rows}</table><p class="muted">For the local sag z = cₓx² + cᵧy² + …, vertex principal curvatures are 2cₓ and 2cᵧ. Nonzero power does not imply equal optical contribution.</p>
{'<img src="visualizations/cma_convergence.png">' if conv else ''}<img src="visualizations/metric_maps.png"><img src="visualizations/design_sideviews.png"><p class="muted">Geometric blur is computed over received rays; survival is shown separately. No diffraction PSF, coating efficiency, finite slit width, tolerance model, or mechanical certification. No WFE improvement is inferred from a moved reference surface.</p><p>{audit_note}</p><p>{links}</p>'''
    (output/'report.html').write_text(text)
    return serial(summary)

def render_seed_gallery(root):
    root=Path(root);cards=[]
    for path in sorted(root.glob('seed_*/result.json')):
        data=json.loads(path.read_text());seed=data['seed'];tr=pm.trace(data['best']['genes'],fields=13,waves=17,pupils=11)
        output=root/'visualizations'/f'seed_{seed}'
        metrics=render_layout(tr['model'],tr,output,f'Sampled CMA candidate · seed {seed}')
        cards.append(f'<section><h2>Seed {seed}</h2><p>{100*tr["good"].mean():.4f}% geometric ray survival on the render grid. Sampled CMA candidate; no independent validation implied.</p><a href="seed_{seed}/gallery.html"><img src="seed_{seed}/design_3d_preview.jpg"></a><p><a href="seed_{seed}/design_3d.png">Full-resolution render</a> · <a href="seed_{seed}/gallery.html">Candidate gallery and metrics</a></p></section>')
    target=root/'visualizations'/'seed_gallery.html'
    target.write_text('<!doctype html><meta charset="utf-8"><title>CMA seed candidates</title><style>body{font:16px/1.6 Arial;color:#172635;background:white;max-width:1200px;margin:35px auto;padding:0 20px}img{width:100%;height:auto}section{margin:30px 0}a{color:#2475c2}</style><h1>Four-powered-mirror CMA candidates</h1><p>Saved winners from three independent CMA populations. Each image traces its own prescription. See the main report for the independently audited selected candidate.</p>'+''.join(cards))
    return str(target)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--out');a=p.parse_args();r=report(a.source,a.out)
    print(json.dumps(dict(candidate={k:v for k,v in r['candidate_imaging'].items() if k!='centroids_mm'},envelopes=r['optical_envelopes']),indent=2))
