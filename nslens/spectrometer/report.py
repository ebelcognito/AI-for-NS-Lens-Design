"""Create a compact, reproducible report comparing all completed pilot stages."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from . import model
from .search import write_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cma',required=True)
    p.add_argument('--hybrid')
    p.add_argument('--quality')
    p.add_argument('--constrained')
    p.add_argument('--final')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for f in sorted(Path(a.cma).glob('seed_*/result.json')):
        r=json.loads(f.read_text()); initial=min(r['initial_population'],key=lambda x:x['metrics']['loss'])
        base=next(s['best'] for s in r['stages'] if s['stage']=='base')
        for stage,genes in [('initial',initial['genes']),('base CMA',base['genes']),('freeform CMA',r['genes'])]:
            metric=model.evaluate(genes,fields=8,waves=8,pupils=10)
            geo=model.geometry_metrics(genes)
            rows.append(dict(seed=r['seed'],stage=stage,metrics=metric,geometry=geo,source=str(f)))
    if a.hybrid:
        for f in sorted(Path(a.hybrid).glob('seed_*/result.json')):
            r=json.loads(f.read_text())
            rows.append(dict(seed=r['seed'],stage='hybrid polish',metrics=r['audit'],geometry=r['geometry'],source=str(f)))
    if a.quality:
        for f in sorted(Path(a.quality).glob('seed_*/result.json')):
            r=json.loads(f.read_text())
            rows.append(dict(seed=r['seed'],stage='quality polish',metrics=r['audit'],geometry=r['geometry'],source=str(f)))
    if a.constrained:
        for f in sorted(Path(a.constrained).glob('seed_*/result.json')):
            r=json.loads(f.read_text())
            rows.append(dict(seed=r['seed'],stage='hybrid, fixed physical apertures',metrics=r['before_audit'],geometry=r['initial_geometry'],source=str(f)))
            rows.append(dict(seed=r['seed'],stage='curvature-constrained polish',metrics=r['audit'],geometry=r['geometry'],source=str(f)))
    if a.final:
        for f in sorted(Path(a.final).glob('seed_*/result.json')):
            r=json.loads(f.read_text())
            rows.append(dict(seed=r['seed'],stage='validated aperture-only correction',metrics=r['audit'],geometry=r['geometry'],source=str(f)))
    write_json(out/'comparison.json',rows)
    lines=['# Architecture-conditioned spectrometer discovery: first pilot','',
        'The optimizer discovered working three-surface spectrometer prescriptions from randomized, bounded chief-ray scaffolds. The runs did not reproduce the paper’s diffraction-limited performance. Full-coverage candidates are distinguished from optically incomplete failures below.','',
        'CMA uses the repository’s existing full-covariance optimizer; the new domain model is a separate 3D sequential tracer with intervening-surface obstruction checks. The primary–convex-grating–tertiary family is supplied. The initial CMA runs received no published prescription or successful archived optical seed; later local stages start from their own discovered candidate. This demonstrates prescription search within a family, not discovery of the family.','',
        '| Seed | Stage | Minimum field/band ray support | RMS spot (µm) | Worst cell RMS WFE (waves) | Dispersion (mm/µm) | Sampled optical AABB (cm³) |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['metrics'];v=r['geometry']['sampled_aabb_volume_cm3']
        lines.append(f"| {r['seed']} | {r['stage']} | {100*m['min_throughput']:.1f}% | {1000*m['rms_spot_mm']:.2f} | {m['max_wfe_waves']:.3f} | {m['dispersion_mm_per_um']:.3f} | {v:.2f} |")
    lines.extend(['',
        'All table rows use 8 slit positions × 8 wavelengths × the circular portion of a 10×10 pupil grid (60 pupil rays, 3,840 launched rays). New interior slit/wavelength samples are present; endpoints overlap training. This is denser same-model validation. Spots and WFE for incomplete cases describe surviving rays only and are not successful image-quality results. The independent polynomial-root audit, where present beside each result, separately checks ray intersections.','',
        'The initial row is the best training-loss candidate from the first sampled CMA population; it is not a matched-budget random-search baseline. Base CMA uses 1,200 evaluations and freeform CMA adds 2,400 per seed (10,800 completed search evaluations total). Each seed owns its random stream and candidate pool. The hybrid rows use a different, stricter all-ray image-mapping least-squares residual; their finite-difference calls are logged separately, so they must not be represented as CMA-only improvement.','',
        'The target slit is 10 mm, the band 500–1100 nm, the input cone F/3.8, and the reflective grating 300 lines/mm. The scaffold permits three fold angles within 100–150°, −150 to −100°, and 100–150°; four path lengths span 25–65, 25–75, 25–75 and 25–65 mm. Six quadratic powers have fixed concave/convex/concave signs. Fifteen base genes expand to thirty-six genes by adding seven higher polynomial coefficients per surface. This is a pilot polynomial basis, not the paper’s exact FRINGE Zernike expansion.','',
        'Hybrid refinement enforces 10 mm/µm spectral scale and unit slit magnification; unit magnification is our chosen instrument requirement, not a verified transcription of the paper. It retains every launched ray in its residual, adds failed-route and aperture-overflow penalties, and softly penalizes sampled optical-surface AABB volume above 100 cm³. Fixed 18 mm-radius optical apertures remain; substrates, mounts and a detector outline are absent. The AABB is neither a certified mechanical package nor a hard volume constraint.','',
        'Reproduction requires full slit/band/pupil support, worst-cell RMS WFE near or below 0.07 waves across a converged sampling grid, appropriate dispersion/distortion, and a validated compact physical assembly. Analytic grating and wavefront tests and root cross-checks reduce implementation risk; they do not certify manufacturing tolerances or full independent optical performance.','',
        '## Reproduce','', '```bash',
        '.venv/bin/python -m nslens.spectrometer.search --out experiments/spectrometer_pilot_v1 --seeds 101 102 103 --base-evals 1200 --freeform-evals 2400 --workers 3',
        '.venv/bin/python -m nslens.spectrometer.polish --source experiments/spectrometer_pilot_v1 --out experiments/spectrometer_hybrid_v1 --seeds 102 103 --max-nfev 120 --workers 2',
        '```','',
        'The additional quality-focused diagnostic, if present, starts only from seed 102’s hybrid prescription. It emphasizes within-cell spot quality, reduces the non-paper exact-unit-magnification mapping weight to 0.1, retains a strong endpoint spectral-span constraint, and trains on 5 slit × 5 wavelength samples with 25 interior plus 32 rim pupil directions. It preserves the failed-ray and soft volume penalties. Its results use the same dense audit in the table; the withheld staggered-ring/rim audit is reported separately.', '', '```bash', '.venv/bin/python -m nslens.spectrometer.polish --source experiments/spectrometer_hybrid_v1 --out experiments/spectrometer_quality_v1 --seeds 102 --max-nfev 160 --workers 1 --mapping-weight 0.1 --rim-points 32 --fields 5 --waves 5 --pupils 7', '```', '',
        'Output directories must be new. The JSON results contain normalized and physical prescriptions, objective histories, stages, audit results and exact configurations. Snapshots preserve the implementation used by each experiment.','',
        '## Model snapshots',''])
    for folder in [Path(a.cma)]+([Path(a.hybrid)] if a.hybrid else [])+([Path(a.quality)] if a.quality else [])+([Path(a.constrained)] if a.constrained else []):
        snap=folder/'model_snapshot.py'
        if snap.exists():lines.append(f'- `{snap}`: SHA-256 `{hashlib.sha256(snap.read_bytes()).hexdigest()}`.')
        summary=folder/'summary.json'
        if summary.exists():
            s=json.loads(summary.read_text());lines.append(f'- `{folder}` wall time: {s["wall_seconds"]:.1f} s.')
            for r in s['runs']:
                if 'residual_calls' in r:lines.append(f'  - Seed {r["seed"]}: {r["residual_calls"]} actual residual calls, {r["scipy_nfev"]} solver-reported evaluations.')
    lines.extend(['','Two partial development runs are preserved and excluded from optical outcome claims: the initial unconstrained-curvature prototype was stopped before the corrected CMA run; a hybrid checkpoint-writing error aborted at 200 residual calls per seed before a deterministic rerun. These overheads are not hidden in the completed-run evaluation counts.',''])
    final_folder=a.final or a.constrained or a.quality
    if final_folder:
        detail=Path(final_folder)/'seed_102'/'distortion.json'
        if detail.exists():
            d=json.loads(detail.read_text())
            lines.extend(['## Final-candidate tradeoffs and artifacts','',
                f"On the dense grid, spectral-direction RMS spot is {d['dispersion_direction_rms_um']:.2f} µm on aggregate and {d['worst_dispersion_direction_rms_um']:.2f} µm in the worst cell. These geometric width measures exclude a finite slit-width convolution, detector sampling and diffraction, and are not a claimed spectral resolving power.",'',
                f"Spectral smile is {1000*d['max_spectral_smile_mm']:.1f} µm and spatial keystone {1000*d['max_spatial_keystone_mm']:.1f} µm under the code's peak-to-peak centroid definitions. The unconstrained diagnostic traded sharper spots for greater spectral smile; the final constrained repair also changes physical aperture sizing, so its gains cannot be attributed to one merit-weight change alone.",'',
                'The unconstrained quality-polish candidate fails the required convex-grating condition even on its illuminated footprint. It is a diagnostic, not a valid final-family result. The subsequent constrained repair starts again from the valid hybrid candidate, sets fixed physical circular apertures from that candidate’s audited footprint plus 1 mm margin, and penalizes sign violations of the sag Hessian over each entire physical disk. Independent dense signed-principal-curvature and all-ray support audits determine acceptance; the penalty alone does not certify it.','',
                f"Final prescription and plots: `{final_folder}/seed_102/PRESCRIPTION.md`, `result.json`, `candidate.png`, `distortion.json`, and the independent `physics_audit.json` or rim-audit artifacts alongside them.",'',
                'The remaining research bottleneck is full-aperture image quality, together with distortion and physical shape/packaging constraints. More evaluation budget alone is not established as the remedy: the earlier local stages converged under their respective merits, while the final repair may be budget-limited; the surface basis, scaffold ranges and merit also restrict the available solutions. A next experiment should widen the validated freeform basis and pose family, preserve curvature constraints over illuminated footprints, and compare the attainable residual with an independently reconstructed reference design.',''])
    if a.constrained:
        lines.extend(['## Curvature-constrained repair','', 'The sharp unconstrained candidate exposed a real representation loophole: assigning a convex quadratic grating base does not prevent higher-order departures from becoming locally concave. The repair uses fixed physical radii 8.6064072098, 8.3708801659 and 12.1071506883 mm, selected from the earlier valid hybrid footprints plus 1 mm margin. Tracing uses these circles as actual apertures; no post-hoc loss of rays is accepted. The smaller aperture AABB must not be compared with the oversized 18 mm-radius AABB as a pure optimizer improvement.', '', '```bash', '.venv/bin/python -m nslens.spectrometer.polish --source experiments/spectrometer_hybrid_v1 --out experiments/spectrometer_constrained_v1 --seeds 102 --max-nfev 160 --workers 1 --mapping-weight 0.1 --rim-points 32 --fields 5 --waves 5 --pupils 7 --aperture-radii 8.606407209819915 8.370880165877303 12.107150688327819 --curvature-weight 10000', '```', ''])
    if a.constrained:
        result_file=Path(a.constrained)/'seed_102'/'result.json'
        if result_file.exists():
            r=json.loads(result_file.read_text())
            lines.extend([f"Repair termination: {r['solver_message']} (status {r['solver_status']}); {r['scipy_nfev']} solver evaluations and {r['residual_calls']} actual residual calls.", ''])
            if r['solver_status']==0:
                lines.extend(['The final repair exhausted its finite solver-evaluation budget. This is a budget-limited result, not evidence that this parameterization or optimizer has reached its attainable quality limit. Any continuing improvement in the saved history must be considered before diagnosing an optimization plateau.', ''])
    if a.final:
        audit_file=Path(a.final)/'seed_102'/'physics_rim_audit.json'
        if audit_file.exists():
            r=json.loads(audit_file.read_text());m=r['polynomial_root_metrics']
            lines.extend(['## Independent final-pupil audit','', f"The final aperture-only correction leaves every optimized gene unchanged and changes only the explicitly saved physical aperture radii. The previous candidate clipped 30 of 8,100 withheld rim/staggered-pupil rays. The corrected candidate receives {100*m['throughput']:.4f}% overall and at least {100*m['min_throughput']:.4f}% in every sampled slit/wavelength cell.", '', f"All-root polynomial tracing agrees with Newton tracing on {r['launched_rays']} rays, with {r['good_mask_disagreements']} good-mask differences and maximum common image discrepancy {r['max_common_image_difference_mm']:.3g} mm. The deliberately rim-heavy stress sample gives {1000*m['rms_spot_mm']:.2f} µm aggregate RMS spot, {1000*m['max_spot_mm']:.2f} µm worst-cell RMS spot, and {m['max_wfe_waves']:.3f} waves worst RMS wavefront error. This differs in weighting from the dense Cartesian pupil table and is not an area-uniform RMS estimate.", '', 'The exact physical circle radii, sampled curvature signs and conservative optical-surface separation bounds are recorded in the same audit JSON. These checks do not include mounts, substrates or manufacturing tolerances.', ''])
    if a.final:
        result_file=Path(a.final)/'seed_102'/'result.json'
        audit_file=Path(a.final)/'seed_102'/'physics_rim_audit.json'
        if result_file.exists() and audit_file.exists():
            r=json.loads(result_file.read_text());q=json.loads(audit_file.read_text())
            overview=['## Final corrected candidate','', '| Evaluation | Launched rays | Minimum cell support | Aggregate RMS spot | Worst-cell RMS spot | Worst-cell RMS WFE |', '|---|---:|---:|---:|---:|---:|']
            for label,n,m in [('Dense Cartesian pupil',3840,r['audit']),('Independent roots, rim-heavy stress',q['launched_rays'],q['polynomial_root_metrics'])]:
                overview.append(f"| {label} | {n} | {100*m['min_throughput']:.2f}% | {1000*m['rms_spot_mm']:.2f} µm | {1000*m['max_spot_mm']:.2f} µm | {m['max_wfe_waves']:.3f} waves |")
            overview.extend(['',f"The corrected optical-surface AABB is {r['geometry']['sampled_aabb_volume_cm3']:.2f} cm³; mounts and housing are excluded. The final grating pupil margin is approximately 50 µm, so manufacturing/tolerance robustness is not established. The paper’s approximately 0.07-wave target remains unmet.",'',f"[Final layout and spot map]({(Path(a.final)/'seed_102'/'candidate.png').resolve()}) · [Physical prescription]({(Path(a.final)/'seed_102'/'PRESCRIPTION.md').resolve()}) · [Independent audit]({audit_file.resolve()})",''])
            lines[4:4]=overview
    (out/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__': main()
