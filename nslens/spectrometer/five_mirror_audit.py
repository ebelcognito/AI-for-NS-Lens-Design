"""Heldout, fixed-aperture checks for six-surface spectrometer candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import model
from .four_mirror_audit import (dense_pupil, audit_paths, detector_obstructions,
    covered_surface_separation, image_metrics)
from .physics_audit import exact_intersect
from .search import serial


def curvature_audit(layout):
    """Independent Hessian construction; inherited primary/grating/tertiary signs."""
    pupil = dense_pupil(65, 360)
    gi = int(layout['grating_index'])
    rows = []
    for index, surface in enumerate(layout['surfaces']):
        x, y = (pupil * surface['aperture']).T
        h = np.zeros((len(x), 2, 2))
        for c, (i, j) in zip(surface['coeff'], model.TERMS):
            if i >= 2: h[:, 0, 0] += c*i*(i-1)*x**(i-2)*y**j
            if i and j: h[:, 0, 1] += c*i*j*x**(i-1)*y**(j-1)
            if j >= 2: h[:, 1, 1] += c*j*(j-1)*x**i*y**(j-2)
        h[:, 1, 0] = h[:, 0, 1]
        eig = np.linalg.eigvalsh(h)
        # Insertion can separate the primary or tertiary from the grating.
        segment = int(layout['insertion_segment'])
        old_index = None if index == segment else index - int(index > segment)
        expected = 1 if old_index == 2 else -1 if old_index in (1, 3) else 0
        signed_min = float(np.linalg.eigvalsh(expected*h).min()) if expected else None
        rows.append(dict(surface=index, original_surface_index=old_index,
            is_grating=index == gi, expected_sign=expected,
            minimum_expected_signed_hessian_per_mm=signed_min,
            sign_preserved=signed_min is None or signed_min > 0,
            minimum_hessian_per_mm=float(eig.min()),maximum_hessian_per_mm=float(eig.max()),
            nonzero_power=bool(np.max(np.abs(eig)) > 1e-10)))
    return dict(surfaces=rows,all_expected_signs_preserved=all(r['sign_preserved'] for r in rows),
        all_five_mirrors_powered=all(r['nonzero_power'] for r in rows if not r['is_grating']),
        limitation='Dense disk/rim Hessian samples, not a continuous curvature proof.')


def audit(genes, powered, hardware_constraints=False):
    dense = powered.trace(genes, fields=13, waves=17, pupil_xy=dense_pupil(5,44))
    layout=dense['model']
    result=dict(genes=np.asarray(genes).tolist(),dense_sampling=dict(fields=13,waves=17,pupil_points=177),
        dense_launched_rays=int(dense['good'].size),dense_received_rays=int(dense['good'].sum()),
        dense_image_metrics=image_metrics(dense),curvature=curvature_audit(layout),
        covered_surface_separation=covered_surface_separation(layout),
        fixed_aperture_radii_mm=[s['aperture'] for s in layout['surfaces']])
    sampling=dict(fields=5,waves=7,pupil_xy=dense_pupil(5,24))
    nominal=powered.trace(genes,**sampling)
    exact=powered.trace(genes,intersect_fn=exact_intersect,**sampling)
    paths=audit_paths(exact['paths'],layout['surfaces'],grating_index=layout['grating_index'],wavelengths=exact['wavelength'])
    detectors=detector_obstructions(exact['paths'],layout)
    common=nominal['good'] & exact['good']
    error=float(np.max(np.abs(nominal['xy'][common]-exact['xy'][common]))) if common.any() else None
    result['polynomial_audit']=dict(launched_rays=int(exact['good'].size),received_rays=int(exact['good'].sum()),
        mask_disagreements=int(np.sum(nominal['good'] != exact['good'])),maximum_image_error_mm=error,
        independent_paths=paths,early_detector_interceptions=detectors)
    result['passed']=bool(dense['good'].all() and exact['good'].all() and np.array_equal(nominal['good'],exact['good'])
        and error is not None and error < 1e-7 and paths['accepted_rays']==exact['good'].size and detectors['blocked_rays']==0
        and result['curvature']['all_expected_signs_preserved']
        and result['covered_surface_separation']['all_optical_surfaces_and_detector_disjoint_certified'])
    from .entrance_access import upstream_access
    result['entrance_access_20mm']=upstream_access(layout,distance_mm=20.,fields=41,pupil_rings=8,pupil_azimuths=48,exact=True)
    result['internal_optical_audit_passed']=result['passed']
    result['passed']=bool(result['passed'] and result['entrance_access_20mm']['passed'])
    if hardware_constraints:
        from .slit_hardware import hardware_check
        result['slit_hardware']=hardware_check(layout,paths=dense['paths'],dense=True)
        result['passed']=bool(result['passed'] and result['slit_hardware']['passed'])
    result['limitation']='Geometric optical validation including sampled 20mm upstream illumination clearance only. All-five-powered status reported separately; flat added mirror is permitted. Excludes substrates, mounts, coating losses, tolerances, diffraction; the 20mm input corridor is a modeling assumption, not a specified feed assembly.'
    return serial(result)


def main():
    from . import powered_five_mirror as powered
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--out');a=p.parse_args()
    source=Path(a.source);data=json.loads(source.read_text())
    configuration=data.get('insertion_configuration',data.get('configuration',{}))
    powered.configure(configuration['segment'],configuration['aperture'])
    g=data.get('genes',data.get('best',{}).get('genes'))
    result=audit(g,powered,hardware_constraints=data.get('configuration',{}).get('hardware_constraints',False));result['source']=str(source);result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    output=Path(a.out) if a.out else source.parent/'independent_audit.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','dense_launched_rays','dense_received_rays','dense_image_metrics')},indent=2))

if __name__=='__main__':main()
