"""Bounded least-squares polishing after camera CMA, with separate provenance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.optimize import least_squares
from . import spectrometer_style_model as model
from .two_stage_search import write_json


def residuals(g, reference_good, parity=1, fields=5, pupils=7):
    """Fixed-length residual; initially transmitted rays cannot be dropped.

    Central-obscuration ghost rays have no image-quality target. Every launched
    ray still contributes to loss/support and aperture penalties. Image errors
    remain active for the fixed initial transmitted set even if a ray is lost.
    This is a local hybrid merit, not the original CMA objective.
    """
    with np.errstate(all='ignore'):
        r = model.trace(g, fields=fields, waves=1, pupils=pupils)
        good = r['good']
        ref = np.asarray(reference_good, bool)
        if ref.shape != good.shape or np.any(ref.sum(axis=2) == 0):
            raise ValueError('reference mask must match sampling and illuminate every field')
        xy = np.nan_to_num(r['xy'], nan=1e3, posinf=1e3, neginf=-1e3)
        count = ref.sum(axis=2)
        centroid = np.sum(np.where(ref[..., None], xy, 0), axis=2)/count[..., None]
        target = parity*np.tan(np.deg2rad(r['fields_deg']))[:, None, :]*model.radial.EFL_TARGET_MM
        mapping = (centroid-target)/model.PIXEL_PITCH_MM
        image = (xy-centroid[:, :, None])/model.PIXEL_PITCH_MM
        image += np.sqrt(.05)*mapping[:, :, None]
        # Fixed initial pupil normalization; changing survival cannot shrink the
        # image residual by removing an initially transmitted ray.
        normalizer = np.sqrt(count.size*count)[..., None, None]
        image = np.where(ref[..., None], image, 0)/normalizer
        n = good.size
        support = good.mean(axis=2)
        pieces = [image.ravel(), np.sqrt(2/n)*(~good).ravel(),
                  20/np.sqrt(n)*(ref & ~good).ravel(),
                  np.sqrt(80/support.size)*np.maximum(.55-support, 0).ravel(),
                  np.sqrt(150/support.size)*np.maximum(.50-support, 0).ravel(),
                  np.sqrt(5/n)*np.minimum(r['aperture_overflow_mm'], 100).ravel()]
        # Newly admitted rays are checked for quality too, without replacing the
        # original transmitted set or changing residual dimension.
        extra = np.where((good & ~ref)[..., None],
                         (xy-centroid[:, :, None])/model.PIXEL_PITCH_MM, 0)/np.sqrt(n)
        pieces.append(extra.ravel())
        m = r['model']
        geom = [10*max(m['optical_depth_mm']-10, 0)]
        for key in ('minimum_plate_thickness_mm', 'minimum_secondary_plate_gap_mm',
                    'minimum_primary_secondary_gap_mm'):
            geom.append(np.sqrt(1000)*max(.05-m[key], 0))
        for name, row in m['freeform_geometry']['surfaces'].items():
            geom.append(10*max(row['max_slope']-(.3 if name == 'plate' else 1.), 0))
        pieces.append(np.asarray(geom))
        return np.nan_to_num(np.concatenate(pieces), nan=1e4, posinf=1e4, neginf=-1e4)


def run(source, out, max_nfev=40, fields=5, pupils=7):
    source, out = Path(source), Path(out)
    original = json.loads(source.read_text())
    if original.get('representation') != 'spectrometer_style_xy':
        raise ValueError('expected a spectrometer_style_xy CMA result')
    genes = np.asarray(original['genes'], float)
    if genes.shape != (model.N_TOTAL,) or not np.isfinite(genes).all() or np.any(abs(genes)>1):
        raise ValueError('expected 41 finite normalized genes within [-1,1]')
    if max_nfev < 1 or fields < 1 or pupils < 1:
        raise ValueError('evaluation budget and sampling must be positive')
    before = model.evaluate(genes, fields=fields, waves=1, pupils=pupils)
    reference = model.trace(genes, fields=fields, waves=1, pupils=pupils)['good']
    parity = before['image_parity']
    initial = residuals(genes, reference, parity, fields, pupils)
    out.mkdir(parents=True, exist_ok=False)
    for filename, module in [('polish_snapshot.py', __file__),
                             ('model_snapshot.py', model.__file__),
                             ('radial_model_snapshot.py', model.radial.__file__)]:
        shutil.copyfile(module, out/filename)
    shutil.copyfile(source, out/'source_result.json')
    config = dict(source=str(source.resolve()), max_nfev=max_nfev,
                  training_sampling=dict(fields=fields,waves=1,pupils=pupils),
                  audit_sampling=dict(fields=7,waves=1,pupils=13),
                  method='trf', jac='2-point', diff_step=1e-4, x_scale='jac',
                  bounds=[-1,1], ftol=1e-8, xtol=1e-8, gtol=1e-8)
    write_json(out/'configuration.json', config)
    started = time.monotonic()
    calls = 0
    history = []
    def fun(g):
        nonlocal calls
        value = residuals(g, reference, parity, fields, pupils)
        calls += 1
        if calls % 100 == 0:
            row = dict(residual_calls=calls, probe_sum_squares=float(value@value),
                       elapsed_seconds=time.monotonic()-started)
            history.append(row)
            write_json(out/'checkpoint.json', dict(**row, probe_genes=g, history=history,
                       warning='May be a numerical Jacobian probe, not an accepted iterate'))
            print(json.dumps(row), flush=True)
        return value
    opt = least_squares(fun, np.clip(genes,-1+1e-10,1-1e-10),
                        bounds=(-1,1),method='trf',jac='2-point',diff_step=1e-4,
                        x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=max_nfev)
    # Select only the solver's returned iterate or the original source, never
    # a finite-difference probe. This decision uses training residual alone.
    final_vector = residuals(opt.x, reference, parity, fields, pupils)
    initial_cost, solver_cost = float(initial@initial), float(final_vector@final_vector)
    accepted = bool(solver_cost <= initial_cost)
    final = opt.x if accepted else genes
    train = config['training_sampling']; audit = config['audit_sampling']
    final_trace = model.trace(final, **train)
    result = dict(seed=original.get('seed'), representation='spectrometer_style_xy',
                  algorithm='bounded least-squares after CMA; different hybrid merit',
                  source=str(source.resolve()), genes=final, decoded=model.decode(final),
                  configuration=config, train_sampling=train, audit_sampling=audit,
                  initial_sum_squares=initial_cost, solver_sum_squares=solver_cost,
                  final_sum_squares=solver_cost if accepted else initial_cost,
                  accepted_solver_iterate=accepted, solver_status=int(opt.status),
                  solver_message=str(opt.message), scipy_nfev=opt.nfev, scipy_njev=opt.njev,
                  solver_residual_calls=calls, total_residual_calls=calls+2,
                  residual_count=len(initial), initial_transmitted_mask=reference,
                  initially_transmitted_rays_lost=int(np.sum(reference & ~final_trace['good'])),
                  elapsed_seconds=time.monotonic()-started,
                  before_training=before, training=model.evaluate(final, **train),
                  before_audit=model.evaluate(genes, **audit), audit=model.evaluate(final, **audit),
                  source_audit=model.evaluate(genes, **audit), history=[], probe_history=history,
                  stages=[], interpretation='Local fixed-reference-pupil polish; hard clipping is nonsmooth. '
                  'Audits use all launched rays. No diffraction, dispersion or packaging validation.')
    write_json(out/'result.json',result)
    print(json.dumps({k:result[k] for k in ('solver_status','solver_message','scipy_nfev',
          'solver_residual_calls','initial_sum_squares','final_sum_squares',
          'initially_transmitted_rays_lost')}),flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--out',required=True)
    p.add_argument('--max-nfev',type=int,default=40)
    p.add_argument('--fields',type=int,default=5); p.add_argument('--pupils',type=int,default=7)
    a = p.parse_args()
    run(a.source,a.out,a.max_nfev,a.fields,a.pupils)
