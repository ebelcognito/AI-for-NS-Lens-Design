"""Per-ray bounded least-squares spot refinement: conventional TRA baseline.

Shapes only, fixed poses/apertures/detector and frozen incident rays. The same
ray-coordinate interface can later host GTRA retargeting. This is not a claim
of implementing general task-loss GTRA.
"""
from copy import deepcopy
import time
import numpy as np
from scipy.optimize import least_squares
from . import model3d as model


def residual_vector(xy, targets, centroid_weight=3.):
    """Centered ray errors plus explicit fixed-image-mapping anchor, all in mm."""
    fields,pupils,_=xy.shape
    centers=xy.mean(axis=1)
    spot=(xy-centers[:,None,:])/np.sqrt(fields*pupils)
    mapping=centroid_weight*(centers-targets)/np.sqrt(fields)
    return np.r_[spot.ravel(),mapping.ravel()]


def refine(layout,fields,pupils,max_nfev=60,max_degree=4):
    start=time.perf_counter();base=deepcopy(layout)
    selected=[(j,k) for j in range(3) for k,p in enumerate(model.POWERS) if sum(p)<=max_degree]
    # One normalized unit changes sag by one mm at the 30 mm basis radius.
    x0=np.array([base['surfaces'][j]['coefficients'][k] for j,k in selected])
    span=np.array([2. if sum(model.POWERS[k])==2 else 2. for j,k in selected])
    # Launch upstream of every surface allowed by these coefficient bounds.
    envelope=deepcopy(base)
    for (j,k),x,w in zip(selected,x0,span):envelope['surfaces'][j]['coefficients'][k]=abs(x)+w
    origins,directions=model.launch_rays(envelope,fields,pupils)
    initial=model.trace_rays(base,origins,directions)
    if not initial['good'].all():raise ValueError('Full initial training-ray support required')
    target=initial['image_mm'].reshape(len(fields),len(pupils),2).mean(axis=1)
    route=initial['route'].copy();calls=0
    def decode(x):
        m=deepcopy(base)
        for (j,k),value in zip(selected,x):m['surfaces'][j]['coefficients'][k]=value
        return m
    def fun(x):
        nonlocal calls
        calls+=1
        r=model.trace_rays(decode(x),origins,directions)
        xy=np.nan_to_num(r['image_mm'],nan=100.,posinf=100.,neginf=-100.).reshape(len(fields),len(pupils),2)
        residual=residual_vector(xy,target)
        invalid=~r['good']|np.any(r['route']!=route,axis=1)
        return np.r_[residual,100.*invalid/np.sqrt(len(origins))]
    initial_residual=fun(x0)
    opt=least_squares(fun,x0,bounds=(x0-span,x0+span),method='trf',jac='3-point',
                      x_scale='jac',diff_step=1e-5,max_nfev=max_nfev,
                      ftol=1e-8,xtol=1e-8,gtol=1e-8)
    final=decode(opt.x);r=model.trace_rays(final,origins,directions)
    accepted=bool(r['good'].all() and np.array_equal(r['route'],route)
                  and 2*opt.cost<float(initial_residual@initial_residual))
    if not accepted:final=base
    return final,dict(accepted=accepted,parameters=len(selected),active_terms=[model.POWERS[k] for j,k in selected if j==0],
        residual_calls=calls,nfev=int(opt.nfev),njev=int(opt.njev),solver_status=int(opt.status),solver_message=opt.message,
        initial_sum_squares=float(initial_residual@initial_residual),final_sum_squares=float(2*opt.cost),
        elapsed_seconds=time.perf_counter()-start,centroid_weight=3.,coefficient_change_bound_mm=2.,
        training_fields=len(fields),training_pupils=len(pupils),initial_parameters=x0.tolist(),final_parameters=opt.x.tolist(),
        fixed='mirror poses, detector pose, aperture radii, entrance pupil, field angles',method='scipy bounded trust-region reflective least squares; central finite-difference per-ray Jacobian')
