"""Conservative slit-assembly keep-out and sampled illumination-access gates.

The reserved box is not an optical surface. Only the initial launch leg may
traverse it. Subsequent segments must avoid its entire closed volume, including
the opening, to prevent a returning beam from hitting slit hardware.
"""
import numpy as np
from . import model
from .entrance_access import upstream_access

BOX_LO=np.array([-6.,-7.,-1.])
BOX_HI=np.array([6.,7.,1.])
RAY_MARGIN_MM=.25
SETTINGS=dict(plate_width_mm=10.,plate_height_mm=12.,slit_width_mm=.025,slit_length_mm=10.,
    reserved_box_low_mm=BOX_LO.tolist(),reserved_box_high_mm=BOX_HI.tolist(),
    later_ray_margin_mm=RAY_MARGIN_MM,upstream_clearance_mm=20.,
    assumptions='Illustrative knife-edge slit at z=0. Reserved plate/holder volume, not a detailed solid mount. Initial launch passes through slit; later legs avoid entire reserved volume. Finite slit width not sampled optically.')


def keepout_corners():
    return np.array([[x,y,z] for x in [BOX_LO[0],BOX_HI[0]] for y in [BOX_LO[1],BOX_HI[1]] for z in [BOX_LO[2],BOX_HI[2]]])


def segments_intersect_box(start,end,low=BOX_LO,high=BOX_HI):
    """Closed continuous segment/AABB slab intersection; tangencies are hits."""
    start,end=np.broadcast_arrays(np.asarray(start,float),np.asarray(end,float))
    direction=end-start;lower=np.zeros(start.shape[:-1]);upper=np.ones(start.shape[:-1]);valid=np.ones(lower.shape,bool)
    for axis in range(3):
        parallel=np.abs(direction[...,axis])<1e-14
        valid &=~parallel|((start[...,axis]>=low[axis])&(start[...,axis]<=high[axis]))
        denom=np.where(parallel,1.,direction[...,axis])
        a=(low[axis]-start[...,axis])/denom;b=(high[axis]-start[...,axis])/denom
        lower=np.maximum(lower,np.where(parallel,-np.inf,np.minimum(a,b)))
        upper=np.minimum(upper,np.where(parallel,np.inf,np.maximum(a,b)))
    finite=np.isfinite(start).all(axis=-1)&np.isfinite(end).all(axis=-1)
    return (~finite)|(valid&(lower<=upper))


def _distance_to_box(points):
    return np.linalg.norm(np.maximum(np.maximum(BOX_LO-points,points-BOX_HI),0.),axis=-1)


def _monomial_max(i,j,r):
    if i+j==0:return 1.
    return r**(i+j)*(i/(i+j))**(i/2)*(j/(i+j))**(j/2)


def surface_box_clearance(layout,dense=False):
    """Conservative whole-mask distance lower bounds using a sampling cover.

    A disk-grid covering radius times an analytic sag Lipschitz bound covers
    every point, so positive lower bounds certify separation from the box.
    Nonpositive is treated as rejection, even when inconclusive.
    """
    nr,na=(48,192) if dense else (8,48)
    angles=np.arange(na)*2*np.pi/na;rows=[]
    for index,s in enumerate(layout['surfaces']):
        radius=float(s['aperture']);rad=np.linspace(0,radius,nr+1)
        x=(rad[:,None]*np.cos(angles)).ravel();y=(rad[:,None]*np.sin(angles)).ravel()
        z=model.sag(x,y,s['coeff'])[0]
        cloud=np.column_stack([x,y,z])@s['basis'].T+s['vertex']
        dx=sum(abs(c*i)*_monomial_max(i-1,j,radius) for c,(i,j) in zip(s['coeff'],model.TERMS) if i)
        dy=sum(abs(c*j)*_monomial_max(i,j-1,radius) for c,(i,j) in zip(s['coeff'],model.TERMS) if j)
        cover=(radius/(2*nr)+2*radius*np.sin(np.pi/(2*na)))*np.sqrt(1+dx*dx+dy*dy)
        gap=float(_distance_to_box(cloud).min()-cover)
        rows.append(dict(surface=index,gap_lower_bound_mm=gap,cover_radius_mm=float(cover)))
    bounds=layout['detector_bounds'];u,v=np.meshgrid(np.linspace(*bounds[:,0],nr+1),np.linspace(*bounds[:,1],nr+1))
    cloud=layout['image']+u.ravel()[:,None]*layout['image_x']+v.ravel()[:,None]*layout['image_y']
    cover=float(np.linalg.norm(bounds[1]-bounds[0])/(2*nr))
    rows.append(dict(surface='detector',gap_lower_bound_mm=float(_distance_to_box(cloud).min()-cover),cover_radius_mm=cover))
    return dict(passed=all(np.isfinite(r['gap_lower_bound_mm']) and r['gap_lower_bound_mm']>0 for r in rows),
        minimum_gap_lower_bound_mm=min(r['gap_lower_bound_mm'] for r in rows),surfaces=rows,
        limitation='Analytic covering bounds in floating point, not interval arithmetic. Physical optical masks only, not mirror substrates or mounts.')


def hardware_check(layout,paths=None,dense=False):
    if paths is None:paths=layout.get('nominal_paths')
    if paths is None:raise ValueError('Actual or nominal ray paths required for slit hardware check')
    paths=np.asarray(paths)
    if paths.ndim!=3 or paths.shape[0]!=len(layout['surfaces'])+2:raise ValueError('Expected paths (Nsurfaces+2, Nrays, 3)')
    # Initial slit-to-first-mirror launch is the only intended access to this box.
    hit=segments_intersect_box(paths[1:-1],paths[2:],BOX_LO-RAY_MARGIN_MM,BOX_HI+RAY_MARGIN_MM)
    blocked=np.any(hit,axis=0);surface=surface_box_clearance(layout,dense=dense)
    access=upstream_access(layout,distance_mm=20.,fields=13 if dense else 5,pupil_rings=5 if dense else 2,pupil_azimuths=32 if dense else 12)
    fraction=float(blocked.mean());gap=surface['minimum_gap_lower_bound_mm']
    passed=bool(not blocked.any() and surface['passed'] and access['passed'])
    penalty=50*fraction+50*access['blocked_fraction']+10*max(.05-gap,0.)**2
    if not passed:penalty+=1.
    if not np.isfinite(penalty):penalty=1e6
    return dict(passed=passed,loss_penalty=float(penalty),later_ray_count=blocked.size,later_blocked_rays=int(blocked.sum()),later_blocked_fraction=fraction,
        segments=[dict(segment=i+1,blocked_rays=int(row.sum())) for i,row in enumerate(hit)],surface_box=surface,upstream=access,settings=SETTINGS)
