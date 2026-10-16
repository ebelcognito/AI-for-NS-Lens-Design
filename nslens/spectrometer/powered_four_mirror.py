"""Coarse-to-fine powered four-mirror refinement around inherited optical scaffolds.

All five optical surfaces retain finite physical masks. Added mirror curvatures
are free in sign. The original three surfaces retain their signed-curvature
architecture. This is seeded refinement, not prescription-free discovery.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import model, four_mirror as fold

N_BASE=28
N_TOTAL=63
SOURCE=fold.DEFAULT_BASELINE
REFERENCE=np.asarray(json.loads(Path(SOURCE).read_text())['genes'])
APERTURES=np.array([9.882,8.9,8.43,12.11,9.169])
# Changes in original normalized coordinates: turns +-8.75deg; lengths
# +-6/7.5mm; quadratic coefficient changes +-.00348/mm; detector slope +-.0625.
BASE_SCALE=np.r_[np.full(3,.35),np.full(4,.30),np.full(6,.20),np.full(2,.25)]
FOLD_LOW=np.array([.2,45.,0.,.2,45.,0.])
FOLD_HIGH=np.array([.85,135.,360.,.85,135.,360.])
HIGH_SCALE=np.array([2e-4,2e-4,8e-6,8e-6,8e-6,2e-7,2e-7])

def seed_genes(params, rotation_xyz_deg=(0,0,0)):
    g=np.zeros(N_TOTAL)
    g[15:21]=2*(np.asarray(params)-FOLD_LOW)/(FOLD_HIGH-FOLD_LOW)-1
    g[25:28]=np.asarray(rotation_xyz_deg)/90
    return g


def decode(g):
    g=np.pad(np.asarray(g,float),(0,max(0,N_TOTAL-len(g))))
    optical=REFERENCE.copy();optical[:15]+=g[:15]*BASE_SCALE
    # Original freeforms remain fixed during coarse pose/curvature refinement.
    optical[15:36]+=g[28:49]*.5
    m=model.decode(optical,aperture_radii=APERTURES[1:4])
    # Only the central scaffold is needed to construct surface poses. Physical
    # apertures are fixed, never inferred from received optimized rays.
    chief=np.array([m['origin'],*[s['vertex'] for s in m['surfaces']],m['image']])[:,None,:]
    scaffold=dict(model=m,trace=dict(paths=chief),detector_bounds=np.array([[-4.2,-6.2],[4.2,6.2]]))
    params=FOLD_LOW+(g[15:21]+1)*.5*(FOLD_HIGH-FOLD_LOW)
    layout=fold.build_layout(params,scaffold)
    for s,a in zip(layout['surfaces'],APERTURES):s['aperture']=float(a)
    for j,index in enumerate([0,4]):
        layout['surfaces'][index]['coeff']=np.r_[.012*g[21+2*j:23+2*j],g[49+7*j:56+7*j]*HIGH_SCALE*.5]
    layout['original_genes']=optical;layout['input_length']=float(m['lengths'][0]);layout['rotation_xyz_deg']=90*g[25:28]
    layout['genes']=g
    return layout


def trace(g,fields=3,waves=3,pupils=5,pupil_xy=None,intersect_fn=None):
    m=decode(g);fn=model.intersect if intersect_fn is None else intersect_fn
    if pupil_xy is None:
        u=np.linspace(-1,1,pupils);px,py=np.meshgrid(u,u);keep=px*px+py*py<=1+1e-12;px=px[keep];py=py[keep]
    else:
        pupil_xy=np.asarray(pupil_xy,float)
        if pupil_xy.ndim!=2 or pupil_xy.shape[1]!=2 or not len(pupil_xy) or not np.isfinite(pupil_xy).all() or np.any(np.sum(pupil_xy*pupil_xy,axis=1)>1+1e-12):raise ValueError('Invalid pupil disk points')
        px,py=pupil_xy.T
    npup=len(px);slit,wave=np.meshgrid(np.linspace(-5,5,fields),np.linspace(*model.WAVELENGTHS,waves),indexing='ij')
    slit=np.repeat(slit.ravel(),npup);wave=np.repeat(wave.ravel(),npup)
    o=np.column_stack([np.zeros(len(slit)),slit,np.zeros(len(slit))]);L=m['input_length']
    d=model.unit(np.array([0,0,L])-o+np.column_stack([np.tile(px,fields*waves),np.tile(py,fields*waves),np.zeros(len(slit))])*L/7.6)
    good=np.ones(len(o),bool);obstruction=np.zeros(len(o),bool);opl=np.zeros(len(o));phase=np.zeros(len(o));history=[o.copy()]
    for j,s in enumerate(m['surfaces']):
        p,n,t,valid=fn(o,d,s)
        for k,other in enumerate(m['surfaces']):
            if k==j:continue
            _,_,ot,ov=fn(o,d,other);obstruction|=good&ov&(ot<t-1e-5)
        dt,dv=fold.detector_intersection(o,d,m);obstruction|=good&dv&(dt<t-1e-5)
        good &= valid;opl+=np.where(valid,t,0)
        if j==2:
            local=(p-s['vertex'])@s['basis'];phase=model.grating_phase_length(local[:,0],wave)
            d,order=model.diffract(d,n,s['basis'][:,0],wave);good &=order
        else:d=model.reflect(d,n)
        o=p;history.append(p.copy())
    den=d@m['image_normal'];t=((m['image']-o)@m['image_normal'])/np.where(abs(den)>1e-12,den,1e-12)
    for s in m['surfaces']:
        _,_,ot,ov=fn(o,d,s);obstruction|=good&ov&(ot<t-1e-5)
    image=o+t[:,None]*d;history.append(image)
    xy=np.column_stack([(image-m['image'])@m['image_x'],(image-m['image'])@m['image_y']])
    bounds=m['detector_bounds'];good &=np.isfinite(t)&(t>0)&~obstruction&np.all((xy>=bounds[0])&(xy<=bounds[1]),axis=1)
    shape=(fields,waves,npup)
    return dict(xy=xy.reshape(*shape,2),good=good.reshape(shape),opl=opl.reshape(shape),phase=phase.reshape(shape),last_surface=o.reshape(*shape,3),image=image.reshape(*shape,3),wavelength=wave.reshape(shape),slit=slit.reshape(shape),model=m,paths=np.array(history),obstructed=obstruction.reshape(shape))


def geometry_metrics(g,samples=9):
    from .four_mirror_search import package_metrics
    m=decode(g)
    package=package_metrics(m,m['rotation_xyz_deg'],samples=samples)
    geometry=fold.geometry_metrics(m,samples=samples)
    return dict(**package,min_surface_gap_mm=geometry['min_surface_gap_mm'],min_detector_gap_mm=geometry['min_detector_gap_mm'],aperture_radii_mm=APERTURES.tolist())


def curvature_metrics(layout,radial_samples=5,azimuth_samples=24):
    ang=np.arange(azimuth_samples)*2*np.pi/azimuth_samples;rad=np.linspace(0,1,radial_samples)
    x=(rad[:,None]*np.cos(ang)).ravel();y=(rad[:,None]*np.sin(ang)).ravel()
    sign_violations=[];max_slope=0.;max_sag=0.;max_curve=0.
    for index,s in enumerate(layout['surfaces']):
        xx=x*s['aperture'];yy=y*s['aperture'];z,zx,zy=model.sag(xx,yy,s['coeff']);h=model.sag_hessian(xx,yy,s['coeff'])
        eig=np.linalg.eigvalsh(h)
        max_slope=max(max_slope,float(np.hypot(zx,zy).max()));max_sag=max(max_sag,float(abs(z).max()));max_curve=max(max_curve,float(abs(eig).max()))
        if index in (1,2,3):
            sign=1 if index==2 else -1
            sign_violations.extend(np.maximum(2e-6-np.linalg.eigvalsh(sign*h),0).ravel())
    return dict(sign_violation=float(np.max(sign_violations)),max_slope=max_slope,max_abs_sag_mm=max_sag,max_abs_hessian_per_mm=max_curve)


def evaluate(g,fields=3,waves=3,pupils=5,pupil_xy=None):
    with np.errstate(all='ignore'):
        r=trace(g,fields,waves,pupils,pupil_xy=pupil_xy);stats=fold.image_metrics(r)
        geo=geometry_metrics(g,samples=7);curvature=curvature_metrics(r['model'])
        empty=float(np.mean(~np.any(r['good'],axis=2)))
        gap=min(geo['min_surface_gap_mm'],geo['min_detector_gap_mm'])
        loss=(stats['rms_spot_mm']+30*(1-stats['throughput'])+10*empty+
              .2*(stats['dispersion_mm_per_um']-10)**2+.5*(stats['slit_magnification']-1)**2+
              .03*geo['volume_cm3']/60+2*max(2-gap,0)**2+
              1e5*curvature['sign_violation']**2+10*max(curvature['max_slope']-.7,0)**2+max(curvature['max_abs_sag_mm']-4,0)**2)
        if not np.isfinite(loss):loss=1e6
    return dict(**stats,loss=float(loss),volume_cm3=geo['volume_cm3'],sampled_min_gap_mm=float(gap),curvature=curvature)


def residuals(g,fields=5,waves=7,pupils=7,pupil_xy=None):
    r=trace(g,fields,waves,pupils,pupil_xy=pupil_xy);xy=r['xy'];n=r['good'].size
    target=np.stack([10000*(r['wavelength']-.0008),-r['slit']],axis=-1)
    error=xy-target;centroid=error.mean(axis=2,keepdims=True)
    error=error-centroid+.1*(centroid-centroid.mean(axis=(0,1,2),keepdims=True))
    pieces=[error.ravel(),10*(~r['good']).ravel()]
    span=np.mean(xy[:,-1,:,0]-xy[:,0,:,0]);pieces.append(np.array([3*np.sqrt(n)*(span-6)]))
    for j,s in enumerate(r['model']['surfaces']):
        local=(r['paths'][j+1]-s['vertex'])@s['basis'];pieces.append(3*np.maximum(np.linalg.norm(local[:,:2],axis=1)-s['aperture'],0))
    curv=curvature_metrics(r['model']);geo=geometry_metrics(g,samples=7)
    pieces.append(np.sqrt(n)*np.array([1000*curv['sign_violation'],max(curv['max_slope']-.7,0),.1*max(geo['volume_cm3']/60-1,0),max(2-min(geo['min_surface_gap_mm'],geo['min_detector_gap_mm']),0)]))
    return np.nan_to_num(np.concatenate(pieces)/np.sqrt(n),nan=1e3,posinf=1e3,neginf=-1e3)
