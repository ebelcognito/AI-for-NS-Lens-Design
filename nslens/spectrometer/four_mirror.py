"""Exact isometric flat-fold extension of the three-surface spectrometer.

Reflections transform all basis columns, including the grating phase axis.
A left-handed basis is intentional: reconstructing it with cross products
would change the prescription. Envelope and clearance are sampled optical
surface diagnostics, and exclude substrates and mounts.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from . import model

DEFAULT_BASELINE='experiments/spectrometer_final_v1/seed_102/result.json'

def load_baseline(path=DEFAULT_BASELINE,fields=5,waves=7,pupils=9,pupil_xy=None):
    row=json.loads(Path(path).read_text())
    apertures=row.get('configuration',{}).get('aperture_radii',[8.9,8.43,12.11])
    r=model.trace(row['genes'],fields,waves,pupils,pupil_xy=pupil_xy,aperture_radii=apertures)
    xy=r['xy'].reshape(-1,2)
    return dict(genes=row['genes'],aperture_radii=apertures,model=r['model'],trace=r,
                detector_bounds=np.array([xy.min(axis=0)-1,xy.max(axis=0)+1]),source=str(path))

def _flat(p,d,turn,azimuth):
    # Azimuth zero lies in global xz plane for the baseline incoming ray.
    ref=np.array([0.,1.,0.]); e=model.unit(np.cross(ref,d))
    if np.linalg.norm(e)<.5:e=model.unit(np.cross([1.,0.,0.],d))
    q=np.cross(d,e); a=np.deg2rad(azimuth); t=np.deg2rad(turn)
    out=np.cos(t)*d+np.sin(t)*(np.cos(a)*e+np.sin(a)*q)
    n=model.unit(d-out); x=model.unit(np.cross(ref,n))
    if np.linalg.norm(x)<.5:x=model.unit(np.cross([1.,0.,0.],n))
    b=np.column_stack([x,np.cross(n,x),n])
    return dict(vertex=p,basis=b,coeff=np.zeros(9),aperture=1.),np.eye(3)-2*np.outer(n,n)

def _point(p,center,Q):return (p-center)@Q.T+center

def _surface(s,center,Q):
    return dict(vertex=_point(s['vertex'],center,Q),basis=Q@s['basis'],coeff=s['coeff'].copy(),aperture=s['aperture'])

def _crossing(a,b,s):
    d=b-a;n=s['basis'][:,2];den=d@n
    f=((s['vertex']-a)@n)/np.where(abs(den)>1e-14,den,1e-14)
    return a+f[:,None]*d,f

def build_layout(params,baseline):
    f1,t1,a1,f2,t2,a2=np.asarray(params,float);m=baseline['model'];h=baseline['trace']['paths']
    p1=m['origin']+f1*(m['surfaces'][0]['vertex']-m['origin'])
    flat1,Q1=_flat(p1,model.unit(m['surfaces'][0]['vertex']-m['origin']),t1,a1)
    surfaces=[_surface(s,p1,Q1) for s in m['surfaces']]
    image1=_point(m['image'],p1,Q1)
    p2=surfaces[-1]['vertex']+f2*(image1-surfaces[-1]['vertex'])
    flat2,Q2=_flat(p2,model.unit(image1-surfaces[-1]['vertex']),t2,a2)
    cross1,frac1=_crossing(h[0],h[1],flat1)
    transformed=_point(h[1:],p1,Q1)
    cross2,frac2=_crossing(transformed[-2],transformed[-1],flat2)
    for s,points in [(flat1,cross1),(flat2,cross2)]:
        local=(points-s['vertex'])@s['basis'];s['aperture']=float(np.max(np.linalg.norm(local[:,:2],axis=1))+1.)
    return dict(surfaces=[flat1,*surfaces,flat2],image=_point(image1,p2,Q2),
                image_x=Q2@Q1@m['image_x'],image_y=Q2@Q1@m['image_y'],image_normal=Q2@Q1@m['image_normal'],
                origin=m['origin'].copy(),params=np.asarray(params),detector_bounds=baseline['detector_bounds'].copy(),
                nominal_paths=np.array([h[0],cross1,*transformed[:-1],cross2,_point(transformed[-1],p2,Q2)]),
                nominal_segment_valid=bool(np.all((frac1>0)&(frac1<1)&(frac2>0)&(frac2<1))),
                fold_fractions=np.array([frac1,frac2]))

def surface_cloud(surface,samples=17):
    u=np.linspace(-1,1,samples);x,y=np.meshgrid(u,u);ok=x*x+y*y<=1+1e-12
    theta=np.linspace(0,2*np.pi,max(128,8*samples),endpoint=False)
    x=np.r_[x[ok],np.cos(theta)]*surface['aperture'];y=np.r_[y[ok],np.sin(theta)]*surface['aperture']
    z,_,_=model.sag(x,y,surface['coeff'])
    return np.column_stack([x,y,z])@surface['basis'].T+surface['vertex']

def geometry_metrics(layout,samples=17):
    clouds=[surface_cloud(s,samples) for s in layout['surfaces']]
    bounds=layout['detector_bounds'];corners=np.array([[x,y] for x in bounds[:,0] for y in bounds[:,1]])
    detector=layout['image']+corners[:,0,None]*layout['image_x']+corners[:,1,None]*layout['image_y']
    slit=layout['origin']+np.array([[0,-5,0],[0,5,0]])
    allpoints=np.concatenate([*clouds,detector,slit]);lo=allpoints.min(axis=0);hi=allpoints.max(axis=0)
    dg=np.array([[x,y] for x in np.linspace(*bounds[:,0],samples) for y in np.linspace(*bounds[:,1],samples)])
    dc=layout['image']+dg[:,0,None]*layout['image_x']+dg[:,1,None]*layout['image_y']
    detector_gaps=[float(cKDTree(c).query(dc)[0].min()) for c in clouds]
    gaps=[float(cKDTree(a).query(b)[0].min()) for j,a in enumerate(clouds) for b in clouds[j+1:]]
    return dict(bbox_mm=(hi-lo).tolist(),bbox_min_mm=lo.tolist(),bbox_max_mm=hi.tolist(),volume_cm3=float(np.prod(hi-lo)/1000),
                min_surface_gap_mm=min(gaps),min_detector_gap_mm=min(detector_gaps),surface_detector_gaps_mm=detector_gaps,surface_pair_gaps_mm=gaps,aperture_radii_mm=[s['aperture'] for s in layout['surfaces']],
                limitation='Sampled finite optical surfaces including rims, slit ends and detector rectangle; no mounts or substrates. Sampled gaps are upper bounds on true minimum gaps.')

def trace(layout,baseline,fields=5,waves=7,pupils=9,pupil_xy=None,intersect_fn=None):
    fn=model.intersect if intersect_fn is None else intersect_fn
    source=model.trace(baseline['genes'],fields,waves,pupils,pupil_xy=pupil_xy,aperture_radii=baseline['aperture_radii'])
    shape=source['good'].shape;wave=source['wavelength'].ravel();o=source['paths'][0].copy()
    d=model.unit(source['paths'][1]-o);good=np.ones(len(o),bool);obstruction=np.zeros(len(o),bool)
    opl=np.zeros(len(o));phase=np.zeros(len(o));history=[o.copy()]
    for j,s in enumerate(layout['surfaces']):
        p,n,t,valid=fn(o,d,s)
        for k,other in enumerate(layout['surfaces']):
            if k==j:continue
            _,_,ot,ov=fn(o,d,other);obstruction|=good&ov&(ot<t-1e-5)
        dt,dv=detector_intersection(o,d,layout)
        obstruction|=good&dv&(dt<t-1e-5)
        good &=valid;opl+=np.where(valid,t,0)
        if j==2:
            local=(p-s['vertex'])@s['basis'];phase=model.grating_phase_length(local[:,0],wave)
            d,valid_order=model.diffract(d,n,s['basis'][:,0],wave);good &=valid_order
        else:d=model.reflect(d,n)
        o=p;history.append(p.copy())
    den=d@layout['image_normal'];t=((layout['image']-o)@layout['image_normal'])/np.where(abs(den)>1e-12,den,1e-12)
    for other in layout['surfaces']:
        _,_,ot,ov=fn(o,d,other);obstruction|=good&ov&(ot<t-1e-5)
    image=o+t[:,None]*d;history.append(image)
    xy=np.column_stack([(image-layout['image'])@layout['image_x'],(image-layout['image'])@layout['image_y']])
    bounds=layout['detector_bounds'];good &=np.isfinite(t)&(t>0)&~obstruction&np.all((xy>=bounds[0])&(xy<=bounds[1]),axis=1)
    return dict(xy=xy.reshape(*shape,2),good=good.reshape(shape),opl=opl.reshape(shape),phase=phase.reshape(shape),last_surface=o.reshape(*shape,3),
                image=image.reshape(*shape,3),wavelength=wave.reshape(shape),slit=source['slit'],model=layout,paths=np.array(history),obstructed=obstruction.reshape(shape))

def image_metrics(r):
    """Same imaging and wavefront statistics as baseline model.evaluate."""
    good=r['good'];xy=r['xy'];count=good.sum(axis=2)
    cent=np.sum(np.where(good[...,None],xy,0),axis=2)/np.maximum(count[...,None],1)
    rms=np.sqrt(np.sum(np.where(good,np.sum((xy-cent[:,:,None,:])**2,axis=-1),0),axis=2)/np.maximum(count,1))
    m=r['model'];target=m['image']+cent[...,0,None]*m['image_x']+cent[...,1,None]*m['image_y']
    path=r['opl']+np.linalg.norm(target[:,:,None,:]-r['last_surface'],axis=-1)+r['phase']
    piston=np.sum(np.where(good,path,0),axis=2)/np.maximum(count,1)
    wfe=np.sqrt(np.sum(np.where(good,(path-piston[:,:,None])**2,0),axis=2)/np.maximum(count,1))/r['wavelength'][:,:,0]
    return dict(throughput=float(good.mean()),min_throughput=float(np.min(count/good.shape[2])),
                rms_spot_mm=float(np.sqrt(np.mean(rms*rms))),max_spot_mm=float(rms.max()),
                rms_wfe_waves=float(np.sqrt(np.mean(wfe*wfe))),max_wfe_waves=float(wfe.max()),
                dispersion_mm_per_um=float(np.mean(abs(cent[:,-1,0]-cent[:,0,0]))/.6),
                slit_magnification=float(abs(np.mean(cent[-1,:,1]-cent[0,:,1]))/10),
                obstruction_fraction=float(r['obstructed'].mean()),imaging_complete=bool(np.all(count>0)),centroids_mm=cent.tolist())


def detector_intersection(o,d,layout):
    den=d@layout['image_normal']
    t=((layout['image']-o)@layout['image_normal'])/np.where(abs(den)>1e-12,den,1e-12)
    p=o+t[:,None]*d-layout['image']
    xy=np.column_stack([p@layout['image_x'],p@layout['image_y']]);bounds=layout['detector_bounds']
    return t,(t>1e-6)&(abs(den)>1e-12)&np.all((xy>=bounds[0])&(xy<=bounds[1]),axis=1)

def screen_layout(layout,max_rays=49,intersect_fn=None):
    """Cheap intended-segment and unintended surface checks on nominal rays.

    This is a search filter; finalists require full trace and independent audit.
    """
    if not layout['nominal_segment_valid']:
        return dict(valid=False,obstruction_fraction=1.,reason='fold outside ray segment')
    fn=model.intersect if intersect_fn is None else intersect_fn
    paths=layout['nominal_paths'];idx=np.unique(np.linspace(0,paths.shape[1]-1,min(max_rays,paths.shape[1]),dtype=int))
    paths=paths[:,idx];blocked=np.zeros(len(idx),bool)
    for j in range(len(paths)-1):
        o=paths[j];v=paths[j+1]-o;length=np.linalg.norm(v,axis=1);d=model.unit(v)
        for k,surface in enumerate(layout['surfaces']):
            if k==j:continue
            _,_,t,valid=fn(o,d,surface);blocked|=valid&(t<length-1e-5)
        if j<len(layout['surfaces']):
            t,valid=detector_intersection(o,d,layout);blocked|=valid&(t<length-1e-5)
    return dict(valid=not bool(blocked.any()),obstruction_fraction=float(blocked.mean()),reason='unintended optical intersection' if blocked.any() else '')
