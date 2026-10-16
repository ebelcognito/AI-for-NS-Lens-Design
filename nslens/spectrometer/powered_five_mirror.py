"""Seeded six-surface spectrometer: five powered mirrors and one curved grating.
Insertion topology is selected by broad folding search, not discovered by CMA.
All finite physical masks are frozen before each refinement run.
"""
import numpy as np
from . import model, four_mirror as fold, powered_four_mirror as parent
N_BASE=33
N_TOTAL=75
SEGMENT=0
APERTURE=10.

def configure(segment, aperture):
    global SEGMENT,APERTURE
    SEGMENT=int(segment);APERTURE=float(aperture)

def insert(layout, segment, params, aperture=None, paths=None):
    fraction,turn,azimuth=params
    nodes=[layout['origin'],*[s['vertex'] for s in layout['surfaces']],layout['image']]
    center=nodes[segment]+fraction*(nodes[segment+1]-nodes[segment])
    flat,Q=fold._flat(center,model.unit(nodes[segment+1]-nodes[segment]),turn,azimuth)
    surfaces=[]
    for j,s in enumerate(layout['surfaces']):
        out=fold._surface(s,center,Q) if j>=segment else dict(s)
        out['original_index']=j;surfaces.append(out)
    flat['original_index']=-1
    result=dict(layout)
    result.update(surfaces=surfaces[:segment]+[flat]+surfaces[segment:],image=fold._point(layout['image'],center,Q),
                  image_x=Q@layout['image_x'],image_y=Q@layout['image_y'],image_normal=Q@layout['image_normal'],
                  grating_index=2+int(segment<=2), insertion_segment=segment,insertion_params=np.array(params))
    if paths is not None:
        cross,frac=fold._crossing(paths[segment],paths[segment+1],flat)
        local=(cross-center)@flat['basis']
        measured=float(np.max(np.linalg.norm(local[:,:2],axis=1))+1.)
        result['nominal_paths']=np.concatenate([paths[:segment+1],cross[None],fold._point(paths[segment+1:],center,Q)])
        result['nominal_segment_valid']=bool(np.all((frac>0)&(frac<1)))
        flat['aperture']=measured if aperture is None else aperture
    else:flat['aperture']=APERTURE if aperture is None else aperture
    return result

def seed_genes(inherited,params):
    g=np.zeros(N_TOTAL);g[:28]=inherited[:28];g[33:68]=inherited[28:]
    g[28:31]=2*(np.asarray(params)-[.15,45,0])/[.7,90,360]-1
    return g

def decode(g):
    g=np.pad(np.asarray(g,float),(0,max(0,N_TOTAL-len(g))))
    old=np.r_[g[:28],g[33:68]]
    m=parent.decode(old)
    params=np.array([.15,45,0])+(g[28:31]+1)*.5*np.array([.7,90,360])
    m=insert(m,SEGMENT,params,APERTURE)
    m['surfaces'][SEGMENT]['coeff']=np.r_[.012*g[31:33],g[68:75]*parent.HIGH_SCALE*.5]
    m['genes']=g
    return m

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
        if j==m['grating_index']:
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
    m=decode(g);package=package_metrics(m,m['rotation_xyz_deg'],samples=samples)
    geometry=fold.geometry_metrics(m,samples=samples)
    return dict(**package,min_surface_gap_mm=geometry['min_surface_gap_mm'],min_detector_gap_mm=geometry['min_detector_gap_mm'],aperture_radii_mm=[s['aperture'] for s in m['surfaces']])

def curvature_metrics(layout,radial_samples=5,azimuth_samples=24):
    ang=np.arange(azimuth_samples)*2*np.pi/azimuth_samples;rad=np.linspace(0,1,radial_samples)
    x=(rad[:,None]*np.cos(ang)).ravel();y=(rad[:,None]*np.sin(ang)).ravel()
    sign_violations=[];max_slope=0.;max_sag=0.;max_curve=0.
    for index,s in enumerate(layout['surfaces']):
        xx=x*s['aperture'];yy=y*s['aperture'];z,zx,zy=model.sag(xx,yy,s['coeff']);h=model.sag_hessian(xx,yy,s['coeff'])
        eig=np.linalg.eigvalsh(h)
        max_slope=max(max_slope,float(np.hypot(zx,zy).max()));max_sag=max(max_sag,float(abs(z).max()));max_curve=max(max_curve,float(abs(eig).max()))
        if s.get('original_index') in (1,2,3):
            sign=1 if s['original_index']==2 else -1
            sign_violations.extend(np.maximum(2e-6-np.linalg.eigvalsh(sign*h),0).ravel())
    return dict(sign_violation=float(np.max(sign_violations)),max_slope=max_slope,max_abs_sag_mm=max_sag,max_abs_hessian_per_mm=max_curve)


def evaluate(g,fields=3,waves=3,pupils=5,pupil_xy=None,hardware_constraints=False):
    with np.errstate(all='ignore'):
        r=trace(g,fields,waves,pupils,pupil_xy=pupil_xy);stats=fold.image_metrics(r)
        geo=geometry_metrics(g,samples=7);curvature=curvature_metrics(r['model'])
        empty=float(np.mean(~np.any(r['good'],axis=2)))
        gap=min(geo['min_surface_gap_mm'],geo['min_detector_gap_mm'])
        loss=(stats['rms_spot_mm']+30*(1-stats['throughput'])+10*empty+
              .2*(stats['dispersion_mm_per_um']-10)**2+.5*(stats['slit_magnification']-1)**2+
              .03*geo['volume_cm3']/60+2*max(2-gap,0)**2+
              1e5*curvature['sign_violation']**2+10*max(curvature['max_slope']-.7,0)**2+max(curvature['max_abs_sag_mm']-4,0)**2)
        hardware={}
        if hardware_constraints:
            from .slit_hardware import hardware_check
            hardware=hardware_check(r['model'],paths=r['paths'],dense=False)
            loss+=hardware['loss_penalty']
        if not np.isfinite(loss):loss=1e6
    result=dict(**stats,loss=float(loss),volume_cm3=geo['volume_cm3'],sampled_min_gap_mm=float(gap),curvature=curvature)
    if hardware_constraints:result['hardware']=hardware
    return result

