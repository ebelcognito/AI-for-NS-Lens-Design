"""Experimental 3D sequential freeform spectrometer, lengths in mm.

Surfaces use polynomial sag in an orthonormal local frame. A uniform grating
phase is density*x cycles, with x the local projected coordinate. This is an
independent 3D domain model, not an extension of the JavaScript 2D tracer.
"""
import numpy as np

N_BASE=15
N_TOTAL=36
WAVELENGTHS=(0.0005,0.0011)
DENSITY=300.
TERMS=((2,0),(0,2),(3,0),(1,2),(4,0),(2,2),(0,4),(5,0),(3,2))

def unit(a):
    a=np.asarray(a,float)
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-30)

def sag(x,y,coeff,terms=None):
    x,y=np.broadcast_arrays(x,y); c=np.asarray(coeff,float)
    # Optional explicit basis for other off-axis mirror applications. Keep the
    # original spectrometer's optimized nine-term path unchanged by default.
    if terms is not None:
        if len(c)!=len(terms):raise ValueError('One coefficient per polynomial term required')
        z=np.zeros_like(x,dtype=float);zx=z.copy();zy=z.copy()
        active=[(value,i,j) for value,(i,j) in zip(c,terms) if value!=0]
        xp=[np.ones_like(x,dtype=float)];yp=[np.ones_like(y,dtype=float)]
        for _ in range(max((i for _,i,_ in active),default=0)):xp.append(xp[-1]*x)
        for _ in range(max((j for _,_,j in active),default=0)):yp.append(yp[-1]*y)
        for value,i,j in active:
            z+=value*xp[i]*yp[j]
            if i:zx+=value*i*xp[i-1]*yp[j]
            if j:zy+=value*j*xp[i]*yp[j-1]
        return z,zx,zy
    if len(c)<9: c=np.pad(c,(0,9-len(c)))
    xx=x*x; yy=y*y; q=c[1]+x*(c[3]+x*(c[5]+x*c[8]))
    z=xx*(c[0]+x*(c[2]+x*(c[4]+x*c[7])))+yy*(q+c[6]*yy)
    zx=x*(2*c[0]+x*(3*c[2]+x*(4*c[4]+5*c[7]*x)))+yy*(c[3]+x*(2*c[5]+3*c[8]*x))
    zy=2*y*(q+2*c[6]*yy)
    return z,zx,zy


def reflect(d,n):
    return d-2*np.sum(d*n,axis=-1,keepdims=True)*n

def diffract(d,n,tangent,wavelength_mm,order=1,density=DENSITY):
    n=unit(n); tangent=np.asarray(tangent); gradient=tangent-n*np.sum(tangent*n,axis=-1,keepdims=True)
    tangential=d-n*np.sum(d*n,axis=-1,keepdims=True)
    q=tangential+np.asarray(wavelength_mm)[...,None]*order*density*gradient
    rem=1-np.sum(q*q,axis=-1)
    side=-np.where(np.sum(d*n,axis=-1)>=0,1.,-1.)
    return q+(side*np.sqrt(np.maximum(rem,0)))[...,None]*n,rem>=0

def grating_phase_length(x,wavelength_mm,order=1,density=DENSITY):
    return order*density*np.asarray(wavelength_mm)*x

def intersect(o,d,surface):
    o=np.asarray(o); d=np.asarray(d); b=surface['basis']; a=(o-surface['vertex'])@b; v=d@b
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        t=-a[:,2]/np.where(abs(v[:,2])>1e-12,v[:,2],1e-12)
        for _ in range(9):
            p=a+t[:,None]*v; z,zx,zy=sag(p[:,0],p[:,1],surface['coeff'],surface.get('terms'))
            residual=p[:,2]-z; derivative=v[:,2]-zx*v[:,0]-zy*v[:,1]
            t-=np.clip(residual/np.where(abs(derivative)>1e-12,derivative,1e-12),-100,100)
        p=a+t[:,None]*v; z,zx,zy=sag(p[:,0],p[:,1],surface['coeff'],surface.get('terms'))
    good=np.isfinite(t)&(t>1e-6)&(abs(p[:,2]-z)<1e-6)&(p[:,0]**2+p[:,1]**2<=surface['aperture']**2)
    normals=unit(np.column_stack([-zx,-zy,np.ones(len(t))])@b.T)
    return o+t[:,None]*d,normals,t,good

def decode(g,aperture_radii=None):
    g=np.pad(np.asarray(g,float),(0,max(0,N_TOTAL-len(g))))
    # Independent scaffold ranges; no published surface prescription is used.
    turns=np.deg2rad(np.array([125.,-125.,125.])+25*g[:3])
    lengths=np.array([45.,50.,50.,45.])+np.array([20.,25.,25.,20.])*g[3:7]
    curves=(.0002+.0174*(g[7:13].reshape(3,2)+1))*np.array([[-1.],[1.],[-1.]])
    higher=g[15:36].reshape(3,7)*np.array([2e-4,2e-4,8e-6,8e-6,8e-6,2e-7,2e-7])
    origin=np.zeros(3); d=np.array([0.,0.,1.]); p=origin+lengths[0]*d; surfaces=[]
    for j,turn in enumerate(turns):
        c,s=np.cos(turn),np.sin(turn); out=np.array([c*d[0]+s*d[2],0.,-s*d[0]+c*d[2]])
        n=unit(d-out); y=np.array([0.,1.,0.]); x=np.cross(y,n)
        b=np.column_stack([x,y,n]); surfaces.append(dict(vertex=p.copy(),basis=b,coeff=np.r_[curves[j],higher[j]],aperture=18. if aperture_radii is None else float(aperture_radii[j])))
        if j==1: out,_=diffract(d,n,x,.0008)
        d=out; p=p+lengths[j+1]*d
    x=unit(np.cross(np.array([0.,1.,0.]),d)); y=np.cross(d,x)
    normal=unit(d+.25*g[13]*x+.25*g[14]*y); x=unit(np.cross(y,normal)); y=np.cross(normal,x)
    return dict(surfaces=surfaces,image=p,image_normal=normal,image_x=x,image_y=y,lengths=lengths,turns=turns,origin=origin)

def trace(g,fields=3,waves=3,pupils=5,pupil_xy=None,aperture_radii=None):
    m=decode(g,aperture_radii=aperture_radii)
    # Uniform Cartesian disk quadrature. Identical pupils for every field/band.
    u=np.linspace(-1,1,pupils); px,py=np.meshgrid(u,u); disk=px*px+py*py<=1+1e-12
    px=px[disk]; py=py[disk]
    if pupil_xy is not None:
        custom=np.asarray(pupil_xy,float)
        if custom.ndim!=2 or custom.shape[1]!=2 or not len(custom) or not np.isfinite(custom).all() or np.any(np.sum(custom*custom,axis=1)>1+1e-12):
            raise ValueError("pupil_xy must contain finite points inside the unit disk")
        px,py=custom.T
    npup=len(px)
    slit,wave=np.meshgrid(np.linspace(-5,5,fields),np.linspace(*WAVELENGTHS,waves),indexing='ij')
    slit=np.repeat(slit.ravel(),npup); wave=np.repeat(wave.ravel(),npup)
    o=np.column_stack([np.zeros(len(slit)),slit,np.zeros(len(slit))])
    # F/3.8 cone at slit, with chief ray directed to primary vertex.
    d=unit(m['surfaces'][0]['vertex']-o+np.column_stack([np.tile(px,fields*waves),np.tile(py,fields*waves),np.zeros(len(slit))])*m['lengths'][0]/7.6)
    good=np.ones(len(o),bool); opl=np.zeros(len(o)); phase=np.zeros(len(o)); histories=[o.copy()]; obstruction=np.zeros(len(o),bool)
    for j,surface in enumerate(m['surfaces']):
        p,n,t,valid=intersect(o,d,surface)
        # Check every unintended surface for nearer forward intersection.
        for k,other in enumerate(m['surfaces']):
            if k==j: continue
            _,_,ot,ov=intersect(o,d,other)
            obstruction|=good&ov&(ot<t-1e-5)
        good &=valid; opl+=np.where(valid,t,0)
        if j==1:
            local=(p-surface['vertex'])@surface['basis']; phase=grating_phase_length(local[:,0],wave)
            d,valid_order=diffract(d,n,surface['basis'][:,0],wave); good &=valid_order
        else: d=reflect(d,n)
        o=p; histories.append(p.copy())
    den=d@m['image_normal']; t=((m['image']-o)@m['image_normal'])/np.where(abs(den)>1e-12,den,1e-12)
    for other in m['surfaces']:
        _,_,ot,ov=intersect(o,d,other); obstruction|=good&ov&(ot<t-1e-5)
    good &=np.isfinite(t)&(t>0)&~obstruction
    image=o+t[:,None]*d; histories.append(image)
    xy=np.column_stack([(image-m['image'])@m['image_x'],(image-m['image'])@m['image_y']])
    shape=(fields,waves,npup)
    return dict(xy=xy.reshape(*shape,2),good=good.reshape(shape),opl=opl.reshape(shape),phase=phase.reshape(shape),last_surface=o.reshape(*shape,3),image=image.reshape(*shape,3),wavelength=wave.reshape(shape),slit=slit.reshape(shape),model=m,paths=np.array(histories),obstructed=obstruction.reshape(shape))

def evaluate(g,fields=3,waves=3,pupils=5,pupil_xy=None,aperture_radii=None):
    with np.errstate(all='ignore'):
        r=trace(g,fields,waves,pupils,pupil_xy=pupil_xy,aperture_radii=aperture_radii); good=r['good']; xy=r['xy']; count=np.sum(good,axis=2); throughput=float(np.mean(good))
        cent=np.sum(np.where(good[...,None],xy,0),axis=2)/np.maximum(count[...,None],1)
        distance=np.sum((xy-cent[:,:,None,:])**2,axis=-1)
        rms=np.sqrt(np.sum(np.where(good,distance,0),axis=2)/np.maximum(count,1))
        # Reference sphere centred on each geometric centroid. Add grating phase.
        m=r['model']; target=m['image']+cent[...,0,None]*m['image_x']+cent[...,1,None]*m['image_y']
        wavepath=r['opl']+np.linalg.norm(target[:,:,None,:]-r['last_surface'],axis=-1)+r['phase']
        piston=np.sum(np.where(good,wavepath,0),axis=2)/np.maximum(count,1)
        wfe=np.sqrt(np.sum(np.where(good,(wavepath-piston[:,:,None])**2,0),axis=2)/np.maximum(count,1))/r['wavelength'][:,:,0]
        dispersion=np.mean(abs(cent[:,-1,0]-cent[:,0,0]))/.6 # mm per micron, desired10
        magnification=abs(np.mean(cent[-1,:,1]-cent[0,:,1]))/10
        # Fixed throughput penalties retain all launched rays in the objective.
        loss=float(np.mean(np.minimum(rms,100))+30*(1-throughput)+10*np.mean(count==0)+.2*(dispersion-10)**2+.5*(magnification-1)**2)
        if not np.isfinite(loss): loss=1e6
    return dict(loss=loss,imaging_complete=bool(np.all(count>0)),empty_field_wave_cells=int(np.sum(count==0)),throughput=throughput,min_throughput=float(np.min(count/good.shape[2])),rms_spot_mm=float(np.sqrt(np.mean(rms*rms))),max_spot_mm=float(np.max(rms)),rms_wfe_waves=float(np.sqrt(np.mean(wfe*wfe))),max_wfe_waves=float(np.max(wfe)),dispersion_mm_per_um=float(dispersion),slit_magnification=float(magnification),centroids_mm=cent.tolist(),obstruction_fraction=float(np.mean(r['obstructed'])))


def geometry_metrics(g, samples=25,aperture_radii=None):
    """Sample nominal finite surfaces; these are diagnostics, not clearance proofs.

    A nearest sample-pair gap can exceed the true gap and miss intersections.
    Bounds cover sampled optical surfaces, slit ends, and detector centre only;
    mounts, substrates, and detector outline are not represented.
    """
    m=decode(g,aperture_radii=aperture_radii)
    grid=np.linspace(-1.,1.,samples)
    xx,yy=np.meshgrid(grid,grid); mask=xx*xx+yy*yy<=1.+1e-12
    clouds=[]; surface_rows=[]
    for surface in m['surfaces']:
        x=xx[mask]*surface['aperture']; y=yy[mask]*surface['aperture']
        z,zx,zy=sag(x,y,surface['coeff'])
        cloud=np.column_stack([x,y,z])@surface['basis'].T+surface['vertex']
        clouds.append(cloud)
        zxx=np.zeros_like(x);zxy=np.zeros_like(x);zyy=np.zeros_like(x)
        for c,(i,j) in zip(surface['coeff'],TERMS):
            if i>=2:zxx+=c*i*(i-1)*x**(i-2)*y**j
            if i and j:zxy+=c*i*j*x**(i-1)*y**(j-1)
            if j>=2:zyy+=c*j*(j-1)*x**i*y**(j-2)
        scale=np.sqrt(1.+zx*zx+zy*zy)
        first=np.empty((len(x),2,2));second=first.copy()
        first[:,0,0]=1.+zx*zx;first[:,0,1]=first[:,1,0]=zx*zy;first[:,1,1]=1.+zy*zy
        second[:,0,0]=zxx/scale;second[:,0,1]=second[:,1,0]=zxy/scale;second[:,1,1]=zyy/scale
        principal=np.linalg.eigvals(np.linalg.solve(first,second)).real
        curvature=float(np.max(np.abs(principal)))
        surface_rows.append(dict(aperture_radius_mm=float(surface['aperture']),
                                 sampled_max_abs_sag_mm=float(np.max(abs(z))),
                                 sampled_max_slope=float(np.max(np.hypot(zx,zy))),
                                 sampled_max_abs_principal_curvature_per_mm=curvature,
                                 sampled_min_signed_principal_curvature_per_mm=float(np.min(principal)),
                                 sampled_max_signed_principal_curvature_per_mm=float(np.max(principal)),
                                 sampled_min_abs_principal_radius_mm=1./curvature if curvature>0 else None))
    points=np.concatenate(clouds+[np.array([[0.,-5.,0.],[0.,5.,0.],m['image']])])
    lo=points.min(axis=0);hi=points.max(axis=0)
    vertices=np.array([s['vertex'] for s in m['surfaces']])
    pair_gaps=[];vertex_gaps=[]
    for i in range(3):
        for j in range(i+1,3):
            pair_gaps.append(float(np.sqrt(np.min(np.sum((clouds[i][:,None]-clouds[j][None,:])**2,axis=-1)))))
            vertex_gaps.append(float(np.linalg.norm(vertices[i]-vertices[j])))
    return dict(sample_count_per_surface=len(clouds[0]),surfaces=surface_rows,
                sampled_bounds_min_mm=lo.tolist(),sampled_bounds_max_mm=hi.tolist(),
                sampled_bounds_size_mm=(hi-lo).tolist(),sampled_aabb_volume_cm3=float(np.prod(hi-lo)/1000.),
                min_vertex_separation_mm=min(vertex_gaps),
                min_sample_pair_gap_mm=min(pair_gaps),
                clearance_certified=False,
                limitation='Sampled optical surfaces only; gaps are not lower bounds on clearance; no mounts, substrates or detector outline.')


def residuals(g,fields=3,waves=3,pupils=5,dispersion_sign=1.,slit_sign=1.,mapping_weight=1.,pupil_xy=None,aperture_radii=None,curvature_weight=0.):
    """All-ray residual for a separate geometric least-squares polishing stage.

Fixed ideal 10 mm/um dispersion and unit slit magnification. Overall image
translation is removed, but no field-dependent centroid fitting is allowed.
Invalid routes retain their image error and an explicit loss penalty.
"""
    with np.errstate(all='ignore'):
        r=trace(g,fields,waves,pupils,pupil_xy=pupil_xy,aperture_radii=aperture_radii); xy=r['xy']; n=r['good'].size
        target=np.stack([dispersion_sign*10000*(r['wavelength']-.0008),slit_sign*r['slit']],axis=-1)
        error=xy-target; centroid_error=np.mean(error,axis=2,keepdims=True)
        error=error-centroid_error+mapping_weight*(centroid_error-np.mean(centroid_error,axis=(0,1,2),keepdims=True))
        pieces=[error.ravel(),10*(~r['good']).ravel()]
        if mapping_weight != 1.:
            spectral_span=np.mean(xy[:,-1,:,0]-xy[:,0,:,0])
            pieces.append(np.array([3*np.sqrt(n)*(spectral_span-dispersion_sign*6)]))
        # Smooth aperture overflow guides the optimiser back before rays are lost.
        for j,s in enumerate(r['model']['surfaces']):
            p=(r['paths'][j+1]-s['vertex'])@s['basis']
            pieces.append(3*np.maximum(np.linalg.norm(p[:,:2],axis=1)-s['aperture'],0))
        if curvature_weight:
            violations=curvature_violations(g,aperture_radii=aperture_radii)
            pieces.append(curvature_weight*np.sqrt(n/len(violations))*violations)
        return np.nan_to_num(np.concatenate(pieces)/np.sqrt(n),nan=1e3,posinf=1e3,neginf=-1e3)


def distortion_metrics(g,fields=8,waves=8,pupils=10,pupil_xy=None,aperture_radii=None):
    """Sampled centroid distortions on the common detector, in millimetres."""
    r=trace(g,fields,waves,pupils,pupil_xy=pupil_xy,aperture_radii=aperture_radii); good=r['good']; count=good.sum(axis=2)
    if np.any(count==0): return dict(valid=False,reason='An empty field/wavelength cell prevents distortion evaluation.')
    cent=np.sum(np.where(good[...,None],r['xy'],0),axis=2)/count[...,None]
    spectral_sign=np.sign(np.mean(cent[:,-1,0]-cent[:,0,0])); slit_sign=np.sign(np.mean(cent[-1,:,1]-cent[0,:,1]))
    target=np.stack([spectral_sign*10000*(r['wavelength'][:,:,0]-.0008),slit_sign*r['slit'][:,:,0]],axis=-1)
    error=cent-target; error-=error.mean(axis=(0,1),keepdims=True)
    return dict(valid=True,max_spectral_smile_mm=float(np.max(np.ptp(cent[:,:,0],axis=0))),max_spatial_keystone_mm=float(np.max(np.ptp(cent[:,:,1],axis=1))),rms_ideal_centroid_map_error_mm=float(np.sqrt(np.mean(np.sum(error*error,axis=-1)))),max_ideal_centroid_map_error_mm=float(np.max(np.linalg.norm(error,axis=-1))))


def sag_hessian(x,y,coeff):
    """Exact sag Hessian; definiteness matches graph principal-curvature signs."""
    c=np.asarray(coeff); x,y=np.broadcast_arrays(x,y)
    xx=2*c[0]+6*c[2]*x+12*c[4]*x*x+2*c[5]*y*y+20*c[7]*x*x*x+6*c[8]*x*y*y
    xy=2*c[3]*y+4*c[5]*x*y+6*c[8]*x*x*y
    yy=2*c[1]+2*c[3]*x+2*c[5]*x*x+12*c[6]*y*y+2*c[8]*x*x*x
    h=np.empty((*x.shape,2,2));h[...,0,0]=xx;h[...,0,1]=h[...,1,0]=xy;h[...,1,1]=yy
    return h


def curvature_violations(g,aperture_radii=None,radial_samples=9,azimuth_samples=48,margin=2e-5):
    """Sample fixed physical disks, including every sampled boundary azimuth.

    Positive semidefiniteness of the signed Hessian enforces the intended
    concave/convex/concave family. This is a sampled optimisation constraint;
    a separate dense audit must accept the final candidate.
    """
    angles=np.arange(azimuth_samples)*2*np.pi/azimuth_samples
    radii=np.linspace(0,1,radial_samples)
    x=(radii[:,None]*np.cos(angles)).ravel();y=(radii[:,None]*np.sin(angles)).ravel()
    result=[]
    for sign,surface in zip((-1.,1.,-1.),decode(g,aperture_radii=aperture_radii)['surfaces']):
        h=sag_hessian(x*surface['aperture'],y*surface['aperture'],surface['coeff'])
        result.append(np.maximum(margin-np.linalg.eigvalsh(sign*h),0).ravel())
    return np.concatenate(result)
