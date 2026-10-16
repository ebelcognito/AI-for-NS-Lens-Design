"""Specification-matched Ge F/1.4 reconstruction, not the paper prescription.
Three tilted rotational aspheres and one plane-symmetric XY freeform (M1).
58 mm geometric on-axis path in material enforced by geometry normalization.
Ge dispersion is a Cauchy interpolation of the paper's 8 and 12 um endpoints;
it is not the unpublished original material catalogue. Dimensions mm.
"""
from dataclasses import dataclass
import numpy as np
from nslens.thermal_camera.model import unit, reflect, snell

N_BASE20 = N_BASE = 17
N_TOTAL = 35
FOCAL_MM = 11.76
PUPIL_RADIUS = 4.2
TERMS = ((3,0),(1,2),(4,0),(2,2),(0,4),(5,0),(3,2),(1,4),(6,0),(4,2),(2,4),(0,6))
PIXEL_MM = .017

def index(wavelength_mm):
    w=np.asarray(wavelength_mm)*1000
    b=(4.0067-4.0029)/(1/8**2-1/12**2)
    return 4.0029+b*(1/w**2-1/12**2)

@dataclass
class Surface:
    center: np.ndarray
    tilt: float
    power: np.ndarray
    higher: np.ndarray
    radius: float
    name: str
    radial: object = None

def sag(s, x, y):
    if s.radial is not None: return asphere_sag(s,x,y)
    x = x-s.center[0]
    z = s.center[2]+s.tilt*x+s.power[0]*x*x+s.power[1]*y*y
    dx = s.tilt+2*s.power[0]*x
    dy = 2*s.power[1]*y
    a,b,c,d,e,f,g,h,i,j,k,l=s.higher
    X=x/3; Y=y/3; X2=X*X; Y2=Y*Y
    z=z+X2*X*(a+X*(c+X*f))+Y2*(X*(b+X*(d+X*g))+e*Y2)
    dx=dx+(X2*(3*a+X*(4*c+5*f*X))+Y2*(b+X*(2*d+3*g*X)))/3
    dy=dy+2*Y*(X*(b+X*(d+X*g))+2*e*Y2)/3
    z=z+h*X*Y2*Y2+i*X2**3+j*X2*X2*Y2+k*X2*Y2*Y2+l*Y2**3
    dx=dx+(h*Y2*Y2+6*i*X**5+4*j*X**3*Y2+2*k*X*Y2*Y2)/3
    dy=dy+(4*h*X*Y**3+2*j*X2*X2*Y+4*k*X2*Y**3+6*l*Y**5)/3
    return z,dx,dy

@dataclass
class Model:
    surfaces: list
    detector: np.ndarray
    f1: float

def asphere_sag(s,x,y):
    dx=np.asarray(x)-s.center[0]
    c=1/np.sqrt(1+s.tilt*s.tilt); sn=s.tilt*c
    A,B,C=s.radial
    dz=s.tilt*dx+A*(dx*dx+y*y)/c
    for _ in range(10):
        u=c*dx+sn*dz; q=u*u+y*y
        w=A*q+B*q*q+C*q*q*q
        slope=2*A+4*B*q+6*C*q*q
        f=c*dz-sn*dx-w; den=c-slope*u*sn
        dz-=np.clip(f/np.where(abs(den)>1e-9,den,1e-9),-10,10)
    u=c*dx+sn*dz; q=u*u+y*y
    slope=2*A+4*B*q+6*C*q*q; den=c-slope*u*sn
    den=np.where(abs(den)>1e-9,den,1e-9)
    return s.center[2]+dz,(sn+slope*u*c)/den,slope*y/den

def decode(genes):
    g=np.zeros(N_TOTAL); g[:len(genes)]=genes
    a=11+4*g[0]; h=15+6*g[1]; et=.3*g[2]; xt=.4*g[5]
    n=float(index(.010)); e=np.array([-a,0.,0.])
    di=snell(np.array([0.,0.,1.]),unit(np.array([-et,0.,1.])),1,n)[0]
    m1=e+di*h/di[2]; m2=np.array([a+3*g[3],0.,3*g[4]])
    do=snell(np.array([0.,0.,1.]),unit(np.array([-xt,0.,1.])),1,n)[0]
    ex=m2+do*(h+3*g[6]-m2[2])/do[2]
    scale=58/(np.linalg.norm(m1-e)+np.linalg.norm(m2-m1)+np.linalg.norm(ex-m2))
    e,m1,m2,ex=[p*scale for p in (e,m1,m2,ex)]
    bf=2.+1.7*g[7]; d=unit(m2-m1); dist=np.linalg.norm(m2-m1)
    v=np.linalg.norm(ex-m2)+n*bf
    f1=n*FOCAL_MM*dist/(n*FOCAL_MM+v)*(1+.2*g[13]); u=dist-f1
    normals=[unit(di-d),unit(d-do)]
    normals=[q if q[2]>0 else -q for q in normals]
    n1,n2=normals; ci=abs(np.dot(n1,di))
    primary=np.array([-ci/(4*f1*n1[2]**3),-1/(4*f1*ci*n1[2])])*(1+.5*g[8:10])
    ci2=abs(np.dot(n2,d))
    secondary=(1/u+1/v)*(ci2+1/ci2)/8*(1+.5*g[10])
    # Match tangential and sagittal first-order conjugates of the spherical
    # secondary independently; this is a Gaussian scaffold, not Fermat synthesis.
    At=(1/u+1/v)*(ci2+1/ci2)/8
    ut=1/(4*At/ci2-1/v); us=1/(4*At*ci2-1/v)
    ft=np.clip(dist-ut,1.,2*dist); fs=np.clip(dist-us,1.,2*dist)
    primary=np.array([-ci/(4*ft*n1[2]**3),-1/(4*fs*ci*n1[2])])*(1+.5*g[8:10])
    radii=[PUPIL_RADIUS,6.+1.7*g[14],5.+1.5*g[15],4.5+1.3*g[16]]
    centers=[e,m1,m2,ex]; tilts=[et,-n1[0]/n1[2],-n2[0]/n2[2],xt]
    amps=np.array([.4,.4,.25,.25,.25,.15,.15,.15,.1,.1,.1,.1])
    # XY normalization remains 3 mm for compatibility with exact root checks.
    amps=amps*np.array([(3/6)**(i+j) for i,j in TERMS])
    surfaces=[]
    for k in range(4):
        high=np.zeros(12); radial=None; power=np.zeros(2)
        if k==1: high=amps*g[17:29]; power=primary
        else:
            slot={0:0,2:1,3:2}[k]
            A={0:.02*g[11],2:secondary,3:.025*g[12]}[k]
            radial=np.array([A,.4*g[29+2*slot]/6**4,.15*g[30+2*slot]/6**6])
        surfaces.append(Surface(centers[k],tilts[k],power,high,radii[k],['entrance','mirror1','mirror2','exit'][k],radial))
    return Model(surfaces,ex+np.array([0,0,bf]),f1)

def body_face(left,right,x,y):
    def clamped(s):
        dx=x-s.center[0]; r=np.hypot(dx,y); fac=np.minimum(1,s.radius/np.maximum(r,1e-12))
        return sag(s,s.center[0]+dx*fac,y*fac)[0]
    lo=left.center[0]+left.radius; hi=right.center[0]-right.radius
    w=np.clip((x-lo)/max(hi-lo,.01),0,1); w=w*w*(3-2*w)
    return (1-w)*clamped(left)+w*clamped(right)

def body(model,x,y):
    e,m1,m2,ex=model.surfaces
    return body_face(e,m2,x,y),body_face(m1,ex,x,y)

def intersect(o,d,s):
    t=(s.center[2]-o[:,2])/np.where(abs(d[:,2])>1e-8,d[:,2],1e-8)
    for _ in range(9):
        p=o+t[:,None]*d
        with np.errstate(over='ignore',invalid='ignore'):
            z,zx,zy=sag(s,np.clip(p[:,0],-40,40),np.clip(p[:,1],-40,40))
            f=p[:,2]-z; der=d[:,2]-zx*d[:,0]-zy*d[:,1]
            t-=np.clip(f/np.where(abs(der)>1e-8,der,1e-8),-20,20)
    p=o+t[:,None]*d; z,zx,zy=sag(s,np.clip(p[:,0],-40,40),np.clip(p[:,1],-40,40))
    overflow=np.maximum(0,np.hypot(p[:,0]-s.center[0],p[:,1])-s.radius)
    valid=(abs(p[:,2]-z)<1e-5)&(t>1e-7)&(overflow<1e-7)
    if s.radial is not None:
        c=1/np.sqrt(1+s.tilt*s.tilt); sn=s.tilt*c
        dx=p[:,0]-s.center[0]; dz=p[:,2]-s.center[2]
        uu=c*dx+sn*dz; q=uu*uu+p[:,1]**2; A,B,C=s.radial
        valid &= abs(c*dz-sn*dx-(A*q+B*q*q+C*q*q*q))<1e-5
    return p,unit(np.stack([-zx,-zy,np.ones_like(z)],-1)),valid,overflow

def trace(genes,fields=3,waves=3,pupils=5,body_samples=7):
    model=decode(genes)
    nx,ny=fields if isinstance(fields,(tuple,list)) else (fields,fields)
    f=np.array([(x,y) for y in np.linspace(-12.5,12.5,ny) for x in np.linspace(-18.5,18.5,nx)])
    wl=np.linspace(.008,.012,waves)
    rad=np.sqrt((np.arange(pupils)+.5)/pupils); theta=np.arange(4*pupils)*2*np.pi/(4*pupils)
    pupil=np.array([(r*np.cos(t),r*np.sin(t)) for r in rad for t in theta])*PUPIL_RADIUS
    shape=(len(f),len(wl),len(pupil)); n=index(np.broadcast_to(wl[None,:,None],shape).ravel())
    ang=np.broadcast_to(f[:,None,None,:],shape+(2,)).reshape(-1,2)
    d=unit(np.c_[np.tan(np.deg2rad(ang)),np.ones(len(ang))])
    xy=np.broadcast_to(pupil[None,None,:,:],shape+(2,)).reshape(-1,2)
    o=np.c_[xy[:,0]+model.surfaces[0].center[0],xy[:,1],np.zeros(len(xy))]-d*4/d[:,2,None]
    paths=[o.copy()]; good=np.ones(len(o),bool); overflow=np.zeros(len(o)); escape=np.zeros(len(o))
    for k,s in enumerate(model.surfaces):
        p,normal,ok,ov=intersect(o,d,s); good &= ok; overflow+=np.minimum(ov,20)
        if k:
            for frac in np.linspace(.025,.975,body_samples):
                q=o+(p-o)*frac; front,back=body(model,q[:,0],q[:,1])
                err=np.maximum.reduce([front-q[:,2],q[:,2]-back,abs(q[:,0])-40,abs(q[:,1])-16,np.zeros(len(q))])
                escape=np.maximum(escape,np.minimum(err,20))
        if k==0: d,ok=snell(d,normal,1,n)
        elif k==3: d,ok=snell(d,normal,n,1)
        else: d=reflect(d,normal); ok=np.ones(len(d),bool)
        good &= ok; paths.append(p.copy()); o=p
    t=(model.detector[2]-o[:,2])/np.where(abs(d[:,2])>1e-8,d[:,2],1e-8)
    p=o+t[:,None]*d; good &= (t>0)&(escape<1e-6)&np.all(np.isfinite(p),axis=1)
    paths.append(p)
    return dict(model=model,xy=(p[:,:2]-model.detector[:2]).reshape(shape+(2,)),good=good.reshape(shape),paths=np.array(paths),fields_deg=f,wavelengths_mm=wl,overflow=overflow.reshape(shape),escape=escape.reshape(shape))

def residuals(genes,reference_good=None,fields=3,waves=3,pupils=5,_trace=None):
    tr=trace(genes,fields,waves,pupils) if _trace is None else _trace; xy=np.clip(np.nan_to_num(tr['xy'],nan=100,posinf=100,neginf=-100),-20,20)
    center=np.mean(xy,axis=(1,2)); delta=xy-center[:,None,None,:]
    target=FOCAL_MM*np.tan(np.deg2rad(tr['fields_deg']))
    # Choose only image parity, never fit away focal length or field mapping.
    sign=np.where(np.sum(center*target,axis=0)>=0,1,-1)
    mapping=(center-target*sign)/PIXEL_MM
    m=tr['model']; e,m1,m2,ex=m.surfaces
    gx,gy=np.meshgrid(np.linspace(-40,40,25),np.linspace(-16,16,13)); front,back=body(m,gx,gy)
    geom=np.array([max(0,.2-(m2.center[0]-m2.radius-e.center[0]-e.radius)),max(0,.2-(ex.center[0]-ex.radius-m1.center[0]-m1.radius)),0.0,max(0,.1-np.min(back-front))])
    return np.concatenate([np.clip(delta/PIXEL_MM,-100,100).ravel()/np.sqrt(delta.size),.3*mapping.ravel()/np.sqrt(mapping.size),20*(np.minimum(1,(tr['overflow']+tr['escape'])/.05)+((~tr['good'])&(tr['overflow']<1e-7)&(tr['escape']<1e-6))).ravel()/np.sqrt(tr['good'].size),10*tr['overflow'].ravel()/np.sqrt(tr['good'].size),20*tr['escape'].ravel()/np.sqrt(tr['good'].size),20*geom])

def evaluate(genes,fields=3,waves=3,pupils=5):
    tr=trace(genes,fields,waves,pupils); good=tr['good']; xy=tr['xy']
    count=good.sum(axis=2); center=np.sum(np.where(good[...,None],xy,0),axis=2)/np.maximum(count[...,None],1)
    rms=np.sqrt(np.sum(np.where(good,np.sum((xy-center[:,:,None,:])**2,axis=-1),0),axis=2)/np.maximum(count,1))
    rms=np.where(count>0,rms,np.nan)
    r=residuals(genes,fields=fields,waves=waves,pupils=pupils,_trace=tr)
    return dict(loss=float(r@r),throughput=float(good.mean()),min_throughput=float(np.min(good.mean(axis=2))),empty_cells=int(np.sum(count==0)),rms_spot_mm=float(np.nanmean(rms)) if np.any(count) else None,max_spot_mm=float(np.nanmax(rms)) if np.any(count) else None,valid_rays=int(good.sum()),launched_rays=int(good.size))










