"""Wide-range experimental four-zone closed-body IG4 objective; not an IOF prescription.

Dimensions mm. IG4 dispersion beyond 12 um is extrapolated. Mirrors coated.
"""
from dataclasses import dataclass
import numpy as np
from nslens.thermal_camera.model import unit, reflect, snell

N_BASE20 = N_BASE = 20
N_TOTAL = 48
TERMS = ((3,0),(1,2),(4,0),(2,2),(0,4),(5,0),(3,2))
PIXEL_MM = .017

def index(wavelength_mm):
    w = np.asarray(wavelength_mm)*1000
    return np.sqrt(3.7832+3.0693*w*w/(w*w-.3966**2)+.8541*w*w/(w*w-42.3492**2))

@dataclass
class Surface:
    center: np.ndarray
    tilt: float
    power: np.ndarray
    higher: np.ndarray
    radius: float
    name: str

def sag(s, x, y):
    x = x-s.center[0]
    z = s.center[2]+s.tilt*x+s.power[0]*x*x+s.power[1]*y*y
    dx = s.tilt+2*s.power[0]*x
    dy = 2*s.power[1]*y
    a,b,c,d,e,f,g=s.higher
    X=x/3; Y=y/3; X2=X*X; Y2=Y*Y
    z=z+X2*X*(a+X*(c+X*f))+Y2*(X*(b+X*(d+X*g))+e*Y2)
    dx=dx+(X2*(3*a+X*(4*c+5*f*X))+Y2*(b+X*(2*d+3*g*X)))/3
    dy=dy+2*Y*(X*(b+X*(d+X*g))+2*e*Y2)/3
    return z,dx,dy

@dataclass
class Model:
    surfaces: list
    detector: np.ndarray
    f1: float

def decode(genes):
    g=np.zeros(N_TOTAL); g[:len(genes)]=genes
    a=4.5+.5*g[0]; h=6+.8*g[1]; et=.15*g[2]; xt=.2*g[5]
    n=float(index(.011)); e=np.array([-a,0.,0.])
    di=snell(np.array([0.,0.,1.]),unit(np.array([-et,0.,1.])),1,n)[0]
    m1=e+di*h/di[2]; m2=np.array([a+.5*g[3],0.,.5*g[4]])
    do=snell(np.array([0.,0.,1.]),unit(np.array([-xt,0.,1.])),1,n)[0]
    ex=m2+do*(h+.5*g[6]-m2[2])/do[2]; bf=.7+.5*g[7]
    d=unit(m2-m1); dist=np.linalg.norm(m2-m1)
    v=np.linalg.norm(ex-m2)+n*bf
    f1=n*6*dist/(n*6+v)*(1+.2*g[16]); u=dist-f1
    normals=[unit(di-d),unit(d-do)]
    normals=[q if q[2]>0 else -q for q in normals]
    powers=[]
    for k,(normal,incident) in enumerate(zip(normals,[di,d])):
        ci=abs(np.dot(normal,incident)); nz=normal[2]
        fac=-1/f1 if k==0 else (1/u+1/v)
        powers.append(np.array([fac*ci/(4*nz**3),fac/(4*ci*nz)])*(1+.3*g[8+2*k:10+2*k]))
    centers=[e,m1,m2,ex]; tilts=[et,-normals[0][0]/normals[0][2],-normals[1][0]/normals[1][2],xt]
    powers=[.015*g[12:14],powers[0],powers[1],.02*g[14:16]]
    radii=[3,3.6+.4*g[17],2.7+.4*g[18],2.3+.3*g[19]]
    amps=5*np.array([.08,.08,.04,.04,.04,.025,.025])
    surfaces=[Surface(c,t,p,amps*g[20+7*k:27+7*k],r,name) for k,(c,t,p,r,name) in enumerate(zip(centers,tilts,powers,radii,['entrance','mirror1','mirror2','exit']))]
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
    return p,unit(np.stack([-zx,-zy,np.ones_like(z)],-1)),valid,overflow

def trace(genes,fields=3,waves=3,pupils=5,body_samples=7):
    model=decode(genes)
    f=np.array([(x,y) for y in np.linspace(-12.5,12.5,fields) for x in np.linspace(-18.5,18.5,fields)])
    wl=np.linspace(.008,.014,waves)
    rad=np.sqrt((np.arange(pupils)+.5)/pupils); theta=np.arange(4*pupils)*2*np.pi/(4*pupils)
    pupil=np.array([(r*np.cos(t),r*np.sin(t)) for r in rad for t in theta])*3
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
                err=np.maximum.reduce([front-q[:,2],q[:,2]-back,abs(q[:,0])-10,abs(q[:,1])-5,np.zeros(len(q))])
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
    target=6*np.tan(np.deg2rad(tr['fields_deg']))
    # Choose only image parity, never fit away focal length or field mapping.
    sign=np.where(np.sum(center*target,axis=0)>=0,1,-1)
    mapping=(center-target*sign)/PIXEL_MM
    m=tr['model']; e,m1,m2,ex=m.surfaces
    gx,gy=np.meshgrid(np.linspace(-10,10,17),np.linspace(-5,5,11)); front,back=body(m,gx,gy)
    geom=np.array([max(0,.2-(m2.center[0]-m2.radius-e.center[0]-e.radius)),max(0,.2-(ex.center[0]-ex.radius-m1.center[0]-m1.radius)),max(0,np.max(back)-np.min(front)-10),max(0,.1-np.min(back-front))])
    return np.concatenate([np.clip(delta/PIXEL_MM,-100,100).ravel()/np.sqrt(delta.size),.3*mapping.ravel()/np.sqrt(mapping.size),20*(~tr['good']).ravel()/np.sqrt(tr['good'].size),10*tr['overflow'].ravel()/np.sqrt(tr['good'].size),20*tr['escape'].ravel()/np.sqrt(tr['good'].size),20*geom])

def evaluate(genes,fields=3,waves=3,pupils=5):
    tr=trace(genes,fields,waves,pupils); good=tr['good']; xy=tr['xy']
    count=good.sum(axis=2); center=np.sum(np.where(good[...,None],xy,0),axis=2)/np.maximum(count[...,None],1)
    rms=np.sqrt(np.sum(np.where(good,np.sum((xy-center[:,:,None,:])**2,axis=-1),0),axis=2)/np.maximum(count,1))
    rms=np.where(count>0,rms,np.nan)
    r=residuals(genes,fields=fields,waves=waves,pupils=pupils,_trace=tr)
    return dict(loss=float(r@r),throughput=float(good.mean()),min_throughput=float(np.min(good.mean(axis=2))),empty_cells=int(np.sum(count==0)),rms_spot_mm=float(np.nanmean(rms)) if np.any(count) else None,max_spot_mm=float(np.nanmax(rms)) if np.any(count) else None,valid_rays=int(good.sum()),launched_rays=int(good.size))




