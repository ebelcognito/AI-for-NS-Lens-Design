"""Convex 3D dielectric pilot. Millimetres, linear monochromatic radiance.

Scalar unpolarized Fresnel at each bounce: polarization memory is approximated.
One convex solid in air; no absorption, dispersion, scattering, or coatings.
"""
import numpy as np


def unit(x):
    x = np.asarray(x, float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def interface(d, normal, n1, n2):
    """Normal points into incident medium; return reflected, transmitted, R, TIR."""
    d, normal = np.broadcast_arrays(np.asarray(d, float), np.asarray(normal, float))
    ci = np.clip(-np.sum(d * normal, axis=-1), 0, 1)
    eta = n1 / n2
    k = 1 - eta * eta * (1 - ci * ci)
    ct = np.sqrt(np.maximum(k, 0))
    reflected = d + 2 * ci[..., None] * normal
    transmitted = eta * d + (eta * ci - ct)[..., None] * normal
    rs = ((n1 * ci - n2 * ct) / np.maximum(n1 * ci + n2 * ct, 1e-30)) ** 2
    rp = ((n2 * ci - n1 * ct) / np.maximum(n2 * ci + n1 * ct, 1e-30)) ** 2
    tir = k < 0
    return reflected, transmitted, np.where(tir, 1., (rs + rp) / 2), tir


def geometry(g):
    """Six bounded genes; 8-fold crown/pavilion with bounded 2-fold asymmetry.

    Fixed 10 mm girdle diameter; thickness .3 mm. This is a simple tiered
    reference family, NOT a conventional 57-facet brilliant prescription.
    """
    g = np.asarray(g, float)
    if g.shape != (6,) or np.any(abs(g) > 1 + 1e-10):
        raise ValueError('Expected six genes in [-1, 1]')
    a = np.arange(8) * np.pi / 4
    crown = np.deg2rad(32 + 12 * g[0] + 5 * g[3] * np.cos(2*a))
    pavilion = np.deg2rad(43 + 12 * g[1] + 5 * g[4] * np.cos(2*a))
    table = .53 + .18 * g[2]
    pa = a + .16 * g[5]
    cn = np.column_stack([np.cos(a)*np.sin(crown), np.sin(a)*np.sin(crown), np.cos(crown)])
    pn = np.column_stack([np.cos(pa)*np.sin(pavilion), np.sin(pa)*np.sin(pavilion), -np.cos(pavilion)])
    gn = np.column_stack([np.cos(a), np.sin(a), np.zeros(8)])
    radius = 5 * np.cos(np.pi/8)
    h = (1-table)*radius*np.tan(np.deg2rad(32+12*g[0]))
    normals = np.vstack([cn, pn, gn, [0,0,1], [0,0,-1]])
    offsets = np.r_[radius*np.sin(crown)+.15*np.cos(crown), radius*np.sin(pavilion)+.15*np.cos(pavilion), np.full(8,radius), .15+h, .15+radius*np.tan(np.deg2rad(43+12*g[1]))*.97]
    return normals, offsets


def intersect(o, d, normals, offsets):
    """Convex half-space interval intersection; returns nearest forward boundary."""
    den = d @ normals.T
    num = offsets - o @ normals.T
    parallel = abs(den) < 1e-12
    t = num / np.where(parallel, 1., den)
    enter = np.where(den < -1e-12, t, -np.inf)
    leave = np.where(den > 1e-12, t, np.inf)
    i0, i1 = np.argmax(enter, axis=1), np.argmin(leave, axis=1)
    rows = np.arange(len(o))
    near, far = enter[rows,i0], leave[rows,i1]
    valid = (far >= np.maximum(near,0)) & ~np.any(parallel & (num < -1e-9), axis=1) & (far > 1e-8)
    outside = near > 1e-8
    dist = np.where(outside,near,far)
    idx = np.where(outside,i0,i1)
    return dist, normals[idx], valid


def environment(d, kind='spots'):
    """Fixed world-space illumination; finite smooth sources, camera-head shadow."""
    if kind == 'constant':
        return np.ones(len(d))
    specs = {
        'spots': ([[-.5,.3,1],[.6,.25,1],[.1,-.65,1]], .075),
        'window': ([[-.7,.2,1]], .30),
        'holdout': ([[.45,-.35,1],[-.4,-.6,.8],[.7,.5,.65]], .12),
    }
    lights, width = specs[kind]
    dots = d @ unit(lights).T
    value = .025 + np.sum(np.exp((dots-1)/(width*width)),axis=1)
    value *= np.where(d[:,2] > np.cos(np.deg2rad(9)), .08, 1.)
    return value


def rotation(ax, ay):
    x,y=np.deg2rad([ax,ay]); cx,sx,cy,sy=np.cos(x),np.sin(x),np.cos(y),np.sin(y)
    return np.array([[cy,sy*sx,sy*cx],[0,cx,-sx],[-sy,cy*sx,cy*cx]])


def render(g, size=48, pose=(0,0), lighting='spots', index=1.6, bounces=32, cutoff=1e-7):
    """Deterministic camera paths with both Fresnel contributions, fixed exposure.

    Convexity means every transmitted internal branch escapes. Only the reflected
    branch continues; unresolved camera-path weight is reported, never relabelled.
    Endpoint media are air, so paired refractive radiance factors cancel.
    """
    normals, offsets=geometry(g); rot=rotation(*pose)
    grid=(np.arange(size)+.5)/size*12-6
    xx,yy=np.meshgrid(grid,grid)
    o=np.column_stack([xx.ravel(),yy.ravel(),np.full(size*size,15.)]) @ rot
    d=np.tile([0.,0.,-1.],(len(o),1)) @ rot
    t,n,mask=intersect(o,d,normals,offsets)
    ids=np.flatnonzero(mask); out=np.zeros(len(o)); escaped=np.zeros(len(o)); residual=np.zeros(len(o))
    if not len(ids):
        return out.reshape(size,size),mask.reshape(size,size),dict(residual=0.,closure_error=0.)
    p=o[ids]+t[ids,None]*d[ids]
    refl,din,r,_=interface(d[ids],n[ids],1.,index)
    out[ids]=r*environment(refl@rot.T,lighting); escaped[ids]=r
    weights=1-r; o=p+din*1e-7; d=din
    for _ in range(bounces):
        if not len(ids): break
        t,n,valid=intersect(o,d,normals,offsets)
        if not np.all(valid):
            # Edge/grazing offsets can place a path outside a very thin feature.
            # Preserve its weight as unresolved; never count it as escaped light.
            residual[ids[~valid]]+=weights[~valid]
            ids=ids[valid]; weights=weights[valid]
            o=o[valid]; d=d[valid]; t=t[valid]; n=n[valid]
            if not len(ids): break
        p=o+t[:,None]*d
        refl,trans,r,tir=interface(d,-n,index,1.)
        wout=weights*(1-r)
        out[ids]+=wout*environment(trans@rot.T,lighting)
        escaped[ids]+=wout
        weights*=r
        keep=weights>cutoff
        residual[ids[~keep]]+=weights[~keep]
        ids=ids[keep]; weights=weights[keep]; d=refl[keep]; o=p[keep]+d*1e-7
    residual[ids]+=weights
    return out.reshape(size,size),mask.reshape(size,size),dict(residual=float(np.mean(residual[mask])),max_residual=float(np.max(residual[mask])),closure_error=float(np.max(abs(escaped[mask]+residual[mask]-1))))


def sequence(g, size=32, frames=9, lighting='spots', axis=0, bounces=32):
    images=[]; masks=[]; audits=[]
    for angle in np.linspace(-15,15,frames):
        pose=(angle,0) if axis==0 else (0,angle)
        im,mask,audit=render(g,size,pose,lighting,bounces=bounces)
        images.append(im); masks.append(mask); audits.append(audit)
    return np.array(images),np.array(masks),audits


def metrics(images,masks):
    from scipy.ndimage import label
    means=np.array([im[m].mean() for im,m in zip(images,masks)])
    areas=[]; counts=[]
    for im,m in zip(images,masks):
        lab,n=label((im>.3)&m)
        sizes=np.bincount(lab.ravel())[1:]
        areas.append(float(np.sum(sizes*sizes)/max(np.sum(sizes),1)/m.sum()))
        counts.append(n)
    common=masks[1:]&masks[:-1]
    activity=np.sum(abs(np.diff(images,axis=0))*common)/max(common.sum(),1)/(30/max(len(images)-1,1))
    return dict(brightness=float(means.mean()),lower_brightness=float(np.quantile(means,.1)),flash_area=float(np.mean(areas)),flash_count=float(np.mean(counts)),activity=float(activity),dark_fraction=float(np.mean(images[masks]<.04)))
