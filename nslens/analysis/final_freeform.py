"""Experimental 2D final-mirror polynomial freeform; nearest-hit ray tracing.

Sag on the last mirror is c*u**2 + sum(a[k]*(u/30)**(k+3),k=0..3).
The four coefficients have units mm. Zero coefficients recover the existing tracer.
"""
import numpy as np
from nslens.analysis import variable_mirror as base
unit=base.unit

def decode(g,n,coefficients):
 m=base.decode(g,n);a=np.asarray(coefficients,dtype=float)
 if a.shape!=(4,) or not np.all(np.isfinite(a)):raise ValueError('Need four finite freeform coefficients in mm')
 m['ff_coeff']=a;return m

def sag(m,j,u):
 u=np.asarray(u);value=m['curves'][j]*u*u
 return value+np.where(np.asarray(j)==len(m['curves'])-1,sum(a*(u/30.)**k for k,a in enumerate(m['ff_coeff'],3)),0.)

def slope(m,j,u):
 u=np.asarray(u);value=2*m['curves'][j]*u
 return value+np.where(np.asarray(j)==len(m['curves'])-1,sum(k*a*(u/30.)**(k-1)/30. for k,a in enumerate(m['ff_coeff'],3)),0.)

def intersections(o,d,m):
 result=base.intersections(o,d,m)
 nonzero=np.flatnonzero(m['ff_coeff'])
 if not len(nonzero):return result
 j=len(m['curves'])-1;w=m['widths'][j];delta=o-m['vertices'][j];u=delta@m['tangents'][j];v=delta@m['normals'][j];du=d@m['tangents'][j];dv=d@m['normals'][j]
 last=np.full(len(o),np.inf);vertical=np.abs(du)<1e-12
 ids=np.flatnonzero(vertical & (np.abs(u)<=w) & (np.abs(dv)>1e-14))
 if len(ids):
  tt=(sag(m,j,u[ids])-v[ids])/dv[ids];last[ids]=np.where(tt>1e-7,tt,np.inf)
 ids=np.flatnonzero(~vertical)
 if len(ids):
  degree=int(nonzero[-1])+3;coef=np.zeros((len(ids),degree+1));ratio=dv[ids]/du[ids]
  coef[:,0]=-v[ids]+ratio*u[ids];coef[:,1]=-30*ratio;coef[:,2]=m['curves'][j]*900
  coef[:,3:]=m['ff_coeff'][:degree-2]
  companion=np.zeros((len(ids),degree,degree));companion[:,np.arange(1,degree),np.arange(degree-1)]=1
  companion[:,:,-1]=-coef[:,:-1]/coef[:,-1,None]
  roots=np.linalg.eigvals(companion);z=roots.real;tt=(30*z-u[ids,None])/du[ids,None]
  valid=(np.abs(roots.imag)<=1e-8*(1+np.abs(z))) & (np.abs(z)<=w/30+1e-10) & (tt>1e-7)
  # Independent substitution residual rejects inaccurate nearly-real eigenvalues.
  residual=np.abs(sag(m,j,30*z)-(v[ids,None]+dv[ids,None]*tt))
  valid &= residual <= 1e-6*(1+np.abs(sag(m,j,30*z)))
  last[ids]=np.min(np.where(valid,tt,np.inf),axis=1)
 result[:,-1]=last;return result


def trace(g, n, coefficients, fields=3, pupils=9, paths=False, *, focal=50.0,
          aperture=27.78, half_field_deg=4.0):
    """Trace all rays to nearest surfaces. Paths mode returns original 5-tuple."""
    for name, value in (("focal", focal), ("aperture", aperture),
                        ("half_field_deg", half_field_deg)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (("fields", fields), ("pupils", pupils)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f"{name} must be an integer >= 1")
    m = decode(g, n, coefficients)
    field = np.repeat(np.linspace(-half_field_deg, half_field_deg, fields), pupils)
    theta = np.deg2rad(field)
    pupil = np.tile(np.linspace(-aperture/2, aperture/2, pupils), fields)
    samples = np.linspace(-1., 1., 65)
    outlines = [v + samples[:, None]*w*t + sag(m,j,samples*w)[:, None]*normal
                for j,(v,t,normal,w) in enumerate(zip(m['vertices'],m['tangents'],m['normals'],m['widths']))]
    x0 = min(np.min(p[:, 0]) for p in outlines) - 20.
    o = np.column_stack((np.full(len(field), x0), pupil + x0*np.tan(theta)))
    d = np.column_stack((np.cos(theta), np.sin(theta)))
    active = np.ones(len(field), bool)
    hits = np.zeros(len(field), int)
    route = np.full((len(field), n+3), -1, int)
    image = np.full(len(field), np.nan)
    histories = [[p.copy()] for p in o] if paths else None
    for step in range(n+3):
        distances = intersections(o, d, m)
        which = np.argmin(distances, axis=1)
        tm = distances[np.arange(len(o)), which]
        den = d @ m['image_normal']
        with np.errstate(divide='ignore', invalid='ignore'):
            ti = ((m['image']-o) @ m['image_normal']) / den
        sensor = active & (ti > 1.e-7) & (ti < tm) & (den > 0)
        if np.any(sensor):
            p = o[sensor] + ti[sensor, None]*d[sensor]
            image[sensor] = (p-m['image']) @ m['image_tangent']
            if paths:
                for idx, point in zip(np.flatnonzero(sensor), p):
                    histories[idx].append(point.copy())
        reflect = active & ~sensor & np.isfinite(tm)
        active = reflect
        if not np.any(reflect):
            break
        ids = np.flatnonzero(reflect)
        j = which[ids]
        p = o[ids] + tm[ids, None]*d[ids]
        u = np.sum((p-m['vertices'][j])*m['tangents'][j], axis=1)
        normal = unit(m['normals'][j] - slope(m,j,u)[:, None]*m['tangents'][j])
        d[ids] -= 2*np.sum(d[ids]*normal, axis=1)[:, None]*normal
        o[ids] = p
        route[ids, step] = j
        hits[ids] += 1
        if paths:
            for idx, point in zip(ids, p):
                histories[idx].append(point.copy())
    good = np.isfinite(image) & (hits == n) & np.all(
        np.sort(route[:, :n], axis=1) == np.arange(n), axis=1)
    target = focal*np.tan(theta)
    errors = [np.where(good, np.minimum(abs(image-sign*target), 100.), 100.)
              for sign in (-1, 1)]
    parity = (-1, 1)[int(np.argmin([np.mean(e*e) for e in errors]))]
    err = errors[0 if parity == -1 else 1]
    support = good.reshape(fields, pupils).mean(axis=1)
    rmse = np.sqrt(np.mean(err.reshape(fields, pupils)**2, axis=1))
    progress = np.sum(np.cumprod(route[:, :n] == np.arange(n), axis=1), axis=1)
    loss = float(np.mean(err**2) + 10000*np.mean(~good)
                 + 1000*np.mean((n-progress)/n))
    rows = []
    field_mapping_rms = []
    for f in range(fields):
        mask = (np.arange(len(field))//pupils == f) & good
        values = image[mask]
        target_value = target[f*pupils]
        rows.append(dict(field_deg=float(field[f*pupils]), support=float(support[f]),
                         rms_spot_mm=float(np.std(values)) if len(values) else None,
                         centroid_error_mm=float(abs(np.mean(values)-parity*target_value))
                         if len(values) else None))
        if len(values):
            field_mapping_rms.append(float(np.sqrt(np.mean(
                (values-parity*target_value)**2))))
        else:
            field_mapping_rms.append(None)
    worst_rms_spot = (max(row["rms_spot_mm"] for row in rows)
                      if all(sum(good[f*pupils:(f+1)*pupils]) >= 2
                             for f in range(fields)) else None)
    worst_centroid_error = (max(row["centroid_error_mm"] for row in rows)
                            if all(sum(good[f*pupils:(f+1)*pupils]) >= 1
                                   for f in range(fields)) else None)
    worst_mapping_rms = (max(field_mapping_rms)
                         if all(value is not None for value in field_mapping_rms)
                         else None)
    result = dict(loss=loss, worst_support=float(min(support)),
                  target_rmse_with_loss_penalty_mm=float(max(rmse)), parity=parity,
                  fields=rows, genes=np.asarray(g).tolist(), n=n,
                  task=dict(focal=float(focal), aperture=float(aperture),
                            half_field_deg=float(half_field_deg), fields=int(fields),
                            pupils=int(pupils)),
                  worst_rms_spot_mm=worst_rms_spot,
                  worst_centroid_error_mm=worst_centroid_error,
                  worst_mapping_rms_mm=worst_mapping_rms)
    result['freeform_coefficients_mm']=m['ff_coeff'].tolist()
    result['ray_image_mm']=np.where(good,image,0.).tolist()
    result['ray_good']=good.tolist()
    result['ray_field_deg']=field.tolist()
    result['ray_route']=route.tolist()
    if paths:
        return result, m, outlines, histories, good
    return result

