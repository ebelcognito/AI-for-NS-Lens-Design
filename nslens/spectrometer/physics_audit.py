"""Independent polynomial-root intersection audit for spectrometer finalists.

Unlike the search's single-seed Newton solve, enumerate every polynomial root
and choose the nearest positive root inside the finite circular aperture. This
checks root selection, not a separate optical engine or mechanical clearance.
"""
from __future__ import annotations
import argparse
import json
from math import comb
from pathlib import Path
import numpy as np
from . import model


def exact_intersect(o,d,surface):
    o=np.asarray(o);d=np.asarray(d);b=surface['basis']
    a=(o-surface['vertex'])@b;v=d@b
    terms=surface.get('terms',model.TERMS)
    degree=max(1,max((i+j for i,j in terms),default=1))
    coefficients=np.zeros((len(o),degree+1));coefficients[:,0]=a[:,2];coefficients[:,1]=v[:,2]
    for c,(i,j) in zip(surface['coeff'],terms):
        if c==0:continue
        for u in range(i+1):
            for w in range(j+1):
                coefficients[:,u+w]-=c*comb(i,u)*comb(j,w)*a[:,0]**(i-u)*v[:,0]**u*a[:,1]**(j-w)*v[:,1]**w
    ts=np.zeros(len(o));good=np.zeros(len(o),bool)
    for index,row in enumerate(coefficients):
        if not np.isfinite(row).all():continue
        # Trim only mathematically zero terms, not small but physical coefficients.
        nonzero=np.flatnonzero(row)
        if not len(nonzero) or nonzero[-1]==0:continue
        roots=np.roots(row[:nonzero[-1]+1][::-1])
        roots=roots.real[(abs(roots.imag)<1e-7*(1+abs(roots.real)))&(roots.real>1e-6)]
        if not len(roots):continue
        p=a[index]+roots[:,None]*v[index]
        z,_,_=model.sag(p[:,0],p[:,1],surface['coeff'],surface.get('terms'))
        usable=(np.sum(p[:,:2]**2,axis=1)<=surface['aperture']**2+1e-9)&(abs(p[:,2]-z)<1e-6)
        if np.any(usable):ts[index]=np.min(roots[usable]);good[index]=True
    local=a+ts[:,None]*v
    _,zx,zy=model.sag(local[:,0],local[:,1],surface['coeff'],surface.get('terms'))
    normals=model.unit(np.column_stack([-zx,-zy,np.ones(len(o))])@b.T)
    return o+ts[:,None]*d,normals,ts,good


def audit(g,fields=8,waves=8,pupils=10,pupil_xy=None,aperture_radii=None):
    sampling=dict(fields=fields,waves=waves,pupils=pupils)
    original=model.intersect
    nominal=model.trace(g,**sampling,pupil_xy=pupil_xy,aperture_radii=aperture_radii)
    try:
        model.intersect=exact_intersect
        independent=model.trace(g,**sampling,pupil_xy=pupil_xy,aperture_radii=aperture_radii)
        metrics=model.evaluate(g,**sampling,pupil_xy=pupil_xy,aperture_radii=aperture_radii)
    finally:
        model.intersect=original
    common=nominal['good']&independent['good']
    delta=np.linalg.norm(nominal['xy'][common]-independent['xy'][common],axis=1)
    return dict(sampling=sampling,launched_rays=int(common.size),
                received_by_both=int(common.sum()),
                good_mask_disagreements=int(np.sum(nominal['good']!=independent['good'])),
                newton_only_received=int(np.sum(nominal['good']&~independent['good'])),
                polynomial_only_received=int(np.sum(independent['good']&~nominal['good'])),
                max_common_image_difference_mm=float(delta.max()) if len(delta) else None,
                polynomial_root_metrics=metrics,geometry=model.geometry_metrics(g,aperture_radii=aperture_radii),
                aperture_radii=aperture_radii,
                limitation='Independent root enumeration; same sag, diffraction and objective formulas. Not an independent optical engine or clearance certification.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fields',type=int,default=8);parser.add_argument('--waves',type=int,default=8);parser.add_argument('--pupils',type=int,default=10)
    args=parser.parse_args();row=json.loads(args.result.read_text())
    apertures=row.get('configuration',{}).get('aperture_radii')
    result=audit(row['genes'],args.fields,args.waves,args.pupils,aperture_radii=apertures)
    result['source_result']=str(args.result)
    result['genes']=row['genes']
    from .search import write_json
    write_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='geometry'},indent=2))



def footprint_curvature_metrics(g, fields=8, waves=8, pupils=10, margin=1., grid=61, pupil_xy=None, aperture_radii=None):
    """Sample signed principal curvature over the illuminated convex hull plus margin.

    The margin offsets convex-hull half-spaces; it is an outer approximation to
    a circular Euclidean dilation. Values are sampled, not certified extrema.
    Only received rays define the footprint; missing rays are reported separately.
    """
    from scipy.spatial import ConvexHull
    r=model.trace(g,fields=fields,waves=waves,pupils=pupils,pupil_xy=pupil_xy,aperture_radii=aperture_radii)
    good=r['good'].ravel();rows=[]
    for j,surface in enumerate(r['model']['surfaces']):
        local=(r['paths'][j+1][good]-surface['vertex'])@surface['basis']
        xy=np.unique(local[:,:2],axis=0)
        if len(xy)<3:
            rows.append(dict(surface=j,status='insufficient_footprint'));continue
        hull=ConvexHull(xy);equations=hull.equations
        lo=xy.min(axis=0)-margin;hi=xy.max(axis=0)+margin
        xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],grid),np.linspace(lo[1],hi[1],grid))
        candidates=np.column_stack([xx.ravel(),yy.ravel()])
        inside=np.all(candidates@equations[:,:2].T+equations[:,2]<=margin+1e-10,axis=1)
        candidates=np.concatenate([candidates[inside],xy])
        candidates=candidates[np.sum(candidates*candidates,axis=1)<=surface['aperture']**2]
        hull_count=len(candidates)
        required_radius=float(np.linalg.norm(xy,axis=1).max()+margin)
        radius=min(required_radius,surface['aperture'])
        cx,cy=np.meshgrid(np.linspace(-radius,radius,grid),np.linspace(-radius,radius,grid))
        disk=cx*cx+cy*cy<=radius*radius+1e-12
        candidates=np.concatenate([candidates,np.column_stack([cx[disk],cy[disk]])])
        circle_end=len(candidates)
        physical_radius=surface['aperture']
        cx,cy=np.meshgrid(np.linspace(-physical_radius,physical_radius,grid),np.linspace(-physical_radius,physical_radius,grid))
        disk=cx*cx+cy*cy<=physical_radius*physical_radius+1e-12
        candidates=np.concatenate([candidates,np.column_stack([cx[disk],cy[disk]])])
        x,y=candidates.T
        _,zx,zy=model.sag(x,y,surface['coeff'])
        zxx=np.zeros_like(x);zxy=np.zeros_like(x);zyy=np.zeros_like(x)
        for c,(i,k) in zip(surface['coeff'],model.TERMS):
            if i>=2:zxx+=c*i*(i-1)*x**(i-2)*y**k
            if i and k:zxy+=c*i*k*x**(i-1)*y**(k-1)
            if k>=2:zyy+=c*k*(k-1)*x**i*y**(k-2)
        scale=np.sqrt(1.+zx*zx+zy*zy)
        first=np.empty((len(x),2,2));second=first.copy()
        first[:,0,0]=1.+zx*zx;first[:,0,1]=first[:,1,0]=zx*zy;first[:,1,1]=1.+zy*zy
        second[:,0,0]=zxx/scale;second[:,0,1]=second[:,1,0]=zxy/scale;second[:,1,1]=zyy/scale
        principal=np.linalg.eigvals(np.linalg.solve(first,second)).real
        low=float(principal[:hull_count].min());high=float(principal[:hull_count].max())
        circle_low=float(principal[hull_count:circle_end].min());circle_high=float(principal[hull_count:circle_end].max())
        physical_low=float(principal[circle_end:].min());physical_high=float(principal[circle_end:].max())
        expected='positive_convex' if j==1 else 'negative_concave'
        preserved=low>0 if j==1 else high<0
        rows.append(dict(surface=j,status='sampled',sample_count=hull_count,
                         physical_aperture_circle=dict(radius_mm=physical_radius,
                             min_signed_curvature_per_mm=physical_low,max_signed_curvature_per_mm=physical_high,
                             expected_sign_preserved=bool(physical_low>0 if j==1 else physical_high<0)),
                         enclosing_circle=dict(required_radius_mm=required_radius,audited_radius_mm=radius,
                             fits_nominal_aperture=bool(required_radius<=surface['aperture']),
                             min_signed_curvature_per_mm=circle_low,max_signed_curvature_per_mm=circle_high,
                             expected_sign_preserved=bool(circle_low>0 if j==1 else circle_high<0)),
                         footprint_min_xy_mm=xy.min(axis=0).tolist(),footprint_max_xy_mm=xy.max(axis=0).tolist(),
                         footprint_max_radius_mm=float(np.linalg.norm(xy,axis=1).max()),
                         sampled_min_signed_principal_curvature_per_mm=low,
                         sampled_max_signed_principal_curvature_per_mm=high,
                         sampled_max_slope=float(np.hypot(zx[:hull_count],zy[:hull_count]).max()),
                         expected_curvature=expected,expected_sign_preserved=bool(preserved)))
    return dict(sampling=dict(fields=fields,waves=waves,pupils=pupils),margin_mm=margin,
                received_fraction=float(good.mean()),surfaces=rows,
                limitation='Sampled received-ray convex hull with outward half-space margin; not an analytic curvature or clearance certificate.')


def circular_aperture_separation(g, footprint, aperture_radii=None):
    """Conservative bounding-sphere gaps for suggested trimmed circular surfaces.

    For each monomial, maximize |x**i y**j| analytically on a disk; the sum
    bounds absolute sag. The resulting sphere encloses the whole polynomial
    optical surface, not just samples. Positive gaps exclude surface intersections
    for these trimmed apertures, but do not account for substrates or mounts.
    """
    m=model.decode(g,aperture_radii=aperture_radii);circles=[]
    for surface,row in zip(m['surfaces'],footprint['surfaces']):
        radius=float(surface['aperture']) if aperture_radii is not None else row['enclosing_circle']['required_radius_mm']
        bound=sum(abs(c)*radius**(i+j)*(i/(i+j))**(i/2)*(j/(i+j))**(j/2)
                  for c,(i,j) in zip(surface['coeff'],model.TERMS))
        circles.append(dict(aperture_radius_mm=radius,absolute_sag_bound_mm=float(bound),
                            enclosing_ball_radius_mm=float(np.hypot(radius,bound))))
    pairs=[]
    for i in range(3):
        for j in range(i+1,3):
            distance=float(np.linalg.norm(m['surfaces'][i]['vertex']-m['surfaces'][j]['vertex']))
            gap=distance-circles[i]['enclosing_ball_radius_mm']-circles[j]['enclosing_ball_radius_mm']
            pairs.append(dict(surface_pair=[i,j],conservative_gap_lower_bound_mm=float(gap)))
    return dict(trimmed_circular_surfaces=circles,pairs=pairs,
                aperture_source='configured_physical_circles' if aperture_radii is not None else 'suggested_footprint_plus_margin_circles',
                all_surfaces_disjoint_by_bound=bool(all(row['conservative_gap_lower_bound_mm']>1e-9 for row in pairs)),
                limitation='Applies to the listed circular optical surfaces only; excludes substrates, detector housing and mounts.')


def dense_hessian_sign_audit(g, aperture_radii, radial_samples=129, azimuth_samples=1440):
    """Independent monomial Hessians, including dense full physical-aperture rims."""
    m=model.decode(g,aperture_radii=aperture_radii)
    angles=np.arange(azimuth_samples)*2*np.pi/azimuth_samples
    radius=np.linspace(0,1,radial_samples)
    xx=(radius[:,None]*np.cos(angles)).ravel();yy=(radius[:,None]*np.sin(angles)).ravel()
    rows=[]
    for index,surface in enumerate(m['surfaces']):
        x=xx*surface['aperture'];y=yy*surface['aperture'];h=np.zeros((len(x),2,2))
        for c,(i,j) in zip(surface['coeff'],model.TERMS):
            if i>=2:h[:,0,0]+=c*i*(i-1)*x**(i-2)*y**j
            if i and j:h[:,0,1]+=c*i*j*x**(i-1)*y**(j-1)
            if j>=2:h[:,1,1]+=c*j*(j-1)*x**i*y**(j-2)
        h[:,1,0]=h[:,0,1];eig=np.linalg.eigvalsh(h)
        low=float(eig.min());high=float(eig.max())
        rows.append(dict(surface=index,aperture_radius_mm=surface['aperture'],sample_count=len(x),
                         minimum_hessian_eigenvalue_per_mm=low,maximum_hessian_eigenvalue_per_mm=high,
                         expected_sign_preserved=bool(low>0 if index==1 else high<0)))
    return dict(radial_samples=radial_samples,azimuth_samples=azimuth_samples,surfaces=rows,
                all_expected_signs_preserved=all(r['expected_sign_preserved'] for r in rows),
                limitation='Dense sampled graph-Hessian definiteness; equivalent principal-curvature signs, not an analytic global certificate.')


if __name__=='__main__':main()
