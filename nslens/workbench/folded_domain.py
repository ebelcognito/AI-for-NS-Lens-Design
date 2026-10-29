"""Closed spline freeform AR/VR prism with explicitly required TIR delivery.

Archive-derived circular faces are only a starting scaffold. Endpoint-tapered
radial polynomial departures are sampled into ONE closed C2 spline. Every new
prescription is traced by the continuous-boundary Fresnel branching kernel.
"""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np

from ns_autodiff.prism_branch import build_arcs_ff, _unwrap
from nslens.freeform_body.boundary import FreeformBody2D
from nslens.freeform_body.trace import InputRay, trace_bundle, project_exits_to_x

PARAMETERS=tuple(f'{face}_p{power}' for face in ('A','B','T','D') for power in (2,4))
FACE_ORDER=('A','D','B','T')
REQUIREMENTS=dict(min_field_folded_throughput=.30,max_field_sigma_deg=1.,min_fov_deg=12.,
                  min_tir_bounces=1,max_unresolved_fraction=1e-4,max_closure_error=1e-9)
LAYOUT=dict(source_points=[[49.,50.],[55.,50.],[61.,50.]],source_aim_deg=-90.,
            fan_half_angle_deg=12.,pupil_x=120.,pupil_half_height=9.)


def surface_body(base,coefficients):
    x=np.asarray([*base,*coefficients],dtype=float)
    if x.shape!=(17,) or not np.isfinite(x).all() or np.any(np.abs(x[9:])>2):
        raise ValueError('nine fixed base values and eight bounded freeform coefficients required')
    arcs,n=build_arcs_ff(x);parts=[]
    for name in FACE_ORDER:
        a=arcs[name];ce=a['center']
        t1=_unwrap(math.atan2(*(a['c1']-ce)[::-1]),a['th_apex'])
        t2=_unwrap(math.atan2(*(a['c2']-ce)[::-1]),a['th_apex'])
        theta=np.linspace(t1,t2,32,endpoint=False)
        s=(theta-a['th_apex'])/a['half_span']
        radius=a['r']+(a['coeffs'][0]*s**2+a['coeffs'][1]*s**4)*(1-s*s)**2
        parts.append(ce+np.stack([radius*np.cos(theta),radius*np.sin(theta)],axis=1))
    return FreeformBody2D(np.concatenate(parts),n=n)


def rays_for(fidelity,layout=None):
    layout=LAYOUT if layout is None else layout
    count={'coarse':17,'dense':33,'audit':61,'convergence':91}[fidelity]
    positions=np.linspace(-1,1,count)
    if fidelity!='coarse':positions[1:-1]+=({'dense':.27,'audit':-.31,'convergence':.17}[fidelity])*2/(count-1)
    rays=[]
    for source in layout['source_points']:
        for position in positions:
            angle=np.deg2rad(layout['source_aim_deg']+layout['fan_half_angle_deg']*position)
            rays.append(InputRay(tuple(source),(float(np.cos(angle)),float(np.sin(angle))),source_id=len(rays)))
    return rays,count


def evaluate(base,coefficients,fidelity='coarse',*,return_trace=False,layout=None,requirements=None):
    # Environment-owned configurations; legacy tool tasks still require exact defaults.
    layout=LAYOUT if layout is None else layout
    requirements=REQUIREMENTS if requirements is None else requirements
    rays,count=rays_for(fidelity,layout)
    try:
        body=surface_body(base,coefficients)
        geometry=body.geometry_report(sample_count=512)
        if not geometry.valid:raise ValueError('; '.join(geometry.reasons))
        if any(body.contains(p) for p in layout['source_points']):raise ValueError('source inside optical body')
        if body.sample(512)[1][:,0].max()>=layout['pupil_x']-1:raise ValueError('body reaches pupil plane')
    except (ValueError,FloatingPointError) as exc:
        return dict(physical_valid=False,success=False,reason=str(exc),fields=[],merit=1e6,rays=len(rays))
    trace=trace_bundle(body,rays,energy_floor=1e-5 if fidelity=='coarse' else 1e-6,
        max_interactions=24,intersection_scan=1024 if fidelity=='coarse' else 2048)
    records=project_exits_to_x(trace,layout['pupil_x'])
    accepted=[r for r in records if abs(r['y'])<=layout['pupil_half_height'] and
              sum(e.kind=='tir' for e in r['events'])>=requirements['min_tir_bounces']]
    fields=[];routes=defaultdict(float)
    for field in range(3):
        rows=[r for r in accepted if r['source_id']//count==field]
        weights=np.array([r['energy'] for r in rows]);total=float(weights.sum())
        angles=np.array([np.rad2deg(np.arctan2(r['direction'][1],r['direction'][0])) for r in rows])
        mean=float(np.average(angles,weights=weights)) if total else 0.
        sigma=float(np.sqrt(np.average((angles-mean)**2,weights=weights))) if total else 180.
        histogram=defaultdict(float)
        contributors=defaultdict(float)
        for row in rows:
            folds=sum(e.kind=='tir' for e in row['events']);histogram[str(folds)]+=row['energy']/count
            contributors[row['source_id']]+=row['energy']
            route=' > '.join(e.kind+'@'+FACE_ORDER[min(3,int((e.u%1)*4))] for e in row['events'])
            routes[route]+=row['energy']/len(rays)
        fields.append(dict(folded_throughput=total/count,centroid_deg=mean,sigma_deg=sigma,
            pupil_ray_fraction=sum(v>=.1 for v in contributors.values())/count,
            throughput_by_tir_count=dict(histogram)))
    means=[f['centroid_deg'] for f in fields]
    ordering=bool(means[0]<means[1]<means[2] or means[0]>means[1]>means[2])
    fov=abs(means[2]-means[0])
    unresolved=(trace.depth_limited_energy+trace.branch_limited_energy)/len(rays)
    closure=abs(trace.closure_residual)/len(rays)
    physical=bool(unresolved<=requirements['max_unresolved_fraction'] and closure<=requirements['max_closure_error'])
    min_t=min(f['folded_throughput'] for f in fields);max_sigma=max(f['sigma_deg'] for f in fields)
    min_throughput=requirements['min_field_folded_throughput']
    max_spread=requirements['max_field_sigma_deg'];min_fov=requirements['min_fov_deg']
    ordering_gate=ordering or not requirements.get('require_ordering',True)
    success=bool(physical and ordering_gate and min_t>=min_throughput and max_sigma<=max_spread and fov>=min_fov)
    violation=sum(max(0.,(min_throughput-f['folded_throughput'])/min_throughput) + max(0.,f['sigma_deg']-max_spread) for f in fields)
    violation+=max(0.,(min_fov-fov)/max(min_fov,1e-12))+float(not ordering_gate)
    merit=100*float(not physical)+10*violation+float(np.mean([f['sigma_deg'] for f in fields]))
    result=dict(physical_valid=physical,success=success,reason='' if success else 'folded throughput, angular spread, FOV, ordering or trace gate',
        fields=fields,fov_deg=fov,ordering_ok=ordering,unresolved_fraction=unresolved,closure_error=closure,
        merit=merit,rays=len(rays),routes=sorted([dict(route=k,energy_fraction=v) for k,v in routes.items()],key=lambda r:-r['energy_fraction'])[:8])
    return (result,body,rays,trace,accepted) if return_trace else result
