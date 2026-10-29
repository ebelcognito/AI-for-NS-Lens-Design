"""Shared, bounded geometry space and outcome-only camera imaging objective."""
from dataclasses import asdict, dataclass, replace
from collections import defaultdict
import numpy as np
from nslens.discovery.polygon import PolygonCodec
from nslens.freeform_body.polynomial_faces import PolynomialFaceBody
from nslens.freeform_body.trace import InputRay, trace_bundle, project_exits_to_plane
from .scene import OpticalScene
from .sequential_grade import grade as sequential_grade


@dataclass(frozen=True)
class CameraCodec:
    corners: tuple = (4, 3, 3)
    n: float = 1.60
    departure_bound_mm: float = 0.
    placement_bounds: tuple = ()

    def __post_init__(self):
        if not 1 <= len(self.corners) <= 3 or any(k not in (3,4,5) for k in self.corners):
            raise ValueError('one to three bodies with 3/4/5 corners required')
        if not np.isfinite(self.departure_bound_mm) or not 0 <= self.departure_bound_mm <= .1:
            raise ValueError('departure bound must be in [0,.1] mm')
        if self.placement_bounds:
            if len(self.placement_bounds) != len(self.corners):
                raise ValueError('one x-placement interval per body required')
            for bounds in self.placement_bounds:
                if (len(bounds) != 2 or not np.isfinite(bounds).all()
                        or bounds[0] < 1 or bounds[1] > 22 or bounds[0] >= bounds[1]):
                    raise ValueError('placement intervals must lie within [1,22]')
        for index, k in enumerate(self.corners): self.polygon(k, index)

    def polygon(self, k, index=0):
        bounds = self.placement_bounds[index] if self.placement_bounds else (1.,22.)
        return PolygonCodec(k, n=self.n, center_x=bounds, center_y=(-1.,1.),
            half_width=(.5,3.), half_height=(.5,3.), min_radius_mm=.5,
            min_edge_mm=.1, max_sag_ratio=.5)

    @property
    def gene_size(self):
        return sum(5+4*k+(3*k if self.departure_bound_mm else 0) for k in self.corners)

    def design(self, genes):
        g = np.asarray(genes, dtype=float)
        if g.shape != (self.gene_size,) or not np.isfinite(g).all() or np.any(abs(g)>1):
            raise ValueError('invalid normalized camera genes')
        result=[]; offset=0
        for index,k in enumerate(self.corners):
            p=self.polygon(k,index); a=g[offset:offset+5+3*k]; offset+=len(a)
            def physical(x, bounds): return bounds[0]+(x+1)/2*(bounds[1]-bounds[0])
            item=dict(center_mm=[physical(a[0],p.center_x),physical(a[1],p.center_y)],
                half_size_mm=[physical(a[2],p.half_width),physical(a[3],p.half_height)],
                rotation_deg=float(180*a[4]),
                angular_offsets_deg=(a[5:5+k]*p.angular_jitter_fraction*360/k).tolist(),
                radial_fractions=[physical(v,p.radial_fraction) for v in a[5+k:5+2*k]],
                curvature_controls=a[5+2*k:].tolist(),
                mirrors=(g[offset:offset+k]>0).tolist(), departures_mm=np.zeros((k,3)).tolist())
            offset+=k
            if self.departure_bound_mm:
                item['departures_mm']=(g[offset:offset+3*k].reshape(k,3)*self.departure_bound_mm).tolist(); offset+=3*k
            result.append(item)
        return {'bodies':result}

    def encode(self, design):
        if not isinstance(design,dict) or set(design)!={'bodies'} or len(design['bodies'])!=len(self.corners):
            raise ValueError('design must contain exactly the declared bodies')
        values=[]
        expected={'center_mm','half_size_mm','rotation_deg','angular_offsets_deg','radial_fractions',
                  'curvature_controls','mirrors','departures_mm'}
        for index,(k,item) in enumerate(zip(self.corners,design['bodies'])):
            if not isinstance(item,dict) or set(item)!=expected:
                raise ValueError('body keys must match the geometry schema')
            p=self.polygon(k,index)
            for key,shape in [('center_mm',(2,)),('half_size_mm',(2,)),('angular_offsets_deg',(k,)),
                              ('radial_fractions',(k,)),('curvature_controls',(k,)),('departures_mm',(k,3))]:
                a=np.asarray(item[key],dtype=float)
                if a.shape!=shape or not np.isfinite(a).all(): raise ValueError('invalid '+key)
            if len(item['mirrors'])!=k or any(type(v) is not bool for v in item['mirrors']):
                raise ValueError('mirrors must be boolean flags')
            def normalized(x,b): return 2*(x-b[0])/(b[1]-b[0])-1
            values.extend(normalized(v,b) for v,b in zip(item['center_mm']+item['half_size_mm'],
                          (p.center_x,p.center_y,p.half_width,p.half_height)))
            values.append(float(item['rotation_deg'])/180)
            values.extend(np.asarray(item['angular_offsets_deg'])/(p.angular_jitter_fraction*360/k))
            values.extend(normalized(v,p.radial_fraction) for v in item['radial_fractions'])
            values.extend(item['curvature_controls']); values.extend(1. if v else -1. for v in item['mirrors'])
            c=np.asarray(item['departures_mm'])
            if self.departure_bound_mm: values.extend(c.ravel()/self.departure_bound_mm)
            elif np.any(c): raise ValueError('polynomial departures disabled in this protocol')
        g=np.asarray(values,dtype=float)
        if not np.isfinite(g).all() or np.any(abs(g)>1+1e-12): raise ValueError('geometry outside declared bounds')
        return np.clip(g,-1,1)

    def decode(self, genes, *, face_scan=64):
        # Round-trip validation also supplies the physical prescription for the model.
        design=self.design(genes); bodies=[]; offset=0
        for index,(k,item) in enumerate(zip(self.corners,design['bodies'])):
            b=self.polygon(k,index).decode(genes[offset:offset+5+3*k])
            offset+=5+4*k+(3*k if self.departure_bound_mm else 0)
            c=np.asarray(item['departures_mm'])
            if np.any(c): b=PolynomialFaceBody(b.corners,b.sags,b.n,c,face_scan=face_scan)
            bodies.append(b)
        return OpticalScene(bodies,[item['mirrors'] for item in design['bodies']])


@dataclass(frozen=True)
class CameraBrief:
    focal_length_mm: float = 12.24
    f_number: float = 2.
    half_fov_deg: float = 11.6
    sensor_half_mm: float = 2.52
    max_thickness_mm: float = 8.
    max_length_mm: float = 24.
    min_throughput: float = .5
    min_root_support: float = .8
    max_rms_mm: float = .10
    max_centroid_error_mm: float = .10
    rays_per_field: int = 5
    field_count: int = 3
    max_interactions: int = 32
    max_branches: int = 200000

    def __post_init__(self):
        if not all(np.isfinite(v) and v>0 for v in asdict(self).values()):
            raise ValueError('brief values must be positive and finite')
        if self.rays_per_field<5 or self.field_count<3 or self.field_count%2!=1:
            raise ValueError('at least 5 pupil rays and odd field count >=3 required')
        if any(type(getattr(self,k)) is not int for k in ('rays_per_field','field_count','max_interactions','max_branches')):
            raise ValueError('counts must be integers')
        if max(self.min_throughput,self.min_root_support)>1: raise ValueError('fractions must not exceed one')

    def rays(self):
        rays=[]
        for field in np.linspace(-self.half_fov_deg,self.half_fov_deg,self.field_count):
            theta=np.deg2rad(field)
            for p in np.linspace(-self.focal_length_mm/self.f_number/2,self.focal_length_mm/self.f_number/2,self.rays_per_field):
                rays.append(InputRay((4.+p,self.max_thickness_mm/2+2),
                    (np.sin(theta),-np.cos(theta)),source_id=len(rays)))
        return rays


def evaluate(codec, genes, brief, *, dense=False, return_trace=False):
    row=dict(genes=np.asarray(genes).tolist(),status='geometry',feasible=False,source_rays=0,
             fields=[],quality=0.,support=0.,route=None,violations={},progress=(0.,0.,0.))
    try:
        scene=codec.decode(genes,face_scan=128 if dense else 64)
        lo,hi=scene.validate(samples=1024 if dense else 256)
        if lo[0]<0 or hi[0]>brief.max_length_mm-.1 or lo[1]<-brief.max_thickness_mm/2 or hi[1]>brief.max_thickness_mm/2:
            raise ValueError('geometry outside camera envelope')
    except (ValueError,TypeError,IndexError,FloatingPointError) as e:
        row['reason']=str(e); return (row,None,None) if return_trace else row
    rays=brief.rays()
    trace=trace_bundle(scene,rays,max_interactions=brief.max_interactions,
                       max_branches=brief.max_branches,energy_floor=1e-6)
    records=project_exits_to_plane(trace,axis=0,value=brief.max_length_mm,travel_sign=1)
    row.update(status='measured',source_rays=len(rays),thickness_mm=float(hi[1]-lo[1]))
    # Score every sensor-reaching branch. Exclude direct bypass rays from signal;
    # report them as stray energy. Do not select a favourable path or pupil subset.
    rows=[r for r in records if abs(r['coord'])<=brief.sensor_half_mm]
    signal=[r for r in rows if r['events']]
    row['bypass_fraction']=sum(r['energy'] for r in rows if not r['events'])/len(rays)
    fields=np.linspace(-brief.half_fov_deg,brief.half_fov_deg,brief.field_count)
    # One common parity for every field; never fit independent image targets.
    def mapping_error(parity):
        return sum(r['energy']*(r['coord']-parity*brief.focal_length_mm*np.tan(np.deg2rad(fields[r['source_id']//brief.rays_per_field])))**2 for r in signal)
    parity=min((-1,1),key=mapping_error); row['image_parity']=parity
    row['sequential_grade'] = sequential_grade(scene, trace, brief, parity)
    violations={}; quality=[]; supports=[]; routes=defaultdict(float)
    for i,field in enumerate(fields):
        rr=[r for r in signal if r['source_id']//brief.rays_per_field==i]
        target=float(parity*brief.focal_length_mm*np.tan(np.deg2rad(field)))
        energy=sum(r['energy'] for r in rr); throughput=energy/brief.rays_per_field
        roots=defaultdict(float)
        for r in rr:
            roots[r['source_id']]+=r['energy']
            key=tuple((e.body_id,e.face_id,e.kind) for e in r['events'])
            routes[key]+=r['energy']
        support=sum(v>=.1 for v in roots.values())/brief.rays_per_field
        centroid=sum(r['energy']*r['coord'] for r in rr)/energy if energy else 0.
        rms=float(np.sqrt(sum(r['energy']*(r['coord']-centroid)**2 for r in rr)/energy)) if energy else None
        error=abs(centroid-target) if energy else None
        row['fields'].append(dict(field_deg=float(field),target_mm=target,throughput=throughput,
            root_support=support,centroid_mm=centroid if energy else None,rms_mm=rms,centroid_error_mm=error))
        violations[f'throughput_{i}']=max(0.,1-throughput/brief.min_throughput)
        violations[f'support_{i}']=max(0.,1-support/brief.min_root_support)
        # Bounded smooth errors give routing progress before imaging is achieved.
        violations[f'rms_{i}']=1. if rms is None else max(0.,1-brief.max_rms_mm/max(rms,1e-12))
        violations[f'centroid_{i}']=1. if error is None else max(0.,1-brief.max_centroid_error_mm/max(error,1e-12))
        quality.append(throughput/(1+(rms or 0)/brief.max_rms_mm+(error or 0)/brief.max_centroid_error_mm))
        supports.append(support)
    unresolved=(trace.depth_limited_energy+trace.branch_limited_energy)/len(rays)
    pruned=trace.pruned_energy/len(rays); closure=abs(trace.closure_residual)/len(rays)
    violations.update(unresolved=max(0.,unresolved/1e-4-1),pruned=max(0.,pruned/1e-3-1),closure=max(0.,closure/1e-9-1))
    row.update(violations=violations,feasible=all(v<=1e-12 for v in violations.values()),
        quality=min(quality),support=min(supports),progress=(min(quality),float(np.mean(quality)),float(np.mean(supports))),
        route=repr(max(routes,key=routes.get)) if routes else None,
        energy=dict(unresolved=unresolved,pruned=pruned,closure=closure),
        folded_signal_fraction=sum(r['energy'] for r in signal if any(e.kind in ('tir','mirror') for e in r['events']))/len(rays))
    return (row,scene,trace) if return_trace else row


def audit(codec, genes, brief):
    dense=replace(brief,field_count=7,rays_per_field=25)
    first=evaluate(codec,genes,dense,dense=True)
    second=evaluate(codec,genes,replace(dense,max_branches=2*dense.max_branches),dense=True)
    agree=(first['status']==second['status'] and first['fields']==second['fields'] and first.get('energy')==second.get('energy'))
    return dict(passed=first['feasible'] and second['feasible'] and agree,
                branch_capacity_agreement=agree,checks=[first,second],
                source_rays=first['source_rays']+second['source_rays'])
