"""Run: python -m nslens.patent_camera.run --output results/patent_camera_spherical"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import load_prescription, Surface, PatentScene
from nslens.freeform_body.trace import InputRay, trace_bundle

EXPECTED = [(0,'refract'),(1,'refract'),(2,'refract'),(3,'mirror'),(2,'tir'),
            (4,'refract'),(5,'refract'),(6,'tir'),(7,'mirror'),(6,'refract')]


def spherical_prescription():
    p=load_prescription()
    for s in p['surfaces']:
        original=s['kind']
        if original=='xy':
            s['radius_mm']=1/(2*s['coefficients']['Y2'])
        s['approximation']='published base radius' if original!='xy' else 'R = 1/(2*Y2), meridional quadratic curvature'
        s['kind']='asphere'
        s['coefficients']={}
    entrance=Surface(p['surfaces'][5],2,1.671)
    p['surfaces'][5]['origin_zy_mm']=(entrance.origin+1.5*entrance.axis).tolist()
    p['model']='spherical meridional approximation; chief rays only'
    p['assumptions']=[
        'All aspheric/freeform coefficients removed. XY faces use R=1/(2*Y2); other faces retain published base radii.',
        'P2S1 is shifted +1.5 mm along its local sag axis to obtain separated, closed spherical prisms. This is an explicit layout adjustment, not patent data.',
        'Spherical faces are trimmed at their mutual intersections; lens clear semi-aperture is 3.15 mm.',
        'Only chief rays through stop center are evaluated. No f/2 imaging, focus, spot-size, or MTF validation is claimed.',
        'Monochromatic d-line refractive indices from Table 1; no dispersion model.',
        'Ideal lossless mirrors on P1S2/P2S3, uncoated Fresnel/TIR on remaining interfaces.',
        'IR filter omitted because its thickness/index/position are not specified in the prescription tables.',
        'Sensor cross-section spans the published semi-diagonal +/-2.52 mm; this is not a specified sensor width/height.'
    ]
    return p


def chief(scene, field, epsilon=1e-7):
    a=np.deg2rad(field);direction=(float(np.cos(a)),float(np.sin(a)))
    origin=(-1.,-float(np.tan(a))) # chief passes through stop center at (Z,Y)=(0,0)
    result=trace_bundle(scene,[InputRay(origin,direction)],energy_floor=1e-4,
        max_interactions=24,max_branches=20000,ray_epsilon=epsilon)
    ray=result.dominant_exit()
    if ray is None:raise RuntimeError('no exiting chief-ray branch')
    route=[(e.face_id,e.kind) for e in ray.events]
    o=np.array(ray.origin);d=np.array(ray.direction)
    sensor=scene.prescription['sensor'];t=(sensor['z_mm']-o[0])/d[0]
    point=o+t*d
    in_sensor=bool(t>0 and abs(point[1]-sensor['y_mm'])<=2.52)
    record={'field_deg':field,'route_correct':route==EXPECTED,'hits_sensor':in_sensor,
        'sensor_y_mm':float(point[1]),'sensor_relative_y_mm':float(point[1]-sensor['y_mm']),
        'dominant_branch_power_fraction':ray.energy,'energy_closure_residual':result.closure_residual,
        'pruned_power':result.pruned_energy,'interaction_limited_power':result.depth_limited_energy,
        'branch_limited_power':result.branch_limited_energy,
        'events':[dict(asdict(e),surface=scene.surfaces[e.face_id].name) for e in ray.events]}
    return record,np.array([origin]+[e.point for e in ray.events]+[tuple(point)])


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',default='results/patent_camera_spherical')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    p=spherical_prescription();scene=PatentScene(p,scan=256);scene.validate()
    fields=(-11.6,-5.8,0.,5.8,11.6)
    traces=[chief(scene,f) for f in fields]
    # Check all chief rays with doubled bracketing resolution and smaller offsets.
    fine=PatentScene(p,scan=512);fine.validate()
    checks=[chief(fine,f,epsilon=1e-8)[0] for f in fields]
    delta=max(abs(a[0]['sensor_y_mm']-b['sensor_y_mm']) for a,b in zip(traces,checks))
    valid=all(r['route_correct'] and r['hits_sensor'] for r,_ in traces)
    stable=all(a[0]['route_correct']==b['route_correct'] for a,b in zip(traces,checks)) and delta<1e-5
    report={'model':p['model'],'all_chiefs_follow_route_and_hit_sensor':valid,
        'convergence_pass':stable,'max_sensor_change_mm':delta,
        'fields':[r for r,_ in traces],'assumptions':p['assumptions']}
    (out/'prescription_spherical.json').write_text(json.dumps(p,indent=2)+'\n')
    (out/'chief_rays.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,9),gridspec_kw={'width_ratios':[1.2,1]})
    ax=axes[0]
    for polygon,label,color in zip(scene.polygons,['Objective','Prism 1','Prism 2'],['#c9e4f1','#bedbdc','#d4ccec']):
        ax.fill(polygon[:,0],polygon[:,1],color=color,alpha=.8,ec='#34495e',lw=1)
    colors=plt.cm.coolwarm(np.linspace(0,1,len(fields)))
    colors[2]=matplotlib.colors.to_rgba('#137547')
    for (record,points),color in zip(traces,colors):
        ax.plot(points[:,0],points[:,1],color=color,lw=2.3 if record['field_deg']==0 else 1.2,label=f"{record['field_deg']:+g}° chief")
    for face in (scene.surfaces[3],scene.surfaces[7]):
        ax.plot(face.points[:,0],face.points[:,1],color='#202938',lw=2.6)
    y=p['sensor']['y_mm'];z=p['sensor']['z_mm']
    ax.plot([z,z],[y-2.52,y+2.52],color='black',lw=3,label='Sensor section')
    ax.scatter([0],[0],c='black',s=22,zorder=5)
    ax.text(-.55,3.8,'Objective',fontsize=9)
    ax.text(2.4,2.5,'Prism 1',fontsize=10)
    ax.text(.25,-10.5,'Prism 2',fontsize=10)
    ax.set(xlabel='Patent Z (mm)',ylabel='Patent Y (mm)',title='Spherical layout · nearest-hit ray tracing')
    ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='lower left',fontsize=8)
    ax=axes[1]
    heights=[r['sensor_relative_y_mm'] for r,_ in traces]
    ax.plot(fields,heights,'o-',color='#247b91',label='Spherical chief rays')
    ax.plot(fields,12.24*np.tan(np.deg2rad(fields)),'--',color='#777',label='Reference: 12.24 tan(field)')
    ax.axhspan(-2.52,2.52,color='#e8edf4',alpha=.6)
    ax.axhline(0,color='#999',lw=.6)
    ax.set(xlabel='Object field angle (degrees)',ylabel='Sensor height from center (mm)',title='Chief-ray mapping · not an image-quality test')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('US20210096338A1 · Figure 3A spherical approximation',fontsize=15)
    fig.text(.5,.025,'Published layout and indices; P2 entrance shifted 1.5 mm. Ideal mirrors; filter omitted. No focus / MTF claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.96))
    fig.savefig(out/'layout.png',dpi=180);fig.savefig(out/'layout.svg');plt.close(fig)
    center=traces[2][0]
    rows='\n'.join(f"| {r['field_deg']:+g} | {r['sensor_relative_y_mm']:.6f} | {r['dominant_branch_power_fraction']:.4f} | {r['hits_sensor']} |" for r,_ in traces)
    route=' → '.join(f"{e['surface']} ({e['kind']})" for e in center['events'])
    text=f'''# Spherical chief-ray reconstruction

![Layout](layout.png)

All five chiefs follow the intended four-fold path and reach the sensor: **{valid}**.
The on-axis chief lands {center['sensor_relative_y_mm']:.6f} mm from sensor center.
This is a 2D spherical layout approximation, not a reproduction of patent imaging performance.

| Field (degrees) | Sensor height from center (mm) | Dominant branch power / input | Hits sensor |
|---|---|---|---|
{rows}

On-axis route: {route}.

The existing `nslens.freeform_body.trace.trace_bundle` performs nearest-hit Fresnel splitting and TIR; no surface order is imposed. Only the dominant exiting branch is plotted. Its power is chief-ray branch power, not full-aperture throughput.

## Numerical checks

Non-overlapping closed solids verified. Doubling intersection brackets (256 to 512) and reducing ray offset (1e-7 to 1e-8 mm) changes sensor intercept by at most {delta:.3g} mm. Convergence passed: {stable}. Maximum energy-ledger residual: {max(abs(r['energy_closure_residual']) for r,_ in traces):.3g}.

## Approximation choices

'''+ '\n'.join('- '+a for a in p['assumptions'])+'''

## Source audit

`nslens/patent_camera/prescription.json` retains all published coefficients and table text separately from this spherical experiment. Table 3I omits ZP1/ZP3/ZP4 (also in the granted version). A zero-filled freeform interpretation using CODE V standard Zernikes gives overlapping prism bodies; it is not a validated full prescription reconstruction. The spherical model does not depend on these missing terms.

Sources: [patent PDF](https://patentimages.storage.googleapis.com/5d/ed/01/4a05fdc9109c69/US20210096338A1.pdf), Tables 1 and 3A–3K, printed pages 11, 13–15; [patent HTML](https://patents.google.com/patent/US20210096338A1/en). CODE V convention reference: [Genberg et al., appendix](https://www.sigmadyne.com/sigweb/downloads/SPIE-4771-33.pdf).

Reproduce: `.venv/bin/python -m nslens.patent_camera.run`
'''
    (out/'report.md').write_text(text)
    print(json.dumps({k:v for k,v in report.items() if k not in ('fields','assumptions')},indent=2))
    print(out/'layout.png')
    if not (valid and stable):raise SystemExit('Chief-ray validation failed')

if __name__=='__main__':main()
