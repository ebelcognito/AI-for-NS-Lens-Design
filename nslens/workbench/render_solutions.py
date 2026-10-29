"""Render actual workbench prescriptions and re-traced optical errors as PNGs."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from nslens.analysis.run_repair_curriculum import digest, write_json
from nslens.learning.freeform_repair import ImagingSpec
from nslens.freeform_body.trace import InputRay, trace_bundle, project_exits_to_x
from .domain import surface_body, measure

COLORS=('#2171b5','#239768','#db6837')


def layout(ax,spec,start,final):
    body=surface_body(spec,final)
    _,boundary=body.sample(1024)
    _,initial=surface_body(spec,start).sample(1024)
    boundary=np.vstack([boundary,boundary[0]])
    initial=np.vstack([initial,initial[0]])
    ax.fill(boundary[:,0],boundary[:,1],color='#dceaf1',alpha=.9)
    ax.plot(initial[:,0],initial[:,1],color='#99a2ad',ls='--',lw=1,label='Initial boundary')
    ax.plot(boundary[:,0],boundary[:,1],color='#20364b',lw=1.4,label='Final boundary')
    angles=spec.chief_deg+np.linspace(-spec.half_field_deg,spec.half_field_deg,3)
    rays=[]
    for angle in angles:
        theta=np.deg2rad(angle)
        for y in np.linspace(-spec.pupil_half,spec.pupil_half,9):
            rays.append(InputRay((0.,float(y-60*np.tan(theta))),
                (float(np.cos(theta)),float(np.sin(theta))),source_id=len(rays)))
    trace=trace_bundle(body,rays,energy_floor=.00002,max_interactions=12,intersection_scan=1024)
    records=project_exits_to_x(trace,60+spec.focal_distance)
    for i,ray in enumerate(rays):
        group=[r for r in records if r['source_id']==i]
        if not group:
            continue
        primary=max(group,key=lambda r:r['energy'])
        points=np.array([ray.origin]+[e.point for e in primary['events']]+[(primary['x'],primary['y'])])
        ax.plot(points[:,0],points[:,1],color=COLORS[i//9],lw=.75,alpha=.65)
    sensor=60+spec.focal_distance
    ax.axvline(sensor,color='#555',lw=1,ls=':')
    for angle,color in zip(angles,COLORS):
        ax.scatter([sensor],[spec.focal_distance*np.tan(np.deg2rad(angle))],color=color,s=22,marker='x',zorder=5)
    ax.text(sensor+1, .9*spec.radius_y,'Image\nplane',fontsize=9,ha='left',va='top')
    ax.set(xlim=(-1,sensor+9),ylim=(-1.12*spec.radius_y,1.12*spec.radius_y),xlabel='Axial position x (mm)',ylabel='Height y (mm)')
    ax.set_aspect('equal',adjustable='box')
    ax.legend(fontsize=8,loc='upper left',frameon=False)
    ax.grid(alpha=.15)


def fans(ax,verdict,spec,title):
    residual=np.asarray(verdict.residual).reshape(9,31)*1000
    pupil=np.linspace(-1,1,31)
    pupil[1:-1]-=.31*2/30
    for row in residual:
        ax.plot(pupil,row,color='#aab3bd',lw=.65,alpha=.4)
    for i,color in zip((0,4,8),COLORS):
        angle=spec.chief_deg+np.linspace(-spec.half_field_deg,spec.half_field_deg,9)[i]
        ax.plot(pupil,residual[i],color=color,lw=1.7,label=f'{angle:+.2f}° field')
    ax.axhline(0,color='#555',lw=.7)
    ax.set(title=f'{title} — audit RMS {verdict.rms*1000:.2f} µm',
           xlabel='Normalized pupil coordinate',ylabel='Image position error (µm)')
    ax.legend(fontsize=8,frameon=False)
    ax.grid(alpha=.18)


def render(path,out,label):
    episode=json.loads(path.read_text())
    task=episode['task']; spec=ImagingSpec(**task['spec'])
    start=task['start']; final=episode['outcome']['final_coefficients']
    before=measure(spec,start,'audit'); after=measure(spec,final,'audit')
    recorded=episode['outcome']['final_audit']
    assert asdict(after)==recorded,'render re-trace differs from saved audit'
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    policy='Qwen3-8B + LoRA' if episode.get('adapter') else 'Frozen Qwen3-8B'
    fig.suptitle(f"{label} | {task['task_id']}\n{policy} + numerical optical tools · 2-D transmitting freeform element",fontsize=15,fontweight='bold')
    layout(axes[0,0],spec,start,final)
    axes[0,0].set_title('Final ray paths and actual lens boundaries')
    body0=surface_body(spec,start); body1=surface_body(spec,final)
    for side,interval,color in [('Entrance',(.25,.75),'#7656a5'),('Exit',(-.25,.25),'#a45d29')]:
        u=np.linspace(*interval,1001)
        a=body0.evaluate(u); b=body1.evaluate(u)
        keep=np.abs(a[:,1])<=1.4*spec.pupil_half
        axes[0,1].plot((b[keep,0]-a[keep,0])*1000,a[keep,1],label=side,color=color,lw=2)
    axes[0,1].axvline(0,color='#777',lw=.7)
    for y in (-spec.pupil_half,spec.pupil_half):axes[0,1].axhline(y,color='#aaa',lw=.7,ls=':')
    axes[0,1].set(title='Surface change through the pupil region',xlabel='Final − initial axial surface position (µm)',ylabel='Height y (mm)')
    axes[0,1].legend(frameon=False);axes[0,1].grid(alpha=.18)
    fans(axes[1,0],before,spec,'Before repair')
    fans(axes[1,1],after,spec,'After repair')
    o=episode['outcome']
    fig.supxlabel(f"Submitted: {o['submitted']} · independent audit: {'PASS' if o['success'] else 'FAIL'} · {o['candidate_calls']}/24 candidate traces · "
                  f"minimum primary energy {after.min_ray_energy:.3f}\n"
                  'Error panels use different vertical scales. Both use the same 279-ray audit grid. '
                  'Layout shows 27 primary rays; faint reflected branches are omitted.',fontsize=9)
    name=out/(label.lower().replace(' ','_')+'.png')
    fig.savefig(name,dpi=180,facecolor='white');plt.close(fig)
    return dict(image=str(name),source_episode=str(path),source_sha256=digest(path),
        task_id=task['task_id'],family_id=task['family_id'],selection_label=label,
        before_audit=asdict(before),after_audit=asdict(after),retrace_matches=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=3)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[(path,json.loads(path.read_text())) for path in sorted(a.run.glob('episode*.json'))]
    rows=[(path,e) for path,e in rows if e['outcome']['success']]
    selected=[];families=set()
    # Select explicitly for visual interest, not as an unbiased performance sample.
    criteria=[('Largest repair',lambda e:e['evaluations'][0]['verdict']['rms']/max(e['outcome']['final_audit']['rms'],1e-12)),
              ('Off axis solution',lambda e:abs(e['task']['spec']['chief_deg'])),
              ('Strong surface change',lambda e:np.linalg.norm(np.asarray(e['outcome']['final_coefficients'])-e['task']['start']))]
    for label,key in criteria[:a.count]:
        candidates=[r for r in rows if r[1]['task']['family_id'] not in families]
        if not candidates:break
        path,e=max(candidates,key=lambda row:key(row[1]))
        selected.append((label,path,e));families.add(e['task']['family_id'])
    if not selected:raise ValueError('no verified solutions to render')
    records=[render(path,a.out,label) for label,path,e in selected]
    fig,axes=plt.subplots(len(selected),1,figsize=(14,4*len(selected)),layout='constrained',squeeze=False)
    for ax,(label,path,e) in zip(axes[:,0],selected):
        spec=ImagingSpec(**e['task']['spec']);o=e['outcome']
        layout(ax,spec,e['task']['start'],o['final_coefficients'])
        ax.set_title(f"{label} · {e['task']['task_id']} · audit RMS {1000*o['final_audit']['rms']:.2f} µm",fontsize=12)
    policy='Qwen3-8B + LoRA' if selected[0][2].get('adapter') else 'Frozen Qwen3-8B'
    fig.suptitle(f'{policy}: verified optical solutions',fontsize=17,fontweight='bold')
    fig.supxlabel('2-D transmission pilot · primary rays only · dashed boundary = initial lens · final acceptance uses 279 rays',fontsize=10)
    fig.savefig(a.out/'solutions_gallery.png',dpi=180,facecolor='white');plt.close(fig)
    write_json(a.out/'manifest.json',dict(source_run=str(a.run),renderer_sha256=digest(__file__),
        selection='Three distinct families, chosen for repair magnitude, off-axis field, and surface change; not an unbiased performance sample.',solutions=records))
    print(a.out/'solutions_gallery.png',flush=True)


if __name__=='__main__':main()
