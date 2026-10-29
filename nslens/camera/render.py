"""Render a saved candidate using the same camera physics (diagnostic, uncharged)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .problem import CameraCodec,CameraBrief,evaluate


def render(root):
    protocol=json.loads((root/'protocol.json').read_text())
    codec=CameraCodec(**protocol['task']['codec']); brief=CameraBrief(**protocol['task']['brief'])
    selected=json.loads((root/'selected.json').read_text())
    row,scene,trace=evaluate(codec,selected['genes'],brief,return_trace=True)
    if scene is None: return None
    fig,ax=plt.subplots(figsize=(12,5))
    for i,body in enumerate(scene.bodies):
        points=body.evaluate(np.linspace(0,1,401))
        ax.fill(points[:,0],points[:,1],alpha=.18,color='steelblue')
        ax.plot(points[:,0],points[:,1],color='steelblue')
        ax.text(*body.corners.mean(axis=0),str(i),ha='center')
        for face,mirror in enumerate(scene.mirrors[i]):
            if mirror:
                points=np.array([body._point_on(face,f) for f in np.linspace(0,1,80)])
                ax.plot(points[:,0],points[:,1],color='black',lw=2)
    inputs=brief.rays(); colors=['#d44','#299455','#426dd1']
    for source in inputs:
        ray=trace.dominant_exit(source.source_id)
        if ray is None: continue
        points=[source.origin]+[e.point for e in ray.events]
        origin=np.asarray(ray.origin);direction=np.asarray(ray.direction)
        t=(brief.max_length_mm-origin[0])/direction[0] if direction[0]>1e-10 else 5.
        points.append(origin+max(0,min(t,35))*direction)
        p=np.asarray(points); ax.plot(p[:,0],p[:,1],lw=.65,alpha=.65,
            color=colors[(source.source_id//brief.rays_per_field)%3])
    ax.plot([brief.max_length_mm]*2,[-brief.sensor_half_mm,brief.sensor_half_mm],color='black',lw=3)
    ax.axhline(brief.max_thickness_mm/2,color='gray',ls=':')
    ax.axhline(-brief.max_thickness_mm/2,color='gray',ls=':')
    ax.set(xlim=(-1,brief.max_length_mm+2),ylim=(-6,7),xlabel='x (mm)',ylabel='z (mm)',
        title=f"2D camera prototype — candidate {selected['candidate']} — coarse pass: {row['feasible']}\nDominant branch per pupil ray only; evaluation includes all branches")
    ax.set_aspect('equal');fig.tight_layout();fig.savefig(root/'geometry.png',dpi=160);plt.close(fig)
    return root/'geometry.png'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(render(p.parse_args().root))
