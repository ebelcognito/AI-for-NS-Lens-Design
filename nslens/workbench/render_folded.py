"""PNG evidence of actual closed freeform surfaces and TIR-delivered ray paths."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .folded_domain import evaluate,LAYOUT
from nslens.analysis.run_repair_curriculum import write_json,digest


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();source=json.loads(a.source.read_text())
 if 'task' in source:
  base=source['task']['base'];coeffs=source['outcome']['final_coefficients']
  label='Qwen folded freeform repair' if source['provenance']=='frozen_language_model' else 'Numerical folded freeform repair'
 else:
  x=source['cell']['genotype'];base=x[:9];coeffs=x[9:];label='Re-certified archive-derived folded freeform starting design'
 result,body,rays,trace,accepted=evaluate(base,coeffs,'audit',return_trace=True)
 if 'task' in source:assert result==source['outcome']['final_audit']
 count=len(rays)//3;_,boundary=body.sample(2048);boundary=np.vstack([boundary,boundary[0]])
 fig,axes=plt.subplots(1,3,figsize=(17,7),layout='constrained')
 colors=['#2285b8','#159876','#d16b32']
 for field,ax in enumerate(axes):
  ax.fill(boundary[:,0],boundary[:,1],color='#dce8ef');ax.plot(boundary[:,0],boundary[:,1],color='#233e50',lw=1.5)
  chosen=[r for r in accepted if r['source_id']//count==field and r['energy']>.01]
  # Sample the full fan across its extent for clarity, keeping every third source ray.
  for row in chosen:
   if row['source_id']%count%3:continue
   ray=rays[row['source_id']]
   pts=np.array([ray.origin]+[e.point for e in row['events']]+[(row['x'],row['y'])])
   ax.plot(pts[:,0],pts[:,1],color=colors[field],lw=.9,alpha=.7)
  if chosen:
   row=max(chosen,key=lambda r:r['energy']);ray=rays[row['source_id']]
   pts=np.array([ray.origin]+[e.point for e in row['events']]+[(row['x'],row['y'])])
   ax.plot(pts[:,0],pts[:,1],color=colors[field],lw=2)
   for event in row['events']:
    if event.kind=='tir':
     ax.scatter(*event.point,marker='D',s=55,color='#b53229',zorder=6)
     ax.annotate('TIR fold',event.point,xytext=(5,-15),textcoords='offset points',fontsize=9,color='#93241d')
  points=np.array(LAYOUT['source_points'])
  ax.scatter(points[:,0],points[:,1],s=30,marker='s',color='#9aabb6')
  ax.scatter(*points[field],s=60,marker='s',color=colors[field],zorder=6)
  ax.text(55,54,'Display sources',ha='center',fontsize=9)
  ax.plot([120,120],[-9,9],color='#333',lw=3)
  ax.text(118,12,'Eyebox',ha='right',fontsize=9)
  ax.set_aspect('equal');ax.set(xlim=(min(10,boundary[:,0].min()-5),127),ylim=(min(-40,boundary[:,1].min()-5),60),xlabel='x (mm)',ylabel='y (mm)')
  f=result['fields'][field]
  ax.set_title(f"Source {field+1}: ({points[field,0]:g}, 50) mm\nFolded throughput {100*f['folded_throughput']:.1f}% · angular σ {f['sigma_deg']:.3f}°",fontsize=11)
  ax.grid(alpha=.15)
 fig.suptitle(label+'\nThree display fields · closed C2 spline · useful rays require TIR',fontsize=16,fontweight='bold')
 submission=str(source['outcome']['submitted']) if 'outcome' in source else 'not a model episode'
 fig.supxlabel(f"Submission: {submission} · dense independent trace: {len(rays)} rays · FOV {result['fov_deg']:.2f}° · optical audit {'PASS' if result['success'] else 'FAIL'}\n"
  'Red diamonds mark actual TIR events. Primary pupil-reaching rays shown; Fresnel branches remain included in reported metrics.\n'
  '2-D prototype, not a production-qualified or 3-D design. Starting design is numerical/archive-derived, not discovered by Qwen.',fontsize=10)
 a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=180,facecolor='white');plt.close(fig)
 write_json(a.out.with_suffix('.json'),dict(source=str(a.source),source_sha256=digest(a.source),audit=result,renderer_sha256=digest(__file__)))
 print(a.out,flush=True)


if __name__=='__main__':main()
