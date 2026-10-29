"""Render a measured narrow-field folded repair and its target-angle errors."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .folded_domain import evaluate
from .pencil_loop import pencil_evaluate,write


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',type=Path);p.add_argument('decision',type=Path)
 p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 decisions=json.loads(a.decision.read_text());decision=next((d for d in decisions if d['success']),decisions[0])
 task=next(t for t in json.loads((a.corpus/'tasks.json').read_text()) if t['task_id']==decision['task_id'])
 if decision['outcome'] is None:raise ValueError('no submitted prescription to render')
 x=decision['outcome']['final_coefficients'];audit=pencil_evaluate(task,x,'audit')
 assert audit==decision['outcome']['final_audit']
 result,body,rays,trace,accepted=evaluate(task['base'],x,'audit',return_trace=True,layout=task['layout'],requirements=task['requirements'])
 initial=pencil_evaluate(task,task['start'],'audit')
 fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained');ax=axes[0]
 _,boundary=body.sample(2048);boundary=np.vstack([boundary,boundary[0]])
 ax.fill(boundary[:,0],boundary[:,1],color='#dce8ef');ax.plot(boundary[:,0],boundary[:,1],color='#233e50')
 colors=['#2285b8','#159876','#d16b32']
 for field in range(3):
  chosen=[r for r in accepted if r['source_id']//61==field and r['energy']>.05 and r['source_id']%61 in [0,15,30,45,60]]
  for row in chosen:
   pts=np.array([rays[row['source_id']].origin]+[e.point for e in row['events']]+[(row['x'],row['y'])])
   ax.plot(pts[:,0],pts[:,1],color=colors[field],alpha=.7,lw=.9)
  if chosen:
   for event in chosen[len(chosen)//2]['events']:
    if event.kind=='tir':ax.scatter(*event.point,marker='D',color='#b53229',s=35)
 ax.plot([120,120],[-9,9],color='black',lw=3);ax.set_aspect('equal');ax.set(xlabel='x (mm)',ylabel='y (mm)',title='Actual TIR paths • 1 mm source field • ±0.25° fan');ax.grid(alpha=.2)
 xaxis=np.arange(3)
 axes[1].axhspan(-.15,.15,color='#159876',alpha=.15,label='Allowed target error')
 axes[1].plot(xaxis,initial['centroid_errors_deg'],'o--',label='Before repair')
 axes[1].plot(xaxis,audit['centroid_errors_deg'],'o-',label='After selected plan')
 axes[1].set(xticks=xaxis,xticklabels=['Left','Center','Right'],xlabel='Source position within 1 mm field',ylabel='Output centroid error (degrees)',title='Shifted 183-ray audit grid');axes[1].legend();axes[1].grid(alpha=.2)
 fig.suptitle(f"Qwen pencil-beam repair-plan selection • {task['task_id']} • plan {decision['plan_id']} • {'PASS' if audit['success'] else 'FAIL'}")
 fig.supxlabel('2-D archive-derived freeform prism; Qwen selected a numerical repair plan. No topology discovery claim. Red diamonds: TIR.',fontsize=9)
 fig.savefig(a.out,dpi=180);plt.close(fig);write(a.out.with_suffix('.json'),dict(task_id=task['task_id'],plan_id=decision['plan_id'],audit=audit,decision_source=str(a.decision)))

if __name__=='__main__':main()
