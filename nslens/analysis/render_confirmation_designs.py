"""Export saved optical designs as inspectable PNG ray diagrams."""
import argparse,json,csv,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nslens.analysis.variable_mirror import trace
from nslens.analysis.final_freeform import trace as freeform_trace

LABELS={'reuse_only':'CMA refinement','optical_residual':'Optical mutation + CMA','direction_random':'Random mutation + CMA','restart':'Random start + CMA (earlier sweep)'}
COLORS=['#2684cf','#22a06b','#d95f02']

def draw(ax,r,compact=False):
 g=r['selected_coarse']['genes'];n=r['selected_n'];task=r['task'];a=r['audit'];j=r.get('source_job',r['job'])
 coeff=r['selected_coarse'].get('freeform_coefficients_mm')
 if coeff is None:_,m,shapes,paths,good=trace(g,n,fields=3,pupils=7,paths=True,**task)
 else:_,m,shapes,paths,good=freeform_trace(g,n,coeff,fields=3,pupils=7,paths=True,**task)
 for i,path in enumerate(paths):
  p=np.asarray(path)
  ax.plot(p[:,0],p[:,1],color=COLORS[i//7] if good[i] else '#999999',alpha=.5 if good[i] else .3,lw=.7,ls='-' if good[i] else '--',zorder=1)
 for i,p in enumerate(shapes):
  ax.plot(p[:,0],p[:,1],color='#172b4d',lw=2.4,zorder=3)
  ax.annotate(f'M{i+1}',m['vertices'][i],xytext=(5,5),textcoords='offset points',fontsize=8,color='#172b4d',bbox=dict(fc='white',ec='none',alpha=.7,pad=.5),zorder=4)
 h=task['focal']*np.tan(np.deg2rad(task['half_field_deg']))+1
 sensor=np.array([m['image']-h*m['image_tangent'],m['image']+h*m['image_tangent']]);ax.plot(sensor[:,0],sensor[:,1],color='#9b36b7',lw=3,zorder=4)
 ax.annotate('Detector',m['image'],xytext=(5,-12),textcoords='offset points',fontsize=8,color='#852a9e',bbox=dict(fc='white',ec='none',alpha=.8,pad=.5))
 error=a['worst_mapping_rms_mm'];err='undefined' if error is None else f'{error:.3f} mm';status='PASS' if a['success'] else 'FAIL'
 method_label=LABELS[j['arm']]+(' + final freeform refinement' if r['job'].get('refiner')=='last_freeform' else '')
 title=f"{method_label}\n{n} mirrors | {status} | worst-field RMS {err}"
 if not compact:title+=f"\n{j['task']} | donor {j['seed']} | repeat {j.get('repeat','original')} | support {a['worst_support']:.1%}"
 ax.set_title(title,fontsize=10 if compact else 12,pad=12)
 ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)');ax.grid(alpha=.15);ax.margins(.12)
 if not compact:
  fields=[-task['half_field_deg'],0,task['half_field_deg']]
  ax.legend([Line2D([0],[0],color=c) for c in COLORS]+[Line2D([0],[0],color='#9b36b7',lw=3)], [f'{f:+g}° field' for f in fields]+['Target detector span'],loc='best',fontsize=8)
 return n

def render(job):
 source,target=job;r=json.loads(Path(source).read_text());fig,ax=plt.subplots(figsize=(9,7));draw(ax,r)
 task=r['task'];fig.text(.5,.015,f"2D ideal parabolic mirrors | aperture {task['aperture']:g} mm | target focal mapping {task['focal']:g} mm\nShown: 3 fields × 7 rays; reported quality: saved 15 fields × 101 rays. Detector is an infinite plane in the tracer.",ha='center',fontsize=8,color='#555555');fig.tight_layout(rect=[0,.065,1,1]);fig.savefig(target,dpi=150);plt.close(fig)
 return target

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 sources=sorted(Path('experiments/hard_confirmation_20261006').glob('*__*__*__r*.json'))
 restarts=[p for task in ['field8','aperture42'] for p in sorted(Path('experiments/hard_optical_20261006').glob(f'{task}__restart__*.json'))]
 jobs=[];records=[]
 for source in sources+restarts:
  r=json.loads(source.read_text());j=r['job'];section='earlier_random_restarts' if j['arm']=='restart' else 'latest_confirmation';folder=out/section/j['task']/j['arm'];folder.mkdir(parents=True,exist_ok=True);target=folder/(source.stem+'.png');jobs.append((str(source),str(target)))
  records.append(dict(source_json=str(source.resolve()),png=str(target.relative_to(out)),task=j['task'],method=j['arm'],donor=j['seed'],repeat=j.get('repeat','original'),mirrors=r['selected_n'],success=r['audit']['success'],worst_mapping_rms_mm=r['audit']['worst_mapping_rms_mm'],worst_support=r['audit']['worst_support']))
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,_ in enumerate(pool.map(render,jobs),1):
   if i%24==0:print(f'{i}/{len(jobs)} PNGs',flush=True)
 fig,axs=plt.subplots(2,4,figsize=(22,14));selected=[]
 for row,task in enumerate(['field8','aperture42']):
  for col,arm in enumerate(['restart','reuse_only','direction_random','optical_residual']):
   eligible=[r for r in records if r['task']==task and r['method']==arm]
   # Illustrative only: pick passes first, then dense RMS. Never feeds search or reported rates.
   best=min(eligible,key=lambda r:(not r['success'],r['worst_mapping_rms_mm'] if r['worst_mapping_rms_mm'] is not None else 1e9))
   draw(axs[row,col],json.loads(Path(best['source_json']).read_text()),compact=True);selected.append(best)
 fig.suptitle('Illustrative best saved designs per method — selected AFTER evaluation\nTop: ±8° field, 35 mm aperture; bottom: ±6° field, 42 mm aperture. Target focal mapping: 75 mm.',fontsize=15)
 fig.text(.5,.012,'Blue / green / orange: negative / zero / positive field rays. Purple: detector span. Each panel uses its own scale. Earlier restart examples have a different trial cohort.',ha='center',fontsize=10)
 fig.subplots_adjust(left=.04,right=.98,bottom=.08,top=.87,wspace=.3,hspace=.55);fig.savefig(out/'overview.png',dpi=140);plt.close(fig)
 with (out/'index.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 (out/'overview_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
 (out/'README.txt').write_text('Optical design PNG export\n\noverview.png: illustrative best saved designs, one per method/task; selection by passing audit then lowest dense RMS AFTER the experiment. This is not a performance comparison. Each panel has its own scale.\nlatest_confirmation/: all144 selected final designs from the latest confirmation, including failures.\nearlier_random_restarts/:12 selected final designs from the earlier fresh-start CMA sweep for these two specifications. These did not share the latest confirmation cohort.\nindex.csv: source JSON, PNG path and saved quality metrics for every design.\n\nDark curves are finite mirrors. Purple line is the target detector span plus1mm margin, not a modeled finite sensor aperture. Colored lines are traced rays from3fields x7pupils; failed plotted rays are dashed gray. Titles quote the original15fields x101rays dense audit, not the sparse display rays. PASS requires <=1mm mapping RMS,>=99% worst-field support and no sampled crossing. These are2D ideal parabolic mirror models, without mounts, thickness, coating, diffraction or tolerance modeling.\n')
 shutil.copy(__file__,out/'render_confirmation_designs.py');print(f'Wrote {len(jobs)} design PNGs and overview to {out}',flush=True)
if __name__=='__main__':main()
