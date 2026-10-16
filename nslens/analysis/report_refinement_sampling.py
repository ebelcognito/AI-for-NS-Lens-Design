"""Post-selection, same-engine physical audit and paired comparison report."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,spearmanr
from nslens.three_mirror import pilot
from nslens.three_mirror.audit import crossing
from nslens.analysis.refinement_sampling_study import save,serial

def main():
 p=argparse.ArgumentParser(); p.add_argument('out'); p.add_argument('--extra-search'); a=p.parse_args(); root=Path(a.out)
 rows=[]
 paths=list((root/'mirror_search').glob('*_[0-9]*.json'))
 if a.extra_search: paths+=list(Path(a.extra_search).glob('*_[0-9]*.json'))
 for path in sorted(paths):
  r=json.loads(path.read_text()); g=r['genes']; m=pilot.decode(g); shapes=[]
  for v,t,n,c,w in zip(m['vertices'],m['tangents'],m['normals'],m['curves'],m['widths']):
   u=np.linspace(-w,w,513); shapes.append(v+u[:,None]*t+(c*u*u)[:,None]*n)
  r['crossings']=[[i,j] for i in range(3) for j in range(i) if crossing(shapes[i],shapes[j])]
  fine=pilot.trace(g,fields=21,pupils=201); r['fine']=fine
  r['qualified']=not r['crossings'] and fine['worst_support']>=.99
  r['max_spot']=max((f['rms_spot_mm'] for f in fine['fields'] if f['rms_spot_mm'] is not None),default=None)
  rows.append(r)
 arms=sorted(set(r['arm'] for r in rows)); grouped={a:{r['seed']:r for r in rows if r['arm']==a} for a in arms}; summary={}
 for arm in arms:
  rr=list(grouped[arm].values()); summary[arm]=dict(n=len(rr),median_coarse_loss=float(np.median([r['best_loss'] for r in rr])),median_dense_loss=float(np.median([r['fine']['loss'] for r in rr])),qualified=sum(r['qualified'] for r in rr),crossed=sum(bool(r['crossings']) for r in rr),median_seconds=float(np.median([r['seconds'] for r in rr])),calls=sum(r['calls'] for r in rr),extra_dense_audit_calls=len(rr))
 comparisons={}; rng=np.random.default_rng(57)
 for arm in arms:
  if arm=='random': continue
  for base in ['random','retained','continuous']:
   if arm==base or base not in grouped:continue
   ss=sorted(set(grouped[arm])&set(grouped[base])); diff=np.array([grouped[arm][s]['fine']['loss']-grouped[base][s]['fine']['loss'] for s in ss]); wins=int(sum(diff<0)); ties=int(sum(diff==0)); n=len(diff)-ties
   boot=np.median(rng.choice(diff,(10000,len(diff)),replace=True),axis=1)
   qualified_wins=sum((grouped[arm][s]['qualified'],-grouped[arm][s]['fine']['loss'])>(grouped[base][s]['qualified'],-grouped[base][s]['fine']['loss']) for s in ss)
   comparisons[arm+'_vs_'+base]=dict(qualification_first_wins=qualified_wins,wins=wins,ties=ties,n=len(diff),median_paired_loss_difference=float(np.median(diff)),bootstrap_95_median_difference=np.quantile(boot,[.025,.975]),two_sided_sign_p=float(binomtest(wins,n).pvalue) if n else 1.)
 save(root/'mirror_audit.json',dict(summary=summary,comparisons=comparisons,rows=rows,limits='2D parabolic mirrors; no sagittal aberrations, diffraction, thickness, coatings, or tolerancing. Sampled proper crossings only. Development comparisons, unadjusted multiple tests.'))
 diag=[json.loads(p.read_text()) for p in sorted((root/'mirror_diagnostics').glob('diagnostic_[0-9]*.json'))]
 test=[r for r in diag if r['split']=='test']; target=[r['endpoints'][1]['loss'] for r in test]; scores={}
 for budget in ([0,24,60,120,240] if test else []):
  pred=[r['initial']['loss'] if not budget else min(r['initial']['loss'],r['endpoints'][0]['curve'][budget-1]) for r in test]
  stat=spearmanr(pred,target); scores[budget]=dict(spearman=float(stat.statistic),p=float(stat.pvalue))
 save(root/'early_refinement_diagnostic.json',dict(n=len(test),scores=scores,interpretation='Post-hoc development analysis: predict independent replicate 480-call refinement using first replicate partial trajectory; not a best-possible performance bound.'))
 import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(11,4))
 for arm in arms:
  rr=list(grouped[arm].values()); curves=np.array([r['curve'] for r in rr]); steps=np.arange(1,curves.shape[1]+1)
  ax[0].plot(steps,np.median(curves,axis=0),label=arm); ax[0].fill_between(steps,*np.quantile(curves,[.25,.75],axis=0),alpha=.12)
 ax[0].set(xlabel='Charged coarse evaluations',ylabel='Median best loss (IQR shaded)',yscale='log'); ax[0].legend()
 ax[1].boxplot([[r['fine']['loss'] for r in grouped[a].values()] for a in arms],tick_labels=arms); ax[1].set(yscale='log',ylabel='21-field × 201-ray audit loss'); ax[1].tick_params(axis='x',rotation=20)
 fig.tight_layout(); fig.savefig(root/'mirror_comparison.png',dpi=160); plt.close(fig)
 print(json.dumps(serial(dict(summary=summary,comparisons=comparisons)),indent=2))
if __name__=='__main__':main()
