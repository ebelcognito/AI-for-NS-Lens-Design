"""Audit equal physical budgets and summarize paired TIR sampler comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from nslens.analysis.refinement_sampling_study import save

def main():
 p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--candidate',default='de_history');a=p.parse_args();root=Path(a.out)
 rows=[]
 for path in sorted(root.glob('*/summary.json')):
  r=json.loads(path.read_text());candidates=[json.loads(x) for x in (path.parent/'candidates.jsonl').read_text().splitlines()]
  assert len(candidates)==r['proposals']
  assert sum(x['source_rays']>0 for x in candidates)==r['search_calls']==r['config']['search_calls']
  assert sum(x['source_rays'] for x in candidates)==r['search_source_rays']
  audited=json.loads((path.parent/'archive.json').read_text())['audits']
  assert sum(x['result']['source_rays'] for x in audited)==r['audit_source_rays']
  rows.append(r)
 arms=sorted(set(r['arm'] for r in rows)); stats={}
 for k in [4,5]:
  stats[k]={}
  for arm in arms:
   rr=[r for r in rows if r['corners']==k and r['arm']==arm]
   stats[k][arm]=dict(n=len(rr),successes=sum(r['audited_success'] for r in rr),mean_verified_quality=float(np.mean([r['best_audited_quality'] for r in rr])),median_seconds=float(np.median([r['wall_seconds'] for r in rr])),physical_calls=sum(r['search_calls'] for r in rr),search_rays=sum(r['search_source_rays'] for r in rr),audit_rays=sum(r['audit_source_rays'] for r in rr))
 paired={}; rng=np.random.default_rng(73)
 for base in ['portfolio','random_restart']:
  if base not in arms:continue
  by={(r['seed'],r['corners'],r['arm']):r for r in rows}; seeds=sorted(set(r['seed'] for r in rows));ds=[];dq=[];wins=losses=0
  for seed in seeds:
   ss=[];qq=[]
   for k in [4,5]:
    if (seed,k,a.candidate) not in by or (seed,k,base) not in by:continue
    x=by[seed,k,a.candidate]; y=by[seed,k,base]; ss.append(int(x['audited_success'])-int(y['audited_success']));qq.append(x['best_audited_quality']-y['best_audited_quality'])
    wins+=int(x['audited_success'] and not y['audited_success']);losses+=int(y['audited_success'] and not x['audited_success'])
   if ss:ds.append(np.mean(ss));dq.append(np.mean(qq))
  ds=np.array(ds);dq=np.array(dq);boot=np.mean(rng.choice(ds,(20000,len(ds)),replace=True),axis=1); bq=np.mean(rng.choice(dq,(20000,len(dq)),replace=True),axis=1)
  sw=int(sum(ds>0));sl=int(sum(ds<0)); paired[base]=dict(seed_blocks=len(ds),success_rate_difference=float(np.mean(ds)),bootstrap95_by_seed=np.quantile(boot,[.025,.975]),discordant_unit_wins=wins,discordant_unit_losses=losses,seed_block_sign_p=float(binomtest(sw,sw+sl).pvalue) if sw+sl else 1.,mean_quality_difference=float(np.mean(dq)),quality_bootstrap95_by_seed=np.quantile(bq,[.025,.975]))
 save(root/'analysis.json',dict(candidate=a.candidate,stats=stats,paired=paired,verified_budget_runs=len(rows),limits='One base routing brief, two corner families. 125-ray audit same engine, not imaging or diffraction certification. Seed-block bootstrap preserves cross-family dependence. Quality=worst-field qualifying folded throughput, zero assigned to no verified design.'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for i,k in enumerate([4,5]):
  axes[i].bar(arms,[stats[k][a]['successes']/stats[k][a]['n'] for a in arms]);axes[i].set(title=f'{k}-corner TIR',ylabel='Fraction with a verified route',ylim=(0,1.05));axes[i].tick_params(axis='x',rotation=15)
 fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160)
 print(json.dumps(json.loads((root/'analysis.json').read_text()),indent=2))
if __name__=='__main__':main()
