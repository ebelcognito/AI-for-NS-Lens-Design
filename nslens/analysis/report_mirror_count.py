"""Audit/report the variable-count pilot without selecting on dense results."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nslens.analysis.variable_mirror import trace

def main():
 out=Path(sys.argv[1]);summary=json.loads((out/'summary.json').read_text());protocol=json.loads((out/'protocol.json').read_text());rows=[]
 for p in out.glob('*.json'):
  r=json.loads(p.read_text())
  if isinstance(r,dict) and 'arm' in r:rows.append(r)
 expected=protocol['rounds']*(1+12*protocol['generations'])
 assert all(r['coarse_calls']==expected and r['dense_calls']==4 for r in rows)
 comparisons={}
 for arm in ['same_count','cross_count']:
  comparisons[arm]={}
  for n in range(3,7):
   deltas=[]
   for seed in protocol['seeds']:
    rr={r['arm']:r for r in rows if r['seed']==seed}
    deltas.append(rr[arm]['audits'][str(n)]['loss']-rr['restart']['audits'][str(n)]['loss'])
   d=np.array(deltas);rng=np.random.default_rng(420+n);boot=np.mean(rng.choice(d,(10000,len(d)),replace=True),axis=1)
   comparisons[arm][str(n)]=dict(mean_delta=float(d.mean()),wins=int(sum(d<0)),ties=int(sum(d==0)),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
 (out/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
 fig,axes=plt.subplots(2,2,figsize=(12,10));geometry={}
 for n,ax in zip(range(3,7),axes.flat):
  # Pick display by COARSE loss only, then display the associated dense trace.
  r=min(rows,key=lambda r:r['coarse'][str(n)]['loss']);g=r['coarse'][str(n)]['genes']
  audit,m,lines,paths,good=trace(g,n,fields=5,pupils=9,paths=True)
  for i,line in enumerate(lines):ax.plot(*line.T,color='black',lw=2);ax.text(*m['vertices'][i],str(i+1))
  for path,okay in zip(paths,good):
   p=np.array(path);ax.plot(*p.T,color='tab:blue' if okay else 'tab:red',alpha=.2,lw=.6)
  det=m['image']+np.array([-15,15])[:,None]*m['image_tangent'];ax.plot(*det.T,color='green',lw=2)
  ax.set_title(f'{n} mirrors: {r["arm"]}, seed {r["seed"]}\ndense support {r["audits"][str(n)]["worst_support"]:.2f}');ax.set_aspect('equal');ax.set_xlabel('mm');ax.set_ylabel('mm')
  geometry[str(n)]=dict(seed=r['seed'],arm=r['arm'],dense_loss=r['audits'][str(n)]['loss'])
 fig.tight_layout();fig.savefig(out/'layouts.png',dpi=140);plt.close(fig)
 report=['VARIABLE MIRROR COUNT PILOT','',f'{len(rows)} runs; {sum(r["coarse_calls"] for r in rows)} coarse calls; {sum(r["dense_calls"] for r in rows)} dense audits.','Same field +/-4deg, entrance aperture27.78mm and target image mapping50mm across3--6 mirrors.','Nine to eighteen variables: turns, intermirror/detector distances, curvatures. Fixed30mm halfwidths.','Ideal2D parabolic cross-sections. Detector position and normal follow final scaffold leg.','No 3D, diffraction, coating loss, manufacturing, intersection clearance or common packaging-envelope validation.','No extra-mirror cost imposed: comparisons are within each count, not claims about optimal mirror count.','','STRATEGIES','restart: new scaffold each round; same_count: perturb incumbent of that count; cross_count: insert/remove one mirror from adjacent-count incumbent.','Each uses identical counts schedule and call budget; no archive cells or learned model.','Insertion uses small-angle flat mirror and splits a leg; removal merges legs and turns. These approximate geometric proposals do not preserve ray paths or power.','Cross-count proposals select lower coarse loss among available adjacent counts. Earlier rounds build their own parents; no pre-solved designs.','','DENSE AUDIT RESULTS (lower loss better; dominated by lost rays for unsuccessful designs)','count | strategy | median loss | mean worst-field support | runs with >=99% support']
 for n in range(3,7):
  for arm,s in summary.items():
   v=s['by_count'][str(n)];report.append(f'{n} | {arm} | {v["median_dense_loss"]:.3f} | {v["mean_worst_support"]:.3f} | {v["full_support_runs"]}/{s["runs"]}')
 report+=['','PAIRED COMPARISONS AGAINST RESTART',json.dumps(comparisons,indent=2),'','INTERPRETATION','Support means fraction reaching detector after exactly n distinct mirrors, not an imaging specification pass.','Dense audit uses11fields x81rays; search uses3x9. Dense outcomes do not drive search or candidate selection.','Route reward is normalized by mirror count, unlike the original three-mirror pilot. Turns allow +/-180deg.','Count order3,4,5,6 favors upward transfer availability early; deletion becomes available in later rounds.','Only a small search budget per architecture; failure to find a design does not establish that architecture is inferior.','Broad angle coordinates have a wrap boundary; transfer uses clipping; these implementation choices may hinder local search.','Verification: four original mirror tests;30 geometry/support/image-error parity cases;400 adjacent proposal smoke cases; exact evaluation ledgers.','','FILES','protocol.json: pre-run conditions. Per-run JSON: all incumbents/history/audits. paired_comparisons.json: seed-paired exploratory statistics. layouts.png: coarse-selected examples (red failed rays, blue successful rays).','REPRODUCE',f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.mirror_count_experiment --out NEW_DIRECTORY --seeds {len(protocol["seeds"])} --rounds {protocol["rounds"]} --gens {protocol["generations"]}']
 report.insert(2, 'Search crossing penalty: '+str(protocol.get('geometry_penalty',False))+'; '+protocol.get('geometry_rule','none'))
 report[-1]+=f" --seed-start {protocol['seeds'][0]}"+(' --geometry-penalty' if protocol.get('geometry_penalty') else '')
 (out/'RESULTS.txt').write_text('\n'.join(report)+'\n');(out/'verification.json').write_text(json.dumps(dict(runs=len(rows),calls=sum(r['coarse_calls'] for r in rows),dense_calls=sum(r['dense_calls'] for r in rows),ledgers_passed=True,display_selection=geometry,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'variable_mirror.py',out/'mirror_count_experiment.py']}),indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
