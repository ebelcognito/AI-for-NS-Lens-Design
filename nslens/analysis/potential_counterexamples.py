"""Find prospective confirmation cases for fitness/potential rank reversals."""
import json,sys
from pathlib import Path
from nslens.analysis.mirror_residual_experiment import save

def main():
 out=Path(sys.argv[1]);rows=[json.loads(p.read_text()) for p in out.glob('*__*__*__*.json')];rows=[r for r in rows if not r['initial_audit']['imaging_pass']['1.0']];pairs=[]
 for a in rows:
  aa=a['initial_audit']
  if not aa['routing_pass']:continue
  pa=sum(t['checkpoints']['256']['success'] for t in a['trials'])/3
  for b in rows:
   bb=b['initial_audit']
   if a['job']['seed']!=b['job']['seed'] or a['job']['taskname']!=b['job']['taskname'] or not bb['routing_pass']:continue
   pb=sum(t['checkpoints']['256']['success'] for t in b['trials'])/3
   if pa-pb<2/3-1e-9:continue
   if a['parent']['raw']['loss']<=1.1*b['parent']['raw']['loss']:continue
   if aa['worst_mapping_rms_mm']<=1.1*bb['worst_mapping_rms_mm']:continue
   pairs.append(dict(worse_initial=a['job'],better_initial=b['job'],initial_losses=[a['parent']['raw']['loss'],b['parent']['raw']['loss']],initial_worst_rms=[aa['worst_mapping_rms_mm'],bb['worst_mapping_rms_mm']],observed_successes=[int(pa*3),int(pb*3)],score=(pa-pb)*(aa['worst_mapping_rms_mm']/bb['worst_mapping_rms_mm'])))
 pairs.sort(key=lambda p:p['score'],reverse=True);chosen=[];seen=set()
 for p in pairs:
  key=(p['worse_initial']['seed'],p['worse_initial']['taskname'])
  if key in seen:continue
  chosen.append(p);seen.add(key)
  if len(chosen)==3:break
 save(out/'rank_reversals.json',dict(criteria='Same donor/task, both initially feasible but>1mm. Worse candidate has>10% worse coarse merit AND>10% worse dense worst-field RMS; observed >=2/3 higher success after256 calls. Posthoc selection, requires fresh replicate confirmation.',all_qualifying_pairs=len(pairs),selected=chosen));print(json.dumps(chosen,indent=2))
if __name__=='__main__':main()
