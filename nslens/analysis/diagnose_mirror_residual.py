"""Posthoc ranking diagnostics using saved counterfactual outcomes only."""
import json,sys
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import spearmanr

def main():
 out=Path(sys.argv[1]);rows=[json.loads(p.read_text()) for p in out.glob('*__*__*.json')];model=joblib.load(out/'learned_physical.joblib')['model'];result={}
 for split in ['seed_heldout','task_heldout','both_heldout']:
  subset=[r for r in rows if r['job']['split']==split];guided=[];random=[];corr=[];oracle=[]
  for r in subset:
   parent=r['parent']['raw']['loss'];true=np.array([np.log1p(parent)-np.log1p(a['guided']['coarse']['loss']) for a in r['actions']]);pred=model.predict([a['features'] for a in r['actions']])
   if np.std(true)>1e-12 and np.std(pred)>1e-12:corr.append(float(spearmanr(true,pred).statistic))
   guided.extend(a['guided_initial_loss']<parent for a in r['actions']);random.extend(a['random_initial_loss']<parent for a in r['actions'])
   if r['job']['n']==r['selected_parent_count']:oracle.append(min(r['actions'],key=lambda a:a['guided']['coarse']['loss'])['guided']['audit']['imaging_pass']['0.5'])
  result[split]=dict(guided_immediate_improvement_fraction=float(np.mean(guided)),random_immediate_improvement_fraction=float(np.mean(random)),mean_within_parent_rank_correlation=float(np.mean(corr)) if corr else None,nonconstant_ranking_parents=len(corr),nondeployable_coarse_refinement_oracle_05_passes=sum(oracle),one_design_parents=len(oracle))
 (out/'selection_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
