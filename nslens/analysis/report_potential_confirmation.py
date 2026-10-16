"""Display fresh-repeat validation separately from exploratory selector results."""
import json,sys,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 out=Path(sys.argv[1]);s=json.loads((out/'summary.json').read_text());pairs=s['pairs'];fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(len(pairs));width=.34
 for sign,side,color,label in [(-1,'worse_initial','tab:orange','Worse initial quality'),(1,'better_initial','tab:blue','Better initial quality')]:
  idx=0 if side=='worse_initial' else 1;axes[0].bar(x+sign*width/2,[p['initial_worst_rms_mm'][idx] for p in pairs],width,color=color,label=label)
  vals=np.array([p['success_rate'][side] for p in pairs]);ci=np.array([p['wilson_95'][side] for p in pairs]);err=np.maximum(0,np.array([vals-ci[:,0],ci[:,1]-vals]));axes[1].bar(x+sign*width/2,vals,width,color=color,label=label,yerr=err,capsize=3)
 labels=['50mm task\n4 vs5mirrors','35mm task\n4 vs5mirrors','Wider field\n3 vs3mirrors']
 for ax in axes:ax.set_xticks(x,labels)
 axes[0].set_ylabel('Initial worst-field RMS mapping error (mm)');axes[0].axhline(1,color='black',ls='--',lw=1,label='1mm target');axes[0].legend(fontsize=8)
 axes[1].set_ylabel('Success fraction after256 refinement evaluations');axes[1].set_ylim(0,1.08);axes[1].set_title('24 new repeats per configuration;95% Wilson intervals')
 fig.tight_layout();fig.savefig(out/'confirmed_reversals.png',dpi=170);plt.close(fig)
 lines=['CONFIRMED QUALITY–POTENTIAL REVERSALS','','Same initial configuration + same task + fixed256-evaluation CMA refinement;24 NEW paired optimizer seeds per configuration. Pass means <=1mm worst-field RMS mapping error,>=99% worst-field support and no sampled mirror crossings.','Cases were selected posthoc from an initial240-config benchmark with3repeats/config, then frozen for the independent optimizer-repeat confirmation. These are conditional results for selected configurations, not prevalence/generalization estimates.','','RESULTS','task | mirror counts(worse/better initial) | initial RMS(worse/better) | new successes(worse/better)']
 for p in pairs:
  c=p['successes_out_of_24'];n=p['n_mirrors'];e=p['initial_worst_rms_mm'];lines.append(f'{p["taskname"]} | {n["worse_initial"]}/{n["better_initial"]} | {e[0]:.3f}/{e[1]:.3f}mm | {c["worse_initial"]}/24 vs{c["better_initial"]}/24')
 lines+=['','The same-count3-mirror pair rules out mirror count alone as the explanation for every reversal. It does not isolate the cause: geometry, local optimization landscape, residual structure and parameter bounds differ.','The first two pairs share a donor lineage across tasks and are not independent architecture populations. Initial audits and source genes were inherited; source snapshots and hashes preserved.','256 calls, sigma.03,pop8, legacy mean-error merit and bounded CMA define the operational potential. This does not establish an architecture-level optimum or optimizer-independent potential.','','RELATION TO THE PREDICTOR BENCHMARK','In donor-held-out selection, refining one candidate for256 calls succeeded57.8% using current scalar fitness,58.9% using current worst-field quality,55.6% using learned cheap optical features, and56.7% using learned Jacobian features. The latter additionally charges about208 screen calls per pool. Random selection18.3%.','Thus the confirmed reversals validate the distinction between current quality and refinement potential, but the fitted predictors have not improved candidate selection. A local first-order correctability proxy is not sufficient evidence of attainable final quality.','In the initial benchmark228/240configurations failed the1mm target initially.27/228passed all3initial256-call trials;172/228passed none. Six donor blocks; repeated task/count/strategy variants correlated.','','COST','Initial benchmark191040 coarse traces and1440dense audits. Confirmation37008coarse and144dense. Total228048new coarse traces and1584dense audits. Candidate/source-generation and initial audit costs inherited; no total-compute superiority claim.','','VERIFICATION','Confirmation sources match hashes, new seeds disjoint from initial rollout, paired sides share a seed, eachtrial257coarse calls (parent+256) and1dense audit.144trials reconciled. Each endpoint chosen only by coarse merit before dense audit.','Source code, protocol, source snapshots and individual endpoint prescriptions included.','','NEXT RESEARCH QUESTION','What observable property distinguishes a repairable large error from a small but stubborn error? Use these confirmed cases to test residual changes over short refinement trajectories, rather than assuming current error magnitude or one Jacobian reveals potential.']
 (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n');shutil.copy(__file__,out/Path(__file__).name)
 for f in ['confirm_potential_pairs.py','potential_counterexamples.py']:shutil.copy(Path('nslens/analysis')/f,out/f)
if __name__=='__main__':main()
