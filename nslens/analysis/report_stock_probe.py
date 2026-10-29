"""Write a compact pilot report and geometry comparison without additional rays."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nslens.workbench.stock_probe import read
from nslens.discovery.polygon import PolygonCodec


def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
 audit=read(root/'audit.json');summary=read(root/'summary.json')
 folders=sorted(p.parent for p in root.glob('*/task.json'))
 lines=['# Five-design local-agent CMA pilot','',
        'Qwen3.8-27B selected six bounded CMA batches per design. Two triangles, three quadrilaterals; three physically failing starts and two verified starts. Each probe started independently, with two 40-candidate batches. No spot or freeform optimization.','',
        '| Start | Sides | Initial support | Repair | 5% body compaction | 2° full FOV |',
        '|---|---:|---:|---|---|---|']
 fig,axes=plt.subplots(5,3,figsize=(12,16))
 for i,folder in enumerate(folders):
  task=read(folder/'task.json');out=read(folder/'outcome.json');codec=PolygonCodec(**task['design']['codec'])
  initial=out['outcomes']['repair']['initial']
  cells=[]
  for j,probe in enumerate(['repair','compact','field']):
   result=out['outcomes'][probe];r=result['final']
   cells.append('PASS' if result['verified'] else 'unreached')
   ax=axes[i,j]
   for genes,color,label in [(task['design']['genes'],'#999999','start'),(r['genes'],'#147a9b','final')]:
    body=codec.decode(genes);b=body.sample(600)[1];b=np.vstack([b,b[0]])
    ax.plot(b[:,0],b[:,1],color=color,label=label,lw=1.6)
   ax.set_aspect('equal');ax.grid(alpha=.15)
   shrink=1-max(r['dimensions'])/max(task['initial_dimensions'])
   ax.set_title(f"{task['task_id']} / {probe}\n{cells[-1]} · support {r['support']:.1%} · shrink {shrink:.1%}",fontsize=9)
   ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
   if i==0 and j==0:ax.legend()
  lines.append(f"| {task['task_id']} | {task['design']['corners']} | {initial['support']:.1%} | {' | '.join(cells)} |")
 fig.suptitle('Starting and final body geometry by probe\nGrey: initial; blue: final. Shrinkage alone is not verified feasibility.',fontsize=13)
 fig.tight_layout(rect=(0,0,1,.96));fig.savefig(root/'geometry_comparison.png',dpi=150);plt.close(fig)
 total={}
 for result in summary['results']:
  for stage,b in result['budget'].items():
   c=total.setdefault(stage,dict(evaluations=0,source_rays=0))
   for k,v in b.items():c[k]+=v
 lines+=['','## Budget and checks','',f"Search: {total['search']['evaluations']:,} proposals, {total['search']['source_rays']:,} actual source rays (270,000 cap).",
         f"Baseline checks: {total['initial']['source_rays']:,} rays. Final audits/replays/capacity checks: {sum(v['source_rays'] for k,v in total.items() if k.startswith('audit')):,} rays.",
         'Selection and full-stock capacity screening are separate preprocessing costs; the stored 40-design screen contains 36,360 source rays, and selected-label checks contain 9,090 rays. These are not search feedback.',
         f"Model responses: {sum(r['model_responses'] for r in audit['results'])}; invalid responses: {sum(r['invalid_responses'] for r in audit['results'])}.",
         f"All CMA populations replay exactly: {all(r['cma_replay_exact'] for r in audit['results'])}. All budget ledgers reconcile: {all(r['budgets_match'] for r in audit['results'])}.",
         f"All final optical audits replay exactly: {all(r['audit_replay_exact'] for r in audit['results'])}. All final 200k/400k capacity checks agree: {all(r['capacity_agrees'] for r in audit['results'])}.",
         'Transport interruption: the initial shared KV pool exhausted. Completed optical batches were retained and the server resumed with a larger pool. Completed-response token totals exclude early failed/interrupted requests; see transport_amendment.json and server_initial.log. This pilot is not a clean language-cost benchmark.',
         'Inference amendment: thinking was capped at 1536 tokens partway through after repeated output-limit failures. Completed batches were preserved. This is a mixed-settings development pilot, not a frozen-policy benchmark; see reasoning_amendment.json.',
         '', '## Interpretation','',
         'PASS means joint package and routing feasibility on the reserved 909-ray grid with exact replay and branch-capacity agreement. A repair pass for a good start may simply preserve its initial feasibility. Field expansion forbids body enlargement. Compaction reduces the longest body dimension by at least 5%, with neither dimension growing. Source, detector, beam and codec bounds stay fixed.',
         'Branch choice mattered: the weak triangle failed its repair probe, but its compaction branch produced a routing-feasible descendant with both body dimensions smaller, longest side down 2.0%, and dense support up from 34.7% to 66.3%. The good quadrilateral retained routing feasibility at 4.1% shrinkage. The good triangle reached 4.6% shrinkage but failed dense routing support (59.4%). The weaker quadrilateral recovered routing in its compaction branch, but its longest side grew 2.3%, violating the package constraint. These are diagnostic partial outcomes, not passes of the 5% target.',
         'The good triangle already passed the 2-degree dense field check before adaptation. Its final field pass is preserved feasibility, not a newly reached FOV. The near-failing quadrilateral improved dense repair support from 59.4% to 61.4%; its search-grid field pass did not survive dense audit.',
         'Unreached means this two-batch trajectory did not produce a verified result. It does not establish impossibility or low eventual potential. Each probe has one optimizer trajectory; this pilot does not compare model superiority or reach the eventual 10–12° field target.',
         'The 40-design preprocessing screen found 39 passes at a 200,000-branch limit, versus 13 in the original 20,000-branch run. The selected five labels were additionally checked at 400,000 branches. The remaining stock has not all received that second capacity check. Original campaign artifacts remain unchanged.',
         'The audits use the same engine and nested evenly spaced pupil grids, not independent-engine or offset-grid certification. Geometry is 2D circular arcs; imaging quality and freeform correctability remain untested.',
         '', '## Portable tasks','', '[Frontier model handoff](frontier_tasks/README.md) · [Frozen protocol](protocol.json) · [Audit ledger](audit.json) · [Geometry comparison](geometry_comparison.png)',
         '', 'Each task directory contains model_calls.jsonl, batch_*.json, state.json, and outcome.json. Reason text records model statements, not guaranteed causal explanations.']
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
