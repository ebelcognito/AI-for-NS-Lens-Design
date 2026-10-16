"""Second bounded overnight experiment, only when the first dense audit fails."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .search import _write

def convert(genes,variant):
    g=list(genes)
    if variant=='xy6':
        return g[:20]+[v for k in range(4) for v in (g[20+7*k:27+7*k]+[0.]*5)]
    if variant=='freeform':
        g[20:]=[x/5 for x in g[20:]]
    else:
        for i,factor in ((0,2),(1,2),(2,3),(3,3),(4,3),(5,3),(6,2),(12,10/3),(13,10/3),(14,3),(15,3),(17,2.5)):
            g[i]/=factor
        g[7]=(.7+.5*g[7]-1.1)/.9
    return g

def worker(job):
    from . import search, model_wide, model_expanded, model_xy6
    seed,source,out,variant=job
    search.model=model_xy6 if variant=='xy6' else (model_expanded if variant=='geometry' else model_wide)
    search.run(out,seed=seed,source=source,base_evals=0,freeform_evals=7200,freeform_sigma=.025)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--previous',required=True); p.add_argument('--out',required=True)
    p.add_argument('--variant',choices=['freeform','geometry','xy6'],default='freeform')
    a=p.parse_args(); previous=Path(a.previous); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw): _write(out/'status.json',dict(stage=stage,updated_unix=time.time(),**kw))
    try:
        status('waiting_for_first_audit')
        deadline=time.monotonic()+10800
        while not (previous/'audit.json').exists():
            if time.monotonic()>deadline: raise TimeoutError('First audit not ready within three hours.')
            if (previous/'status.json').exists() and json.loads((previous/'status.json').read_text())['stage']=='failed':
                raise RuntimeError('First experiment failed; inspect its status.json.')
            time.sleep(5)
        audit=json.loads((previous/'audit.json').read_text())
        if audit['passes_all_gates']:
            status('not_needed',reason='Previous candidate passed all pilot gates.'); return
        source=json.loads((previous/'selected.json').read_text())
        source['genes']=convert(source['genes'],a.variant)
        source['conversion']=f'Physical-prescription-preserving {a.variant} bound expansion; see refine_run.convert.'
        _write(out/'warm_start.json',source)
        # Prefix is the actual previous winning lineage expressed in the new, equivalent units.
        while not (previous/'trajectory.json').exists(): time.sleep(2)
        prefix=json.loads((previous/'trajectory.json').read_text())
        for row in prefix:
            row['genes']=convert(row['genes'],a.variant)
            row['stage']='previous_'+row.get('stage','initial')
        _write(out/'prefix_trajectory.json',prefix)
        status('wide_CMA',seeds=[301,302,303],candidate_evaluations_per_seed=7200)
        with ProcessPoolExecutor(max_workers=3) as pool:
            list(pool.map(worker,[(s,str(out/'warm_start.json'),str(out/'search'/f'seed{s}'),a.variant) for s in (301,302,303)]))
        status('polish_audit_render')
        subprocess.run([sys.executable,'-m','nslens.monolithic_ir.finish_run','--search',str(out/'search'),'--out',str(out/'polish'),'--prefix-trajectory',str(out/'prefix_trajectory.json')],check=True)
        status('complete',final_status=str(out/'polish'/'status.json'))
    except BaseException as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc()); raise

if __name__=='__main__': main()


