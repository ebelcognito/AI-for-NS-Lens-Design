"""Durable unattended polishing, audit and visualization for the launched CMA run."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .search import _write

def polish_one(job):
    from .polish import run
    source,out,max_nfev=job
    run(source,out,max_nfev=max_nfev)
    return str(Path(out)/'result.json')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--search',required=True); p.add_argument('--out',required=True)
    p.add_argument('--prefix-trajectory')
    p.add_argument('--max-nfev',type=int,default=250)
    a=p.parse_args(); root=Path(a.out); root.mkdir(parents=True,exist_ok=False)
    sources=[Path(a.search)/f'seed{s}'/'result.json' for s in (301,302,303)]
    def status(stage,**kw): _write(root/'status.json',dict(stage=stage,updated_unix=time.time(),**kw))
    try:
        status('waiting_for_CMA',sources=[str(x) for x in sources])
        deadline=time.monotonic()+7200
        while not all(x.exists() for x in sources):
            if time.monotonic()>deadline: raise TimeoutError('CMA results not ready within two hours; checkpoints preserved.')
            time.sleep(5)
        status('least_squares',max_nfev_per_seed=a.max_nfev)
        jobs=[(str(s),str(root/s.parent.name),a.max_nfev) for s in sources]
        with ProcessPoolExecutor(max_workers=3) as pool: polished=list(pool.map(polish_one,jobs))
        # Training merit alone chooses the displayed candidate; independent audits are not optimization feedback.
        candidates=[str(s) for s in sources]+polished
        best=min(candidates,key=lambda x:json.loads(Path(x).read_text())['training']['loss'])
        selected=json.loads(Path(best).read_text()); _write(root/'selected.json',selected)
        status('dense_independent_audit',selected_source=best)
        subprocess.run([sys.executable,'-m','nslens.monolithic_ir.audit','--source',best,'--out',str(root/'audit.json')],check=True)
        seed=selected['seed']; search_dir=Path(a.search)/f'seed{seed}'
        # Actual initial state, generation incumbents, and accepted LS iterations; never interpolated genes.
        config=json.loads((search_dir/'configuration.json').read_text())
        trajectory=json.loads(Path(a.prefix_trajectory).read_text()) if a.prefix_trajectory else [dict(stage='initial',evaluations=0,genes=config['start_genes'])]
        trajectory+=json.loads((search_dir/'trajectory.json').read_text())
        if best in polished:
            for row in json.loads((Path(best).parent/'trajectory.json').read_text()):
                trajectory.append(dict(row,stage='least_squares',evaluations=trajectory[-1].get('evaluations',4800)))
        _write(root/'trajectory.json',trajectory)
        status('rendering',selected_source=best,states=len(trajectory))
        subprocess.run([sys.executable,'-m','nslens.monolithic_ir.render','--source',best,'--out',str(root/'visualizations'),'--trajectory',str(root/'trajectory.json')],check=True)
        subprocess.run([sys.executable,'-m','nslens.monolithic_ir.report','--out',str(root)],check=True)
        audit=json.loads((root/'audit.json').read_text())
        status('complete',selected_source=best,training=selected['training'],audit=audit,
               note='Completed experiment does not imply published performance was achieved; inspect audit gates.')
    except BaseException as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc())
        raise

if __name__=='__main__': main()



