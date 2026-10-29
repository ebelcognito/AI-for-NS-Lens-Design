"""Fixed-plan control with identical candidate allowances and final audits."""
from concurrent.futures import ProcessPoolExecutor
from nslens.analysis.prepare_angular_probe import ROOT
from nslens.workbench.stock_probe import initialize,step,finalize,read,dump

def run(folder):
    initialize(folder)
    while read(folder/'state.json')['counts']['field']<6:
        step(folder,dict(probe='field',group='all',sigma=.03,parent='best',reason='Predeclared fixed numerical control.'))
    result=finalize(folder)
    print(folder.name,result['outcomes']['field']['verified'],flush=True)
    return result

if __name__=='__main__':
    folders=sorted(p.parent for p in (ROOT/'numerical').glob('*/task.json'))
    with ProcessPoolExecutor(max_workers=4) as pool: results=list(pool.map(run,folders))
    dump(ROOT/'numerical/summary.json',dict(results=results))
