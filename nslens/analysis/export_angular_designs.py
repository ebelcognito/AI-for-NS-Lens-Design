"""Export verified prescriptions with the widened brief attached explicitly."""
from pathlib import Path
import numpy as np
from nslens.analysis.prepare_angular_probe import ROOT
from nslens.workbench.stock_probe import read,dump,digest


def main():
    audits=read(ROOT/'dense_verification.json')['results']
    dest=ROOT/'designs';dest.mkdir(exist_ok=True);selected=[];exports=[]
    for p in sorted((ROOT/'qwen').glob('*/task.json')):
        task=read(p);name=task['task_id'];full=task['full_fov_deg']['field'];options=[]
        for arm in ['qwen','numerical']:
            pair=audits[name][arm][str(full)];r=pair['200000']
            if not r['verified'] or r!=pair['400000']:continue
            brief=task['design']['brief'] | dict(fields_deg=np.linspace(-full/2,full/2,17).tolist(),rays_per_field=101)
            export=dict(task_id=name,method=arm,source_unit=task['design']['unit'],
                source_archive=task['design']['source_archive'],source_coarse_evaluation=task['design']['coarse_evaluation'],
                codec=task['design']['codec'],genes=r['genes'],brief=brief,full_input_acceptance_deg=full,
                initial_dimensions=task['initial_dimensions'],final_dimensions=r['dimensions'],
                dense_audit=r,branch_capacity_agrees=True,
                verification='17 fields x 101 aperture positions; same analytic engine at 200k/400k branches; sampled 2D angular routing only.')
            output=dest/f'{name}_{arm}.json';dump(output,export)
            item=dict(task_id=name,method=arm,path=str(output),sha256=digest(output),full_input_acceptance_deg=full,
                      support=r['support'],folded_throughput=r['quality'])
            exports.append(item);options.append(item)
        if options:
            # Comparison is within one fixed brief only. This margin is not a tolerance test.
            selected.append(max(options,key=lambda x:(min(x['support']/.6-1,x['folded_throughput']/.25-1),x['folded_throughput'])))
    dump(dest/'manifest.json',dict(exports=exports,recommended_per_family=selected,
         selection='Among verified finals within each identical task, prefer larger minimum relative support/throughput margin; throughput breaks ties. No cross-receiver ranking or manufacturing robustness claim.'))
    print('Exported',len(exports),'verified prescriptions; retained',len(selected),'family recommendations.')

if __name__=='__main__':main()
