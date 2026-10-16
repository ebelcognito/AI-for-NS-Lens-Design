"""Separate bounded coefficient capacity from frozen-hit basis compatibility."""
import argparse
import json
from pathlib import Path

import numpy as np

from nslens.analysis.refinement_probe_3d import prepare, _ray_fit_data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    records=[]
    for path in sorted(out.glob('*__*__*.json')):
        row=json.loads(path.read_text());job=row['job']
        context=prepare(job['environment'],job['genes'],fields=3,pupils=3)
        mask,required,jacobian=_ray_fit_data(context)
        rhs=required.ravel()
        if not len(rhs):continue
        step,_,rank,_=np.linalg.lstsq(jacobian,rhs,rcond=1e-10)
        rms=float(np.sqrt(np.mean(rhs**2)))
        unexplained=float(np.sqrt(np.mean((rhs-jacobian@step)**2)))
        bounded=row['compatibility']['rms_unexplained_slope']
        active=context['active_indices']
        records.append(dict(case=path.name,environment=job['environment'],
            required_rms=rms,bounded_unexplained_rms=bounded,unbounded_unexplained_rms=unexplained,
            bounded_explained_fraction=1-(bounded/rms)**2,
            unbounded_explained_fraction=1-(unexplained/rms)**2,
            unbounded_outside_limits=int(np.sum(abs(context['genes'][active]+step)>1)),
            rank=int(rank),active_parameters=len(active),
            bound_saturated=row['compatibility']['bound_saturated']))
    assert len(records)==48
    summary={}
    for env in ('thermal','monolithic'):
        rows=[r for r in records if r['environment']==env]
        summary[env]={k:float(np.median([r[k] for r in rows])) for k in
                      ('bounded_explained_fraction','unbounded_explained_fraction','bound_saturated','rank','active_parameters')}
    result=dict(extra_initial_traces=48,summary=summary,records=records,
        interpretation='Unbounded least-squares is a frozen-hit diagnostic only; it is not a physically retraced design, lower bound on attainable RMS, or permission to violate coefficient limits.')
    (out/'capacity_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
