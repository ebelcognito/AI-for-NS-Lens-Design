"""Compare saved camera runs only when task and physical budget contracts match."""
import argparse
import json
from pathlib import Path


def compare(roots):
    records=[]; reference=None
    for root in roots:
        protocol=json.loads((root/'protocol.json').read_text())
        summary=json.loads((root/'summary.json').read_text())
        # Search/model interface sources can vary, but physics and geometry cannot.
        physics={str(Path(k).name):v for k,v in protocol['sources'].items()
                 if Path(k).name in ('problem.py','scene.py','trace.py','arc_polygon.py','polynomial_faces.py','polygon.py')}
        contract=(protocol['task'],protocol['candidate_budget'],protocol['ray_budget'],physics)
        if reference is None: reference=contract
        elif reference!=contract: raise ValueError('task, budget or physics mismatch; not a matched comparison')
        audit=summary.get('audit')
        records.append(dict(path=str(root),arm=protocol['arm'],seed=protocol['seed'],
            success=summary['success'],candidates=summary['candidates'],source_rays=summary['source_rays'],
            audit_source_rays=audit['source_rays'] if audit else 0,
            wall_seconds=summary.get('wall_seconds'),reported_total_tokens=summary.get('reported_total_tokens'),
            selected_candidate=summary['selected_candidate'],
            inference_protocol={k:protocol.get(k) for k in ('max_calls','max_output_tokens','reasoning_tokens')}
                if protocol['arm']=='model' else None))
    return dict(caveat='Development smoke results; use repeated matched seeds before drawing comparative conclusions.',runs=records)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+')
    print(json.dumps(compare(p.parse_args().roots),indent=2))
