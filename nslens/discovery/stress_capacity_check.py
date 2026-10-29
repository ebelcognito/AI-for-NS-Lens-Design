"""Diagnose bundle-wide branch caps without changing the frozen search evaluator."""
from __future__ import annotations
import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .folded import RoutingBrief, evaluate
from .polygon import PolygonCodec
from .run import write
from nslens.freeform_body.trace import trace_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    a=p.parse_args()
    stress=json.loads((a.run/'replay_and_stress.json').read_text())['stress_examples']
    examples=json.loads((a.run/'example_verification.json').read_text())['examples']
    output=[]
    for example in examples:
        previous=next(r for r in stress if r['unit']==example['unit'])
        brief=RoutingBrief(**example['brief'])
        fields=sorted(brief.fields_deg)
        brief=replace(brief,rays_per_field=101,fields_deg=tuple(np.linspace(fields[0],fields[-1],9)))
        checks=[]
        for cap in (200000,400000):
            # Scoped diagnostic injection only. The frozen benchmark evaluator,
            # search results and original failed stress checks remain unmodified.
            with patch('nslens.discovery.folded.trace_bundle',partial(trace_bundle,max_branches=cap)):
                row,body,trace=evaluate(PolygonCodec(**example['codec']),example['result']['genes'],
                                       brief,geometry_samples=4096,return_trace=True)
            checks.append(dict(branch_cap=cap,branch_limited_fraction=trace.branch_limited_energy/trace.emitted_energy,
                               depth_limited_fraction=trace.depth_limited_energy/trace.emitted_energy,result=row))
        same=checks[0]['result']==checks[1]['result']
        output.append(dict(unit=example['unit'],original_20000_cap_pass=previous['feasible'],
                           larger_caps_agree_exactly=same,checks=checks))
        print(example['unit'],'higher-cap pass',checks[-1]['result']['feasible'],'converged',same,flush=True)
    write(a.run/'stress_capacity_check.json',dict(examples=output,
          additional_source_rays=sum(c['result']['source_rays'] for r in output for c in r['checks']),
          caveat='Post-run diagnostic on six selected examples. Optical requirements unchanged; only numerical branch capacity increased. Original capped failures preserved in replay_and_stress.json.'))


if __name__=='__main__':
    main()
