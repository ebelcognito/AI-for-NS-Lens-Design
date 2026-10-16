"""Separate historical routing repair from already-routed refinement moves."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from nslens.analysis.routing_transition_policy import TransitionPolicy
from nslens.analysis.mirror_adapter_pilot import save


class StagedRepairPolicy:
    def __init__(self,regular,repair):self.regular=regular;self.repair=repair
    def propose(self,obs,state,count=8,radius=.15):
        obs=np.asarray(obs);support=obs[121:226].reshape(5,21).mean(axis=1)
        if min(support)>=.99:return self.regular.propose(obs,state,count=count,radius=radius)
        # Large historical repair moves get both conservative and full tests.
        candidates=self.repair.propose(obs,state,count=(count+1)//2,radius=2*radius)
        result=[]
        for x in candidates:
            result.extend([np.asarray(state)+.5*(x-state),x])
        return np.asarray(result[:count])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--teacher',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    old=Path('experiments/routing_transition_adapter_20261006/transitions.json')
    oldrows=[r for r in json.loads(old.read_text()) if r['accepted']]
    newpaths=sorted(Path(a.teacher).glob('seed_*.json'));newrows=[r for p in newpaths for r in json.loads(p.read_text())['transitions']]
    repairs=[r for r in oldrows+newrows if r['parent_support']<.99 and r['child_support']>r['parent_support']+.01]
    regular=TransitionPolicy().fit([r['features'] for r in oldrows],[r['delta'] for r in oldrows])
    repair=TransitionPolicy().fit([r['features'] for r in repairs],[r['delta'] for r in repairs])
    policy=StagedRepairPolicy(regular,repair);joblib.dump(dict(retrieval=policy),out/'policies.joblib')
    save(out/'protocol.json',dict(old_transitions=len(oldrows),new_transitions=len(newrows),repair_transitions=len(repairs),routing_gate=.99,repair_radius=.3,half_step_radius=.15,
         source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old]+newpaths},
         teacher='Separate nearest-neighbor banks: regular uses old verified moves; repair uses strict support-increasing moves from old and64 new independent searches. No confirmation donors in either bank. Four repair moves each tested at half/full magnitude. Same8 online proposal calls.'))
    print(json.dumps(dict(repair_transitions=len(repairs),new_transitions=len(newrows))),flush=True)

if __name__=='__main__':
    from nslens.analysis.staged_repair_policy import main as run_main
    run_main()
