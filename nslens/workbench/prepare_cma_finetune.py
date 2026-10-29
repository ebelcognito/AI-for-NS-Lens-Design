"""One SFT action target per state; development-validation split by task setting."""
import hashlib,json
from pathlib import Path
from nslens.workbench.interactive_cma_pilot import SYSTEM,read,write,digest
ROOT=Path('/playpen/hisham26/qwen_cma_finetune_20260915')
SOURCES=[Path('/playpen/hisham26/cma_midpoint_pilot_20260915'),Path('/playpen/hisham26/qwen_new_midpoints_20260915')]

def main():
    rows=[];seen=set()
    for source in SOURCES:
        for path in sorted((source/'episodes').glob('*/result.json')):
            r=read(path);obs=read(path.parent/'observation.json');h=digest(obs)
            if h in seen:continue
            seen.add(h)
            arms={k:v for k,v in r['arms'].items() if v['action'] is not None}
            order=['fixed','half','double','qwen']
            best=max(arms,key=lambda k:(arms[k]['outcome']['reward'],-order.index(k)))
            rewards=[a['outcome']['reward'] for a in arms.values()]
            informative=max(rewards)-min(rewards)>=.05
            anchor=all(a['outcome']['success'] for a in arms.values())
            target=dict(arms[best]['action']);target['rationale']=''
            prompt=SYSTEM+'\nReturn exactly {"actions":[one action]}. Set rationale to an empty string. Choose using only the supplied observation.'
            rows.append(dict(id=source.name+'/'+r['key'],family_id=r['family_id'],source_result=str(path),midpoint_path=str(path.parent/'midpoint.json'),seed=r['seed'],task_source=str(source/'tasks.json'),task_id=r['task_id'],observation=obs,observation_hash=h,informative=informative,anchor=anchor,target_arm=best,target=target,system=prompt,user=json.dumps(dict(observation=obs),separators=(',',':')),assistant=json.dumps(dict(actions=[target]),separators=(',',':'))))
    families=sorted({r['family_id'] for r in rows},key=lambda f:hashlib.sha256(('cma-sft-v1/'+f).encode()).hexdigest())
    validation=set(families[::4]);train=[r for r in rows if r['family_id'] not in validation and r['informative']]
    anchors=[r for r in rows if r['family_id'] not in validation and r['anchor'] and not r['informative']]
    train+=anchors[:min(len(anchors),len(train))]
    val=[r for r in rows if r['family_id'] in validation]
    assert train and val and not ({r['family_id'] for r in train}&{r['family_id'] for r in val})
    for name,data in [('train',train),('validation',val),('all_states',rows)]:write(ROOT/f'{name}.json',data)
    write(ROOT/'data_protocol.json',dict(train_states=len(train),informative_train_states=sum(r['informative'] for r in train),validation_states=len(val),train_families=sorted({r['family_id'] for r in train}),validation_families=sorted(validation),split='Deterministic setting-level development validation, not untouched test',labels='Best measured single-seed action per informative state; equal number or fewer all-pass unchanged-continuation anchors; empty rationales, no invented reasoning',sources=[str(p) for p in SOURCES],source_observation_hashes=[r['observation_hash'] for r in rows]))
    print(read(ROOT/'data_protocol.json'))
if __name__=='__main__':main()
