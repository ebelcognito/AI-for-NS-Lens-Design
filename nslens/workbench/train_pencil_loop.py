"""Two-round, three-arm Qwen LoRA curriculum pilot with private optical oracle cache."""
from __future__ import annotations
import argparse,copy,gc,json,random,time
from pathlib import Path
import numpy as np
from .pencil_loop import INSTRUCTIONS,TOOLS,PLANS,observation,write
from .qwen_baseline import LocalQwen,MODEL,REVISION


def select_tasks(arm,eligible,previous,rng,count=4):
    if any(t['split']!='train' for t in eligible):raise ValueError('selection accessed nontraining task')
    if not eligible:raise ValueError('no successful numerical teachers')
    weights=np.ones(len(eligible),dtype=float)
    if arm=='adaptive':
        failed=np.array([not previous[t['task_id']]['success'] for t in eligible],dtype=float)
        if failed.sum():weights=.25/len(eligible)+.75*failed/failed.sum()
    elif arm=='fixed':
        # Fixed round-robin schedule, independent of learner outcomes.
        offset=int(rng.integers(len(eligible)))
        return [eligible[(offset+i)%len(eligible)] for i in range(count)],None
    elif arm!='uniform':raise ValueError('unknown curriculum')
    weights/=weights.sum()
    indices=rng.choice(len(eligible),size=count,replace=True,p=weights)
    return [eligible[i] for i in indices],weights.tolist()


def prompt(task,cache):
    initial=cache[(task['task_id'],0)]['evaluations'][0]['verdict']
    return dict(instructions=INSTRUCTIONS,input=[dict(role='user',content=json.dumps(observation(task,initial)))],
                tools=TOOLS,max_output_tokens=128)


def assess(transport,tasks,cache,seed):
    rows=[]
    for i,task in enumerate(tasks):
        transport.torch.manual_seed(seed+i)
        response=transport(prompt(task,cache));calls=[c for c in response['output'] if c['type']=='function_call']
        index=None
        if len(calls)==1 and calls[0]['name']=='execute_plan':
            args=json.loads(calls[0]['arguments'])
            if set(args)=={'plan_id'} and type(args['plan_id']) is int and 0<=args['plan_id']<8:index=args['plan_id']
        outcome=cache[(task['task_id'],index)]['outcome'] if index is not None else None
        rows.append(dict(task_id=task['task_id'],plan_id=index,response=response,
            success=bool(outcome and outcome['success']),outcome=outcome,
            charged_optical_rays=outcome['nominal_rays'] if outcome else 51))
    return rows


def teacher(task,cache):
    candidates=[cache[(task['task_id'],i)] for i in range(8)]
    return max(candidates,key=lambda c:(c['outcome']['success'],-c['outcome']['final_audit']['merit']))


def train_round(transport,selected,cache,out,seed):
    import torch
    from .qwen_baseline import chat_messages
    model=transport.model;tok=transport.tokenizer
    data=[];selection=[]
    for task in selected:
        if task['split']!='train':raise ValueError('training boundary violation')
        chosen=teacher(task,cache)
        if not chosen['outcome']['success']:raise ValueError('unverified teacher')
        payload=prompt(task,cache)
        messages=chat_messages(payload)
        native_tools=[dict(type='function',function={k:t[k] for k in ('name','description','parameters')}) for t in TOOLS]
        prefix=tok.apply_chat_template(messages,tools=native_tools,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        messages.append(dict(role='assistant',tool_calls=[dict(type='function',function=dict(name='execute_plan',arguments=dict(plan_id=chosen['plan_id'])))]))
        full=tok.apply_chat_template(messages,tools=native_tools,tokenize=True,add_generation_prompt=False,enable_thinking=False)
        if full[:len(prefix)]!=prefix or len(full)>4096:raise ValueError('unsafe template or context truncation')
        target=len(full)-len(prefix)
        ids=[tok.pad_token_id or tok.eos_token_id]*(4096-len(full))+full
        mask=[0]*(4096-len(full))+[1]*len(full)
        data.append((ids,mask,target));selection.append(dict(task_id=task['task_id'],plan_id=chosen['plan_id'],
            unpadded_tokens=len(full),target_tokens=target))
    # Equal padded-token count and optimizer updates across all arms and rounds.
    random.Random(seed).shuffle(data);torch.manual_seed(seed)
    model.train();model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    log=[];start=time.monotonic()
    for step in range(8):
        ids,mask,target=data[step%len(data)]
        ids=torch.tensor([ids],device=model.device);mask=torch.tensor([mask],device=model.device)
        optimizer.zero_grad(set_to_none=True)
        output=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=target+1)
        loss=torch.nn.functional.cross_entropy(output.logits[0,:-1].float(),ids[0,-target:])
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();log.append(dict(step=step+1,loss=loss.item()))
        print('training',out,step+1,loss.item(),flush=True)
        del ids,mask,output,loss
    out.mkdir(parents=True,exist_ok=False);model.save_pretrained(out/'adapter')
    write(out/'training.json',dict(selection=selection,steps=log,padded_training_tokens=8*4096,
        supervised_tokens=sum(data[i%len(data)][2] for i in range(8)),elapsed_seconds=time.monotonic()-start,
        optimizer='AdamW fresh each round',learning_rate=1e-4))
    model.gradient_checkpointing_disable();model.config.use_cache=True;model.eval()
    del optimizer;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',type=Path)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if not (a.corpus/'build_complete.json').exists():raise ValueError('unfinished corpus')
    tasks=json.loads((a.corpus/'tasks.json').read_text())
    cache={}
    for f in (a.corpus/'private_plans').glob('*.json'):
        r=json.loads(f.read_text());cache[(r['task_id'],r['plan_id'])]=r
    train=[t for t in tasks if t['split']=='train'];val=[t for t in tasks if t['split']=='validation'];test=[t for t in tasks if t['split']=='test']
    if not train or not val or not test:raise ValueError('missing split')
    if any({t['family_id'] for t in left}&{t['family_id'] for t in right} for left,right in [(train,val),(train,test),(val,test)]):
        raise ValueError('prescription family leakage')
    eligible=[t for t in train if teacher(t,cache)['outcome']['success']]
    write(a.out/'config.json',dict(model=MODEL,revision=REVISION,seed=20260914,rounds=2,arms=['uniform','fixed','adaptive'],
        eligible_train=[t['task_id'] for t in eligible],all_train=[t['task_id'] for t in train],
        lora=dict(rank=8,alpha=16,targets=['q_proj','v_proj']),teacher_cache_rays=sum(r['outcome']['nominal_rays'] for r in cache.values()),
        note='Offline finite-pool curriculum; cache shared and equally charged; no learned proposer or novel topology.'))
    transport=LocalQwen(a.weights)
    baseline=assess(transport,train+val,cache,20260914)
    write(a.out/'baseline_development.json',baseline)
    initial={r['task_id']:r for r in baseline}
    del transport;gc.collect()
    import torch
    from peft import LoraConfig,get_peft_model
    torch.cuda.empty_cache()
    for arm in ['uniform','fixed','adaptive']:
        transport=LocalQwen(a.weights);torch.manual_seed(20260914)
        transport.model=get_peft_model(transport.model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
        previous=initial
        for round_index in range(2):
            rng=np.random.default_rng(20260914+round_index)
            selected,weights=select_tasks(arm,eligible,previous,rng)
            out=a.out/arm/f'round{round_index+1}'
            write(a.out/arm/f'selection{round_index+1}.json',dict(tasks=[t['task_id'] for t in selected],weights=weights,
                 eligible_order=[t['task_id'] for t in eligible],feedback_task_ids=[t['task_id'] for t in train]))
            train_round(transport,selected,cache,out,20260914+round_index)
            rows=assess(transport,train+val,cache,20260914)
            write(out/'development.json',rows);previous={r['task_id']:r for r in rows}
        del transport;gc.collect();torch.cuda.empty_cache()
    # Test is evaluated only after every arm and update has finished. No promotion or reselection follows.
    for arm,round_index in [('baseline',0)]+[(a,r) for a in ['uniform','fixed','adaptive'] for r in [1,2]]:
        adapter=None if arm=='baseline' else a.out/arm/f'round{round_index}'/'adapter'
        transport=LocalQwen(a.weights,adapter)
        rows=assess(transport,test,cache,20260914)
        write(a.out/f'test_{arm}_{round_index}.json',rows)
        del transport;gc.collect();torch.cuda.empty_cache()
    summary={}
    for f in a.out.glob('test_*.json'):
        rows=json.loads(f.read_text());summary[f.stem]=dict(successes=sum(r['success'] for r in rows),tasks=len(rows),
            mean_merit=float(np.mean([r['outcome']['final_audit']['merit'] if r['outcome'] else 1e6 for r in rows])))
    write(a.out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
