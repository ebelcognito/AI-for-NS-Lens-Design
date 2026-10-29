"""One predeclared LoRA pass over verified, train-only optical tool transcripts."""
from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import time

from nslens.analysis.run_repair_curriculum import digest, write_json
from .qwen_baseline import MODEL, REVISION


def examples(rows, tokenizer):
    result = []
    for row in rows:
        messages = copy.deepcopy(row['messages'])
        task = json.loads(messages[1]['content'])
        if task['split'] != 'train' or row['metadata']['family_id'] != task['family_id']:
            raise ValueError('training data boundary violation')
        tools = [dict(type='function', function={k:t[k] for k in ('name','description','parameters')}) for t in row['tools']]
        for message in messages:
            for call in message.get('tool_calls', []):
                call['function']['arguments'] = json.loads(call['function']['arguments'])
        for i, message in enumerate(messages):
            if message['role'] != 'assistant':
                continue
            prefix = tokenizer.apply_chat_template(messages[:i], tools=tools, tokenize=True,
                                                   add_generation_prompt=True, enable_thinking=False)
            full = tokenizer.apply_chat_template(messages[:i+1], tools=tools, tokenize=True,
                                                 add_generation_prompt=False, enable_thinking=False)
            # Use exactly the native inference prefix, including empty think block.
            if full[:len(prefix)] != prefix:
                raise ValueError('chat template does not preserve the inference prefix')
            if len(full) >= 32768 or len(full) <= len(prefix):
                raise ValueError('invalid or oversized training example; never silently truncate')
            result.append(dict(task_id=task['task_id'], family_id=task['family_id'],
                               strategy=row['metadata']['strategy'], input_ids=full,
                               prefix_length=len(prefix), target_tokens=len(full)-len(prefix)))
    return result


def select_training_rows(rows, verified, mode):
    if mode == 'all_successes':
        return rows, {}
    strategies=sorted({e['strategy'] for e in verified.values()})
    stats={}
    task_sets=[]
    for strategy in strategies:
        episodes=[e for e in verified.values() if e['strategy']==strategy]
        if any(e['task']['split']!='train' for e in episodes):
            raise ValueError('teacher selection may only use training families')
        task_sets.append({e['task']['task_id'] for e in episodes})
        stats[strategy]=dict(success_rate=statistics.mean(e['outcome']['success'] for e in episodes),
            worst_rms=max(e['outcome']['final_audit']['rms'] for e in episodes),
            mean_rms=statistics.mean(e['outcome']['final_audit']['rms'] for e in episodes))
    if not all(tasks==task_sets[0] for tasks in task_sets):
        raise ValueError('strategies must be compared on identical training tasks')
    chosen=max(strategies,key=lambda name:(stats[name]['success_rate'],-stats[name]['worst_rms'],-stats[name]['mean_rms']))
    selected=[row for row in rows if row['metadata']['strategy']==chosen]
    return selected,dict(chosen=chosen,training_only_scores=stats,
                        rule='maximum success rate, then minimum worst RMS, then minimum mean RMS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('corpus',type=Path)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--teacher-selection',choices=['all_successes','robust_plan'],default='all_successes')
    a=p.parse_args()
    if a.out.exists():
        p.error('new output directory required')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    seed=20260910
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    source=a.corpus/'train_sft.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    verified = {}
    for line in (a.corpus/'train_episodes.jsonl').read_text().splitlines():
        episode = json.loads(line)
        verified[(episode['task']['task_id'], episode['strategy'])] = episode
    for row in rows:
        key = (row['metadata']['task_id'], row['metadata']['strategy'])
        episode = verified[key]
        if not episode['outcome']['success'] or episode['task']['split'] != 'train':
            raise ValueError('teacher is not a verified training success')
        if row['messages'] != episode['messages']:
            raise ValueError('training transcript differs from executed teacher')
    rows,selection=select_training_rows(rows,verified,a.teacher_selection)
    tokenizer=AutoTokenizer.from_pretrained(a.weights,local_files_only=True)
    data=examples(rows,tokenizer)
    a.out.mkdir(parents=True)
    config=dict(model=MODEL,revision=REVISION,seed=seed,epochs=1,learning_rate=1e-4,
        rank=8,alpha=16,dropout=0.,target_modules=['q_proj','v_proj'],
        microbatch=1,gradient_accumulation=4,loss='assistant completion tokens only',
        selection=a.teacher_selection,selection_details=selection,
        trajectories=len(rows),examples=len(data),families=sorted({d['family_id'] for d in data}),
        max_length=max(len(d['input_ids']) for d in data),
        source_sha256=digest(source),trainer_sha256=digest(__file__),
        packages={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','accelerate']})
    write_json(a.out/'config.json',config)
    (a.out/'trainer_snapshot.py').write_text(Path(__file__).read_text())
    write_json(a.out/'examples.json', [{k:v for k,v in d.items() if k!='input_ids'} for d in data])
    print(json.dumps(config),flush=True)
    model=AutoModelForCausalLM.from_pretrained(a.weights,local_files_only=True,dtype=torch.bfloat16,
                                             attn_implementation='sdpa').to('cuda:0')
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
        target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    model.train()
    model.print_trainable_parameters()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    random.shuffle(data)
    start=time.monotonic()
    history=[]
    for i,d in enumerate(data):
        ids=torch.tensor([d['input_ids']],device='cuda:0')
        count=d['target_tokens']
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=count+1)
        loss=torch.nn.functional.cross_entropy(output.logits[0,:-1].float(),ids[0,-count:])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        group_start=(i//4)*4
        group_size=min(4,len(data)-group_start)
        (loss/group_size).backward()
        if (i+1)%4==0 or i+1==len(data):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append(dict(step=i+1,loss=loss.item(),elapsed_seconds=time.monotonic()-start))
        if (i+1)%4==0 or i+1==len(data):
            print(json.dumps(history[-1]),flush=True)
            write_json(a.out/'training_log.json',history)
        del output,loss,ids
    model.save_pretrained(a.out/'adapter')
    write_json(a.out/'complete.json',dict(complete=True,examples=len(data),
        elapsed_seconds=time.monotonic()-start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        adapter_files={p.name:digest(p) for p in (a.out/'adapter').iterdir() if p.is_file()}))


if __name__=='__main__':
    main()
