"""Small QLoRA SFT pilot, with an adapter-disabled baseline and simulator evaluation."""
import argparse,json,math,random,time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from nslens.workbench.interactive_cma_pilot import read,write,validate_action,finalize
from nslens.workbench.cma_midpoint_pilot import evaluate_action


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);a=p.parse_args();root=a.out
    def status(phase,**kw):write(root/'status.json',dict(phase=phase,updated_unix=time.time(),**kw))
    torch.set_num_threads(4);torch.manual_seed(1515);random.seed(1515)
    status('loading_model')
    tokenizer=AutoTokenizer.from_pretrained(a.weights,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(a.weights,local_files_only=True,device_map={'':'cuda:0'},dtype=torch.bfloat16,attn_implementation='sdpa')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    model.get_input_embeddings().to(dtype=torch.bfloat16)
    model.get_output_embeddings().to(dtype=torch.bfloat16)
    train=read(root/'train.json');val=read(root/'validation.json')
    def encode(row):
        prefix=tokenizer.apply_chat_template([dict(role='system',content=row['system']),dict(role='user',content=row['user'])],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        target=tokenizer.encode(row['assistant'],add_special_tokens=False)+[tokenizer.eos_token_id]
        assert len(prefix)+len(target)<=8192,'No silent truncation'
        return prefix,target
    encoded={r['id']:encode(r) for r in train+val}
    params=[p for p in model.parameters() if p.requires_grad]
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    write(root/'training_protocol.json',dict(method='4-bit QLoRA supervised action tuning',weights=str(a.weights),model='Qwen3.8-27B',quantization='unsloth NF4; differs from prior GGUF Q8 baseline',rank=8,alpha=16,target_modules=['q_proj','v_proj'],epochs=3,learning_rate=2e-5,gradient_accumulation=4,seed=1515,trainable_parameters=sum(p.numel() for p in params),max_sequence_length=max(len(x)+len(y) for x,y in encoded.values()),decoding='greedy, thinking disabled, max_new_tokens=256; identical frozen/trained prompts',evaluation='All 60 setting-held-out development-validation states; final epoch predeclared, no test claim',labels='Single-seed best actions; label noise remains; not rationale training'))
    def loss_for(row):
        prefix,target=encoded[row['id']];ids=torch.tensor([prefix+target],device='cuda:0')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(input_ids=ids,use_cache=False,logits_to_keep=len(target)+1).logits[:,:-1,:]
        return torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),torch.tensor(target,device='cuda:0'))
    # Real forward/backward on the longest example, without changing weights.
    longest=max(train,key=lambda r:sum(map(len,encoded[r['id']])))
    status('memory_smoke_test');model.train();loss=loss_for(longest);assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    write(root/'memory_smoke.json',dict(passed=True,loss=loss.item(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),trainable_parameters=sum(p.numel() for p in params)))
    model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    def actions(phase,frozen=False):
        model.eval();model.gradient_checkpointing_disable();outputs=[]
        for i,row in enumerate(val):
            path=root/'model_evaluation'/phase/f'{i:03d}.json'
            if path.exists():outputs.append(read(path));continue
            prefix,_=encoded[row['id']];ids=torch.tensor([prefix],device='cuda:0')
            import contextlib
            with (model.disable_adapter() if frozen else contextlib.nullcontext()),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
            text=tokenizer.decode(generated[0,len(prefix):],skip_special_tokens=True).strip()
            action=None;error=None
            try:
                obj=json.loads(text);assert set(obj)=={'actions'} and len(obj['actions'])==1
                action=obj['actions'][0];validate_action(action,len(row['observation']['parameters']),True)
            except (ValueError,TypeError,KeyError,AssertionError) as exc:action=None;error=str(exc) or 'Invalid JSON action wrapper'
            result=dict(id=row['id'],action=action,text=text,error=error,output_tokens=len(generated[0])-len(prefix));write(path,result);outputs.append(result)
            status(phase,completed=i+1,expected=len(val))
        return outputs
    frozen=actions('frozen_actions',True)
    optimizer=torch.optim.AdamW(params,lr=2e-5)
    updates=0;logs=[];model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for epoch in range(3):
        order=list(train);random.Random(1515+epoch).shuffle(order);model.train();optimizer.zero_grad(set_to_none=True)
        for i,row in enumerate(order):
            loss=loss_for(row);assert torch.isfinite(loss)
            # Last accumulation group may have fewer than four examples.
            group_size=min(4,len(order)-(i//4)*4)
            (loss/group_size).backward();logs.append(dict(epoch=epoch+1,sample=i+1,loss=loss.item()))
            if (i+1)%4==0 or i+1==len(order):
                norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                status('training',epoch=epoch+1,sample=i+1,examples_per_epoch=len(order),optimizer_updates=updates,loss=loss.item(),peak_gpu_bytes=torch.cuda.max_memory_allocated())
                write(root/'training_log.json',logs)
                if updates==1:model.save_pretrained(root/'adapter_step_1')
        model.save_pretrained(root/f'adapter_epoch_{epoch+1}')
    delta=sum((p.detach().cpu()-initial[n]).float().square().sum().item() for n,p in model.named_parameters() if n in initial)
    assert delta>0
    write(root/'adapter_update.json',dict(squared_weight_change=delta,optimizer_updates=updates))
    model.save_pretrained(root/'adapter_final');tokenizer.save_pretrained(root/'adapter_final')
    trained=actions('trained_actions')
    # Numerical evaluation uses the existing workspace's small numerical environment.
    write(root/'evaluation_actions.json',dict(frozen=frozen,trained=trained))
    status('actions_complete',adapter=str(root/'adapter_final'),optimizer_updates=updates)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback,sys
        try:
            i=sys.argv.index('--out');write(Path(sys.argv[i+1])/'status.json',dict(phase='failed',error=traceback.format_exc(),updated_unix=time.time()))
        except Exception:pass
        raise
