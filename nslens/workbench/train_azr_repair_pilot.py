"""Finite-action optical self-play with Qwen proposer/solver LoRA policies.

Uses exact finite-action expected-reward gradients, not AZR's original PPO/TRR++.
Optical counterfactuals are private reward tables, never policy inputs. No prose
teacher or original repair prescription is used as a supervised target.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from nslens.analysis.azr_repair_pilot import public_observation, STYLES, PLANS, write


def rounded(value):
    if isinstance(value,float): return round(value,5)
    if isinstance(value,dict): return {k:rounded(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [rounded(v) for v in value]
    return value


def solver_prompt(task):
    return ('Select a repair plan for a 2D folded optical body. Restore at least 0.80 '
            'worst-field folded throughput and 0.80 root support, retaining all stricter '
            'original requirements. Material, illumination and receiver cannot change. '
            'Use the measured initial state. Each plan gets 40 candidate evaluations. '
            'Answer exactly one letter A-F. Plans in order: '+json.dumps(PLANS)+
            '\nOptical state: '+json.dumps(rounded(public_observation(task)),separators=(',',':')))


def proposer_prompt(family):
    # Proposer sees the good starting design; it cannot send free text to the solver.
    return ('Select a perturbation recipe to create a useful optical repair exercise. '
            'Aim for tasks the current solver can sometimes solve, avoiding trivial or '
            'unrecoverable ones. Geometry is perturbed, never the optical requirements. '
            'Answer exactly one letter A-F. Recipes in order: '+json.dumps(STYLES)+
            '\nStarting family: '+json.dumps(rounded(dict(config=family['config'],
                initial=family['witness_audit'])),separators=(',',':')))


class Policy:
    def __init__(self, weights):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        self.torch=torch
        torch.set_num_threads(4);torch.manual_seed(8123)
        self.tokenizer=AutoTokenizer.from_pretrained(weights,local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(weights,local_files_only=True,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0')
        cfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],
                       bias='none',task_type='CAUSAL_LM')
        self.model=get_peft_model(base,cfg,adapter_name='solver')
        self.model.add_adapter('proposer',cfg)
        self.model.eval()
        self.ids=[self.tokenizer.encode(c,add_special_tokens=False) for c in 'ABCDEF']
        if any(len(x)!=1 for x in self.ids):raise ValueError('Action letters must be single tokens')
        self.ids=[x[0] for x in self.ids]
        self.initial={name:p.detach().cpu().clone() for name,p in self.model.named_parameters()
                      if 'lora_' in name}
        self.optimizers={}
        self.token_cache={}
        self.calls=0;self.tokens=0

    def reset(self):
        with self.torch.no_grad():
            for name,p in self.model.named_parameters():
                if name in self.initial:p.copy_(self.initial[name].to(p.device))
        self.optimizers={}

    def logits(self, prompt, role, grad=False, frozen=False):
        self.model.set_adapter(role)
        if prompt not in self.token_cache:
            ids=self.tokenizer.apply_chat_template([dict(role='user',content=prompt)],
                    tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if len(ids)>4096:raise ValueError('Prompt too long; no silent truncation')
            self.token_cache[prompt]=self.torch.tensor([ids],device='cuda:0')
        ids=self.token_cache[prompt];self.calls+=1;self.tokens+=ids.shape[1]
        def forward():
            return self.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1,self.ids].float()
        if frozen:
            with self.model.disable_adapter(),self.torch.no_grad():return forward()
        if grad:return forward()
        with self.torch.no_grad():return forward()

    def probabilities(self,prompt,role='solver',frozen=False):
        return self.logits(prompt,role,frozen=frozen).softmax(-1).cpu().numpy()

    def update(self,prompt,role,rewards,mask=None):
        torch=self.torch
        self.model.set_adapter(role)
        if role not in self.optimizers:
            self.optimizers[role]=torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],lr=2e-5)
        optimizer=self.optimizers[role]
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.train()
        reference=self.logits(prompt,role,frozen=True).softmax(-1).detach()
        log=[]
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits=self.logits(prompt,role,grad=True)
            if mask is not None:
                logits=logits.masked_fill(~torch.tensor(mask,device=logits.device),-1e9)
            probs=logits.softmax(-1)
            returns=torch.tensor(rewards,dtype=torch.float32,device=logits.device)
            kl=(probs*(probs.clamp_min(1e-9).log()-reference.clamp_min(1e-9).log())).sum()
            # Full-action reward gradient; rewards have no gradient and are simulator-derived.
            loss=-(probs*returns).sum()+.02*kl
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
            optimizer.step()
            log.append(dict(loss=loss.item(),expected_reward=(probs*returns).sum().item(),kl=kl.item()))
        self.model.gradient_checkpointing_disable();self.model.eval()
        return log


def assess(policy,tasks,cache,frozen=False):
    rows=[]
    for task in tasks:
        probs=policy.probabilities(solver_prompt(task),frozen=frozen)
        results=[cache[task['task_id'],i] for i in range(6)]
        selected=int(probs.argmax());r=results[selected]
        rows.append(dict(task_id=task['task_id'],family_id=task['family_id'],
                         plan=selected,probabilities=probs.tolist(),success=r['success'],
                         expected_success=float(sum(p*x['success'] for p,x in zip(probs,results))),
                         throughput=r['stress']['throughput'],reward=r['reward']))
    return rows


def metrics(rows):
    return dict(tasks=len(rows),successes=sum(r['success'] for r in rows),
                expected_success=float(np.mean([r['expected_success'] for r in rows])),
                mean_throughput=float(np.mean([r['throughput'] for r in rows])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('corpus',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);a=p.parse_args()
    if not (a.corpus/'build_complete.json').exists():raise ValueError('Corpus incomplete')
    a.out.mkdir(parents=True,exist_ok=False)
    tasks=json.loads((a.corpus/'tasks.json').read_text())
    families=json.loads((a.corpus/'private_families.json').read_text())
    cache={}
    for path in (a.corpus/'private_outcomes').glob('*.json'):
        r=json.loads(path.read_text());cache[r['task_id'],r['plan_id']]=r
    train=[t for t in tasks if t['split']=='train'];test=[t for t in tasks if t['split']=='test']
    val=[t for t in tasks if t['split']=='validation']
    if {t['family_id'] for t in train}&{t['family_id'] for t in val+test}:raise ValueError('Leakage')
    if not any(cache[t['task_id'],i]['success'] for t in train for i in range(6)):
        write(a.out/'blocked.json',dict(reason='No recoverable training task at this solver budget; do not train on fabricated successes.'))
        return
    write(a.out/'protocol.json',dict(model='Qwen3-8B',weights=str(a.weights),seeds=[8123,8124],
        rounds=2,steps_per_task=2,learning_rate=2e-5,kl_weight=.02,rank=8,
        arms=['uniform','learned_proposer'],floor=.8,
        method='Exact expected-reward gradient on six actions, separate proposer/solver adapters on one frozen backbone.',
        proposer_reward='4*p_success*(1-p_success), current solver probabilities over verified outcomes.',
        limits='Finite cached generator and finite CMA plans. No open-ended task synthesis, free-form editing, compaction, or original AZR algorithm.',
        comparison='Matched solver updates and common counterfactual corpus; learned proposer has additional training/inference overhead, reported separately.',
        trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256((a.corpus/'tasks.json').read_bytes()).hexdigest()))
    policy=Policy(a.weights)
    checkpoints=[]
    for seed in (8123,8124):
        for arm in ('uniform','learned_proposer'):
            policy.reset();rng=np.random.default_rng(seed);policy.torch.manual_seed(seed)
            rows=[];start=time.monotonic();before_calls=policy.calls;before_tokens=policy.tokens
            for round_id in range(2):
                for family in [f for f in families if f['split']=='train']:
                    candidates=[t for t in train if t['family_id']==family['family_id']]
                    if not candidates:continue
                    rewards=np.zeros(6);valid=np.zeros(6,dtype=bool)
                    # Solver probes only train tasks. No test-dependent curriculum or selection.
                    for t in candidates:
                        probs=policy.probabilities(solver_prompt(t))
                        success=sum(probs[i]*cache[t['task_id'],i]['success'] for i in range(6))
                        rewards[t['style']]=4*success*(1-success);valid[t['style']]=True
                    if arm=='learned_proposer':
                        proposal=policy.probabilities(proposer_prompt(family),'proposer')*valid
                        proposal/=proposal.sum()
                    else:proposal=valid/valid.sum()
                    style=int(rng.choice(6,p=proposal))
                    chosen=next(t for t in candidates if t['style']==style)
                    solver_rewards=[cache[chosen['task_id'],i]['reward'] for i in range(6)]
                    # Separate adapters prevent proposer gradients from altering solver directly.
                    proposer_log=(policy.update(proposer_prompt(family),'proposer',rewards,valid)
                                  if arm=='learned_proposer' else [])
                    solver_log=policy.update(solver_prompt(chosen),'solver',solver_rewards)
                    rows.append(dict(round=round_id+1,family=family['family_id'],task=chosen['task_id'],
                         proposal=proposal.tolist(),learnability=rewards.tolist(),solver_rewards=solver_rewards,
                         proposer_updates=proposer_log,solver_updates=solver_log))
                    write(a.out/f'{arm}_seed{seed}'/'training.json',rows)
                    print(arm,seed,'round',round_id+1,'task',chosen['task_id'],flush=True)
                path=a.out/f'{arm}_seed{seed}'/f'round{round_id+1}'
                policy.model.save_pretrained(path,selected_adapters=['solver','proposer'])
                delta={role:sum(float((p.detach().float().cpu()-policy.initial[name].float()).square().sum())
                               for name,p in policy.model.named_parameters()
                               if name in policy.initial and f'.{role}.' in name)**.5
                       for role in ('solver','proposer')}
                write(path/'parameter_change.json',dict(l2_from_initial=delta))
                checkpoints.append(dict(arm=arm,seed=seed,round=round_id+1,path=str(path)))
            write(a.out/f'{arm}_seed{seed}'/'cost.json',dict(seconds=time.monotonic()-start,
                  model_calls=policy.calls-before_calls,prompt_tokens=policy.tokens-before_tokens,
                  solver_steps=sum(len(r['solver_updates']) for r in rows),
                  proposer_steps=sum(len(r['proposer_updates']) for r in rows)))
    # All training and checkpoint creation precedes held-out evaluation.
    report={}
    for split,eval_tasks in [('validation',val),('test',test)]:
        baseline=assess(policy,eval_tasks,cache,frozen=True)
        write(a.out/f'{split}_frozen.json',baseline);report[f'{split}_frozen']=metrics(baseline)
        fixed=[]
        # Choose the fixed plan using train only, never the test oracle.
        plan=max(range(6),key=lambda i:sum(cache[t['task_id'],i]['success'] for t in train))
        for t in eval_tasks:
            r=cache[t['task_id'],plan]
            fixed.append(dict(task_id=t['task_id'],family_id=t['family_id'],plan=plan,
                              success=r['success'],expected_success=float(r['success']),
                              throughput=r['stress']['throughput']))
        write(a.out/f'{split}_fixed.json',fixed);report[f'{split}_fixed']=metrics(fixed)
        report[f'{split}_oracle']=dict(tasks=len(eval_tasks),successes=sum(
            any(cache[t['task_id'],i]['success'] for i in range(6)) for t in eval_tasks))
        for c in checkpoints:
            # Load adapter tensors without constructing another 8B backbone.
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            weights=load_file(str(Path(c['path'])/'solver'/'adapter_model.safetensors'))
            set_peft_model_state_dict(policy.model,weights,adapter_name='solver')
            result=assess(policy,eval_tasks,cache)
            name=f"{split}_{c['arm']}_seed{c['seed']}_round{c['round']}"
            write(a.out/f'{name}.json',result);report[name]=metrics(result)
            print(name,report[name],flush=True)
    write(a.out/'summary.json',report)
    write(a.out/'completion.json',dict(status='complete',checkpoints=checkpoints,
                                      total_model_calls=policy.calls,total_prompt_tokens=policy.tokens))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
