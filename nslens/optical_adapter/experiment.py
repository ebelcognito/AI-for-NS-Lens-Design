"""Train adapters and measure raw and guarded optical corrections on held-out families."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from .data import write, validate_dataset
from .network import NumericPolicy, tiny_qwen, pretrained_qwen, save_adapter, load_adapter
from .state import observe, apply_action
from nslens.spectrometer import model


def tensors(rows, device):
    return [torch.tensor(np.asarray([r[k] for r in rows]), dtype=torch.float32, device=device)
            for k in ('cells', 'state', 'action')]


def scalar_metrics(g, dense=False):
    r = model.evaluate(g, **(dict(fields=5, waves=5, pupils=7) if dense else {}))
    return {k: v for k, v in r.items() if not isinstance(v, (list, dict))}


def evaluate_policy(policy, rows, device, rollout_steps, seed):
    policy.eval(); result=[]; rng=np.random.default_rng(seed)
    cells, state, target = tensors(rows, device)
    with torch.no_grad():
        prediction = policy(cells, state).cpu().numpy()
    for row, action in zip(rows, prediction):
        g=np.asarray(row['genes']); radius=row['radius']
        candidate=apply_action(g, action, radius)
        before=scalar_metrics(g); after=scalar_metrics(candidate)
        random_g=apply_action(g, rng.uniform(-1, 1, 36), radius)
        random_result=scalar_metrics(random_g)
        result.append(dict(family=row['family'], step=row['step'], before=before, raw_after=after,
            retained_loss=min(before['loss'], after['loss']), random_after=random_result,
            action=action.tolist(), teacher_loss=row['teacher_loss']))
    rollouts=[]
    # One independent closed-loop episode per held-out family. No selecting on audit.
    for family in sorted({r['family'] for r in rows}):
        first=next(r for r in rows if r['family']==family)
        g=np.array(first['genes']); radius=first['radius']; initial=g.copy()
        best=scalar_metrics(g); history=[]
        for step in range(rollout_steps):
            c,s=observe(g,radius)
            with torch.no_grad():
                action=policy(torch.from_numpy(c)[None].to(device),torch.from_numpy(s)[None].to(device))[0].cpu().numpy()
            candidate=apply_action(g,action,radius); metric=scalar_metrics(candidate)
            accepted=metric['loss']<best['loss']
            history.append(dict(step=step,raw_loss=metric['loss'],accepted=accepted))
            if accepted:g,best=candidate,metric
        rollouts.append(dict(family=family,initial_genes=initial.tolist(),final_genes=g.tolist(),
            before_audit=scalar_metrics(initial,True),after_audit=scalar_metrics(g,True),history=history))
    return dict(action_mse=float(np.mean((prediction-target.cpu().numpy())**2)),
        raw_improvement_fraction=float(np.mean([r['raw_after']['loss']<r['before']['loss'] for r in result])),
        mean_raw_loss_change=float(np.mean([r['raw_after']['loss']-r['before']['loss'] for r in result])),
        mean_retained_loss_change=float(np.mean([r['retained_loss']-r['before']['loss'] for r in result])),
        random_improvement_fraction=float(np.mean([r['random_after']['loss']<r['before']['loss'] for r in result])),
        one_step=result,rollouts=rollouts,
        evaluation_traces=dict(one_step=3*len(rows),rollout=len(rollouts)*(3+2*rollout_steps)),
        interpretation='Development validation only. Loss guard is not a physical feasibility certificate. '
                       'Teacher spends multiple probes; it is not a matched-budget performance baseline.')


def run(args):
    if args.epochs<1 or args.batch_size<1 or args.rollout_steps<1 or args.lr<=0:
        raise ValueError('Training and rollout settings must be positive')
    data_path=Path(args.data); data=json.loads(data_path.read_text());validate_dataset(data)
    if data['protocol']['model_sha256'] != hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest():
        raise ValueError('Simulator changed since data generation')
    train=[r for r in data['rows'] if r['split']=='train']
    validation=[r for r in data['rows'] if r['split']=='validation']
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'source';snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(source,snapshot/source.name)
    shutil.copyfile(model.__file__,snapshot/'spectrometer_model.py')
    write(out/'protocol.json',dict(**vars(args),dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        train_rows=len(train),validation_rows=len(validation),selection='final epoch; validation never used for updates or checkpoint selection',
        loss='smooth L1 on continuous teacher corrections',backbone='frozen; gradients traverse backbone to optical encoder',
        pretrained_comparison=args.mode=='pretrained',scope='local correction feasibility pilot, fixed optical target',
        dependencies=dict(torch=torch.__version__)))
    torch.set_num_threads(args.threads)
    summaries={}
    arms=['numeric']+(['tiny_qwen'] if args.mode=='tiny' else ['pretrained_qwen'] if args.mode=='pretrained' else [])
    for arm in arms:
        torch.manual_seed(args.seed);started=time.monotonic()
        if arm=='numeric':policy=NumericPolicy();device='cpu'
        elif arm=='tiny_qwen':policy=tiny_qwen(args.seed);device='cpu'
        else:
            if not args.weights:raise ValueError('--weights is required for pretrained mode')
            policy=pretrained_qwen(args.weights,args.device);device=args.device
        params=[p for p in policy.parameters() if p.requires_grad]
        initial=[p.detach().cpu().clone() for p in params]
        optimizer=torch.optim.AdamW(params,lr=args.lr)
        cells,state,target=tensors(train,device);generator=torch.Generator().manual_seed(args.seed)
        losses=[]; gradient_evidence=None
        policy.train()
        for epoch in range(args.epochs):
            order=torch.randperm(len(train),generator=generator);total=0.
            for ids in order.split(args.batch_size):
                ids=ids.to(device);optimizer.zero_grad(set_to_none=True)
                prediction=policy(cells[ids],state[ids]);loss=F.smooth_l1_loss(prediction,target[ids])
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                loss.backward()
                if gradient_evidence is None:
                    gradient_evidence={name:sum(p.grad.float().square().sum().item() for p in module.parameters() if p.grad is not None)
                        for name,module in [('encoder',policy.encoder),('head',policy.head)]}
                    if hasattr(policy,'backbone'):
                        gradient_evidence['backbone_has_gradients']=any(p.grad is not None for p in policy.backbone.parameters())
                        gradient_evidence['cell_projection']=sum(p.grad.float().square().sum().item() for p in policy.cell_projection.parameters() if p.grad is not None)
                        if gradient_evidence['backbone_has_gradients']:raise RuntimeError('Frozen backbone received gradients')
                    if not gradient_evidence['encoder']>0 or not gradient_evidence['head']>0:
                        raise RuntimeError('Optical encoder or head is disconnected')
                    write(out/f'{arm}_gradient_check.json',gradient_evidence)
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm):raise RuntimeError('Non-finite gradients')
                optimizer.step();total+=loss.item()*len(ids)
            losses.append(total/len(train))
            write(out/'status.json',dict(phase='training',arm=arm,epoch=epoch+1,epochs=args.epochs,train_loss=losses[-1]))
            if epoch==0 or (epoch+1)%10==0 or epoch+1==args.epochs:
                print(json.dumps(dict(arm=arm,epoch=epoch+1,train_loss=losses[-1])),flush=True)
        training_seconds=time.monotonic()-started
        delta=sum((p.detach().cpu()-old).square().sum().item() for p,old in zip(params,initial))
        if not delta>0:raise RuntimeError('No adapter parameter update')
        save_adapter(policy,out/f'{arm}.pt')
        # Verify a serialization round trip before reporting measurements.
        policy.eval()
        with torch.no_grad():reference=policy(cells[:1],state[:1]).clone()
        load_adapter(policy,out/f'{arm}.pt')
        with torch.no_grad():torch.testing.assert_close(policy(cells[:1],state[:1]),reference,rtol=0,atol=0)
        evaluation=evaluate_policy(policy,validation,device,args.rollout_steps,args.seed+1000)
        summary=dict(metadata=policy.metadata,trainable_parameters=sum(p.numel() for p in params),
            frozen_parameter_storage_numel=sum(p.numel() for p in policy.parameters() if not p.requires_grad),
            frozen_logical_parameters=sum(math.prod(p.quant_state.shape) if getattr(p,'quant_state',None) is not None else p.numel() for p in policy.parameters() if not p.requires_grad),
            gradient_evidence=gradient_evidence,train_losses=losses,adapter_squared_change=delta,training_seconds=training_seconds,
            total_seconds=time.monotonic()-started,**evaluation)
        write(out/f'{arm}.json',summary)
        summaries[arm]={k:v for k,v in summary.items() if k not in ('one_step','rollouts','train_losses')}
        del policy,optimizer,params,initial,cells,state,target
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write(out/'summary.json',summaries)
    lines=['# Optical neural adapter development pilot','',
        'A continuous optical encoder and residual action head were trained on simulator-verified corrections.',
        ('Pretrained mode uses the cached 27B NF4 backbone with frozen weights and trainable optical adapters.' if args.mode=='pretrained' else
         'Tiny mode, if selected, uses random reduced-size Qwen3.5 layers: an architecture smoke experiment, not pretrained Qwen.'),
        'Final epochs were evaluated on held-out CMA seed families at one fixed spectrometer target. No paper-reproduction claim.','',
        '| Arm | Trainable parameters | Raw improving actions | Mean raw loss change | Training seconds |',
        '|---|---:|---:|---:|---:|']
    for name,s in summaries.items():
        lines.append(f"| {name} | {s['trainable_parameters']} | {s['raw_improvement_fraction']:.1%} | {s['mean_raw_loss_change']:.6g} | {s['training_seconds']:.1f} |")
    lines+=['','Negative loss change is better. Raw proposals can degrade optics; retained results use a simulator loss guard.',
        'The random-action control uses one candidate per state; teacher labels use multiple probes and have unequal cost.',
        'Dense rollout audits report WFE and survival separately in each arm JSON. Validation has few independent families.',
        'Training data costs are in dataset.json; observation traces and all candidate evaluations are counted.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(out/'status.json',dict(phase='complete',arms=arms))
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['tiny','numeric','pretrained'],default='tiny')
    p.add_argument('--weights');p.add_argument('--device',default='cuda:1')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--rollout-steps',type=int,default=4)
    run(p.parse_args())


if __name__=='__main__':main()
