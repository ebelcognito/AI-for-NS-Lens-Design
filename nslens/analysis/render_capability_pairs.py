"""Paired setup views of selected, independently verified capability outcomes."""
from pathlib import Path
from dataclasses import replace
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nslens.discovery.folded import RoutingBrief,evaluate
from nslens.discovery.polygon import PolygonCodec


def scene(ax,task,genes,metrics,title):
    codec=PolygonCodec(**task['config']['codec']);base=RoutingBrief(**task['config']['brief'])
    brief=replace(base,fields_deg=tuple(np.linspace(min(base.fields_deg),max(base.fields_deg),3)),rays_per_field=9)
    _,body,trace=evaluate(codec,genes,brief,return_trace=True,geometry_samples=1024)
    boundary=body.sample(1000)[1];boundary=np.vstack([boundary,boundary[0]])
    ax.fill(*boundary.T,color='#dcecf1');ax.plot(*boundary.T,color='#244e62',lw=1.5)
    axis,pos,center,half,sign=brief.detector();det=np.zeros((2,2));det[:,axis]=pos;det[:,1-axis]=[center-half,center+half]
    ax.plot(*det.T,color='#72439b',lw=3,label='Detector')
    rays=brief.rays();origins=np.array([r.origin for r in rays]);ax.scatter(*origins.T,s=3,color='#187f61')
    # Show the strongest exiting branch per sampled input, including failed routes.
    strongest={}
    for ray in trace.exits:
        if ray.source_id not in strongest or ray.energy>strongest[ray.source_id].energy:strongest[ray.source_id]=ray
    colors=['#3085c1','#e29327','#cb5288'];points=[boundary,det,origins]
    for sid,ray in strongest.items():
        path=[rays[sid].origin,*[e.point for e in ray.events]]
        direction=np.asarray(ray.direction);distance=(pos-ray.origin[axis])/direction[axis] if abs(direction[axis])>1e-12 else -1
        if distance>0 and distance<150:path.append(np.asarray(ray.origin)+distance*direction)
        else:path.append(np.asarray(ray.origin)+12*direction)
        xy=np.array(path);ax.plot(*xy.T,color=colors[sid//9],alpha=.45,lw=.65)
    rms=metrics['worst_spot_rms'];ax.set_title(f'{title}\nThroughput {metrics["worst_folded_throughput"]:.1%} · RMS {rms:.3f} mm',fontsize=11)
    ax.set_aspect('equal');ax.grid(alpha=.12);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
    return np.vstack(points)


def main(root):
    read=lambda p:json.loads(p.read_text());tasks=read(root/'tasks_with_initial.json');out=root/'before_after';out.mkdir(exist_ok=True)
    manifest=[]
    for task in tasks:
        episodes=[read(p) for p in (root/'episodes'/task['id']).glob('qwen_*.json')]
        if len(episodes)!=4:raise ValueError('Finish all Qwen attempts before rendering')
        def score(r):
            m=r['outcome']['stress'];v=max(m['violations'].values())
            return (r['outcome']['success'],-v,m['worst_folded_throughput'],-m['worst_spot_rms'])
        chosen=max(episodes,key=score)
        fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
        bounds=[scene(axes[0],task,task['start'],task['initial_audits']['stress'],'Before'),scene(axes[1],task,chosen['genes'],chosen['outcome']['stress'],f'After: {chosen["arm"]}, seed {chosen["seed"]}')]
        points=np.vstack(bounds);lo=points.min(axis=0)-5;hi=points.max(axis=0)+5
        for ax in axes:ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
        status='VERIFIED REPAIR' if chosen['outcome']['success'] else 'TARGET STILL UNMET'
        fig.suptitle(f'{task["id"]} · {task["failure_group"]} · {status}',fontsize=14)
        fig.supxlabel('Illustration: 3 fields × 9 input rays, strongest exiting branch per ray. Metrics: 33 fields × 151 rays.\nBest of four Qwen outcomes selected after auditing for display only; this is not a single-attempt success rate.',fontsize=9)
        name=f'{task["id"]}_before_after.png';fig.savefig(out/name,dpi=160);plt.close(fig)
        manifest.append(dict(case_id=task['id'],arm=chosen['arm'],seed=chosen['seed'],success=chosen['outcome']['success'],file=name,selection='Post-audit best of both Qwen arms and both attempts, for illustration only.'))
        print(task['id'],status,flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
