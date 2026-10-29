"""Render real folded branches at negative, central and positive target angles."""
from dataclasses import replace
from functools import partial
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nslens.analysis.prepare_angular_probe import ROOT
from nslens.workbench.stock_probe import read,dump
from nslens.discovery.folded import RoutingBrief,evaluate
from nslens.discovery.polygon import PolygonCodec
from nslens.freeform_body.trace import trace_bundle


def main():
    tasks=sorted(p.parent/'task.json' for p in (ROOT/'qwen').glob('*/outcome.json'))
    fig,axes=plt.subplots(len(tasks),2,figsize=(13,4*len(tasks)),squeeze=False);ledger=[]
    for row,p in enumerate(tasks):
        task=read(p);folder=p.parent;full=task['full_fov_deg']['field']
        final=read(folder/'outcome.json')['outcomes']['field']['final']
        codec=PolygonCodec(**task['design']['codec'])
        # Trace the exact optimization grid; draw five roots in each of three fields.
        brief=replace(RoutingBrief(**task['design']['brief']),fields_deg=tuple(np.linspace(-full/2,full/2,9)),rays_per_field=25)
        for col,(label,genes) in enumerate([('Initial',task['design']['genes']),('Qwen',final['genes'])]):
            ax=axes[row,col]
            with patch('nslens.discovery.folded.trace_bundle',partial(trace_bundle,max_branches=200000)):
                result,body,trace=evaluate(codec,genes,brief,return_trace=True,geometry_samples=4096)
            boundary=body.sample(1000)[1];boundary=np.vstack([boundary,boundary[0]])
            ax.fill(boundary[:,0],boundary[:,1],color='#dceaf3');ax.plot(boundary[:,0],boundary[:,1],color='#143c56')
            axis,pos,center,half,sign=brief.detector()
            detector=np.array([[pos,center-half],[pos,center+half]]) if axis==0 else np.array([[center-half,pos],[center+half,pos]])
            ax.plot(*detector.T,color='#8157aa',lw=3,label='Receiver')
            rays=brief.rays();points=[boundary,detector];npaths=0
            for field,color in [(0,'#3478b8'),(4,'#c88a23'),(8,'#cf5949')]:
                for local in [0,6,12,18,24]:
                    source_id=field*25+local;options=[]
                    for ray in trace.exits:
                        if ray.source_id!=source_id:continue
                        direction=np.array(ray.direction)
                        if direction[axis]*sign<=1e-9:continue
                        events=[e for e in ray.events if e.kind=='tir']
                        if len(events)<brief.min_tir or any(e.critical_margin_deg<brief.min_tir_margin_deg for e in events):continue
                        distance=(pos-ray.origin[axis])/direction[axis]
                        if distance<=0:continue
                        end=np.array(ray.origin)+distance*direction
                        options.append((ray,end,abs(end[1-axis]-center)<=half))
                    if not options:continue
                    ray,end,hit=max(options,key=lambda x:x[0].energy)
                    path=np.array([rays[source_id].origin,*[e.point for e in ray.events],end])
                    ax.plot(*path.T,color=color,lw=.7,alpha=.65,ls='-' if hit else '--');points.append(path);npaths+=1
            source=np.array([r.origin for r in rays[:25]])
            ax.plot(*source.T,color='#15966f',lw=3)
            scene=np.vstack([boundary,detector,source]);margin=5
            ax.set_xlim(scene[:,0].min()-margin,scene[:,0].max()+margin);ax.set_ylim(scene[:,1].min()-margin,scene[:,1].max()+margin)
            ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
            ax.set_title(f"{task['task_id']} / {label} / ±{full/2:g}° target\nSampled ray paths; acceptance is reported in the dense audit",fontsize=9)
            ledger.append(dict(task_id=task['task_id'],arm=label,source_rays=result['source_rays'],drawn_paths=npaths,result=result))
        xlim=(min(a.get_xlim()[0] for a in axes[row]),max(a.get_xlim()[1] for a in axes[row]))
        ylim=(min(a.get_ylim()[0] for a in axes[row]),max(a.get_ylim()[1] for a in axes[row]))
        for a in axes[row]:a.set_xlim(xlim);a.set_ylim(ylim)
    fig.suptitle('Initial and Qwen-refined designs\nBlue / gold / red: negative / central / positive input angles. Dashed paths miss the receiver.\nStrongest forward folded branch for selected roots; absent paths are not proof of zero delivered energy.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(ROOT/'ray_comparison.png',dpi=140);plt.close(fig)
    dump(ROOT/'render_verification.json',dict(records=ledger,source_rays=sum(x['source_rays'] for x in ledger)))

if __name__=='__main__':main()
