"""Static comparison chart and explicitly selected traced benchmark examples."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .folded import RoutingBrief, evaluate
from .polygon import PolygonCodec
from .run import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    a = p.parse_args()
    comparison = json.loads((a.run/'comparison.json').read_text())
    summaries = json.loads((a.run/'summary.json').read_text())
    candidate = comparison['candidate']
    arms = [arm for arm in ('naive','legacy_route','random','pilot',candidate) if arm in comparison['totals']]
    arms = list(dict.fromkeys(arms))
    labels = {'naive':'Naive CMA','random':'Uniform random','pilot':'Earlier guided CMA','portfolio':'Retained-candidate CMA'}
    labels['legacy_route'] = 'Legacy CMA-ME\n(route adaptation)'
    rates = [comparison['totals'][arm]['successes']/comparison['totals'][arm]['runs'] for arm in arms]
    fig, ax = plt.subplots(figsize=(9,4.8))
    ax.bar([labels.get(arm,arm) for arm in arms],np.array(rates)*100,
           color=['#208775' if arm == candidate else '#8094a7' for arm in arms])
    for i,arm in enumerate(arms):
        t = comparison['totals'][arm]
        ax.text(i,100*rates[i]+2,f"{t['successes']}/{t['runs']} ({rates[i]:.1%})",ha='center')
    ax.set_ylim(0,110)
    ax.set_ylabel('Runs yielding ≥1 dense-verified folded design (%)')
    ax.set_title('Fresh-seed confirmation · equal search and reserved audit budgets')
    c = next(c for c in comparison['comparisons'] if c['control']=='naive')
    lo,hi = np.array(c['seed_block_bootstrap_95'])*100
    fig.text(.5,.015,f"Gain over naive: {100*c['mean_difference']:.1f} percentage points; 95% seed-block interval [{lo:.1f}, {hi:.1f}]\nFixed 3/4/5-corner tasks and declared beam widths; same ray engine for search and verification.",ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,1))
    fig.savefig(a.run/'comparison.png',dpi=180)
    plt.close(fig)
    chosen = {}
    for s in summaries:
        if s['arm'] != candidate:
            continue
        archive = json.loads((Path(s['config']['out'])/'archive.json').read_text())
        for audit in archive['audits']:
            row = audit['result']
            key = (s['brief_id'],s['corners'])
            if row['feasible'] and (key not in chosen or row['quality']>chosen[key][1]['quality']):
                chosen[key]=(s,row)
    briefs = sorted({s['brief_id'] for s in summaries})
    corners = sorted({s['corners'] for s in summaries})
    fig, axes = plt.subplots(len(briefs),len(corners),figsize=(6*len(corners),5*len(briefs)),squeeze=False)
    verified=[]
    for i,name in enumerate(briefs):
        for j,k in enumerate(corners):
            ax=axes[i,j]
            if (name,k) not in chosen:
                ax.set_title(f'{name}, {k} corners: no passing example')
                continue
            s,row = chosen[name,k]
            brief = RoutingBrief(**s['config']['brief'])
            fields=sorted(brief.fields_deg)
            brief=replace(brief,rays_per_field=25,fields_deg=tuple(sorted(set(fields+[(x+y)/2 for x,y in zip(fields,fields[1:])]))))
            result,body,trace=evaluate(PolygonCodec(**s['config']['codec']),row['genes'],brief,return_trace=True,geometry_samples=1024)
            if not result['feasible']:
                raise RuntimeError('selected example failed replay')
            verified.append(dict(unit=s['unit'],arm=candidate,codec=s['config']['codec'],
                                 brief=s['config']['brief'],result=result))
            boundary=body.sample(1200)[1]
            boundary=np.vstack((boundary,boundary[0]))
            ax.fill(boundary[:,0],boundary[:,1],color='#d7eafa')
            ax.plot(boundary[:,0],boundary[:,1],color='#153d56',lw=1.5)
            ax.scatter(body.corners[:,0],body.corners[:,1],color='#153d56',s=14)
            rays=brief.rays()
            mid=len(brief.fields_deg)//2
            for sid in range(mid*25,(mid+1)*25,3):
                branches=[]
                for ray in trace.exits:
                    if ray.source_id != sid or ray.direction[0]<=0:
                        continue
                    tir=[e for e in ray.events if e.kind=='tir']
                    if len(tir)<brief.min_tir or any(e.critical_margin_deg<brief.min_tir_margin_deg for e in tir):
                        continue
                    distance=(brief.detector_x-ray.origin[0])/ray.direction[0]
                    end=np.asarray(ray.origin)+distance*np.asarray(ray.direction)
                    if distance>0 and abs(end[1]-brief.detector_center_y)<=brief.detector_half_height:
                        branches.append((ray,end))
                if not branches:
                    continue
                ray,end=max(branches,key=lambda pair:pair[0].energy)
                path=np.asarray([rays[sid].origin,*(e.point for e in ray.events),end])
                ax.plot(path[:,0],path[:,1],lw=.8,alpha=.7,color='#c75b20')
            src=np.asarray([r.origin for r in rays[:25]])
            ax.plot(src[:,0],src[:,1],color='#11946c',lw=4)
            ax.plot([brief.detector_x]*2,[brief.detector_center_y-brief.detector_half_height,brief.detector_center_y+brief.detector_half_height],color='#7958a3',lw=3)
            ax.set_title(f"{k} corners · {brief.beam_width_mm:g} mm beam\nworst-field folded energy {result['quality']:.1%}, source support {result['support']:.1%}",fontsize=10)
            ax.set(xlim=(0,128),ylim=(-55,58),xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
            ax.grid(alpha=.15)
    fig.suptitle('Selected confirmed designs · central-field qualifying paths shown\nFinite-width routing examples, before compaction and imaging refinement',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(a.run/'folded_examples.png',dpi=170)
    plt.close(fig)
    write(a.run/'example_verification.json',dict(examples=verified,additional_source_rays=sum(r['result']['source_rays'] for r in verified)))
    groups=[]
    for name in briefs:
        for k in corners:
            for arm in arms:
                rows=[s for s in summaries if s['brief_id']==name and s['corners']==k and s['arm']==arm]
                groups.append(dict(brief=name,corners=k,arm=arm,runs=len(rows),successes=sum(r['audited_success'] for r in rows),routes=sum(r['audited_routes'] for r in rows)))
    write(a.run/'family_breakdown.json',groups)


if __name__=='__main__':
    main()
