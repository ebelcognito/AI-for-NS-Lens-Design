"""Independent post-search sampling checks and an angular acceptance report."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from unittest.mock import patch
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nslens.analysis.prepare_angular_probe import ROOT
from nslens.workbench.stock_probe import read,dump,measure
from nslens.discovery.folded import RoutingBrief,evaluate
from nslens.discovery.polygon import PolygonCodec,exact_extent
from nslens.freeform_body.trace import trace_bundle

def check(job):
    task,genes,full,cap=job
    cache=ROOT/'verification_cache'/ (hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()+'.json')
    if cache.exists(): return read(cache)['result']
    brief=replace(RoutingBrief(**task['design']['brief']), fields_deg=tuple(np.linspace(-full/2,full/2,17)),rays_per_field=101)
    codec=PolygonCodec(**task['design']['codec'])
    with patch('nslens.discovery.folded.trace_bundle',partial(trace_bundle,max_branches=cap)):
        r=evaluate(codec,genes,brief,geometry_samples=4096)
    lo,hi=exact_extent(codec.decode(genes))
    r['dimensions']=(hi-lo).tolist()
    r['package_pass']=bool(np.all((hi-lo)<=np.array(task['initial_dimensions'])+1e-10))
    r['verified']=r['feasible'] and r['package_pass']
    cache.parent.mkdir(exist_ok=True)
    dump(cache,dict(input=job,result=r))
    return r

def widths(task):
    screening=read(ROOT/f"selection_{task['task_id'][4:]}.json")['screen']
    passing=[x['full_fov_deg'] for x in screening if x['result']['feasible']]
    return sorted(set([task['original_full_fov_deg'], task['full_fov_deg']['field'], max(passing)]))

def report():
    tasks=sorted((ROOT/'qwen').glob('*/task.json'))
    jobs=[];keys=[]
    for p in tasks:
        t=read(p);name=t['task_id'];target=t['full_fov_deg']['field']
        for arm in ['initial','qwen','numerical']:
            genes=t['design']['genes'] if arm=='initial' else read(ROOT/arm/name/'outcome.json')['outcomes']['field']['final']['genes']
            for full in widths(t):
                for cap in [200000,400000]:
                    jobs.append((t,genes,full,cap));keys.append((name,arm,str(full),str(cap)))
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(check,jobs))
    audits={}
    for key,r in zip(keys,results):
        name,arm,full,cap=key
        audits.setdefault(name,{}).setdefault(arm,{}).setdefault(full,{})[cap]=r
    dump(ROOT/'dense_verification.json',dict(grid='17 angular fields x 101 endpoint-inclusive aperture rays, 4096 geometry samples',
          source_rays=sum(r['source_rays'] for r in results),results=audits))
    fig,axes=plt.subplots(3,2,figsize=(12,12))
    lines=['# Wider angular acceptance: Qwen refinement','',
      'Six selected families from the interrupted run. Each arm used six batches of 40 CMA proposals per design. Qwen chose variable groups, step sizes and restart parents; the fixed control used all variables, sigma 0.03 and best parent.', '',
      'Full widths below are input angular acceptance, not image FOV. Beam, receiver, material and codec bounds are fixed; neither body bounding-box dimension may grow. A pass requires every sampled field to satisfy the original routing gates.', '',
      '| Seed | Starting verified full width | Target full width | Initial target support | Qwen target support | Qwen target | Numerical target |',
      '|---|---:|---:|---:|---:|---|---|']
    for ax,p in zip(axes.flat,tasks):
        t=read(p);name=t['task_id'];full=str(t['full_fov_deg']['field']);rows={}
        for arm,color in [('initial','gray'),('qwen','#007b96'),('numerical','#ca7b24')]:
            pair=audits[name][arm][full];r=dict(pair['200000']);rows[arm]=r
            r['capacity_agrees']=r==pair['400000']
            ax.plot([f['field_deg'] for f in r['fields']], [f['root_fraction'] for f in r['fields']],'.-',label=arm,color=color)
        def passed(arm):
            r=rows[arm];return 'PASS' if r['verified'] and r['capacity_agrees'] else 'unreached'
        initial_width=max(float(w) for w,pair in audits[name]['initial'].items() if pair['200000']['verified'] and pair['200000']==pair['400000'])
        lines.append(f"| {name} | {initial_width:g}° | {float(full):g}° | {rows['initial']['support']:.1%} | {rows['qwen']['support']:.1%} | {passed('qwen')} | {passed('numerical')} |")
        ax.axhline(.6,color='red',ls=':',label='support gate');ax.set_title(f'{name}: {float(full):g}° full target')
        ax.set_xlabel('Input angle relative to aim (degrees)');ax.set_ylabel('Contributing source-ray fraction');ax.set_ylim(0,1.04);ax.grid(alpha=.2)
    axes.flat[0].legend();fig.tight_layout();fig.savefig(ROOT/'angular_comparison.png',dpi=160);plt.close(fig)
    lines += ['', 'Final verification used 17 angular fields × 101 aperture rays, including fields between the nine optimization fields, with matching 200k/400k branch-capacity checks. Full results include throughput, all violations and package dimensions in dense_verification.json. Final audits are separate from search feedback.', '',
      'These are selected development cases and bounded local searches, not a general Qwen advantage or proof that an unreached target is impossible. A sampled routing pass does not establish continuous-angle, imaging, manufacturing or independent-engine certification.', '',
      '[Angular comparison](angular_comparison.png) · [Protocol](protocol.json) · [Dense verification](dense_verification.json)']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':report()
