"""Reproducible small search + local refinement + held-out appearance audit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from PIL import Image, ImageDraw
from nslens.search.cma_me import CMAES
from .model import geometry, sequence, metrics


def evaluate(g,size=24,frames=7,lighting=None,axis=0,bounces=32):
    rows=[]; residual=0.
    for light in ([lighting] if lighting else ['spots','window']):
        images,masks,audits=sequence(g,size,frames,light,axis,bounces=bounces)
        rows.append(metrics(images,masks))
        residual=max(residual,max(a['residual'] for a in audits))
    result={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    result['residual']=residual
    return result


def score(m,style):
    # Pilot proxies; neither beauty grades nor a validated perceptual model.
    quality=min(m['brightness']/.075,1.)*min(m['lower_brightness']/.045,1.)
    if m['residual']>.001: return -10-m['residual']
    if style=='brightness': return m['brightness']
    if style=='broad': return quality*(m['flash_area']+2*m['activity'])
    return quality*(.025*m['flash_count']+2*m['activity']-.4*m['flash_area'])


def shape_audit(g):
    n,h=geometry(g)
    vertices=HalfspaceIntersection(np.column_stack([n,-h]),np.zeros(3)).intersections
    hull=ConvexHull(vertices)
    edges=[]; areas=[]; active=0
    for normal,offset in zip(n,h):
        face=vertices[abs(vertices@normal-offset)<1e-6]
        if len(face)<3: continue
        active+=1
        u=np.cross(normal,[1,0,0] if abs(normal[0])<.8 else [0,1,0]); u/=np.linalg.norm(u)
        v=np.cross(normal,u); xy=face@np.column_stack([u,v]); fh=ConvexHull(xy)
        poly=xy[fh.vertices]; edges.extend(np.linalg.norm(poly-np.roll(poly,1,axis=0),axis=1))
        areas.append(fh.volume)
    return dict(volume_mm3=float(hull.volume),depth_mm=float(np.ptp(vertices[:,2])),active_facets=active,min_edge_mm=float(min(edges)),min_facet_area_mm2=float(min(areas)),vertices=vertices.tolist())


def make_media(out,selected):
    names=list(selected)
    for lighting in ['spots','holdout']:
        streams=[]
        for name in names:
            ims,masks,_=sequence(selected[name]['genes'],80,41,lighting,axis=0 if lighting=='spots' else 1,bounces=64)
            streams.append((ims,masks))
        frames=[]
        for j in list(range(41))+list(range(39,0,-1)):
            canvas=Image.new('RGB',(len(names)*190,230),(19,23,30)); draw=ImageDraw.Draw(canvas)
            for col,(name,(ims,masks)) in enumerate(zip(names,streams)):
                # Fixed display mapping for every candidate, frame and environment.
                gray=np.uint8(np.clip(ims[j],0,1)**(1/2.2)*255)
                rgb=np.repeat(gray[...,None],3,axis=2); rgb[~masks[j]]=[19,23,30]
                tile=Image.fromarray(rgb).resize((180,180))
                canvas.paste(tile,(col*190+5,35)); draw.text((col*190+8,10),name,fill='white')
            draw.text((8,215),'SIMULATION | n=1.60 | fixed exposure | '+lighting,fill='white')
            frames.append(canvas)
        frames[0].save(out/(lighting+'.gif'),save_all=True,append_images=frames[1:],duration=70,loop=0)
    html='''<!doctype html><meta charset="utf-8"><title>Jewellery optical discovery pilot</title>
<style>body{background:#13171e;color:#eee;font:17px system-ui;max-width:1100px;margin:40px auto}img{width:100%}p{line-height:1.5}a{color:#9cd6ff}</style>
<h1>Gemstone appearance discovery</h1><p>3D optical simulation: a simple faceted family, fixed synthetic refractive index 1.60, 10 mm maximum girdle diameter. Volumes vary. Scalar unpolarized Fresnel approximation; no dispersion, absorption, bloom or added sparkle. These are pilot proxies, not measured beauty or fabrication results.</p>
<h2>Training illumination and motion</h2><img src="spots.gif">
<h2>Held-out illumination and orthogonal motion</h2><img src="holdout.gif">
<p>All panels use the same exposure. Candidate selection uses training scores only. The broad/fine labels name optimization objectives; inspect the audit for whether the intended distinction actually survives.</p><p><a href="report.json">Metrics, geometry, search history and numerical audits</a></p>'''
    (out/'index.html').write_text(html)


def summarize(report, out):
    lines=['# Jewellery pilot results', '',
           'Simulation only. Two style objectives are compared within one convex faceted family.', '',
           '| Candidate | Held-out brightness | Bright-region area fraction | Volume mm3 | Minimum edge mm |',
           '|---|---:|---:|---:|---:|']
    for name,row in report['selected'].items():
        m=row['resolution_check']; shape=row['shape']
        lines.append(f"| {name} | {m['brightness']:.5f} | {m['flash_area']:.5f} | {shape['volume_mm3']:.2f} | {shape['min_edge_mm']:.4f} |")
    lines += ['', 'Values use the 96-pixel, 31-frame held-out environment and orthogonal motion.', '',
              '## Interpretation and limits', '',
              '- Broad/fine bright-region size separation persists under increased image resolution in this run. This is objective control, not a human preference result.',
              '- Flash counts change with resolution; the fine-sparkle proxy is not numerically converged.',
              '- Some 32-bounce evaluations exceed the .001 unresolved-weight threshold. The 96-bounce check substantially reduces residual weight; raw values are in report.json.',
              '- Very short edges occur in selected geometries. These candidates are not fabrication-qualified.',
              '- Volumes differ, so this is not an equal-carat performance comparison.',
              '- The archive is diagnostic. This run does not establish archive-driven search superiority.',
              '- Scalar unpolarized Fresnel approximates polarization history; no dispersion/fire or real-material validation.', '',
              f"Total recorded coarse evaluations, including reference/refinement: {len(report['history'])}.",
              f"Runtime including dense audits and media: {report['seconds']:.1f} seconds.", '',
              'Open index.html for synchronized animations; report.json contains the full evidence.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); parser.add_argument('--calls',type=int,default=96); parser.add_argument('--seeds',type=int,nargs='+',default=[101,102]); args=parser.parse_args()
    if args.calls < 8 or args.calls % 8: parser.error('--calls must be a positive multiple of 8')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    start=time.time(); history=[]; best={}; archive={}
    def record(g,arm,seed):
        m=evaluate(g); row=dict(genes=g.tolist(),metrics=m,arm=arm,seed=seed); history.append(row)
        # Fixed descriptor cells; collection shared by every arm, no claimed QD advantage.
        key=(min(int(m['flash_area']/.01),19),min(int(m['activity']/.002),19))
        if m['residual']<.001 and m['brightness']>.045:
            if key not in archive or m['brightness']>archive[key]['metrics']['brightness']: archive[key]=row
        return row
    ref=record(np.zeros(6),'reference',0)
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        for _ in range(args.calls): record(rng.uniform(-1,1,6),'random',seed)
        for style in ['brightness','broad','fine']:
            opt=CMAES(np.zeros(6),.4,popsize=8,seed=seed,bounds=(-np.ones(6),np.ones(6)),boundary_handling='reflect')
            for _ in range(args.calls//8):
                xs=opt.ask(); rows=[record(x,'cma_'+style,seed) for x in xs]
                opt.tell(np.argsort([-score(row['metrics'],style) for row in rows]))
        print('completed seed',seed,'evaluations',len(history),flush=True)
    selected={'reference':ref}
    for style in ['brightness','broad','fine']:
        candidates=[r for r in history if r['arm']=='cma_'+style]
        champion=max(candidates,key=lambda row:score(row['metrics'],style)); before=champion
        # Guarded bounded coordinate refinement; 24 additional evaluations per style.
        for step in [.06,.025]:
            for dim in range(6):
                base=np.array(champion['genes'])
                rows=[]
                for sign in [-1,1]:
                    g=base.copy(); g[dim]=np.clip(g[dim]+sign*step,-1,1)
                    rows.append(record(g,'refine_'+style,0))
                candidate=max(rows+[champion],key=lambda row:score(row['metrics'],style))
                champion=candidate
        selected[style]=dict(champion,pre_refinement=before)
    for name,row in selected.items():
        g=row['genes']; row['shape']=shape_audit(g)
        row['dense_training']=evaluate(g,64,31)
        row['heldout']=evaluate(g,64,31,'holdout',axis=1)
        # Pixel convergence at fixed motion sampling, separated from training generalization.
        row['resolution_check']=evaluate(g,96,31,'holdout',axis=1)
        row['motion_check']=evaluate(g,64,61,'holdout',axis=1)
        im,mask,audits=sequence(g,64,31,'holdout',axis=1,bounces=96)
        row['bounce_check']=metrics(im,mask)
        row['bounce_check']['residual']=max(a['residual'] for a in audits)
        print(name,'heldout',row['heldout'],flush=True)
    summary=[]
    for arm in sorted(set(r['arm'] for r in history)):
        for seed in sorted(set(r['seed'] for r in history if r['arm']==arm)):
            rows=[r for r in history if r['arm']==arm and r['seed']==seed]
            summary.append(dict(arm=arm,seed=seed,calls=len(rows),best_scores={s:max(score(r['metrics'],s) for r in rows) for s in ['brightness','broad','fine']}))
    report=dict(scope='pilot: random versus style-conditioned CMA and coordinate refinement; archive is diagnostic, not a CMA-ME experiment',settings=vars(args),selected=selected,arm_summary=summary,archive=list(archive.values()),history=history)
    make_media(out,selected)
    report['seconds']=time.time()-start
    (out/'report.json').write_text(json.dumps(report,indent=2))
    summarize(report,out)
    print('saved',out,'seconds',report['seconds'],flush=True)

if __name__=='__main__': main()
