"""Matched archive-guided CMA experiment. Archive controls ranking AND restarts."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from nslens.search.cma_me import CMAES
from .pilot import evaluate, score, shape_audit
from .model import sequence

BINS=(12,15)
WIDTHS=(.01,.002)


def cell(m):
    values=np.array([m['flash_area'],m['activity']])
    if not np.isfinite(values).all() or np.any(values<0): return None
    key=tuple(int(v) for v in np.floor(values/np.array(WIDTHS)))
    if any(k>=n for k,n in zip(key,BINS)): return None
    if not np.isfinite([m['brightness'],m['lower_brightness'],m['residual']]).all(): return None
    if m['brightness']<.045 or m['lower_brightness']<.025 or m['residual']>.001: return None
    return key


def add(archive,row,metric='metrics'):
    m=row[metric]; key=cell(m)
    if key is None: return False
    if key not in archive or m['brightness']>archive[key][metric]['brightness']:
        archive[key]=row
        return True
    return False


def ranks(rows,archive):
    """Rank against pre-generation archive, independent of insertion order."""
    result=[]
    for row in rows:
        m=row['metrics']; key=cell(m)
        if key is None:
            result.append((0,-max(0,.045-m['brightness'])-max(0,.025-m['lower_brightness'])-max(0,m['residual']-.001)))
        elif key not in archive: result.append((2,m['brightness']))
        else: result.append((1,m['brightness']-archive[key]['metrics']['brightness']))
    return np.array(sorted(range(len(rows)),key=lambda i:result[i],reverse=True))


def run_arm(arm,seed,calls,evaluator=evaluate):
    rng=np.random.default_rng(seed); archive={}; history=[]; restarts=[]; curve=[]
    opt=None; stale=0
    for batch in range(calls//8):
        # All arms start from the same centre. Portfolio changes style every calls/3.
        style=['brightness','broad','fine'][min(2,batch//(calls//24))]
        restart=opt is None or batch%4==0 or (arm=='archive' and stale>=3)
        if arm!='random' and restart:
            parent=None
            if batch==0: center=np.zeros(6)
            elif arm=='archive' and archive:
                parent=sorted(archive)[int(rng.integers(len(archive)))]; center=np.array(archive[parent]['genes'])
            else: center=rng.uniform(-1,1,6)
            opt=CMAES(center,.4,popsize=8,seed=int(rng.integers(2**31)),bounds=(-np.ones(6),np.ones(6)),boundary_handling='reflect')
            restarts.append(dict(batch=batch,parent_cell=parent,center=center.tolist())); stale=0
        xs=rng.uniform(-1,1,(8,6)) if arm=='random' else opt.ask()
        rows=[dict(genes=x.tolist(),metrics=evaluator(x),arm=arm,seed=seed,eval=len(history)+i+1) for i,x in enumerate(xs)]
        if arm=='archive': opt.tell(ranks(rows,archive))
        elif arm=='portfolio': opt.tell(np.argsort([-score(r['metrics'],style) for r in rows]))
        gains=sum(add(archive,row) for row in rows)
        stale=0 if gains else stale+1
        history.extend(rows)
        curve.append(dict(calls=len(history),cells=len(archive),qd_score=sum(r['metrics']['brightness'] for r in archive.values())))
    return dict(arm=arm,seed=seed,history=history,archive=list(archive.values()),restarts=restarts,curve=curve)


def audit(run):
    # Every coarse elite gets identical audit settings; no held-out winner selection.
    for row in run['archive']:
        row['dense']=evaluate(row['genes'],48,15,bounces=96)
        row['heldout']=evaluate(row['genes'],48,15,'holdout',axis=1,bounces=96)
    dense={}; hold={}
    for row in run['archive']:
        add(dense,row,'dense'); add(hold,row,'heldout')
    run['summary']=dict(coarse_cells=len(run['archive']),dense_cells=len(dense),heldout_cells=len(hold),dense_qd=sum(r['dense']['brightness'] for r in dense.values()),heldout_qd=sum(r['heldout']['brightness'] for r in hold.values()),audited_candidates=len(run['archive']))
    return dense


def select_gallery(rows,count=6):
    archive={}
    for row in rows: add(archive,row,'dense')
    pool=list(archive.values())
    if not pool: return []
    # Greedy spread selected solely on dense training descriptors.
    chosen=[max(pool,key=lambda r:r['dense']['brightness'])]
    def xy(r): return np.array([r['dense']['flash_area']/.01,r['dense']['activity']/.002])
    while len(chosen)<min(count,len(pool)):
        remaining=[r for r in pool if all(r is not c for c in chosen)]
        chosen.append(max(remaining,key=lambda r:min(np.linalg.norm(xy(r)-xy(c)) for c in chosen)))
    return sorted(chosen,key=lambda r:r['dense']['flash_area'])


def media(out,galleries):
    for arm,rows in galleries.items():
        if not rows: continue
        for lighting in ['spots','holdout']:
            streams=[sequence(r['genes'],64,25,lighting,axis=int(lighting=='holdout'),bounces=96)[:2] for r in rows]
            frames=[]
            for j in list(range(25))+list(range(23,0,-1)):
                canvas=Image.new('RGB',(len(rows)*155,190),(19,23,30)); draw=ImageDraw.Draw(canvas)
                for k,((ims,masks),row) in enumerate(zip(streams,rows)):
                    gray=np.uint8(np.clip(ims[j],0,1)**(1/2.2)*255); rgb=np.repeat(gray[...,None],3,axis=2); rgb[~masks[j]]=[19,23,30]
                    canvas.paste(Image.fromarray(rgb).resize((145,145)),(k*155+5,25))
                    draw.text((k*155+5,5),f"{arm} {k+1}",fill='white')
                draw.text((5,175),'SIMULATION | fixed exposure | '+lighting,fill='white'); frames.append(canvas)
            frames[0].save(out/f'{arm}_{lighting}.gif',save_all=True,append_images=frames[1:],duration=90,loop=0)
    parts=['<!doctype html><meta charset="utf-8"><title>Archive-driven gemstone discovery</title><style>body{background:#13171e;color:#eee;font:17px system-ui;max-width:1100px;margin:40px auto}img{width:100%}a{color:#9cd6ff}td,th{padding:8px;border-bottom:1px solid #555}</style><h1>Archive-driven gemstone discovery</h1><p>Each method receives the same search budget. Galleries use the same training-only selection rule. All methods retain their full candidate history. New lighting and motion are reserved for the second animation.</p>']
    for arm in ['archive','portfolio','random']:
        parts.append(f'<h2>{arm}</h2><p>Training illumination</p><img src="{arm}_spots.gif"><p>Held-out illumination and orthogonal motion</p><img src="{arm}_holdout.gif">')
    parts.append('<p>Fixed six-parameter convex family, synthetic n=1.60, scalar Fresnel approximation. No fabrication or beauty claim. Volumes differ. Descriptor coverage is sensitive to rendering resolution and grid definitions.</p><a href="RESULTS.md">Measured comparison and limitations</a> | <a href="report.json">Full report</a>')
    (out/'index.html').write_text('\n'.join(parts))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--calls',type=int,default=192); p.add_argument('--seeds',type=int,nargs='+',default=[201,202,203]); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    if a.calls<96 or a.calls%96: p.error('--calls must be a positive multiple of 96')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=a.resume); start=time.time(); runs=[]
    config=dict(calls=a.calls,seeds=a.seeds,bins=BINS,widths=WIDTHS)
    config=json.loads(json.dumps(config))
    config_path=out/'config.json'
    if a.resume and (not config_path.exists() or json.loads(config_path.read_text())!=config):
        raise ValueError('Resume requires matching experiment configuration')
    config_path.write_text(json.dumps(config,indent=2))
    for seed in a.seeds:
        for arm in ['random','portfolio','archive']:
            checkpoint=out/f'{arm}_{seed}_search.json'
            run=json.loads(checkpoint.read_text()) if a.resume and checkpoint.exists() else run_arm(arm,seed,a.calls)
            runs.append(run)
            checkpoint.write_text(json.dumps(run,indent=2))
            print(arm,seed,'search cells',len(run['archive']),flush=True)
    for run in runs:
        audit(run)
        (out/f"{run['arm']}_{run['seed']}_audit.json").write_text(json.dumps(run,indent=2))
        print(run['arm'],run['seed'],run['summary'],flush=True)
    galleries={arm:select_gallery([r for run in runs if run['arm']==arm for r in run['archive']]) for arm in ['random','portfolio','archive']}
    for rows in galleries.values():
        for r in rows: r['shape']=shape_audit(r['genes'])
    report=dict(settings=vars(a),bins=BINS,widths=WIDTHS,runs=runs,galleries=galleries,scope='Archive-guided CMA ranking and elite restarts versus equal-budget style portfolio and random sampling. No refinement in any arm. All coarse elites re-evaluated; audit costs reported separately.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# Archive-driven jewellery comparison','','| Method | Seed | Search calls | Coarse cells | Dense training cells | Held-out cells | Audited elites |','|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        s=r['summary']; lines.append(f"| {r['arm']} | {r['seed']} | {a.calls} | {s['coarse_cells']} | {s['dense_cells']} | {s['heldout_cells']} | {s['audited_candidates']} |")
    lines+=['','Cells use fixed bright-region-area and motion-activity bins, with common brightness, lower-tail brightness and unresolved-weight gates. Dense and held-out coverage rebuild the archive after evaluation, allowing cell migration/collisions. Each baseline retains all candidates for its archive.','', 'Search budgets are equal. Audit budgets vary because every coarse elite is audited; counts are above. Coverage is only among those retained elites, not a dense re-evaluation of every search candidate. The 48-pixel/15-frame audit is a fidelity increase, not a convergence proof.','', 'Gallery selection uses dense training descriptors only, pooled over seeds. Held-out images do not influence selection. The same greedy descriptor-spread rule applies to all methods. No local refinement is included, avoiding extra adaptive budgets.','', 'These are small-sample algorithmic observations, not statistical significance, a universal beauty measure, arbitrary-facet discovery, or manufacturing qualification. Physical limitations of the first pilot still apply.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    media(out,galleries); report['seconds']=time.time()-start
    (out/'report.json').write_text(json.dumps(report,indent=2)); print('finished',out,'seconds',report['seconds'],flush=True)

if __name__=='__main__': main()
