"""CMA discovery of three parabolic mirror cross-sections with nearest-hit tracing."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from nslens.search.cma_me import CMAES

FOCAL = 50.
APERTURE = 27.78

def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def decode(g):
    """Chief-ray scaffold only; every pupil ray is traced non-sequentially."""
    g = np.asarray(g, dtype=float)
    angles = np.array([-150., 180., -180.]) + np.array([28., 58., 58.])*g[:3]
    lengths = 65. + 40.*g[3:6]
    curves = .025*g[6:9]
    widths = 30. + 16.*g[9:12]
    vertices=[]; normals=[]; tangents=[]
    p=np.zeros(2); d=np.array([1.,0.])
    for angle,length in zip(angles,lengths):
        a=np.deg2rad(angle)
        out=np.array([np.cos(a)*d[0]-np.sin(a)*d[1],np.sin(a)*d[0]+np.cos(a)*d[1]])
        n=unit(d-out); t=np.array([-n[1],n[0]])
        vertices.append(p.copy()); normals.append(n); tangents.append(t)
        p=p+length*out; d=out
    return dict(vertices=np.array(vertices),normals=np.array(normals),tangents=np.array(tangents),
                curves=curves,widths=widths,image=p,image_normal=d,
                image_tangent=np.array([-d[1],d[0]]),angles=angles,lengths=lengths)


def intersections(o,d,m):
    """Exact forward intersections with finite parabolas; stable quadratic roots."""
    delta=o[:,None,:]-m['vertices'][None,:,:]
    u=np.einsum('rmi,mi->rm',delta,m['tangents'])
    v=np.einsum('rmi,mi->rm',delta,m['normals'])
    du=d@m['tangents'].T; dv=d@m['normals'].T
    a=m['curves'][None,:]*du**2
    b=2*m['curves'][None,:]*u*du-dv
    c=m['curves'][None,:]*u**2-v
    disc=b*b-4*a*c
    q=-.5*(b+np.where(b>=0,1.,-1.)*np.sqrt(np.maximum(0.,disc)))
    with np.errstate(divide='ignore',invalid='ignore'):
        r1=q/a; r2=c/q
        linear=-c/b
    small=abs(a)<1e-14
    r1=np.where(small,linear,r1); r2=np.where(small,np.inf,r2)
    roots=np.stack([r1,r2],axis=-1)
    with np.errstate(invalid='ignore'):
        local=u[:,:,None]+roots*du[:,:,None]
    valid=np.isfinite(roots)&(roots>1e-7)&(abs(local)<=m['widths'][None,:,None])
    valid &= ((disc>=0)|small)[:,:,None]
    return np.min(np.where(valid,roots,np.inf),axis=-1)


def trace(g,fields=3,pupils=9,paths=False):
    m=decode(g)
    field=np.repeat(np.linspace(-4.,4.,fields),pupils)
    theta=np.deg2rad(field)
    pupil=np.tile(np.linspace(-APERTURE/2,APERTURE/2,pupils),fields)
    samples=np.linspace(-1,1,65)
    outlines=[v+samples[:,None]*w*t+(a*(samples*w)**2)[:,None]*n
              for v,t,n,a,w in zip(m['vertices'],m['tangents'],m['normals'],m['curves'],m['widths'])]
    x0=min(np.min(p[:,0]) for p in outlines)-20.
    o=np.column_stack([np.full(len(field),x0),pupil+x0*np.tan(theta)])
    d=np.column_stack([np.cos(theta),np.sin(theta)])
    active=np.ones(len(field),bool); hits=np.zeros(len(field),int)
    route=np.full((len(field),6),-1,int); image=np.full(len(field),np.nan)
    histories=[[p.copy()] for p in o] if paths else None
    for step in range(6):
        distances=intersections(o,d,m)
        which=np.argmin(distances,axis=1); tm=distances[np.arange(len(o)),which]
        den=d@m['image_normal']
        with np.errstate(divide='ignore',invalid='ignore'):
            ti=((m['image']-o)@m['image_normal'])/den
        sensor=active&(ti>1e-7)&(ti<tm)&(den>0)
        if np.any(sensor):
            p=o[sensor]+ti[sensor,None]*d[sensor]
            image[sensor]=(p-m['image'])@m['image_tangent']
            if paths:
                for idx,point in zip(np.flatnonzero(sensor),p): histories[idx].append(point.copy())
        reflect=active&~sensor&np.isfinite(tm)
        active=reflect
        if not np.any(reflect): break
        ids=np.flatnonzero(reflect); j=which[ids]
        p=o[ids]+tm[ids,None]*d[ids]
        u=np.sum((p-m['vertices'][j])*m['tangents'][j],axis=1)
        n=unit(m['normals'][j]-2*(m['curves'][j]*u)[:,None]*m['tangents'][j])
        d[ids]-=2*np.sum(d[ids]*n,axis=1)[:,None]*n
        o[ids]=p; route[ids,step]=j; hits[ids]+=1
        if paths:
            for idx,point in zip(ids,p): histories[idx].append(point.copy())
    good=np.isfinite(image)&(hits==3)&np.all(np.sort(route[:,:3],axis=1)==np.arange(3),axis=1)
    # All entrance rays remain in the denominator; lost rays cannot improve quality.
    target=FOCAL*np.tan(theta)
    errors=[np.where(good,np.minimum(abs(image-sign*target),100.),100.) for sign in (-1,1)]
    parity=(-1,1)[int(np.argmin([np.mean(e*e) for e in errors]))]
    err=errors[0 if parity==-1 else 1]
    support=good.reshape(fields,pupils).mean(axis=1)
    rmse=np.sqrt(np.mean(err.reshape(fields,pupils)**2,axis=1))
    # Continuous imaging score once rays route; coverage dominates large misses.
    progress=np.sum(np.cumprod(route[:,:3]==np.arange(3),axis=1),axis=1)
    loss=float(np.mean(err**2)+10000*np.mean(~good)+1000*np.mean(3-progress))
    rows=[]
    for f in range(fields):
        mask=(np.arange(len(field))//pupils==f)&good
        values=image[mask]
        rows.append(dict(field_deg=float(field[f*pupils]),support=float(support[f]),
            rms_spot_mm=float(np.std(values)) if len(values) else None,
            centroid_error_mm=float(abs(np.mean(values)-parity*target[f*pupils])) if len(values) else None))
    result=dict(loss=loss,worst_support=float(min(support)),target_rmse_with_loss_penalty_mm=float(max(rmse)),
                parity=parity,fields=rows,genes=np.asarray(g).tolist())
    if paths: return result,m,outlines,histories,good
    return result


def cell(g):
    m=decode(g)
    signs=tuple((m['angles']-np.array([-180,180,-180])>0).astype(int))
    # Layout category plus relative spacing; optical power signs are also retained.
    return signs+tuple((m['curves']>0).astype(int))+(int(np.clip(m['lengths'][0]/m['lengths'][1],0,3)*2),)


def run(out,budget,seeds):
    out.mkdir(parents=True,exist_ok=False)
    archive={}; logs=[]; start=time.monotonic()
    for seed in range(seeds):
        rng=np.random.default_rng(seed); best=None; emitter=None; stale=0
        for generation in range(budget//24):
            if emitter is None or stale>=35 or emitter.restart_needed:
                elites=list(archive.values())
                mean=np.array(elites[rng.integers(len(elites))]['genes']) if elites and rng.random()<.7 else rng.uniform(-.8,.8,12)
                emitter=CMAES(mean,.22,popsize=24,seed=int(rng.integers(2**30)),bounds=(-np.ones(12),np.ones(12)),boundary_handling='reflect')
                stale=0
            pop=emitter.ask(); rows=[trace(g) for g in pop]
            order=np.argsort([r['loss'] for r in rows]); emitter.tell(order)
            improved=False
            for row in rows:
                key=cell(row['genes'])
                if key not in archive or row['loss']<archive[key]['loss']:
                    archive[key]=row
                if best is None or row['loss']<best['loss']:
                    best=row; improved=True
            stale=0 if improved else stale+1
            if generation%25==0:
                log=dict(seed=seed,evaluations=(generation+1)*24,loss=best['loss'],support=best['worst_support'],cells=len(archive))
                logs.append(log); print(json.dumps(log),flush=True)
    selected=sorted(archive.values(),key=lambda r:r['loss'])[:12]
    audited=[trace(r['genes'],fields=11,pupils=101) for r in selected]
    audited.sort(key=lambda r:r['loss'])
    summary=dict(caveat='2D parabolic cross-sections; ideal mirrors; geometric imaging only; no 3D or diffraction claim.',
        focal_length_mm=FOCAL,entrance_aperture_mm=APERTURE,full_field_deg=8,
        evaluations=seeds*(budget//24)*24,seeds=seeds,wall_seconds=time.monotonic()-start,
        archive_cells=len(archive),audit_fields=11,audit_pupil_rays=101,selected=audited,progress=logs)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'archive.json').write_text(json.dumps(list(archive.values()),indent=2,allow_nan=False)+'\n')
    render(out,audited)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('selected','progress')}),flush=True)


def render(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit_path=out/'geometry_audit.json'
    audits=json.loads(audit_path.read_text()) if audit_path.exists() else []
    fig,axs=plt.subplots(3,4,figsize=(19,12),squeeze=False)
    for ax,row in zip(axs.flat,rows):
        result,m,outlines,histories,good=trace(row['genes'],fields=3,pupils=7,paths=True)
        for i,p in enumerate(outlines):
            ax.plot(*p.T,color='black',lw=2)
            ax.text(*m['vertices'][i],f'M{i+1}',fontsize=8)
        extent=FOCAL*np.tan(np.deg2rad(4.))*1.3
        detector=m['image']+np.array([-extent,extent])[:,None]*m['image_tangent']
        ax.plot(*detector.T,color='red',lw=3)
        for idx,path in enumerate(histories):
            p=np.array(path)
            ax.plot(*p.T,color=['#4477aa','#228833','#cc6677'][idx//7],alpha=.45,lw=.7)
        ax.set_aspect('equal'); ax.grid(alpha=.15)
        index=list(axs.flat).index(ax)
        checked=audits[index] if audits else None
        measured=checked['finer_audit'] if checked else row
        warning=' — MIRRORS CROSS' if checked and checked['mirror_crossings_sampled'] else ''
        ax.set_title(f"#{index+1}: support {measured['worst_support']:.1%}; loss {measured['loss']:.3g}{warning}",fontsize=9, color='red' if warning else 'black')
        ax.set_xlabel('mm'); ax.set_ylabel('mm')
    for ax in list(axs.flat)[len(rows):]: ax.set_visible(False)
    fig.suptitle('CMA three-mirror exploration — 2D parabolic cross-sections, not 3D validated designs')
    fig.tight_layout(); fig.savefig(out/'gallery.png',dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--budget',type=int,default=12000)
    p.add_argument('--seeds',type=int,default=3)
    a=p.parse_args()
    if a.budget<24 or a.seeds<1: p.error('budget must be >=24; seeds >=1')
    run(a.out,a.budget,a.seeds)

if __name__=='__main__': main()
