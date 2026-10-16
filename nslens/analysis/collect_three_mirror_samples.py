"""Collect saved 2D three-mirror samples with provenance and a local PNG gallery."""
import json, hashlib, csv, shutil, argparse, html
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

DEFAULT = dict(focal=50., aperture=27.78, half_field_deg=4.)
PREFIXES = ('mirror_count','mirror_spec','mirror_residual','optical_potential','potential_', 'hard_optical','hard_confirmation','shared_freeform','online_freeform','final_freeform','targeted_insertion','folded_insertion')
LEGACY = ('three_mirror_pilot','optical_mutation','optical_target_repair','refinement_sampling')

def render(args):
    path, target = args
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nslens.analysis.variable_mirror import trace
    from nslens.analysis.final_freeform import trace as fftrace
    from nslens.three_mirror.pilot import trace as oldtrace
    r=json.loads(Path(path).read_text()); task=r['task']; g=r['genes']
    try:
        if r['model']=='legacy_three_mirror': result=oldtrace(g,fields=3,pupils=7,paths=True)
        else: result=fftrace(g,3,r['freeform_coefficients_mm'],fields=3,pupils=7,paths=True,**task)
        _,m,shapes,paths,good=result
        fig,ax=plt.subplots(figsize=(7,5))
        for i,p in enumerate(paths):
            p=np.asarray(p);ax.plot(p[:,0],p[:,1],color=['#2684cf','#22a06b','#d95f02'][i//7] if good[i] else '#aaa',alpha=.55,lw=.65)
        for i,s in enumerate(shapes):
            ax.plot(s[:,0],s[:,1],color='#172b4d',lw=2);ax.annotate(f'M{i+1}',m['vertices'][i],fontsize=8)
        h=task['focal']*np.tan(np.deg2rad(task['half_field_deg']))+1
        sensor=np.array([m['image']-h*m['image_tangent'],m['image']+h*m['image_tangent']]);ax.plot(sensor[:,0],sensor[:,1],color='#9b36b7',lw=2)
        ax.set(aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title=f"{r['id']} | {r['model']}\nFocal {task['focal']:g} mm · aperture {task['aperture']:g} mm · ±{task['half_field_deg']:g}°")
        ax.grid(alpha=.15);fig.text(.5,.015,'Saved sample; 3 fields × 7 rays for display only. Gray rays fail routing.',ha='center',fontsize=7)
        fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(target,dpi=100);plt.close(fig)
        return None
    except Exception as exc:return dict(id=r['id'],error=repr(exc))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for s in ['samples','pngs','source_records','gallery']: (out/s).mkdir()
    samples={}; scanned=[]; errors=[]; occurrences=0
    def walk(x,ctx,ptr,source,legacy):
        nonlocal occurrences
        if isinstance(x,list):
            for i,v in enumerate(x):walk(v,ctx,ptr+'/'+str(i),source,legacy)
        elif isinstance(x,dict):
            ctx=ctx.copy()
            for name in ('job','chosen_job'):
                j=x.get(name)
                if isinstance(j,dict):
                    if isinstance(j.get('task'),dict):ctx['task']=j['task']
                    if 'backend' in j:ctx['backend']=j['backend']
            if isinstance(x.get('task'),dict):ctx['task']=x['task']
            if 'backend' in x:ctx['backend']=x['backend']
            for k,g in x.items():
                if 'genes' not in k or not isinstance(g,list) or not all(isinstance(v,(int,float)) and np.isfinite(v) for v in g):continue
                if legacy:
                    if len(g)!=12 or ctx.get('backend')=='tir' or '/tir' in str(source):continue
                    model='legacy_three_mirror';task=DEFAULT.copy()
                else:
                    if len(g)!=9:continue
                    model='variable_mirror_n3';task={**DEFAULT,**{k:v for k,v in ctx.get('task',{}).items() if k in DEFAULT}}
                coeff=x.get('freeform_coefficients_mm',[0.]*4) if k=='genes' else [0.]*4
                identity=dict(model=model,genes=g,task=task,freeform_coefficients_mm=coeff)
                key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                row=samples.setdefault(key,{**identity,'id':'three_'+key[:16],'occurrences':[]})
                row['occurrences'].append(dict(source=str(source),pointer=ptr+'/'+k,saved_metrics={k:v for k,v in x.items() if k in ('loss','success','worst_mapping_rms_mm','worst_support','crossing')}));occurrences+=1
            for k,v in x.items():walk(v,ctx,ptr+'/'+k.replace('~','~0').replace('/','~1'),source,legacy)
    for d in sorted(Path('experiments').iterdir()):
        if not d.is_dir() or d==out:continue
        legacy=d.name.startswith(LEGACY)
        if not legacy and not d.name.startswith(PREFIXES):continue
        for f in sorted(d.rglob('*.json')):
            try:
                before=occurrences;walk(json.loads(f.read_text()),{},'',f,legacy);scanned.append(str(f))
                if occurrences>before:
                    target=out/'source_records'/f.relative_to('experiments');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
            except Exception as exc:errors.append(dict(source=str(f),error=repr(exc)))
    rows=sorted(samples.values(),key=lambda r:r['id']);jobs=[]
    for r in rows:
        target=out/'samples'/(r['id']+'.json');target.write_text(json.dumps(r,indent=2)+'\n');jobs.append((str(target),str(out/'pngs'/(r['id']+'.png'))))
    print(f'Collected {len(rows)} unique samples from {occurrences} occurrences; rendering.',flush=True)
    render_errors=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(render,jobs,chunksize=8),1):
            if result:render_errors.append(result)
            if i%200==0:print(f'{i}/{len(rows)} rendered',flush=True)
    with (out/'index.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['id','model','focal_mm','aperture_mm','half_field_deg','freeform','occurrences','json','png','first_source'])
        for r in rows:w.writerow([r['id'],r['model'],*r['task'].values(),any(r['freeform_coefficients_mm']),len(r['occurrences']),'samples/'+r['id']+'.json','pngs/'+r['id']+'.png',r['occurrences'][0]['source']])
    pages=[]
    for start in range(0,len(rows),100):
        name=f'page_{start//100+1:03}.html';pages.append(name)
        content='<meta charset="utf-8"><style>body{font-family:sans-serif}main{display:grid;grid-template-columns:repeat(3,1fr)}img{width:100%}figure{margin:8px}small{overflow-wrap:anywhere}</style><a href="../index.html">All pages</a><main>'
        for r in rows[start:start+100]:
            stem=r['id'];content+=f'<figure><a href="../pngs/{stem}.png"><img loading="lazy" src="../pngs/{stem}.png"></a><a href="../samples/{stem}.json">{stem}</a><br><small>{html.escape(r["occurrences"][0]["source"])}</small></figure>'
        (out/'gallery'/name).write_text(content+'</main>')
    (out/'index.html').write_text('<meta charset="utf-8"><h1>Stored three-mirror designs</h1><p>'+str(len(rows))+' unique geometry/specification records; includes failed and intermediate samples.</p>'+''.join(f'<p><a href="gallery/{name}">Samples {i*100+1}–{min((i+1)*100,len(rows))}</a></p>' for i,name in enumerate(pages)))
    report=dict(unique_samples=len(rows),stored_occurrences=occurrences,scanned_files=len(scanned),models={m:sum(r['model']==m for r in rows) for m in {r['model'] for r in rows}},read_errors=errors,render_errors=render_errors,scanned_sources=scanned)
    (out/'collection_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'README.txt').write_text('Open index.html for the paginated PNG gallery. samples/ contains complete vectors and every source JSON pointer; source_records/ preserves contributing originals. index.csv is searchable. Exact model + genes + freeform coefficients + specification duplicates are merged, preserving all occurrences. Includes failures, smoke runs, intermediate candidates, and refined designs; no quality filter. Covers saved JSON vectors from the three-mirror pilot and subsequent optical mutation, variable-count, potential, and freeform experiment families. It cannot recover unsaved optimizer evaluations. Variable-count records with 9 genes are n=3; legacy 12-gene records have a separate decoder and variable mirror widths. TIR and unrelated spectrometer/camera models are excluded. PNGs use sparse display rays, not new quality audits. Defaults for records without explicit task: focal50mm, aperture27.78mm, half-field4deg.\n')
    shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps({k:v for k,v in report.items() if k!='scanned_sources'}),flush=True)
if __name__=='__main__':main()
