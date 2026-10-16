"""Forty-second, provenance-linked reconstruction of saved spectrometer evolution."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from . import powered_four_mirror as pm, four_mirror as fm, model
from .four_mirror_render import load_design

INK='#172635'; MUTED='#64748b'; BLUE='#77a8be'; GOLD='#d8ad52'; PURPLE='#ad84ba'
COLORS=['#2475c2','#2b9a63','#d6534d']
ROOT=Path('experiments/spectrometer_powered_four_mirror_cma_v3')


def reconstruct():
    seed=ROOT/'seed_603'
    initial=json.loads((seed/'inherited_seed.json').read_text())
    best=initial;records=[]
    for line in (seed/'populations.jsonl').read_text().splitlines():
        row=json.loads(line)
        for g,metric in zip(row['genes'],row['metrics']):
            if metric['loss']<best['metrics']['loss']:
                genes=np.pad(g,(0,pm.N_TOTAL-len(g))).tolist()
                best=dict(genes=genes,metrics=metric)
        records.append(dict(genes=best['genes'],metrics=best['metrics'],evaluations=row['evaluations']-1,stage=row['stage']))
    final=json.loads((seed/'result.json').read_text())
    if not np.allclose(records[-1]['genes'],final['best']['genes'],atol=1e-14,rtol=0):
        raise RuntimeError('Reconstructed incumbent does not match saved final result')
    return initial,records


def surface_mesh(s):
    radius=np.linspace(0,s['aperture'],10);theta=np.linspace(0,2*np.pi,40)
    x=radius[:,None]*np.cos(theta);y=radius[:,None]*np.sin(theta)
    z=model.sag(x,y,s['coeff'])[0]
    return np.stack([x,y,z],axis=-1)@s['basis'].T+s['vertex']


def scene(layout,trace):
    meshes=[surface_mesh(s) for s in layout['surfaces']]
    bounds=layout['detector_bounds'];uv=np.array([[bounds[0,0],bounds[0,1]],[bounds[1,0],bounds[0,1]],[bounds[1,0],bounds[1,1]],[bounds[0,0],bounds[1,1]]])
    detector=layout['image']+uv[:,0,None]*layout['image_x']+uv[:,1,None]*layout['image_y']
    slit=layout['origin']+np.array([[0,-5,0],[0,5,0]])
    cloud=np.concatenate([p.reshape(-1,3) for p in meshes]+[detector,slit])
    center=(cloud.min(0)+cloud.max(0))/2
    rays=[];nf,nw,npu=trace['good'].shape
    for fi in [0,nf//2,nf-1]:
        for wi,c in zip([0,nw//2,nw-1],COLORS):
            for pi in [npu//2,0,npu-1]:
                idx=(fi*nw+wi)*npu+pi
                rays.append((trace['paths'][:,idx]-center,c))
    return dict(meshes=[p-center for p in meshes],detector=detector-center,slit=slit-center,rays=rays,
                centers=[s['vertex']-center for s in layout['surfaces']])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(ROOT/'visualizations/evolution'))
    parser.add_argument('--fps',type=int,default=24);parser.add_argument('--duration',type=int,default=40)
    parser.add_argument('--width',type=int,default=1920);parser.add_argument('--height',type=int,default=1080)
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--concise',action='store_true',help='Omit the extra commentary block and technical footer')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    initial,records=reconstruct();audit=json.loads((ROOT/'independent_audit.json').read_text())
    report=json.loads((ROOT/'report_metrics.json').read_text())
    original=fm.load_baseline(fields=3,waves=3,pupils=5)
    base=dict(original['model'],detector_bounds=np.array([[-4.2,-6.2],[4.2,6.2]]))
    _,probe_base,probe_layout=load_design('experiments/spectrometer_four_mirror_v1/result.json')
    scenes={'original':scene(base,original['trace']),
            'rejected':scene(probe_layout,fm.trace(probe_layout,probe_base,fields=3,waves=3,pupils=5)),
            'initial':scene(pm.decode(initial['genes']),pm.trace(initial['genes'],fields=3,waves=3,pupils=5))}
    # Only recorded incumbents are displayed. No parameter interpolation.
    for index,row in enumerate(records):
        if index==0 or row['genes']!=records[index-1]['genes']:
            scenes[index]=scene(pm.decode(row['genes']),pm.trace(row['genes'],fields=3,waves=3,pupils=5))
        else:scenes[index]=scenes[index-1]
    manifest=dict(duration_seconds=args.duration,fps=args.fps,resolution=[args.width,args.height],winning_seed=603,
                  chapters=[dict(seconds=[0,4],label='Original two-mirror baseline'),dict(seconds=[4,7],label='Rejected compact geometry probe'),dict(seconds=[7,10],label='Feasible folded starting scaffold'),dict(seconds=[10,32],label='Actual winning CMA run: 1200 coarse + 2400 fine evaluations'),dict(seconds=[32,40],label='Final validated four-powered-mirror design')],
                  source_population_history=str(ROOT/'seed_603/populations.jsonl'),source_result=str(ROOT/'result.json'),
                  source_audit=str(ROOT/'independent_audit.json'),reconstruction='Strict incumbent replay, verified against saved final genes. Camera orbit and cuts are presentation; optical parameters are never interpolated. Earlier failed CMA pilots condensed out; this is not a screen recording.',
                  records=records)
    (out/'timeline.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'axes.labelcolor':MUTED,'figure.facecolor':'white'})
    fig=plt.figure(figsize=(args.width/120,args.height/120),dpi=120,facecolor='white')
    ax=fig.add_axes([.005,.145,.625,.69],projection='3d')
    fig.text(.05,.936,'FOUR MIRRORS + A CURVED GRATING',fontsize=25,weight='bold',color=INK)
    subtitle=fig.text(.05,.889,'',fontsize=17,color=MUTED)
    tag=fig.text(.66,.79,'',fontsize=13,weight='bold',color=INK)
    headline=fig.text(.66,.705,'',fontsize=34,weight='bold',color=INK)
    details=fig.text(.66,.59,'',fontsize=13,linespacing=1.6,color=MUTED)
    description=fig.text(.66,.425,'',fontsize=12,linespacing=1.5,color=INK)
    chart=fig.add_axes([.665,.185,.29,.17]);xs=np.array([r['evaluations'] for r in records]);ys=np.array([r['metrics']['loss'] for r in records])
    chart.set_xlim(0,3600);chart.set_ylim(ys.min()-.00012,initial['metrics']['loss']+.00012)
    chart.axvspan(0,1200,color=BLUE,alpha=.12);chart.axvspan(1200,3600,color=GOLD,alpha=.12)
    chart.axvline(1200,color=MUTED,alpha=.3,lw=.8)
    chart.plot(xs,ys,color='#e2e8f0',lw=2);curve,=chart.plot([],[],color='#2475c2',lw=2.5)
    dot,=chart.plot([],[],'o',color='#2475c2',ms=5)
    chart.set_xlabel('Population evaluations',fontsize=10);chart.set_ylabel('Training objective',fontsize=10)
    chart.tick_params(labelsize=9);chart.spines[['top','right']].set_visible(False);chart.grid(alpha=.12)
    chart.text(600,1.04,'COARSE',transform=chart.get_xaxis_transform(),ha='center',fontsize=9,color=MUTED)
    chart.text(2400,1.04,'FINE',transform=chart.get_xaxis_transform(),ha='center',fontsize=9,color=MUTED)
    footer=fig.text(.05,.055,'',fontsize=10,color=MUTED)
    clock=fig.text(.955,.055,'',fontsize=11,color=MUTED,ha='right')
    progress=fig.add_axes([.05,.1,.9,.008]);progress.set_xlim(0,40);progress.set_ylim(0,1);progress.axis('off')
    progress.axhspan(0,1,color='#edf1f5');bar=progress.barh(.5,0,height=1,color=BLUE)[0]
    for x,c,w in zip([.08,.20,.32],COLORS,[500,800,1100]):
        fig.text(x,.135,'━━',color=c,fontsize=15);fig.text(x+.026,.136,f'{w} nm',fontsize=10,color=MUTED)
    lastkey=None; annotations=[]
    def frame(t):
        nonlocal lastkey, annotations
        if t<4:
            key='original';idx=-1;sub='01  /  Original two-mirror spectrometer';status='STARTING REFERENCE';big='83.0 cm³'
            desc='10 mm slit  ·  F/3.8\n500–1100 nm band\nTwo powered mirrors + grating'
            note='Matched optical-envelope reference.\nMounts and substrates excluded.'
        elif t<7:
            key='rejected';idx=-1;sub='02  /  Compactness alone is not enough';status='GEOMETRY PROBE  ·  REJECTED';big='44.8 cm³'
            desc='Looks compact — but clips rays.\n17 of 17,352 audit rays blocked.\nNot carried forward as a valid design.'
            note='A real saved feasibility trial.\nIndependent ray checks caught the failure.'
        elif t<10:
            key='initial';idx=-1;sub='03  /  A feasible starting scaffold';status='INHERITED OPTICS + TWO FOLDS';big='50.8 cm³'
            desc='Existing optical prescription retained.\nA feasible five-surface route.\nAll four mirror powers may now evolve.'
            note='Construction step before powered CMA.\nThe two added mirrors start flat.'
        else:
            idx=min(len(records)-1,int(max(0,t-10)/22*len(records)))
            key=idx;row=records[idx]
            if t<32:
                stage='Coarse geometry + curvature' if row['stage']=='base' else 'Fine freeform refinement'
                sub='04  /  '+stage;status=f'CMA SEED 603  ·  {row["evaluations"]:,} / 3,600';big=f'{row["metrics"]["volume_cm3"]:.2f} cm³'
                desc='Best recorded candidate so far.\n28 coarse → 63 fine variables.\nPositions, powers, then freeform terms.'
                note='The optical changes are subtle.\nTrace and geometry are recomputed\nfrom each saved prescription.'
            else:
                sub='05  /  Final design — independently checked';status='FOUR POWERED MIRRORS  ·  VALIDATED';big='39.9% smaller'
                desc='49.9 cm³ matched optical envelope.\n39,117 / 39,117 dense rays received.\n3,395 independent-root rays received.'
                note='Added mirror powers remain weak.\nGeometric RMS blur is ~6% worse.\nThe gain is compactness, not sharper imaging.'
        if key!=lastkey:
            ax.clear();s=scenes[key];annotations=[]
            gi=1 if len(s['meshes'])==3 else 2
            for j,p in enumerate(s['meshes']):
                ax.plot_surface(*np.moveaxis(p,-1,0),color=GOLD if j==gi else BLUE,alpha=.9,linewidth=0,shade=True,rcount=10,ccount=40)
                ax.plot(*p[-1].T,color='#334155',lw=.6)
                label=('G' if j==gi else (['M1','G','M2'][j] if gi==1 else ['M1','M2','G','M3','M4'][j]))
                positions=([(.23,.79),(.71,.39),(.78,.74)] if gi==1 else [(.77,.66),(.20,.78),(.80,.41),(.14,.34),(.67,.15)])
                annotation=ax.annotate(label,xy=(0,0),xycoords='data',xytext=positions[j],textcoords='axes fraction',fontsize=11,color=INK,weight='bold',bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.85),arrowprops=dict(arrowstyle='-',color=MUTED,lw=.8))
                annotations.append((annotation,s['centers'][j]))
            ax.add_collection3d(Poly3DCollection([s['detector']],facecolor=PURPLE,edgecolor='#75458a',alpha=.6))
            ax.plot(*s['slit'].T,color=INK,lw=2.5)
            for p,c in s['rays']:ax.plot(*p.T,color=c,alpha=.35,lw=.8)
            ax.set(xlim=(-38,38),ylim=(-23,23),zlim=(-38,38),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
            ax.set_box_aspect((76,46,76));ax.tick_params(labelsize=8)
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
            ax.grid(alpha=.12);lastkey=key
        # Continuous camera motion only: no invented intermediate optical states.
        ax.view_init(elev=22+3*np.sin(t/40*np.pi),azim=-72+24*t/40)
        for annotation,point in annotations:
            xp,yp,_=proj3d.proj_transform(*point,ax.get_proj());annotation.xy=(xp,yp)
        subtitle.set_text(sub);tag.set_text(status);tag.set_color('#a75c22' if key=='rejected' else INK)
        headline.set_text(big);details.set_text(desc);description.set_text('' if args.concise else note)
        if idx>=0:curve.set_data(xs[:idx+1],ys[:idx+1]);dot.set_data([xs[idx]],[ys[idx]])
        else:curve.set_data([],[]);dot.set_data([],[])
        chart.set_visible(t>=10)
        footer.set_text('' if args.concise else 'Saved-checkpoint replay  ·  No interpolated designs  ·  Equal scale, recentered views  ·  Optical surfaces only')
        clock.set_text(f'{min(int(t),39):02d} / 40 s');bar.set_width(t)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
    if args.preview_only:
        for t in [1,5,8,16,27,36]:
            frame(t);fig.savefig(out/f'preview_{t:02d}.jpg',dpi=120)
        return
    movie=out/'spectrometer_evolution_40s.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s',f'{args.width}x{args.height}','-pix_fmt','rgb24','-r',str(args.fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
        for i in range(args.duration*args.fps):
            rgb=frame(i/args.fps);encoder.stdin.write(rgb.tobytes())
            if i==args.fps*36:fig.savefig(out/'poster.jpg',dpi=120)
            if i%(args.fps*5)==0:print(f'rendered {i/args.fps:.0f}/{args.duration}s',flush=True)
        encoder.stdin.close();code=encoder.wait()
        if code:raise RuntimeError(f'ffmpeg failed: {code}')
    plt.close(fig)
    (out/'watch.html').write_text('<!doctype html><meta charset="utf-8"><title>Spectrometer evolution</title><style>body{font:17px system-ui;background:#f6f8fa;color:#172635;max-width:1200px;margin:40px auto;padding:20px}video{width:100%;border-radius:12px}a{color:#2475c2}</style><h1>Four-mirror spectrometer · 40-second evolution</h1><video controls poster="poster.jpg" src="spectrometer_evolution_40s.mp4"></video><p>A reconstruction of saved designs and the actual winning CMA history. Optical prescriptions are not interpolated; camera motion is illustrative. Earlier unsuccessful CMA pilots are condensed out.</p><p><a href="spectrometer_evolution_40s.mp4" download>Download MP4</a> · <a href="timeline.json">Timeline and source history</a></p>')
    print(movie,flush=True)

if __name__=='__main__':main()
