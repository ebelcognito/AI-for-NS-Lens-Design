"""Forty-second replay of saved geometry/CMA candidates; camera motion only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from . import powered_five_mirror as pm, four_mirror as fm
from .five_mirror_render import draw_hardware3d,hardware_scene,surface_names
from .four_mirror_search import package_cloud
from .render_evolution_video import surface_mesh
from .search import serial,write_json
from scipy.spatial.transform import Rotation

INK='#172635';BLUE='#77a8be';GOLD='#d8ad52';PURPLE='#ad84ba';COLORS=['#2475c2','#2b9a63','#d6534d']


def replay_records(source):
    source=Path(source);out=source.parent;data=json.loads(source.read_text());config=data['configuration']
    geometry=[];best_volume=np.inf;population=out/'geometry_population.jsonl'
    if population.exists():
        for line in population.open():
            r=json.loads(line)
            if not(r.get('hardware',{}).get('passed') and r.get('screen',{}).get('valid')):continue
            if r['volume_cm3']>=best_volume:continue
            best_volume=r['volume_cm3'];g=pm.seed_genes(np.asarray(r['parent_genes']),r['params'])
            geometry.append(dict(stage='Geometry',counter=r['evaluation'],genes=g,segment=r['segment'],aperture=r['aperture'],source=str(population)))
    seed=data['best_seed'];seed_dir=out/f'seed_{seed}'
    initial=json.loads((seed_dir/'inherited_seed.json').read_text())
    scaffold=np.asarray(config['initial_genes']);best=initial;cma=[]
    source_pop=seed_dir/'populations.jsonl'
    for line in source_pop.open():
        r=json.loads(line)
        for sample,metric in zip(r['genes'],r['metrics']):
            feasible=(not config.get('hardware_constraints',False)) or (metric['hardware']['passed'] and metric['throughput']==1 and metric['curvature']['sign_violation']==0)
            if feasible and metric['loss']<best['metrics']['loss']:
                genes=scaffold.copy();genes[:len(sample)]=sample
                best=dict(genes=genes,metrics=metric)
        cma.append(dict(stage='Coarse' if r['stage']=='base' else 'Fine',counter=r['evaluations']-1,
                        genes=np.asarray(best['genes']).copy(),segment=config['segment'],aperture=config['aperture'],
                        loss=float(best['metrics']['loss']),source=str(source_pop)))
    raw=json.loads((seed_dir/'result.json').read_text())
    if not np.allclose(cma[-1]['genes'],raw['best']['genes'],rtol=0,atol=1e-14):
        raise RuntimeError('Population replay differs from saved raw seed incumbent')
    final=dict(stage='Final',counter=data['evaluations'],genes=np.asarray(data['genes']),segment=config['segment'],aperture=config['aperture'],source=str(source))
    if not geometry:
        geometry=[dict(final,stage='Initial',genes=np.asarray(initial['genes']),counter=0)]
    return data,geometry,cma,final


def render_video(source,out=None,duration=40,fps=24,width=1920,height=1080,preview_only=False,
                 zoom=1.0,show_fine_loss=False,movie_name='spectrometer_evolution_40s.mp4'):
    source=Path(source);out=Path(out) if out else source.parent/'visualizations/evolution';out.mkdir(parents=True,exist_ok=True)
    data,geometry,cma,final=replay_records(source);hardware=bool(data['configuration'].get('hardware_constraints',False))
    audit_path=source.parent/'independent_audit.json';audit=json.loads(audit_path.read_text())
    if not(audit.get('passed') and np.array_equal(audit.get('genes'),data['genes'])):
        raise ValueError('Automatic video requires a passing audit for the exact saved final genes')
    # Replay only actual states. Repeated incumbents share a precomputed scene.
    allrecords=geometry+cma+[final];scenes={};record_keys=[];bounds=[]
    for record in allrecords:
        key=hashlib.sha256(np.asarray(record['genes']).tobytes()+str((record['segment'],record['aperture'])).encode()).hexdigest()
        record_keys.append(key)
        if key in scenes:continue
        pm.configure(record['segment'],record['aperture']);trace=pm.trace(record['genes'],fields=13,waves=17,pupils=11);m=trace['model']
        meshes=[surface_mesh(s) for s in m['surfaces']];lo,hi=m['detector_bounds']
        uv=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
        detector=m['image']+uv[:,0,None]*m['image_x']+uv[:,1,None]*m['image_y'];slit=np.array([[0,-5,0],[0,5,0]])
        points=package_cloud(m,17)
        if hardware:
            h=hardware_scene(m);points=np.concatenate([points,h['corners']]);bounds.append(np.concatenate([points,*h['feed']]))
        else:bounds.append(points)
        q=points@Rotation.from_euler('xyz',m['rotation_xyz_deg'],degrees=True).as_matrix().T
        box=np.ptp(q,axis=0);volume=float(np.prod(box)/1000);metrics=fm.image_metrics(trace)
        rays=[];nf,nw,npup=trace['good'].shape
        for fi in [0,nf//2,nf-1]:
            for wi,color in zip([0,nw//2,nw-1],COLORS):
                for pi in [0,npup//2,npup-1]:
                    index=(fi*nw+wi)*npup+pi
                    rays.append((trace['paths'][:,index],color))
        scenes[key]=dict(layout=m,meshes=meshes,detector=detector,slit=slit,rays=rays,volume_cm3=volume,bbox_mm=box,rms_spot_um=metrics['rms_spot_mm']*1000)
    report_path=source.parent/'report_metrics.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        final_key=record_keys[-1]+'-final';scenes[final_key]=dict(scenes[record_keys[-1]]);record_keys[-1]=final_key
        if hardware and 'envelopes' in report:
            final_geometry=report['envelopes']['with_slit_assembly']['geometry']
            scenes[final_key].update(volume_cm3=final_geometry['volume_cm3'],bbox_mm=np.asarray(final_geometry['bbox_mm']),rms_spot_um=report['quality']['rms_spot_mm']*1000)
    cloud=np.concatenate(bounds);low=cloud.min(0)-4;high=cloud.max(0)+4
    manifest=dict(source_result=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  duration_seconds=duration,fps=fps,resolution=[width,height],sampling=dict(fields=13,waves=17,pupils_grid=11),
                  hardware_shown=hardware,visualization=dict(zoom=zoom,show_fine_loss=show_fine_loss,
                  fine_loss_source='Best feasible incumbent in the winning seed population history' if show_fine_loss else None),
                  metric_methods='All spot metrics use the same 13x17x11 Cartesian grid. Earlier envelope dimensions use saved package angles and sampled optical masks plus reserved box; Final uses the denser independently optimized rigid package envelope from report_metrics.json. Optical prescriptions are never interpolated.',chapters=[dict(seconds=[0,10],stage='Geometry'),dict(seconds=[10,34],stage='CMA'),dict(seconds=[34,40],stage='Final')],
                  reconstruction='Recorded hardware-qualified geometry incumbents and feasible winning-seed CMA incumbents. Coarse genes overwrite the saved full scaffold; inherited higher terms are preserved. Final may differ by held-out archive selection. Only camera orientation moves continuously.',
                  records=[dict(r,scene_key=k,volume_cm3=scenes[k]['volume_cm3'],bbox_mm=scenes[k]['bbox_mm'],rms_spot_um=scenes[k]['rms_spot_um']) for r,k in zip(allrecords,record_keys)])
    write_json(out/'timeline.json',manifest)
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white','font.size':11})
    fig=plt.figure(figsize=(width/120,height/120),dpi=120)
    ax=fig.add_axes([.01,.10,.73,.79],projection='3d')
    fine_start=next((i for i,r in enumerate(cma) if r['stage']=='Fine'),len(cma))
    loss_ax=None
    if show_fine_loss and fine_start<len(cma):
        fine=cma[fine_start:]
        fine_x=np.array([r['counter']-fine[0]['counter'] for r in fine])
        fine_y=np.array([r['loss'] for r in fine])
        loss_ax=fig.add_axes([.755,.145,.205,.205])
        loss_ax.set_title('FINE REFINEMENT · LOSS',loc='left',fontsize=11,weight='bold',color=INK,pad=12)
        loss_ax.set_xlabel('Fine evaluations',fontsize=9,color='#64748b')
        loss_ax.set_ylabel('Best feasible loss',fontsize=9,color='#64748b')
        loss_ax.set_xlim(0,max(1,fine_x[-1]))
        spread=max(float(np.ptp(fine_y)),1e-6)
        loss_ax.set_ylim(float(fine_y.min()-spread*.12),float(fine_y.max()+spread*.12))
        loss_ax.tick_params(labelsize=8,colors='#64748b')
        loss_ax.grid(alpha=.18)
        for spine in loss_ax.spines.values():spine.set_color('#cbd5e1')
        loss_line,=loss_ax.plot([],[],color='#2475c2',lw=2.5)
        loss_dot,=loss_ax.plot([],[],marker='o',markersize=5,color='#d8ad52')
        loss_ax.set_visible(False)
    fig.text(.045,.94,'FIVE MIRRORS + CURVED GRATING',fontsize=24,weight='bold',color=INK)
    stage_text=fig.text(.76,.77,'',fontsize=18,weight='bold',color=INK)
    counter=fig.text(.76,.72,'',fontsize=12,color='#64748b')
    envelope=fig.text(.76,.60,'',fontsize=29,weight='bold',color=INK)
    fig.text(.76,.555,'Envelope incl. slit' if hardware else 'Optical envelope',fontsize=12,color='#64748b')
    dimensions=fig.text(.76,.51,'',fontsize=11,color='#64748b')
    blur=fig.text(.76,.445,'',fontsize=27,weight='bold',color=INK)
    fig.text(.76,.40,'RMS spot',fontsize=12,color='#64748b')
    fig.text(.045,.045,'Slit plate 10 × 12 mm  ·  reserved volume 12 × 14 × 2 mm' if hardware else 'Saved optical candidates',fontsize=11,color='#64748b')
    clock=fig.text(.96,.045,'',fontsize=11,color='#64748b',ha='right')
    last=None
    def frame(t):
        nonlocal last
        phase=t/duration
        if phase<.25:index=min(len(geometry)-1,int(phase/.25*len(geometry)))
        elif phase<.85:index=len(geometry)+min(len(cma)-1,int((phase-.25)/.60*len(cma)))
        else:index=len(allrecords)-1
        record=allrecords[index];key=record_keys[index];scene=scenes[key]
        if key!=last:
            ax.clear();m=scene['layout'];_,short=surface_names(m)
            for j,p in enumerate(scene['meshes']):
                ax.plot_surface(*np.moveaxis(p,-1,0),color=GOLD if j==m['grating_index'] else BLUE,alpha=.88,linewidth=0,rcount=10,ccount=40)
                ax.plot(*p[-1].T,color='#334155',lw=.55);v=m['surfaces'][j]['vertex']
                ax.text(*v,short[j],fontsize=9,color=INK)
            ax.add_collection3d(Poly3DCollection([scene['detector']],facecolor=PURPLE,edgecolor='#75458a',alpha=.55))
            ax.plot(*scene['slit'].T,color=INK,lw=2)
            if hardware:
                draw_hardware3d(ax,m);ax.text(0,0,-20,'Feed',color='#268b72',fontsize=9)
            for p,c in scene['rays']:ax.plot(*p.T,color=c,alpha=.3,lw=.65)
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
            ax.set_box_aspect(high-low,zoom=zoom);ax.tick_params(labelsize=8)
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
            ax.grid(alpha=.1);last=key
        ax.view_init(elev=23+3*np.sin(phase*np.pi),azim=-76+32*phase)
        stage_text.set_text(record['stage']);counter.set_text(f'{record["counter"]:,} evaluations')
        envelope.set_text(f'{scene["volume_cm3"]:.2f} cm³');dimensions.set_text(' × '.join(f'{v:.1f}' for v in sorted(scene['bbox_mm']))+' mm');blur.set_text(f'{scene["rms_spot_um"]:.2f} µm')
        if loss_ax is not None:
            fine_index=index-len(geometry)-fine_start
            loss_ax.set_visible(fine_index>=0)
            if fine_index>=0:
                visible=min(fine_index,len(fine)-1)
                loss_line.set_data(fine_x[:visible+1],fine_y[:visible+1])
                loss_dot.set_data([fine_x[visible]],[fine_y[visible]])
        clock.set_text(f'{int(t):02d} / {duration} s');fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
    if preview_only:
        for t in [2,15,28,37]:frame(t);fig.savefig(out/f'preview_{t:02d}.jpg')
        plt.close(fig);return dict(preview_directory=str(out))
    movie=out/movie_name
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s',f'{width}x{height}','-pix_fmt','rgb24','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        for i in range(duration*fps):
            encoder.stdin.write(frame(i/fps).tobytes())
            if i==fps*(duration-3):fig.savefig(out/'poster.jpg')
            if i%(fps*5)==0:print(f'video {i/fps:.0f}/{duration}s',flush=True)
        encoder.stdin.close();code=encoder.wait()
        if code:raise RuntimeError(f'ffmpeg failed: {code}')
    plt.close(fig)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames','-of','json',str(movie)]))
    write_json(out/'ffprobe.json',probe)
    stream=probe['streams'][0]
    if stream.get('codec_name')!='h264' or stream.get('width')!=width or stream.get('height')!=height or stream.get('avg_frame_rate')!=f'{fps}/1' or int(stream.get('nb_frames',-1))!=duration*fps or abs(float(probe['format']['duration'])-duration)>.1:
        raise RuntimeError('Encoded movie does not match requested codec, resolution, or duration')
    subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True)
    write_json(out/'decode_verification.json',dict(full_decode_passed=True,expected_frames=duration*fps,codec='h264',fps=fps,resolution=[width,height],duration_seconds=duration))
    (out/'video_source_snapshot.py').write_text(Path(__file__).read_text())
    (out/'watch.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Five-mirror evolution</title><style>body{{font:16px Arial;color:#172635;background:white;max-width:1200px;margin:30px auto}}video{{width:100%}}</style><h1>Five mirrors · 40 seconds</h1><video controls poster="poster.jpg" src="{movie_name}"></video><p><a href="{movie_name}">MP4</a> · <a href="timeline.json">Timeline</a></p>')
    report=source.parent/'report.html'
    if report.exists() and out==source.parent/'visualizations/evolution':
        replacement='<video controls poster="visualizations/evolution/poster.jpg" src="visualizations/evolution/spectrometer_evolution_40s.mp4"></video><p><a href="visualizations/evolution/watch.html">40-second evolution</a> · <a href="visualizations/evolution/timeline.json">Timeline</a></p>'
        report.write_text(report.read_text().replace('<!--VIDEO-->',replacement))
    return dict(movie=str(movie),timeline=str(out/'timeline.json'),probe=probe)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--out');p.add_argument('--preview-only',action='store_true');p.add_argument('--fps',type=int,default=24);p.add_argument('--duration',type=int,default=40)
    p.add_argument('--zoom',type=float,default=1.0);p.add_argument('--show-fine-loss',action='store_true');p.add_argument('--movie-name',default='spectrometer_evolution_40s.mp4')
    a=p.parse_args();print(json.dumps(serial(render_video(a.source,a.out,duration=a.duration,fps=a.fps,preview_only=a.preview_only,zoom=a.zoom,show_fine_loss=a.show_fine_loss,movie_name=a.movie_name)),indent=2))

if __name__=='__main__':main()
