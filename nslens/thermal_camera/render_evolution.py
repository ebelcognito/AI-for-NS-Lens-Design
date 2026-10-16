"""Spectrometer-theme camera frames/video from genuine recorded prescriptions.

No optimization, invented geometry or interpolation. All views keep fixed axes.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .plot import _backend, _surface_profile

INK='#172635'; MUTED='#586576'; RAY='#2475c2'


def render_one(record,index,total,destination):
    genes=record['genes']; optical=_backend(record,genes)
    r=optical.trace(genes,fields=3,waves=1,pupils=13)
    m=r['model'];good=r['good'];xy=r['xy'];npupil=good.shape[2]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                        'figure.facecolor':'white','axes.facecolor':'white',
                        'text.color':INK,'axes.labelcolor':INK,
                        'xtick.color':INK,'ytick.color':INK})
    fig=plt.figure(figsize=(16,9),facecolor='white')
    grid=fig.add_gridspec(2,2,width_ratios=[1.44,1],left=.035,right=.975,
                         bottom=.11,top=.88,wspace=.16,hspace=.35)
    a=fig.add_subplot(grid[:,0],projection='3d')
    mer=fig.add_subplot(grid[0,1])
    lower=grid[1,1].subgridspec(1,2,wspace=.34)
    det=fig.add_subplot(lower[0,0]);spot=fig.add_subplot(lower[0,1])
    surfaces=[('plate','Corrector','#77a8be',.28),
              ('primary','Primary mirror','#77a8be',.72),
              ('secondary','Secondary mirror','#d8ad52',.78)]
    for name,label,color,alpha in surfaces:
        s=m[name];radius=m['secondary_radius'] if name=='secondary' else m['primary_radius']
        inner=m['hole_radius'] if name=='primary' else 0
        rr=np.linspace(inner,radius,19);tt=np.linspace(0,2*np.pi,61)
        x=rr[:,None]*np.cos(tt);y=rr[:,None]*np.sin(tt)
        z=s[0]+optical.sag_xy(x,y,*s[1:])[0]
        a.plot_surface(x,y,z,color=color,alpha=alpha,linewidth=0,shade=True)
        for edge in ([0,-1] if inner else [-1]):
            a.plot(x[edge],y[edge],z[edge],color='#334155',lw=.6)
        a.text(radius+.2,0,float(np.mean(z[-1])),label,fontsize=10,color=INK)
        _surface_profile(mer,optical,s,radius,color,label,inner)
    a.plot([-5,5],[0,0],[0,0],color='#77a8be',lw=.8)
    dz=m['detector_z'];corners=np.array([[-.96,-.72,dz],[.96,-.72,dz],[.96,.72,dz],[-.96,.72,dz]])
    a.add_collection3d(Poly3DCollection([corners],facecolor='#ad84ba',edgecolor='#75458a',alpha=.4))
    a.text(1.1,0,dz+.1,'Detector',color='#75458a',fontsize=10)
    mer.plot([-.96,.96],[dz,dz],color='#75458a',lw=2.5)
    choices=[int(np.argmin(np.sum((r['pupil_xy']-p)**2,axis=1)))
             for p in [(-.9,0),(.9,0),(0,-.9),(0,.9),(.65,.65),(-.65,-.65)]]
    for fi in (3,4,5):
        for pi in choices:
            if not good[fi,0,pi]:continue
            ray=r['paths'][:,fi*npupil+pi]
            if not np.isfinite(ray).all():continue
            a.plot(*ray.T,color=RAY,alpha=.38,lw=.85)
            mer.plot(ray[:,0],ray[:,2],color=RAY,alpha=.3,lw=.7)
    out_count=0
    for fi in range(9):
        p=xy[fi,0,good[fi,0]];p=p[np.isfinite(p).all(axis=1)]
        if len(p):
            det.scatter(*p.T,s=3,color=RAY,alpha=.45,linewidths=0)
            out_count+=int(np.sum(np.any(abs(p)>[1.05,.85],axis=1)))
    p=xy[4,0,good[4,0]];p=p[np.isfinite(p).all(axis=1)]
    if len(p):
        centered=(p-p.mean(axis=0))*1000
        spot.scatter(*centered.T,s=5,color=RAY,alpha=.55,linewidths=0)
        rms=float(np.sqrt(np.mean(np.sum(centered**2,axis=1))))
        clipped=int(np.sum(np.any(abs(centered)>30,axis=1)))
        stat=f'RMS {rms:.2f} µm'+(f'\n{clipped} rays outside view' if clipped else '')
    else:
        rms=None;stat='No surviving central rays'
    a.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',
          xlim=(-6,6),ylim=(-6,6),zlim=(-1.2,9))
    a.set_box_aspect((12,12,10.2));a.view_init(elev=23,azim=-66)
    a.xaxis.pane.fill=a.yaxis.pane.fill=a.zaxis.pane.fill=False
    a.grid(alpha=.15);a.set_title('3D optical layout · fixed camera',loc='left',fontsize=13)
    mer.set(xlabel='x (mm)',ylabel='z (mm)',xlim=(-5.6,5.6),ylim=(-1.2,9),
            title='Side view · x–z projection')
    mer.set_aspect('equal',adjustable='box');mer.grid(alpha=.2)
    det.add_patch(Rectangle((-.96,-.72),1.92,1.44,fill=False,color='#75458a',lw=.8,ls='--'))
    det.set(xlabel='Sensor x (mm)',ylabel='Sensor y (mm)',xlim=(-1.05,1.05),ylim=(-.85,.85),
            title='Detector · nine fields');det.set_aspect('equal');det.grid(alpha=.2)
    if out_count:det.text(.02,.02,f'{out_count} rays outside view',transform=det.transAxes,fontsize=8,color='#9a3412')
    spot.add_patch(Rectangle((-6,-6),12,12,fill=False,color='#94a3b8',lw=.8,ls='--'))
    spot.set(xlabel='Δx (µm)',ylabel='Δy (µm)',xlim=(-30,30),ylim=(-30,30),title='On-axis spot · 11 µm')
    spot.set_aspect('equal');spot.grid(alpha=.2)
    spot.text(.98,.98,stat,transform=spot.transAxes,ha='right',va='top',fontsize=8,
              bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    key=record.get('stage','Saved design')
    stage={'cma_base':'CMA base · 8 variables','cma_asphere':'Radial aspheres · 14 variables',
           'cma_final':'Radial CMA result','base':'XY base · 20 variables',
           'freeform':'Full XY freeform · 41 variables','least_squares':'Least-squares polish',
           'least_squares_final':'Polished result'}.get(key,key.replace('_',' '))
    count=record.get('evaluations',record.get('calls'))
    suffix=f' · {int(count):,} '+('residual calls' if 'calls' in record else 'candidate evaluations') if count else ''
    fig.suptitle('Thermal camera evolution  ·  '+stage+suffix,x=.04,ha='left',fontsize=19,weight='bold',color=INK)
    fig.text(.04,.925,record.get('caption','Actual saved/replayed prescription · no interpolation'),fontsize=10,color=MUTED)
    fig.text(.04,.055,f'State {index+1}/{total}  ·  {int(good.sum()):,}/{good.size:,} rays survive on this rendering grid'
             '  ·  fixed axes and camera  ·  dashed spot square = 12 µm pixel',fontsize=10,color=MUTED)
    fig.text(.04,.029,'One reconstructed channel; steering prisms and packaging omitted. Blue = rays at 11 µm. Geometric spots, not diffraction PSFs.',fontsize=9,color=MUTED)
    fig.savefig(destination,dpi=120);plt.close(fig)
    return dict(stage=record.get('stage'),survived=int(good.sum()),launched=int(good.size),on_axis_rms_um=rms)


def _render_job(args):
    return render_one(*args)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trajectory');p.add_argument('--out',required=True)
    p.add_argument('--fps',type=float,default=2);p.add_argument('--limit',type=int,default=65);p.add_argument('--still',action='store_true')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--changes-only',action='store_true')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    payload=json.loads(Path(args.trajectory).read_text());records=payload['records']
    if args.changes_only:
        records=[row for i,row in enumerate(records) if i==0 or i==len(records)-1
                 or row['genes']!=records[i-1]['genes'] or row['stage']!=records[i-1]['stage']
                 or (i+1<len(records) and row['stage']!=records[i+1]['stage'])]
    # Keep all stage boundaries; subsample only actual states, never interpolate.
    if args.limit and len(records)>args.limit:
        keep=set(np.linspace(0,len(records)-1,args.limit,dtype=int).tolist())
        for i in range(len(records)):
            if i==0 or i==len(records)-1 or records[i]['stage']!=records[i-1]['stage']:keep.add(i)
            if i+1<len(records) and records[i]['stage']!=records[i+1]['stage']:keep.add(i)
        records=[records[i] for i in sorted(keep)]
    jobs=[(record,i,len(records),out/f'frame_{i:04d}.png') for i,record in enumerate(records)]
    metadata=[]
    with ProcessPoolExecutor(max_workers=max(1,min(args.workers,len(records)))) as pool:
        for i,row in enumerate(pool.map(_render_job,jobs)):
            metadata.append(row)
            print(f'{i+1}/{len(records)} {jobs[i][-1]}',flush=True)
    (out/'frames.json').write_text(json.dumps(dict(frames=metadata,source=str(Path(args.trajectory).resolve()),
                      note='All frames are genuine recorded prescriptions; unchanged states may be held.'),indent=2)+'\n')
    render_one(records[-1],len(records)-1,len(records),out/'final_design.png')
    if not args.still:
        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(args.fps),'-i',str(out/'frame_%04d.png'),
                        '-frames:v',str(len(records)+int(3*args.fps)),'-vf',
                        f'trim=end_frame={len(records)},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=3',
                        '-c:v','libx264','-pix_fmt','yuv420p','-crf','19',str(out/'evolution.mp4')],check=True)
        print(out/'evolution.mp4')


if __name__=='__main__':main()




