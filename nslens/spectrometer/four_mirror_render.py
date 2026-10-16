"""Render saved folded layouts using the final spectrometer's visual theme."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from . import four_mirror as fm
from .model import sag

MIRROR='#77a8be';GRATING='#d8ad52';DETECTOR='#ad84ba';INK='#172635'
RAYCOLS=['#2475c2','#2b9a63','#d6534d']
NAMES=['Entrance fold F1','Primary mirror','Curved grating','Tertiary mirror','Detector fold F2']

def load_design(source):
    data=json.loads(Path(source).read_text())
    baseline_path=data.get('configuration',{}).get('source',data.get('baseline_source',fm.DEFAULT_BASELINE))
    baseline=fm.load_baseline(baseline_path,fields=5,waves=7,pupils=7 if 'fallback' in str(source) else 9)
    row=data.get('best',data)
    if 'params' not in row and data.get('verified'):row=data['verified'][1]
    if 'layout' in data:
        layout={k:np.asarray(v) if isinstance(v,list) else v for k,v in data['layout'].items() if k!='surfaces'}
        layout['surfaces']=[{k:np.asarray(v) if isinstance(v,list) else v for k,v in s.items()} for s in data['layout']['surfaces']]
    else:layout=fm.build_layout(row['params'],baseline)
    return data,baseline,layout

def _surface_points(s):
    radius=np.linspace(0,s['aperture'],26);theta=np.linspace(0,2*np.pi,90)
    x=radius[:,None]*np.cos(theta);y=radius[:,None]*np.sin(theta);z=sag(x,y,s['coeff'])[0]
    return s['vertex']+x[...,None]*s['basis'][:,0]+y[...,None]*s['basis'][:,1]+z[...,None]*s['basis'][:,2]

def render(source,out=None,title='Geometry feasibility probe',layout_override=None,trace_override=None):
    source=Path(source);out=Path(out) if out else source.parent/'visualizations';out.mkdir(parents=True,exist_ok=True)
    if layout_override is None:
        data,baseline,m=load_design(source)
        r=fm.trace(m,baseline,fields=13,waves=17,pupils=11)
    else:
        m=layout_override;r=trace_override
    nf,nw,nrays=r['good'].shape;npupil=11
    good=r['good'];xy=r['xy'];count=good.sum(axis=2)
    cent=np.sum(np.where(good[...,None],xy,0),axis=2)/np.maximum(count[...,None],1)
    delta=xy-cent[:,:,None,:]
    rms_xy=np.sqrt(np.sum(np.where(good[...,None],delta**2,0),axis=2)/np.maximum(count[...,None],1))*1000
    rms_spot=np.linalg.norm(rms_xy,axis=-1);survival=good.mean(axis=2)
    rms_xy[count==0]=np.nan;rms_spot[count==0]=np.nan
    slits=r['slit'][:,0,0];waves=r['wavelength'][0,:,0]*1e6
    np.savez(out/'metric_grid.npz',slit_mm=slits,wavelength_nm=waves,rms_spot_um=rms_spot,rms_spectral_um=rms_xy[:,:,0],rms_spatial_um=rms_xy[:,:,1],throughput=survival)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'figure.facecolor':'white'})
    clouds=[_surface_points(s) for s in m['surfaces']]
    names=NAMES if not np.any(m['surfaces'][0]['coeff']) and not np.any(m['surfaces'][4]['coeff']) else ['Mirror 1','Mirror 2','Curved grating','Mirror 3','Mirror 4']
    lo,hi=m['detector_bounds'];uv=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    detector=m['image']+uv[:,0,None]*m['image_x']+uv[:,1,None]*m['image_y']
    slit=m['origin']+np.array([[0,-5,0],[0,5,0]])
    points=np.concatenate([*[p.reshape(-1,3) for p in clouds],detector,slit]);low=points.min(axis=0)-5;high=points.max(axis=0)+5
    u=np.linspace(-1,1,npupil);pu,pv=np.meshgrid(u,u);mask=pu**2+pv**2<=1+1e-12;pu=pu[mask];pv=pv[mask]
    choices=[int(np.argmin((pu-a)**2+(pv-b)**2)) for a,b in [(0,0),(.8,0),(-.8,0),(0,.8),(0,-.8)]] if nrays==len(pu) else list(np.unique(np.linspace(0,nrays-1,min(5,nrays),dtype=int)))
    rays=[]
    for fi in [0,nf//2,nf-1]:
        for wi,c in zip([0,nw//2,nw-1],RAYCOLS):
            for pi in choices:
                idx=(fi*nw+wi)*nrays+pi
                if good[fi,wi,pi]:rays.append((r['paths'][:,idx],c))
    fig=plt.figure(figsize=(13,9),layout='constrained');ax=fig.add_subplot(111,projection='3d')
    offsets=[(3,1,3),(-6,1,6),(3,2,1),(-6,2,-5),(3,2,-4)]
    for j,(s,p) in enumerate(zip(m['surfaces'],clouds)):
        ax.plot_surface(p[:,:,0],p[:,:,1],p[:,:,2],color=GRATING if j==2 else MIRROR,alpha=.86,linewidth=0,shade=True)
        ax.plot(*p[-1].T,color='#334155',lw=.8)
        pass  # Screen-space labels are placed after the final camera setup.
    ax.plot(*slit.T,color=INK,lw=3)
    ax.add_collection3d(Poly3DCollection([detector],facecolor=DETECTOR,edgecolor='#75458a',alpha=.48))
    pass
    for p,c in rays:ax.plot(*p.T,color=c,alpha=.30,lw=.75)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]))
    ax.set_box_aspect(high-low);ax.view_init(elev=23,azim=-66)
    ax.xaxis.pane.fill=False;ax.yaxis.pane.fill=False;ax.zaxis.pane.fill=False;ax.grid(alpha=.15)
    ax.legend(handles=[Line2D([0],[0],color=c,lw=2,label=f'{w} nm') for c,w in zip(RAYCOLS,[500,800,1100])],loc='upper left',frameon=False)
    fig.suptitle(title+'\nFour mirrors + curved grating',fontsize=19,fontweight='bold',color=INK)
    fig.text(.5,.014,'Saved prescription · representative rays · optical surfaces only',ha='center',fontsize=10,color='#586576')
    fig.get_layout_engine().set(rect=(0,.045,1,.91))
    label_specs=[(s['vertex'],names[j],[(.70,.72),(.06,.78),(.79,.40),(.04,.28),(.39,.10)][j],INK) for j,s in enumerate(m['surfaces'])]
    label_specs += [(m['image'],'Common detector',(.05,.48),'#75458a'),(m['origin'],'Entrance slit · 10 mm',(.72,.15),INK)]
    for point,label,position,color in label_specs:
        xp,yp,_=proj3d.proj_transform(*point,ax.get_proj())
        ax.annotate(label,xy=(xp,yp),xycoords='data',xytext=position,textcoords='axes fraction',fontsize=11,color=color,fontweight='bold',ha='left',va='center',bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.86),arrowprops=dict(arrowstyle='-',color='#586576',lw=.8,alpha=.8))
    fig.savefig(out/'design_3d.png',dpi=190);fig.savefig(out/'design_3d_preview.jpg',dpi=110);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    for ax,(a,b),label in zip(axs,[(0,2),(1,2)],['Dispersion-plane projection','Transverse projection']):
        for j,p in enumerate(clouds):
            edge=p[-1];ax.fill(edge[:,a],edge[:,b],color=GRATING if j==2 else MIRROR,alpha=.75,edgecolor='#334155',lw=.8)
            v=m['surfaces'][j]['vertex'];ax.annotate((['F1','M1','G','M2','F2'] if names is NAMES else ['M1','M2','G','M3','M4'])[j],(v[a],v[b]),xytext=(7,7),textcoords='offset points',color=INK,fontweight='bold')
        ax.fill(detector[:,a],detector[:,b],color=DETECTOR,alpha=.5,edgecolor='#75458a');ax.plot(slit[:,a],slit[:,b],color=INK,lw=3)
        for p,c in rays:ax.plot(p[:,a],p[:,b],color=c,alpha=.22,lw=.7)
        ax.set(xlabel=f'{"xyz"[a]} (mm)',ylabel=f'{"xyz"[b]} (mm)',title=label);ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle(title+' · optical routing',fontsize=18,fontweight='bold',color=INK)
    fig.savefig(out/'design_sideviews.png',dpi=190);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,5.7),layout='constrained')
    metrics=[(rms_spot,'RMS spot radius','µm','magma',None),(rms_xy[:,:,0],'Spectral-direction RMS blur','µm','magma',None),(survival*100,'Geometric ray survival','%','viridis',100)]
    for ax,(v,label,units,cmap,vmax) in zip(axs,metrics):
        mesh=ax.pcolormesh(waves,slits,v,shading='nearest',cmap=cmap,vmin=0,vmax=vmax)
        ax.set(xlabel='Wavelength (nm)',ylabel='Slit position (mm)',xticks=[500,650,800,950,1100],yticks=[-5,-2.5,0,2.5,5])
        ax.set_title(f'{label}\n{np.nanmin(v):.2f}–{np.nanmax(v):.2f} {units}',pad=12);fig.colorbar(mesh,ax=ax,label=units,shrink=.85)
    fig.suptitle('Image quality across the full slit and wavelength band',fontsize=18,fontweight='bold',color=INK)
    fig.text(.5,.008,f'{nf} slit positions × {nw} wavelengths × {nrays} pupil rays · {100*good.mean():.3f}% geometric ray survival\nGeometric blur on the common detector. No diffraction PSF, finite slit width or tolerance model.',ha='center',fontsize=10,color='#586576')
    fig.get_layout_engine().set(rect=(0,.12,1,.88));fig.savefig(out/'metric_maps.png',dpi=190);plt.close(fig)
    summary=dict(source=str(source),title=title,aperture_radii_mm=[s['aperture'] for s in m['surfaces']],sampling=dict(slit_positions=nf,wavelengths=nw,pupil_rays=nrays),throughput=float(good.mean()),rms_spot_um_range=[float(np.nanmin(rms_spot)),float(np.nanmax(rms_spot))],spectral_rms_um_range=[float(np.nanmin(rms_xy[:,:,0])),float(np.nanmax(rms_xy[:,:,0]))])
    (out/'render_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px Arial;color:#172635;background:white;max-width:1250px;margin:35px auto}img{width:100%;height:auto}p{color:#586576}</style><h1>'+title+'</h1><p>Four mirrors + curved grating. Saved optical prescription.</p>'+''.join('<img src="'+f+'">' for f in ['design_3d.png','design_sideviews.png','metric_maps.png']))
    return summary

def render_layout(layout,trace,output,label='Four-powered-mirror candidate'):
    return render(Path(output)/'saved_layout',output,label,layout_override=layout,trace_override=trace)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source');parser.add_argument('--out');parser.add_argument('--title',default='Geometry feasibility probe');args=parser.parse_args()
    print(json.dumps(render(args.source,args.out,args.title),indent=2))
