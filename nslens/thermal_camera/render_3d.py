"""Presentation render of saved camera surfaces and traced rays; no redesign."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from .plot import _backend

BG='#080f1e'; WHITE='#e8f0fc'; MUTED='#8dabc8'


def render(source, out):
    source=Path(source); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    data=json.loads(source.read_text()); optical=_backend(data,data['genes'])
    r=optical.trace(data['genes'],fields=3,waves=1,pupils=17);m=r['model']
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':WHITE,
                        'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED})
    fig=plt.figure(figsize=(16,10),facecolor=BG)
    ax=fig.add_axes([.00,.10,.72,.78],projection='3d',facecolor=BG,computed_zorder=False)
    ax.set_proj_type('ortho')
    def disk(surface,radius,color,alpha,inner=0):
        rr=np.linspace(inner,radius,34);tt=np.linspace(0,2*np.pi,121)
        x=rr[:,None]*np.cos(tt);y=rr[:,None]*np.sin(tt)
        z=surface[0]+optical.sag_xy(x,y,*surface[1:])[0]
        ax.plot_surface(x,y,z,color=color,alpha=alpha,linewidth=0,shade=True,
                        rcount=34,ccount=121,zorder=2)
        for row in ([0,-1] if inner else [-1]):
            ax.plot(x[row],y[row],z[row],color=color,alpha=.95,lw=1.3,zorder=5)
        # Sparse actual-surface contours reveal the freeform geometry.
        for j in (10,20,28):
            ax.plot(x[j],y[j],z[j],color=color,alpha=.16,lw=.5,zorder=3)
    disk((0,0,0,[]),m['primary_radius'],'#64e5d2',.045)
    disk(m['plate'],m['primary_radius'],'#64e5d2',.12)
    disk(m['primary'],m['primary_radius'],'#83caff',.30,m['hole_radius'])
    disk(m['secondary'],m['secondary_radius'],'#f6be69',.62)
    z=m['detector_z']; corners=np.array([[-.96,-.72,z],[.96,-.72,z],[.96,.72,z],[-.96,.72,z]])
    ax.add_collection3d(Poly3DCollection([corners],facecolors='#ae87ff',edgecolors='#dfceff',alpha=.65,zorder=7))
    colors=['#57d8ff','#b2f594','#ff89c5']; fields=[3,4,5]
    npup=r['good'].shape[-1]
    for fi,color in zip(fields,colors):
        valid=np.flatnonzero(r['good'][fi,0])
        # Select actual accepted rays around the pupil, not invented guide lines.
        theta=np.arctan2(r['pupil_xy'][valid,1],r['pupil_xy'][valid,0])
        order=valid[np.argsort(theta)]
        chosen=order[np.linspace(0,len(order)-1,9,dtype=int)]
        for pi in chosen:
            p=r['paths'][:,fi*npup+pi]
            ax.plot(*p.T,color=color,lw=4,alpha=.035,zorder=8)
            ax.plot(*p.T,color=color,lw=.8,alpha=.64,zorder=9)
        pts=r['xy'][fi,0,r['good'][fi,0]]
        ax.scatter(pts[:,0],pts[:,1],np.full(len(pts),z),color=color,s=.3,alpha=.3,zorder=10)
    # Dimension bracket is in model coordinates, not an invented housing.
    ax.plot([-5.8,-5.8],[0,0],[0,z],color=MUTED,lw=.9,zorder=10)
    for zz in [0,z]:ax.plot([-6,-5.6],[0,0],[zz,zz],color=MUTED,lw=.9,zorder=10)
    ax.text(-6.15,0,z/2,f'{z:.2f} mm',color=MUTED,fontsize=10,zorder=11)
    ax.set(xlim=(-6.3,5.5),ylim=(-5.5,5.5),zlim=(-1.1,8.5))
    ax.set_box_aspect((11.8,11,9.6),zoom=1.15);ax.view_init(elev=24,azim=-57);ax.set_axis_off()
    fig.text(.055,.94,'FOLDED / THERMAL',fontsize=29,weight='bold',color=WHITE)
    fig.text(.057,.905,'XY FREEFORM CAMERA     /     CMA + LEAST-SQUARES',fontsize=11,color=MUTED)
    fig.text(.735,.82,'THE OPTICAL STACK',fontsize=12,weight='bold')
    labels=[('#ae87ff','04  IMAGE PLANE','Assumed 12 µm-pitch detector'),
            ('#83caff','03  ANNULAR PRIMARY','Freeform mirror · central exit aperture'),
            ('#f6be69','02  SECONDARY','Beam returns toward the detector'),
            ('#64e5d2','01  CORRECTOR','Finite refractive plate · entrance')]
    for i,(c,title,detail) in enumerate(labels):
        yy=.765-i*.079
        fig.text(.735,yy,title,color=c,fontsize=11,weight='bold')
        fig.text(.735,yy-.025,detail,color=MUTED,fontsize=9)
    fig.text(.735,.418,'LIGHT PATH',fontsize=10,weight='bold')
    fig.text(.735,.385,'Corrector → primary → secondary\n→ primary hole → image plane',fontsize=10,color=MUTED,linespacing=1.6,va='top')
    fig.text(.735,.302,'ON-AXIS SPOT',fontsize=10,weight='bold')
    spot=fig.add_axes([.755,.105,.125,.175],facecolor=BG)
    p=r['xy'][4,0,r['good'][4,0]]*1000;p-=p.mean(axis=0)
    spot.scatter(*p.T,s=3,color='#b2f594',alpha=.55)
    spot.add_patch(Rectangle((-6,-6),12,12,fill=False,color=MUTED,ls='--',lw=.8))
    spot.set(xlim=(-12,12),ylim=(-12,12),aspect='equal',xticks=[-10,0,10],yticks=[-10,0,10])
    for spine in spot.spines.values():spine.set_color('#26394f')
    spot.tick_params(labelsize=8);spot.grid(color='#26394f',alpha=.5)
    fig.text(.9,.205,f'{np.sqrt(np.mean(np.sum(p*p,axis=1))):.2f} µm',fontsize=15,color='#b2f594',ha='center')
    fig.text(.9,.17,'RMS radius',fontsize=9,color=MUTED,ha='center')
    fig.text(.9,.125,'Dashed square:\n12 µm pixel',fontsize=8,color=MUTED,ha='center')
    fig.text(.055,.082,'ONE CHANNEL  /  ACTUAL SAVED SURFACES  /  REPRESENTATIVE TRACED RAYS',fontsize=10,color=MUTED)
    fig.text(.055,.045,'Surfaces shown translucent for visibility. Ray colors distinguish field angles, not wavelength.\n'
             'Reconstruction, not the Fraunhofer prescription. Steering prisms and packaging omitted; geometric spots, no diffraction.',
             fontsize=9,color=MUTED,linespacing=1.6)
    fig.savefig(out/'design_3d.png',dpi=190,facecolor=BG)
    fig.savefig(out/'design_3d_preview.jpg',dpi=100,facecolor=BG)
    ax.view_init(elev=38,azim=125)
    fig.savefig(out/'design_3d_reverse.png',dpi=190,facecolor=BG)
    plt.close(fig)
    print(out/'design_3d.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--out',required=True)
    a=p.parse_args();render(a.source,a.out)


