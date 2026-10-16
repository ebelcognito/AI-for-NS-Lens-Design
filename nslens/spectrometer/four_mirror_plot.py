"""Render serialized baseline and folded optical layouts on a common scale."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import sag


def draw_layout(ax, layout, title, paths=None):
    colors = ['#d27d28', '#3066be', '#8c4bb8', '#3066be', '#d27d28']
    surfaces = layout['surfaces']
    if len(surfaces) == 3:
        colors = ['#3066be', '#8c4bb8', '#3066be']
    points = []
    for index, s in enumerate(surfaces):
        b = np.asarray(s['basis']); v = np.asarray(s['vertex'])
        angles = np.linspace(0, 2*np.pi, 97)
        x = float(s['aperture'])*np.cos(angles)
        y = float(s['aperture'])*np.sin(angles)
        z = sag(x, y, s['coeff'])[0]
        edge = np.column_stack([x,y,z]) @ b.T + v
        points.append(edge)
        ax.plot(*edge.T, color=colors[index], lw=1.8)
        for angle in [0, np.pi/2]:
            t=np.linspace(-s['aperture'],s['aperture'],61)
            x=t*np.cos(angle); y=t*np.sin(angle)
            p=np.column_stack([x,y,sag(x,y,s['coeff'])[0]]) @ b.T + v
            ax.plot(*p.T, color=colors[index], lw=.6, alpha=.5)
        label=s.get('name', ['M1','Grating','M2'][index] if len(surfaces)==3 else ['F1','M1','Grating','M2','F2'][index])
        ax.text(*v, label, fontsize=8)
    image=np.asarray(layout['image'])
    origin=np.asarray(layout.get('origin',[0.,0.,0.]))
    points.extend([image[None,:],origin[None,:]])
    if 'detector_bounds' in layout:
        bounds=np.asarray(layout['detector_bounds'])
        corners=np.array([[bounds[0,0],bounds[0,1]],[bounds[1,0],bounds[0,1]],
                          [bounds[1,0],bounds[1,1]],[bounds[0,0],bounds[1,1]],[bounds[0,0],bounds[0,1]]])
        detector=image+corners[:,0,None]*np.asarray(layout['image_x'])+corners[:,1,None]*np.asarray(layout['image_y'])
        ax.plot(*detector.T,color='crimson',lw=1.3);points.append(detector)
    slit=origin+np.array([[0.,-5.,0.],[0.,5.,0.]])
    ax.plot(*slit.T,color='black',lw=1.5);points.append(slit)
    if paths is None:
        p=np.vstack([origin]+[np.asarray(s['vertex']) for s in surfaces]+[image])
        ax.plot(*p.T,color='#269c8d',lw=1.2,alpha=.8)
    else:
        paths=np.asarray(paths)
        for k in range(0,paths.shape[1],max(1,paths.shape[1]//40)):
            ax.plot(*paths[:,k].T,color='#269c8d',lw=.5,alpha=.25)
    ax.scatter(*origin,color='black',s=15); ax.text(*origin,'Slit',fontsize=8)
    ax.scatter(*image,color='crimson',s=15); ax.text(*image,'Detector',fontsize=8)
    ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
    ax.view_init(elev=23,azim=-63)
    return np.vstack(points)


def render(layout, baseline, output, title='Four mirrors + curved grating', paths=None):
    fig=plt.figure(figsize=(14,7),layout='constrained')
    axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
    clouds=[draw_layout(axes[0],baseline,'Baseline: two powered mirrors + grating'),
            draw_layout(axes[1],layout,title,paths=paths)]
    span=max(float(np.ptp(c,axis=0).max()) for c in clouds)*1.12
    for ax,c in zip(axes,clouds):
        center=(c.max(axis=0)+c.min(axis=0))/2
        ax.set_xlim(center[0]-span/2,center[0]+span/2)
        ax.set_ylim(center[1]-span/2,center[1]+span/2)
        ax.set_zlim(center[2]-span/2,center[2]+span/2)
        ax.set_box_aspect([1,1,1])
    fig.suptitle('Optical geometry comparison — equal axis scales',fontsize=16)
    fig.text(.5,.015,'Blue: powered mirrors  |  Purple: curved grating  |  Orange: added flat folds\nOptical surfaces only; mounts and substrates are not modeled.',ha='center',fontsize=10)
    output=Path(output);fig.savefig(output,dpi=180);plt.close(fig)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result');p.add_argument('--output',default=None)
    a=p.parse_args();r=json.loads(Path(a.result).read_text())
    render(r.get('layout',r.get('best',{}).get('layout')),r['baseline']['layout'],a.output or str(Path(a.result).with_name('comparison.png')))
