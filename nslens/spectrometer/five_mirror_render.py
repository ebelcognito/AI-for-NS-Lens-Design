"""Render five powered mirrors and a curved grating with an arbitrary grating index."""
import argparse
import json
import html
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from .model import sag

MIRROR='#77a8be';GRATING='#d8ad52';DETECTOR='#ad84ba';INK='#172635'
RAYCOLS=['#2475c2','#2b9a63','#d6534d']

def load_design(source):
    """Load a frozen six-surface layout; configure the matching tracer explicitly."""
    from . import powered_five_mirror as pm
    data=json.loads(Path(source).read_text())
    config=data.get('configuration',{})
    pm.configure(config['segment'],config['aperture'])
    genes=data.get('genes',data.get('best',{}).get('genes'))
    if 'layout' in data:
        layout={k:np.asarray(v) if isinstance(v,list) else v for k,v in data['layout'].items() if k!='surfaces'}
        layout['surfaces']=[{k:np.asarray(v) if isinstance(v,list) else v for k,v in surf.items()} for surf in data['layout']['surfaces']]
    elif genes is not None:
        layout=pm.decode(genes)
    else:
        raise ValueError('Saved result must contain a layout or normalized genes')
    layout['_show_hardware']=bool(config.get('hardware_constraints',False))
    trace=pm.trace(genes,fields=13,waves=17,pupils=11)
    return data,layout,trace


def surface_names(layout):
    names=[];short=[];mirror=0
    for j,_ in enumerate(layout['surfaces']):
        if j==int(layout['grating_index']):names.append('Curved grating');short.append('G')
        else:
            mirror+=1;names.append(f'Mirror {mirror}');short.append(f'M{mirror}')
    return names,short

def _surface_points(s):
    radius=np.linspace(0,s['aperture'],26);theta=np.linspace(0,2*np.pi,90)
    x=radius[:,None]*np.cos(theta);y=radius[:,None]*np.sin(theta);z=sag(x,y,s['coeff'])[0]
    return s['vertex']+x[...,None]*s['basis'][:,0]+y[...,None]*s['basis'][:,1]+z[...,None]*s['basis'][:,2]

HARDWARE='#c97345'
FEED='#268b72'

def hardware_scene(layout,center=None):
    from .slit_hardware import keepout_corners,SETTINGS
    shift=np.zeros(3) if center is None else np.asarray(center)
    corners=keepout_corners()-shift
    faces=[corners[list(ids)] for ids in [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]]
    w=SETTINGS['plate_width_mm']/2;h=SETTINGS['plate_height_mm']/2;gap=SETTINGS['slit_width_mm']/2
    plates=[np.array([[a,-h,0],[b,-h,0],[b,h,0],[a,h,0]])-shift for a,b in [(-w,-gap),(gap,w)]]
    slit_half=SETTINGS['slit_length_mm']/2
    plates += [np.array([[-gap,a,0],[gap,a,0],[gap,b,0],[-gap,b,0]])-shift for a,b in [(-h,-slit_half),(slit_half,h)]]
    L=layout['input_length'];feed=[]
    for yy in [-5,0,5]:
        o=np.array([0.,yy,0.]);angles=np.arange(8)*2*np.pi/8
        for aa in angles:
            d=np.array([np.cos(aa)*L/7.6,-yy+np.sin(aa)*L/7.6,L]);d/=np.linalg.norm(d)
            feed.append(np.array([o-20*d,o])-shift)
    arrow=np.array([[0.,0.,-20],[0.,0.,-3]])-shift
    return dict(corners=corners,faces=faces,plates=plates,feed=feed,arrow=arrow)

def draw_hardware3d(ax,layout,center=None):
    h=hardware_scene(layout,center)
    ax.add_collection3d(Poly3DCollection(h['faces'],facecolor=HARDWARE,edgecolor=HARDWARE,alpha=.12,linewidths=.8))
    ax.add_collection3d(Poly3DCollection(h['plates'],facecolor='#64748b',edgecolor='#475569',alpha=.8,linewidths=.6))
    for p in h['feed']:ax.plot(*p.T,color=FEED,alpha=.15,lw=.6)
    p,q=h['arrow'];ax.quiver(*p,*(q-p),color=FEED,arrow_length_ratio=.16,linewidth=2)
    return h

def draw_hardware2d(ax,layout,a,b):
    from matplotlib.patches import Rectangle
    h=hardware_scene(layout);low=h['corners'].min(0);high=h['corners'].max(0)
    ax.add_patch(Rectangle((low[a],low[b]),high[a]-low[a],high[b]-low[b],facecolor=HARDWARE,edgecolor=HARDWARE,alpha=.3,lw=1.2,zorder=7))
    for plate in h['plates']:ax.plot(plate[[0,1,2,3,0],a],plate[[0,1,2,3,0],b],color='#475569',lw=1.3,zorder=8)
    for p in h['feed']:ax.plot(p[:,a],p[:,b],color=FEED,alpha=.15,lw=.6)
    p,q=h['arrow'];ax.annotate('',xy=q[[a,b]],xytext=p[[a,b]],arrowprops=dict(arrowstyle='->',color=FEED,lw=1.8))
    ax.annotate('Incoming feed',xy=p[[a,b]],xytext=(5,-12),textcoords='offset points',color=FEED,fontsize=9)
    ax.annotate('Slit assembly',xy=(0,0),xytext=(7,9),textcoords='offset points',color=HARDWARE,fontsize=9,fontweight='bold')

def render(source,out=None,title='Geometry feasibility probe',layout_override=None,trace_override=None):
    source=Path(source);out=Path(out) if out else source.parent/'visualizations';out.mkdir(parents=True,exist_ok=True)
    if layout_override is None:
        data,m,r=load_design(source)
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
    names,short_names=surface_names(m)
    grating_index=int(m['grating_index'])
    if len(m['surfaces'])!=6:raise ValueError('Five-mirror renderer requires exactly six optical surfaces')
    lo,hi=m['detector_bounds'];uv=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    detector=m['image']+uv[:,0,None]*m['image_x']+uv[:,1,None]*m['image_y']
    slit=m['origin']+np.array([[0,-5,0],[0,5,0]])
    points=np.concatenate([*[p.reshape(-1,3) for p in clouds],detector,slit])
    hardware=bool(m.get('_show_hardware',False))
    if hardware:
        hw=hardware_scene(m);points=np.concatenate([points,hw['corners'],*hw['feed']])
    low=points.min(axis=0)-5;high=points.max(axis=0)+5
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
        ax.plot_surface(p[:,:,0],p[:,:,1],p[:,:,2],color=GRATING if j==grating_index else MIRROR,alpha=.86,linewidth=0,shade=True)
        ax.plot(*p[-1].T,color='#334155',lw=.8)
        pass  # Screen-space labels are placed after the final camera setup.
    ax.plot(*slit.T,color=INK,lw=3)
    ax.add_collection3d(Poly3DCollection([detector],facecolor=DETECTOR,edgecolor='#75458a',alpha=.48))
    if hardware:draw_hardware3d(ax,m)
    for p,c in rays:ax.plot(*p.T,color=c,alpha=.30,lw=.75)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]))
    ax.set_box_aspect(high-low);ax.view_init(elev=23,azim=-66)
    ax.xaxis.pane.fill=False;ax.yaxis.pane.fill=False;ax.zaxis.pane.fill=False;ax.grid(alpha=.15)
    ax.legend(handles=[Line2D([0],[0],color=c,lw=2,label=f'{w} nm') for c,w in zip(RAYCOLS,[500,800,1100])],loc='upper left',frameon=False)
    fig.suptitle('Five mirrors + curved grating' if hardware else title+'\nFive mirrors + curved grating',fontsize=19,fontweight='bold',color=INK)
    fig.text(.5,.014,('Slit plate 10 × 12 mm · reserved box 12 × 14 × 2 mm · incoming feed 20 mm' if hardware else 'Saved prescription · representative rays · optical surfaces only'),ha='center',fontsize=10,color='#586576')
    fig.get_layout_engine().set(rect=(0,.045,1,.91))
    label_specs=[(s['vertex'],names[j],[(.05,.86),(.98,.84),(.04,.63),(.98,.56),(.03,.34),(.68,.14)][j],INK) for j,s in enumerate(m['surfaces'])]
    label_specs += [(m['image'],'Common detector',(.98,.34),'#75458a'),(m['origin'],'Entrance slit · 10 mm',(.04,.12),INK)]
    if hardware:
        label_specs += [(np.zeros(3),'Slit plate + reserved box',(.02,.21),HARDWARE),(np.array([0.,0.,-18.]),'Incoming feed',(.98,.14),FEED)]
    for point,label,position,color in label_specs:
        xp,yp,_=proj3d.proj_transform(*point,ax.get_proj())
        annotation=ax.annotate(label,xy=(xp,yp),xycoords='data',xytext=position,textcoords='axes fraction',fontsize=11,color=color,fontweight='bold',ha='left',va='center',bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.86),arrowprops=dict(arrowstyle='-',color='#586576',lw=.8,alpha=.8))
        annotation.set_in_layout(False)
    fig.savefig(out/'design_3d.png',dpi=190);fig.savefig(out/'design_3d_preview.jpg',dpi=110);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    for ax,(a,b),label in zip(axs,[(0,2),(1,2)],['Dispersion-plane projection','Transverse projection']):
        for j,p in enumerate(clouds):
            edge=p[-1];ax.fill(edge[:,a],edge[:,b],color=GRATING if j==grating_index else MIRROR,alpha=.75,edgecolor='#334155',lw=.8)
            v=m['surfaces'][j]['vertex'];ax.annotate(short_names[j],(v[a],v[b]),xytext=(7,7),textcoords='offset points',color=INK,fontweight='bold')
        ax.fill(detector[:,a],detector[:,b],color=DETECTOR,alpha=.5,edgecolor='#75458a');ax.plot(slit[:,a],slit[:,b],color=INK,lw=3)
        for p,c in rays:ax.plot(p[:,a],p[:,b],color=c,alpha=.22,lw=.7)
        if hardware:draw_hardware2d(ax,m,a,b)
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
    summary=dict(source=str(source),title=title,hardware_shown=hardware,aperture_radii_mm=[s['aperture'] for s in m['surfaces']],grating_index=grating_index,surface_names=names,sampling=dict(slit_positions=nf,wavelengths=nw,pupil_rays=nrays),throughput=float(good.mean()),rms_spot_um_range=[float(np.nanmin(rms_spot)),float(np.nanmax(rms_spot))],spectral_rms_um_range=[float(np.nanmin(rms_xy[:,:,0])),float(np.nanmax(rms_xy[:,:,0]))])
    (out/'render_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px Arial;color:#172635;background:white;max-width:1250px;margin:35px auto}img{width:100%;height:auto}p{color:#586576}</style><h1>'+title+'</h1><p>Five mirrors + curved grating. Saved optical prescription.</p>'+''.join('<img src="'+f+'">' for f in ['design_3d.png','design_sideviews.png','metric_maps.png']))
    return summary

def render_layout(layout,trace,output,label='Five-powered-mirror candidate'):
    return render(Path(output)/'saved_layout',output,label,layout_override=layout,trace_override=trace)

def build_report(source, four_source='experiments/spectrometer_powered_four_mirror_cma_v3/result.json', reuse_comparison=False):
    """Compare frozen prescriptions with matched ray and package settings."""
    if json.loads(Path(source).read_text()).get('configuration',{}).get('hardware_constraints',False):
        return build_hardware_report(source,reuse_comparison=reuse_comparison)
    from . import powered_four_mirror as four, powered_five_mirror as five
    from .four_mirror import image_metrics
    from .four_mirror_search import optimized_package, package_metrics
    from .search import write_json
    source=Path(source);directory=source.parent
    data,layout,trace=load_design(source)
    previous=json.loads(Path(four_source).read_text());old=four.decode(previous['genes'])
    old_trace=four.trace(previous['genes'],fields=13,waves=17,pupils=11)
    quality={'four':image_metrics(old_trace),'five':image_metrics(trace)}
    package={};settings=dict(seeds=[406,907],iterations=60,final_surface_samples=81)
    cached_file=directory/'report_metrics.json'
    if reuse_comparison and cached_file.exists():
        cached=json.loads(cached_file.read_text())
        if cached['source']!=str(source) or cached['four_source']!=str(four_source) or cached['orientation_search']!=settings:
            raise ValueError('Cached matched comparison does not match the requested sources/settings')
        package=cached['package']
    else:
        for label,m in [('four',old),('five',layout)]:
            attempts=[]
            for seed in settings['seeds']:
                fit=optimized_package(m,seed=seed,iterations=settings['iterations'])
                dense=package_metrics(m,fit['rotation_xyz_deg'],samples=settings['final_surface_samples'])
                attempts.append(dict(seed=seed,geometry=dense,orientation_evaluations=fit['orientation_evaluations']))
            chosen=min(attempts,key=lambda row:row['geometry']['volume_cm3'])
            package[label]=dict(geometry=chosen['geometry'],attempts=attempts)
    reduction=1-package['five']['geometry']['volume_cm3']/package['four']['geometry']['volume_cm3']
    audit_file=directory/'independent_audit.json'
    audit=json.loads(audit_file.read_text()) if audit_file.exists() else None
    saved_genes=data.get('genes',data.get('best',{}).get('genes'))
    audit_matches=bool(audit is not None and np.array_equal(np.asarray(audit.get('genes')),np.asarray(saved_genes)))
    old_access_path=Path(four_source).parent/'slit_access_audit.json'
    old_access=json.loads(old_access_path.read_text()) if old_access_path.exists() else None
    access_link=os.path.relpath(Path(four_source).parent/'visualizations/slit_access.html',directory)
    powers=[]
    for j,surf in enumerate(layout['surfaces']):
        if j!=int(layout['grating_index']):
            powers.append(dict(surface_index=j,quadratic_x_per_mm=float(surf['coeff'][0]),quadratic_y_per_mm=float(surf['coeff'][1]),powered=bool(np.any(np.abs(surf['coeff'])>0))))
    report=dict(source=str(source),four_source=str(four_source),orientation_search=settings,
                sampling=dict(slit_positions=13,wavelengths=17,pupil_rays=int(trace['good'].shape[2])),
                quality=quality,package=package,volume_reduction_fraction=float(reduction),quadratic_powers=powers,
                audit_file=str(audit_file) if audit is not None else None,audit_matches_saved_genes=audit_matches,four_entrance_access=old_access,five_entrance_access=audit.get('entrance_access_20mm') if audit else None,
                limitations=['Envelope includes sampled finite optical masks, 10 mm slit, and full detector rectangle; excludes substrates and mounts.',
                             'Cartesian comparison grid is distinct from the independent ray audit.',
                             'The model launches rays from a zero-width 10 mm slit; external illumination hardware and finite-width slit are not modeled.'])
    write_json(directory/'report_metrics.json',report)
    def cells(key,scale=1,unit='',digits=3):
        return ''.join(f'<td>{quality[label][key]*scale:.{digits}f}{unit}</td>' for label in ['four','five'])
    rows=[]
    for label,key,scale,unit,digits in [('RMS spot radius','rms_spot_mm',1000,' µm',3),('Worst RMS spot radius','max_spot_mm',1000,' µm',3),('Geometric ray survival','throughput',100,'%',3),('Dispersion','dispersion_mm_per_um',1,' mm/µm',4),('Slit magnification','slit_magnification',1,'×',5)]:
        rows.append('<tr><th>'+label+'</th>'+cells(key,scale,unit,digits)+'</tr>')
    rows.insert(0,'<tr><th>Rotated optical envelope</th>'+''.join(f'<td>{package[k]["geometry"]["volume_cm3"]:.3f} cm³</td>' for k in ['four','five'])+'</tr>')
    rows.insert(1,'<tr><th>Envelope dimensions (sorted)</th>'+''.join('<td>'+' × '.join(f'{v:.2f}' for v in sorted(package[k]['geometry']['bbox_mm']))+' mm</td>' for k in ['four','five'])+'</tr>')
    # Audit conclusions are injected only from explicit evidence, never inferred from file existence.
    audit_status='Independent finite-ray and curvature audit is pending. The comparison below is not an acceptance certificate.'
    if audit is not None:
        if not audit_matches:
            audit_status='An audit file exists, but its saved genes do not match this result. Acceptance remains pending for this prescription.'
        elif audit.get('passed') is True:
            received=audit.get('dense_received_rays',audit.get('dense_newton_received_rays'));launched=audit.get('dense_launched_rays',audit.get('dense_newton_launched_rays'))
            evidence=f' Dense-grid rays received: {received:,} of {launched:,}.' if isinstance(received,int) and isinstance(launched,int) else ''
            audit_status='The independent geometric audit passed its recorded acceptance gates.'+evidence+' This is not mechanical certification; see the audit record for exact sampling and limitations.'
        elif audit.get('passed') is False:
            audit_status='The independent audit rejected this candidate. The search result is retained for review, not accepted as a working optical design.'
        else:
            audit_status='Independent audit data is available. See the audit record for its explicit pass/fail gates; received-ray fraction alone does not establish optical or mechanical validity.'
    active_count=sum(row['powered'] for row in powers)
    title='Five powered mirrors + curved grating' if active_count==5 else 'Five mirrors + curved grating'
    incoming=('The entrance is modeled as a 10 mm-long, zero-width slit with an F/3.8 ray cone. '
              'A clear traced path after the slit does not establish access for a real illuminator, slit mount, or upstream optics. '
              'Finite slit width, package walls, substrates, and mount envelopes still need to be included before mechanical sign-off.')
    incoming_evidence=''
    if old_access:
        hit_distances=[r['nearest_positive_distance_mm'] for r in old_access['upstream_obstructions'] if r['nearest_positive_distance_mm'] is not None]
        first_hit=min(hit_distances);fraction=old_access['blocked_fraction_within_distance_mm']['20']
        incoming_evidence=(f'<p>The four-mirror reference has restricted upstream access: the first sampled obstruction is {first_hit:.3f} mm before the slit, and {100*fraction:.2f}% of the reverse-cone samples encounter an optical surface or detector within 20 mm. This measures external feed access, not internal spectrometer throughput. <a href="{html.escape(access_link)}">View the four-mirror entrance-access study.</a></p>')
    if audit_matches and audit.get('entrance_access_20mm'):
        access=audit['entrance_access_20mm']
        incoming_evidence+=f'<p>Five-mirror upstream check: {access["blocked_rays"]:,} of {access["rays"]:,} sampled reverse-cone rays blocked within {access["distance_mm"]:.0f} mm. '
        incoming_evidence+=('The recorded entrance-access gate passed.' if access['passed'] else 'The entrance-access gate failed.')+' This sampled optical-mask check does not include mounts or an illumination assembly.</p>'
    if (directory/'visualizations/slit_access.png').exists():
        incoming_evidence+='<img src="visualizations/slit_access.png" alt="Upstream illumination cone and optical masks before the entrance slit"><p><a href="visualizations/slit_access.html">Open the five-mirror entrance-access view.</a></p>'
    html_text=('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>'+title+'</title><style>body{font:16px/1.6 "DejaVu Sans",Arial;color:#172635;background:white;max-width:1200px;margin:35px auto;padding:0 25px}h1,h2{line-height:1.25}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{padding:10px;border-bottom:1px solid #dae1e6;text-align:left}.muted{color:#586576}.note{background:#f3f7fa;padding:16px 20px;border-left:4px solid #77a8be}a{color:#2475c2}code{font-size:14px}</style>'
        '<h1>'+title+'</h1><p class="muted">Population CMA · coarse-to-freeform refinement of the inherited four-mirror prescription. '+str(active_count)+' of five saved mirror surfaces have nonzero sag coefficients.</p>'
        '<p class="note">'+html.escape(audit_status)+'</p>'
        '<img src="visualizations/design_3d.png" alt="Saved five-mirror optical layout with curved grating and representative spectral rays">'
        '<h2>Matched four-versus-five comparison</h2><p>The saved five-mirror prescription changes the sampled optical envelope by '
        +f'{-100*reduction:+.2f}% relative to the saved four-mirror CMA winner. '
        +f'The longest dimension grows from {max(package["four"]["geometry"]["bbox_mm"]):.2f} to {max(package["five"]["geometry"]["bbox_mm"]):.2f} mm: the five-mirror package is narrower, but not smaller in every dimension. '
        +'Both rigid package orientations use the same search settings: seeds 406 and 907, 60 differential-evolution iterations per seed followed by orientation-only polishing. Final envelopes use 81 surface samples per grid axis plus aperture rims. This rotates the package without changing the optical prescription.</p>'
        '<table><thead><tr><th>Measure</th><th>Four mirrors + grating</th><th>Five mirrors + grating</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
        '<p class="muted">Quality comparison: identical 13 slit positions × 17 wavelengths × '+str(trace['good'].shape[2])+' pupil rays. Geometric blur uses received rays; survival is reported separately. These values do not include diffraction PSF, slit width, coatings, or tolerances.</p>'
        '<h2>Routing and detector quality</h2><img src="visualizations/design_sideviews.png" alt="Two orthogonal views of the five-mirror routing"><img src="visualizations/metric_maps.png" alt="Blur and survival across the slit and wavelength band">'
        '<h2>Entrance slit and physical access</h2>'+incoming_evidence+'<p>'+incoming+'</p>'
        '<p class="muted">The volume includes finite optical surface masks, the full detector rectangle, and slit endpoints. It excludes substrates, mounts, baffles, electronics, and a detector package.</p>'
        '<p><a href="result.json">Saved search result</a> · <a href="report_metrics.json">Matched comparison data</a> · <a href="visualizations/gallery.html">Render gallery</a>'
        +(' · <a href="independent_audit.json">Independent audit</a>' if audit is not None else '')+'</p></html>')
    hardware_warning=directory/'slit_hardware_warning.html'
    if hardware_warning.exists():
        html_text=html_text.replace('<p class="note">',hardware_warning.read_text()+'<p class="note">',1)
    (directory/'report.html').write_text(html_text)
    return report


def build_hardware_report(source,reuse_comparison=False):
    from scipy.optimize import differential_evolution,minimize
    from scipy.spatial.transform import Rotation
    from .four_mirror_search import package_cloud
    from .four_mirror import image_metrics
    from .slit_hardware import keepout_corners,SETTINGS
    from .search import write_json
    import hashlib
    source=Path(source);out=source.parent;data,m,trace=load_design(source)
    digest=hashlib.sha256(source.read_bytes()).hexdigest();cache=out/'report_metrics.json'
    settings=dict(seeds=[406,907],iterations=60,popsize=8,final_surface_samples=81)
    cached=json.loads(cache.read_text()) if cache.exists() else {}
    if reuse_comparison and cached.get('source_sha256')==digest and cached.get('hardware_constraints'):
        envelopes=cached['envelopes']
    else:
        envelopes={}
        for label,extra in [('optical',np.empty((0,3))),('with_slit_assembly',keepout_corners())]:
            points=np.concatenate([package_cloud(m,11),extra]);dense=np.concatenate([package_cloud(m,81),extra])
            def measure(a,cloud=points):
                q=cloud@Rotation.from_euler('xyz',a,degrees=True).as_matrix().T
                box=np.ptp(q,axis=0)
                return dict(volume_cm3=float(np.prod(box)/1000),bbox_mm=box,rotation_xyz_deg=np.asarray(a))
            attempts=[]
            for seed in settings['seeds']:
                f=lambda a:measure(a)['volume_cm3']
                opt=differential_evolution(f,[(-90,90)]*3,seed=seed,maxiter=60,popsize=8,tol=1e-7,polish=False)
                local=minimize(f,opt.x,method='Powell',bounds=[(-90,90)]*3,options=dict(maxiter=80,maxfev=3000,xtol=1e-7,ftol=1e-9))
                angles=local.x if local.fun<opt.fun else opt.x
                attempts.append(dict(seed=seed,geometry=measure(angles,dense),evaluations=opt.nfev+local.nfev))
            chosen=min(attempts,key=lambda r:r['geometry']['volume_cm3'])
            envelopes[label]=dict(geometry=chosen['geometry'],attempts=attempts)
    audit_path=out/'independent_audit.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    matched=bool(np.array_equal(np.asarray(audit.get('genes')),np.asarray(data['genes'])))
    passed=bool(matched and audit.get('passed') and audit.get('slit_hardware',{}).get('passed'))
    quality=image_metrics(trace)
    report=dict(source=str(source),source_sha256=digest,hardware_constraints=True,hardware_settings=SETTINGS,
                orientation_search=settings,envelopes=envelopes,quality=quality,audit_matches_saved_genes=matched,
                audit_passed=passed,sampling=dict(fields=13,waves=17,pupil_points=trace['good'].shape[2]))
    write_json(cache,report)
    (out/'five_mirror_render_snapshot.py').write_text(Path(__file__).read_text())
    status='Audit passed' if passed else ('Audit rejected' if matched and audit.get('passed') is False else 'Audit pending')
    rows=''
    for label,key in [('Optical surfaces','optical'),('Optics + reserved slit assembly','with_slit_assembly')]:
        g=envelopes[key]['geometry'];dims=' × '.join(f'{x:.2f}' for x in sorted(g['bbox_mm']))
        rows+=f'<tr><th>{label}</th><td>{g["volume_cm3"]:.3f} cm³</td><td>{dims} mm</td></tr>'
    video=out/'visualizations/evolution/spectrometer_evolution_40s.mp4'
    video_html=('<video controls poster="visualizations/evolution/poster.jpg" src="visualizations/evolution/spectrometer_evolution_40s.mp4"></video><p><a href="visualizations/evolution/watch.html">40-second evolution</a> · <a href="visualizations/evolution/timeline.json">Timeline</a></p>') if video.exists() else '<!--VIDEO-->'
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Five mirrors · slit hardware</title><style>body{{font:16px/1.5 Arial;color:#172635;background:white;max-width:1200px;margin:32px auto;padding:0 24px}}img,video{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}a{{color:#2475c2}}.muted{{color:#64748b}}</style><h1>Five mirrors + curved grating</h1><p>{status} · {data.get('evaluations',0):,} CMA evaluations · RMS spot {quality['rms_spot_mm']*1000:.3f} µm</p>{video_html}<img src="visualizations/design_3d.png" alt="Six optical surfaces, physical slit plate, reserved slit box, and incoming feed"><table><tr><th>Envelope</th><th>Volume</th><th>Sorted dimensions</th></tr>{rows}</table><p class="muted">Plate 10 × 12 mm; reserved volume 12 × 14 × 2 mm; later-ray margin 0.25 mm; upstream feed 20 mm. Other mounts and substrates are not modeled. Finite slit width is not propagated optically.</p><img src="visualizations/design_sideviews.png" alt="Slit hardware and beam routing"><img src="visualizations/metric_maps.png" alt="Spot blur and received rays across slit and wavelength"><p><a href="result.json">Result</a> · <a href="report_metrics.json">Metrics</a> · <a href="independent_audit.json">Audit</a> · <a href="visualizations/gallery.html">Gallery</a></p></html>'''
    (out/'report.html').write_text(document)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('--out');parser.add_argument('--title',default='CMA coarse-to-fine · five-mirror candidate')
    parser.add_argument('--report',action='store_true');parser.add_argument('--four-source',default='experiments/spectrometer_powered_four_mirror_cma_v3/result.json')
    args=parser.parse_args();print(json.dumps(render(args.source,args.out,args.title),indent=2))
    if args.report:
        summary=build_report(args.source,args.four_source)
        print(json.dumps(dict(report=str(Path(args.source).parent/'report.html')),indent=2))

if __name__=='__main__':main()
