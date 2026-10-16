"""Physical-pose/finite-slit screening. Geometric metrics only, NOT fab certification."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import model, four_mirror as fold, powered_five_mirror as five

REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO/'experiments/spectrometer_powered_five_mirror_cma_v2_hardware/prescription.json'
REFERENCE = REPO/'experiments/spectrometer_final_v1/seed_102/result.json'
SPEC = REPO/'nslens/spectrometer/fabrication_study_spec_v1.json'
LIMITATIONS = [
 'Geometric screening only: no diffraction, physical grating efficiency, coatings, detector QE or stray light.',
 'Spectral RMS and central-80-percent widths are NOT FWHM and cannot certify the 3/4 nm FWHM requirement.',
 'No surface figure/slope errors, substrate/mount solids, thermal deformation or manufacturing correlations.',
 'Pose axes are global XYZ; rotations about optical vertices, not yet physical mounting datums.',
 'Only detector axial translation is offered as a bounded compensator; no inferred build-ready tolerances.',
 'Local Newton intersections and same-engine calculations; separate independent-engine validation remains required.',
 'Sampling must be converged before drawing tolerance conclusions; one trial alignment serves all cells.',
]

def load_layout(design):
    if design == 'five_mirror':
        raw=json.loads(CANDIDATE.read_text())['layout']
        m=copy.deepcopy(raw)
        for s in m['surfaces']:
            for k in ('vertex','basis','coeff'): s[k]=np.asarray(s[k],float)
        for k in ('image','image_x','image_y','image_normal','origin','detector_bounds'): m[k]=np.asarray(m[k],float)
    elif design == 'three_surface':
        raw=json.loads(REFERENCE.read_text())
        m=model.decode(raw['genes'],aperture_radii=raw.get('configuration',{}).get('aperture_radii',[8.9,8.43,12.11]))
        m.update(input_length=float(m['lengths'][0]),grating_index=1)
    else: raise ValueError('Unknown design')
    # Explicit proposed 8 x 12 mm detector for matched comparisons.
    m['detector_bounds']=np.array([[-4.,-6.],[4.,6.]])
    return m

def rays(m,fields,waves,pupils,slit_samples,slit_width_um):
    u=np.linspace(-1,1,pupils);x,y=np.meshgrid(u,u);keep=x*x+y*y<=1+1e-12
    pupil=np.column_stack([x[keep],y[keep]])
    offsets=(np.arange(slit_samples)+.5)/slit_samples-.5
    f,w,s,p=np.meshgrid(np.linspace(-5,5,fields),np.linspace(*model.WAVELENGTHS,waves),offsets*slit_width_um/1000,np.arange(len(pupil)),indexing='ij')
    origin=np.column_stack([s.ravel(),f.ravel(),np.zeros(f.size)])
    L=m['input_length'];d=model.unit(np.array([0,0,L])-origin+np.column_stack([pupil[p.ravel()],np.zeros(f.size)])*L/7.6)
    return origin,d,w.ravel(),(fields,waves,slit_samples*len(pupil))

def trace_layout(m,launch):
    origin,d,wave,shape=launch;o=origin.copy();d=d.copy();good=np.ones(len(o),bool);blocked=np.zeros(len(o),bool)
    with np.errstate(all='ignore'):
        for j,s in enumerate(m['surfaces']):
            p,n,t,valid=model.intersect(o,d,s)
            for k,other in enumerate(m['surfaces']):
                if k!=j:
                    _,_,ot,ov=model.intersect(o,d,other);blocked|=good&ov&(ot<t-1e-5)
            dt,dv=fold.detector_intersection(o,d,m);blocked|=good&dv&(dt<t-1e-5)
            good &= valid
            if j==m['grating_index']:
                d,valid_order=model.diffract(d,n,s['basis'][:,0],wave);good &= valid_order
            else: d=model.reflect(d,n)
            o=p
        den=d@m['image_normal'];t=((m['image']-o)@m['image_normal'])/np.where(abs(den)>1e-12,den,1e-12)
        for s in m['surfaces']:
            _,_,ot,ov=model.intersect(o,d,s);blocked|=good&ov&(ot<t-1e-5)
        image=o+t[:,None]*d
        xy=np.column_stack([(image-m['image'])@m['image_x'],(image-m['image'])@m['image_y']])
        b=m['detector_bounds'];good &= np.isfinite(xy).all(axis=1)&np.isfinite(t)&(t>0)&~blocked&np.all((xy>=b[0])&(xy<=b[1]),axis=1)
    return xy.reshape(*shape,2),good.reshape(shape)

def metrics(m,launch,pixel_um=10,reference_centroids=None):
    xy,good=trace_layout(m,launch);n=good.sum(axis=2)
    cent=np.sum(np.where(good[...,None],xy,0),axis=2)/np.maximum(n[...,None],1)
    variance=np.sum(np.where(good,(xy[...,0]-cent[:,:,None,0])**2,0),axis=2)/np.maximum(n,1)
    # Nominal fixed dispersion conversion keeps perturbations comparable. No fitted per-trial calibration.
    rms=np.sqrt(variance+(pixel_um/1000)**2/12)*100
    widths=[]
    for idx in np.ndindex(good.shape[:2]):
        vals=xy[idx][good[idx],0]
        widths.append(float(np.diff(np.quantile(vals,[.1,.9]))[0]*100) if len(vals) else None)
    valid=bool((n>0).all())
    out={'all_cells_have_rays':valid,'launched_rays':int(good.size),'received_rays':int(good.sum()),'geometric_survival':float(good.mean()),'minimum_cell_survival':float((n/good.shape[2]).min()),'max_spectral_rms_nm_with_pixel_variance':float(rms.max()) if valid else None,'max_geometric_central80_width_nm':max(widths) if valid else None,'centroids_mm':cent.tolist(),'conversion_nm_per_mm':100}
    if reference_centroids is not None:
        out['max_uncalibrated_centroid_shift_nm']=float(np.max(np.abs(cent[...,0]-reference_centroids[...,0]))*100) if valid else None
    return out

def perturb(m,component,axis,translation_um=0,rotation_arcsec=0):
    out=copy.deepcopy(m);vec=np.eye(3)[axis]
    rot=Rotation.from_rotvec(vec*np.deg2rad(rotation_arcsec/3600)).as_matrix()
    if component=='detector':
        out['image']+=vec*translation_um/1000
        for k in ('image_x','image_y','image_normal'):out[k]=rot@out[k]
    else:
        s=out['surfaces'][int(component)];s['vertex']+=vec*translation_um/1000;s['basis']=rot@s['basis']
    return out

def compact(row):return {k:v for k,v in row.items() if k!='centroids_mm'}
def score(row):
    if not row['all_cells_have_rays']:return 1e9
    return row['max_spectral_rms_nm_with_pixel_variance']+1000*(1-row['minimum_cell_survival'])
def compensate(m,launch,pixel,reference,focus_um):
    choices=[]
    for z in sorted(set([-focus_um,-focus_um/2,0,focus_um/2,focus_um])):
        trial=copy.deepcopy(m);trial['image']+=z/1000*m['image_normal']
        row=metrics(trial,launch,pixel,reference);choices.append((score(row),z,row))
    _,z,row=min(choices,key=lambda item:item[0])
    return {'detector_focus_um':z,'at_travel_limit':bool(focus_um and abs(z)==focus_um),'metrics':compact(row)}

def run(config,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    def save(name,data):
        temp=out/(name+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(out/name)
    save('configuration.json',config)
    design=config['design'];m=load_layout(design)
    launch=rays(m,config['fields'],config['waves'],config['pupils'],config['slit_samples'],config['slit_width_um'])
    nominal=metrics(m,launch,config['pixel_um']);reference=np.asarray(nominal['centroids_mm'])
    source=CANDIDATE if design=='five_mirror' else REFERENCE
    result={'configuration':config,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'limitations':LIMITATIONS,'nominal':nominal,'rows':[]}
    result['code_sha256']={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),REPO/'nslens/spectrometer/model.py',REPO/'nslens/spectrometer/four_mirror.py',SPEC)}
    save('progress.json',{'state':'running','completed':0,'mode':config['mode']});save('result.json',result)
    components=list(range(len(m['surfaces'])))+['detector']
    if config['mode']=='sensitivity':
        jobs=[(c,a,k,sign) for c in components for a in range(3) for k in ('translation','rotation') for sign in (-1,1)]
        for c,a,k,sign in jobs:
            kwargs={('translation_um' if k=='translation' else 'rotation_arcsec'):sign*config['translation_um' if k=='translation' else 'rotation_arcsec']}
            changed=perturb(m,c,a,**kwargs)
            row={'component':c,'axis':'XYZ'[a],'kind':k,'perturbation':kwargs,'uncompensated':compact(metrics(changed,launch,config['pixel_um'],reference))}
            if config['focus_um']>0:row['compensated']=compensate(changed,launch,config['pixel_um'],reference,config['focus_um'])
            result['rows'].append(row);save('result.json',result);save('progress.json',{'state':'running','completed':len(result['rows']),'total':len(jobs)})
    elif config['mode']=='monte_carlo':
        rng=np.random.default_rng(config['seed'])
        result['distribution']='Independent zero-mean Gaussian; translation_um and rotation_arcsec are 1-sigma, not drawing tolerances.'
        for trial in range(config['trials']):
            changed=copy.deepcopy(m);draws=[]
            for c in components:
                for a in range(3):
                    t=float(rng.normal(0,config['translation_um']));r=float(rng.normal(0,config['rotation_arcsec']))
                    changed=perturb(changed,c,a,t,r);draws.append([c,a,t,r])
            row={'trial':trial,'draws':draws,'uncompensated':compact(metrics(changed,launch,config['pixel_um'],reference))}
            if config['focus_um']>0:row['compensated']=compensate(changed,launch,config['pixel_um'],reference,config['focus_um'])
            result['rows'].append(row);save('result.json',result);save('progress.json',{'state':'running','completed':trial+1,'total':config['trials']})
    save('progress.json',{'state':'complete','completed':len(result['rows'])});return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.out)
