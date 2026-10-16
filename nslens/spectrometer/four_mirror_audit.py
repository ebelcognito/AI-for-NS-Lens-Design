"""Independent finite-surface checks for folded spectrometer ray paths.

Checks use polynomial root enumeration, not the search Newton solver. All
fractions use launched rays, including rays marked lost by the nominal trace.
"""
from __future__ import annotations
import numpy as np
from . import model
from .physics_audit import exact_intersect


def dense_pupil(radial_samples=9, azimuth_samples=48):
    """Disk quadrature with an explicit full rim (not Cartesian-disk corners)."""
    a=np.arange(azimuth_samples)*2*np.pi/azimuth_samples
    r=np.linspace(0.,1.,radial_samples)[1:]
    return np.concatenate([np.zeros((1,2)),np.stack([r[:,None]*np.cos(a),r[:,None]*np.sin(a)],axis=-1).reshape(-1,2)])


def audit_paths(paths, surfaces, grating_index=2, wavelengths=None):
    """Audit N+2 supplied path vertices against N ordered finite surfaces.

    Tests intended hits, physical reflection/diffraction, and every unintended
    surface on all N+1 open segments. This independently validates paths; it
    does not silently replace their endpoints with alternative intersections.
    """
    paths=np.asarray(paths,float)
    if paths.ndim!=3 or paths.shape[0]!=len(surfaces)+2:
        raise ValueError('paths must be (number of surfaces + 2, launched rays, 3)')
    nr=paths.shape[1]; blocked=np.zeros(nr,bool); invalid=np.zeros(nr,bool)
    rows=[]; hits=[]; law_errors=[]
    for j in range(len(surfaces)+1):
        start,end=paths[j:j+2]; length=np.linalg.norm(end-start,axis=1)
        direction=model.unit(end-start)
        valid_segment=np.isfinite(length)&(length>1e-6)&np.isfinite(direction).all(axis=1)
        invalid|=~valid_segment
        segment_blocked=np.zeros(nr,bool); blockers=[]
        for k,surface in enumerate(surfaces):
            point,normal,t,good=exact_intersect(start,direction,surface)
            if j==k:
                error=np.linalg.norm(point-end,axis=1)
                intended=good&(error<2e-6)&valid_segment
                invalid|=~intended
                outgoing=model.unit(paths[j+2]-end)
                if j==grating_index and wavelengths is not None:
                    prediction,order_valid=model.diffract(direction,normal,surface['basis'][:,0],np.asarray(wavelengths).reshape(-1))
                    invalid|=~order_valid
                else:
                    prediction=model.reflect(direction,normal)
                angular_error=np.linalg.norm(prediction-outgoing,axis=1)
                invalid|=intended&(angular_error>2e-6)
                hits.append(dict(surface=k,hit_count=int(intended.sum()),max_hit_error_mm=float(error[intended].max()) if intended.any() else None))
                law_errors.append(float(angular_error[intended].max()) if intended.any() else None)
            else:
                obstruct=good&(t<length-1e-5)&valid_segment
                segment_blocked|=obstruct
                if obstruct.any():blockers.append(dict(surface=k,ray_count=int(obstruct.sum())))
        blocked|=segment_blocked
        rows.append(dict(segment=j,obstructed_rays=int(segment_blocked.sum()),blockers=blockers))
    accepted=~(blocked|invalid)
    return dict(launched_rays=nr,accepted_rays=int(accepted.sum()),accepted_fraction=float(accepted.mean()),
                obstructed_rays=int(blocked.sum()),invalid_path_rays=int(invalid.sum()),segments=rows,intended_hits=hits,
                maximum_direction_law_error=max((x for x in law_errors if x is not None),default=None),
                limitation='Independent roots along supplied paths and vector-law checks; excludes mechanical substrates and mounts.')


def sphere_separation(surfaces):
    """Positive lower gaps certify disjoint optical surfaces; negative is inconclusive."""
    radii=[]
    for s in surfaces:
        r=float(s['aperture'])
        bound=sum(abs(c)*r**(i+j)*(i/(i+j))**(i/2)*(j/(i+j))**(j/2) for c,(i,j) in zip(s['coeff'],model.TERMS))
        radii.append(float(np.hypot(r,bound)))
    pairs=[]
    for i in range(len(surfaces)):
        for j in range(i+1,len(surfaces)):
            gap=float(np.linalg.norm(surfaces[i]['vertex']-surfaces[j]['vertex'])-radii[i]-radii[j])
            pairs.append(dict(surface_pair=[i,j],conservative_gap_lower_bound_mm=gap,disjoint_certified=gap>0))
    return dict(enclosing_sphere_radii_mm=radii,pairs=pairs,all_optical_surfaces_disjoint_certified=all(p['disjoint_certified'] for p in pairs),limitation='Negative bounds are inconclusive, not proof of collision. No mount or substrate envelopes included.')


def detector_obstructions(paths, layout):
    """Finite active detector rectangle blocks any earlier open ray segment."""
    paths=np.asarray(paths);nr=paths.shape[1];blocked=np.zeros(nr,bool);rows=[]
    for j in range(len(paths)-2):
        a,b=paths[j:j+2];d=b-a;den=d@layout['image_normal']
        fraction=((layout['image']-a)@layout['image_normal'])/np.where(abs(den)>1e-14,den,1e-14)
        p=a+fraction[:,None]*d-layout['image']
        xy=np.column_stack([p@layout['image_x'],p@layout['image_y']]);bounds=layout['detector_bounds']
        hit=(abs(den)>1e-14)&(fraction>1e-7)&(fraction<1-1e-7)&np.all((xy>=bounds[0])&(xy<=bounds[1]),axis=1)
        blocked|=hit;rows.append(dict(segment=j,detector_hits=int(hit.sum())))
    return dict(launched_rays=nr,blocked_rays=int(blocked.sum()),segments=rows)


def audit_layout(layout, baseline, fields=8, waves=9, radial_samples=7, azimuth_samples=40):
    """Heldout ray audit with immutable manufactured apertures/detector bounds."""
    from . import four_mirror as fold
    pupil=dense_pupil(radial_samples,azimuth_samples)
    sampling=dict(fields=fields,waves=waves,pupil_xy=pupil)
    reference=model.trace(baseline['genes'],aperture_radii=baseline['aperture_radii'],**sampling)
    # Rebuild only nominal isometric paths to test segment crossing parameters;
    # the physical layout, apertures and detector stay frozen from training.
    heldout=dict(baseline,trace=reference)
    nominal=fold.build_layout(layout['params'],heldout)
    numerical=fold.trace(layout,baseline,**sampling)
    exact=fold.trace(layout,baseline,intersect_fn=exact_intersect,**sampling)
    common=exact['good']&reference['good']
    def max_common(a,b):
        delta=np.asarray(a)[common]-np.asarray(b)[common]
        return float(np.max(np.abs(delta))) if delta.size else None
    path_length=lambda r:np.sum(np.linalg.norm(np.diff(r['paths'],axis=0),axis=-1),axis=0).reshape(r['good'].shape)
    independent=audit_paths(exact['paths'],layout['surfaces'],wavelengths=exact['wavelength'])
    detectors=detector_obstructions(exact['paths'],layout)
    fractions=nominal['fold_fractions'];segments=(fractions>0)&(fractions<1)&np.isfinite(fractions)
    return dict(sampling=dict(fields=fields,waves=waves,radial_samples=radial_samples,azimuth_samples=azimuth_samples,pupil_points=len(pupil)),
        launched_rays=int(exact['good'].size),baseline_received_fraction=float(reference['good'].mean()),
        polynomial_received_fraction=float(exact['good'].mean()),newton_received_fraction=float(numerical['good'].mean()),
        good_mask_disagreements=int(np.sum(exact['good']!=numerical['good'])),
        maximum_image_coordinate_error_mm=max_common(exact['xy'],reference['xy']),
        maximum_total_geometric_path_error_mm=max_common(path_length(exact),path_length(reference)),
        maximum_grating_phase_length_error_mm=max_common(exact['phase'],reference['phase']),
        fold_segment_fractions=[dict(minimum=float(f.min()),maximum=float(f.max()),outside_rays=int(np.sum(~v))) for f,v in zip(fractions,segments)],
        fixed_aperture_radii_mm=[float(s['aperture']) for s in layout['surfaces']],fixed_detector_bounds_mm=layout['detector_bounds'].tolist(),
        heldout_aperture_edge_margins_mm=[float(s['aperture']-np.linalg.norm(((exact['paths'][i+1]-s['vertex'])@s['basis'])[:,:2],axis=1).max()) for i,s in enumerate(layout['surfaces'])],
        independent_paths=independent,early_detector_interceptions=detectors,sphere_separation=sphere_separation(layout['surfaces']),
        passed=bool(exact['good'].all() and not np.any(exact['good']!=numerical['good']) and segments.all() and independent['accepted_rays']==exact['good'].size and detectors['blocked_rays']==0 and reference['good'].all() and (max_common(exact['xy'],reference['xy']) or 0)<1e-7 and (max_common(path_length(exact),path_length(reference)) or 0)<1e-7 and (max_common(exact['phase'],reference['phase']) or 0)<1e-7),
        limitation='Sampled heldout rays with polynomial root enumeration, not exhaustive continuous-ray or mechanical-clearance certification.')


def image_metrics(trace):
    """Same centroid reference-sphere convention as the baseline model."""
    r=trace;good=r['good'];count=good.sum(axis=2);xy=r['xy']
    cent=np.sum(np.where(good[...,None],xy,0),axis=2)/np.maximum(count[...,None],1)
    rms=np.sqrt(np.sum(np.where(good,np.sum((xy-cent[:,:,None,:])**2,axis=-1),0),axis=2)/np.maximum(count,1))
    m=r['model'];target=m['image']+cent[...,0,None]*m['image_x']+cent[...,1,None]*m['image_y']
    path=r['opl']+np.linalg.norm(target[:,:,None,:]-r['last_surface'],axis=-1)+r['phase']
    piston=np.sum(np.where(good,path,0),axis=2)/np.maximum(count,1)
    wfe=np.sqrt(np.sum(np.where(good,(path-piston[:,:,None])**2,0),axis=2)/np.maximum(count,1))/r['wavelength'][:,:,0]
    return dict(throughput=float(good.mean()),minimum_cell_throughput=float((count/good.shape[2]).min()),
                rms_spot_mm=float(np.sqrt(np.mean(rms*rms))),maximum_spot_mm=float(rms.max()),
                rms_wavefront_waves=float(np.sqrt(np.mean(wfe*wfe))),maximum_wavefront_waves=float(wfe.max()),
                dispersion_mm_per_um=float(np.mean(abs(cent[:,-1,0]-cent[:,0,0]))/.6),
                slit_magnification=float(abs(np.mean(cent[-1,:,1]-cent[0,:,1]))/10))


def covered_surface_separation(layout, radial_intervals=64, azimuth_samples=360, detector_intervals=64):
    """Conservative gaps from complete parameter-domain sampling cover bounds.

    Polynomial derivative monomial maxima bound the graph's Lipschitz constant.
    Subtract both covering radii from sampled nearest distances. This bounds
    complete optical surfaces, unlike reporting nearest sample distances alone.
    """
    from scipy.spatial import cKDTree
    def monomial_disk_max(i,j,r):
        if i+j==0:return 1.
        return r**(i+j)*(i/(i+j))**(i/2)*(j/(i+j))**(j/2)
    clouds=[];covers=[];names=[];details=[]
    angles=np.arange(azimuth_samples)*2*np.pi/azimuth_samples
    for index,s in enumerate(layout['surfaces']):
        r=float(s['aperture']);radii=np.linspace(0,r,radial_intervals+1)
        xy=np.stack([radii[:,None]*np.cos(angles),radii[:,None]*np.sin(angles)],axis=-1).reshape(-1,2)
        z=model.sag(xy[:,0],xy[:,1],s['coeff'])[0]
        cloud=np.column_stack([xy,z])@s['basis'].T+s['vertex']
        dx=sum(abs(c*i)*monomial_disk_max(i-1,j,r) for c,(i,j) in zip(s['coeff'],model.TERMS) if i)
        dy=sum(abs(c*j)*monomial_disk_max(i,j-1,r) for c,(i,j) in zip(s['coeff'],model.TERMS) if j)
        xy_cover=r/(2*radial_intervals)+2*r*np.sin(np.pi/(2*azimuth_samples))
        cover=float(xy_cover*np.sqrt(1+dx*dx+dy*dy))
        clouds.append(cloud);covers.append(cover);names.append(index)
        details.append(dict(surface=index,sample_count=len(cloud),global_gradient_norm_upper_bound=float(np.hypot(dx,dy)),spatial_cover_radius_mm=cover))
    bounds=layout['detector_bounds'];x,y=np.meshgrid(np.linspace(*bounds[:,0],detector_intervals+1),np.linspace(*bounds[:,1],detector_intervals+1))
    clouds.append(layout['image']+x.ravel()[:,None]*layout['image_x']+y.ravel()[:,None]*layout['image_y'])
    covers.append(float(np.linalg.norm(bounds[1]-bounds[0])/(2*detector_intervals)));names.append('detector')
    details.append(dict(surface='detector',sample_count=x.size,spatial_cover_radius_mm=covers[-1]))
    pairs=[]
    for i in range(len(clouds)):
        tree=cKDTree(clouds[i])
        for j in range(i+1,len(clouds)):
            sampled=float(tree.query(clouds[j])[0].min());lower=sampled-covers[i]-covers[j]
            pairs.append(dict(surface_pair=[names[i],names[j]],sampled_gap_mm=sampled,conservative_gap_lower_bound_mm=lower,disjoint_certified=lower>1e-9))
    return dict(radial_intervals=radial_intervals,azimuth_samples=azimuth_samples,detector_intervals=detector_intervals,
                surfaces=details,pairs=pairs,minimum_gap_lower_bound_mm=min(p['conservative_gap_lower_bound_mm'] for p in pairs),
                all_optical_surfaces_and_detector_disjoint_certified=all(p['disjoint_certified'] for p in pairs),
                limitation='Analytic covering bound with floating-point arithmetic, not outward-rounded interval arithmetic. Applies to zero-thickness optical surfaces and active detector rectangle; excludes substrates, mounts and detector housing.')


def main():
    import argparse
    import json
    from pathlib import Path
    from . import four_mirror as fold
    from .search import write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fields',type=int,default=8)
    parser.add_argument('--waves',type=int,default=9)
    parser.add_argument('--radial-samples',type=int,default=7)
    parser.add_argument('--azimuth-samples',type=int,default=40)
    args=parser.parse_args();row=json.loads(args.result.read_text())
    baseline=fold.load_baseline(row['configuration']['source'])
    layout=fold.build_layout(row['best']['params'],baseline)
    # Use persisted manufactured dimensions, even if training sampling changes.
    for surface,saved in zip(layout['surfaces'],row['layout']['surfaces']):surface['aperture']=saved['aperture']
    layout['detector_bounds']=np.asarray(row['layout']['detector_bounds'])
    result=audit_layout(layout,baseline,args.fields,args.waves,args.radial_samples,args.azimuth_samples)
    result['params']=row['best']['params'];result['source_result']=str(args.result)
    result['covered_surface_separation']=covered_surface_separation(layout)
    sampling=dict(fields=args.fields,waves=args.waves,pupil_xy=dense_pupil(args.radial_samples,args.azimuth_samples))
    result['baseline_image_metrics']=image_metrics(model.trace(baseline['genes'],aperture_radii=baseline['aperture_radii'],**sampling))
    result['folded_image_metrics']=image_metrics(fold.trace(layout,baseline,intersect_fn=exact_intersect,**sampling))
    result['wavefront_comparison_note']='The raw centroid-reference WFE estimator uses distance from the last optical surface, which changes when F2 is inserted. Raw WFE differences do not represent an optical gain or loss. The exact isometry preserves full ray path, grating phase and image coordinates; baseline wavefront performance is inherited under the corresponding unfolded reference construction.'
    result['curvature_note']='The three powered surfaces retain baseline coefficients and aperture radii under orthogonal isometries; added flat folds have zero curvature.'
    write_json(args.out,result)
    print(json.dumps({k:result[k] for k in ('passed','launched_rays','polynomial_received_fraction','maximum_image_coordinate_error_mm')},indent=2))


def audit_trace_pair(nominal, exact, include_surface_certificate=True):
    """Validate powered-layout traces without assuming baseline image invariance.

    The caller must generate `exact` using exact_intersect with every physical
    aperture and detector bound fixed before choosing heldout sample points.
    """
    good=exact['good'];common=good&nominal['good'];layout=exact['model']
    paths=audit_paths(exact['paths'],layout['surfaces'],wavelengths=exact['wavelength'])
    detector=detector_obstructions(exact['paths'],layout)
    xy_error=float(np.max(np.abs(exact['xy'][common]-nominal['xy'][common]))) if common.any() else None
    disagreement=int(np.sum(good!=nominal['good']))
    result=dict(launched_rays=int(good.size),polynomial_received_fraction=float(good.mean()),
        newton_received_fraction=float(nominal['good'].mean()),good_mask_disagreements=disagreement,
        maximum_common_image_coordinate_disagreement_mm=xy_error,
        independent_paths=paths,early_detector_interceptions=detector,
        image_metrics=image_metrics(exact),fixed_aperture_radii_mm=[float(s['aperture']) for s in layout['surfaces']],
        fixed_detector_bounds_mm=np.asarray(layout['detector_bounds']).tolist(),
        heldout_aperture_edge_margins_mm=[float(s['aperture']-np.linalg.norm(((exact['paths'][i+1]-s['vertex'])@s['basis'])[:,:2],axis=1).max()) for i,s in enumerate(layout['surfaces'])],
        passed=bool(good.all() and disagreement==0 and paths['accepted_rays']==good.size and detector['blocked_rays']==0 and xy_error is not None and xy_error<1e-7),
        limitation='Checks sampled powered-layout rays and physical laws. Does not assume image/OPL invariance relative to the old prescription; no fabrication or mount certification.')
    if include_surface_certificate:result['covered_surface_separation']=covered_surface_separation(layout)
    return result


def powered_curvature_audit(layout, radial_samples=65, azimuth_samples=360):
    """Independent monomial Hessians on full physical disks including dense rims."""
    pupil=dense_pupil(radial_samples,azimuth_samples);rows=[]
    for index,surface in enumerate(layout['surfaces']):
        x,y=(pupil*surface['aperture']).T;h=np.zeros((len(x),2,2))
        for c,(i,j) in zip(surface['coeff'],model.TERMS):
            if i>=2:h[:,0,0]+=c*i*(i-1)*x**(i-2)*y**j
            if i and j:h[:,0,1]+=c*i*j*x**(i-1)*y**(j-1)
            if j>=2:h[:,1,1]+=c*j*(j-1)*x**i*y**(j-2)
        h[:,1,0]=h[:,0,1];eigen=np.linalg.eigvalsh(h)
        minimum=float(eigen.min());maximum=float(eigen.max())
        expected='positive' if index==2 else 'negative' if index in (1,3) else 'unconstrained'
        preserved=minimum>0 if expected=='positive' else maximum<0 if expected=='negative' else True
        rows.append(dict(surface=index,aperture_radius_mm=float(surface['aperture']),sample_count=len(x),
            minimum_hessian_eigenvalue_per_mm=minimum,maximum_hessian_eigenvalue_per_mm=maximum,
            expected_sign=expected,expected_sign_preserved=bool(preserved),
            minimum_expected_signed_hessian_per_mm=(minimum if expected=='positive' else -maximum if expected=='negative' else None),
            target_2e_minus6_margin_deficit_per_mm=(max(2e-6-(minimum if expected=='positive' else -maximum),0.) if expected!='unconstrained' else None),
            nonzero_surface_power=bool(np.max(np.abs(eigen))>1e-10),
            vertex_x_curvature_per_mm=float(2*surface['coeff'][0]),vertex_y_curvature_per_mm=float(2*surface['coeff'][1])))
    return dict(radial_samples=radial_samples,azimuth_samples=azimuth_samples,surfaces=rows,
        all_expected_signs_preserved=all(r['expected_sign_preserved'] for r in rows),
        all_four_mirrors_powered=all(rows[i]['nonzero_surface_power'] for i in (0,1,3,4)),
        limitation='Dense graph-Hessian sign sampling, including complete physical aperture rims; not analytic curvature certification. Added mirror curvature signs are unrestricted.')


def audit_powered_genes(genes):
    """Dense Newton plus independent moderate exact-root audit of fixed hardware."""
    from . import powered_four_mirror as powered
    dense=powered.trace(genes,fields=13,waves=17,pupil_xy=dense_pupil(5,44))
    result=dict(genes=np.asarray(genes).tolist(),dense_newton_sampling=dict(fields=13,waves=17,radial_samples=5,azimuth_samples=44,pupil_points=177),
                dense_newton_launched_rays=int(dense['good'].size),dense_newton_received_rays=int(dense['good'].sum()),
                dense_newton_image_metrics=image_metrics(dense),
                powered_curvature=powered_curvature_audit(dense['model']),
                covered_surface_separation=covered_surface_separation(dense['model']))
    failure_reasons=[]
    if not dense['good'].all():failure_reasons.append('Dense Newton rays lost')
    if not result['powered_curvature']['all_expected_signs_preserved']:failure_reasons.append('Inherited curvature sign failed on full physical disk')
    if not result['powered_curvature']['all_four_mirrors_powered']:failure_reasons.append('Not all four mirrors have nonzero power')
    if not result['covered_surface_separation']['all_optical_surfaces_and_detector_disjoint_certified']:failure_reasons.append('Positive optical-surface separation not certified')
    if failure_reasons:
        result.update(passed=False,reason='; '.join(failure_reasons)+'. Expensive full independent ray audit deferred.')
        return result
    sampling=dict(fields=5,waves=7,pupil_xy=dense_pupil(5,24))
    nominal=powered.trace(genes,**sampling)
    exact=powered.trace(genes,intersect_fn=exact_intersect,**sampling)
    result['polynomial_sampling']=dict(fields=5,waves=7,radial_samples=5,azimuth_samples=24,pupil_points=97)
    result['independent_polynomial_audit']=audit_trace_pair(nominal,exact,include_surface_certificate=False)
    curvature=result['powered_curvature']
    result['passed']=bool(result['independent_polynomial_audit']['passed'] and curvature['all_four_mirrors_powered'] and
                          curvature['all_expected_signs_preserved'] and result['covered_surface_separation']['all_optical_surfaces_and_detector_disjoint_certified'])
    result['limitation']='Geometric sampled-ray and optical-surface verification only; no image invariance assumption, diffraction PSF, throughput loss from coatings, fabrication tolerances, mounts or substrate envelopes.'
    return result


if __name__=='__main__':main()
