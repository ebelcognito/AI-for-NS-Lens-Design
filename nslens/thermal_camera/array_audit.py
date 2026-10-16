"""Finite steering-prism and eight-channel geometry audit."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from . import model as radial
from . import freeform_model as freeform
from . import spectrometer_style_model as spectrometer_style
from .search import write_json

CENTERS = np.array([(x, y) for y in (-3.25, 3.25) for x in (-12.75, -4.25, 4.25, 12.75)])
SENSOR_PIXELS = np.array([160, 120])


def steering_bundle(center, local_fields, pupils, waves=1):
    center, local_fields = np.asarray(center), np.asarray(local_fields)
    nf, npup, n = len(local_fields), len(pupils), radial.N_CORRECTOR
    chief = radial.unit(np.r_[np.tan(np.deg2rad(center)), 1.])
    inside, _ = radial.snell(chief, np.array([0., 0., 1.]), 1., n)
    normal = radial.unit(n*inside-np.array([0., 0., 1.]))
    slope = -normal[:2]/normal[2]
    radius = radial.APERTURE_DIAMETER_MM/2+.3
    extent = radius*np.linalg.norm(slope)
    exit_z, entrance_z = -.15-extent, -.30-2*extent
    global_fields = local_fields+center
    fld = np.repeat(global_fields, waves*npup, axis=0)
    incoming = radial.unit(np.column_stack([np.tan(np.deg2rad(fld)), np.ones(len(fld))]))
    inside, valid1 = radial.snell(incoming, np.array([0., 0., 1.]), 1., n)
    outgoing, valid2 = radial.snell(inside, normal, n, 1.)
    xy = np.tile(pupils*radial.APERTURE_DIAMETER_MM/2, (nf*waves, 1))
    at_stop = np.column_stack([xy, np.zeros(len(xy))])
    t = ((np.array([0., 0., exit_z])-at_stop)@normal)/(outgoing@normal)
    at_exit = at_stop+t[:, None]*outgoing
    u = (entrance_z-at_exit[:, 2])/inside[:, 2]
    at_entrance = at_exit+u[:, None]*inside
    valid = (valid1 & valid2 & (t < 0) & (u < 0)
             & (np.linalg.norm(at_exit[:, :2], axis=1) <= radius)
             & (np.linalg.norm(at_entrance[:, :2], axis=1) <= radius))
    return dict(entrance_o=at_exit, entrance_d=outgoing, valid=valid,
                prism=dict(index=n, radius_mm=radius, entrance_z_mm=entrance_z,
                           exit_center_z_mm=exit_z, exit_normal=normal.tolist(),
                           maximum_thickness_mm=2*extent+.15),
                global_fields_deg=global_fields, prism_paths=np.array([at_entrance, at_exit, at_stop]))


def _backend(result):
    genes = np.asarray(result['genes']).ravel()
    representation = result.get('representation')
    if representation == 'spectrometer_style_xy':
        return spectrometer_style
    if representation == 'xy_freeform' or len(genes) > radial.N_TOTAL:
        return freeform
    return radial


def run_audit(result_path, fields=7, pupils=13):
    path = Path(result_path)
    result = json.loads(path.read_text())
    genes = np.asarray(result['genes'], float).ravel()
    optical = _backend(result)
    decoded = optical.decode(genes)
    geometry = decoded.get('freeform_geometry')
    xx, yy = np.meshgrid(np.linspace(-4.25, 4.25, fields), np.linspace(-3.25, 3.25, fields))
    local = np.column_stack([xx.ravel(), yy.ravel()])
    pupil = radial._grid(pupils)
    half_sensor = SENSOR_PIXELS*radial.PIXEL_PITCH_MM/2
    rows = []
    for channel, center in enumerate(CENTERS):
        bundle = steering_bundle(center, local, pupil)
        rays = optical.trace(genes, waves=1, field_xy=local, pupil_xy=pupil,
                             entrance_o=bundle['entrance_o'], entrance_d=bundle['entrance_d'])
        good = rays['good'][:, 0] & bundle['valid'].reshape(len(local), len(pupil))
        xy = rays['xy'][:, 0]
        opt_count = good.sum(axis=1)
        opt_cent = np.sum(np.where(good[..., None], xy, 0), axis=1)/np.maximum(opt_count[:, None], 1)
        unclipped = np.sqrt(np.sum(np.where(good, np.sum((xy-opt_cent[:, None])**2, axis=-1), 0), axis=1)/np.maximum(opt_count, 1))
        on_sensor = np.all(abs(xy) <= half_sensor, axis=-1)
        clipped_fraction = float(np.mean(good & ~on_sensor))
        good &= on_sensor
        count = good.sum(axis=1)
        centroid = np.sum(np.where(good[..., None], xy, 0), axis=1)/np.maximum(count[:, None], 1)
        rms = np.sqrt(np.sum(np.where(good, np.sum((xy-centroid[:, None])**2, axis=-1), 0), axis=1)/np.maximum(count, 1))
        support = count/len(pupil)
        design = np.column_stack([local, np.ones(len(local))])
        affine = np.linalg.lstsq(design, centroid, rcond=None)[0]
        ppd = np.linalg.svd(affine[:2]/radial.PIXEL_PITCH_MM, compute_uv=False)
        maperr = float(np.sqrt(np.mean(np.sum((centroid-design@affine)**2, axis=1)))/radial.PIXEL_PITCH_MM)
        depth = float(decoded['detector_z']-bundle['prism']['entrance_z_mm'])
        rows.append(dict(channel=channel, center_deg=center.tolist(), prism=bundle['prism'],
                         fields_deg=bundle['global_fields_deg'].tolist(),
                         throughput_per_field=support.tolist(), spot_rms_mm_per_field=rms.tolist(),
                         centroids_mm=centroid.tolist(), optical_depth_with_prism_mm=depth,
                         unclipped_spot_rms_mm_per_field=unclipped.tolist(),
                         max_unclipped_spot_pixels=float(unclipped.max()/radial.PIXEL_PITCH_MM),
                         sensor_clipped_fraction=clipped_fraction, fitted_pixels_per_degree=ppd.tolist(),
                         affine_mapping_residual_pixels=maperr,
                         area_equivalent_f_number_nominal=1.1/np.sqrt(float(support.mean())) if support.mean()>0 else None,
                         min_throughput=float(support.min()), mean_throughput=float(support.mean()),
                         max_spot_pixels=float(rms.max()/radial.PIXEL_PITCH_MM),
                         rms_spot_pixels=float(np.sqrt(np.mean(rms*rms))/radial.PIXEL_PITCH_MM),
                         empty_fields=int(np.sum(count == 0))))
    worst = max(row['max_unclipped_spot_pixels'] for row in rows)
    min_support = min(row['min_throughput'] for row in rows)
    max_depth = max(row['optical_depth_with_prism_mm'] for row in rows)
    gates = dict(all_fields_have_rays=all(row['empty_fields']==0 for row in rows),
                 min_geometric_support_at_least_0_5=min_support>=.5,
                 worst_geometric_rms_at_most_one_assumed_pixel=worst<=1.,
                 optical_depth_including_prisms_at_most_10mm=max_depth<=10.,
                 angular_sampling_within_ten_percent_of_16=all(14.4<=v<=17.6 for row in rows for v in row['fitted_pixels_per_degree']))
    if geometry is not None:
        s = geometry['surfaces']
        geometry_ok = (geometry['minimum_plate_thickness_mm']>.05
                       and geometry['minimum_secondary_plate_gap_mm']>.05
                       and geometry['minimum_primary_secondary_gap_mm']>.05
                       and s['plate']['max_slope']<=.3
                       and s['primary']['max_slope']<=1.
                       and s['secondary']['max_slope']<=1.)
        gates['sampled_freeform_geometry_valid'] = bool(geometry_ok)
    report = dict(target='2024 Fraunhofer flat camera; reconstruction pilot',
                  backend=result.get('representation', 'xy_freeform' if geometry is not None else 'radial'),
                  sampling=dict(fields_per_channel=fields**2, pupil_rays=len(pupil), wavelength_mm=.011,
                                spectral_geometry='constant index; no dispersion validation'),
                  assumed_sensor_pixels=SENSOR_PIXELS.tolist(), assumed_pixel_pitch_mm=radial.PIXEL_PITCH_MM,
                  field_extent_deg=[34., 13.], channels=rows, gates=gates,
                  geometric_pilot_pass=all(gates.values()), exact_reproduction=False,
                  maximum_optical_depth_mm=max_depth, minimum_geometric_support=min_support,
                  worst_geometric_rms_pixels=worst, sampled_freeform_geometry=geometry,
                  limitations=['No diffraction PSF/MTF, coatings, absorption, temperature or tolerances',
                               'No native detector specification or measured prescription available',
                               'No mechanical supports, vacuum package, electronics or crosstalk model',
                               'Stop-plane quadrature after prism; support is not radiometric efficiency',
                               'Fitted angular sampling does not establish diffraction-limited resolved pixel count',
                               'Sampled surface slope and gaps are not certified bounds'])
    write_json(path.with_name('array_audit.json'), report)
    render(report, path.with_name('array_coverage.png'))
    print(json.dumps({k:report[k] for k in ('backend','gates','geometric_pilot_pass','maximum_optical_depth_mm','minimum_geometric_support','worst_geometric_rms_pixels')}, indent=2))
    return report


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    for key, ax, label, limits in zip(('throughput_per_field','spot_rms_mm_per_field'), axes,
                                       ('Geometric pupil support','Geometric RMS spot / 12 µm pixel'),
                                       ((0,1),(0,max(1.,report['worst_geometric_rms_pixels'])))):
        for row in report['channels']:
            field = np.array(row['fields_deg'])
            val = np.array(row[key])/(radial.PIXEL_PITCH_MM if key.startswith('spot') else 1)
            artist = ax.scatter(*field.T, c=val, marker='s', s=38, vmin=limits[0], vmax=limits[1], cmap='viridis')
        ax.set(xlabel='Horizontal field (deg)', ylabel='Vertical field (deg)', title=label, aspect='equal')
        fig.colorbar(artist, ax=ax, shrink=.7)
    fig.suptitle('Eight channels with finite refracting steering prisms — geometric audit, assumed sensor')
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result'); p.add_argument('--fields',type=int,default=7); p.add_argument('--pupils',type=int,default=13)
    a=p.parse_args(); run_audit(a.result,a.fields,a.pupils)
