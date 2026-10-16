"""Plot XY surface departures and stage-by-stage dense audit metrics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import freeform_model
from . import spectrometer_style_model


def _backend(result):
    if result.get('representation') == 'spectrometer_style_xy':
        return spectrometer_style_model
    # Preserve interpretation of historical 50-gene XY artifacts. The
    # zero-padded radial control also displays as zero XY departure here.
    return freeform_model


def render(path, control=None):
    path = Path(path)
    result = json.loads(path.read_text())
    model = _backend(result)
    decoded = model.decode(result['genes'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, name in zip(axes, model.SURFACES):
        radius = decoded['secondary_radius'] if name == 'secondary' else decoded['primary_radius']
        q = np.linspace(-radius, radius, 151)
        x, y = np.meshgrid(q, q)
        z, c, k, coeff = decoded[name]
        sag = model.sag_xy(x, y, c, k, coeff)[0]
        base = model._radial(np.hypot(x, y), c, k, coeff['radial'])[0]
        departure = (sag-base)*1000
        mask = x*x+y*y > radius*radius
        if name == 'primary':
            mask |= x*x+y*y < decoded['hole_radius']**2
        departure[mask] = np.nan
        extent = max(float(np.nanmax(abs(departure))), 1e-9)
        im = ax.imshow(departure, origin='lower', extent=(-radius,radius,-radius,radius),
                       cmap='coolwarm', vmin=-extent, vmax=extent)
        fig.colorbar(im, ax=ax, label='XY departure (µm)')
        ax.set(title=name, xlabel='x (mm)', ylabel='y (mm)')
    fig.suptitle(f"XY departures from radial surface terms — {result.get('representation', 'freeform')}")
    fig.savefig(path.with_name('freeform_surfaces.png'), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    runs = [('XY freeform', result)]
    if control:
        runs.append(('Radial control', json.loads(Path(control).read_text())))
    stage_labels = None
    for label, data in runs:
        stages = data['stages'] or [dict(name='least-squares', audit=data['audit'])]
        rows = [data['source_audit']]+[stage['audit'] for stage in stages]
        labels = ['source']+[stage.get('name', f"stage {i+1}")
                             for i, stage in enumerate(stages)]
        stage_labels = labels
        xvals = np.arange(len(rows))
        for ax, key in zip(axes, ('rms_spot_pixels', 'max_spot_mm', 'min_throughput')):
            values = [r[key]/.012 if key == 'max_spot_mm' else r[key] for r in rows]
            ax.plot(xvals, values, '-o', label=label)
            ax.set_xticks(xvals, labels, rotation=15, ha='right')
            ax.grid(alpha=.2)
    for ax, title in zip(axes, ('RMS spot (pixels)', 'Worst field RMS (pixels)', 'Minimum pupil support')):
        ax.set_title(title)
        ax.legend()
    audit_cfg = result.get('audit_sampling', {})
    fields = int(audit_cfg.get('fields', 7))
    pupils = int(audit_cfg.get('pupils', 13))
    fig.suptitle(f"Fixed dense audit: {fields}×{fields} field grid; {len(model._grid(pupils))} pupil rays/field; not training loss")
    fig.savefig(path.with_name('stage_comparison.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result')
    parser.add_argument('--control')
    args = parser.parse_args()
    render(args.result, args.control)


