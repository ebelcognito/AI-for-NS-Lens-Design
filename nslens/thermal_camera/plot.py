"""Render a saved thermal-camera candidate and its ray-trace diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import model
from . import freeform_model
from . import spectrometer_style_model


def _backend(result, genes):
    representation = result.get("representation")
    if representation == "spectrometer_style_xy":
        return spectrometer_style_model
    if representation == "xy_freeform" or len(genes) > model.N_TOTAL:
        return freeform_model
    return model


def _surface_profile(ax, optical, surface, radius, color, label, hole=0.0):
    z0, curvature, conic, coeff = surface
    x = np.linspace(-radius, radius, 501)
    sag, _, _ = optical.sag_xy(x, np.zeros_like(x), curvature, conic, coeff)
    if hole <= 0:
        ax.plot(x, z0+sag, color=color, lw=2, label=label)
    else:
        left, right = x <= -hole, x >= hole
        ax.plot(x[left], (z0+sag)[left], color=color, lw=2, label=label)
        ax.plot(x[right], (z0+sag)[right], color=color, lw=2)


def _representative_fields(fields_deg, count=9):
    fields_deg = np.asarray(fields_deg)
    if len(fields_deg) <= count:
        return np.arange(len(fields_deg))
    xs = np.linspace(fields_deg[:, 0].min(), fields_deg[:, 0].max(), 3)
    ys = np.linspace(fields_deg[:, 1].min(), fields_deg[:, 1].max(), 3)
    selected = []
    for y in ys:
        for x in xs:
            idx = int(np.argmin(np.sum((fields_deg-[x, y])**2, axis=1)))
            if idx not in selected:
                selected.append(idx)
    if len(selected) < min(count, len(fields_deg)):
        selected.extend(i for i in range(len(fields_deg)) if i not in selected)
    return np.asarray(selected[:count], dtype=int)


def render(path):
    path = Path(path)
    result = json.loads(path.read_text())
    genes = np.asarray(result["genes"], dtype=float)
    optical = _backend(result, genes)
    sampling = result.get("audit_sampling", {"fields": 9, "waves": 1, "pupils": 9})
    trace = optical.trace(genes, **sampling)
    decoded = trace["model"]
    _, nw, _ = trace["good"].shape

    fig = plt.figure(figsize=(16, 13), layout="constrained")
    grid = fig.add_gridspec(4, 3, height_ratios=(1.1, 1, 1, 1))
    layout_ax = fig.add_subplot(grid[0, 0])
    coverage_ax = fig.add_subplot(grid[0, 1])
    history_ax = fig.add_subplot(grid[0, 2])

    paths = trace["paths"]
    ids = np.flatnonzero(trace["good"].reshape(-1))
    if len(ids) > 250:
        ids = ids[np.linspace(0, len(ids)-1, 250, dtype=int)]
    for idx in ids:
        ray = paths[:, idx]
        layout_ax.plot(ray[:, 0], ray[:, 2], color="tab:blue", alpha=.18, lw=.5)
    _surface_profile(layout_ax, optical, decoded["plate"], decoded["primary_radius"], "tab:green", "corrector")
    _surface_profile(layout_ax, optical, decoded["primary"], decoded["primary_radius"], "black", "primary",
                     hole=decoded["hole_radius"])
    _surface_profile(layout_ax, optical, decoded["secondary"], decoded["secondary_radius"], "tab:orange", "secondary")
    detx = np.array([-decoded["hole_radius"], decoded["hole_radius"]])
    layout_ax.plot(detx, np.full(2, decoded["detector_z"]), color="magenta", lw=2, label="detector")
    layout_ax.set(title="Ray-path x–z projection (surfaces at y=0)", xlabel="x (mm)", ylabel="z (mm)")
    layout_ax.set_aspect("equal", adjustable="datalim")
    layout_ax.grid(alpha=.2)
    layout_ax.legend(fontsize=7, loc="best")

    for cy in (-3.25, 3.25):
        for cx in (-12.75, -4.25, 4.25, 12.75):
            coverage_ax.add_patch(plt.Rectangle((cx-4.5, cy-3.5), 9, 7,
                                   facecolor="tab:blue", alpha=.16,
                                   edgecolor="tab:blue", lw=1.2))
    coverage_ax.add_patch(plt.Rectangle((-17, -6.5), 34, 13, fill=False,
                           edgecolor="black", lw=2, linestyle="--"))
    coverage_ax.set(xlim=(-18, 18), ylim=(-8, 8), aspect="equal",
                    title="Idealized 2×4 channel coverage", xlabel="field x (deg)", ylabel="field y (deg)")
    coverage_ax.grid(alpha=.2)
    coverage_ax.text(.02, .02, "34° × 13° target\nSteering is not ray-traced.",
                     transform=coverage_ax.transAxes, fontsize=8, va="bottom")

    history = result.get("history", [])
    if history:
        history_ax.plot([r["evaluations"] for r in history], [r["best_loss"] for r in history],
                        color="tab:purple")
        history_ax.set_yscale("log")
    history_ax.set(title="Best training loss", xlabel="actual model evaluations", ylabel="loss")
    history_ax.grid(alpha=.2)

    fields = np.asarray(trace["fields_deg"])
    colors = plt.cm.viridis(np.linspace(.1, .9, nw))
    for panel, fi in enumerate(_representative_fields(fields)):
        row, col = divmod(panel, 3)
        ax = fig.add_subplot(grid[row+1, col])
        for wi in range(nw):
            mask = trace["good"][fi, wi]
            points = trace["xy"][fi, wi, mask]
            if len(points):
                points = (points-points.mean(axis=0))*1000
                ax.scatter(points[:, 0], points[:, 1], s=8, alpha=.55, color=colors[wi],
                           label=f"{trace['wavelengths_mm'][wi]*1000:.0f} µm")
        fxy = fields[fi]
        ax.set(title=f"Field ({fxy[0]:.1f}°, {fxy[1]:.1f}°)",
               xlabel="centred x (µm)", ylabel="centred y (µm)")
        ax.grid(alpha=.2)
        ax.set_aspect("equal", adjustable="datalim")
        if panel == 0:
            ax.legend(fontsize=7, loc="best")

    audit = result.get("audit", {})
    method = "least-squares polish" if 'solver_residual_calls' in result else "CMA candidate"
    count_label = (f"{result['solver_residual_calls']} solver residual calls"
                   if 'solver_residual_calls' in result else f"{result.get('evaluations')} training evaluations")
    fig.suptitle(f"Thermal-camera {method} | {result.get('representation', 'radial')} | "
                 f"seed {result.get('seed')} | {count_label} | "
                 f"audit spot RMS {audit.get('rms_spot_mm', float('nan')):.4g} mm | "
                 f"throughput {audit.get('throughput', float('nan')):.3f}\n"
                 "Assumed 12 µm pitch; optical depth target is 10 mm; not packaged or certified.", fontsize=12)
    target = path.with_name("candidate.png")
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="path to a seed result.json")
    render(parser.parse_args(argv).result)


if __name__ == "__main__":
    main()


