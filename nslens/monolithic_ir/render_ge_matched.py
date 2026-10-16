"""Render saved monolithic IR prescriptions and genuine search states.

The source run's model snapshot is used when present, so geometry stays tied to
the physics version that produced the saved genes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import model as _package_model

model = _package_model
INK = "#172635"
MUTED = "#586576"
RAY = "#2475c2"
PATCHES = ("#77a8be", "#d8ad52", "#d8ad52", "#77a8be")
PATCH_LABELS = ("Entrance", "Mirror 1", "Mirror 2", "Exit")


def _load_model_snapshot(source_path):
    """Load the saved run's physics version before decoding its genes."""
    global model
    snapshot = Path(source_path).resolve().parent / "model_snapshot.py"
    if not snapshot.is_file():
        return None
    name = f"_nslens_monolithic_ir_snapshot_{abs(hash(str(snapshot)))}"
    spec = importlib.util.spec_from_file_location(name, snapshot)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model snapshot {snapshot}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    model = module
    return snapshot


def _load_json(path):
    return json.loads(Path(path).read_text())


def _genes(data):
    genes = np.asarray(data["genes"], dtype=float).reshape(-1)
    if genes.shape != (model.N_TOTAL,) or not np.isfinite(genes).all():
        raise ValueError(f"Expected {model.N_TOTAL} finite genes")
    return genes


def load_records(path):
    text = Path(path).read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    records = payload.get("trajectory", payload.get("records", [])) if isinstance(payload, dict) else payload
    if not records:
        raise ValueError("Trajectory contains no records")
    for record in records:
        _genes(record)
    return records


def _surface_mesh(surface, nr=21, nt=65):
    radius = np.linspace(0, surface.radius, nr)
    theta = np.linspace(0, 2*np.pi, nt)
    x = radius[:, None] * np.cos(theta)
    y = radius[:, None] * np.sin(theta)
    z = model.sag(surface, surface.center[0]+x, y)[0]
    return surface.center[0]+x, y, z


def _body_mesh(optical):
    x = np.linspace(-40, 40, 61)
    y = np.linspace(-16, 16, 33)
    xx, yy = np.meshgrid(x, y)
    front, back = model.body(optical, xx, yy)
    return xx, yy, front, back


def _spot_stats(tr):
    good, xy = tr["good"], tr["xy"]
    stats = []
    for fi in range(good.shape[0]):
        pts = xy[fi][good[fi]]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts):
            centered = (pts-pts.mean(axis=0))*1000
            rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        else:
            centered = np.empty((0, 2))
            rms = None
        stats.append((len(pts), rms, centered))
    return stats


def save_spot_maps(tr, destination):
    """Write a readable adaptive-scale 7×7 field plot for the saved design."""
    stats = _spot_stats(tr)
    side = int(round(np.sqrt(len(stats))))
    extent = max((float(np.max(np.abs(p))) for _, _, p in stats if len(p)), default=25.0)
    limit = max(30.0, float(np.ceil(extent*1.15/10)*10))
    fig, axes = plt.subplots(side, side, figsize=(15, 14), facecolor="white", squeeze=False)
    fields = tr["fields_deg"]
    for fi, (count, rms, points) in enumerate(stats):
        iy, ix = divmod(fi, side)
        ax = axes[iy, ix]
        if count:
            ax.scatter(points[:, 0], points[:, 1], s=7, color=RAY, alpha=.52, linewidths=0)
        ax.add_patch(Rectangle((-8.5, -8.5), 17, 17, fill=False,
                               edgecolor="#75458a", linewidth=.8, linestyle="--"))
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.grid(alpha=.16, linewidth=.45)
        ax.tick_params(labelsize=6, length=2)
        ax.set_title(f"x={fields[fi, 0]:g}°, y={fields[fi, 1]:g}°", fontsize=8, pad=2)
        rms_text = f"{rms:.1f} µm RMS" if rms is not None else "no spots"
        ax.text(.03, .97, f"{count}/{tr['good'].shape[1]*tr['good'].shape[2]} rays\n{rms_text}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=1))
        if iy == side-1:
            ax.set_xlabel("Δ detector x (µm)", fontsize=7)
        if ix == 0:
            ax.set_ylabel("Δ detector y (µm)", fontsize=7)
    fig.suptitle("Monolithic IR · all-field geometric spots", x=.04, ha="left",
                 fontsize=18, weight="bold", color=INK)
    fig.text(.04, .02, f"Centroid across all wavelengths · dashed square = 17 µm reference · adaptive ±{limit:g} µm axes",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.055, right=.99, bottom=.06, top=.93, wspace=.28, hspace=.34)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def render_one(record, index, total, destination):
    genes = _genes(record)
    tr = model.trace(genes, fields=7, waves=3, pupils=7)
    optical, good, paths = tr["model"], tr["good"], tr["paths"]
    nf, nw, npupil = good.shape
    side = int(round(np.sqrt(nf)))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.42, 1), left=.035, right=.98,
                            bottom=.105, top=.90, wspace=.14, hspace=.27)
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    mer = fig.add_subplot(grid[0, 1])
    spotgrid = grid[1, 1].subgridspec(side, side, wspace=.03, hspace=.03)
    spot_axes = np.array([[fig.add_subplot(spotgrid[i, j]) for j in range(side)] for i in range(side)])

    # Front and rear body faces are evaluated from the physical envelope model.
    xx, yy, front, back = _body_mesh(optical)
    ax.plot_surface(xx, yy, front, color="#cbd5e1", alpha=.16, linewidth=0, shade=False)
    ax.plot_surface(xx, yy, back, color="#cbd5e1", alpha=.16, linewidth=0, shade=False)
    ax.plot_surface(np.vstack((xx[0], xx[0])), np.vstack((yy[0], yy[0])),
                    np.vstack((front[0], back[0])), color="#cbd5e1", alpha=.12, linewidth=0)
    for k, surface in enumerate(optical.surfaces):
        x, y, z = _surface_mesh(surface)
        ax.plot_surface(x, y, z, color=PATCHES[k], alpha=.88, linewidth=0, shade=True)
        ax.plot(x[-1], y[-1], z[-1], color="#334155", lw=.55)
        ax.text(surface.center[0], surface.radius+.18, surface.center[2],
                PATCH_LABELS[k], color=INK, fontsize=8)
        mx = np.linspace(surface.center[0]-surface.radius, surface.center[0]+surface.radius, 121)
        mz = model.sag(surface, mx, np.zeros_like(mx))[0]
        mer.plot(mx, mz, color=PATCHES[k], lw=4.2)
    xm = np.linspace(-40, 40, 301)
    fm, bm = model.body(optical, xm, np.zeros_like(xm))
    mer.fill_between(xm, fm, bm, color="#cbd5e1", alpha=.3)
    mer.plot(xm, fm, color="#94a3b8", lw=.7)
    mer.plot(xm, bm, color="#94a3b8", lw=.7)

    # The detector is centered at its decoded global position and scaled to its panel.
    detector = optical.detector
    panel = np.array([[detector[0]-4.2, detector[1]-2.8, detector[2]],
                      [detector[0]+4.2, detector[1]-2.8, detector[2]],
                      [detector[0]+4.2, detector[1]+2.8, detector[2]],
                      [detector[0]-4.2, detector[1]+2.8, detector[2]]])
    ax.add_collection3d(Poly3DCollection([panel], facecolor="#ad84ba",
                                         edgecolor="#75458a", alpha=.36))
    mer.plot([detector[0]-4.2, detector[0]+4.2], [detector[2], detector[2]],
             color="#75458a", lw=2.3)
    ax.text(detector[0], detector[1]+1.8, detector[2], "Detector",
            color="#75458a", fontsize=9)

    # Fifteen sample rays: five pupil positions at three fields, central wavelength.
    chosen_fields = (0, nf//2, nf-1)
    chosen_pupil = np.linspace(0, npupil-1, 5, dtype=int)
    central_wave = nw//2
    for fi in chosen_fields:
        for pi in chosen_pupil:
            if not good[fi, central_wave, pi]:
                continue
            ray = paths[:, (fi*nw+central_wave)*npupil+pi]
            if not np.isfinite(ray).all():
                continue
            ax.plot(*ray.T, color=RAY, alpha=.48, lw=.9)
            mer.plot(ray[:, 0], ray[:, 2], color=RAY, alpha=.32, lw=.75)

    stats = _spot_stats(tr)
    limit_um = 500.0
    fields = tr["fields_deg"]
    for fi, (count, rms, points) in enumerate(stats):
        iy, ix = divmod(fi, side)
        a = spot_axes[iy, ix]
        if count:
            a.scatter(points[:, 0], points[:, 1], s=4, color=RAY, alpha=.48, linewidths=0)
        a.set(xlim=(-limit_um, limit_um), ylim=(-limit_um, limit_um))
        a.set_aspect("equal", adjustable="box")
        a.set_xticks([]); a.set_yticks([])
        for spine in a.spines.values():
            spine.set_color("#d8dee8"); spine.set_linewidth(.45)
        a.set_facecolor("#f8fafc" if count else "#fff7ed")
        if iy == 0:
            a.set_title(f"{fields[fi, 0]:g}°", fontsize=5, pad=1, color=MUTED)
        if ix == 0:
            a.set_ylabel(f"{fields[fi, 1]:g}°", fontsize=5, rotation=0,
                         labelpad=8, va="center", color=MUTED)

    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)",
           xlim=(-42, 42), ylim=(-18, 18), zlim=(-10, 34))
    ax.set_box_aspect((84, 36, 44))
    ax.view_init(elev=23, azim=-66)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(alpha=.13)
    ax.set_title("3D optical layout · fixed camera", loc="left", fontsize=12, pad=8)
    mer.set(xlabel="x (mm)", ylabel="z (mm)", xlim=(-42, 42), ylim=(-10, 34),
            title="Body and optical surfaces · x–z section")
    mer.set_aspect("equal", adjustable="box"); mer.grid(alpha=.18)
    mer.legend(handles=[Line2D([0], [0], color=RAY, lw=2, label="10 µm rays")],
               loc="upper right", fontsize=7, framealpha=.9)
    fig.suptitle(f"Monolithic IR evolution · {record.get('stage', 'saved design').replace('_',' ').title()}",
                 x=.04, ha="left", fontsize=19, weight="bold", color=INK)
    fig.text(.04, .925, "Ge F/1.4 · 58 mm internal chief path · sparse central-wavelength rays · geometric spots",
             fontsize=9, color=MUTED)
    fig.text(.74, .485, "All-field spots · combined wavelengths · centroid removed",
             fontsize=8, color=MUTED, ha="center")
    fig.text(.04, .025,
             f"State {index+1}/{total} · {int(good.sum()):,}/{good.size:,} rays reach detector · fixed axes and camera",
             fontsize=9, color=MUTED)
    fig.savefig(destination, dpi=130)
    plt.close(fig)
    return {"frame": Path(destination).name, "stage": record.get("stage"),
            "evaluations": record.get("evaluations"), "survived": int(good.sum()),
            "launched": int(good.size)}


def _render_job(args):
    return render_one(*args)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path, help="Saved result.json or checkpoint.json")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--trajectory", type=Path, help="Optional trajectory.json or trajectory.jsonl")
    p.add_argument("--fps", type=float, default=2)
    p.add_argument("--limit", type=int, default=65, help="Evenly sample real trajectory states; 0 keeps all")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--still", action="store_true", help="Write frames without encoding a video")
    args = p.parse_args(argv)
    source_data = _load_json(args.source)
    snapshot = _load_model_snapshot(args.source)
    source_genes = _genes(source_data)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.trajectory:
        records = load_records(args.trajectory)
        # Keep only genuine saved states; never inject an endpoint from another file.
        if args.limit and len(records) > args.limit:
            keep = set(np.linspace(0, len(records)-1, args.limit, dtype=int).tolist())
            for i, row in enumerate(records):
                if i == 0 or i == len(records)-1 or row.get("stage") != records[i-1].get("stage"):
                    keep.add(i)
                if i+1 < len(records) and row.get("stage") != records[i+1].get("stage"):
                    keep.add(i)
            records = [records[i] for i in sorted(keep)]
        jobs = [(row, i, len(records), args.out / f"frame_{i:04d}.png")
                for i, row in enumerate(records)]
        metadata = []
        with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
            for i, item in enumerate(pool.map(_render_job, jobs)):
                metadata.append(item)
                print(f"{i+1}/{len(records)} {jobs[i][3]}", flush=True)
        if not args.still:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                            "-i", str(args.out / "frame_%04d.png"),
                            "-vf", "tpad=stop_mode=clone:stop_duration=2",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
                            str(args.out / "evolution.mp4")], check=True)
        (args.out / "frames.json").write_text(json.dumps({
            "source": str(args.source.resolve()),
            "trajectory": str(args.trajectory.resolve()),
            "model_snapshot": str(snapshot) if snapshot else None,
            "frames": metadata}, indent=2) + "\n")

    # Always provide final source stills, including when an evolution was requested.
    final_record = {"stage": "final saved result", "genes": source_genes.tolist()}
    render_one(final_record, 0, 1, args.out / "final_design.png")
    save_spot_maps(model.trace(source_genes, fields=7, waves=3, pupils=7),
                   args.out / "spot_maps.png")
    from .report_ge_matched import write_report
    write_report(args.out.parent)


if __name__ == "__main__":
    main()



