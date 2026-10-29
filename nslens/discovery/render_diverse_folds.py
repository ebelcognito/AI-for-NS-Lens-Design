"""Render a configurable quality-diverse gallery from one discovery run.

Candidates are re-evaluated with the run's dense audit settings.  Only passing
designs are eligible; selection uses the same normalized-gene farthest-first
rule as the retained-candidate proposer, applied to the best 2*N designs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .folded import RoutingBrief, evaluate
from .polygon import PolygonCodec
from .portfolio import select_diverse


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualifying_path(ray, source, brief):
    tir = [event for event in ray.events if event.kind == "tir"]
    if (ray.direction[0] <= 0 or len(tir) < brief.min_tir
            or any(event.critical_margin_deg < brief.min_tir_margin_deg for event in tir)):
        return None
    distance = (brief.detector_x - ray.origin[0]) / ray.direction[0]
    end = np.asarray(ray.origin) + distance * np.asarray(ray.direction)
    if distance <= 0 or abs(end[1] - brief.detector_center_y) > brief.detector_half_height:
        return None
    points = np.asarray([source, *(event.point for event in ray.events), end])
    tir_points = np.asarray([event.point for event in tir])
    return points, tir_points


def render(
    run: Path,
    output: Path,
    count: int = 16,
    rows: int = 4,
    columns: int = 4,
) -> dict:
    if min(count, rows, columns) < 1 or count > rows * columns:
        raise ValueError("count must be positive and fit within rows * columns")
    summary = json.loads((run / "summary.json").read_text())
    codec = PolygonCodec(**summary["config"]["codec"])
    brief = RoutingBrief(**summary["config"]["brief"])
    fields = sorted(brief.fields_deg)
    dense = replace(
        brief,
        rays_per_field=summary["config"]["audit_rays"],
        fields_deg=tuple(sorted(set(fields + [(a + b) / 2 for a, b in zip(fields, fields[1:])]))),
    )

    candidates_path = run / "candidates.jsonl"
    coarse = [json.loads(line) for line in candidates_path.read_text().splitlines()]
    coarse = [row for row in coarse if row.get("feasible")]
    verified = []
    for source_row in coarse:
        result = evaluate(codec, source_row["genes"], dense, geometry_samples=1024)
        if result["feasible"]:
            result["coarse_evaluation"] = source_row["evaluation"]
            verified.append(result)
    if len(verified) < count:
        raise RuntimeError(f"only {len(verified)} dense-verified designs; need {count}")

    selected = select_diverse(verified, count)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.5 * columns, 4 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    manifest_rows = []
    for number, (ax, choice) in enumerate(zip(axes.flat, selected), 1):
        result, body, trace = evaluate(
            codec, choice["genes"], dense, return_trace=True, geometry_samples=1024
        )
        if not result["feasible"]:
            raise RuntimeError(f"selected design {number} failed repeated replay")

        boundary = body.sample(1200)[1]
        boundary = np.vstack((boundary, boundary[0]))
        ax.fill(boundary[:, 0], boundary[:, 1], color="#d7eafa", alpha=.88)
        ax.plot(boundary[:, 0], boundary[:, 1], color="#153d56", lw=1.35)
        ax.scatter(body.corners[:, 0], body.corners[:, 1], color="#153d56", s=10, zorder=4)

        rays = dense.rays()
        central = min(range(len(dense.fields_deg)), key=lambda i: abs(dense.fields_deg[i]))
        source_ids = np.arange(
            central * dense.rays_per_field, (central + 1) * dense.rays_per_field
        )
        shown = 0
        for source_id in source_ids[::3]:
            paths = []
            for ray in trace.exits:
                if ray.source_id != source_id:
                    continue
                path = _qualifying_path(ray, rays[source_id].origin, dense)
                if path is not None:
                    paths.append((ray.energy, *path))
            if not paths:
                continue
            _, points, tir_points = max(paths, key=lambda item: item[0])
            ax.plot(points[:, 0], points[:, 1], color="#d4642b", lw=.8, alpha=.82)
            ax.scatter(tir_points[:, 0], tir_points[:, 1], color="#b13f1e", s=7, zorder=5)
            shown += 1

        origins = np.asarray([r.origin for r in rays[central*dense.rays_per_field:(central+1)*dense.rays_per_field]])
        ax.plot(origins[:, 0], origins[:, 1], color="#11946c", lw=3)
        ax.plot(
            [dense.detector_x] * 2,
            [dense.detector_center_y - dense.detector_half_height,
             dense.detector_center_y + dense.detector_half_height],
            color="#7958a3", lw=2.5,
        )
        ax.set_title(
            f"#{number:02d}  Q {result['quality']:.1%}  support {result['support']:.0%}",
            fontsize=9,
        )
        ax.set_aspect("equal")
        ax.grid(alpha=.12)
        ax.tick_params(labelsize=7)
        manifest_rows.append({
            "panel": number,
            "coarse_evaluation": choice["coarse_evaluation"],
            "displayed_central_field_paths": shown,
            "genes": result["genes"],
            "dense_result": result,
        })

    for ax in axes.flat[count:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("x (mm)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (mm)", fontsize=8)
    axes[0, 0].set_xlim(20, dense.detector_x + 5)
    axes[0, 0].set_ylim(-35, 55)
    fig.suptitle(
        "Best quality-diverse folded designs from one proposer run\n"
        f"{codec.corners} corners · {dense.beam_width_mm:g} mm beam · dense verified; "
        "orange paths include ≥2 TIR folds",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, .955))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)

    genes = np.asarray([row["genes"] for row in selected])
    pairwise = np.sqrt(np.mean((genes[:, None, :] - genes[None, :, :]) ** 2, axis=2))
    nonzero = pairwise[np.triu_indices(count, 1)]
    manifest = {
        "source_run": str(run.resolve()),
        "source_candidates_sha256": _sha256(candidates_path),
        "renderer_sha256": _sha256(Path(__file__)),
        "selection": f"dense-pass only; top {2*count} by quality, then normalized-gene farthest-first",
        "dense_brief": dense.__dict__,
        "eligible_dense_verified": len(verified),
        "selected": count,
        "layout": {"rows": rows, "columns": columns},
        "selected_quality_range": [min(r["quality"] for r in selected), max(r["quality"] for r in selected)],
        "selected_pairwise_gene_rms": {
            "min": float(nonzero.min()), "median": float(np.median(nonzero)), "max": float(nonzero.max())
        },
        "panels": manifest_rows,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    result = render(
        args.run,
        args.output,
        count=args.rows * args.columns,
        rows=args.rows,
        columns=args.columns,
    )
    print(f"wrote {args.output} ({result['selected']} of {result['eligible_dense_verified']} verified)")


if __name__ == "__main__":
    main()
