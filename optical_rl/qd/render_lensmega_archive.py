"""Generic renderer for any lensmega QD archive.

Works with any lens type -- EF50, Cooke triplet, doublet, etc. -- by reading
the run's config.yaml and using lensmega's own LensTemplate +
GeneralizedParaxialSolver to map theta -> lens.

Usage:
  python -m optical_rl.qd.render_lensmega_archive --archive path/to/archive_cells.json
  python -m optical_rl.qd.render_lensmega_archive --archive path/to/run_dir/
  python -m optical_rl.qd.render_lensmega_archive --count 12 --grid-cols 4
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEEPLENS_ROOT = PROJECT_ROOT / "DeepLens"
for _p in [str(PROJECT_ROOT), str(_DEEPLENS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optical_rl.qd.lensmega.config.loader import load_config as load_lensmega_config
from optical_rl.qd.lensmega.config.schema import RunConfig
from optical_rl.qd.lensmega.problem.lens_template import LensTemplate, load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_singlet_archive_designs import (
    ArchiveDesign,
    PlotTheme,
    _compute_rms_spot,
    _lens_fill_polygons,
    _parse_fov_fracs,
    resolve_theme,
    select_diverse_top,
)
from optical_rl.qd.render_ef50_zernike import save_archive_with_selected_plot
from optical_rl.qd.singlet_zernike_measures import NOLL_TERM_NAMES

TAG = "render-lensmega"


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def _discover_config(archive_path: Path, explicit_config: Optional[str]) -> RunConfig:
    """Discover and load the lensmega RunConfig for a given archive.

    Resolution order:
      1. Explicit --config CLI argument
      2. config.yaml in archive_path's parent directory (RunPaths convention)
      3. Any *.yaml in archive_path's parent directory
    """
    paths_to_try: list[Path] = []
    if explicit_config:
        paths_to_try.append(Path(explicit_config))
    paths_to_try.append(archive_path.parent / "config.yaml")
    for p in sorted(archive_path.parent.glob("*.yaml")):
        if p not in paths_to_try:
            paths_to_try.append(p)

    errors: list[str] = []
    for p in paths_to_try:
        if not p.is_file():
            continue
        try:
            return load_lensmega_config(p)
        except Exception as exc:
            errors.append(f"  {p}: {exc}")

    msg = (
        f"Could not load a valid lensmega config for archive {archive_path}.\n"
        f"Tried: {[str(p) for p in paths_to_try]}"
    )
    if errors:
        msg += "\nErrors:\n" + "\n".join(errors)
    raise FileNotFoundError(msg)


def _resolve_lens_path(lens_path_str: str) -> Path:
    """Resolve a lens_path from config (may be relative to project root)."""
    p = Path(lens_path_str)
    if p.is_file():
        return p.resolve()
    candidate = PROJECT_ROOT / lens_path_str
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find lens template: tried '{lens_path_str}' and '{candidate}'"
    )


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------

def load_lensmega_archive(path: Path) -> list[ArchiveDesign]:
    """Load designs from a lensmega archive_cells.json file."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        rows = raw
    else:
        rows = raw.get("cells", [])

    out: list[ArchiveDesign] = []
    for row in rows:
        try:
            cell_raw = row.get("cell_index") or row.get("cell")
            cell = tuple(int(x) for x in cell_raw)
            fitness = float(row["fitness"])
            measures = tuple(float(x) for x in row["measures"])
            theta = np.asarray(row["theta"], dtype=np.float64).reshape(-1)
            metrics = row.get("metrics", {})
            rms_spot = metrics.get("rms_spot", None)
            if rms_spot is not None:
                rms_spot = float(rms_spot)
        except Exception:
            continue
        if not np.isfinite(fitness):
            continue
        if not all(np.isfinite(m) for m in measures):
            continue
        if not np.all(np.isfinite(theta)):
            continue
        if rms_spot is not None and not np.isfinite(rms_spot):
            rms_spot = None
        out.append(
            ArchiveDesign(
                cell=(cell[0], cell[1]),
                fitness=fitness,
                measures=(measures[0], measures[1]),
                theta=theta,
                rms_spot=rms_spot,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Archive auto-discovery
# ---------------------------------------------------------------------------

def _find_latest_archive() -> Path:
    roots = [
        PROJECT_ROOT / "results" / "lensmega",
        PROJECT_ROOT / "results" / "models",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.is_dir():
            candidates.extend(root.glob("**/archive_cells.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No archive_cells.json found under {[str(r) for r in roots]}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_archive_path(parts: list[str] | None) -> Path:
    """Resolve --archive to a single archive_cells.json file."""
    if not parts:
        return _find_latest_archive()
    candidates: list[Path] = []
    for part in parts:
        s = part.strip()
        if any(ch in s for ch in "*?[]"):
            for m in glob(s):
                p = Path(m).resolve()
                if p.is_file():
                    candidates.append(p)
        else:
            p = Path(s).resolve()
            if p.is_dir():
                archive_in_dir = p / "archive_cells.json"
                if archive_in_dir.is_file():
                    candidates.append(archive_in_dir)
                else:
                    raise FileNotFoundError(
                        f"Directory {s} does not contain archive_cells.json"
                    )
            elif p.is_file():
                candidates.append(p)
            else:
                raise FileNotFoundError(f"Archive not found: {s}")
    if not candidates:
        raise FileNotFoundError(f"No archive file matched {parts!r}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    if len(candidates) > 1:
        print(
            f"[{TAG}] {len(candidates)} archive(s) matched; using latest: {chosen}",
            flush=True,
        )
    return chosen


# ---------------------------------------------------------------------------
# Generic theta-to-lens applicator
# ---------------------------------------------------------------------------

def _power_budget_theta_to_curvatures(
    theta: torch.Tensor,
    template: LensTemplate,
    focal_length_target_mm: float,
) -> torch.Tensor:
    """Convert power-fraction theta to curvatures (for power_budget mode)."""
    P_total = 1.0 / focal_length_target_mm
    theta_sum = theta.sum()
    safe_sum = torch.where(
        torch.abs(theta_sum) > 1e-6, theta_sum, torch.full_like(theta_sum, 1e-6)
    )
    fractions = theta / safe_sum
    powers = fractions * P_total
    delta_ns = []
    for idx in template.free_indices:
        surf = template.surfaces[idx]
        delta_ns.append(surf.n_after - surf.n_before)
    delta_n_t = torch.tensor(delta_ns, dtype=theta.dtype, device=theta.device)
    return powers / delta_n_t


def _apply_theta_full_dof(
    lens,
    theta: np.ndarray,
    *,
    template: LensTemplate,
    free_curvature_indices: Sequence[int],
    free_gap_indices: Sequence[int],
    variable_flange: bool,
    variable_materials: bool = False,
    free_material_indices: Sequence[int] = (),
    device: str = "cpu",
) -> None:
    """Apply a full-DOF theta (curvatures + gaps + flange + materials) to a GeoLens."""
    theta_t = torch.tensor(theta, dtype=torch.float64, device=device)
    nc = len(free_curvature_indices)
    ng = len(free_gap_indices)
    curvatures = theta_t[:nc]
    gaps = theta_t[nc:nc + ng] if ng > 0 else None
    offset = nc + ng
    flange_val = theta_t[offset] if variable_flange else None
    if variable_flange:
        offset += 1
    n_mat = len(free_material_indices) if variable_materials else 0
    mat_params = theta_t[offset:offset + 2 * n_mat] if n_mat > 0 else None

    dtype = lens.d_sensor.dtype

    # Set curvatures.
    for i_local, i_global in enumerate(free_curvature_indices):
        lens.surfaces[i_global].c = torch.tensor(
            float(curvatures[i_local].item()), dtype=dtype, device=device,
        )

    # Recompute surface positions from gaps.
    d_pos = 0.0
    if gaps is not None:
        free_set = set(free_gap_indices)
        all_gaps = []
        cursor = 0
        for i in range(len(template.surfaces)):
            if i in free_set:
                all_gaps.append(float(gaps[cursor].item()))
                cursor += 1
            else:
                all_gaps.append(template.surfaces[i].gap_to_next)
        d_pos = 0.0
        for i in range(len(template.surfaces)):
            lens.surfaces[i].d = torch.tensor(d_pos, dtype=dtype, device=device)
            if i < len(template.surfaces) - 1:
                d_pos += all_gaps[i]

    # Set d_sensor: flange is relative to last surface position.
    if flange_val is not None:
        d_sensor = d_pos + float(flange_val.item())
        lens.d_sensor = torch.tensor(d_sensor, dtype=dtype, device=device)
    else:
        from .lensmega.problem.full_dof import FullDOFParaxialSolver
        fd_solver = FullDOFParaxialSolver(template)
        # Build c_full and gaps_full for BFL computation.
        c_parts = []
        free_c_set = set(free_curvature_indices)
        c_cursor = 0
        for i in range(len(template.surfaces)):
            if i in free_c_set:
                c_parts.append(curvatures[c_cursor])
                c_cursor += 1
            else:
                c_parts.append(torch.tensor(
                    template.surfaces[i].initial_c, dtype=torch.float64, device=device,
                ))
        c_full = torch.stack(c_parts)
        g_parts = []
        free_g_set = set(free_gap_indices)
        g_cursor = 0
        for i in range(len(template.surfaces)):
            if i in free_g_set:
                g_parts.append(gaps[g_cursor])
                g_cursor += 1
            else:
                g_parts.append(torch.tensor(
                    template.surfaces[i].gap_to_next, dtype=torch.float64, device=device,
                ))
        gaps_full = torch.stack(g_parts)
        bfl_t = fd_solver.bfl(c_full, gaps_full)
        z_last = sum(float(g) for g in g_parts[:-1])
        d_sensor = z_last + max(float(bfl_t.item()), 1.0)
        lens.d_sensor = torch.tensor(d_sensor, dtype=dtype, device=device)

    # Apply materials to GeoLens.
    if mat_params is not None:
        from deeplens.optics.material.materials import Material
        for mat_idx, elem_idx in enumerate(free_material_indices):
            n_d = float(mat_params[2 * mat_idx].item())
            V_d = float(mat_params[2 * mat_idx + 1].item())
            for si in template.element_groups[elem_idx]:
                s = lens.surfaces[si]
                if hasattr(s, "mat2") and s.mat2.name != "air":
                    s.mat2.n = n_d
                    s.mat2.V = V_d
                    A, B = Material.nV_to_AB(n_d, V_d)
                    s.mat2.A = A
                    s.mat2.B = B
                    s.mat2.dispersion = "cauchy"

    try:
        lens.post_computation()
    except Exception:
        pass


@torch.no_grad()
def _apply_theta_generic(
    lens,
    theta: np.ndarray,
    *,
    template: LensTemplate,
    solver: GeneralizedParaxialSolver,
    focal_length_target_mm: Optional[float],
    device: str = "cpu",
    parameterization: str = "curvature",
    free_curvature_indices: Optional[Sequence[int]] = None,
    free_gap_indices: Optional[Sequence[int]] = None,
    variable_flange: bool = False,
    variable_materials: bool = False,
    free_material_indices: Optional[Sequence[int]] = None,
) -> None:
    """Apply a theta vector to a GeoLens using lensmega's generalized solver.

    Supports curvature, power_budget, and full_dof parameterizations.
    """
    if parameterization == "full_dof":
        _apply_theta_full_dof(
            lens, theta,
            template=template,
            free_curvature_indices=free_curvature_indices or template.free_indices,
            free_gap_indices=free_gap_indices or [],
            variable_flange=variable_flange,
            variable_materials=variable_materials,
            free_material_indices=free_material_indices or [],
            device=device,
        )
        return

    theta_t = torch.tensor(theta, dtype=torch.float64, device=device)

    if parameterization == "power_budget" and focal_length_target_mm is not None:
        curvatures = _power_budget_theta_to_curvatures(
            theta_t, template, focal_length_target_mm
        )
        parts = []
        free_set = set(template.free_indices)
        cursor = 0
        for i in range(len(template.surfaces)):
            if i in free_set:
                parts.append(curvatures[cursor])
                cursor += 1
            else:
                parts.append(torch.tensor(
                    template.surfaces[i].initial_c,
                    dtype=theta_t.dtype, device=theta_t.device,
                ))
        c_full = torch.stack(parts)
        bfl_t = solver.bfl(c_full)
        d_sensor = template.z_last_surface_mm + max(float(bfl_t.item()), 1.0)
    else:
        c_full = solver.expand_theta(theta_t)
        if focal_length_target_mm is not None and template.derived_index is not None:
            c_derived, d_sensor, bfl_t = solver.derive_curvature_for_efl(
                c_full, focal_length_target_mm
            )
            c_full = c_full.clone()
            c_full[template.derived_index] = c_derived
        else:
            bfl_t = solver.bfl(c_full)
            d_sensor = template.z_last_surface_mm + max(float(bfl_t.item()), 1.0)

        if focal_length_target_mm is not None and template.derived_index is not None:
            bfl_val = float(bfl_t.item())
            if bfl_val < 1.0:
                d_sensor = template.z_last_surface_mm + 1.0

    dtype = lens.d_sensor.dtype
    for i in template.free_indices:
        lens.surfaces[i].c = torch.tensor(
            float(c_full[i].item()), dtype=dtype, device=device,
        )
    if template.derived_index is not None:
        lens.surfaces[template.derived_index].c = torch.tensor(
            float(c_full[template.derived_index].item()), dtype=dtype, device=device,
        )
    d_sensor_val = float(d_sensor.item()) if torch.is_tensor(d_sensor) else float(d_sensor)
    lens.d_sensor = torch.tensor(d_sensor_val, dtype=dtype, device=device)

    try:
        lens.post_computation()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Single-design renderer
# ---------------------------------------------------------------------------

def render_design_lensmega(
    *,
    lens_json_path: Path,
    design: ArchiveDesign,
    theme: PlotTheme,
    ax,
    template: LensTemplate,
    solver: GeneralizedParaxialSolver,
    focal_length_target_mm: Optional[float],
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs: Sequence[float],
    rank: int,
    device: str,
    free_curvature_indices: Sequence[int],
    title: Optional[str] = None,
    parameterization: str = "curvature",
    free_gap_indices: Optional[Sequence[int]] = None,
    variable_flange: bool = False,
    variable_materials: bool = False,
    free_material_indices: Optional[Sequence[int]] = None,
) -> dict[str, float]:
    from deeplens.optics import GeoLens

    lens = GeoLens(filename=str(lens_json_path), device=device)
    _apply_theta_generic(
        lens, design.theta,
        template=template,
        solver=solver,
        focal_length_target_mm=focal_length_target_mm,
        device=device,
        parameterization=parameterization,
        free_curvature_indices=free_curvature_indices,
        free_gap_indices=free_gap_indices,
        variable_flange=variable_flange,
        variable_materials=variable_materials,
        free_material_indices=free_material_indices,
    )

    ax.set_facecolor(theme.background)
    if theme.lens_fill is not None:
        for z_poly, x_poly in _lens_fill_polygons(lens):
            ax.fill(z_poly, x_poly, color=theme.lens_fill, alpha=1.0,
                    linewidth=0.0, zorder=1)

    lens.draw_lens_2d(ax=ax, fig=ax.figure, color=theme.lens_outline)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fov_values = [float(f) * rfov_deg for f in fov_fracs] or [0.0]

    for i, fov in enumerate(fov_values):
        ray = lens.sample_point_source_2D(
            fov=fov, depth=float(depth_mm),
            num_rays=int(num_rays), wvln=float(wavelength_um),
        )
        ray.prop_to(-1.0)
        _ray_out, ray_o_record = lens.trace2sensor(ray=ray, record=True)
        # draw_ray_2d uses hardcoded linewidth=0.8; re-style after drawing.
        ray_color = theme.ray_colors[i % len(theme.ray_colors)]
        n_lines_before = len(ax.lines)
        lens.draw_ray_2d(
            ray_o_record, ax=ax, fig=ax.figure,
            color=ray_color,
        )
        for line in ax.lines[n_lines_before:]:
            line.set_linewidth(1.8)

    spot = design.rms_spot
    if spot is None:
        spot = _compute_rms_spot(
            lens=lens, depth_mm=float(depth_mm),
            wavelength_um=float(wavelength_um), num_rays=int(num_rays),
        )

    foclen = float(getattr(lens, "foclen", float("nan")))

    if title is not None:
        ax.set_title(title, fontsize=8)
    else:
        theta_arr = np.asarray(design.theta, dtype=np.float64).reshape(-1)
        theta_str = "  ".join(
            f"c{free_curvature_indices[k]}={float(theta_arr[k]):.4f}"
            for k in range(min(len(theta_arr), len(free_curvature_indices)))
        )
        efl_note = (
            f"  EFL={focal_length_target_mm:.2f}mm(fixed)"
            if focal_length_target_mm is not None and template.derived_index is not None
            else f"  f={foclen:.1f}mm"
        )
        ax.set_title(
            f"#{rank+1}  spot={spot:.4f}mm{efl_note}\n{theta_str}",
            fontsize=5,
        )
    ax.set_axis_off()

    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"spot_size_mm": float(spot), "focal_length_mm": float(foclen)}


# ---------------------------------------------------------------------------
# Grid renderer
# ---------------------------------------------------------------------------

def render_archive_grid_lensmega(
    *,
    selected: Sequence[ArchiveDesign],
    lens_json_path: Path,
    template: LensTemplate,
    solver: GeneralizedParaxialSolver,
    focal_length_target_mm: Optional[float],
    theme: PlotTheme,
    out_path: Path,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs: Sequence[float],
    cols: int,
    dpi: int,
    device: str,
    free_curvature_indices: Sequence[int],
    panel_titles: Optional[Sequence[Optional[str]]] = None,
    parameterization: str = "curvature",
    free_gap_indices: Optional[Sequence[int]] = None,
    variable_flange: bool = False,
    variable_materials: bool = False,
    free_material_indices: Optional[Sequence[int]] = None,
) -> list[dict[str, float]]:
    import matplotlib.pyplot as plt

    n = len(selected)
    if n == 0:
        return []
    cols = max(1, int(cols))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.3 * cols, 3.9 * rows), dpi=int(dpi),
    )
    fig.patch.set_facecolor(theme.background)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])

    panel_metrics: list[dict[str, float]] = []
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.set_facecolor(theme.background)
        ax.axis("off")
        if idx >= n:
            continue

        print(
            f"[{TAG}] rendering #{idx+1}/{n} fitness={selected[idx].fitness:.5f}",
            flush=True,
        )
        ptitle = (
            panel_titles[idx]
            if panel_titles and idx < len(panel_titles)
            else None
        )
        metrics = render_design_lensmega(
            lens_json_path=lens_json_path,
            design=selected[idx],
            theme=theme,
            ax=ax,
            template=template,
            solver=solver,
            focal_length_target_mm=focal_length_target_mm,
            depth_mm=float(depth_mm),
            wavelength_um=float(wavelength_um),
            num_rays=int(num_rays),
            fov_fracs=fov_fracs,
            rank=idx,
            device=device,
            free_curvature_indices=free_curvature_indices,
            title=ptitle,
            parameterization=parameterization,
            free_gap_indices=free_gap_indices,
            variable_flange=variable_flange,
            variable_materials=variable_materials,
            free_material_indices=free_material_indices,
        )
        panel_metrics.append(metrics)

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), facecolor=theme.background)
    plt.close(fig)
    return panel_metrics


# ---------------------------------------------------------------------------
# Descriptor coordinate labels
# ---------------------------------------------------------------------------

def _descriptor_coord_labels(cfg: RunConfig) -> tuple[str, str]:
    """Build human-readable axis labels from the config."""
    if cfg.descriptor.kind == "zernike":
        indices = cfg.descriptor.params.get("noll_indices", [])
        if len(indices) >= 2:
            j0, j1 = int(indices[0]), int(indices[1])
            return (
                f"Z{j0} {NOLL_TERM_NAMES.get(j0, '')} (mm)",
                f"Z{j1} {NOLL_TERM_NAMES.get(j1, '')} (mm)",
            )
    return ("m0", "m1")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Generic renderer for lensmega QD archives (any lens type).",
    )
    p.add_argument(
        "--archive", nargs="+", default=None, metavar="PATH",
        help="archive_cells.json path(s), run dir(s), or glob(s). "
             "Default: newest under results/lensmega/.",
    )
    p.add_argument(
        "--config", default=None,
        help="Override: path to lensmega config YAML.",
    )
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: renders_lensmega_<ts> in run dir).")
    p.add_argument("--count", type=int, default=9)
    p.add_argument("--top-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-mm", type=float, default=None,
                   help="Object depth (mm). Default: from config.")
    p.add_argument("--wavelength-um", type=float, default=None,
                   help="Wavelength (um). Default: from config.")
    p.add_argument("--focal-length", type=float, default=None,
                   help="Override focal_length_target_mm from config.")
    p.add_argument("--num-rays", type=int, default=11)
    p.add_argument("--fov-fracs", default="0.0,0.5,0.9")
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-fitness", type=float, default=-0.8)
    p.add_argument("--use-default-theme", action="store_true")
    p.add_argument("--bg-color", default="#F0EDE4")
    p.add_argument("--lens-fill-color", default="#7BAFD4")
    p.add_argument("--lens-outline-color", default="#1A4E8C")
    p.add_argument("--ray-color", default="#E07B39")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)

    # 1. Resolve archive path.
    archive_path = _resolve_archive_path(args.archive)

    # 2. Discover and load config.
    cfg = _discover_config(archive_path, explicit_config=args.config)
    prob = cfg.problem

    # 3. Resolve lens path.
    lens_json_path = _resolve_lens_path(prob.lens_path)

    # 4. Apply CLI overrides.
    depth_mm = args.depth_mm if args.depth_mm is not None else prob.object_depth_mm
    wavelength_um = (
        args.wavelength_um if args.wavelength_um is not None else prob.wavelength_um
    )
    focal_length_target_mm = (
        args.focal_length
        if args.focal_length is not None
        else prob.focal_length_target_mm
    )

    # 5. Build LensTemplate + GeneralizedParaxialSolver.
    template, _base_lens = load_lens_template(
        str(lens_json_path),
        wavelength_um=wavelength_um,
        stop_surface_index=prob.stop_surface_index,
        free_curvature_indices=tuple(prob.free_curvature_indices),
        derived_curvature_index=prob.derived_curvature_index,
        device=args.device,
        cemented_interfaces=tuple(getattr(prob, "cemented_interfaces", ()) or ()),
    )
    solver = GeneralizedParaxialSolver(template)

    # 6. Output directory.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else archive_path.parent / f"renders_lensmega_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{TAG}] archive:       {archive_path}", flush=True)
    print(f"[{TAG}] config:        {cfg.io.run_name}", flush=True)
    print(f"[{TAG}] lens template: {lens_json_path}", flush=True)
    print(f"[{TAG}] output dir:    {out_dir}", flush=True)
    if focal_length_target_mm is not None:
        print(
            f"[{TAG}] EFL target:    {focal_length_target_mm} mm "
            f"(derived surface {prob.derived_curvature_index})",
            flush=True,
        )

    # 7. Load and filter designs.
    designs = load_lensmega_archive(archive_path)
    print(f"[{TAG}] loaded {len(designs)} designs", flush=True)

    if args.min_fitness is not None:
        designs = [d for d in designs if d.fitness >= args.min_fitness]
        print(f"[{TAG}] {len(designs)} designs after min-fitness filter", flush=True)

    if not designs:
        print(f"[{TAG}] no designs to render — exiting.", flush=True)
        return

    # 8. Select diverse top designs.
    selected = select_diverse_top(
        designs, count=args.count, top_frac=args.top_frac, seed=args.seed,
    )
    selected = sorted(
        selected,
        key=lambda d: (
            float(d.rms_spot)
            if d.rms_spot is not None and np.isfinite(d.rms_spot)
            else float("inf"),
            -float(d.fitness),
        ),
    )
    print(f"[{TAG}] selected {len(selected)} diverse designs", flush=True)

    # 9. Theme and FoV.
    theme = resolve_theme(
        use_default_theme=bool(args.use_default_theme),
        bg_color=str(args.bg_color),
        lens_fill_color=str(args.lens_fill_color),
        lens_outline_color=str(args.lens_outline_color),
        ray_color=str(args.ray_color),
    )
    fov_fracs = _parse_fov_fracs(str(args.fov_fracs))

    # 10. Render grid.
    parameterization = getattr(prob, "parameterization", "curvature")
    grid_path = out_dir / "archive_diverse_top_grid.png"
    panel_metrics = render_archive_grid_lensmega(
        selected=selected,
        lens_json_path=lens_json_path,
        template=template,
        solver=solver,
        focal_length_target_mm=focal_length_target_mm,
        theme=theme,
        out_path=grid_path,
        depth_mm=depth_mm,
        wavelength_um=wavelength_um,
        num_rays=args.num_rays,
        fov_fracs=fov_fracs,
        cols=args.grid_cols,
        dpi=args.dpi,
        device=args.device,
        free_curvature_indices=prob.free_curvature_indices,
        parameterization=parameterization,
        free_gap_indices=getattr(prob, "free_gap_indices", None),
        variable_flange=getattr(prob, "variable_flange", False),
        variable_materials=getattr(prob, "variable_materials", False),
        free_material_indices=getattr(prob, "free_material_indices", None),
    )
    print(f"[{TAG}] grid saved: {grid_path}", flush=True)

    # 11. Save selected_designs.json summary.
    selected_meta = []
    for rank, (design, panel) in enumerate(zip(selected, panel_metrics)):
        selected_meta.append({
            "rank": rank,
            "cell": [int(design.cell[0]), int(design.cell[1])],
            "fitness": float(design.fitness),
            "spot_size_mm": float(panel["spot_size_mm"]),
            "focal_length_mm": float(panel["focal_length_mm"]),
            "measures": [float(design.measures[0]), float(design.measures[1])],
            "theta": [float(x) for x in design.theta.tolist()],
            "rms_spot_from_archive": (
                float(design.rms_spot)
                if design.rms_spot is not None and np.isfinite(design.rms_spot)
                else None
            ),
        })

    summary = {
        "archive": str(archive_path),
        "lens_template": str(lens_json_path),
        "config_run_name": cfg.io.run_name,
        "free_curvature_indices": list(prob.free_curvature_indices),
        "derived_curvature_index": prob.derived_curvature_index,
        "focal_length_target_mm": focal_length_target_mm,
        "count": len(selected),
        "top_frac": float(args.top_frac),
        "depth_mm": float(depth_mm),
        "wavelength_um": float(wavelength_um),
        "fov_fracs": [float(x) for x in fov_fracs],
        "grid_image": grid_path.name,
        "selected": selected_meta,
    }
    (out_dir / "selected_designs.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    # 12. Archive map with rendered-selection overlay.
    sel_plot_path = out_dir / "archive_with_selected.png"
    coord_labels = _descriptor_coord_labels(cfg)
    save_archive_with_selected_plot(
        archive_path=archive_path,
        all_designs=designs,
        selected=selected,
        out_path=sel_plot_path,
        coord_labels=coord_labels,
        dpi=args.dpi,
    )
    print(f"[{TAG}] archive+selection plot: {sel_plot_path}", flush=True)

    print(f"[{TAG}] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
