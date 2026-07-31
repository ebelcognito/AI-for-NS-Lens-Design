"""
Render a 1×N grid of singlet designs picked by MAP-Elites archive bin (cell).

Default cells match three example individuals (~c1 ≈ -0.1, ~0, ~0.1):

  --cell 18,36 --cell 134,244 --cell 241,82

Usage
-----
  python3 optical_rl/qd/render_singlet_archive_by_cells.py
  python3 optical_rl/qd/render_singlet_archive_by_cells.py --archive path/to/archive.json
  python3 optical_rl/qd/render_singlet_archive_by_cells.py --cell 10,20 --cell 100,100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.qd.render_singlet_archive_designs import (  # noqa: E402
    ArchiveDesign,
    _default_lens_template,
    _find_latest_archive,
    _parse_fov_fracs,
    load_archive,
    render_archive_grid,
    resolve_theme,
)

DEFAULT_CELLS: list[tuple[int, int]] = [
    (18, 36),
    (134, 244),
    (241, 82),
]


def _parse_cell(spec: str) -> tuple[int, int]:
    parts = spec.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected R,C, got {spec!r}")
    return int(parts[0]), int(parts[1])


def _find_design(designs: list[ArchiveDesign], cell: tuple[int, int]) -> ArchiveDesign | None:
    for d in designs:
        if d.cell == cell:
            return d
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Render singlet lens/rays for archive entries identified by cell indices."
    )
    p.add_argument("--archive", default=None, help="archive.json (default: latest singlet_cmamega)")
    p.add_argument("--lens-template", default=str(_default_lens_template()))
    p.add_argument(
        "--cell",
        action="append",
        dest="cells",
        type=_parse_cell,
        default=None,
        metavar="R,C",
        help="Bin cell row,col; repeat once per panel, left-to-right (default: three c1-sweep examples)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: <run_dir>/renders_picked_cells.png)",
    )
    p.add_argument("--depth-mm", type=float, default=-1000.0)
    p.add_argument("--wavelength-um", type=float, default=0.587)
    p.add_argument("--num-rays", type=int, default=11)
    p.add_argument("--fov-fracs", default="0.0,0.5,0.9")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-default-theme", action="store_true")
    p.add_argument("--bg-color", default="#F8F5E9")
    p.add_argument("--lens-fill-color", default="#9DC08B")
    p.add_argument("--lens-outline-color", default="#3A7D44")
    p.add_argument("--ray-color", default="#DF6D14")
    args = p.parse_args(argv)

    cells = args.cells if args.cells else list(DEFAULT_CELLS)
    archive_path = Path(args.archive) if args.archive else _find_latest_archive()
    lens_template = Path(args.lens_template)
    out_path = args.out if args.out else archive_path.resolve().parent / "renders_picked_cells.png"

    designs = load_archive(archive_path)
    selected: list[ArchiveDesign] = []
    for cell in cells:
        d = _find_design(designs, cell)
        if d is None:
            sys.exit(f"No archive entry with cell {list(cell)} in {archive_path}")
        selected.append(d)

    titles: list[str] = []
    for d in selected:
        c1 = float(np.asarray(d.theta, dtype=np.float64).reshape(-1)[0])
        spot = d.rms_spot if d.rms_spot is not None else float("nan")
        titles.append(
            f"cell {d.cell[0]},{d.cell[1]}  c1={c1:.4f}  spot={spot:.4f} mm"
        )

    theme = resolve_theme(
        use_default_theme=bool(args.use_default_theme),
        bg_color=str(args.bg_color),
        lens_fill_color=str(args.lens_fill_color),
        lens_outline_color=str(args.lens_outline_color),
        ray_color=str(args.ray_color),
    )
    fov_fracs = _parse_fov_fracs(str(args.fov_fracs))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_archive_grid(
        selected=selected,
        lens_template=lens_template,
        theme=theme,
        out_path=out_path,
        depth_mm=float(args.depth_mm),
        wavelength_um=float(args.wavelength_um),
        num_rays=int(args.num_rays),
        fov_fracs=fov_fracs,
        cols=len(selected),
        dpi=int(args.dpi),
        device=str(args.device),
        panel_titles=titles,
    )
    print(f"[render-by-cells] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
