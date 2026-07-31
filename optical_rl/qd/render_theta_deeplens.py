"""Render an EF50 lens design from a theta vector using DeepLens analysis.

Always saves the PNG and prints RMS spot sizes at 0%, 50%, and 90% FOV.

Usage:
  # Defaults to rank-0 example from the latest render_summary.json:
  python3 optical_rl/qd/render_theta_deeplens.py

  # Supply your own theta (space- or comma-separated, 10 values):
  python3 optical_rl/qd/render_theta_deeplens.py --theta "0.026 0.022 0.025 0.008 0.025 0.052 -0.080 -0.015 -0.058 0.002"

  # Load rank N from a render_summary.json:
  python3 optical_rl/qd/render_theta_deeplens.py --from-summary path/to/render_summary.json --rank 0

  # Custom output path:
  python3 optical_rl/qd/render_theta_deeplens.py --output /tmp/my_lens.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup (mirrors render_ef50_zernike.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEEPLENS_ROOT = PROJECT_ROOT / "DeepLens"
for _p in [str(PROJECT_ROOT), str(_DEEPLENS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optical_rl.qd.render_ef50_zernike import _apply_theta_to_lens_ef50  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_LENS_TEMPLATE = PROJECT_ROOT / "DeepLens/datasets/lenses/camera/ef50mm_f1.8.json"
_DEFAULT_FOCAL_LENGTH_MM = 50.52
_DEFAULT_OUTPUT = Path("deeplens_render.png")
_FOV_FRACS = [0.0, 0.5, 0.9]
_DEPTH_MM = -20000.0
_WAVELENGTH_UM = 0.587
_NUM_RAYS = 64

# Rank-0 panel from the most recent render_summary.json (2026-04-09)
_EXAMPLE_THETA = np.array([
    0.026128659936854343,
    0.02155183135937308,
    0.024785206766414925,
    0.007819488032164612,
    0.02493818963221161,
    0.05218501470563203,
    -0.07968403105095712,
    -0.015266665182887081,
    -0.05809203796589087,
    0.0021655426161041767,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_theta(s: str) -> np.ndarray:
    parts = s.replace(",", " ").split()
    arr = np.array([float(x) for x in parts], dtype=np.float64)
    if arr.size != 10:
        raise ValueError(f"Expected 10 theta values, got {arr.size}")
    return arr


def _load_theta_from_summary(path: Path, rank: int) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    panels = data.get("panels", [])
    for panel in panels:
        if panel.get("rank") == rank:
            return np.array(panel["theta"], dtype=np.float64)
    if rank < len(panels):
        return np.array(panels[rank]["theta"], dtype=np.float64)
    raise ValueError(f"Rank {rank} not found in {path} (found {len(panels)} panels)")


def _rms_spot_at_fov(lens, fov_deg: float) -> float:
    """Compute RMS spot size (mm) at a given field angle."""
    import torch

    ray = lens.sample_point_source_2D(
        fov=fov_deg,
        depth=float(_DEPTH_MM),
        num_rays=_NUM_RAYS,
        wvln=float(_WAVELENGTH_UM),
    )
    ray.prop_to(-1.0)
    ray_out = lens.trace2sensor(ray=ray)
    valid = ray_out.is_valid
    denom = valid.sum().clamp_min(1e-8)
    xy = ray_out.o[..., :2]
    centroid = (xy * valid.unsqueeze(-1)).sum(dim=0) / denom
    err2 = ((xy - centroid) ** 2).sum(dim=-1)
    rms = ((err2 * valid).sum() / denom + 1e-12).sqrt()
    return float(rms.detach().cpu().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Render an EF50 theta vector with DeepLens analysis.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--theta", type=str, default=None,
        help="10 curvature values (space- or comma-separated, units mm⁻¹)",
    )
    group.add_argument(
        "--from-summary", type=Path, default=None, metavar="PATH",
        help="Load theta from a render_summary.json file",
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Which panel rank to load from --from-summary (default: 0)",
    )
    parser.add_argument(
        "--focal-length", type=float, default=_DEFAULT_FOCAL_LENGTH_MM,
        metavar="MM",
        help=f"Target EFL in mm for c11 derivation (default: {_DEFAULT_FOCAL_LENGTH_MM})",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device (default: cpu)",
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    # Resolve theta
    if args.theta is not None:
        theta = _parse_theta(args.theta)
    elif args.from_summary is not None:
        theta = _load_theta_from_summary(args.from_summary, args.rank)
        print(f"Loaded rank-{args.rank} theta from {args.from_summary}")
    else:
        theta = _EXAMPLE_THETA
        print("Using built-in example theta (rank-0 from 2026-04-09 run).")

    print(f"theta            = {[f'{v:.6f}' for v in theta.tolist()]}")
    print(f"focal_length_mm  = {args.focal_length}")

    # Load EF50 template and apply theta
    from deeplens.optics import GeoLens  # type: ignore

    lens = GeoLens(filename=str(_DEFAULT_LENS_TEMPLATE), device=args.device)
    _apply_theta_to_lens_ef50(
        lens, theta,
        wvln_um=_WAVELENGTH_UM,
        focal_length_target_mm=args.focal_length,
    )

    # Print RMS spot sizes at multiple FOV fractions
    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    print(f"\nRMS spot sizes  (rfov = {rfov_deg:.2f} deg, depth = {_DEPTH_MM} mm, λ = {_WAVELENGTH_UM} µm):")
    print(f"  {'FOV (%)':>8}  {'FOV (deg)':>10}  {'RMS spot (mm)':>14}  {'RMS spot (µm)':>14}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*14}")
    for frac in _FOV_FRACS:
        fov_deg = frac * rfov_deg
        rms_mm = _rms_spot_at_fov(lens, fov_deg)
        print(f"  {frac*100:>7.0f}%  {fov_deg:>10.2f}  {rms_mm:>14.6f}  {rms_mm*1e3:>14.3f}")

    # render=False skips the expensive NBS chart image simulation;
    # we still get lens layout, spot diagrams, and MTF plots.
    print(f"\nRunning DeepLens analysis...")
    lens.analysis(render=False)

    output = args.output
    plt.savefig(str(output), dpi=150, bbox_inches="tight")
    print(f"Saved → {output.resolve()}")


if __name__ == "__main__":
    main()
