"""Noll Zernike basis evaluation and OPL-based OPD fitting for singlet CMA-MEGA.

Pupil convention: unit disk, Noll (1976) single-index ordering.

  j   n  m   name
  --  -  --  ----
   1  0   0  Piston
   2  1   1  Tilt x (cos)
   3  1  -1  Tilt y (sin)
   4  2   0  Defocus
   5  2  -2  Oblique astigmatism (sin)
   6  2   2  Vertical astigmatism (cos)
   7  3  -1  Vertical coma (sin)
   8  3   1  Horizontal coma (cos)
   9  3  -3  Oblique trefoil (sin)
  10  3   3  Vertical trefoil (cos)
  11  4   0  Primary spherical
  12  4  -2  Secondary oblique astigmatism (sin)
  13  4   2  Secondary vertical astigmatism (cos)
  14  4  -4  Tetrafoil (sin)
  15  4   4  Tetrafoil (cos)
  16  5  -1  Secondary coma (sin)

Typical descriptor choices for a singlet (on-axis, fixed EFL):
  (4, 11)   — defocus + primary spherical            [default]
  (11, 16)  — primary spherical + secondary coma
  (7, 11)   — vertical coma + primary spherical       [good for off-axis stages]
  (5, 11)   — oblique astigmatism + primary spherical

Set descriptor_zernike_indices in the YAML config to try different combos.
"""

from __future__ import annotations

import math

import numpy as np


# Maximum j implemented in _noll_zernike_matrix.
NOLL_N_TERMS = 16

# Human-readable names keyed by 1-based Noll index j.
NOLL_TERM_NAMES: dict[int, str] = {
    1:  "piston",
    2:  "tilt_x",
    3:  "tilt_y",
    4:  "defocus",
    5:  "oblique_astigmatism",
    6:  "vertical_astigmatism",
    7:  "vertical_coma",
    8:  "horizontal_coma",
    9:  "oblique_trefoil",
    10: "vertical_trefoil",
    11: "primary_spherical",
    12: "secondary_oblique_astigmatism",
    13: "secondary_vertical_astigmatism",
    14: "tetrafoil_sin",
    15: "tetrafoil_cos",
    16: "secondary_coma",
}


def _noll_zernike_matrix(rho: np.ndarray, phi: np.ndarray, j_max: int = NOLL_N_TERMS) -> np.ndarray:
    """Evaluate Noll Zernike polynomials j=1..j_max at given pupil coordinates.

    Args:
        rho: radial coordinate in [0, 1], shape (N,).
        phi: azimuthal coordinate in [-pi, pi], shape (N,).
        j_max: highest Noll index to evaluate (1-based); max supported = 16.

    Returns:
        Z: shape (N, j_max), column j-1 is Z_j.
    """
    if j_max > NOLL_N_TERMS:
        raise ValueError(f"j_max={j_max} exceeds implemented maximum {NOLL_N_TERMS}.")
    N = len(rho)
    Z = np.zeros((N, j_max), dtype=np.float64)
    r = rho.astype(np.float64)
    t = phi.astype(np.float64)

    # n=0 -----------------------------------------------------------------------
    if j_max >= 1:
        Z[:, 0] = 1.0                                                  # j=1  piston

    # n=1 -----------------------------------------------------------------------
    if j_max >= 2:
        Z[:, 1] = 2.0 * r * np.cos(t)                                 # j=2  tilt x
    if j_max >= 3:
        Z[:, 2] = 2.0 * r * np.sin(t)                                 # j=3  tilt y

    # n=2 -----------------------------------------------------------------------
    if j_max >= 4:
        Z[:, 3] = math.sqrt(3.0) * (2.0 * r**2 - 1.0)                # j=4  defocus
    if j_max >= 5:
        Z[:, 4] = math.sqrt(6.0) * r**2 * np.sin(2.0 * t)            # j=5  oblique astigmatism
    if j_max >= 6:
        Z[:, 5] = math.sqrt(6.0) * r**2 * np.cos(2.0 * t)            # j=6  vertical astigmatism

    # n=3 -----------------------------------------------------------------------
    if j_max >= 7:
        Z[:, 6] = math.sqrt(8.0) * (3.0 * r**3 - 2.0 * r) * np.sin(t)  # j=7  vertical coma
    if j_max >= 8:
        Z[:, 7] = math.sqrt(8.0) * (3.0 * r**3 - 2.0 * r) * np.cos(t)  # j=8  horizontal coma
    if j_max >= 9:
        Z[:, 8] = math.sqrt(8.0) * r**3 * np.sin(3.0 * t)            # j=9  oblique trefoil
    if j_max >= 10:
        Z[:, 9] = math.sqrt(8.0) * r**3 * np.cos(3.0 * t)            # j=10 vertical trefoil

    # n=4 -----------------------------------------------------------------------
    if j_max >= 11:
        Z[:, 10] = math.sqrt(5.0) * (6.0 * r**4 - 6.0 * r**2 + 1.0) # j=11 primary spherical
    if j_max >= 12:
        Z[:, 11] = math.sqrt(10.0) * (4.0 * r**4 - 3.0 * r**2) * np.sin(2.0 * t)  # j=12
    if j_max >= 13:
        Z[:, 12] = math.sqrt(10.0) * (4.0 * r**4 - 3.0 * r**2) * np.cos(2.0 * t)  # j=13
    if j_max >= 14:
        Z[:, 13] = math.sqrt(10.0) * r**4 * np.sin(4.0 * t)          # j=14 tetrafoil sin
    if j_max >= 15:
        Z[:, 14] = math.sqrt(10.0) * r**4 * np.cos(4.0 * t)          # j=15 tetrafoil cos

    # n=5 -----------------------------------------------------------------------
    if j_max >= 16:
        Z[:, 15] = math.sqrt(12.0) * (10.0 * r**5 - 12.0 * r**3 + 3.0 * r) * np.sin(t)  # j=16

    return Z


def build_zernike_pinv(pupil_xy_unit: np.ndarray, j_max: int = NOLL_N_TERMS) -> np.ndarray:
    """Precompute the pseudo-inverse of the Zernike matrix for a fixed pupil sampling.

    Since pupil positions are constant across evaluations, the pseudo-inverse is
    computed once at initialisation.  The Zernike fit then reduces to a single
    matrix-vector multiply:  a = pinv_Z @ opd_vector.

    Args:
        pupil_xy_unit: (N, 2) unit-disk (x, y) coordinates; rho = sqrt(x²+y²) ≤ 1.
        j_max: number of Zernike terms to fit (highest 1-based Noll index needed).

    Returns:
        pinv_Z: (j_max, N) pseudo-inverse, ready for torch matmul after conversion.
    """
    x = pupil_xy_unit[:, 0].astype(np.float64)
    y = pupil_xy_unit[:, 1].astype(np.float64)
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    Z = _noll_zernike_matrix(rho, phi, j_max)
    return np.linalg.pinv(Z)   # shape (j_max, N)
