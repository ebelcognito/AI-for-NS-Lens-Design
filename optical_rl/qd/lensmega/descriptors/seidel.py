"""Paraxial Seidel aberration coefficients as behavioral descriptors.

Computes the five third-order (Seidel) aberration sums from a paraxial
marginal + chief ray trace through the lens.  Fully differentiable via
torch so CMA-MEGA can compute gradient-based search directions.

Seidel terms
------------
S1 — Spherical aberration
S2 — Coma
S3 — Astigmatism
S4 — Petzval field curvature
S5 — Distortion

Usage::

    descriptor:
      kind: seidel
      params:
        terms: ["S1", "S3"]   # 2-D BC space: spherical vs astigmatism
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ..problem.lens_template import LensTemplate

SEIDEL_TERM_LABELS = ("S1_spherical", "S2_coma", "S3_astigmatism",
                      "S4_petzval", "S5_distortion")

_TERM_INDEX = {"S1": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4}
_TERM_NAMES = {
    "S1": "S1_spherical",
    "S2": "S2_coma",
    "S3": "S3_astigmatism",
    "S4": "S4_petzval",
    "S5": "S5_distortion",
}


class SeidelDescriptor:
    """Paraxial Seidel aberration sums as MAP-Elites behaviour coordinates.

    The descriptor traces a normalised marginal ray ``[h=1, nu=0]`` and a
    chief ray (passing through the stop centre at the requested field angle)
    through the lens system.  At every refracting surface it accumulates the
    five Seidel contributions using the Welford convention:

    .. math::

        S_{I,k}   &= -A_k^2       \\, h_k \\, \\Delta(u/n)_k  \\\\
        S_{II,k}  &= -A_k \\bar A_k \\, h_k \\, \\Delta(u/n)_k  \\\\
        S_{III,k} &= -\\bar A_k^2   \\, h_k \\, \\Delta(u/n)_k  \\\\
        S_{IV,k}  &= -H^2 c_k \\, \\Delta(1/n)_k                \\\\
        S_{V,k}   &= -(\\bar A_k/A_k)(S_{III,k}+S_{IV,k})

    where *A*, *Ā* are the Snell invariants for the marginal and chief rays,
    *H* is the Lagrange invariant, and Δ denotes the jump across the surface.

    The computation uses only torch scalar operations so that autograd can
    propagate gradients back to the lens curvatures and gap parameters.
    """

    def __init__(
        self,
        seidel_terms: Sequence[str],
        fixed_bounds: Optional[np.ndarray] = None,
    ) -> None:
        self._terms = tuple(str(t) for t in seidel_terms)
        for t in self._terms:
            if t not in _TERM_INDEX:
                raise ValueError(
                    f"unknown Seidel term {t!r}; choose from {list(_TERM_INDEX)}"
                )
        self._indices = [_TERM_INDEX[t] for t in self._terms]
        self._fixed_bounds = fixed_bounds
        self._template: Optional[LensTemplate] = None
        self._prepared = False

    # ------------------------------------------------------------------
    # Descriptor protocol
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return len(self._terms)

    @property
    def coord_names(self) -> Tuple[str, ...]:
        return tuple(_TERM_NAMES[t] for t in self._terms)

    @property
    def fixed_bounds(self) -> Optional[np.ndarray]:
        return self._fixed_bounds

    def prepare(self, template: LensTemplate, *, device: str) -> None:
        self._template = template
        self._prepared = True

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        lens,
        *,
        field_y_deg: float,
        object_depth_mm: float,
        wavelength_um: float,
        device: str,
        n_bef_override=None,
        n_aft_override=None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute selected Seidel aberration sums.

        Returns a shape-``(K,)`` differentiable tensor where *K* =
        ``len(seidel_terms)``.
        """
        if not self._prepared:
            raise RuntimeError("SeidelDescriptor.prepare() was not called")
        template = self._template
        assert template is not None

        N = len(template.surfaces)
        dtype = torch.float64

        # ---- curvatures from the live lens (differentiable) -----------
        c: list[torch.Tensor] = []
        for i in range(N):
            if template.surfaces[i].is_stop:
                c.append(torch.tensor(0.0, dtype=dtype, device=device))
            else:
                c.append(lens.surfaces[i].c.to(dtype=dtype))

        # ---- inter-surface gaps from positions (differentiable) -------
        gaps: list[torch.Tensor] = []
        for i in range(N - 1):
            d_i = lens.surfaces[i].d.to(dtype=dtype)
            d_next = lens.surfaces[i + 1].d.to(dtype=dtype)
            gaps.append(d_next - d_i)

        # ---- refractive indices (overridable for variable materials) ---
        if n_bef_override is not None:
            n_bef = n_bef_override
            n_aft = n_aft_override
        else:
            n_bef = [template.surfaces[i].n_before for i in range(N)]
            n_aft = [template.surfaces[i].n_after for i in range(N)]

        # ==============================================================
        # 1. Marginal ray trace  y₀ = 1, nu₀ = 0
        # ==============================================================
        h_m: list[torch.Tensor] = []
        nu_bm: list[torch.Tensor] = []   # reduced angle *before* refraction
        nu_am: list[torch.Tensor] = []   # reduced angle *after*  refraction

        y = torch.ones(1, dtype=dtype, device=device)
        nu = torch.zeros(1, dtype=dtype, device=device)
        for i in range(N):
            h_m.append(y)
            nu_bm.append(nu)
            phi = (n_aft[i] - n_bef[i]) * c[i]
            nu = nu - phi * y
            nu_am.append(nu)
            if i < N - 1:
                y = y + (gaps[i] / n_aft[i]) * nu

        # ==============================================================
        # 2. Chief ray  —  h̄(stop) = 0, ν̄₀ = tan(field)
        # ==============================================================
        stop_idx = template.stop_index
        field_rad = math.radians(field_y_deg)
        tan_field = math.tan(field_rad) if abs(field_rad) > 1e-12 else 0.0
        nubar_0 = torch.tensor([tan_field], dtype=dtype, device=device)

        # Trace two basis rays from surface 0 to the stop to find the
        # ABCD transfer-matrix elements M₁₁ (yA) and M₁₂ (yB).
        yA = torch.ones(1, dtype=dtype, device=device)
        nuA = torch.zeros(1, dtype=dtype, device=device)
        for i in range(stop_idx):
            phi = (n_aft[i] - n_bef[i]) * c[i]
            nuA = nuA - phi * yA
            yA = yA + (gaps[i] / n_aft[i]) * nuA

        yB = torch.zeros(1, dtype=dtype, device=device)
        nuB = torch.ones(1, dtype=dtype, device=device)
        for i in range(stop_idx):
            phi = (n_aft[i] - n_bef[i]) * c[i]
            nuB = nuB - phi * yB
            yB = yB + (gaps[i] / n_aft[i]) * nuB

        # h̄₀·yA + ν̄₀·yB = 0  ⟹  h̄₀ = −ν̄₀·yB/yA
        safe_yA = torch.where(
            torch.abs(yA) > 1e-10, yA, torch.full_like(yA, 1e-10)
        )
        hbar_0 = -nubar_0 * yB / safe_yA

        # Full chief ray trace through all surfaces
        h_c: list[torch.Tensor] = []
        nu_bc: list[torch.Tensor] = []
        nu_ac: list[torch.Tensor] = []

        ybar = hbar_0
        nubar = nubar_0
        for i in range(N):
            h_c.append(ybar)
            nu_bc.append(nubar)
            phi = (n_aft[i] - n_bef[i]) * c[i]
            nubar = nubar - phi * ybar
            nu_ac.append(nubar)
            if i < N - 1:
                ybar = ybar + (gaps[i] / n_aft[i]) * nubar

        # ==============================================================
        # 3. Lagrange invariant  H = n₀(u₀ h̄₀ − ū₀ h₀)
        # ==============================================================
        u_0 = nu_bm[0] / n_bef[0]       # = 0 (marginal parallel to axis)
        ubar_0 = nu_bc[0] / n_bef[0]    # = tan(θ)
        H = n_bef[0] * (u_0 * h_c[0] - ubar_0 * h_m[0])

        # ==============================================================
        # 4. Accumulate per-surface Seidel contributions
        # ==============================================================
        S = [torch.zeros(1, dtype=dtype, device=device) for _ in range(5)]

        for k in range(N):
            nb = n_bef[k]
            na = n_aft[k]
            # Use template stop flag for skip — avoids tensor comparison issues
            if template.surfaces[k].is_stop:
                continue
            delta_n = na - nb
            _dn = float(delta_n.detach()) if torch.is_tensor(delta_n) else float(delta_n)
            if abs(_dn) < 1e-12:
                continue

            c_k = c[k]
            h_k = h_m[k]
            hbar_k = h_c[k]

            # physical angles (u = nu / n)
            u_k = nu_bm[k] / nb
            up_k = nu_am[k] / na
            ubar_k = nu_bc[k] / nb

            # paraxial incidence angles  i = h·c + u
            i_k = h_k * c_k + u_k
            ibar_k = hbar_k * c_k + ubar_k

            # Snell invariants  A = n·i
            A_k = nb * i_k
            Abar_k = nb * ibar_k

            # jumps across the surface
            delta_u_n = up_k / na - u_k / nb     # Δ(u/n)
            delta_inv_n = 1.0 / na - 1.0 / nb    # Δ(1/n)

            # five Seidel contributions (Welford sign convention)
            S_I_k = -(A_k ** 2) * h_k * delta_u_n
            S_II_k = -(A_k * Abar_k) * h_k * delta_u_n
            S_III_k = -(Abar_k ** 2) * h_k * delta_u_n
            S_IV_k = -(H ** 2) * c_k * delta_inv_n

            safe_A_k = torch.where(
                torch.abs(A_k) > 1e-10, A_k, torch.full_like(A_k, 1e-10)
            )
            S_V_k = -(Abar_k / safe_A_k) * (S_III_k + S_IV_k)

            S[0] = S[0] + S_I_k
            S[1] = S[1] + S_II_k
            S[2] = S[2] + S_III_k
            S[3] = S[3] + S_IV_k
            S[4] = S[4] + S_V_k

        # ==============================================================
        # 5. Return the user-selected subset
        # ==============================================================
        all_seidel = torch.stack([s.squeeze() for s in S])
        # Side-channel: expose the full 5-Seidel vector for lineage logging.
        # Non-differentiable snapshot — the autograd graph stays on `picks`.
        self._last_all_seidel = tuple(
            float(all_seidel[j].detach().cpu().item()) for j in range(5)
        )
        picks = torch.stack([all_seidel[idx] for idx in self._indices])
        return picks


# ======================================================================
# Per-element Seidel fingerprint (standalone, non-differentiable)
# ======================================================================

def _detect_elements(template: LensTemplate):
    """Group surfaces into optical elements.

    An element is a contiguous run of surfaces sharing a glass medium
    (n_after != 1.0 for non-air).  Returns list of (name, [surface_indices]).
    """
    N = len(template.surfaces)
    elements = []
    i = 0
    elem_idx = 0
    while i < N:
        s = template.surfaces[i]
        if s.is_stop:
            i += 1
            continue
        # Start of an element: surface entering glass (n_after != 1.0)
        if abs(s.n_after - 1.0) > 0.01:
            group = [i]
            glass_name = f"E{elem_idx + 1}"
            # Walk forward while we're still in glass
            j = i + 1
            while j < N:
                sj = template.surfaces[j]
                if sj.is_stop:
                    j += 1
                    continue
                group.append(j)
                if abs(sj.n_after - 1.0) < 0.01:
                    # Exited to air — element ends
                    break
                j += 1
            elements.append((glass_name, group))
            elem_idx += 1
            i = j + 1
        else:
            i += 1
    return elements


def compute_per_element_seidel(
    lens,
    template: LensTemplate,
    *,
    field_y_deg: float,
    wavelength_um: float = 0.587,
    device: str = "cpu",
) -> dict:
    """Compute per-element Seidel fingerprint (non-differentiable).

    Returns a dict like::

        {
            "elements": [
                {"name": "E1", "surfaces": [0, 1], "glass": "n-sk16",
                 "S1": 0.030, "S2": 0.008, ...},
                ...
            ],
            "system_total": {"S1": 0.001, "S2": -0.002, ...},
        }
    """
    N = len(template.surfaces)
    dtype = torch.float64

    # Extract curvatures and gaps
    c = []
    for i in range(N):
        if template.surfaces[i].is_stop:
            c.append(torch.tensor(0.0, dtype=dtype, device=device))
        else:
            c.append(lens.surfaces[i].c.to(dtype=dtype).detach())

    gaps = []
    for i in range(N - 1):
        d_i = lens.surfaces[i].d.to(dtype=dtype).detach()
        d_next = lens.surfaces[i + 1].d.to(dtype=dtype).detach()
        gaps.append(d_next - d_i)

    n_bef = [template.surfaces[i].n_before for i in range(N)]
    n_aft = [template.surfaces[i].n_after for i in range(N)]

    # Marginal ray
    y = torch.ones(1, dtype=dtype, device=device)
    nu = torch.zeros(1, dtype=dtype, device=device)
    h_m, nu_bm, nu_am = [], [], []
    for i in range(N):
        h_m.append(y)
        nu_bm.append(nu)
        phi = (n_aft[i] - n_bef[i]) * c[i]
        nu = nu - phi * y
        nu_am.append(nu)
        if i < N - 1:
            y = y + (gaps[i] / n_aft[i]) * nu

    # Chief ray
    stop_idx = template.stop_index
    field_rad = math.radians(field_y_deg)
    tan_field = math.tan(field_rad) if abs(field_rad) > 1e-12 else 0.0
    nubar_0 = torch.tensor([tan_field], dtype=dtype, device=device)

    yA = torch.ones(1, dtype=dtype, device=device)
    nuA = torch.zeros(1, dtype=dtype, device=device)
    for i in range(stop_idx):
        phi = (n_aft[i] - n_bef[i]) * c[i]
        nuA = nuA - phi * yA
        yA = yA + (gaps[i] / n_aft[i]) * nuA

    yB = torch.zeros(1, dtype=dtype, device=device)
    nuB = torch.ones(1, dtype=dtype, device=device)
    for i in range(stop_idx):
        phi = (n_aft[i] - n_bef[i]) * c[i]
        nuB = nuB - phi * yB
        yB = yB + (gaps[i] / n_aft[i]) * nuB

    safe_yA = torch.where(torch.abs(yA) > 1e-10, yA, torch.full_like(yA, 1e-10))
    hbar_0 = -nubar_0 * yB / safe_yA

    h_c, nu_bc, nu_ac = [], [], []
    ybar, nubar = hbar_0, nubar_0
    for i in range(N):
        h_c.append(ybar)
        nu_bc.append(nubar)
        phi = (n_aft[i] - n_bef[i]) * c[i]
        nubar = nubar - phi * ybar
        nu_ac.append(nubar)
        if i < N - 1:
            ybar = ybar + (gaps[i] / n_aft[i]) * nubar

    # Lagrange invariant
    u_0 = nu_bm[0] / n_bef[0]
    ubar_0 = nu_bc[0] / n_bef[0]
    H = n_bef[0] * (u_0 * h_c[0] - ubar_0 * h_m[0])

    # Per-surface Seidel contributions (store individually)
    per_surface = {}  # surface_index -> [S1, S2, S3, S4, S5]
    for k in range(N):
        nb, na = n_bef[k], n_aft[k]
        if abs(na - nb) < 1e-12:
            continue
        c_k, h_k, hbar_k = c[k], h_m[k], h_c[k]
        u_k = nu_bm[k] / nb
        up_k = nu_am[k] / na
        ubar_k = nu_bc[k] / nb
        i_k = h_k * c_k + u_k
        ibar_k = hbar_k * c_k + ubar_k
        A_k = nb * i_k
        Abar_k = nb * ibar_k
        delta_u_n = up_k / na - u_k / nb
        delta_inv_n = 1.0 / na - 1.0 / nb

        S_I = float((-(A_k ** 2) * h_k * delta_u_n).item())
        S_II = float((-(A_k * Abar_k) * h_k * delta_u_n).item())
        S_III = float((-(Abar_k ** 2) * h_k * delta_u_n).item())
        S_IV = float((-(H ** 2) * c_k * delta_inv_n).item())
        safe_A = max(abs(float(A_k.item())), 1e-10)
        S_V = -(float(Abar_k.item()) / safe_A) * (S_III + S_IV)

        per_surface[k] = [S_I, S_II, S_III, S_IV, S_V]

    # Group into elements
    elements_info = _detect_elements(template)
    result_elements = []
    system_total = [0.0] * 5

    for name, surf_indices in elements_info:
        elem_seidel = [0.0] * 5
        for si in surf_indices:
            if si in per_surface:
                for j in range(5):
                    elem_seidel[j] += per_surface[si][j]
        for j in range(5):
            system_total[j] += elem_seidel[j]

        # Detect glass name from the template
        glass = "air"
        for si in surf_indices:
            s = template.surfaces[si]
            if abs(s.n_after - 1.0) > 0.01:
                glass = f"n={s.n_after:.3f}"
                break

        result_elements.append({
            "name": name,
            "surfaces": surf_indices,
            "glass": glass,
            **{SEIDEL_TERM_LABELS[j]: round(elem_seidel[j], 8)
               for j in range(5)},
        })

    return {
        "field_y_deg": field_y_deg,
        "elements": result_elements,
        "system_total": {
            SEIDEL_TERM_LABELS[j]: round(system_total[j], 8)
            for j in range(5)
        },
    }
