"""Full-DOF lens problem: variable curvatures + thicknesses + flange + materials.

Theta layout:  [curvatures..., gaps..., flange?, n_0, V_0, n_1, V_1, ...]

All curvatures are raw values (not power-budget fractions).  Gaps are direct
thickness values in mm.  When ``variable_flange`` is True the flange distance
(last surface to sensor) is in theta and d_sensor is derived as
``z_last_surface + flange``.  Otherwise d_sensor is computed from the paraxial
BFL.

When ``variable_materials`` is True, continuous (n_d, V_d) pairs per glass
element are appended to theta.  These flow through the paraxial solver
(EFL/BFL), the Seidel descriptor, and the ray tracer.

A soft quadratic EFL penalty steers the system toward the target focal length.
No paraxial solve or derived surface — every parameter is a genuine free
variable for CMA-MEGA.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .base import EvalResult
from .lens_problem import LensProblem, LensProblemConfig
from .lens_template import LensTemplate
from .penalties import intersection_penalty
from .tracing import (
    disk_samples,
    latin_hypercube,
    project_box,
    rms_spot_batch,
    trace_fields_batch,
)


# ---------------------------------------------------------------------------
# Paraxial solver that accepts variable gaps
# ---------------------------------------------------------------------------

class FullDOFParaxialSolver:
    """Paraxial ABCD solver that takes both curvatures AND gaps as tensors.

    Unlike ``GeneralizedParaxialSolver`` which reads gaps from the frozen
    template, this solver accepts ``gaps_full`` as a differentiable tensor
    so that gradients flow through thickness parameters.
    """

    def __init__(self, template: LensTemplate) -> None:
        self.template = template
        self._num_surfaces = len(template.surfaces)

    def trace_marginal(
        self,
        c_full: torch.Tensor,
        gaps_full: torch.Tensor,
        n_bef=None,
        n_aft=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trace marginal ray ``[y=1, nu=0]`` with variable gaps.

        When *n_bef* / *n_aft* are provided (lists of tensors or floats, one
        per surface) they override the frozen template values so that
        gradients flow through material parameters.

        No transfer after the last surface (BFL convention).
        """
        dtype = c_full.dtype
        device = c_full.device
        y = torch.ones(1, dtype=dtype, device=device)
        nu = torch.zeros(1, dtype=dtype, device=device)
        N = self._num_surfaces
        for i in range(N):
            surf = self.template.surfaces[i]
            nb = n_bef[i] if n_bef is not None else surf.n_before
            na = n_aft[i] if n_aft is not None else surf.n_after
            phi = (na - nb) * c_full[i]
            nu = nu - phi * y
            if i < N - 1:
                y = y + (gaps_full[i] / na) * nu
        return y, nu

    def bfl(
        self, c_full: torch.Tensor, gaps_full: torch.Tensor,
        n_bef=None, n_aft=None,
    ) -> torch.Tensor:
        """Back focal length: ``-y / nu`` after last refraction."""
        y, nu = self.trace_marginal(c_full, gaps_full, n_bef=n_bef, n_aft=n_aft)
        safe_nu = torch.where(
            torch.abs(nu) > 1e-6, nu, torch.full_like(nu, 1e-6)
        )
        return (-y / safe_nu).squeeze()

    def efl(
        self, c_full: torch.Tensor, gaps_full: torch.Tensor,
        n_bef=None, n_aft=None,
    ) -> torch.Tensor:
        """Effective focal length: ``-1 / nu`` after last refraction."""
        _, nu = self.trace_marginal(c_full, gaps_full, n_bef=n_bef, n_aft=n_aft)
        safe_nu = torch.where(
            torch.abs(nu) > 1e-6, nu, torch.full_like(nu, 1e-6)
        )
        return (-1.0 / safe_nu).squeeze()

    def paraxial_pupil_radius(
        self,
        c_full: torch.Tensor,
        gaps_full: torch.Tensor,
        aper_idx: int,
        r_stop_mm: float,
        n_bef=None,
        n_aft=None,
    ) -> torch.Tensor:
        """Differentiable paraxial entrance-pupil radius (full-DOF variant).

        Walks a unit parallel ray ``(y=1, nu=0)`` from object space through
        surfaces ``[0, aper_idx)`` using the ACTUAL ``gaps_full`` (and
        optionally per-surface ``n_bef``/``n_aft``), then returns
        ``r_stop / |y_at_stop|``. See
        ``GeneralizedParaxialSolver.paraxial_pupil_radius`` for derivation.
        Gradients flow through curvatures, gaps, and material indices.
        """
        dtype = c_full.dtype
        device = c_full.device
        y = torch.ones(1, dtype=dtype, device=device)
        nu = torch.zeros(1, dtype=dtype, device=device)
        for i in range(int(aper_idx)):
            surf = self.template.surfaces[i]
            nb = n_bef[i] if n_bef is not None else surf.n_before
            na = n_aft[i] if n_aft is not None else surf.n_after
            phi = (na - nb) * c_full[i]
            nu = nu - phi * y
            y = y + (gaps_full[i] / na) * nu
        safe_y = torch.where(
            torch.abs(y) > 1e-6, y, torch.full_like(y, 1e-6)
        )
        return (float(r_stop_mm) / torch.abs(safe_y)).squeeze()


# ---------------------------------------------------------------------------
# Full-DOF problem
# ---------------------------------------------------------------------------

class FullDOFLensProblem(LensProblem):
    """LensProblem with variable curvatures, thicknesses, and flange.

    Theta vector layout::

        [c_0, c_1, ..., c_Nc,   g_0, g_1, ..., g_Ng,   flange]
         |--- Nc curvatures ---|  |--- Ng gaps -------|  optional

    All parameters are direct physical values (mm for gaps/flange,
    1/mm for curvatures).  A soft EFL penalty steers toward the target.
    """

    def __init__(self, cfg: LensProblemConfig, descriptor) -> None:
        super().__init__(cfg, descriptor)
        if cfg.focal_length_target_mm is None:
            raise ValueError("FullDOFLensProblem requires focal_length_target_mm")
        self._efl_target = cfg.focal_length_target_mm
        self._efl_penalty_weight = cfg.efl_penalty_weight
        self._n_curv = len(cfg.free_curvature_indices)
        self._n_gaps = len(cfg.free_gap_indices)
        self._has_flange = cfg.variable_flange
        self._n_mat = len(cfg.free_material_indices) if cfg.variable_materials else 0
        self._full_dof_solver: Optional[FullDOFParaxialSolver] = None

    def _ensure_backend(self) -> None:
        super()._ensure_backend()
        self._full_dof_solver = FullDOFParaxialSolver(self.template)

    # ------------------------------------------------------------------
    # Theta unpacking
    # ------------------------------------------------------------------

    def _unpack_theta(self, theta_t: torch.Tensor):
        """Split theta into curvature, gap, flange, and material parts.

        Returns (curvatures, gaps, flange_or_None, nV_pairs_or_None).
        ``nV_pairs`` has shape ``(n_mat, 2)`` with columns ``[n_d, V_d]``.
        """
        nc = self._n_curv
        ng = self._n_gaps
        curvatures = theta_t[:nc]
        gaps = theta_t[nc : nc + ng] if ng > 0 else None
        offset = nc + ng
        flange = theta_t[offset] if self._has_flange else None
        if self._has_flange:
            offset += 1
        if self._n_mat > 0:
            nV_pairs = theta_t[offset : offset + 2 * self._n_mat].reshape(
                self._n_mat, 2
            )
        else:
            nV_pairs = None
        return curvatures, gaps, flange, nV_pairs

    def _build_c_full(self, curvatures: torch.Tensor, theta_t: torch.Tensor):
        """Expand free curvatures to full curvature vector."""
        parts = []
        free_set = set(self.cfg.free_curvature_indices)
        cursor = 0
        for i in range(len(self.template.surfaces)):
            if i in free_set:
                parts.append(curvatures[cursor])
                cursor += 1
            else:
                parts.append(torch.tensor(
                    self.template.surfaces[i].initial_c,
                    dtype=theta_t.dtype, device=theta_t.device,
                ))
        return torch.stack(parts)

    def _build_gaps_full(self, gaps: Optional[torch.Tensor], theta_t: torch.Tensor):
        """Expand free gaps to full gap vector.

        Non-free gaps use the template's initial values.
        """
        free_set = set(self.cfg.free_gap_indices)
        parts = []
        cursor = 0
        for i in range(len(self.template.surfaces)):
            if i in free_set:
                parts.append(gaps[cursor])
                cursor += 1
            else:
                parts.append(torch.tensor(
                    self.template.surfaces[i].gap_to_next,
                    dtype=theta_t.dtype, device=theta_t.device,
                ))
        return torch.stack(parts)

    def _build_n_full(self, nV_pairs, theta_t: torch.Tensor):
        """Build per-surface n_before / n_after from element-level (n, V).

        When *nV_pairs* is ``None`` returns plain float lists from the
        template (zero overhead, identical to the old code-path).
        """
        N = len(self.template.surfaces)
        if nV_pairs is None:
            return None, None  # callers pass None → solver uses template

        dtype = theta_t.dtype
        device = theta_t.device
        n_bef = [
            torch.tensor(self.template.surfaces[i].n_before, dtype=dtype, device=device)
            for i in range(N)
        ]
        n_aft = [
            torch.tensor(self.template.surfaces[i].n_after, dtype=dtype, device=device)
            for i in range(N)
        ]

        for mat_idx, elem_idx in enumerate(self.cfg.free_material_indices):
            n_d = nV_pairs[mat_idx, 0]
            surf_indices = self.template.element_groups[elem_idx]
            for j, si in enumerate(surf_indices):
                if j == 0:
                    # First surface of element: entering glass
                    n_aft[si] = n_d
                elif j == len(surf_indices) - 1:
                    # Last surface: exiting glass (n_after stays air)
                    n_bef[si] = n_d
                else:
                    # Interior of cemented group
                    n_bef[si] = n_d
                    n_aft[si] = n_d

        return n_bef, n_aft

    # ------------------------------------------------------------------
    # Lens building
    # ------------------------------------------------------------------

    def _build_lens(self, theta_t: torch.Tensor):
        assert self.template is not None and self._full_dof_solver is not None
        curvatures, gaps_free, flange_val, nV_pairs = self._unpack_theta(theta_t)

        c_full = self._build_c_full(curvatures, theta_t)
        gaps_full = self._build_gaps_full(gaps_free, theta_t)
        n_bef, n_aft = self._build_n_full(nV_pairs, theta_t)

        # Compute surface axial positions from gaps.
        dtype = self._reusable_lens.d_sensor.dtype
        lens = self._reusable_lens
        d_pos = torch.zeros(1, dtype=theta_t.dtype, device=theta_t.device)
        for i in range(len(self.template.surfaces)):
            lens.surfaces[i].d = d_pos.to(dtype=dtype)
            if i < len(self.template.surfaces) - 1:
                d_pos = d_pos + gaps_full[i]

        # Compute d_sensor: from flange (relative to last surface) or from
        # paraxial BFL.
        if flange_val is not None:
            d_sensor = d_pos + flange_val
            bfl_t = flange_val
        else:
            bfl_t = self._full_dof_solver.bfl(c_full, gaps_full, n_bef=n_bef, n_aft=n_aft)
            d_sensor = d_pos + bfl_t

        # Set curvatures.
        for i_local, i_global in enumerate(self.cfg.free_curvature_indices):
            lens.surfaces[i_global].c = curvatures[i_local].to(dtype=dtype)

        lens.d_sensor = d_sensor.to(dtype=dtype)

        # Apply materials to GeoLens for the ray tracer.
        if nV_pairs is not None:
            for mat_idx, elem_idx in enumerate(self.cfg.free_material_indices):
                n_d = nV_pairs[mat_idx, 0]
                V_d = nV_pairs[mat_idx, 1]
                for si in self.template.element_groups[elem_idx]:
                    s = lens.surfaces[si]
                    if hasattr(s, "mat2") and s.mat2.name != "air":
                        s.mat2.n = n_d.to(dtype=dtype)
                        s.mat2.V = V_d.to(dtype=dtype)
                        s.mat2.dispersion = "optimizable"

        return lens, bfl_t

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _fitness_and_metrics(self, theta_t: torch.Tensor):
        assert self.template is not None and self._full_dof_solver is not None
        K = self.descriptor.dim
        device = self.cfg.device

        curvatures, gaps_free, flange_val, nV_pairs = self._unpack_theta(theta_t)
        c_full = self._build_c_full(curvatures, theta_t)
        gaps_full = self._build_gaps_full(gaps_free, theta_t)
        n_bef, n_aft = self._build_n_full(nV_pairs, theta_t)

        # Check BFL positivity — soft penalty with gradients instead of
        # hard reject so that CMA-MEGA can learn to steer away from
        # diverging configurations.
        if flange_val is not None:
            bfl_t = flange_val
            bfl_val = float(bfl_t.detach().item())
        else:
            bfl_t = self._full_dof_solver.bfl(c_full, gaps_full, n_bef=n_bef, n_aft=n_aft)
            bfl_val = float(bfl_t.detach().item())

        if bfl_val <= 0.0:
            # Differentiable penalty: gradients point toward positive BFL.
            bfl_penalty = 50.0 * torch.clamp(-bfl_t, min=0.0)
            efl_actual = self._full_dof_solver.efl(c_full, gaps_full, n_bef=n_bef, n_aft=n_aft)
            efl_error_rel = (efl_actual - self._efl_target) / self._efl_target
            efl_penalty = self._efl_penalty_weight * efl_error_rel ** 2
            fitness = -1e6 - bfl_penalty - efl_penalty

            # Build lens and compute descriptor (Seidel is always valid).
            lens, _ = self._build_lens(theta_t)
            hfov = self._calc_hfov_deg(theta_t)
            field_pts = self._field_points(hfov)
            field_y_deg = float(field_pts[self.cfg.descriptor_field_index][1])
            try:
                m_t = self.descriptor.compute(
                    lens, field_y_deg=field_y_deg,
                    object_depth_mm=self.cfg.object_depth_mm,
                    wavelength_um=self.cfg.wavelength_um, device=device,
                    n_bef_override=n_bef, n_aft_override=n_aft,
                    theta_tensor=theta_t,
                )
                measures_list = [m_t[k] for k in range(K)]
            except Exception:
                measures_list = [theta_t[0] * 0.0 for _ in range(K)]

            metrics = {
                "invalid_fraction": 1.0,
                "rejected_bfl": 1.0,
                "bfl": bfl_val,
            }
            for k, name in enumerate(self.descriptor.coord_names):
                metrics[f"m{k}_{name}"] = float(
                    measures_list[k].detach().cpu().item()
                )
            last_all = getattr(self.descriptor, "_last_all_seidel", None)
            if last_all is not None:
                for j, label in enumerate(("S1", "S2", "S3", "S4", "S5")):
                    metrics[f"seidel_{label}"] = float(last_all[j])
            return fitness, measures_list, metrics

        lens, _ = self._build_lens(theta_t)
        hfov = self._calc_hfov_deg(theta_t)

        with torch.no_grad():
            pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)

        if self.cfg.strict_entrance_pupil:
            fb_z = float(lens.surfaces[0].d.item())
            fb_r = float(lens.surfaces[0].r)
            if (
                abs(float(pupilr) - fb_r) < 1e-9
                and abs(float(pupilz) - fb_z) < 1e-9
            ):
                bad = theta_t.sum() * 0.0 - 1e9
                zero_measures = [theta_t[0] * 0.0 for _ in range(K)]
                metrics = {
                    "invalid_fraction": 1.0,
                    "rejected": 1.0,
                    "rejected_pupil": 1.0,
                }
                return bad, zero_measures, metrics

        fitness_aperture_xy = self._fitness_unit_disk * float(pupilr)
        pupil_z_f = float(pupilz)

        field_pts = self._field_points(hfov)
        rms_per_wvln, inv_per_wvln = [], []
        trace_dtype = lens.surfaces[
            self.cfg.free_curvature_indices[0]
        ].c.dtype
        for w in self.cfg.fitness_wavelengths_um:
            ray = trace_fields_batch(
                lens, field_pts, fitness_aperture_xy, pupil_z_f,
                object_depth_mm=self.cfg.object_depth_mm,
                wavelength_um=float(w), device=device, dtype=trace_dtype,
            )
            rms, inv_frac = rms_spot_batch(ray)
            rms_per_wvln.append(rms)
            inv_per_wvln.append(inv_frac)

        rms_all = torch.stack(rms_per_wvln)   # (W, F)
        inv_all = torch.stack(inv_per_wvln)

        # Apply per-field weights (default: uniform).
        if self.cfg.field_weights is not None:
            fw = torch.tensor(
                self.cfg.field_weights, dtype=rms_all.dtype, device=rms_all.device
            )
            fw = fw / fw.sum()  # normalize
            rms_mean = (rms_all * fw.unsqueeze(0)).sum() / rms_all.shape[0]
            invalid_fraction_mean = (inv_all * fw.unsqueeze(0)).sum() / inv_all.shape[0]
        else:
            rms_mean = rms_all.mean()
            invalid_fraction_mean = inv_all.mean()
        fitness = -rms_mean

        # Intersection penalty.
        intersect_loss = intersection_penalty(
            lens,
            air_min_mm=self.cfg.air_min_mm,
            thick_min_mm=self.cfg.thick_min_mm,
            device=device,
        )
        if float(intersect_loss.detach()) > 0:
            fitness = fitness - self.cfg.intersect_penalty_weight * intersect_loss

        # Invalid ray fraction penalty.
        if float(invalid_fraction_mean.detach().item()) > self.cfg.invalid_tol:
            excess = invalid_fraction_mean - self.cfg.invalid_tol
            fitness = fitness - self.cfg.invalid_penalty_weight * excess

        # Soft EFL penalty.
        efl_actual = self._full_dof_solver.efl(c_full, gaps_full, n_bef=n_bef, n_aft=n_aft)
        efl_error_rel = (efl_actual - self._efl_target) / self._efl_target
        efl_penalty = self._efl_penalty_weight * efl_error_rel ** 2
        fitness = fitness - efl_penalty

        # Soft F-number penalty (gated by enforce_fnum). Differentiable via
        # the FullDOFParaxialSolver.paraxial_pupil_radius walk through the
        # front group, so gradients flow through curvatures, gaps, and
        # materials. Skipped entirely when disabled — loss is unchanged for
        # legacy configs.
        fnum_actual_t = None
        fnum_dev_t = None
        fnum_penalty_t = None
        if self.cfg.enforce_fnum:
            aper_idx = self.template.stop_index
            r_stop_mm = float(lens.surfaces[aper_idx].r)
            pupilr_diff = self._full_dof_solver.paraxial_pupil_radius(
                c_full, gaps_full, aper_idx, r_stop_mm,
                n_bef=n_bef, n_aft=n_aft,
            )
            fnum_actual_t = efl_actual / (2.0 * pupilr_diff)
            # Relative error, matches the EFL penalty's dimensionless form.
            fnum_target = float(self.cfg.f_number_target)
            fnum_dev_t = (fnum_actual_t - fnum_target) / fnum_target
            fnum_penalty_t = self.cfg.fnum_penalty_weight * fnum_dev_t ** 2
            fitness = fitness - fnum_penalty_t

        # Descriptor.
        field_y_deg = float(field_pts[self.cfg.descriptor_field_index][1])
        m_t = self.descriptor.compute(
            lens, field_y_deg=field_y_deg,
            object_depth_mm=self.cfg.object_depth_mm,
            wavelength_um=self.cfg.wavelength_um, device=device,
            n_bef_override=n_bef, n_aft_override=n_aft,
            theta_tensor=theta_t,
        )
        measures_list = [m_t[k] for k in range(K)]

        # F# is reported in metrics regardless of enforce_fnum so all logs
        # and PNG titles can show it. Cheap: pupil radius was already
        # computed for ray sampling above.
        pupilr_metric = float(pupilr)
        fnum_actual_metric = (
            float(efl_actual.detach().cpu().item()) / (2.0 * pupilr_metric)
            if pupilr_metric > 1e-6 else float("nan")
        )

        metrics = {
            "rms_spot": float(rms_mean.detach().cpu().item()),
            "invalid_fraction": float(
                invalid_fraction_mean.detach().cpu().item()
            ),
            "efl_actual": float(efl_actual.detach().cpu().item()),
            "efl_error_pct": float(
                abs(efl_error_rel.detach().cpu().item()) * 100
            ),
            "efl_penalty": float(efl_penalty.detach().cpu().item()),
            "d_sensor": float(lens.d_sensor.detach().cpu().item()),
            "flange": float(bfl_val),
            "fnum_actual": fnum_actual_metric,
        }
        if fnum_penalty_t is not None:
            # When the soft penalty is on, the differentiable F# matches the
            # one above to floating-point precision; also report deviation
            # and penalty value.
            metrics["fnum_actual"] = float(fnum_actual_t.detach().cpu().item())
            metrics["fnum_deviation"] = float(fnum_dev_t.detach().cpu().item())
            metrics["fnum_penalty"] = float(fnum_penalty_t.detach().cpu().item())
        for k, name in enumerate(self.descriptor.coord_names):
            metrics[f"m{k}_{name}"] = float(
                measures_list[k].detach().cpu().item()
            )
        last_all = getattr(self.descriptor, "_last_all_seidel", None)
        if last_all is not None:
            for j, label in enumerate(("S1", "S2", "S3", "S4", "S5")):
                metrics[f"seidel_{label}"] = float(last_all[j])
        return fitness, measures_list, metrics

    def _calc_hfov_deg(self, theta_t: torch.Tensor) -> float:
        curvatures, gaps_free, _, nV_pairs = self._unpack_theta(theta_t)
        c_full = self._build_c_full(curvatures, theta_t)
        gaps_full = self._build_gaps_full(gaps_free, theta_t)
        n_bef, n_aft = self._build_n_full(nV_pairs, theta_t)
        efl = float(self._full_dof_solver.efl(c_full, gaps_full, n_bef=n_bef, n_aft=n_aft).detach())
        if abs(efl) < 1e-6:
            efl = self._efl_target
        half_sensor = self.template.sensor_size_mm[0] / 2.0
        return math.degrees(math.atan(half_sensor / abs(efl)))
