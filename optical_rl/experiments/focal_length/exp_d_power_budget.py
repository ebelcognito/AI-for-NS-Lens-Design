"""Experiment D: Power-budget reparameterization.

Instead of CMA-MEGA searching over raw curvatures, theta controls the
**fraction of total system power** each surface contributes.  The total
optical power is fixed at ``P_total = 1/target_efl``, and each ``theta_i``
controls how much of that power surface ``i`` carries.

    curvature_i = (fraction_i * P_total) / (n_after_i - n_before_i)

This keeps the total power approximately correct by construction (exact in
the thin-lens / contact-lens limit).  For separated elements the Gullstrand
equation means the actual EFL will differ from the target, so a soft EFL
penalty handles the residual.

No surface is "special" — CMA-MEGA explores the distribution of optical
power across all surfaces, and the total is constrained.
"""
from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import numpy as np
import torch

from optical_rl.qd.lensmega.problem.lens_problem import LensProblem, LensProblemConfig
from optical_rl.qd.lensmega.problem.base import EvalResult
from optical_rl.qd.lensmega.problem.penalties import intersection_penalty
from optical_rl.qd.lensmega.problem.tracing import (
    disk_samples,
    latin_hypercube,
    project_box,
    rms_spot_batch,
    trace_fields_batch,
)


class PowerBudgetLensProblem(LensProblem):
    """LensProblem variant where theta parameterises power fractions.

    CMA-MEGA proposes ``theta`` in unbounded space.  This class maps theta
    to power fractions that sum to 1 (via normalisation), then to curvatures
    via the thin-lens relation.  A soft EFL penalty corrects for the
    thick-lens residual.

    Parameters
    ----------
    cfg : LensProblemConfig
        ``derived_curvature_index`` must be None.
        ``free_curvature_indices`` lists ALL refractive surfaces.
    descriptor : Descriptor
    efl_target_mm : float
        Target effective focal length.
    efl_penalty_weight : float
        Weight on the residual EFL penalty.
    """

    def __init__(self, cfg, descriptor, *, efl_target_mm: float, efl_penalty_weight: float = 5.0):
        super().__init__(cfg, descriptor)
        self.efl_target_mm = efl_target_mm
        self.efl_penalty_weight = efl_penalty_weight
        self._delta_n_cache: Optional[torch.Tensor] = None

    def _ensure_backend(self) -> None:
        super()._ensure_backend()
        # Cache the refractive index differences for each free surface.
        device = self.cfg.device
        delta_ns = []
        for idx in self.cfg.free_curvature_indices:
            surf = self.template.surfaces[idx]
            delta_ns.append(surf.n_after - surf.n_before)
        self._delta_n_cache = torch.tensor(delta_ns, dtype=torch.float64, device=device)

    def _theta_to_curvatures(self, theta_t: torch.Tensor) -> torch.Tensor:
        """Convert power-fraction theta to curvatures.

        ``theta`` is in raw (unbounded) space — each entry controls the
        relative power of one surface.  We normalise so fractions sum to 1,
        then convert via ``c_i = (f_i * P_total) / delta_n_i``.

        Note: fractions can be negative (negative elements exist in real
        lenses), so we do NOT use softmax.  Instead: ``f_i = theta_i / sum(theta)``.
        If ``sum(theta)`` is near zero (all powers cancel), fall back to
        uniform distribution.
        """
        P_total = 1.0 / self.efl_target_mm
        theta_sum = theta_t.sum()

        # Avoid division by near-zero sum.
        safe_sum = torch.where(
            torch.abs(theta_sum) > 1e-6,
            theta_sum,
            torch.full_like(theta_sum, 1e-6),
        )
        fractions = theta_t / safe_sum  # sum to 1.0

        # power_i = fraction_i * P_total
        # c_i = power_i / delta_n_i
        powers = fractions * P_total
        curvatures = powers / self._delta_n_cache
        return curvatures

    def _build_lens(self, theta_t: torch.Tensor):
        """Build lens from power-budget parameterisation."""
        assert self.solver is not None and self.template is not None

        curvatures = self._theta_to_curvatures(theta_t)

        # Build the full curvature vector (with non-free surfaces at their
        # initial values).
        c_full_parts = []
        free_set = set(self.cfg.free_curvature_indices)
        free_cursor = 0
        for i in range(len(self.template.surfaces)):
            if i in free_set:
                c_full_parts.append(curvatures[free_cursor])
                free_cursor += 1
            else:
                c_full_parts.append(
                    torch.tensor(
                        self.template.surfaces[i].initial_c,
                        dtype=theta_t.dtype,
                        device=theta_t.device,
                    )
                )
        c_full = torch.stack(c_full_parts)

        # d_sensor from paraxial BFL.
        bfl_t = self.solver.bfl(c_full)
        d_sensor = self.template.z_last_surface_mm + bfl_t

        lens = self._reusable_lens
        dtype = lens.surfaces[self.cfg.free_curvature_indices[0]].c.dtype
        for i_local, i_global in enumerate(self.cfg.free_curvature_indices):
            lens.surfaces[i_global].c = curvatures[i_local].to(dtype=dtype)
        lens.d_sensor = d_sensor.to(dtype=dtype)
        return lens, bfl_t

    def _fitness_and_metrics(self, theta_t: torch.Tensor):
        assert self.solver is not None and self.template is not None
        K = self.descriptor.dim
        device = self.cfg.device

        curvatures = self._theta_to_curvatures(theta_t)

        # Build c_full for BFL check.
        free_set = set(self.cfg.free_curvature_indices)
        c_parts = []
        free_cursor = 0
        for i in range(len(self.template.surfaces)):
            if i in free_set:
                c_parts.append(curvatures[free_cursor])
                free_cursor += 1
            else:
                c_parts.append(
                    torch.tensor(
                        self.template.surfaces[i].initial_c,
                        dtype=theta_t.dtype,
                        device=theta_t.device,
                    )
                )
        c_full = torch.stack(c_parts)

        bfl_val = float(self.solver.bfl(c_full).detach().item())
        if bfl_val <= 0.0:
            bad = theta_t.sum() * 0.0 - 1e9
            zero_measures = [theta_t[0] * 0.0 for _ in range(K)]
            metrics = {"invalid_fraction": 1.0, "rejected": 1.0, "rejected_bfl": 1.0}
            return bad, zero_measures, metrics

        lens, _ = self._build_lens(theta_t)
        hfov = self._calc_hfov_deg(theta_t)

        with torch.no_grad():
            pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)

        if self.cfg.strict_entrance_pupil:
            fb_z = float(lens.surfaces[0].d.item())
            fb_r = float(lens.surfaces[0].r)
            if (abs(float(pupilr) - fb_r) < 1e-9
                    and abs(float(pupilz) - fb_z) < 1e-9):
                bad = theta_t.sum() * 0.0 - 1e9
                zero_measures = [theta_t[0] * 0.0 for _ in range(K)]
                metrics = {"invalid_fraction": 1.0, "rejected": 1.0, "rejected_pupil": 1.0}
                return bad, zero_measures, metrics

        fitness_aperture_xy = self._fitness_unit_disk * float(pupilr)
        pupil_z_f = float(pupilz)

        field_pts = self._field_points(hfov)
        rms_per_wvln, inv_per_wvln = [], []
        trace_dtype = lens.surfaces[self.cfg.free_curvature_indices[0]].c.dtype
        for w in self.cfg.fitness_wavelengths_um:
            ray = trace_fields_batch(
                lens, field_pts, fitness_aperture_xy, pupil_z_f,
                object_depth_mm=self.cfg.object_depth_mm,
                wavelength_um=float(w), device=device, dtype=trace_dtype,
            )
            rms, inv_frac = rms_spot_batch(ray)
            rms_per_wvln.append(rms)
            inv_per_wvln.append(inv_frac)

        rms_all = torch.stack(rms_per_wvln)
        inv_all = torch.stack(inv_per_wvln)
        rms_mean = rms_all.mean()
        invalid_fraction_mean = inv_all.mean()
        fitness = -rms_mean

        # --- Intersection penalty ---
        intersect_loss = intersection_penalty(
            lens, air_min_mm=self.cfg.air_min_mm,
            thick_min_mm=self.cfg.thick_min_mm, device=device,
        )
        if float(intersect_loss.detach()) > 0:
            fitness = fitness - self.cfg.intersect_penalty_weight * intersect_loss

        # --- Invalid ray fraction penalty ---
        if float(invalid_fraction_mean.detach().item()) > self.cfg.invalid_tol:
            excess = invalid_fraction_mean - self.cfg.invalid_tol
            fitness = fitness - self.cfg.invalid_penalty_weight * excess

        # --- Soft EFL penalty for thick-lens residual ---
        efl_actual = self.solver.efl(c_full)
        efl_error_rel = (efl_actual - self.efl_target_mm) / self.efl_target_mm
        efl_penalty = self.efl_penalty_weight * efl_error_rel ** 2
        fitness = fitness - efl_penalty

        # --- Power distribution diagnostics ---
        P_total = 1.0 / self.efl_target_mm
        powers = curvatures * self._delta_n_cache
        power_sum = float(powers.sum().detach().cpu().item())

        # --- Descriptor ---
        field_y_deg = float(field_pts[self.cfg.descriptor_field_index][1])
        m_t = self.descriptor.compute(
            lens, field_y_deg=field_y_deg,
            object_depth_mm=self.cfg.object_depth_mm,
            wavelength_um=self.cfg.wavelength_um, device=device,
        )
        measures_list = [m_t[k] for k in range(K)]

        metrics = {
            "rms_spot": float(rms_mean.detach().cpu().item()),
            "invalid_fraction": float(invalid_fraction_mean.detach().cpu().item()),
            "efl_actual": float(efl_actual.detach().cpu().item()),
            "efl_error_rel": float(efl_error_rel.detach().cpu().item()),
            "efl_penalty": float(efl_penalty.detach().cpu().item()),
            "power_sum": power_sum,
            "power_sum_target": P_total,
        }
        for k, name in enumerate(self.descriptor.coord_names):
            metrics[f"m{k}_{name}"] = float(measures_list[k].detach().cpu().item())
        return fitness, measures_list, metrics

    def _calc_hfov_deg(self, theta_t: torch.Tensor) -> float:
        """Compute half-FoV using actual EFL from the power-budget curvatures."""
        curvatures = self._theta_to_curvatures(theta_t)
        free_set = set(self.cfg.free_curvature_indices)
        c_parts = []
        free_cursor = 0
        for i in range(len(self.template.surfaces)):
            if i in free_set:
                c_parts.append(curvatures[free_cursor])
                free_cursor += 1
            else:
                c_parts.append(
                    torch.tensor(
                        self.template.surfaces[i].initial_c,
                        dtype=theta_t.dtype,
                        device=theta_t.device,
                    )
                )
        c_full = torch.stack(c_parts)
        efl = float(self.solver.efl(c_full).detach())
        if abs(efl) < 1e-6:
            efl = self.efl_target_mm
        half_sensor = self.template.sensor_size_mm[0] / 2.0
        return math.degrees(math.atan(half_sensor / abs(efl)))
