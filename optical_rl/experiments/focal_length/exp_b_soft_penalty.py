"""Experiment B: All-free surfaces + soft EFL penalty.

No derived surface at all.  Every curvature surface is a free variable for
CMA-MEGA.  A soft quadratic penalty steers the system toward the target EFL.
``d_sensor`` is computed from the paraxial BFL so the image always lands at
the system's natural focus.

The key difference from the baseline: no single surface is "consumed" by the
focal-length constraint.  CMA-MEGA can explore all N curvatures freely, and
the penalty guides — rather than forces — the system toward the target EFL.
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


class SoftPenaltyLensProblem(LensProblem):
    """LensProblem variant where EFL is enforced via a soft penalty, not a solve.

    All curvature surfaces are free.  ``d_sensor`` is set to the paraxial
    back-focal-length position so the image plane always sits at the system's
    natural focus.  A quadratic penalty on relative EFL error is added to
    fitness.

    Parameters
    ----------
    cfg : LensProblemConfig
        Must have ``derived_curvature_index = None`` and
        ``focal_length_target_mm = None`` (the penalty handles it).
    descriptor : Descriptor
        The measure descriptor (Zernike, Seidel, etc.).
    efl_target_mm : float
        Target effective focal length in mm.
    efl_penalty_weight : float
        Multiplier on the quadratic relative-EFL error.
    """

    def __init__(self, cfg, descriptor, *, efl_target_mm: float, efl_penalty_weight: float = 10.0):
        super().__init__(cfg, descriptor)
        self.efl_target_mm = efl_target_mm
        self.efl_penalty_weight = efl_penalty_weight

    def _build_lens(self, theta_t: torch.Tensor):
        """Build lens with all-free curvatures; d_sensor from paraxial BFL."""
        assert self.solver is not None and self.template is not None
        c_full = self.solver.expand_theta(theta_t)

        bfl_t = self.solver.bfl(c_full)
        d_sensor = self.template.z_last_surface_mm + bfl_t

        lens = self._reusable_lens
        dtype = lens.surfaces[self.cfg.free_curvature_indices[0]].c.dtype
        for i in range(len(self.template.surfaces)):
            if i in self.cfg.free_curvature_indices:
                lens.surfaces[i].c = c_full[i].to(dtype=dtype)
        lens.d_sensor = d_sensor.to(dtype=dtype)
        return lens, bfl_t

    def _fitness_and_metrics(self, theta_t: torch.Tensor):
        """Evaluate fitness with soft EFL penalty instead of hard solve."""
        assert self.solver is not None and self.template is not None
        K = self.descriptor.dim
        device = self.cfg.device

        # Check BFL positivity — reject diverging lenses.
        c_full_check = self.solver.expand_theta(theta_t)
        bfl_val = float(self.solver.bfl(c_full_check).detach().item())
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

        # --- Intersection penalty (same as base) ---
        intersect_loss = intersection_penalty(
            lens, air_min_mm=self.cfg.air_min_mm,
            thick_min_mm=self.cfg.thick_min_mm, device=device,
        )
        if float(intersect_loss.detach()) > 0:
            fitness = fitness - self.cfg.intersect_penalty_weight * intersect_loss

        # --- Invalid ray fraction penalty (same as base) ---
        if float(invalid_fraction_mean.detach().item()) > self.cfg.invalid_tol:
            excess = invalid_fraction_mean - self.cfg.invalid_tol
            fitness = fitness - self.cfg.invalid_penalty_weight * excess

        # --- SOFT EFL PENALTY (the new part) ---
        efl_actual = self.solver.efl(c_full_check)
        efl_error_rel = (efl_actual - self.efl_target_mm) / self.efl_target_mm
        efl_penalty = self.efl_penalty_weight * efl_error_rel ** 2
        fitness = fitness - efl_penalty

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
        }
        for k, name in enumerate(self.descriptor.coord_names):
            metrics[f"m{k}_{name}"] = float(measures_list[k].detach().cpu().item())
        return fitness, measures_list, metrics

    def _calc_hfov_deg(self, theta_t: torch.Tensor) -> float:
        """Compute half-FoV from actual EFL (not assumed target)."""
        assert self.solver is not None and self.template is not None
        c_full = self.solver.expand_theta(theta_t)
        efl = float(self.solver.efl(c_full).detach())
        if abs(efl) < 1e-6:
            efl = self.efl_target_mm
        half_sensor = self.template.sensor_size_mm[0] / 2.0
        return math.degrees(math.atan(half_sensor / abs(efl)))
