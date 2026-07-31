"""Experiment C: Image-height focal loss (real-ray).

Instead of constraining paraxial EFL, penalise the real-ray image height
deviation from ``f_target * tan(field_angle)``.  This distributes the
focal-length constraint through the actual ray trace — every surface that
bends a ray contributes to the gradient signal.

All curvatures are free (no derived surface).  ``d_sensor`` is set from
the paraxial BFL as in Experiment B.  The focal-length signal comes entirely
from the image-height loss on off-axis field points.

Requires ``stage: stage2`` so that off-axis field points are traced (on-axis
gives no focal info because ``tan(0) = 0``).

Reference: ``automated_lens_design/dlt/opt_utils.py:272-282``
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
    project_box,
    rms_spot_batch,
    trace_fields_batch,
)

from .common import rms_spot_batch_with_centroids


class ImageHeightLensProblem(LensProblem):
    """LensProblem variant using real-ray image-height loss for focal length.

    All curvature surfaces are free.  The focal length constraint is enforced
    by penalising the deviation of the actual spot centroid height from the
    ideal ``f * tan(angle)`` at each off-axis field point.  Gradients flow
    through the full ray trace to all surfaces.

    Parameters
    ----------
    cfg : LensProblemConfig
        Must have ``derived_curvature_index = None``.
    descriptor : Descriptor
        The measure descriptor.
    efl_target_mm : float
        Target effective focal length.
    focal_loss_weight : float
        Weight on the image-height focal loss.
    efl_penalty_weight : float
        Optional additional soft paraxial EFL penalty weight (belt-and-suspenders).
        Set to 0 to rely solely on the image-height loss.
    """

    def __init__(
        self, cfg, descriptor, *,
        efl_target_mm: float,
        focal_loss_weight: float = 5.0,
        efl_penalty_weight: float = 1.0,
    ):
        super().__init__(cfg, descriptor)
        self.efl_target_mm = efl_target_mm
        self.focal_loss_weight = focal_loss_weight
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
        assert self.solver is not None and self.template is not None
        K = self.descriptor.dim
        device = self.cfg.device

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
        # Collect centroids from the design wavelength for the focal loss.
        centroids_design = None
        trace_dtype = lens.surfaces[self.cfg.free_curvature_indices[0]].c.dtype
        for w in self.cfg.fitness_wavelengths_um:
            ray = trace_fields_batch(
                lens, field_pts, fitness_aperture_xy, pupil_z_f,
                object_depth_mm=self.cfg.object_depth_mm,
                wavelength_um=float(w), device=device, dtype=trace_dtype,
            )
            rms, inv_frac, centroid = rms_spot_batch_with_centroids(ray)
            rms_per_wvln.append(rms)
            inv_per_wvln.append(inv_frac)
            # Use the design wavelength (d-line) centroids for focal loss.
            if abs(float(w) - self.cfg.wavelength_um) < 1e-6:
                centroids_design = centroid  # shape (F, 2)

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

        # --- IMAGE-HEIGHT FOCAL LOSS (the new part) ---
        focal_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        n_focal_fields = 0
        if centroids_design is not None:
            for f_idx, (fx, fy) in enumerate(field_pts):
                field_angle_deg = fy
                if abs(field_angle_deg) < 1e-6:
                    continue  # skip on-axis: tan(0) = 0, no info
                ideal_height = self.efl_target_mm * math.tan(math.radians(field_angle_deg))
                actual_height = centroids_design[f_idx, 1]  # y-component
                focal_loss = focal_loss + (actual_height - ideal_height) ** 2
                n_focal_fields += 1
            if n_focal_fields > 0:
                focal_loss = focal_loss / n_focal_fields

        fitness = fitness - self.focal_loss_weight * focal_loss

        # --- Optional soft paraxial EFL penalty (belt-and-suspenders) ---
        efl_actual = self.solver.efl(c_full_check)
        efl_penalty = torch.tensor(0.0, device=device)
        if self.efl_penalty_weight > 0:
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
            "focal_loss": float(focal_loss.detach().cpu().item()),
            "efl_penalty": float(efl_penalty.detach().cpu().item()),
        }
        for k, name in enumerate(self.descriptor.coord_names):
            metrics[f"m{k}_{name}"] = float(measures_list[k].detach().cpu().item())
        return fitness, measures_list, metrics

    def _calc_hfov_deg(self, theta_t: torch.Tensor) -> float:
        assert self.solver is not None and self.template is not None
        c_full = self.solver.expand_theta(theta_t)
        efl = float(self.solver.efl(c_full).detach())
        if abs(efl) < 1e-6:
            efl = self.efl_target_mm
        half_sensor = self.template.sensor_size_mm[0] / 2.0
        return math.degrees(math.atan(half_sensor / abs(efl)))
