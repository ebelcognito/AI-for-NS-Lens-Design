"""Power-budget reparameterization for LensMEGA.

Instead of CMA-MEGA searching over raw curvatures, theta controls the
fraction of total system power each surface contributes.  Total power is
fixed at ``1 / focal_length_target``, so the focal length is approximately
correct by construction (exact in the thin-lens limit).  A soft quadratic
EFL penalty handles the thick-lens residual.

This keeps every surface as a genuine free variable — no single surface
is "consumed" by a focal-length solve.  CMA-MEGA explores the distribution
of optical power across surfaces rather than raw curvatures.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .base import EvalResult
from .lens_problem import LensProblem, LensProblemConfig
from .penalties import intersection_penalty
from .tracing import (
    disk_samples,
    latin_hypercube,
    project_box,
    rms_spot_batch,
    trace_fields_batch,
)


class PowerBudgetLensProblem(LensProblem):
    """LensProblem variant where theta parameterises power fractions.

    CMA-MEGA proposes theta in bounded space.  This class normalises them
    to fractions that sum to 1, multiplies by ``P_total = 1/target_efl``,
    then converts to curvatures via ``c_i = power_i / delta_n_i``.

    Parameters
    ----------
    cfg : LensProblemConfig
        Must have ``derived_curvature_index = None`` and
        ``parameterization = "power_budget"``.
        ``focal_length_target_mm`` is used to set the total power.
        ``free_curvature_indices`` lists ALL refractive surfaces.
    descriptor : Descriptor
    """

    def __init__(self, cfg: LensProblemConfig, descriptor) -> None:
        super().__init__(cfg, descriptor)
        if cfg.focal_length_target_mm is None:
            raise ValueError(
                "PowerBudgetLensProblem requires focal_length_target_mm"
            )
        self._efl_target = cfg.focal_length_target_mm
        self._efl_penalty_weight = cfg.efl_penalty_weight
        self._delta_n_cache: Optional[torch.Tensor] = None

    def _ensure_backend(self) -> None:
        super()._ensure_backend()
        device = self.cfg.device
        delta_ns = []
        for idx in self.cfg.free_curvature_indices:
            surf = self.template.surfaces[idx]
            delta_ns.append(surf.n_after - surf.n_before)
        self._delta_n_cache = torch.tensor(
            delta_ns, dtype=torch.float64, device=device
        )

    # ------------------------------------------------------------------
    # Power-fraction → curvature conversion
    # ------------------------------------------------------------------

    def _theta_to_curvatures(
        self, theta_t: torch.Tensor
    ) -> torch.Tensor:
        """Convert power-fraction theta to curvatures.

        Normalises so fractions sum to 1, then:
            power_i  = fraction_i * P_total
            c_i      = power_i / delta_n_i

        Negative fractions are allowed (negative elements).
        """
        P_total = 1.0 / self._efl_target
        theta_sum = theta_t.sum()
        safe_sum = torch.where(
            torch.abs(theta_sum) > 1e-6,
            theta_sum,
            torch.full_like(theta_sum, 1e-6),
        )
        fractions = theta_t / safe_sum
        powers = fractions * P_total
        return powers / self._delta_n_cache

    def _curvatures_to_c_full(
        self, curvatures: torch.Tensor, theta_t: torch.Tensor
    ) -> torch.Tensor:
        """Build the full curvature vector from power-budget curvatures."""
        parts = []
        free_set = set(self.cfg.free_curvature_indices)
        cursor = 0
        for i in range(len(self.template.surfaces)):
            if i in free_set:
                parts.append(curvatures[cursor])
                cursor += 1
            else:
                parts.append(
                    torch.tensor(
                        self.template.surfaces[i].initial_c,
                        dtype=theta_t.dtype,
                        device=theta_t.device,
                    )
                )
        return torch.stack(parts)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _build_lens(self, theta_t: torch.Tensor):
        assert self.solver is not None and self.template is not None

        curvatures = self._theta_to_curvatures(theta_t)
        c_full = self._curvatures_to_c_full(curvatures, theta_t)

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
        c_full = self._curvatures_to_c_full(curvatures, theta_t)

        bfl_val = float(self.solver.bfl(c_full).detach().item())
        if bfl_val <= 0.0:
            bad = theta_t.sum() * 0.0 - 1e9
            zero_measures = [theta_t[0] * 0.0 for _ in range(K)]
            metrics = {
                "invalid_fraction": 1.0,
                "rejected": 1.0,
                "rejected_bfl": 1.0,
            }
            return bad, zero_measures, metrics

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
                lens,
                field_pts,
                fitness_aperture_xy,
                pupil_z_f,
                object_depth_mm=self.cfg.object_depth_mm,
                wavelength_um=float(w),
                device=device,
                dtype=trace_dtype,
            )
            rms, inv_frac = rms_spot_batch(ray)
            rms_per_wvln.append(rms)
            inv_per_wvln.append(inv_frac)

        rms_all = torch.stack(rms_per_wvln)
        inv_all = torch.stack(inv_per_wvln)
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

        # Soft EFL penalty for thick-lens residual.
        efl_actual = self.solver.efl(c_full)
        efl_error_rel = (efl_actual - self._efl_target) / self._efl_target
        efl_penalty = self._efl_penalty_weight * efl_error_rel ** 2
        fitness = fitness - efl_penalty

        # Soft F-number penalty (gated by enforce_fnum). Differentiable
        # through curvatures via the paraxial walk; skipped entirely when
        # disabled so legacy configs are bit-identical.
        fnum_actual_t = None
        fnum_dev_t = None
        fnum_penalty_t = None
        if self.cfg.enforce_fnum:
            aper_idx = self.template.stop_index
            r_stop_mm = float(lens.surfaces[aper_idx].r)
            pupilr_diff = self.solver.paraxial_pupil_radius(
                c_full, aper_idx, r_stop_mm
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
            lens,
            field_y_deg=field_y_deg,
            object_depth_mm=self.cfg.object_depth_mm,
            wavelength_um=self.cfg.wavelength_um,
            device=device,
            theta_tensor=theta_t,
        )
        measures_list = [m_t[k] for k in range(K)]

        P_total = 1.0 / self._efl_target
        powers = curvatures * self._delta_n_cache
        power_sum = float(powers.sum().detach().cpu().item())

        # F# always reported in metrics for logging/PNG titles regardless
        # of enforce_fnum. Pupil radius was already computed for ray
        # sampling above.
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
            "power_sum": power_sum,
            "power_sum_target": P_total,
            "fnum_actual": fnum_actual_metric,
        }
        if fnum_penalty_t is not None:
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
        curvatures = self._theta_to_curvatures(theta_t)
        c_full = self._curvatures_to_c_full(curvatures, theta_t)
        efl = float(self.solver.efl(c_full).detach())
        if abs(efl) < 1e-6:
            efl = self._efl_target
        half_sensor = self.template.sensor_size_mm[0] / 2.0
        return math.degrees(math.atan(half_sensor / abs(efl)))
