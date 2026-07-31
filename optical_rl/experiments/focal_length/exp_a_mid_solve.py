"""Experiment A: Generalized mid-surface paraxial solve.

Extends the paraxial ABCD solve so ANY surface can be the derived one — not
just the last refractive surface.  This is achieved by tracing the marginal
ray ``[y=1, nu=0]`` in two halves:

  R = subsystem matrix for surfaces ``[0, i_d)``   (before the derived surface)
  L = subsystem matrix for surfaces ``(i_d, N)``   (after the derived surface)

Then solving ``c_d`` so that the full system achieves ``nu_final = -1/EFL``.

This generalises the existing ``derive_curvature_for_efl`` which only works
when the derived surface is the last refractive surface.

Reference: ``automated_lens_design/dlt/system_matrix.py:138-169``
"""
from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import numpy as np
import torch

from optical_rl.qd.lensmega.problem.lens_problem import LensProblem, LensProblemConfig
from optical_rl.qd.lensmega.problem.lens_template import LensTemplate, Surface, load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.lensmega.problem.base import EvalResult
from optical_rl.qd.lensmega.problem.penalties import intersection_penalty
from optical_rl.qd.lensmega.problem.tracing import (
    disk_samples,
    latin_hypercube,
    project_box,
    rms_spot_batch,
    trace_fields_batch,
)


class MidSurfaceParaxialSolver(GeneralizedParaxialSolver):
    """Paraxial solver that can derive ANY surface's curvature for a target EFL.

    The key addition is ``_build_after_matrix`` which traces the subsystem
    AFTER the derived surface, and ``derive_curvature_general`` which uses
    both halves to solve for ``c_d``.
    """

    def _build_after_matrix(
        self, c_full: torch.Tensor, start_inclusive: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build 2x2 transfer matrix for surfaces ``[start_inclusive, N)``.

        The last surface gets refraction only (no transfer), matching
        ``trace_marginal`` convention where BFL is computed from the exit of
        the last refractive surface.

        Returns the four elements ``(L00, L01, L10, L11)`` of the matrix as
        separate scalars to keep autograd happy.
        """
        N = self._num_surfaces
        dtype = c_full.dtype
        device = c_full.device

        # Start with identity.
        L00 = torch.ones(1, dtype=dtype, device=device)
        L01 = torch.zeros(1, dtype=dtype, device=device)
        L10 = torch.zeros(1, dtype=dtype, device=device)
        L11 = torch.ones(1, dtype=dtype, device=device)

        for i in range(start_inclusive, N):
            surf = self.template.surfaces[i]
            phi = (surf.n_after - surf.n_before) * c_full[i]

            # Refraction: R = [[1, 0], [-phi, 1]]
            # L_new = R @ L
            new_L00 = L00
            new_L01 = L01
            new_L10 = L10 - phi * L00
            new_L11 = L11 - phi * L01

            if i < N - 1:
                # Transfer: T = [[1, alpha], [0, 1]] where alpha = gap/n_after
                alpha = surf.gap_to_next / surf.n_after
                L00 = new_L00 + alpha * new_L10
                L01 = new_L01 + alpha * new_L11
                L10 = new_L10
                L11 = new_L11
            else:
                # Last surface: no transfer.
                L00 = new_L00
                L01 = new_L01
                L10 = new_L10
                L11 = new_L11

        return L00, L01, L10, L11

    def derive_curvature_general(
        self,
        c_full: torch.Tensor,
        target_efl: float,
        derived_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve curvature at ``derived_index`` so the system has EFL = target_efl.

        Handles any surface position, including mid-system surfaces with
        refractive elements after them.

        The derivation:
          1. Trace ``[y=1, nu=0]`` through surfaces ``[0, i_d)`` → ``(y_R, nu_R)``
          2. At surface ``i_d``: refraction ``nu_mid = nu_R - phi_d * y_R``
          3. Transfer after ``i_d``: ``y_mid = y_R + alpha_d * nu_mid``
             (skipped if ``i_d`` is last surface)
          4. Through after-subsystem ``L``: require ``nu_final = -1/EFL``

        Solving for ``phi_d``:
          ``phi_d = [L10*y_R + (L10*alpha_d + L11)*nu_R + P] /
                    [y_R * (L10*alpha_d + L11)]``

        where ``P = 1/target_efl``.

        Returns ``(c_derived, d_sensor, bfl)`` — all differentiable.
        """
        i_d = derived_index
        N = self._num_surfaces
        surf_d = self.template.surfaces[i_d]
        delta_n = surf_d.n_after - surf_d.n_before
        P = 1.0 / float(target_efl)

        # Trace before the derived surface.
        y_R, nu_R = self._walk(c_full, stop_exclusive=i_d)

        if i_d == N - 1:
            # Last surface: no transfer after, no "after" subsystem.
            # Simplifies to the existing formula.
            safe_y = torch.where(
                torch.abs(y_R) > 1e-6, y_R, torch.full_like(y_R, 1e-6)
            )
            phi_d = (nu_R + P) / safe_y
            c_derived = phi_d / delta_n

            # After refraction at the derived (last) surface:
            # nu_final = nu_R - phi_d * y_R = -P
            # y_final = y_R (no transfer)
            # BFL = -y_final / nu_final = y_R * target_efl
            bfl = y_R * float(target_efl)
        else:
            # General case: there are surfaces after the derived one.
            alpha_d = surf_d.gap_to_next / surf_d.n_after

            # Build the "after" matrix for surfaces [i_d+1, N).
            L00, L01, L10, L11 = self._build_after_matrix(
                c_full, start_inclusive=i_d + 1
            )

            # Solve for phi_d.
            # phi_d = [L10*y_R + (L10*alpha_d + L11)*nu_R + P] /
            #         [y_R * (L10*alpha_d + L11)]
            coeff = L10 * alpha_d + L11
            numerator = L10 * y_R + coeff * nu_R + P
            denominator = y_R * coeff

            safe_denom = torch.where(
                torch.abs(denominator) > 1e-10,
                denominator,
                torch.full_like(denominator, 1e-10),
            )
            phi_d = numerator / safe_denom
            c_derived = phi_d / delta_n

            # Compute the final ray state to get BFL.
            nu_mid = nu_R - phi_d * y_R
            y_mid = y_R + alpha_d * nu_mid

            # Through the "after" subsystem:
            y_final = L00 * y_mid + L01 * nu_mid
            # nu_final = -P by construction

            # BFL = -y_final / nu_final = y_final * target_efl
            bfl = y_final * float(target_efl)

        d_sensor = self.template.z_last_surface_mm + bfl.squeeze()
        return c_derived.squeeze(), d_sensor, bfl.squeeze()


class MidSolveLensProblem(LensProblem):
    """LensProblem variant where any surface (not just the last) is derived.

    Uses ``MidSurfaceParaxialSolver`` to analytically solve the chosen
    surface's curvature so the system always has the target EFL.

    Parameters
    ----------
    cfg : LensProblemConfig
        ``derived_curvature_index`` should be set to the desired mid-surface.
        ``focal_length_target_mm`` specifies the target.
    descriptor : Descriptor
    mid_solve_index : int
        Which surface to use as the derived (solve) surface.  Can be any
        refractive surface — doesn't have to be the last one.
    """

    def __init__(self, cfg, descriptor, *, mid_solve_index: int, efl_target_mm: float):
        # The config must have derived_curvature_index=None to avoid the
        # LensTemplate validation that only allows last-surface solves.
        # We handle the mid-surface solve entirely within this class.
        super().__init__(cfg, descriptor)
        self.mid_solve_index = mid_solve_index
        self.efl_target_mm = efl_target_mm

    def _ensure_backend(self) -> None:
        """Override to use MidSurfaceParaxialSolver and bypass the
        LensTemplate validation that forbids mid-system derived surfaces.
        """
        if self._backend_loaded:
            return
        # Load template+lens with derived_curvature_index=None (bypasses
        # the assertion that the derived surface must be the last one).
        # The config should already have derived_curvature_index=None.
        super()._ensure_backend()
        # Replace with our generalised solver that can derive any surface.
        self.solver = MidSurfaceParaxialSolver(self.template)

    def _build_lens(self, theta_t: torch.Tensor):
        """Build lens with mid-surface curvature solved for target EFL."""
        assert self.solver is not None and self.template is not None
        assert isinstance(self.solver, MidSurfaceParaxialSolver)

        c_full = self.solver.expand_theta(theta_t)

        if self.efl_target_mm is not None:
            c_derived, d_sensor, bfl_t = self.solver.derive_curvature_general(
                c_full, self.efl_target_mm, self.mid_solve_index,
            )
            c_full = c_full.clone()
            c_full[self.mid_solve_index] = c_derived
        else:
            bfl_t = self.solver.bfl(c_full)
            d_sensor = self.template.z_last_surface_mm + bfl_t

        lens = self._reusable_lens
        dtype = lens.surfaces[self.cfg.free_curvature_indices[0]].c.dtype

        # Set all free curvatures + the derived one.
        indices_to_set = set(self.cfg.free_curvature_indices)
        indices_to_set.add(self.mid_solve_index)
        for i in indices_to_set:
            lens.surfaces[i].c = c_full[i].to(dtype=dtype)
        lens.d_sensor = d_sensor.to(dtype=dtype)
        return lens, bfl_t

    def _fitness_and_metrics(self, theta_t: torch.Tensor):
        """Standard fitness eval but with mid-surface solve."""
        assert self.solver is not None and self.template is not None
        assert isinstance(self.solver, MidSurfaceParaxialSolver)
        K = self.descriptor.dim
        device = self.cfg.device

        # Check BFL positivity.
        c_full_check = self.solver.expand_theta(theta_t)
        if self.efl_target_mm is not None:
            c_derived, _, bfl_t = self.solver.derive_curvature_general(
                c_full_check, self.efl_target_mm, self.mid_solve_index,
            )
            bfl_val = float(bfl_t.detach().item())
            # Also check for extreme derived curvatures.
            c_derived_val = float(c_derived.detach().item())
        else:
            bfl_val = float(self.solver.bfl(c_full_check).detach().item())
            c_derived_val = 0.0

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

        intersect_loss = intersection_penalty(
            lens, air_min_mm=self.cfg.air_min_mm,
            thick_min_mm=self.cfg.thick_min_mm, device=device,
        )
        if float(intersect_loss.detach()) > 0:
            fitness = fitness - self.cfg.intersect_penalty_weight * intersect_loss

        if float(invalid_fraction_mean.detach().item()) > self.cfg.invalid_tol:
            excess = invalid_fraction_mean - self.cfg.invalid_tol
            fitness = fitness - self.cfg.invalid_penalty_weight * excess

        # Descriptor.
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
            "c_derived": c_derived_val,
            "mid_solve_index": float(self.mid_solve_index),
        }
        for k, name in enumerate(self.descriptor.coord_names):
            metrics[f"m{k}_{name}"] = float(measures_list[k].detach().cpu().item())
        return fitness, measures_list, metrics
