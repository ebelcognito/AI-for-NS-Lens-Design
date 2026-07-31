from __future__ import annotations

from typing import Tuple

import torch

from .lens_template import LensTemplate


class GeneralizedParaxialSolver:
    """Differentiable paraxial ABCD marginal-ray solver for arbitrary templates.

    Walks a LensTemplate surface by surface in reduced coordinates
    ``[y, nu = n * u]``.  For each surface:

      - refraction : ``nu -= (n_after - n_before) * c * y``
      - transfer   : ``y  += (gap_to_next / n_after) * nu``

    Stops have ``n_before == n_after == 1.0`` and ``c == 0``, so the refraction
    step is identically zero and the transfer step degenerates to ``y += gap *
    nu``.  No branching is required: the stop is just a transfer-only step.
    """

    def __init__(self, template: LensTemplate) -> None:
        self.template = template
        self._free_pos = {i: k for k, i in enumerate(template.free_indices)}
        self._num_surfaces = len(template.surfaces)

    def expand_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Insert ``theta`` into the free-curvature slots, initial values elsewhere.

        The derived slot (if any) is left at zero; it is irrelevant because
        ``derive_curvature_for_efl`` never reads it (it short-circuits the trace
        at ``derived_index``).
        """
        parts = []
        for i in range(self._num_surfaces):
            if i in self._free_pos:
                parts.append(theta[self._free_pos[i]])
            else:
                parts.append(
                    torch.tensor(
                        self.template.surfaces[i].initial_c,
                        dtype=theta.dtype,
                        device=theta.device,
                    )
                )
        return torch.stack(parts)

    def _walk(
        self,
        c_full: torch.Tensor,
        stop_exclusive: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trace surfaces ``[0, stop_exclusive)`` with full refract + transfer.

        Returns ``(y, nu)`` in reduced coordinates at the **entry** of
        ``stop_exclusive`` (i.e. the transfer after surface ``stop_exclusive-1``
        has been applied).
        """
        dtype = c_full.dtype
        device = c_full.device
        y = torch.ones(1, dtype=dtype, device=device)
        nu = torch.zeros(1, dtype=dtype, device=device)
        for i in range(stop_exclusive):
            surf = self.template.surfaces[i]
            phi = (surf.n_after - surf.n_before) * c_full[i]
            nu = nu - phi * y
            y = y + (surf.gap_to_next / surf.n_after) * nu
        return y, nu

    def trace_marginal(
        self, c_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trace the full marginal ray ``[y=1, nu=0]`` through all surfaces.

        No transfer is applied after the last surface, matching the unrolled
        convention where ``BFL = -y / nu`` is computed at the exit of the last
        refractive surface.
        """
        dtype = c_full.dtype
        device = c_full.device
        y = torch.ones(1, dtype=dtype, device=device)
        nu = torch.zeros(1, dtype=dtype, device=device)
        N = self._num_surfaces
        for i in range(N):
            surf = self.template.surfaces[i]
            phi = (surf.n_after - surf.n_before) * c_full[i]
            nu = nu - phi * y
            if i < N - 1:
                y = y + (surf.gap_to_next / surf.n_after) * nu
        return y, nu

    def bfl(self, c_full: torch.Tensor) -> torch.Tensor:
        """Back focal length at the design wavelength: ``-y / nu`` after last refraction."""
        y, nu = self.trace_marginal(c_full)
        safe_nu = torch.where(
            torch.abs(nu) > 1e-6, nu, torch.full_like(nu, 1e-6)
        )
        return (-y / safe_nu).squeeze()

    def efl(self, c_full: torch.Tensor) -> torch.Tensor:
        """Effective focal length: ``-1 / nu`` after the final refraction (y_0=1)."""
        _, nu = self.trace_marginal(c_full)
        safe_nu = torch.where(
            torch.abs(nu) > 1e-6, nu, torch.full_like(nu, 1e-6)
        )
        return (-1.0 / safe_nu).squeeze()

    def paraxial_pupil_radius(
        self,
        c_full: torch.Tensor,
        aper_idx: int,
        r_stop_mm: float,
    ) -> torch.Tensor:
        """Differentiable paraxial entrance-pupil radius.

        Sends a unit-height parallel ray ``(y=1, nu=0)`` from object space
        through surfaces ``[0, aper_idx)`` and reads ``y_at_stop``. By the
        usual paraxial reciprocity the entrance-pupil radius is

            r_pupil = r_stop / |y_at_stop|

        because ``y_at_stop`` is the lateral magnification A11 of the front
        group from object plane to stop plane. Gradients flow through
        ``c_full``, so the F# soft penalty receives a meaningful descent
        direction in θ. Mirrors the paraxial mode of ``calc_entrance_pupil``
        but stays inside the autograd graph.
        """
        y, _ = self._walk(c_full, stop_exclusive=int(aper_idx))
        safe_y = torch.where(
            torch.abs(y) > 1e-6, y, torch.full_like(y, 1e-6)
        )
        return (float(r_stop_mm) / torch.abs(safe_y)).squeeze()

    def derive_curvature_for_efl(
        self,
        c_full: torch.Tensor,
        target_efl: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the derived surface's curvature so the system has EFL = target_efl.

        Traces surfaces ``[0, derived_index)`` with full refract + transfer, then
        enforces ``nu_after_derived = -1/target_efl`` at the derived surface:

            phi_d = (nu + P) / y        (with P = 1/target_efl)
            c_d   = phi_d / (n_after - n_before)
            bfl   = y * target_efl

        Returns ``(c_derived, d_sensor, bfl)``, all differentiable w.r.t. ``c_full``.
        Matches ``_paraxial_fixed_efl`` in the unrolled EF50 code.
        """
        if self.template.derived_index is None:
            raise ValueError("template has no derived_index")
        i_d = self.template.derived_index
        y, nu = self._walk(c_full, stop_exclusive=i_d)
        surf_d = self.template.surfaces[i_d]
        P = 1.0 / float(target_efl)
        safe_y = torch.where(
            torch.abs(y) > 1e-6, y, torch.full_like(y, 1e-6)
        )
        phi_d = (nu + P) / safe_y
        c_derived = phi_d / (surf_d.n_after - surf_d.n_before)
        bfl = y * float(target_efl)
        d_sensor = self.template.z_last_surface_mm + bfl.squeeze()
        return c_derived.squeeze(), d_sensor, bfl.squeeze()
