from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from deeplens.optics.light.ray import Ray

from optical_rl.qd.singlet_zernike_measures import build_zernike_pinv, NOLL_TERM_NAMES

from ..problem.lens_template import LensTemplate
from ..problem.tracing import disk_samples


class ZernikeDescriptor:
    """Wavefront-OPD → Noll-Zernike-fit behavioral descriptor.

    Faithfully ports the 6-step pipeline from
    ``ef50_zernike_parallel.py:_zernike_measures`` (L865-L929):

        1. Coherent ray trace from the first surface toward the object.
        2. Restore the incoming OPL segment (object → first surface).
        3. Trace a chief ray at the target field angle.
        4. Build a reference sphere centered on the chief ray's exit-pupil hit.
        5. OPD = (OPL_marginal - OPL_chief) + ref_sphere, zeroed where invalid.
        6. Fit Noll Zernike coefficients via a precomputed pseudo-inverse.

    The returned tensor is the selected subset of fitted coefficients --
    ``noll_indices`` is the length-K list of 1-based Noll indices the user
    wants as BC coordinates.  2D, 3D, 4D behavior spaces are all trivially
    supported by changing the length of that list.
    """

    def __init__(
        self,
        noll_indices: Sequence[int],
        num_rays: int = 512,
        fixed_bounds: Optional[np.ndarray] = None,
    ) -> None:
        self._noll_indices: Tuple[int, ...] = tuple(int(j) for j in noll_indices)
        assert len(self._noll_indices) >= 1
        assert all(j >= 1 for j in self._noll_indices), "Noll indices are 1-based"
        self._num_rays = int(num_rays)
        self._j_max = max(self._noll_indices)
        self._fixed_bounds = fixed_bounds
        self._aperture_xy: Optional[torch.Tensor] = None
        self._pinv_Z: Optional[torch.Tensor] = None
        self._prepared = False

    @property
    def dim(self) -> int:
        return len(self._noll_indices)

    @property
    def coord_names(self) -> Tuple[str, ...]:
        return tuple(
            NOLL_TERM_NAMES.get(j, f"z{j}") for j in self._noll_indices
        )

    @property
    def fixed_bounds(self) -> Optional[np.ndarray]:
        return self._fixed_bounds

    def prepare(self, template: LensTemplate, *, device: str) -> None:
        stop_r = float(template.surfaces[template.stop_index].semi_diameter)
        disk = disk_samples(self._num_rays)
        self._aperture_xy = torch.tensor(
            disk * stop_r, dtype=torch.float64, device=device
        )
        pinv_np = build_zernike_pinv(disk, j_max=self._j_max)
        self._pinv_Z = torch.tensor(pinv_np, dtype=torch.float64, device=device)
        self._prepared = True

    def compute(
        self,
        lens,
        *,
        field_y_deg: float,
        object_depth_mm: float,
        wavelength_um: float,
        device: str,
        **kwargs,
    ) -> torch.Tensor:
        if not self._prepared:
            raise RuntimeError("ZernikeDescriptor.prepare() was not called")
        aperture_xy = self._aperture_xy
        pinv_Z = self._pinv_Z
        assert aperture_xy is not None and pinv_Z is not None

        dtype = torch.float64
        N = aperture_xy.shape[0]

        z_obj_val = float(object_depth_mm)
        z_obj_abs = abs(z_obj_val)
        y_obj = z_obj_val * math.tan(math.radians(field_y_deg))

        # Step 1: coherent ray trace from the first surface toward the object.
        # Rays start at the stop-sized Fibonacci disk laid on the front surface
        # plane (z = surfaces[0].d ~ 0), pointed at the object point.
        s0_z = float(lens.surfaces[0].d.item())
        origins = torch.zeros((N, 3), dtype=dtype, device=device)
        origins[:, :2] = aperture_xy
        origins[:, 2] = s0_z
        obj = torch.tensor(
            [0.0, y_obj, z_obj_val], dtype=dtype, device=device
        )
        d_vec = torch.nn.functional.normalize(origins - obj.unsqueeze(0), dim=-1)
        ray = Ray(
            origins, d_vec, wvln=float(wavelength_um), coherent=True, device=device
        )
        ray = lens.trace2sensor(ray)
        pupil_z, _ = lens.get_exit_pupil()
        ray = ray.prop_to(pupil_z)
        opl = ray.opl.squeeze(-1)
        valid = ray.is_valid.float()
        x_ep = ray.o[:, 0].to(dtype)
        y_ep = ray.o[:, 1].to(dtype)

        # Step 2: restore incoming OPL (object -> first surface plane).
        xp = aperture_xy[:, 0]
        yp = aperture_xy[:, 1]
        opl_marginal_in = torch.sqrt(
            xp ** 2 + (yp - y_obj) ** 2 + z_obj_abs ** 2
        )
        opl_chief_in = math.sqrt(y_obj ** 2 + z_obj_abs ** 2)
        opl = opl + (opl_marginal_in - opl_chief_in)

        # Step 3: chief ray at the target field angle (meridional plane).
        if field_y_deg == 0.0:
            chief_rfov = torch.tensor([0.0], device=device)
        else:
            chief_rfov = torch.tensor([0.0, field_y_deg], device=device)
        chief_o, chief_d = lens.calc_chief_ray_infinite(
            chief_rfov, plane="meridional"
        )
        chief_ray = Ray(
            chief_o[-1:].to(dtype),
            chief_d[-1:].to(dtype),
            wvln=float(wavelength_um),
            coherent=True,
            device=device,
        )
        chief_ray = lens.trace2sensor(chief_ray)
        chief_ray = chief_ray.prop_to(pupil_z)
        opl_chief = chief_ray.opl.squeeze()
        x_chief_ep = chief_ray.o[0, 0].to(dtype)
        y_chief_ep = chief_ray.o[0, 1].to(dtype)

        # Step 4: reference sphere centered on chief-ray exit-pupil position.
        BFL = lens.d_sensor - pupil_z
        ref_sphere = torch.sqrt(
            (x_ep - x_chief_ep) ** 2 + (y_ep - y_chief_ep) ** 2 + BFL ** 2
        ) - BFL

        # Step 5: OPD (invalid rays zeroed).
        opd = ((opl - opl_chief) + ref_sphere) * valid

        # Step 6: Noll Zernike fit.
        coeffs = pinv_Z @ opd.to(pinv_Z.dtype)
        picks = torch.stack([coeffs[j - 1] for j in self._noll_indices])
        return picks
