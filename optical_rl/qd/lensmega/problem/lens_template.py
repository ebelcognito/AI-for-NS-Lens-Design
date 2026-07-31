from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from deeplens import GeoLens
from deeplens.optics.geometric_surface import Aperture


@dataclass(frozen=True)
class Surface:
    n_before: float
    n_after: float
    gap_to_next: float
    is_free_curvature: bool
    initial_c: float
    semi_diameter: float
    d_axial: float
    is_stop: bool


@dataclass(frozen=True)
class LensTemplate:
    name: str
    surfaces: Tuple[Surface, ...]
    stop_index: int
    sensor_size_mm: Tuple[float, float]
    z_last_surface_mm: float
    free_indices: Tuple[int, ...]
    derived_index: Optional[int]
    object_depth_mm: float
    wavelength_um: float
    json_path: str
    entrance_pupil_z: float
    entrance_pupil_r: float
    element_groups: Tuple[Tuple[int, ...], ...] = ()
    element_n_d: Tuple[float, ...] = ()
    element_V_d: Tuple[float, ...] = ()
    cemented_interfaces: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.surfaces)
        assert n >= 2, "lens must have at least two surfaces"
        assert 0 <= self.stop_index < n
        for i in self.free_indices:
            assert 0 <= i < n, f"free_indices entry {i} out of range"
            assert not self.surfaces[i].is_stop, (
                f"free_indices entry {i} is a stop surface; stops cannot be free"
            )
        if self.derived_index is not None:
            assert 0 <= self.derived_index < n
            assert self.derived_index not in self.free_indices, (
                "derived_index cannot also be a free curvature"
            )
            assert self.surfaces[self.derived_index].n_after == 1.0, (
                "derived curvature surface must exit to air (n_after == 1.0); "
                "the EFL closure formula only handles this case"
            )
            for j in range(self.derived_index + 1, n):
                s = self.surfaces[j]
                assert s.n_before == 1.0 and s.n_after == 1.0, (
                    f"surface {j} after derived_index must be in air on both sides; "
                    "no further refraction is supported after the derived surface"
                )


def load_lens_template(
    json_path: str,
    *,
    wavelength_um: float,
    stop_surface_index: int,
    free_curvature_indices: Tuple[int, ...],
    derived_curvature_index: Optional[int] = None,
    device: str = "cpu",
    name: Optional[str] = None,
    cemented_interfaces: Tuple[int, ...] = (),
) -> Tuple[LensTemplate, GeoLens]:
    lens = GeoLens(filename=json_path, device=device)

    num_surfaces = len(lens.surfaces)
    assert 0 <= stop_surface_index < num_surfaces, (
        f"stop_surface_index {stop_surface_index} out of range [0, {num_surfaces})"
    )
    assert isinstance(lens.surfaces[stop_surface_index], Aperture), (
        f"surface {stop_surface_index} is not an Aperture; "
        f"got {type(lens.surfaces[stop_surface_index]).__name__}"
    )

    d_axial = [float(lens.surfaces[i].d.item()) for i in range(num_surfaces)]
    gaps: list[float] = []
    for i in range(num_surfaces):
        if i < num_surfaces - 1:
            gaps.append(d_axial[i + 1] - d_axial[i])
        else:
            gaps.append(0.0)

    running_n = 1.0
    surfaces: list[Surface] = []
    for i in range(num_surfaces):
        surf = lens.surfaces[i]
        if isinstance(surf, Aperture):
            n_before = running_n
            n_after = running_n
            initial_c = 0.0
            is_stop = True
        else:
            n_before = running_n
            n_after = float(surf.mat2.refractive_index(wavelength_um))
            running_n = n_after
            initial_c = float(surf.c.item()) if hasattr(surf, "c") else 0.0
            is_stop = False

        semi_d = float(surf.r)
        surfaces.append(
            Surface(
                n_before=n_before,
                n_after=n_after,
                gap_to_next=gaps[i],
                is_free_curvature=(i in tuple(free_curvature_indices)),
                initial_c=initial_c,
                semi_diameter=semi_d,
                d_axial=d_axial[i],
                is_stop=is_stop,
            )
        )

    last_refractive = max(
        (i for i, s in enumerate(surfaces) if not s.is_stop),
        default=num_surfaces - 1,
    )
    z_last = d_axial[last_refractive]

    sensor_size = lens.sensor_size
    if hasattr(sensor_size, "__iter__"):
        sensor_size_mm = (float(sensor_size[0]), float(sensor_size[1]))
    else:
        sensor_size_mm = (float(sensor_size), float(sensor_size))

    with torch.no_grad():
        lens.calc_pupil()
    entr_z = float(lens.entr_pupilz)
    entr_r = float(lens.entr_pupilr)

    # GeoLens.__init__ resets the global default dtype to float32; the
    # DeepLens coherent OPL tracer raises if t > 100 mm under float32 default.
    # Override after loading so subsequent rays + chief-ray traces run in
    # float64.  Matches ef50_zernike_parallel.py line 417.
    torch.set_default_dtype(torch.float64)

    for i in free_curvature_indices:
        lens.surfaces[i].c = torch.tensor(0.0, dtype=torch.float64, device=device)
    if derived_curvature_index is not None:
        lens.surfaces[derived_curvature_index].c = torch.tensor(
            0.0, dtype=torch.float64, device=device
        )
    lens.d_sensor = torch.tensor(
        float(lens.d_sensor.item()), dtype=torch.float64, device=device
    )

    # Detect glass elements: group consecutive non-air, non-stop surfaces.
    # When a surface is listed in ``cemented_interfaces``, it acts as the BACK
    # of the current element AND the FRONT of the next element, so the search
    # gets independent (n, V) pairs on each side of the cement.  This is the
    # mechanism that turns a "single thick element with a ghost cement DOF"
    # into a real achromatic doublet whose interface curvature is optically
    # active.  Empty list (default) → original single-glass-per-element
    # behaviour, byte-identical to before this change.
    cement_set = set(int(i) for i in cemented_interfaces)
    element_groups: list[Tuple[int, ...]] = []
    element_n_d: list[float] = []
    element_V_d: list[float] = []
    i_elem = 0
    while i_elem < num_surfaces:
        s = surfaces[i_elem]
        if s.is_stop or abs(s.n_after - 1.0) < 0.01:
            i_elem += 1
            continue
        # Start of an element: surface entering glass (n_after != 1.0)
        group = [i_elem]
        glass_n = s.n_after
        glass_V = float(lens.surfaces[i_elem].mat2.V) if hasattr(lens.surfaces[i_elem].mat2, "V") else 50.0
        j = i_elem + 1
        split_at_cement = False
        while j < num_surfaces:
            sj = surfaces[j]
            if sj.is_stop:
                j += 1
                continue
            group.append(j)
            if abs(sj.n_after - 1.0) < 0.01:
                break  # exited to air
            if j in cement_set:
                # Cement boundary: end this sub-element here.  Surface j is
                # the BACK of the current group (its n_bef gets glass_curr in
                # _build_n_full) AND will become the FRONT of the next group
                # (its n_aft gets glass_next).  The next outer-loop iteration
                # must restart at j (not j+1) so it appears in both groups.
                split_at_cement = True
                break
            j += 1
        element_groups.append(tuple(group))
        element_n_d.append(glass_n)
        element_V_d.append(glass_V)
        i_elem = j if split_at_cement else j + 1

    template = LensTemplate(
        name=name or json_path,
        surfaces=tuple(surfaces),
        stop_index=stop_surface_index,
        sensor_size_mm=sensor_size_mm,
        z_last_surface_mm=z_last,
        free_indices=tuple(free_curvature_indices),
        derived_index=derived_curvature_index,
        object_depth_mm=float("nan"),
        wavelength_um=float(wavelength_um),
        json_path=json_path,
        entrance_pupil_z=entr_z,
        entrance_pupil_r=entr_r,
        element_groups=tuple(element_groups),
        element_n_d=tuple(element_n_d),
        element_V_d=tuple(element_V_d),
        cemented_interfaces=tuple(int(i) for i in cemented_interfaces),
    )
    return template, lens
