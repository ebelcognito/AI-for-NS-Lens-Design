from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch

from ..descriptors.base import Descriptor
from .base import EvalResult
from .lens_template import LensTemplate, load_lens_template
from .paraxial import GeneralizedParaxialSolver
from .penalties import intersection_penalty, invalid_fraction_penalty
from .tracing import (
    disk_samples,
    latin_hypercube,
    project_box,
    rms_spot_batch,
    trace_fields_batch,
)


@dataclass(frozen=True)
class LensProblemConfig:
    lens_path: str
    wavelength_um: float
    fitness_wavelengths_um: Tuple[float, ...]
    object_depth_mm: float
    stop_surface_index: int
    free_curvature_indices: Tuple[int, ...]
    derived_curvature_index: Optional[int]
    focal_length_target_mm: Optional[float]
    theta_bounds: Tuple[Tuple[float, float], ...]
    stage: str = "stage1"
    num_rays: int = 512
    invalid_tol: float = 0.15
    invalid_penalty_weight: float = 5.0
    intersect_penalty_weight: float = 1.0
    air_min_mm: float = 0.1
    thick_min_mm: float = 1.0
    strict_entrance_pupil: bool = False
    descriptor_field_index: int = 0
    parameterization: str = "curvature"  # "curvature", "power_budget", or "full_dof"
    efl_penalty_weight: float = 5.0
    free_gap_indices: Tuple[int, ...] = ()
    variable_flange: bool = False
    flange_bounds: Optional[Tuple[float, float]] = None
    variable_materials: bool = False
    free_material_indices: Tuple[int, ...] = ()
    cemented_interfaces: Tuple[int, ...] = ()
    field_weights: Optional[Tuple[float, ...]] = None
    device: str = "cpu"
    # F-number soft constraint. When ``enforce_fnum`` is False the penalty is
    # never computed and the loss is unchanged from pre-change behaviour.
    enforce_fnum: bool = False
    f_number_target: float = 4.0
    fnum_penalty_weight: float = 1.0


class LensProblem:
    """Generic ray-traced RMS-spot problem with a pluggable Descriptor.

    This is the ONLY concrete Problem in the lensmega package.  The lens
    template, paraxial closure mode, penalties, tracing, and descriptor are
    all swappable via config + composition, so adding a new lens means
    writing a new YAML, not a new class.
    """

    def __init__(self, cfg: LensProblemConfig, descriptor: Descriptor) -> None:
        self.cfg = cfg
        self.descriptor = descriptor
        self._backend_loaded = False
        self.template: Optional[LensTemplate] = None
        self.solver: Optional[GeneralizedParaxialSolver] = None
        self._reusable_lens = None
        self._fitness_unit_disk: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Backend bring-up
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> None:
        if self._backend_loaded:
            return
        template, lens = load_lens_template(
            self.cfg.lens_path,
            wavelength_um=self.cfg.wavelength_um,
            stop_surface_index=self.cfg.stop_surface_index,
            free_curvature_indices=self.cfg.free_curvature_indices,
            derived_curvature_index=self.cfg.derived_curvature_index,
            device=self.cfg.device,
            cemented_interfaces=self.cfg.cemented_interfaces,
        )
        self.template = template
        self.solver = GeneralizedParaxialSolver(template)
        self._reusable_lens = copy.deepcopy(lens)
        self._fitness_unit_disk = torch.tensor(
            disk_samples(int(self.cfg.num_rays)),
            dtype=torch.float32,
            device=self.cfg.device,
        )
        self.descriptor.prepare(template, device=self.cfg.device)
        self._backend_loaded = True

    # ------------------------------------------------------------------
    # Protocol plumbing
    # ------------------------------------------------------------------

    @property
    def theta_dim(self) -> int:
        return len(self.cfg.theta_bounds)

    @property
    def theta_bounds(self) -> np.ndarray:
        return np.asarray(self.cfg.theta_bounds, dtype=np.float64)

    @property
    def descriptor_dim(self) -> int:
        return self.descriptor.dim

    def initial_theta(self) -> np.ndarray:
        return np.zeros(self.theta_dim, dtype=np.float64)

    def sample_theta(self, n: int, seed: int) -> np.ndarray:
        unit = latin_hypercube(int(n), self.theta_dim, int(seed))
        lo = np.array([b[0] for b in self.cfg.theta_bounds], dtype=np.float64)
        hi = np.array([b[1] for b in self.cfg.theta_bounds], dtype=np.float64)
        return lo + unit * (hi - lo)

    def project_theta(self, theta: np.ndarray) -> np.ndarray:
        return project_box(theta, self.cfg.theta_bounds)

    # ------------------------------------------------------------------
    # Paraxial closure + lens build
    # ------------------------------------------------------------------

    def _build_lens(self, theta_t: torch.Tensor):
        """Apply theta_t to the reusable lens in-place; returns bfl_val.

        Handles both fixed-EFL and free modes; sets every free-curvature
        surface from theta, the derived surface from the closure formula,
        and ``lens.d_sensor`` from the computed BFL.
        """
        assert self.solver is not None and self.template is not None
        c_full = self.solver.expand_theta(theta_t)
        if self.cfg.focal_length_target_mm is not None:
            c_derived, d_sensor, bfl_t = self.solver.derive_curvature_for_efl(
                c_full, self.cfg.focal_length_target_mm
            )
            d_idx = self.template.derived_index
            assert d_idx is not None
            c_full = c_full.clone()
            c_full[d_idx] = c_derived
        else:
            bfl_t = self.solver.bfl(c_full)
            d_sensor = self.template.z_last_surface_mm + bfl_t

        lens = self._reusable_lens
        dtype = lens.surfaces[self.cfg.free_curvature_indices[0]].c.dtype
        for i in range(len(self.template.surfaces)):
            if i in self.cfg.free_curvature_indices or i == self.template.derived_index:
                lens.surfaces[i].c = c_full[i].to(dtype=dtype)
        lens.d_sensor = d_sensor.to(dtype=dtype)
        return lens, bfl_t

    def _calc_hfov_deg(self, theta_t: torch.Tensor) -> float:
        assert self.solver is not None and self.template is not None
        if self.cfg.focal_length_target_mm is not None:
            efl = self.cfg.focal_length_target_mm
        else:
            c_full = self.solver.expand_theta(theta_t)
            efl = float(self.solver.efl(c_full).detach())
        half_sensor = self.template.sensor_size_mm[0] / 2.0
        return math.degrees(math.atan(half_sensor / abs(efl)))

    def _field_points(self, hfov_deg: float):
        if str(self.cfg.stage).lower().strip() == "stage2":
            return ((0.0, 0.0), (0.0, 0.5 * hfov_deg), (0.0, hfov_deg))
        return ((0.0, 0.0),)

    # ------------------------------------------------------------------
    # Core evaluate
    # ------------------------------------------------------------------

    def _fitness_and_metrics(self, theta_t: torch.Tensor):
        assert self.solver is not None and self.template is not None
        K = self.descriptor.dim
        device = self.cfg.device

        # Reject diverging lenses.
        c_full_check = self.solver.expand_theta(theta_t)
        if self.cfg.focal_length_target_mm is not None:
            _, _, bfl_t = self.solver.derive_curvature_for_efl(
                c_full_check, self.cfg.focal_length_target_mm
            )
            bfl_val = float(bfl_t.detach().item())
        else:
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
                metrics = {
                    "invalid_fraction": 1.0,
                    "rejected": 1.0,
                    "rejected_pupil": 1.0,
                }
                return bad, zero_measures, metrics

        fitness_aperture_xy = self._fitness_unit_disk * float(pupilr)
        pupil_z_f = float(pupilz)

        # Spot-size fitness: batch all field points per wavelength.
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

        intersect_loss = intersection_penalty(
            lens,
            air_min_mm=self.cfg.air_min_mm,
            thick_min_mm=self.cfg.thick_min_mm,
            device=device,
        )
        if float(intersect_loss.detach()) > 0:
            fitness = fitness - self.cfg.intersect_penalty_weight * intersect_loss

        if float(invalid_fraction_mean.detach().item()) > self.cfg.invalid_tol:
            excess = invalid_fraction_mean - self.cfg.invalid_tol
            fitness = fitness - self.cfg.invalid_penalty_weight * excess

        # --- F-number soft penalty (gated by enforce_fnum) -------------------
        # Reaches here only when the strict_entrance_pupil rejection above
        # did NOT short-circuit, so the pupil computation is meaningful.
        fnum_metrics = None
        if self.cfg.enforce_fnum:
            # Recompute c_full inline (post-closure curvatures) so gradients
            # flow correctly when derived_curvature_index lies in the front
            # group. Cheap: one paraxial walk.
            c_full_fnum = self.solver.expand_theta(theta_t)
            if (self.cfg.focal_length_target_mm is not None
                    and self.template.derived_index is not None):
                c_derived, _, _ = self.solver.derive_curvature_for_efl(
                    c_full_fnum, self.cfg.focal_length_target_mm
                )
                c_full_fnum = c_full_fnum.clone()
                c_full_fnum[self.template.derived_index] = c_derived
            aper_idx = self.template.stop_index
            r_stop_mm = float(lens.surfaces[aper_idx].r)
            pupilr_diff = self.solver.paraxial_pupil_radius(
                c_full_fnum, aper_idx, r_stop_mm
            )
            if self.cfg.focal_length_target_mm is not None:
                # Closure makes EFL exact; treat as a constant so the F# gradient
                # flows purely through the pupil, not through a fictitious EFL term.
                efl_for_fnum = float(self.cfg.focal_length_target_mm)
            else:
                efl_for_fnum = self.solver.efl(c_full_fnum)
            fnum_actual = efl_for_fnum / (2.0 * pupilr_diff)
            # Relative-error formulation, mirrors how the EFL penalty is built
            # (`(efl_actual - efl_target) / efl_target`). Keeps the penalty
            # dimensionless and on the same scale as EFL so a single weight
            # works across F# specs from F/2 to F/16.
            fnum_target = float(self.cfg.f_number_target)
            fnum_dev = (fnum_actual - fnum_target) / fnum_target
            fnum_penalty = self.cfg.fnum_penalty_weight * fnum_dev ** 2
            fitness = fitness - fnum_penalty
            fnum_metrics = {
                "fnum_actual": float(
                    fnum_actual.detach().cpu().item()
                ) if torch.is_tensor(fnum_actual) else float(fnum_actual),
                "fnum_deviation": float(
                    fnum_dev.detach().cpu().item()
                ) if torch.is_tensor(fnum_dev) else float(fnum_dev),
                "fnum_penalty": float(fnum_penalty.detach().cpu().item()),
            }

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

        # Always report fnum_actual in metrics so logging/PNG titles can show
        # F# regardless of whether the soft penalty is enabled. EFL is exact
        # via the closure formula when focal_length_target_mm is set,
        # otherwise read from the differentiable solver.
        if self.cfg.focal_length_target_mm is not None:
            efl_for_metric = float(self.cfg.focal_length_target_mm)
        else:
            efl_for_metric = float(self.solver.efl(c_full_check).detach())
        pupilr_metric = float(pupilr)
        fnum_actual_metric = (efl_for_metric / (2.0 * pupilr_metric)
                              if pupilr_metric > 1e-6 else float("nan"))

        metrics = {
            "rms_spot": float(rms_mean.detach().cpu().item()),
            "invalid_fraction": float(
                invalid_fraction_mean.detach().cpu().item()
            ),
            "fnum_actual": fnum_actual_metric,
        }
        if fnum_metrics is not None:
            # When the soft penalty is on, also report deviation + penalty
            # value (overwrites fnum_actual with the same number).
            metrics.update(fnum_metrics)
        for k, name in enumerate(self.descriptor.coord_names):
            metrics[f"m{k}_{name}"] = float(measures_list[k].detach().cpu().item())
        # Full 5-Seidel snapshot for lineage logging (populated by
        # SeidelDescriptor on its last compute call).
        last_all = getattr(self.descriptor, "_last_all_seidel", None)
        if last_all is not None:
            for j, label in enumerate(("S1", "S2", "S3", "S4", "S5")):
                metrics[f"seidel_{label}"] = float(last_all[j])
        return fitness, measures_list, metrics

    def evaluate(self, theta: np.ndarray, with_grads: bool = True) -> EvalResult:
        self._ensure_backend()
        theta_proj = self.project_theta(theta)
        theta_t = torch.tensor(
            theta_proj,
            dtype=torch.float32,
            device=self.cfg.device,
            requires_grad=with_grads,
        )
        fitness, measures_list, metrics = self._fitness_and_metrics(theta_t)
        K = self.descriptor.dim

        grads = None
        if with_grads:
            grads = {}
            grads["fitness"] = (
                torch.autograd.grad(
                    fitness, theta_t, retain_graph=True, allow_unused=False
                )[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            for k in range(K):
                retain = k < K - 1
                grads[f"m{k}"] = (
                    torch.autograd.grad(
                        measures_list[k],
                        theta_t,
                        retain_graph=retain,
                        allow_unused=False,
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )

        measures_np = np.array(
            [float(m.detach().cpu().item()) for m in measures_list],
            dtype=np.float64,
        )
        return EvalResult(
            theta=theta_proj.astype(np.float64, copy=False),
            fitness=float(fitness.detach().cpu().item()),
            measures=measures_np,
            metrics=metrics,
            grads=grads,
        )
