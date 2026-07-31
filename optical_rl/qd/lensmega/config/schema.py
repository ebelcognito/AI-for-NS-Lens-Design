from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProblemCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["lens"] = "lens"
    lens_path: str
    wavelength_um: float = 0.587
    fitness_wavelengths_um: List[float] = Field(
        default_factory=lambda: [0.656, 0.587, 0.486]
    )
    object_depth_mm: float = -20000.0
    stop_surface_index: int
    free_curvature_indices: List[int]
    derived_curvature_index: Optional[int] = None
    focal_length_target_mm: Optional[float] = None
    theta_bounds: List[Tuple[float, float]]
    stage: Literal["stage1", "stage2"] = "stage1"
    num_rays: int = 512
    invalid_tol: float = 0.15
    invalid_penalty_weight: float = 5.0
    intersect_penalty_weight: float = 1.0
    air_min_mm: float = 0.1
    thick_min_mm: float = 1.0
    strict_entrance_pupil: bool = True
    descriptor_field_index: int = 0
    parameterization: Literal["curvature", "power_budget", "full_dof"] = "curvature"
    efl_penalty_weight: float = 5.0
    free_gap_indices: List[int] = Field(default_factory=list)
    variable_flange: bool = False
    flange_bounds: Optional[Tuple[float, float]] = None
    variable_materials: bool = False
    free_material_indices: List[int] = Field(default_factory=list)
    # Surface indices that are glass-glass cement boundaries.  Splits the
    # auto-detected element group at each listed surface so each side gets
    # its own (n, V) pair in the search.  See lens_template.py for details.
    cemented_interfaces: List[int] = Field(default_factory=list)
    field_weights: Optional[List[float]] = None

    # --- F-number soft-constraint settings -----------------------------------
    # When ``enforce_fnum`` is False (default) the F-number penalty term is
    # never computed and the rest of the loss is bit-identical to pre-change
    # behaviour, so existing configs need no edits.
    enforce_fnum: bool = False
    f_number_target: float = 4.0
    fnum_penalty_weight: float = 1.0

    @model_validator(mode="after")
    def _check_bounds_match_free(self):
        expected = len(self.free_curvature_indices) + len(self.free_gap_indices)
        if self.variable_flange:
            expected += 1
        if self.variable_materials:
            expected += 2 * len(self.free_material_indices)
        if len(self.theta_bounds) != expected:
            raise ValueError(
                f"theta_bounds length ({len(self.theta_bounds)}) must equal "
                f"free_curvature_indices ({len(self.free_curvature_indices)}) + "
                f"free_gap_indices ({len(self.free_gap_indices)}) + "
                f"variable_flange ({1 if self.variable_flange else 0}) + "
                f"variable_materials (2*{len(self.free_material_indices)}) = {expected}"
            )
        if self.variable_materials and self.parameterization != "full_dof":
            raise ValueError(
                "variable_materials requires parameterization='full_dof'"
            )
        return self


class DescriptorCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["zernike", "seidel", "random"]
    params: Dict[str, Any]


class AlgorithmCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    population_size: int = 16
    initial_sigma: float = 0.75
    cma_eta: float = 0.05
    cma_seed: int = 0
    seed_count: int = 128
    calibration_count: int = 512
    iterations: int = 1000
    new_cell_bonus: float = 1.0e6
    # Optional fitness-quality scaling for the new-cell bonus. When enabled,
    # the bonus paid for INSERT_NEW is multiplied by
    #   sigmoid((fitness - bonus_fitness_threshold) / bonus_fitness_scale)
    # so a new cell at high fitness gets the full bonus while a new cell at
    # garbage fitness gets near-zero bonus. Stops CMA from chasing novelty
    # into pathological corners of BC space. Default: disabled (preserves
    # the original constant-bonus behavior).
    scale_bonus_by_fitness: bool = False
    bonus_fitness_threshold: float = 0.0
    bonus_fitness_scale: float = 1.0
    log_every: int = 50
    restart_patience: int = 1
    # Restart when best_fitness has not improved by `fitness_improve_eps` for
    # this many consecutive generations. 0 disables (preserves original
    # behavior, where only insertion-based no_improve and Hansen terms can
    # trigger a restart). Use this to break sigma-runaway equilibria where
    # corner-saturated offspring keep filling new BC cells but the fitness
    # frontier is frozen.
    fitness_patience: int = 0
    fitness_improve_eps: float = 1.0e-6
    # Ablation: if True, every CMA-MEGA offspring uses a purely random
    # direction in theta-space instead of the descriptor-gradient-weighted
    # one. Isolates the contribution of Seidel gradients — with this on
    # the method degenerates to MAP-Elites with Gaussian mutation.
    ablate_gradients: bool = False


class ArchiveCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bins: List[int]
    bounds: Optional[List[Tuple[float, float]]] = None
    bound_calibration_pad: float = 0.05
    bound_calibration_pct: Tuple[float, float] = (2.0, 98.0)


class IOCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_dir: str = "results/lensmega"
    run_name: str
    checkpoint_every: int = 50
    parallel_workers: Optional[int] = None
    device: str = "auto"
    log_seed_accepts: bool = True
    elite_fitness_threshold: Optional[float] = None
    # L-BFGS refiner settings (used when elite_fitness_threshold is set)
    refine_efl_penalty_weight: Optional[float] = None
    refine_invalid_penalty_weight: Optional[float] = None
    refine_intersect_penalty_weight: Optional[float] = None
    # Heavier F-number penalty for the L-BFGS polishing stage, mirroring
    # how ``refine_efl_penalty_weight`` ramps up. Ignored when
    # ``problem.enforce_fnum`` is False or when this field is None.
    refine_fnum_penalty_weight: Optional[float] = 5.0
    refine_lbfgs_max_steps: int = 100
    refine_lbfgs_lr: float = 0.5
    refine_field_weights: Optional[List[float]] = None
    refine_converge_tol: float = 1e-4
    # Basin stability analysis
    refine_stability_samples: int = 20
    refine_stability_perturbation: float = 0.01
    # When True, write raw parent gradients (fitness + per-descriptor) to
    # ``run_dir/gradient_log.npz`` once per CMA-MEGA generation. Used by the
    # gradient-alignment experiment.
    log_gradients: bool = False


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: ProblemCfg
    descriptor: DescriptorCfg
    algorithm: AlgorithmCfg
    archive: ArchiveCfg
    io: IOCfg

    @model_validator(mode="after")
    def _crosscheck(self):
        if self.descriptor.kind == "zernike":
            indices = self.descriptor.params.get("noll_indices")
            if indices is None:
                raise ValueError(
                    "descriptor.params.noll_indices is required for zernike"
                )
            k = len(indices)
        elif self.descriptor.kind == "seidel":
            terms = self.descriptor.params.get("terms")
            if terms is None:
                raise ValueError("descriptor.params.terms is required for seidel")
            k = len(terms)
        elif self.descriptor.kind == "random":
            k = int(self.descriptor.params.get("dim", 2))
        else:
            raise ValueError(f"unknown descriptor kind {self.descriptor.kind!r}")

        if len(self.archive.bins) != k:
            raise ValueError(
                f"archive.bins has length {len(self.archive.bins)} but descriptor "
                f"has dim {k}"
            )
        if self.archive.bounds is not None and len(self.archive.bounds) != k:
            raise ValueError(
                f"archive.bounds has length {len(self.archive.bounds)} but descriptor "
                f"has dim {k}"
            )
        return self
