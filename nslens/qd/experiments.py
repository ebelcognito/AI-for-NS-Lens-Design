"""The experiment registry -- one entry per row of EXPERIMENTS.md.

Each entry below replaces a ~170-line runner script whose only distinguishing content
was the fitness, the descriptor pair, and the source/detector placement.

    python -m nslens.qd fold_qd --evals 14000 --seed 0 --out runs/fold_qd
    python -m nslens.qd --list
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from .driver import ExperimentConfig
from .arvr import SIGMA0 as ARVR_SIGMA0
from .arvr import ArvrProblem
from .fixed_environments import problem_from_environment_spec
from .problems import FoldProblem, PrismFoldProblem


DEFAULT_FIXED_ENVIRONMENT_SPECS = (
    Path(__file__).resolve().parents[2]
    / "runs" / "environment_pilot_20260903_v2" / "environment_specs.json")
FIXED_FIELDS = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)


def _experiments() -> Dict[str, ExperimentConfig]:
    return {
        # -- #5, the headline result -------------------------------------------
        "fold_qd": ExperimentConfig(
            name="fold_qd",
            label="fold_qd_tilt_uncoated",
            problem=FoldProblem(),
            fitness="focused_throughput",
            descriptors=("tir_bounces", "thinness"),
        ),
        # -- #7, the tilt ablation: same as #5 with face tilts locked to 0 ------
        "fold_qd_notilt": ExperimentConfig(
            name="fold_qd_notilt",
            label="fold_qd_tilt_uncoated",
            problem=FoldProblem(),
            fitness="focused_throughput",
            descriptors=("tir_bounces", "thinness"),
            lock_tilt=True,
        ),
        # -- #6, coherence rather than raw throughput ---------------------------
        "orderly_qd": ExperimentConfig(
            name="orderly_qd",
            label="orderly_fold_qd",
            problem=FoldProblem(),
            fitness="coherent_focused_throughput",
            descriptors=("tir_bounces", "thinness"),
        ),
        # -- #8, robustness across +-3 deg of field -----------------------------
        # NOTE n_rays=17, not the 21 used everywhere else: three fields per
        # evaluation makes this 3x more expensive, and the legacy runner traded
        # bundle resolution for evaluations. Changing it moves the archive.
        "multifield_qd": ExperimentConfig(
            name="multifield_qd",
            label="multifield_fold_qd",
            problem=FoldProblem(n_rays=17),
            fitness="focused_throughput",
            descriptors=("tir_bounces", "thinness"),
            fields=(-3.0, 0.0, 3.0),
            evals=12000,
        ),
        # -- the AR/VR 3-field relay: its own measurement model -----------------
        # descriptors are named only for the meta labels; ArvrProblem.evaluate
        # computes the behaviour coordinates itself (FOV, prism depth).
        "arvr_qd": ExperimentConfig(
            name="arvr_qd",
            label="arvr_prism_qd_tir",
            problem=ArvrProblem(),
            fitness="arvr_relay_3field",
            descriptors=("fov", "prism_depth"),
            bc_bins=(20, 28),                      # coverage is not the goal here;
            bc_bounds=((0.0, 40.0), (4.0, 88.0)),  # fine FOV/form-factor resolution is
            cell_bc_keys=("fov", "depth"),
            meta_extra=dict(sigma0=ARVR_SIGMA0),
            # NOTE 8 emitters and 40k evals, not the 5/14k used by the fold envs.
            # The emitter count changes the search trajectory, so getting this wrong
            # shifts the archive even though every evaluation is identical.
            emitters=8,
            evals=40000,
        ),
        # -- AR/VR relay forced to fold at least twice --------------------------
        # arvr_qd's archive is almost entirely single-bounce: the eye is to the right of
        # the display, so one TIR already gets light there and nothing rewards a second.
        # min_tir=2 refuses to archive anything shallower, which makes the 2-4 fold
        # relays compete with each other instead of with the easy single-fold solution.
        "arvr_qd_multifold": ExperimentConfig(
            name="arvr_qd_multifold",
            label="arvr_prism_qd_tir_multifold",
            problem=ArvrProblem(min_tir=2),
            fitness="arvr_relay_3field",
            descriptors=("fov", "prism_depth"),
            bc_bins=(20, 28),
            bc_bounds=((0.0, 40.0), (4.0, 88.0)),
            cell_bc_keys=("fov", "depth"),
            meta_extra=dict(sigma0=ARVR_SIGMA0, min_tir=2),
            param_scaling="normalized",
            emitters=8,
            evals=40000,
        ),
        # -- 2-4 folds with WIDE FIELD ACCEPTANCE -------------------------------
        # The target design class, not a diversity study. Three deliberate choices:
        #   fitness    worst field, not mean -- a design that is brilliant on axis and
        #              blind at +-8 deg scores its blindness, so it cannot win
        #   BC1        fold count restricted to 2..5 with reject_out_of_bounds, so an
        #              unfolded design cannot clamp into the "2 folds" niche and evict
        #              a real fold from it
        #   BC2        field acceptance rather than thinness -- the archive's diversity
        #              axis IS the property we want more of
        # 9 fields at 2 deg spacing means ~9x the engine calls per evaluation, hence the
        # smaller eval budget; normalized scaling because the isotropic step leaves the
        # refractive index bang-bang clipped (see runs/prism_sigma_sweep).
        "fold24_wide": ExperimentConfig(
            name="fold24_wide",
            label="fold24_wide_field",
            problem=FoldProblem(n_rays=17),
            fitness="focused_throughput",
            field_aggregate="worst",
            fields=(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0),
            descriptors=("tir_bounces", "field_accept"),
            bc_bins=(3, 5),
            bc_bounds=((2.0, 5.0), (0.0, 10.0)),
            reject_out_of_bounds=True,
            param_scaling="normalized",
            cell_bc_keys=("tir_count", "accept_deg"),
            info_descriptors=("thinness", "crit_margin", "ns_frac"),
            evals=6000,
        ),
        # -- packing folds: 2-4 folds, wide field, and the beam must CONVERGE ----
        # fold24_wide's archive contains designs whose bundle expands through the
        # element (worst ratio 15x) and designs that collapse to a caustic inside the
        # glass. Neither is a fold that pays for itself. Three changes on top of it:
        #   relative_sag     the sag genes are fractions of their own face's half-chord,
        #                    so +-0.5 means the same curvature at every thickness and a
        #                    bulb is unreachable BY CONSTRUCTION rather than by rejection
        #   max_sag_ratio    belt and braces on the sheared chord, which relative_sag
        #                    only approximates
        #   monotone         the dominant bundle must narrow from entry to exit, which
        #                    also rules out an internal waist that re-expands
        "fold24_pack": ExperimentConfig(
            name="fold24_pack",
            label="fold24_wide_packing",
            problem=FoldProblem(
                n_rays=17, relative_sag=True, max_sag_ratio=0.55,
                # fsag/bsag now in units of half-chord, not mm
                lo=np.array([-20.0, -55.0, 35.0, 6.0, -0.5, -0.5, -25.0, -25.0, 1.50]),
                hi=np.array([20.0, 55.0, 85.0, 45.0, 0.5, 0.5, 25.0, 25.0, 2.00])),
            fitness="focused_throughput",
            field_aggregate="worst",
            fields=(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0),
            descriptors=("tir_bounces", "field_accept"),
            bc_bins=(3, 5),
            bc_bounds=((2.0, 5.0), (0.0, 10.0)),
            reject_out_of_bounds=True,
            require_monotone_convergence=True,
            param_scaling="normalized",
            cell_bc_keys=("tir_count", "accept_deg"),
            info_descriptors=("thinness", "crit_margin", "beam_convergence"),
            evals=6000,
        ),
        # -- fold24_pack, but the beam may not cross an internal focus -----------
        # fold24_pack's width-only gate does not do what its comment claims: 26 of its
        # own 42 elites (62%) put a line focus inside the glass, because a caustic is
        # narrow at the waist and `regrow` only sees it if it re-expands before the exit.
        # convergence_gate="clean_pack" adds the ray-order-inversion test, which detects
        # a focus with no length scale to calibrate. Re-scored offline on fold24_pack's
        # archive it keeps 16 of 42 and the best design (0.790) survives, so this is a
        # real tightening rather than a rename.
        "fold24_clean": ExperimentConfig(
            name="fold24_clean",
            label="fold24_wide_packing_nocaustic",
            problem=FoldProblem(
                n_rays=17, relative_sag=True, max_sag_ratio=0.55,
                lo=np.array([-20.0, -55.0, 35.0, 6.0, -0.5, -0.5, -25.0, -25.0, 1.50]),
                hi=np.array([20.0, 55.0, 85.0, 45.0, 0.5, 0.5, 25.0, 25.0, 2.00])),
            fitness="focused_throughput",
            field_aggregate="worst",
            fields=(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0),
            # field_accept was the second axis and it bought no diversity: it is a
            # threshold walk over a 2 deg field list, so it can only ever return
            # 0/2/4/6/8 -- five attainable values into five bins, and the same design
            # family landed in three adjacent cells depending on where the threshold
            # happened to fall. thinness is continuous, it is the reason a fold exists
            # (pack light into a thin volume), and it is not a restatement of fitness
            # the way acceptance is. Bounds keep the registered low end even though the
            # 40 previous elites only reached T=70.2: reject_out_of_bounds DISCARDS
            # rather than clamps, so a floor fitted to the old elites would throw away
            # every design thinner than the old gate happened to find -- the one
            # direction this axis exists to reward. Top end is x_sensor (120) + margin.
            descriptors=("tir_bounces", "thinness"),
            bc_bins=(3, 16),
            bc_bounds=((2.0, 5.0), (10.0, 122.0)),
            reject_out_of_bounds=True,
            require_monotone_convergence=True,
            convergence_gate="clean_pack",
            gate_all_fields=True,
            # cold start cannot survive a 14.7% gate; seed from the previous run's
            # elites, pre-filtered to the ones that pass it (see runs/fold24_clean_seed)
            resume_from=str(Path(__file__).resolve().parents[2]
                            / "runs" / "fold24_clean_seed"),
            param_scaling="normalized",
            cell_bc_keys=("tir_count", "thinness_T"),
            info_descriptors=("field_accept", "crit_margin", "beam_convergence"),
            evals=6000,
        ),
        # -- the topology axis (doc 19) -----------------------------------------
        # fold24_clean, with ONE change: the first behaviour axis is the multiset of TIR
        # faces instead of the count of TIR bounces. Everything else -- genome box,
        # fitness, field set, every gate, param_scaling, eval budget -- is copied
        # verbatim so the pair is a controlled comparison of the AXIS.
        #
        # Why: fold24_clean's first axis has bc_bins=(3, ...) -- three attainable fold
        # counts. Measured over all 58,980 scored trace rows, 41.7 distinct interaction
        # signatures land in a single (tir_bounces, thinness) cell, 94% of occupied
        # cells hold more than one topology, and 133 of the 171 catalogued topologies
        # never win a cell anywhere. The archive is deleting the diversity it exists to
        # keep. `tir_faces` resolves 79 of those classes instead of 3.
        #
        # thinness drops to 4 bins because the grid is a product: 126 x 16 = 2016 cells
        # against ~900 designs that survive the gate stack in 6k evals would be a lookup
        # table with no selection pressure. 126 x 4 = 504 keeps competition inside a cell
        # while still letting the topology axis do the separating.
        "fold24_topo": ExperimentConfig(
            name="fold24_topo",
            label="fold24_topology_axis",
            problem=FoldProblem(
                n_rays=17, relative_sag=True, max_sag_ratio=0.55,
                lo=np.array([-20.0, -55.0, 35.0, 6.0, -0.5, -0.5, -25.0, -25.0, 1.50]),
                hi=np.array([20.0, 55.0, 85.0, 45.0, 0.5, 0.5, 25.0, 25.0, 2.00])),
            fitness="focused_throughput",
            field_aggregate="worst",
            fields=(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0),
            descriptors=("tir_faces", "thinness"),
            # MUST match the baseline's fold range. fold24_clean enforces 2-4 folds via
            # bc_bounds=(2.0, 5.0) on tir_count plus reject_out_of_bounds; the topology
            # axis has to enforce the same range or the comparison is rigged. The first
            # five vocabulary entries are the k=0 and k=1 multisets (unfolded, and one
            # bounce on each of the four faces), so the low bound is 5.0.
            #
            # Measured cost of getting this wrong: with the low bound at 0.0 the arm
            # filled its archive with `T@A -> T@B` (no fold, 0.922) and `TIR@T -> T@B`
            # (one bounce, 0.960) and "beat" the baseline 0.960 to 0.843 -- entirely
            # because straight transmission pays no fold losses. The baseline was
            # forbidden from keeping exactly those designs.
            #
            # 121 bins over [5, 126] puts exactly one vocabulary entry in each bin.
            bc_bins=(121, 4),
            bc_bounds=((5.0, 126.0), (10.0, 122.0)),
            reject_out_of_bounds=True,
            require_monotone_convergence=True,
            convergence_gate="clean_pack",
            gate_all_fields=True,
            resume_from=str(Path(__file__).resolve().parents[2]
                            / "runs" / "fold24_topo_seed"),
            param_scaling="normalized",
            cell_bc_keys=("tir_faces", "thinness_T"),
            info_descriptors=("field_accept", "crit_margin", "beam_convergence"),
            evals=6000,
        ),
        # -- gated re-run: both axes, exit convergence ON, detector clearance ON --
        # doc 19's comparison was measured on archives that turned out to be ~45%
        # non-imaging designs: 39 of the 87 ab40k elites that pass the fully hardened
        # config fail the exit test (see require_exit_convergence). So the retention
        # result -- 17 vs 7 distinct ray paths -- was counted over ray paths, some of
        # which never put an image on the detector. These two configs re-run the pair
        # with the leak closed, and they are the archives to quote.
        #
        # det_clearance=12 goes on at the same time. It is not part of the exit gate and
        # it is not cosmetic: FoldProblem has NEVER enforced it (Problem.det_clearance
        # defaults to None), so a fold env could hand back a design whose glass passes
        # through the sensor, and 1-6 elites per fold24 run do exactly that. Their
        # thinness_T is meaningless because T is measured against a plane the body
        # intersects, which makes the SECOND behaviour axis wrong for those cells.
        # Measured on the seed it costs nothing: 5 of 7 cells survive with or without it.
        "fold24_clean_gated": ExperimentConfig(
            name="fold24_clean_gated",
            label="fold24_fold_count_axis_gated",
            problem=FoldProblem(
                n_rays=17, relative_sag=True, max_sag_ratio=0.55, det_clearance=12.0,
                lo=np.array([-20.0, -55.0, 35.0, 6.0, -0.5, -0.5, -25.0, -25.0, 1.50]),
                hi=np.array([20.0, 55.0, 85.0, 45.0, 0.5, 0.5, 25.0, 25.0, 2.00])),
            fitness="focused_throughput",
            field_aggregate="worst",
            fields=(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0),
            descriptors=("tir_bounces", "thinness"),
            bc_bins=(3, 16),
            bc_bounds=((2.0, 5.0), (10.0, 122.0)),
            reject_out_of_bounds=True,
            require_monotone_convergence=True,
            convergence_gate="clean_pack",
            gate_all_fields=True,
            require_exit_convergence=True,
            resume_from=str(Path(__file__).resolve().parents[2]
                            / "runs" / "fold24_gated_seed"),
            param_scaling="normalized",
            cell_bc_keys=("tir_count", "thinness_T"),
            info_descriptors=("field_accept", "crit_margin", "beam_convergence"),
            evals=40000,
        ),
        "fold24_topo_gated": ExperimentConfig(
            name="fold24_topo_gated",
            label="fold24_topology_axis_gated",
            problem=FoldProblem(
                n_rays=17, relative_sag=True, max_sag_ratio=0.55, det_clearance=12.0,
                lo=np.array([-20.0, -55.0, 35.0, 6.0, -0.5, -0.5, -25.0, -25.0, 1.50]),
                hi=np.array([20.0, 55.0, 85.0, 45.0, 0.5, 0.5, 25.0, 25.0, 2.00])),
            fitness="focused_throughput",
            field_aggregate="worst",
            fields=(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0),
            descriptors=("tir_faces", "thinness"),
            bc_bins=(121, 4),
            bc_bounds=((5.0, 126.0), (10.0, 122.0)),
            reject_out_of_bounds=True,
            require_monotone_convergence=True,
            convergence_gate="clean_pack",
            gate_all_fields=True,
            require_exit_convergence=True,
            resume_from=str(Path(__file__).resolve().parents[2]
                            / "runs" / "fold24_gated_seed"),
            param_scaling="normalized",
            cell_bc_keys=("tir_faces", "thinness_T"),
            info_descriptors=("field_accept", "crit_margin", "beam_convergence"),
            evals=40000,
        ),
        # -- third curved face ---------------------------------------------------
        # fold24_topo_gated with ONE extra degree of freedom: the BOTTOM face carries a
        # sag too, so the element has three curved surfaces instead of two.
        #
        # The gene is `dsag`, and that is not a typo. LEGACY_SAG_KEY crosses the pair:
        # `dsag` -> "bottom" -> the face signatures label 'T', and `tsag` -> "top" -> 'D'.
        # The bottom face is the one the best designs in the program actually fold on
        # (0.878 and 0.860 both route T@A -> TIR@T -> TIR@B -> TIR@A -> T@B), so curving
        # it is the DOF most likely to buy something. Curving the top instead is the
        # obvious follow-up and is a one-line change to `tsag`.
        #
        # Under relative_sag the new gene is a fraction of its own half-chord, and the
        # bottom face's chord is `thick`, not `half_h` -- so +-0.5 here means +-1.5 mm on
        # a 6 mm element and +-11 mm on a 45 mm one, which is the point of relative sag.
        # Same bounds as fsag/bsag so no face gets a privileged range.
        #
        # A flat bottom face is dsag=0, i.e. the current env is the interior of this
        # genome box. The seed is the gated seed lifted with dsag=0 and re-scored here,
        # so this run starts from exactly the 2-curved designs and can only win by
        # USING the new surface. That makes the comparison against fold24_topo_gated
        # read directly as "what does the third curved face buy".
        "fold24_topo3_gated": ExperimentConfig(
            name="fold24_topo3_gated",
            label="fold24_topology_axis_gated_3curved",
            problem=FoldProblem(
                n_rays=17, relative_sag=True, max_sag_ratio=0.55, det_clearance=12.0,
                param_names=["ray_y", "ang", "cx", "thick", "fsag", "bsag", "dsag",
                             "tilt_a", "tilt_b", "n"],
                lo=np.array([-20.0, -55.0, 35.0, 6.0, -0.5, -0.5, -0.5,
                             -25.0, -25.0, 1.50]),
                hi=np.array([20.0, 55.0, 85.0, 45.0, 0.5, 0.5, 0.5,
                             25.0, 25.0, 2.00])),
            fitness="focused_throughput",
            field_aggregate="worst",
            fields=(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0),
            descriptors=("tir_faces", "thinness"),
            bc_bins=(121, 4),
            bc_bounds=((5.0, 126.0), (10.0, 122.0)),
            reject_out_of_bounds=True,
            require_monotone_convergence=True,
            convergence_gate="clean_pack",
            gate_all_fields=True,
            require_exit_convergence=True,
            resume_from=str(Path(__file__).resolve().parents[2]
                            / "runs" / "fold24_gated3_seed"),
            param_scaling="normalized",
            cell_bc_keys=("tir_faces", "thinness_T"),
            info_descriptors=("field_accept", "crit_margin", "beam_convergence"),
            evals=40000,
        ),
        # -- the perpendicular / forced-fold geometry (doc 16) ------------------
        # NOTE thinness bounds are (70,170), not (10,120): the detector sits at
        # x=170 here, so T is measured against a different sensor plane.
        "prism_fold_qd": ExperimentConfig(
            name="prism_fold_qd",
            label="prism_fold_qd",
            problem=PrismFoldProblem(),
            fitness="focused_throughput",
            descriptors=("tir_bounces", "thinness"),
            bc_bounds=((0.0, 9.0), (70.0, 170.0)),
        ),
    }


#: legacy `--bc` value -> descriptor name (run_fold_qd_descriptors.BC1 keys)
ABLATION_AXES = {
    "baseline": "tir_bounces",
    "nsfrac": "ns_frac",
    "tilt": "tilt_mag",
    "critmargin": "crit_margin",
    "locus": "fold_locus",
    "robust": "robustness",
    "thinness": "thinness",
}

#: every descriptor the ablation measures for every design, for info parity with
#: run_fold_qd_descriptors.py
_ABLATION_INFO = ("tir_bounces", "ns_frac", "tilt_mag", "crit_margin",
                  "fold_locus", "robustness", "thinness")


def descriptor_ablation(bc: str = "baseline", bc2: str = "thinness") -> ExperimentConfig:
    """Doc 12: which BC1 axis surfaces more good non-sequential designs, with the
    fold-blind fitness held fixed? Replaces run_fold_qd_descriptors.py --bc X --bc2 Y.

    Each axis brings its own calibrated binning from the descriptor registry, which is
    what the legacy BC1 dict had to keep in parallel with the descriptor code.
    """
    if bc not in ABLATION_AXES:
        raise KeyError(f"unknown --bc {bc!r}; known: {sorted(ABLATION_AXES)}")
    if bc2 not in ABLATION_AXES:
        raise KeyError(f"unknown --bc2 {bc2!r}; known: {sorted(ABLATION_AXES)}")
    return ExperimentConfig(
        name=f"desc_{bc}" + ("" if bc2 == "thinness" else f"_x_{bc2}"),
        label=f"fold_qd_desc_{bc}_x_{bc2}",
        problem=FoldProblem(),
        fitness="focused_throughput",
        descriptors=(ABLATION_AXES[bc], ABLATION_AXES[bc2]),
        info_descriptors=_ABLATION_INFO,
        cell_bc_keys=("bc1", "T"),
        meta_extra=dict(bc1_axis=bc, bc2_axis=bc2,
                        bc1_label=_axis_label(ABLATION_AXES[bc])),
    )


def _axis_label(name: str) -> str:
    from . import descriptors as _d
    return _d.axis_label(name)


def load_fixed_environment_specs(
        path: Path | str = DEFAULT_FIXED_ENVIRONMENT_SPECS) -> list[dict]:
    """Load and validate the accepted fixed-condition environment catalogue."""
    path = Path(path)
    payload = json.loads(path.read_text())
    rows = payload.get("environments")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} contains no fixed environments")
    accepted = [row for row in rows if row.get("status") == "accepted"]
    ids = [row.get("environment_id") for row in accepted]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{path} contains an accepted environment without an id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate fixed environment ids")
    return accepted


def fixed_environment(
        environment_id: str,
        specs_path: Path | str = DEFAULT_FIXED_ENVIRONMENT_SPECS) -> ExperimentConfig:
    """Build the interaction-aware CMA-ME config for one fixed environment.

    Topology is a contract here, not an archive axis.  The archive instead preserves
    form-factor/packing diversity while every admitted design has already demonstrated
    the requested complete path across the full field sweep.
    """
    rows = load_fixed_environment_specs(specs_path)
    found = [row for row in rows if row["environment_id"] == environment_id]
    if not found:
        known = ", ".join(row["environment_id"] for row in rows)
        raise KeyError(f"unknown fixed environment {environment_id!r}; known: {known}")
    spec = found[0]
    problem = problem_from_environment_spec(spec, n_rays=17)
    midpoint = ((np.asarray(problem.lo, dtype=float)
                 + np.asarray(problem.hi, dtype=float)) / 2.0)
    return ExperimentConfig(
        name=f"fixed_{environment_id}",
        label="fixed_fold_interaction_aware",
        problem=problem,
        fitness="intended_focused_throughput",
        intended_sequence=problem.intended_sequence,
        min_intended_root_support=0.80,
        min_intended_delivered_energy_share=0.05,
        require_intended_dominant_all_fields=True,
        min_intended_critical_margin_deg=0.50,
        fields=FIXED_FIELDS,
        field_aggregate="worst",
        descriptors=("thinness", "beam_convergence"),
        bc_bins=(12, 10),
        bc_bounds=((10.0, 120.0), (0.0, 2.0)),
        cell_bc_keys=("thinness_T", "beam_ratio"),
        info_descriptors=("field_accept", "crit_margin", "ns_frac"),
        reject_out_of_bounds=True,
        require_monotone_convergence=True,
        convergence_gate="clean_pack",
        gate_all_fields=True,
        require_exit_convergence=True,
        param_scaling="normalized",
        initial_genotype=midpoint,
        trace=True,
        evals=20000,
        meta_extra={
            "fixed_environment_id": environment_id,
            "fixed_environment_schema": spec.get("schema_version"),
            "fixed_environment_specs": str(Path(specs_path).resolve()),
            "seed_design_id": spec.get("seed_design_id"),
            "fixed_conditions": dict(spec["fixed"]),
            "intended_sequence": problem.intended_sequence,
        },
    )


REGISTRY: Dict[str, ExperimentConfig] = _experiments()


def get(name: str) -> ExperimentConfig:
    """Look up an experiment. `desc_<bc>` and `desc_<bc>_x_<bc2>` are generated."""
    if name.startswith("desc_"):
        rest = name[len("desc_"):]
        bc, _, bc2 = rest.partition("_x_")
        return descriptor_ablation(bc, bc2 or "thinness")
    if name not in REGISTRY:
        raise KeyError(f"unknown experiment {name!r}; known: {sorted(REGISTRY)}")
    # fresh instance each call: configs are mutated by CLI overrides
    return _experiments()[name]


def names():
    return sorted(REGISTRY)
