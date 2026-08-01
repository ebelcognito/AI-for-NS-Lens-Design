"""The experiment registry -- one entry per row of EXPERIMENTS.md.

Each entry below replaces a ~170-line runner script whose only distinguishing content
was the fitness, the descriptor pair, and the source/detector placement.

    python -m nslens.qd fold_qd --evals 14000 --seed 0 --out runs/fold_qd
    python -m nslens.qd --list
"""
from __future__ import annotations

from typing import Dict

from .driver import ExperimentConfig
from .arvr import SIGMA0 as ARVR_SIGMA0
from .arvr import ArvrProblem
from .problems import FoldProblem, PrismFoldProblem


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
