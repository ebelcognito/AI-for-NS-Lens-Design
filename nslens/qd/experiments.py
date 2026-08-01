"""The experiment registry -- one entry per row of EXPERIMENTS.md.

Each entry below replaces a ~170-line runner script whose only distinguishing content
was the fitness, the descriptor pair, and the source/detector placement.

    python -m nslens.qd fold_qd --evals 14000 --seed 0 --out runs/fold_qd
    python -m nslens.qd --list
"""
from __future__ import annotations

from typing import Dict

from .driver import ExperimentConfig
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


REGISTRY: Dict[str, ExperimentConfig] = _experiments()


def get(name: str) -> ExperimentConfig:
    if name not in REGISTRY:
        raise KeyError(f"unknown experiment {name!r}; known: {sorted(REGISTRY)}")
    # fresh instance each call: configs are mutated by CLI overrides
    return _experiments()[name]


def names():
    return sorted(REGISTRY)
