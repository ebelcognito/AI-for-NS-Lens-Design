"""The hardened RL environment config, and the record of how it diverges.

`ExperimentConfig` ships every gate OFF. That is the right default there and the wrong
one here, and the reason is worth stating once: the frozen archives on disk were built
before most of these gates existed, and `tests/test_qd_equivalence.py` reproduces them
cell-for-cell. Turning a gate on by default would move results the test compares against.

So the permissive values survive as the QD default, and each is a hole that CMA-ME already
walked through -- measured, not hypothesised. Doc 18 tabulates them; `DIVERGENCES` below is
that table in executable form, and `tests/test_rl_env_hardened.py` asserts every row still
differs from the `ExperimentConfig` default. If somebody "fixes" this file back to the
frozen defaults, that test fails and says which exploit they just reopened.

Two knobs the shared evaluator does not have live on `RLEnv` instead of on the config,
because adding them to `ExperimentConfig` would mean touching the frozen search path:

    min_crit_margin    knife-edge TIR. A fold whose dominant ray clears the critical angle
                       by 0.2 deg is a fold in the tracer and not in glass. `thin_design.py`
                       needs --min-margin 3 for exactly this reason.
    source_clearance   the embedded-source cheat: source geometry inside the element traces
                       as a spectacular fold and cannot be built. It taints every TIR@D cell
                       in the older archives.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..qd.driver import ExperimentConfig
from ..qd.problems import FoldProblem, PrismFoldProblem, Problem

#: the nine fields every hardened score in this program is quoted at (the catalogue's
#: canonical set). Fixed per env so a cached reward stays valid -- see server.py.
CANONICAL_FIELDS: Tuple[float, ...] = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)

#: |sag|/half-chord cap. 0.268 is radius >= 2x half-chord; the fold24 runs used the far
#: looser 0.55 and the legacy default permits q up to 5.0, which makes a "face" a bulb.
#: This is the catalogue's hardened value, so scores here are comparable to catalogue/.
MAX_SAG_RATIO = 0.268

#: mm of clear air between the element's rightmost point and the detector plane.
DET_CLEARANCE = 12.0

#: Degrees past critical that the DOMINANT BUNDLE's weakest TIR event must clear, at the
#: worst field that still folds. Two numbers because a single cliff is wrong here:
#: `MIN_CRIT_MARGIN` is the floor below which the design is a knife edge, and
#: `CRIT_MARGIN_TARGET` is where the reward stops paying more for robustness.
#:
#: Priced on the 95 ab40k elites (`reward.dominant_crit_margin`, sentinel-skipped), as
#: fraction kept / how many of the top ten survive / best design still kept:
#:     0.5 deg   59/85   10/10   0.878      <- the floor
#:     1.0 deg   53/85    7/10   0.878
#:     2.0 deg   42/85    3/10   0.858
#:     3.0 deg   34/85    1/10   0.843
#: 3.0 is where `thin_design.py --min-margin 3` sits and it CANNOT be the floor here: that
#: number was calibrated on the `crit_margin` descriptor, which reads the single brightest
#: ray at the central field, and the bundle statistic is much stricter. At 3.0 the gate
#: discards nine of the ten best designs in the program, including the 0.878 that is the
#: best fitness ever reached. This is the etendue-growth mistake the driver warns about,
#: caught before it was shipped rather than after.
#:
#: The finding underneath is worth stating: the fold24 family folds on 0.6-2 deg of margin
#: at its best. Every one of these designs is knife-edge by the standard the single-ray
#: descriptor implies, and a tolerance study is owed before any of them is called buildable.
MIN_CRIT_MARGIN = 0.5
CRIT_MARGIN_TARGET = 3.0

#: mm of clear air between the source and the element body.
SOURCE_CLEARANCE = 2.0


#: knob -> (frozen ExperimentConfig default, hardened value, the exploit it closes).
#: `Problem`-level knobs are named `problem.<attr>`.
DIVERGENCES: Dict[str, Tuple[object, object, str]] = {
    "field_aggregate": (
        "mean", "worst",
        "on-axis excellence, blind off-axis: the mean hides a field that catches nothing"),
    "gate_all_fields": (
        False, True,
        "beam gates measured at field 0 only -- 40/40 elites clean on axis, 16/40 clean "
        "at every field"),
    "convergence_gate": (
        "packing", "clean_pack",
        "caustics pass the width-only packing test: 26 of 42 elites in the run that "
        "exists to reject caustics"),
    "require_monotone_convergence": (
        False, True,
        "without it the packing verdict is recorded and never enforced"),
    "require_exit_convergence": (
        False, True,
        "clean_pack is structurally blind to the exit->detector leg, so 39 of 87 hardened "
        "elites never form an image"),
    "reject_out_of_bounds": (
        False, True,
        "GridArchive clamps, so an unfolded slab occupies a folded niche"),
    "problem.max_sag_ratio": (
        None, MAX_SAG_RATIO,
        "faces that are bulbs (arc > 180 deg); the legacy default permits q up to 5.0"),
    "problem.relative_sag": (
        False, True,
        "an absolute sag box means something different at every thickness, so a 6 mm "
        "element can carry a 15 mm bulb"),
    "problem.det_clearance": (
        None, DET_CLEARANCE,
        "body passes THROUGH the detector plane, which makes its thinness_T meaningless"),
}

#: `RLEnv` knobs with no `ExperimentConfig` counterpart, same shape as DIVERGENCES.
EXTRA_GATES: Dict[str, Tuple[object, object, str]] = {
    "min_crit_margin": (
        0.0, MIN_CRIT_MARGIN,
        "knife-edge TIR: continuation cheats via grazing incidence at a corner"),
    "crit_margin_target": (
        0.0, CRIT_MARGIN_TARGET,
        "above the floor, robustness is graded rather than gated -- see MIN_CRIT_MARGIN"),
    "source_clearance": (
        0.0, SOURCE_CLEARANCE,
        "embedded-source cheat: source geometry inside the glass"),
}

#: measured and reported, never gated on. Recorded here so the reason survives.
NOT_GATES: Dict[str, str] = {
    "etendue_growth":
        "the source is collimated, so entry angular spread is ~0 and ANY focusing reads "
        "as 6-10x growth. It rejects the two best designs in fold24_wide. Read emittance.",
    "sensor_ratio":
        "good designs focus early (20-45 mm of an ~85 mm gap) and re-expand to a spot that "
        "still lands in the window; a ratio to the exit width keeps 1 of 42 elites.",
}


@dataclass(frozen=True)
class RLEnv:
    """One RL environment: the scored config, plus the gates that live outside it.

    `cfg` carries the hardened THRESHOLDS and the descriptor axes. Its `require_*` flags are
    set so that `nslens.qd.driver.run(cfg)` -- used for task certification (`tasks.certify`)
    and as the held-out hardened eval config -- rejects. `reward.evaluate` deliberately does
    NOT reject: it reads the same thresholds and prices the verdicts. See `reward`.
    """

    name: str
    cfg: ExperimentConfig
    #: half-FOV in degrees that `cfg.fields` spans. Carried so a variant can be identified
    #: without re-deriving it from the field list.
    fov: float = 8.0
    min_crit_margin: float = MIN_CRIT_MARGIN
    crit_margin_target: float = CRIT_MARGIN_TARGET
    source_clearance: float = SOURCE_CLEARANCE
    #: held out of training entirely (doc 18 §3.2 test_env split)
    held_out: bool = False

    @property
    def problem(self) -> Problem:
        return self.cfg.problem

    @property
    def fields(self) -> Tuple[float, ...]:
        return tuple(self.cfg.fields or (0.0,))

    @property
    def param_names(self) -> Sequence[str]:
        return self.cfg.problem.param_names

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Genome box as {name: (lo, hi)} -- what goes into the prompt."""
        p = self.cfg.problem
        return {n: (float(lo), float(hi))
                for n, lo, hi in zip(p.param_names, p.lo, p.hi)}


def _fold_problem(**kw) -> FoldProblem:
    """The fold24 genome box, hardened. Sag genes are FRACTIONS of the face half-chord
    (relative_sag), which is why their box is [-0.5, 0.5] and not millimetres.

    That box is copied verbatim from `fold24_topo_gated` and is deliberately NOT retuned to
    match `MAX_SAG_RATIO`. It could be: the +-0.5 sag genes reach roughly twice the 0.268
    cap, so a large part of the box is unreachable, and measured on 4,000 uniform genotypes
    the breakdown is

        bulb_face 58%   degenerate 13%   detector_intersect 1%   buildable 28%

    -- i.e. the curvature cap alone refuses the majority of the box. Tightening it would
    raise the buildable fraction and would also make this a different search space from the
    one every CMA-ME archive on disk was built in, which would silently invalidate the
    equal-engine-call comparison that doc 18 §6 item 3 exists to run. Comparability is worth
    more than the throughput; the dead region is a cost, recorded here rather than removed.
    """
    return FoldProblem(
        n_rays=17, relative_sag=True, max_sag_ratio=MAX_SAG_RATIO,
        det_clearance=DET_CLEARANCE,
        lo=np.array([-20.0, -55.0, 35.0, 6.0, -0.5, -0.5, -25.0, -25.0, 1.50]),
        hi=np.array([20.0, 55.0, 85.0, 45.0, 0.5, 0.5, 25.0, 25.0, 2.00]),
        **kw)


def _prism_problem(**kw) -> PrismFoldProblem:
    """The perpendicular env: bundle enters from below, detector on the right, so straight
    transmission is geometrically impossible. Four curved faces -- a +-25 deg face cannot
    turn 90 deg. Held out as the transfer split.

    Only 6% of this box is buildable under the hardened geometry (55% bulb_face, 39%
    degenerate on 4,000 uniform genotypes), against 28% for `fold`. Two extra curved faces
    and +-50 deg tilts is a much wider box being cut by the same cap. It matters less than
    it looks because this env is never trained on -- but it does mean cold-start
    certification here will reach very few cells, and that limit should be quoted rather
    than worked around by loosening the gates for the transfer split.
    """
    return PrismFoldProblem(
        n_rays=17, relative_sag=True, max_sag_ratio=MAX_SAG_RATIO,
        det_clearance=DET_CLEARANCE,
        lo=np.array([15.0, 35.0, 6.0, -0.5, -0.5, -0.5, -0.5, -50.0, -50.0, 1.50]),
        hi=np.array([105.0, 85.0, 45.0, 0.5, 0.5, 0.5, 0.5, 50.0, 50.0, 2.00]),
        **kw)


def _hardened_cfg(name: str, problem: Problem, *, evals: int = 2000,
                  seed: int = 0) -> ExperimentConfig:
    """Every knob in DIVERGENCES, set to its hardened value.

    The topology axis (`tir_faces`) is the first descriptor rather than `tir_bounces`
    because it is what a spec can actually NAME. Under `tir_bounces` the fold24 regime
    offers three attainable values, which is not a conditioning label; `tir_faces` resolves
    79 observed classes and its cells ARE the ray paths (doc 19).

    bc_bounds start at 5.0 for the same reason `fold24_topo` does: vocabulary entries 0-4
    are the k=0 and k=1 multisets, so admitting them lets straight transmission -- which
    pays no fold losses and scores ~0.92 -- win a folding task.
    """
    return ExperimentConfig(
        name=name,
        label=f"rl_{name}_hardened",
        problem=problem,
        fitness="focused_throughput",
        field_aggregate="worst",
        fields=CANONICAL_FIELDS,
        descriptors=("tir_faces", "thinness"),
        bc_bins=(121, 4),
        bc_bounds=((5.0, 126.0), (10.0, 122.0)),
        reject_out_of_bounds=True,
        require_monotone_convergence=True,
        convergence_gate="clean_pack",
        gate_all_fields=True,
        require_exit_convergence=True,
        exit_window_mult=1.0,
        param_scaling="normalized",
        cell_bc_keys=("tir_faces", "thinness_T"),
        info_descriptors=("field_accept", "crit_margin", "beam_convergence"),
        evals=evals,
        seed=seed,
    )


def _envs() -> Dict[str, RLEnv]:
    return {
        "fold": RLEnv(name="fold", cfg=_hardened_cfg("fold", _fold_problem())),
        # doc 18 §3.2: an entire environment never trained on, so held-out performance
        # measures transfer rather than interpolation. Doc 16 already shows the
        # perpendicular geometry behaves differently, which is what makes it a real test.
        "prism": RLEnv(name="prism", cfg=_hardened_cfg("prism", _prism_problem()),
                       held_out=True),
    }


_CACHE: Dict[str, RLEnv] = {}
_VARIANTS: Dict[Tuple[str, float, float], RLEnv] = {}

#: half-FOV values a spec may ask for, in degrees. The canonical 8.0 is the set every
#: hardened score in this program is quoted at; the others widen the task distribution,
#: which is what §3.1 of doc 18 needs and what the 484-cell descriptor grid cannot supply
#: on its own.
FOV_CHOICES: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0)

#: half-width of the scoring window at the detector, mm. Smaller = a tighter spot demanded.
FOCUS_W_CHOICES: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0)

#: rays per field. Fixed, never a spec axis: a cached verdict must stay valid, and ray
#: count changes the score.
N_RAYS = 17


def field_set(fov: float, n: int = 9) -> Tuple[float, ...]:
    """`n` field angles spanning +-fov, always including 0.0 -- the central field defines
    the behaviour coordinates, so it cannot be sampled away."""
    return tuple(round(float(v), 6) for v in np.linspace(-fov, fov, n))


def get(name: str, *, fov: Optional[float] = None,
        focus_w: Optional[float] = None) -> RLEnv:
    """The env, optionally at a different FOV or detector window.

    A spec names a target topology at a FOV and a spot tolerance, so those two have to be
    part of the environment rather than of the prompt text alone -- `evaluate` reads them
    off the env. Everything else (genome box, gates, descriptor axes, ray count) is fixed,
    which is what keeps a cached verdict valid across specs that share a variant.
    """
    if not _CACHE:
        _CACHE.update(_envs())
    if name not in _CACHE:
        raise KeyError(f"unknown RL env {name!r}; known: {sorted(_CACHE)}")
    base = _CACHE[name]
    if fov is None and focus_w is None:
        return base
    fov = float(base.fov if fov is None else fov)
    focus_w = float(base.problem.focus_w if focus_w is None else focus_w)
    key = (name, fov, focus_w)
    if key not in _VARIANTS:
        problem = replace(base.problem, focus_w=focus_w)
        cfg = replace(base.cfg, problem=problem, fields=field_set(fov),
                      name=f"{name}_fov{fov:g}_w{focus_w:g}")
        _VARIANTS[key] = replace(base, cfg=cfg, fov=fov)
    return _VARIANTS[key]


def names(*, include_held_out: bool = True) -> list:
    if not _CACHE:
        _CACHE.update(_envs())
    return sorted(n for n, e in _CACHE.items() if include_held_out or not e.held_out)


def train_names() -> list:
    return names(include_held_out=False)


def divergences(env: RLEnv) -> Dict[str, Tuple[object, object, str]]:
    """Which hardened knobs actually differ from the frozen defaults, as measured on
    `env` rather than as written down. The test asserts this covers all of DIVERGENCES;
    an empty entry means a gate silently reverted."""
    frozen = {f.name: f.default for f in dc_fields(ExperimentConfig)}
    prob_frozen = {f.name: f.default for f in dc_fields(Problem)}
    out: Dict[str, Tuple[object, object, str]] = {}
    for knob, (expect_frozen, expect_hard, why) in DIVERGENCES.items():
        if knob.startswith("problem."):
            attr = knob.split(".", 1)[1]
            actual, default = getattr(env.problem, attr), prob_frozen.get(attr)
        else:
            actual, default = getattr(env.cfg, knob), frozen.get(knob)
        if actual != default:
            out[knob] = (default, actual, why)
    for knob, (expect_frozen, expect_hard, why) in EXTRA_GATES.items():
        actual = getattr(env, knob)
        if actual != expect_frozen:
            out[knob] = (expect_frozen, actual, why)
    return out


def report(env: Optional[RLEnv] = None) -> str:
    """The divergence table, printable. Used by the milestone-1 check."""
    env = env or get("fold")
    rows = divergences(env)
    w = max(len(k) for k in rows)
    lines = [f"RL env {env.name!r}: {len(rows)} gates hardened against the frozen defaults",
             ""]
    for knob, (frozen, hard, why) in rows.items():
        lines.append(f"  {knob:<{w}}  {frozen!r:>8} -> {hard!r:<8}  {why}")
    lines += ["", "measured but deliberately NOT gated:"]
    for knob, why in NOT_GATES.items():
        lines.append(f"  {knob:<{w}}  {why}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
