"""Genotype -> reward, with every gate verdict kept instead of thrown away.

`nslens.qd.driver._evaluate` returns None the moment a design fails a gate. That is right
for CMA-ME -- an invalid proposal simply ranks last -- and wrong for a policy gradient,
which needs to know that THIS failure was a bulb face and THAT one was a caustic, and needs
partial credit in between or every rollout in a group scores the same and the step is
wasted (doc 18 §4.3, advantage collapse).

So this module measures what `_evaluate` measures, in the same order, from the same
primitives, and rejects nothing. Gates become stages of a staged reward. `tests/
test_rl_reward.py` asserts that on any design reaching `scored`, the fitness and behaviour
coordinates here are bit-identical to the driver's under the gated config -- the reward and
the search must agree about what a design is worth, or the RL result cannot be compared to
the CMA-ME baseline at all.

Two prices are deliberately NOT paid here:

  * the driver gates on the central field before tracing the other eight, which turns a
    rejected design into one engine call instead of nine. That optimisation cannot survive
    a graded reward, because the grade needs the other eight fields. At 291 nine-field
    designs/s the whole thing is noise against LLM generation, so this traces all nine
    always (doc 18 §0.1).
  * `etendue_growth` and the ratio-flavoured exit tests are measured and reported but never
    priced. Both were measured to reject the best designs in the program. See
    `env_config.NOT_GATES`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..qd import descriptors as descriptor_registry
from ..qd import fitness as fitness_registry
from ..qd.driver import _beam_across_fields
from ..qd.observation import measure
from ..qd.descriptors import TIR_FACE_VOCAB, tir_face_label
from ..engine.signatures import format_signature
from .env_config import RLEnv

# ---------------------------------------------------------------- reward weights
#: in-bounds parameter vector. Assigned by the caller (see `parse`), never earned here.
R_PARSED = 0.05
#: a buildable element: non-degenerate, no bulb face, clear of the detector and the source.
R_GEOM = 0.15
W_FIT = 0.35      # worst-field throughput, normalised by the spec's certified achievable
W_NICHE = 0.20    # did it land in the requested descriptor cell
W_GATE = 0.15     # beam quality, graded over fields rather than all-or-nothing
#: 0.85, not 1.0. GRPO uses group-relative advantage, so the absolute ceiling is
#: irrelevant; keeping the plan's weights literal is worth more than a round number.
R_MAX = R_GEOM + W_FIT + W_NICHE + W_GATE

#: The binary acceptance rule used for positive datasets and final evaluation. During RL
#: these same gates remain graded through ``gate_score`` so near-misses provide a gradient;
#: ``hardened_pass`` is logged separately so a high shaped reward cannot be mistaken for a
#: physically acceptable design.
HARD_GATE_KEYS = ("clean_all", "exit_ok_all", "margin_ok")

#: stages, worst to best. `off_class` sits above `no_light` because the design does image,
#: it just images outside the fold class the task asked for.
STAGES = ("degenerate", "bulb_face", "detector_intersect", "embedded_source",
          "engine_error", "no_light", "off_class", "scored")

#: everything at or below this stage earns R_PARSED only -- the element cannot be built.
_UNBUILDABLE = ("degenerate", "bulb_face", "detector_intersect", "embedded_source",
                "engine_error")


@dataclass(frozen=True)
class Spec:
    """One task: build a design of THIS topology, at THIS thickness, in THIS env.

    `achievable` is the score a short CMA-ME probe actually reached on this spec
    (`tasks.certify`), not a guess. Normalising `f_norm` by it is what stops easy specs
    dominating the gradient purely because they score higher in absolute terms -- without
    it the policy learns to ignore the hard half of the curriculum (doc 18 §2.2).
    """

    env: str
    topology: int                 # index into descriptors.TIR_FACE_VOCAB
    thinness_bin: int             # index into the thinness axis of the env's grid
    achievable: float = 1.0
    spec_id: str = ""
    #: measured by `tasks.certify`, carried for the curriculum sampler
    difficulty: Optional[float] = None
    #: the two axes that widen the spec space past the 484-cell descriptor grid. They
    #: select an ENV VARIANT rather than only prompt text -- half-FOV in degrees and the
    #: half-width of the detector window fitness scores in. See tasks.generate.
    fov: float = 8.0
    focus_w: float = 8.0

    @property
    def variant(self) -> Tuple[str, float, float]:
        return (self.env, self.fov, self.focus_w)

    @property
    def topology_label(self) -> str:
        return tir_face_label(self.topology)

    def describe(self) -> str:
        return (f"{self.env} (fov +-{self.fov:g} deg, window +-{self.focus_w:g} mm): "
                f"topology {self.topology_label!r} (cell {self.topology}), "
                f"thinness bin {self.thinness_bin}, achievable {self.achievable:.3f}")


@dataclass
class Verdict:
    """Everything one traced design is known to be. Nothing here is a rejection."""

    stage: str
    reason: str = ""
    fitness: float = 0.0                       # worst-field focused throughput
    per_field: Tuple[float, ...] = ()
    bc: Optional[Tuple[float, float]] = None
    topology: Optional[int] = None
    thinness_bin: Optional[int] = None
    thinness_T: Optional[float] = None
    signature: str = ""
    spot: Optional[float] = None
    crit_margin: Optional[float] = None
    gates: Dict[str, object] = field(default_factory=dict)
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        return self.stage == "scored"

    @property
    def buildable(self) -> bool:
        return self.stage not in _UNBUILDABLE


@dataclass
class Reward:
    """A scalar plus the reason it is that number. Log the parts, not just the total."""

    total: float
    stage: str
    f_norm: float = 0.0
    niche: float = 0.0
    gate: float = 0.0
    hardened_pass: bool = False
    parts: Dict[str, float] = field(default_factory=dict)


def hardened_pass(v: Verdict) -> bool:
    """True only for an in-class design clearing every non-negotiable final gate.

    Geometry faults (including detector intersection and embedded source) cannot reach the
    ``scored`` stage, so this single predicate covers both pre-trace physical validity and
    the post-trace beam/exit/critical-margin requirements.
    """
    return bool(v.stage == "scored" and all((v.gates or {}).get(k)
                                             for k in HARD_GATE_KEYS))


# ------------------------------------------------------------------ geometry gates
def _source_clear(env: RLEnv, p: Dict[str, float]) -> Tuple[bool, float]:
    """Does the glass keep clear of the source line?

    Source geometry INSIDE the element traces as a spectacular fold and cannot be built;
    it taints every TIR@D cell in the older archives. The source is a line at a fixed
    coordinate on one axis (x for the collinear fold env, y for the perpendicular prism),
    with the bundle propagating away from it, so the test is one-dimensional: the whole
    boundary -- arc control points included -- must sit past it by `source_clearance`.
    """
    layout = env.problem.layout(p)
    src = layout.source
    verts = env.problem.element(p).path()
    key = "x" if getattr(src, "axis", "y") == "y" else "y"
    near = min(float(v[key]) for v in verts)
    return bool(near > src.origin + env.source_clearance), near - float(src.origin)


def dominant_crit_margin(obs, n_glass: float) -> float:
    """Degrees past critical at the weakest TIR event of the DOMINANT BUNDLE.

    The registered `crit_margin` descriptor takes the single BRIGHTEST ray, which is not
    the same thing and is wrong for a gate. Measured on the catalogue's 0.742 four-fold
    design: its brightest ray carries 0.79 and goes straight through (`T@A -> T@B`, no TIR
    at all), while the 11-ray bundle that defines the signature and sets the fitness folds
    four times. The descriptor therefore returns the no-TIR sentinel (-5.0) for a design
    whose whole point is four folds, and gating on it would reject the design for a
    property of a ray it is not about.

    Min over every TIR event of every ray in the bundle, which is the conservative reading:
    the knife-edge cheat is one ray of the working bundle grazing a corner, and an average
    hides exactly that. Returns `descriptors.CRIT_NOTIR` when the bundle really has no TIR.
    """
    grp = obs.dominant_rays()
    if not grp:
        return descriptor_registry.CRIT_NOTIR
    theta_c = math.degrees(math.asin(min(1.0 / float(n_glass), 1.0)))
    margins = []
    for r in grp:
        seq = r["seq"]
        for k in range(1, len(seq) - 1):
            if seq[k].get("t") == "tir":
                ti = descriptor_registry.incidence_deg(seq, k)
                if ti is not None:
                    margins.append(ti - theta_c)
    return float(min(margins)) if margins else descriptor_registry.CRIT_NOTIR


def _bin_of(value: float, lo: float, hi: float, bins: int) -> int:
    """Which grid bin a behaviour coordinate falls in; -1 when it is out of bounds.

    Matches GridArchive's edges but does NOT clamp -- clamping is what let an unfolded
    design occupy the '2 folds' niche, and is the failure `reject_out_of_bounds` exists
    to fix (doc 19 §5.1).
    """
    if not (lo <= value <= hi):
        return -1
    idx = int((value - lo) / (hi - lo) * bins)
    return min(idx, bins - 1)


# ------------------------------------------------------------------ the evaluation
def geometry_verdict(env: RLEnv, x: Sequence[float]) -> Optional[Verdict]:
    """The tests decided before a ray is traced, or None if the element is buildable.

    Pure arithmetic on the element polygon, so it costs nothing and can be run anywhere --
    which is the point: four of the exploits in `env_config.DIVERGENCES` are geometry
    faults, and a design that fails one should never reach the engine at all.
    """
    problem = env.problem
    p = problem.vec_to_params(x)
    el = problem.element(p)

    if not el.is_valid():
        return Verdict("degenerate", "tilts fold the body inside out (edge width <= 1 mm)")
    q = el.max_sag_ratio()
    if not problem.sag_ok(el):
        return Verdict("bulb_face",
                       f"steepest face q={q:.3f} > {problem.max_sag_ratio}",
                       info=dict(max_sag_ratio=float(q)))
    if not problem.detector_clear(el):
        return Verdict("detector_intersect",
                       f"glass reaches x={el.x_extent()[1]:.1f}, detector plane at "
                       f"{problem.x_sensor:.1f} needs {problem.det_clearance:.0f} mm clear")
    src_ok, src_gap = _source_clear(env, p)
    if not src_ok:
        return Verdict("embedded_source",
                       f"boundary sits {src_gap:.1f} mm from the source line, "
                       f"needs {env.source_clearance:.1f}")
    return None


def evaluate(runner, env: RLEnv, x: Sequence[float]) -> Verdict:
    """Trace one genotype at every field of `env` and report everything measured.

    Ordered cheapest-first so a design that cannot be built never reaches the engine.
    """
    problem, cfg = env.problem, env.cfg
    p = problem.vec_to_params(x)
    bad = geometry_verdict(env, x)
    if bad is not None:
        return bad
    q = problem.element(p).max_sag_ratio()

    fit_fn = fitness_registry.get(cfg.fitness)
    face_of = problem.face_of(p)
    fields = list(env.fields)
    i0 = fields.index(0.0)

    def observe(field_deg: float):
        try:
            res, _ = runner.simulate(
                problem.layout(p, field_deg=field_deg).to_scene_dict())
        except Exception as exc:                       # engine death, not a design fault
            return exc
        return measure(res, face_of, x_sensor=problem.x_sensor, allow_empty=True)

    obs_list = []
    for f in fields:
        o = observe(f)
        if isinstance(o, Exception):
            return Verdict("engine_error", f"{type(o).__name__}: {o}")
        if o is None:
            return Verdict("engine_error", "tracer returned an error result")
        obs_list.append(o)
    central = obs_list[i0]
    if central.is_empty:
        return Verdict("no_light",
                       "no forward ray reaches the detector with a valid signature "
                       "on axis")

    vals = [0.0 if o.is_empty else float(fit_fn(o, focus_w=problem.focus_w))
            for o in obs_list]
    fitness = float(min(vals))                          # field_aggregate="worst"

    rise_tol, env_tol, regrow_tol = cfg.convergence_tol
    convs = [o.convergence(rise_tol=rise_tol, envelope_tol=env_tol,
                           regrow_tol=regrow_tol, n_glass=p.get("n"))
             for o in obs_list]
    exits = [o.exit_convergence() for o in obs_list]
    sweep = _beam_across_fields(convs, fields)

    exit_w_max = 2.0 * problem.focus_w * cfg.exit_window_mult

    def exit_ok(e) -> bool:
        return bool(e["measured"] and e["focus_dist"] is not None and e["focus_dist"] > 0.0
                    and e["width_sensor"] is not None and e["width_sensor"] <= exit_w_max)

    live_e = [e for e in exits if e and e["measured"]]
    exit_flags = [exit_ok(e) for e in exits]
    exit_all = bool(all(exit_flags) and len(live_e) == len(fields))

    ctx = descriptor_registry.DescriptorContext(
        obs=central, problem=problem, params=p, genotype=np.asarray(x, float),
        fitness=fitness, runner=runner, with_robust=False,
        fields=tuple(fields), per_field=tuple(vals), convergence=convs[i0])
    measured = {}
    for d in tuple(cfg.descriptors) + tuple(cfg.info_descriptors):
        fn = descriptor_registry.get(d)
        if fn.descriptor_name not in measured:
            measured[fn.descriptor_name] = float(fn(ctx))
    bc = tuple(measured[descriptor_registry.get(d).descriptor_name]
               for d in cfg.descriptors)

    # The gate reads the BUNDLE's margin at its worst field; the descriptor's
    # brightest-ray value is kept in `info` so archive-era numbers stay comparable.
    #
    # CRIT_NOTIR (-5.0) is a SENTINEL for "this bundle does not TIR", not a margin, and
    # min()-ing over it is the mistake it invites: 36 of 40 fold24_clean elites change
    # dominant signature across the field sweep, so an edge field whose bundle stops
    # folding would drag the worst-field margin to -5.0 and the gate would read a
    # topology change as a knife edge. Measured, that is not a corner case -- it moves
    # the median margin over the ab40k elites from -5.0 to 0.66. Fields with no TIR are
    # skipped here and counted in `margin_fields`; fitness and `clean_all` are what price
    # a field that stopped working.
    margins = [m for m in (dominant_crit_margin(o, p["n"])
                           for o in obs_list if not o.is_empty)
               if m > descriptor_registry.CRIT_NOTIR]
    margin = float(min(margins)) if margins else descriptor_registry.CRIT_NOTIR
    (t_lo, t_hi), (th_lo, th_hi) = cfg.bc_bounds
    topo_idx = int(bc[0]) if t_lo <= bc[0] <= t_hi else -1
    thin_bin = _bin_of(bc[1], th_lo, th_hi, cfg.bc_bins[1])

    gates = dict(
        clean_all=bool(sweep["clean_all"]), clean_frac=float(sweep["clean_frac"] or 0.0),
        exit_ok_all=exit_all,
        exit_ok_frac=float(sum(exit_flags)) / len(fields),
        margin_ok=bool(margin >= env.min_crit_margin),
        crit_margin=margin,
        crit_margin_brightest=float(measured.get("crit_margin", 0.0)),
        margin_fields=len(margins),
        worst_field=sweep["worst_field"],
        # reported, never priced -- see env_config.NOT_GATES
        etendue_max_worst=sweep["etendue_max_worst"],
        emittance_max_worst=sweep["emittance_max_worst"],
        crossings_worst=sweep["crossings_worst"],
    )
    info = dict(
        max_sag_ratio=float(q), spot=float(central.spot),
        efficiency=float(central.delivered) / central.emitted,
        dom_energy_share=float(central.dominant_energy / central.emitted),
        n_forward=len(central.records),
        exit_width_sensor_worst=(float(max(e["width_sensor"] for e in live_e))
                                 if live_e else None),
        **{k: v for k, v in measured.items() if k not in ("tir_faces", "thinness")},
    )

    stage = "scored" if (topo_idx >= 0 and thin_bin >= 0) else "off_class"
    reason = "" if stage == "scored" else (
        f"behaviour coordinates {bc} fall outside {cfg.bc_bounds}: "
        f"{'fold depth outside 2-5' if topo_idx < 0 else 'thickness outside the grid'}")
    return Verdict(
        stage=stage, reason=reason, fitness=fitness,
        per_field=tuple(round(float(v), 6) for v in vals), bc=bc,
        topology=topo_idx if topo_idx >= 0 else None,
        thinness_bin=thin_bin if thin_bin >= 0 else None,
        thinness_T=float(measured.get("thinness", bc[1])),
        signature=format_signature(central.dominant_signature),
        spot=float(central.spot), crit_margin=margin, gates=gates, info=info)


# ------------------------------------------------------------------ the reward
def topology_distance(a: int, b: int) -> int:
    """Size of the symmetric difference of two TIR-face multisets.

    Vocabulary INDEX distance is meaningless -- the enumeration is grouped by fold depth,
    so neighbouring indices are unrelated routes. What a designer would call 'close' is
    how many bounces you would have to add or remove: A2 B1 is one bounce from A2 B1 T1
    and three from B1 T2.
    """
    if a == b:
        return 0
    if not (0 <= a < len(TIR_FACE_VOCAB) and 0 <= b < len(TIR_FACE_VOCAB)):
        return 99
    ma, mb = dict(TIR_FACE_VOCAB[a]), dict(TIR_FACE_VOCAB[b])
    return sum(abs(ma.get(f, 0) - mb.get(f, 0)) for f in set(ma) | set(mb))


def niche_match(v: Verdict, spec: Spec) -> float:
    """How close the design landed to the requested cell, in [0, 1].

    Graded, not binary: a 0/1 niche term makes every near-miss identical to a wild miss,
    and on a hard spec that is a group of rollouts with zero variance. Weighted 0.7/0.3
    toward topology because the topology is the task -- thickness is a continuous knob on
    a route that already exists.
    """
    if v.topology is None or v.thinness_bin is None:
        return 0.0
    d = topology_distance(v.topology, spec.topology)
    topo = 1.0 if d == 0 else max(0.0, 1.0 - d / 4.0)
    dt = abs(v.thinness_bin - spec.thinness_bin)
    thin = 1.0 if dt == 0 else max(0.0, 1.0 - dt / 3.0)
    return 0.7 * topo + 0.3 * thin


def gate_score(v: Verdict, env: RLEnv) -> float:
    """Beam quality in [0, 1], graded over fields.

    `clean_all` and `exit_ok_all` are the verdicts the search gates on, and as a reward
    term they are cliffs: a design clean at 8 of 9 fields scores exactly what one clean at
    0 of 9 does. The per-field fractions are already measured, so grade on those.

    The margin term is a ramp from `min_crit_margin` (the floor, below which the fold is a
    knife edge) to `crit_margin_target` (where more robustness stops being paid for). A
    binary term at the target would be worse than useless: measured on the ab40k elites it
    scores nine of the ten best designs in the program at zero. See env_config.MIN_CRIT_MARGIN.
    """
    g = v.gates
    m = float(g.get("crit_margin", 0.0))
    span = max(env.crit_margin_target - env.min_crit_margin, 1e-9)
    margin = float(np.clip((m - env.min_crit_margin) / span, 0.0, 1.0))
    return float(np.mean([float(g.get("clean_frac", 0.0)),
                          float(g.get("exit_ok_frac", 0.0)),
                          margin]))


def reward_of(v: Verdict, spec: Spec, env: RLEnv) -> Reward:
    """Stage the verdict into a scalar. See doc 18 §2.2 for the shape.

    `off_class` earns the geometry credit and nothing else. It is a buildable element that
    images -- it simply is not in the fold class the task named -- and paying it fitness
    credit would reward the doc-19 §5.1 failure directly: straight transmission scores
    ~0.92 precisely because it pays no fold losses, so any task-blind fitness term makes
    'do not fold' the dominant strategy.

    The crit-margin floor is the one gate in `env_config` NOT enforced as a stage here. It
    is the bottom of the ramp in `gate_score` instead, because as a stage it is a cliff at
    a threshold that 26 of 85 known-good elites sit below, and cliffs are what produce the
    zero-variance groups in doc 18 §4.3. It IS enforced where rejection is the right
    behaviour: `tasks.certify` refuses to certify a spec whose achievable design is a
    knife edge, so no spec ever asks for one.
    """
    if v.stage in _UNBUILDABLE:
        return Reward(R_PARSED, v.stage, hardened_pass=False,
                      parts=dict(parsed=R_PARSED))
    if v.stage in ("no_light", "off_class"):
        return Reward(R_GEOM, v.stage, hardened_pass=False,
                      parts=dict(geometry=R_GEOM))

    denom = max(float(spec.achievable), 1e-6)
    f_norm = float(np.clip(v.fitness / denom, 0.0, 1.0))
    niche = niche_match(v, spec)
    gate = gate_score(v, env)
    parts = dict(geometry=R_GEOM, fitness=W_FIT * f_norm,
                 niche=W_NICHE * niche, gate=W_GATE * gate)
    return Reward(float(sum(parts.values())), v.stage,
                  f_norm=f_norm, niche=niche, gate=gate,
                  hardened_pass=hardened_pass(v), parts=parts)


def score(runner, env: RLEnv, x: Sequence[float], spec: Spec) -> Tuple[Reward, Verdict]:
    """The whole path: genotype -> reward + everything the reward was computed from."""
    v = evaluate(runner, env, x)
    return reward_of(v, spec, env), v
