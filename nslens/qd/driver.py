"""The CMA-ME driver: the loop that was copy-pasted into six runners.

The RNG protocol is reproduced EXACTLY as the legacy runners had it, because archives
are compared cell-for-cell against committed results:

  rng          = default_rng(seed)
  emitter i    seeded with seed + i at startup
  restarts     seeded from rng.integers(0, 1<<30)
  new emitter  starts from archive.random_elite(rng), else rng.uniform(lo, hi)

The ranking keys are also the legacy ones: 2e12 + fitness for a NEW cell, 1e12 + delta
for an IMPROVED cell, plain fitness otherwise, -1e9 for an invalid evaluation. Those
magic offsets impose new > improved > kept, which is what makes this CMA-ME rather than
plain CMA-ES. Changing any of it changes the search.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..engine import Runner
from ..engine.signatures import format_signature, parse_target
from ..search import CMAES, GridArchive
from . import descriptors as descriptor_registry
from . import fitness as fitness_registry
from .interaction_objective import (target_critical_margin,
                                    target_path_evidence)
from .observation import measure
from .problems import Problem, lock_tilt


@dataclass
class ExperimentConfig:
    """One QD experiment. This replaces a ~170-line runner script."""

    name: str
    problem: Problem
    fitness: str = "focused_throughput"
    descriptors: Sequence[str] = ("tir_bounces", "thinness")
    bc_bins: Optional[Sequence[int]] = None       # None -> take from the descriptors
    bc_bounds: Optional[Sequence[Sequence[float]]] = None
    evals: int = 14000
    seed: int = 0
    emitters: int = 5
    #: emitter update policy. "cma" is the frozen default; "evotf" is the official
    #: release; the trained arms load an nslens Optical-EvoTF checkpoint.
    emitter: str = "cma"
    popsize: int = 10
    sigma0: float = 0.12
    #: EvoTF's bounded generation-history window.  The released default is 100.
    evotf_context_len: int = 100
    evotf_checkpoint: Optional[str] = None
    optical_adapters: Sequence[str] = (
        "parameter_role", "environment", "status_failure", "route", "archive")
    optical_categorical_mode: str = "learned"
    #: "isotropic" (the frozen default) gives every genotype dimension the SAME absolute
    #: step, sigma0 * mean(hi - lo). Dimensions whose own range is far below that mean
    #: are proposed outside their bounds and come back clipped -- in the prism envs the
    #: step is ~1100% of n's range, so 94% of arvr elites sit on an index rail.
    #: "normalized" moves the emitter into the unit cube, so sigma0 means "this fraction
    #: of each parameter's own range" in every dimension. It CHANGES THE SEARCH, so it
    #: is opt-in and archives from the two arms are not comparable cell-for-cell.
    param_scaling: str = "isotropic"
    lock_tilt: bool = False
    fields: Optional[Sequence[float]] = None      # multifield: aggregate over these
    #: A fixed-condition environment names the optical route it exists to realise.
    #: Empty preserves the historical discovery experiments, where topology is inferred
    #: from the dominant delivered signature instead of prescribed.
    intended_sequence: str = ""
    #: Minimum fraction of distinct source-ray roots that must arrive on the complete
    #: intended path at every sampled field. Descendant branches never inflate this.
    min_intended_root_support: float = 0.0
    #: Minimum mean input-energy fraction delivered through the complete intended path.
    min_intended_delivered_energy_share: float = 0.0
    #: Reject a design when any sampled field is dominated by a different path.
    require_intended_dominant_all_fields: bool = False
    #: Conservative minimum over every TIR event of every intended-path ray and field.
    #: None leaves the historical experiments unchanged.
    min_intended_critical_margin_deg: Optional[float] = None
    #: how a multifield evaluation collapses to one number. "mean" is the legacy
    #: aggregate; "worst" scores the WORST field, so a design only survives by working
    #: across the whole field set rather than by being excellent on axis and blind off it.
    field_aggregate: str = "mean"
    #: require the dominant bundle to converge MONOTONICALLY from the entry surface to
    #: the exit. A fold exists to pack light; a bundle that leaves a surface wider than
    #: it arrived has spent volume for nothing, and one that collapses to a waist inside
    #: the glass and re-expands is a caustic, not a packed beam. Both fail this test.
    #: See Observation.convergence(): strict leg-by-leg monotonicity is too brittle to
    #: gate on (it rejects fold24_wide's best design over a 5.4% wobble at the entry
    #: surface), so this requires the ENVELOPE to converge -- net narrowing, no bulge
    #: above the entry width, and no re-expansion after the waist.
    require_monotone_convergence: bool = False
    #: thresholds for that test, in the order (single-leg rise, envelope, regrow)
    convergence_tol: tuple = (0.02, 0.20, 1.5)
    #: WHICH verdict the gate reads. "packing" is the width-only envelope test described
    #: above. It is not enough: measured on the archives it admits caustics -- 10 of
    #: fold24_wide's 27 survivors and 26 of fold24_pack's own 42 elites cross an internal
    #: focus, i.e. the run that exists to reject caustics is 62% caustics. "clean_pack"
    #: additionally requires no internal ray-order inversion, which detects a focus
    #: without needing a length scale. See Observation.focus_crossings().
    convergence_gate: str = "packing"
    #: require the convergence_gate verdict at EVERY field, not just the central one.
    #: Applied only to designs that already passed the cheap central gate and have
    #: therefore already been traced at every field, so it costs no extra engine calls.
    #:
    #: Priced on 900 designs resampled from the previous run's trace -- the population the
    #: search really proposes, not uniform noise -- as fraction of viable kept:
    #:     clean_pack at every field   14.7%
    #:     clean_frac >= 0.8           28.4%
    #:     etendue growth <= 2.0       42.3%
    #: The etendue threshold looks like the better trade and is NOT: rendered, it rejects
    #: the two best designs in the run (0.838 and 0.832), both of which are tight, stable
    #: bundles identical at all nine fields. Their etendue "grows" 6.3x and 9.8x purely
    #: because the source is collimated -- entry angular spread is ~0, so any focusing at
    #: all divides by nearly nothing. beam_legs' own docstring says this: etendue grows on
    #: a converging leg with zero aberration, read emittance instead. So it cannot gate.
    #:
    #: clean_all is the verdict that separates them (0.78 clean_frac for the bad design,
    #: 1.00 for both good ones). Its 14.7% is a COLD-START problem, not a correctness one:
    #: CMA-ME emitters need elites to sample around, and from an empty archive they never
    #: get them (the first attempt filled 0-3 of 48 cells in 4k evals, best fitness 0.03).
    #: Pair it with resume_from so the emitters start from a populated archive.
    gate_all_fields: bool = False
    #: reject designs whose behaviour coordinates fall outside bc_bounds instead of
    #: letting GridArchive clamp them into the edge cells. Clamping is what lets an
    #: unfolded design occupy the "2 folds" niche when the axis is restricted to 2-4.
    reject_out_of_bounds: bool = False
    #: require the bundle to still be converging when it LEAVES, on the one leg the
    #: convergence_gate above structurally cannot see: exit face -> detector. See
    #: Observation.exit_convergence -- leg widths are measured at the vertex that starts
    #: the leg, so the final leg contributes the width at the glass, and a final-leg ray
    #: crossing is excluded on purpose. A design that leaves narrow and fans out over the
    #: gap therefore passes clean_pack untouched and is only PRICED by fitness, never
    #: rejected, which is how ~44% of the ab40k elites came to be designs that put no
    #: image on the detector at all.
    #:
    #: Two conditions, both measured at every field (ns_probe/check_exit_convergence.py):
    #:     focus_dist > 0        the waist is ahead of the exit face. Negative means the
    #:                           bundle already passed focus inside the glass and is
    #:                           monotonically fanning out -- threshold-free, like the
    #:                           ray-order inversion test, and it needs no calibration.
    #:     width_sensor <= 2 * focus_w * exit_window_mult
    #:                           the bundle lands inside the window fitness scores it in.
    #:
    #: Priced on 95 ab40k elites + the 171-topology catalogue: keeps 50-57% of elites,
    #: 100% of the top ten in BOTH arms, best kept 0.878 either way -- i.e. it costs
    #: nothing at the top and clears out the half of the archive that never images.
    #: The ratio-flavoured versions of this test (sensor_ratio <= 0.8, focus_ratio in a
    #: band) are WRONG and were measured to be wrong: the good designs focus early, at
    #: 20-45 mm of an ~85 mm gap, and re-expand to a spot that still lands in the window,
    #: so a ratio to the exit width keeps 1 of 42 elites and discards everything above
    #: 0.639. That is the etendue-growth mistake in the gate_all_fields comment, again.
    require_exit_convergence: bool = False
    #: how much of the scoring window the bundle may fill at the detector. 1.0 = the whole
    #: window; 0.5 starts costing top designs (6 of the top 10), so it is not the default.
    exit_window_mult: float = 1.0
    label: str = ""                                # meta.experiment string
    #: descriptors measured for every design and written to info, beyond the two axes
    info_descriptors: Sequence[str] = ()
    #: per-cell key names for the two behaviour coordinates
    cell_bc_keys: Sequence[str] = ("tir_count", "T")
    #: extra key/values merged into archive.json meta
    meta_extra: dict = field(default_factory=dict)
    #: write trace.jsonl: one record per EVALUATION, including the emitter it came from,
    #: the emitter mean it was sampled around (the "parent") and the step that produced
    #: it. The archive keeps only winners; this keeps the whole search, which is what a
    #: what-worked/what-did-not dataset needs.
    trace: bool = False
    #: write optimizer_transitions.jsonl: one complete, replayable emitter generation
    #: per row. This is intentionally separate from the compact legacy trace.
    optimizer_transitions: bool = False
    #: warm start: load the cells of a previous archive.json into the archive before
    #: searching, so emitters restart from those elites. This is what makes a cascade of
    #: runs a ratchet rather than N independent searches.
    resume_from: Optional[str] = None
    #: Optional deterministic starting mean for a newly empty archive. Fixed local boxes
    #: use their midpoint; legacy experiments keep their original random initialization.
    initial_genotype: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if self.emitter not in ("cma", "evotf", "numeric_evotf", "optical_evotf"):
            raise ValueError(
                "emitter must be cma, evotf, numeric_evotf, or optical_evotf; "
                f"got {self.emitter!r}")
        if self.optical_categorical_mode not in ("learned", "one_hot"):
            raise ValueError("optical_categorical_mode must be 'learned' or 'one_hot'")
        if self.evotf_context_len <= 0:
            raise ValueError("evotf_context_len must be positive")
        if self.param_scaling not in ("isotropic", "normalized"):
            raise ValueError("param_scaling must be 'isotropic' or 'normalized', "
                             f"got {self.param_scaling!r}")
        if self.field_aggregate not in ("mean", "worst"):
            raise ValueError("field_aggregate must be 'mean' or 'worst', "
                             f"got {self.field_aggregate!r}")
        if self.convergence_gate not in ("packing", "clean_pack"):
            raise ValueError("convergence_gate must be 'packing' or 'clean_pack', "
                             f"got {self.convergence_gate!r}")
        for name, value in (
                ("min_intended_root_support", self.min_intended_root_support),
                ("min_intended_delivered_energy_share",
                 self.min_intended_delivered_energy_share)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
        has_intended_gate = bool(
            self.require_intended_dominant_all_fields
            or self.min_intended_root_support > 0.0
            or self.min_intended_delivered_energy_share > 0.0
            or self.min_intended_critical_margin_deg is not None
            or self.fitness == "intended_focused_throughput")
        if has_intended_gate and not self.intended_sequence:
            raise ValueError("intended-path fitness/gates require intended_sequence")
        if self.intended_sequence:
            parse_target(self.intended_sequence.replace("->", " "))
        fitness_registry.get(self.fitness)
        for d in tuple(self.descriptors) + tuple(self.info_descriptors):
            descriptor_registry.get(d)
        # an axis inherits the binning it was calibrated with unless overridden
        specs = [descriptor_registry.spec(d) for d in self.descriptors]
        if self.bc_bins is None:
            self.bc_bins = tuple(s[0] for s in specs)
        if self.bc_bounds is None:
            self.bc_bounds = tuple(s[1] for s in specs)
        if not self.label:
            self.label = self.name

    @property
    def with_robust(self) -> bool:
        """robustness is expensive, so it is only measured when it is an axis."""
        return any(descriptor_registry.get(d) is descriptor_registry.get("robustness")
                   for d in self.descriptors)


def _beam_across_fields(convs, fields) -> dict:
    """Worst case of the beam diagnostics over the WHOLE field sweep.

    The `beam_*` columns describe the central field, because the gate runs there and
    rejects 87-97% of proposals for one engine call instead of nine. But fitness is the
    worst field, so a design is scored on an edge field whose bundle nothing ever looked
    at -- and the edge fields are not a perturbation of the central one: on fold24_clean
    36 of 40 elites change dominant SIGNATURE across the field, i.e. the light takes a
    different path through the element. Measured that way, 40/40 elites are `clean_pack`
    on axis and only 16/40 are clean everywhere, with a median worst-field etendue growth
    of 5.0x against the 1.0x the central column reports.

    Every observation here has already been traced to score it, so this costs no engine
    calls -- only a second walk of the legs it already has.

    A field that catches nothing has no bundle to describe, so it cannot contribute a
    growth ratio and is skipped by the `worst` columns; `n_fields` says how many of the
    requested fields actually did. It is NOT skipped by `clean_all` / `clean_frac` --
    declaring a design clean everywhere on the strength of the five fields that happened
    to produce a bundle is the same mistake as declaring it clean on the strength of the
    central one, and 7 of the 9 partial-sweep fold24_clean elites would pass that way.
    """
    n_req = len(list(fields))
    pairs = [(f, c) for f, c in zip(fields, convs) if c and c.get("measured")]
    if not pairs:
        return dict(clean_all=None, clean_frac=None, worst_field=None, n_fields=0,
                    etendue_max_worst=None, spread_growth_worst=None,
                    ratio_worst=None, emittance_max_worst=None, crossings_worst=None)

    def worst(key):
        """Largest value over the sweep; every one of these keys is bad when big."""
        vs = [c[key] for _, c in pairs if c.get(key) is not None]
        return float(max(vs)) if vs else None

    clean = [bool(c["clean_pack"]) for _, c in pairs]
    # the field to open first: the least clean one, tie-broken by angular blow-up
    wf = max(pairs, key=lambda fc: (not fc[1]["clean_pack"],
                                    fc[1].get("spread_growth") or 0.0))[0]
    return dict(
        clean_all=bool(all(clean) and len(pairs) == n_req),
        clean_frac=float(sum(clean)) / n_req,
        worst_field=float(wf),
        n_fields=len(pairs),
        etendue_max_worst=worst("etendue_growth"),
        spread_growth_worst=worst("spread_growth"),
        ratio_worst=worst("ratio"),
        emittance_max_worst=worst("emittance_max"),
        crossings_worst=worst("n_crossings"),
    )


def _evaluate(runner, problem, cfg, x, pert_rng=None,
              diagnostics: Optional[dict] = None) -> Optional[dict]:
    """Genotype -> {fitness, bc, info}, or None if unusable.

    When supplied, ``diagnostics`` receives a stable failure ``stage`` and ``reason``.
    This is intentionally optional so every historical caller and numerical result keeps
    the original contract while new training traces retain why a proposal was refused.
    """
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics["engine_calls"] = 0

    def reject(stage: str, reason: str):
        if diagnostics is not None:
            engine_calls = int(diagnostics.get("engine_calls", 0))
            diagnostics.clear()
            diagnostics.update(stage=stage, reason=reason,
                               engine_calls=engine_calls)
        return None

    p = problem.vec_to_params(x)
    if not problem.is_valid(p):
        return reject("invalid_geometry", "problem geometry/clearance gate failed")

    fit_fn = fitness_registry.get(cfg.fitness)
    coherent = cfg.fitness in ("coherent_focused_throughput",
                               "intended_focused_throughput")
    face_of = problem.face_of(p)
    target = (parse_target(cfg.intended_sequence.replace("->", " "))
              if cfg.intended_sequence else ())
    intended_by_field = {}

    def observe(field_deg: float, *, allow_empty: bool):
        try:
            if diagnostics is not None:
                diagnostics["engine_calls"] += 1
            res, _ = runner.simulate(
                problem.layout(p, field_deg=field_deg).to_scene_dict())
        except Exception as exc:
            reject("engine_error", f"{type(exc).__name__}: {exc}")
            return None
        if target:
            intended_by_field[float(field_deg)] = target_path_evidence(
                res, face_of, target, problem.n_rays)
        return measure(res, face_of, x_sensor=problem.x_sensor,
                       dominant_only_centroid=coherent, allow_empty=allow_empty)

    def fitness_of(o):
        if o.is_empty:
            return 0.0
        if cfg.fitness == "intended_focused_throughput":
            return float(fit_fn(o, focus_w=problem.focus_w, target=target))
        return float(fit_fn(o, focus_w=problem.focus_w))

    def intended_field_failure(field_deg: float):
        if not target:
            return None
        evidence = intended_by_field[float(field_deg)]
        if evidence["root_support"] < cfg.min_intended_root_support:
            return ("intended_root_support",
                    f"field {field_deg:g}: target root support "
                    f"{evidence['root_support']:.3f} < "
                    f"{cfg.min_intended_root_support:.3f}")
        if (cfg.require_intended_dominant_all_fields
                and not evidence["target_is_dominant"]):
            return ("intended_not_dominant",
                    f"field {field_deg:g}: another delivered path dominates target")
        return None

    # measured once: the gate, the descriptor and the info block all read this
    rise_tol, env_tol, regrow_tol = cfg.convergence_tol

    def convergence_of(o):
        return o.convergence(rise_tol=rise_tol, envelope_tol=env_tol,
                             regrow_tol=regrow_tol, n_glass=p.get("n"))

    # the scoring window is +-focus_w about the centroid, so it is 2*focus_w wide; the
    # bundle is allowed to fill exit_window_mult of it. Expressed against focus_w rather
    # than in mm so the gate follows a problem with a different detector.
    exit_w_max = 2.0 * problem.focus_w * cfg.exit_window_mult

    def exit_ok(e) -> bool:
        return bool(e["measured"] and e["focus_dist"] is not None
                    and e["focus_dist"] > 0.0
                    and e["width_sensor"] is not None
                    and e["width_sensor"] <= exit_w_max)

    def gate_failure(c, e=None):
        if cfg.require_monotone_convergence and not c[cfg.convergence_gate]:
            return ("beam_gate", f"central field fails {cfg.convergence_gate}")
        if cfg.require_exit_convergence and e is not None and not exit_ok(e):
            return ("exit_gate", "central field fails exit convergence")
        return None

    if cfg.fields:
        # a field that catches nothing scores 0; only an engine error is fatal, and
        # only the CENTRAL field must carry a dominant signature (it defines the bc)
        fields = list(cfg.fields)
        i0 = fields.index(0.0)
        central = observe(0.0, allow_empty=True)
        if central is None:
            if diagnostics and diagnostics.get("stage") == "engine_error":
                return None
            return reject("engine_error", "central trace could not be measured")
        if central.is_empty:
            return reject("no_light", "central field has no forward delivered bundle")
        failure = intended_field_failure(0.0)
        if failure:
            return reject(*failure)
        # The gate reads the CENTRAL field only, and it rejects 87-97% of everything the
        # emitter proposes here. Applying it before the other eight fields are traced
        # costs a rejected design one engine call instead of nine. The accepted set is
        # identical either way -- a design the gate refuses never reached the archive.
        conv = convergence_of(central)
        exitc = central.exit_convergence()
        failure = gate_failure(conv, exitc)
        if failure:
            return reject(*failure)
        obs_list = []
        for i, f in enumerate(fields):
            o = central if i == i0 else observe(f, allow_empty=True)
            if o is None:
                if diagnostics and diagnostics.get("stage") == "engine_error":
                    return None
                return reject("engine_error", f"field {f:g} could not be measured")
            obs_list.append(o)
        for field_deg in fields:
            failure = intended_field_failure(field_deg)
            if failure:
                return reject(*failure)
        vals = [fitness_of(o) for o in obs_list]
        fitness = float(min(vals) if cfg.field_aggregate == "worst" else np.mean(vals))
        obs = central
        # `conv` (central) stays the gate and the beam_convergence descriptor; the sweep
        # is what gets REPORTED, so the archive can be filtered on how the design behaves
        # at the fields that actually set its fitness. Central is reused, not re-walked.
        conv_sweep = [conv if i == i0 else convergence_of(o)
                      for i, o in enumerate(obs_list)]
        exit_sweep = [exitc if i == i0 else o.exit_convergence()
                      for i, o in enumerate(obs_list)]
        extra = dict(mean_focthru=float(np.mean(vals)), worst_focthru=float(min(vals)),
                     per_field=[round(float(v), 4) for v in vals],
                     central_spot=float(central.spot))
    else:
        obs = observe(0.0, allow_empty=False)
        if obs is None:
            if diagnostics and diagnostics.get("stage") == "engine_error":
                return None
            return reject("engine_error", "single-field trace could not be measured")
        failure = intended_field_failure(0.0)
        if failure:
            return reject(*failure)
        conv = convergence_of(obs)
        exitc = obs.exit_convergence()
        failure = gate_failure(conv, exitc)
        if failure:
            return reject(*failure)
        fitness = fitness_of(obs)
        fields = [0.0]
        obs_list = [obs]
        conv_sweep = [conv]      # one field: the sweep IS the central field
        exit_sweep = [exitc]
        extra = {}

    intended_info = {}
    if target:
        evidence = [intended_by_field[float(field)] for field in fields]
        energy_shares = [float(item["delivered_energy_share"]) for item in evidence]
        energy_mean = float(np.mean(energy_shares))
        if energy_mean < cfg.min_intended_delivered_energy_share:
            return reject(
                "intended_low_energy",
                f"mean target delivered energy {energy_mean:.4f} < "
                f"{cfg.min_intended_delivered_energy_share:.4f}")
        margins = [target_critical_margin(o, target, p["n"]) for o in obs_list]
        measured_margins = [float(value) for value in margins if value is not None]
        margin_min = min(measured_margins) if measured_margins else None
        if cfg.min_intended_critical_margin_deg is not None and (
                len(measured_margins) != len(fields)
                or margin_min < cfg.min_intended_critical_margin_deg):
            return reject(
                "intended_tir_margin",
                "target TIR margin missing at a field" if margin_min is None else
                f"worst target TIR margin {margin_min:.3f} deg < "
                f"{cfg.min_intended_critical_margin_deg:.3f} deg")
        intended_info = dict(
            intended_sequence=format_signature(target),
            intended_root_support_min=float(min(
                item["root_support"] for item in evidence)),
            intended_root_support_mean=float(np.mean([
                item["root_support"] for item in evidence])),
            intended_energy_share_mean=energy_mean,
            intended_energy_share_min=float(min(energy_shares)),
            intended_dominant_all=bool(all(
                item["target_is_dominant"] for item in evidence)),
            intended_critical_margin_min=margin_min,
            intended_per_field=[{
                "field_deg": float(field),
                "root_support": round(float(item["root_support"]), 6),
                "delivered_energy_share": round(
                    float(item["delivered_energy_share"]), 6),
                "dominant": bool(item["target_is_dominant"]),
                "critical_margin_deg": (
                    None if margin is None else round(float(margin), 6)),
            } for field, item, margin in zip(fields, evidence, margins)],
        )

    # Beam behaviour over the whole sweep, not just the central field. Everything reaching
    # here has already been traced at every field to be scored, so this rejects for free.
    sweep = _beam_across_fields(conv_sweep, cfg.fields or (0.0,))
    if cfg.gate_all_fields and not sweep["clean_all"]:
        return reject("beam_gate_all_fields",
                      f"at least one field fails {cfg.convergence_gate}")

    # The exit leg over the whole sweep. Measured on 95 ab40k elites, gating this at the
    # central field alone would be worth little: the designs that fail it usually fail it
    # at EVERY field, but the ones that fail it at only some (5/9, 6/9) are exactly the
    # ones a central-field test would wave through -- the same asymmetry _beam_across_fields
    # exists for. A field with no bundle cannot be declared converging.
    n_req_e = len(list(cfg.fields or (0.0,)))
    live_e = [e for e in exit_sweep if e and e["measured"]]
    exit_flags = [exit_ok(e) for e in exit_sweep if e]
    exit_all = bool(exit_flags and all(exit_flags) and len(live_e) == n_req_e)
    if cfg.require_exit_convergence and cfg.gate_all_fields and not exit_all:
        return reject("exit_gate_all_fields",
                      "at least one field fails exit convergence")
    exit_info = dict(
        exit_ok_all=exit_all,
        exit_ok_frac=(float(sum(exit_flags)) / n_req_e if n_req_e else None),
        exit_fields=len(live_e),
        exit_width_sensor_worst=(float(max(e["width_sensor"] for e in live_e))
                                 if live_e else None),
        exit_focus_dist=exitc["focus_dist"], exit_sensor_dist=exitc["sensor_dist"],
        exit_focus_ratio=exitc["focus_ratio"], exit_width_sensor=exitc["width_sensor"],
        exit_rms_sensor=exitc["rms_sensor"], exit_converges=exitc["converges"],
    )

    # Ray-order inversion and etendue growth are SHARPER caustic tests than the envelope
    # (measured on fold24_wide: gating on "no crossing AND etendue <= 2x" lifts the median
    # kept fitness 0.233 -> 0.314). They are recorded rather than gated on, because the
    # archive can be filtered after the fact for free whereas a tighter gate costs search
    # budget -- and rejection already runs 84-96% here.
    # conv already walked the legs and ran the crossing test; re-deriving them here would
    # trace the bundle twice per evaluation, which at 40k evals is not free.
    legs = obs.beam_legs(n_glass=p["n"])
    et = [lg["etendue"] for lg in legs[1:]] if len(legs) > 1 else []
    beam = dict(
        etendue_ratio=(float(et[-1] / et[0]) if et and et[0] > 1e-9 else None),
        etendue_max=(float(max(et) / et[0]) if et and et[0] > 1e-9 else None),
        spread_out_deg=(float(legs[-1]["spread_deg"]) if legs else None),
    )

    ctx = descriptor_registry.DescriptorContext(
        obs=obs, problem=problem, params=p, genotype=np.asarray(x, float),
        fitness=float(fitness), runner=runner, pert_rng=pert_rng,
        with_robust=cfg.with_robust,
        fields=tuple(cfg.fields or ()), per_field=tuple(extra.get("per_field", ())),
        convergence=conv)

    # measured once, reused for both the axes and the info block
    measured = {}
    for d in tuple(cfg.descriptors) + tuple(cfg.info_descriptors):
        fn = descriptor_registry.get(d)
        if fn.descriptor_name not in measured:
            measured[fn.descriptor_name] = float(fn(ctx))

    bc = tuple(measured[descriptor_registry.get(d).descriptor_name]
               for d in cfg.descriptors)
    if cfg.reject_out_of_bounds and any(
            not (blo <= v <= bhi) for v, (blo, bhi) in zip(bc, cfg.bc_bounds)):
        return reject("descriptor_out_of_bounds",
                      f"behaviour coordinates {bc} outside {cfg.bc_bounds}")
    info = dict(
        fitness=float(fitness),
        efficiency=float(obs.delivered) / obs.emitted,
        spot=float(obs.spot),
        tir_count=int(obs.tir_count),
        thinness_T=float(obs.thinness(problem.x_sensor)),
        dom_energy_share=float(obs.dominant_energy / obs.emitted),
        dom_signature=format_signature(obs.dominant_signature),
        centroid=float(obs.centroid),
        n_forward=len(obs.records),
        n_dom=obs.n_dominant,
        beam_ratio=conv["ratio"], beam_monotone=conv["monotone"],
        beam_packing=conv["packing"], beam_envelope=conv["envelope"],
        beam_regrow=conv["regrow"], beam_waist=conv["waist"],
        # angular: what the width-only columns above cannot see
        beam_clean_pack=conv["clean_pack"], beam_crossings=conv["n_crossings"],
        beam_entry_spread=conv["entry_spread"], beam_exit_spread=conv["exit_spread"],
        beam_waist_spread=conv["waist_spread"], beam_spread_growth=conv["spread_growth"],
        beam_emittance_exit=conv["emittance_exit"],
        beam_etendue_ratio=beam["etendue_ratio"],
        beam_etendue_max=beam["etendue_max"], beam_spread_out=beam["spread_out_deg"],
        # every beam_ column above is the CENTRAL field; these are the whole sweep, and
        # they are the ones to filter an archive on -- see _beam_across_fields
        beam_clean_all=sweep["clean_all"], beam_clean_frac=sweep["clean_frac"],
        beam_worst_field=sweep["worst_field"], beam_fields=sweep["n_fields"],
        beam_etendue_max_worst=sweep["etendue_max_worst"],
        beam_spread_growth_worst=sweep["spread_growth_worst"],
        beam_ratio_worst=sweep["ratio_worst"],
        beam_emittance_max_worst=sweep["emittance_max_worst"],
        beam_crossings_worst=sweep["crossings_worst"],
        max_sag_ratio=float(problem.element(p).max_sag_ratio()),
        **intended_info,
        **exit_info,
    )
    # extra descriptors land in info under their own names; tir_bounces and thinness
    # are already reported as tir_count / thinness_T
    info.update({k: v for k, v in measured.items()
                 if k not in ("tir_bounces", "thinness")})
    info.update(extra)
    return dict(fitness=float(fitness), bc=bc, info=info)


def _trace(fh, n, emitter_idx, em, parent, x, ev, status, old,
           diagnostics=None) -> None:
    """One evaluation, as a (state, action, outcome) triple.

    parent = the emitter mean the sample was drawn around, x = the proposal, so
    x - parent IS the action CMA-ES chose. `status` is the archive's verdict on it, which
    is the label: new > improved > none > invalid. That is the whole supervision signal
    the search itself runs on, so it is the natural signal to train anything else on.
    """
    if fh is None:
        return
    rec = dict(eval=int(n), emitter=int(emitter_idx), gen=int(em.gen),
               sigma=float(em.sigma), status=status,
               parent=[round(float(v), 6) for v in parent],
               step=[round(float(a - b), 6) for a, b in zip(x, parent)],
               genotype=[round(float(v), 6) for v in x])
    if diagnostics is not None and "engine_calls" in diagnostics:
        rec["engine_calls"] = int(diagnostics["engine_calls"])
    if ev is None and diagnostics:
        rec["rejection"] = dict(diagnostics)
    if ev is not None:
        rec.update(fitness=round(float(ev["fitness"]), 6),
                   bc=[round(float(v), 4) for v in ev["bc"]],
                   delta=(round(float(ev["fitness"] - old), 6)
                          if status == "improved" and old is not None else None),
                   info={k: ev["info"][k] for k in
                         ("tir_count", "spot", "efficiency", "dom_signature",
                          "thinness_T", "per_field", "worst_focthru",
                          "intended_sequence", "intended_root_support_min",
                          "intended_energy_share_mean", "intended_dominant_all",
                          "intended_critical_margin_min", "intended_per_field")
                         if k in ev["info"]})
    fh.write(json.dumps(rec) + "\n")


def _archive_summary(archive: GridArchive) -> dict:
    fitnesses = [float(cell.fitness) for cell in archive.cells.values()]
    return {
        "filled": len(archive.cells),
        "coverage": float(archive.coverage),
        "qd_score": float(sum(fitnesses)),
        "best_fitness": max(fitnesses) if fitnesses else None,
    }


def _parent_archive_context(archive: GridArchive, parent: np.ndarray) -> dict:
    for index, cell in archive.cells.items():
        if np.array_equal(np.asarray(cell.genotype), np.asarray(parent)):
            return {
                "cell": [int(value) for value in index],
                "fitness": float(cell.fitness),
                "behavior": [float(value) for value in cell.bc],
                "route_family": cell.info.get("dom_signature"),
            }
    return {"cell": None, "fitness": None, "behavior": None, "route_family": None}


def _candidate_observation(
    *, candidate_index, search_x, genotype, clipping_mask, status, key, ev, old,
    diagnostics, archive_coverage, archive_filled, parent_context,
    proposed_index=None, previous_cell=None,
) -> dict:
    """The evaluated information available before the distribution update."""
    from ..search.optimizer_transition import json_value, route_from_info

    valid = ev is not None
    info = (ev or {}).get("info") or {}
    improvement = None
    if status == "improved" and old is not None:
        improvement = (float(ev["fitness"]) - float(old)) / max(abs(float(old)), 1e-12)
    within_tier = (
        float(ev["fitness"] - old) if status == "improved" and old is not None else
        float(ev["fitness"]) if valid else None
    )
    previous_fitness = (float(previous_cell.fitness)
                        if previous_cell is not None else None)
    return {
        "candidate_index": int(candidate_index),
        "search_coordinates": [float(value) for value in search_x],
        "genotype": [float(value) for value in genotype],
        "clipping_mask": np.asarray(clipping_mask, dtype=int).tolist(),
        "rank": -1,
        "rank_percentile": None,
        "status": status,
        "ranking_key": float(key),
        "within_tier_value": within_tier,
        "fitness": float(ev["fitness"]) if valid else None,
        "archive_improvement": improvement,
        "behavior": ([float(value) for value in ev["bc"]] if valid else None),
        "failure": (None if valid else {
            "stage": str(diagnostics.get("stage") or "invalid"),
            "reason": str(diagnostics.get("reason") or "evaluation returned no result"),
        }),
        "optical_metrics": json_value(info),
        "constraint_margins": {
            key: json_value(info.get(key)) for key in (
                "crit_margin", "intended_critical_margin_min", "max_sag_ratio",
                "beam_ratio", "beam_ratio_worst", "beam_etendue_max_worst")
            if info.get(key) is not None
        },
        "route": route_from_info(info),
        "archive_context": {
            "coverage": float(archive_coverage),
            "filled": int(archive_filled),
            "proposed_cell": ([int(value) for value in proposed_index]
                              if proposed_index is not None else None),
            "cell_was_empty": previous_cell is None if proposed_index is not None else None,
            "previous_elite_fitness": previous_fitness,
            "parent": parent_context,
        },
    }


def _write_optimizer_transition(
    fh, *, cfg, problem, emitter, emitter_index, state_before, state_after,
    sampling, candidates, ranking, archive_before, archive_after, remaining_budget,
    restart_triggered, restart_reason, fingerprints, context,
) -> None:
    from ..rl.data.schema import (OPTIMIZER_TRANSITION_SCHEMA,
                                  validate_optimizer_transition)
    from ..search.optimizer_transition import canonical_hash, json_value

    generation = int(state_before["generation"])
    transition_id = canonical_hash({
        "lifecycle": emitter.lifecycle_id,
        "generation": generation,
        "sample_id": sampling["sample_id"],
        "mean": state_before["mean"],
    })[:32]
    ranked_candidates = [candidates[int(index)] for index in ranking]
    route_family = next(
        (candidate["route"]["signature"] for candidate in ranked_candidates
         if candidate["route"]["signature"]),
        "invalid",
    )
    record = {
        "schema_version": OPTIMIZER_TRANSITION_SCHEMA,
        "transition_id": transition_id,
        "task": cfg.name,
        "representation": context["representation"]["id"],
        "fingerprints": fingerprints,
        "context": context,
        "emitter_algorithm": "full_covariance_cma_es",
        "emitter_index": int(emitter_index),
        "lifecycle_id": emitter.lifecycle_id,
        "generation": generation,
        "seed": int(emitter.lifecycle_seed),
        "search_seed": int(cfg.seed),
        "remaining_budget": int(remaining_budget),
        "parameter_names": list(problem.param_names),
        # State and samples are recorded in the emitter's coordinate system.  The
        # physical bounds remain available in context.representation.bounds.
        "bounds": ([[0.0, 1.0] for _ in problem.param_names]
                   if cfg.param_scaling == "normalized" else
                   [[float(a), float(b)] for a, b in zip(problem.lo, problem.hi)]),
        "state_before": state_before,
        "sampling": sampling,
        "candidates": candidates,
        "ranking_order": [int(index) for index in ranking],
        "state_after": state_after,
        "archive_before": archive_before,
        "archive_after": archive_after,
        "route_family": route_family,
        "restart": {"triggered": bool(restart_triggered), "reason": restart_reason},
    }
    record = json_value(record)
    errors = validate_optimizer_transition(record)
    if errors:
        raise ValueError(f"invalid optimizer transition {transition_id}: {errors}")
    fh.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def run(cfg: ExperimentConfig, out_dir: Path, *, log=print) -> dict:
    """Run one experiment and write archive.json. Returns the payload."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = cfg.problem
    lo, hi = np.asarray(problem.lo, float), np.asarray(problem.hi, float)
    if cfg.optimizer_transitions and cfg.emitter != "cma":
        raise ValueError("optimizer transition recording currently requires emitter='cma'")
    if cfg.emitter in ("numeric_evotf", "optical_evotf") \
            and not cfg.evotf_checkpoint:
        raise ValueError(f"emitter={cfg.emitter!r} requires evotf_checkpoint")

    archive = GridArchive(tuple(cfg.bc_bins), tuple(tuple(b) for b in cfg.bc_bounds))
    n_resumed = 0
    seed_cells = []
    if cfg.resume_from:
        prev = json.loads((Path(cfg.resume_from) / "archive.json").read_text())
        k0_prev, k1_prev = cfg.cell_bc_keys
        seed_cells = [(np.asarray(c["genotype"], float), float(c["fitness"]),
                       (c[k0_prev], c[k1_prev]),
                       {k: v for k, v in c.items()
                        if k not in ("i0", "i1", "genotype", "fitness")})
                      for c in prev["cells"]]
    rng = np.random.default_rng(cfg.seed)
    # separate stream so that turning the robustness descriptor on does not perturb the
    # search RNG and silently move every other experiment's archive
    pert_rng = np.random.default_rng(cfg.seed + 9999)
    scale = hi - lo
    t0 = time.time()
    n_engine_calls = 0
    n_seed_engine_calls = 0

    # Aggregate lifecycle telemetry as emitters restart. Keeping every complete state
    # summary would make long sparse-archive runs produce enormous archive metadata.
    emitter_telemetry = dict(
        lifecycles=0, generations=0, ask_calls=0, tell_calls=0,
        ask_seconds=0.0, inference_seconds=0.0,
        clipped_candidates=0, clipped_values=0,
        clipped_by_gene=[0 for _ in problem.param_names],
        fallback_events=0, fallback_candidates=0, nonfinite_events=0,
        restart_events=0, numerical_failure_lifecycles=0, restart_reasons={},
    )

    def finish_emitter(em) -> None:
        if cfg.emitter == "cma":
            return
        summary = em.state_summary()
        # The historical loop constructs a replacement even when the budget was just
        # exhausted. It never asks that object, so it is not an emitter lifecycle.
        if int(summary["ask_calls"]) == 0:
            return
        emitter_telemetry["lifecycles"] += 1
        for key in ("generation", "ask_calls", "tell_calls", "clipped_candidates",
                    "clipped_values", "fallback_events", "fallback_candidates",
                    "nonfinite_events", "restart_events"):
            target = "generations" if key == "generation" else key
            emitter_telemetry[target] += int(summary[key])
        for key in ("ask_seconds", "inference_seconds"):
            emitter_telemetry[key] += float(summary[key])
        emitter_telemetry["clipped_by_gene"] = [
            int(a) + int(b) for a, b in zip(
                emitter_telemetry["clipped_by_gene"], summary["clipped_by_gene"])
        ]
        if summary["numerical_failure"] is not None:
            emitter_telemetry["numerical_failure_lifecycles"] += 1
        reason = summary["restart_reason"] or "run_complete"
        emitter_telemetry["restart_reasons"][reason] = (
            emitter_telemetry["restart_reasons"].get(reason, 0) + 1
        )

    normalized = cfg.param_scaling == "normalized"
    transition_f = None
    transition_fingerprints = None
    transition_context = None
    lifecycle_serial = [0 for _ in range(cfg.emitters)]
    if cfg.optimizer_transitions:
        from ..search.optimizer_transition import optimizer_fingerprints
        transition_fingerprints, transition_context = optimizer_fingerprints(cfg, problem)
        transition_f = open(out_dir / "optimizer_transitions.jsonl", "w")
    optical_environment_context = None
    if cfg.emitter in ("numeric_evotf", "optical_evotf"):
        from ..search.optimizer_transition import optimizer_fingerprints
        optical_fingerprints, optical_context = optimizer_fingerprints(cfg, problem)
        optical_environment_context = {
            **optical_context, "fingerprints": optical_fingerprints}
    initial = (None if cfg.initial_genotype is None
               else np.asarray(cfg.initial_genotype, dtype=float))
    if initial is not None:
        if initial.shape != lo.shape:
            raise ValueError(
                f"initial_genotype has shape {initial.shape}, expected {lo.shape}")
        if np.any(initial < lo) or np.any(initial > hi):
            raise ValueError("initial_genotype lies outside the problem bounds")
    # the RNG is drawn identically in both arms -- only the space the emitter lives in
    # differs -- so switching arms cannot shift the isotropic archives
    def new_emitter(s, emitter_index):
        x0 = archive.random_elite(rng)
        if x0 is None:
            x0 = initial.copy() if initial is not None else rng.uniform(lo, hi)
        x0 = np.asarray(x0, float)
        if normalized:
            emitter_x0 = (x0 - lo) / scale
            emitter_sigma0 = cfg.sigma0
            emitter_bounds = (np.zeros_like(lo), np.ones_like(hi))
        else:
            emitter_x0 = x0
            emitter_sigma0 = cfg.sigma0 * float(np.mean(scale))
            emitter_bounds = (lo, hi)
        if cfg.emitter == "cma":
            emitter = CMAES(emitter_x0, emitter_sigma0, popsize=cfg.popsize, seed=s,
                            bounds=emitter_bounds)
        elif cfg.emitter == "evotf":
            # Lazy construction preserves the repository's dependency-free CMA path.
            from ..search.evotf_emitter import EvoTFEmitter
            emitter = EvoTFEmitter(
                emitter_x0, emitter_sigma0, popsize=cfg.popsize, seed=s,
                bounds=emitter_bounds, max_context_len=cfg.evotf_context_len)
        else:
            from ..learning.optical_evotf.emitter import OpticalEvoTFEmitter
            enabled = (() if cfg.emitter == "numeric_evotf"
                       else tuple(cfg.optical_adapters))
            emitter = OpticalEvoTFEmitter(
                emitter_x0, emitter_sigma0, popsize=cfg.popsize, seed=s,
                bounds=emitter_bounds, max_context_len=cfg.evotf_context_len,
                parameter_names=problem.param_names,
                environment_context=optical_environment_context,
                checkpoint_path=(Path(cfg.evotf_checkpoint)
                                 if cfg.evotf_checkpoint else None),
                enabled_adapters=enabled,
                categorical_mode=cfg.optical_categorical_mode)
        emitter.lifecycle_id = (
            f"{cfg.name}:seed{cfg.seed}:emitter{emitter_index}:"
            f"lifecycle{lifecycle_serial[emitter_index]}"
        )
        lifecycle_serial[emitter_index] += 1
        emitter.lifecycle_seed = int(s)
        return emitter

    # A Problem may own its measurement model (the AR/VR relay does); everything else
    # -- loop, RNG protocol, archive, ranking keys, payload -- stays shared.
    if hasattr(problem, "evaluate"):
        evaluate_fn = lambda r, pr, c, xx, pert_rng=None: problem.evaluate(  # noqa: E731
            r, c, xx, pert_rng=pert_rng)
    else:
        evaluate_fn = _evaluate

    if seed_cells:
        # resume_from inserts prior elites VERBATIM -- no evaluation, therefore no gate.
        # Every fold24 run warm-starts (a 14.7% gate cannot cold-start), so a gate that
        # skips this path is bypassable in exactly the configuration everyone uses: the
        # seed was built before the exit test existed, and its elites would sit in the
        # archive untouched, holding the very cells the gate exists to clear. Measured on
        # the fold24_common_seed: 2 of its 7 cells fail the exit gate.
        #
        # Re-scoring is opt-in, and deliberately tied to require_exit_convergence rather
        # than done always: every frozen archive was built by loading its seed unchecked,
        # and re-scoring by default would move results that tests/test_qd_equivalence.py
        # compares cell-for-cell. It costs n_cells * len(fields) engine calls -- well
        # under a second -- so there is no reason to skip it when the gate is on.
        if cfg.require_exit_convergence or cfg.intended_sequence:
            kept = []
            with Runner(timeout_s=20) as seed_runner:
                for g, fit, bc_, info in seed_cells:
                    diagnostics = {}
                    if evaluate_fn is _evaluate:
                        res = evaluate_fn(seed_runner, problem, cfg, g, pert_rng,
                                          diagnostics=diagnostics)
                        calls = int(diagnostics.get("engine_calls", 0))
                        n_engine_calls += calls
                        n_seed_engine_calls += calls
                    else:
                        res = evaluate_fn(seed_runner, problem, cfg, g, pert_rng)
                    if res is not None:
                        kept.append((g, float(res["fitness"]), bc_, res["info"]))
            n_dropped = len(seed_cells) - len(kept)
            seed_cells = kept
            if n_dropped:
                log(f"[{cfg.name}] resume: dropped {n_dropped} seed cells that fail "
                    f"the run's own gates")
        for g, fit, bc_, info in seed_cells:
            archive.add(g, fit, bc_, info=info)
        n_resumed = len(archive.cells)
        log(f"[{cfg.name}] resumed {n_resumed} cells from {cfg.resume_from}")

    n = 0
    n_invalid = 0
    history: List[dict] = []
    trace_f = open(out_dir / "trace.jsonl", "w") if cfg.trace else None
    with Runner(timeout_s=20) as runner:
        emitters = [new_emitter(cfg.seed + i, i) for i in range(cfg.emitters)]
        while n < cfg.evals:
            for ei in range(cfg.emitters):
                if n >= cfg.evals:
                    break
                em = emitters[ei]
                n_before_generation = n
                state_before = (em.state_snapshot()
                                if transition_f is not None else None)
                archive_before = (
                    _archive_summary(archive) if transition_f is not None else None)
                search_X = em.ask()
                sampling = (em.sampling_snapshot()
                            if transition_f is not None else None)
                # the emitter searches the unit cube under "normalized"; the evaluator
                # and the archive always speak world units
                parent = lo + em.mean * scale if normalized else em.mean.copy()
                X = lo + search_X * scale if normalized else search_X
                keys = np.empty(len(X))
                added = False
                observations = []
                parent_context = _parent_archive_context(archive, parent)
                for k, xx in enumerate(X):
                    if cfg.lock_tilt:
                        xx = lock_tilt(xx)
                    diagnostics = {}
                    if evaluate_fn is _evaluate:
                        ev = evaluate_fn(runner, problem, cfg, xx,
                                         pert_rng=pert_rng,
                                         diagnostics=diagnostics)
                    else:
                        ev = evaluate_fn(runner, problem, cfg, xx,
                                         pert_rng=pert_rng)
                    n += 1
                    n_engine_calls += int(diagnostics.get("engine_calls", 0))
                    candidate_archive_coverage = archive.coverage
                    candidate_archive_filled = len(archive.cells)
                    if ev is None:
                        keys[k] = -1e9
                        n_invalid += 1
                        observations.append(_candidate_observation(
                            candidate_index=k, search_x=search_X[k], genotype=xx,
                            clipping_mask=em.last_clipping_mask[k], status="invalid",
                            key=keys[k], ev=None, old=None, diagnostics=diagnostics,
                            archive_coverage=candidate_archive_coverage,
                            archive_filled=candidate_archive_filled,
                            parent_context=parent_context))
                        _trace(trace_f, n, ei, em, parent, xx, None, "invalid", None,
                               diagnostics)
                        continue
                    proposed_index = archive.index(ev["bc"])
                    previous_cell = archive.cells.get(proposed_index)
                    st, old = archive.add(xx, ev["fitness"], ev["bc"], info=ev["info"])
                    if st == "new":
                        keys[k] = 2e12 + ev["fitness"]; added = True
                    elif st == "improved":
                        keys[k] = 1e12 + (ev["fitness"] - (old or 0.0)); added = True
                    else:
                        keys[k] = ev["fitness"]
                    observations.append(_candidate_observation(
                        candidate_index=k, search_x=search_X[k], genotype=xx,
                        clipping_mask=em.last_clipping_mask[k], status=st,
                        key=keys[k], ev=ev, old=old, diagnostics=diagnostics,
                        archive_coverage=candidate_archive_coverage,
                        archive_filled=candidate_archive_filled,
                        parent_context=parent_context,
                        proposed_index=proposed_index, previous_cell=previous_cell))
                    _trace(trace_f, n, ei, em, parent, xx, ev, st, old,
                           diagnostics)
                order = np.argsort(-keys)
                for rank, candidate_index in enumerate(order):
                    observations[int(candidate_index)]["rank"] = int(rank)
                    observations[int(candidate_index)]["rank_percentile"] = (
                        float(rank) / max(len(order) - 1, 1))
                em.tell(order, observations)
                restart_triggered = (not added) or em.restart_needed
                restart_reason = None
                if restart_triggered:
                    if not added and em.restart_needed:
                        restart_reason = "no_archive_addition+numerical_failure"
                    elif em.restart_needed:
                        restart_reason = "numerical_failure"
                    else:
                        restart_reason = "no_archive_addition"
                    if hasattr(em, "note_restart"):
                        em.note_restart(restart_reason)
                if transition_f is not None:
                    _write_optimizer_transition(
                        transition_f, cfg=cfg, problem=problem, emitter=em,
                        emitter_index=ei, state_before=state_before,
                        state_after=em.state_snapshot(), sampling=sampling,
                        candidates=observations, ranking=order,
                        archive_before=archive_before,
                        archive_after=_archive_summary(archive),
                        remaining_budget=max(cfg.evals - n_before_generation, 0),
                        restart_triggered=restart_triggered,
                        restart_reason=restart_reason,
                        fingerprints=transition_fingerprints,
                        context=transition_context)
                if restart_triggered:
                    finish_emitter(em)
                    emitters[ei] = new_emitter(
                        int(rng.integers(0, 1 << 30)), ei)
            if n // 1000 > len(history):
                rec = dict(evals=n, filled=len(archive.cells),
                           coverage=archive.coverage, best=archive.best_fitness,
                           invalid=n_invalid)
                history.append(rec)
                log(f"[{cfg.name}] evals={n:6d} filled={rec['filled']:3d} "
                    f"cov={rec['coverage']*100:5.1f}% best={rec['best']:.3f} "
                    f"invalid={n_invalid} ({time.time()-t0:6.1f}s)")
    if trace_f is not None:
        trace_f.close()
        log(f"[{cfg.name}] wrote {out_dir/'trace.jsonl'} ({n} evaluations)")
    if transition_f is not None:
        transition_f.close()
        log(f"[{cfg.name}] wrote {out_dir/'optimizer_transitions.jsonl'}")

    for em in emitters:
        finish_emitter(em)

    k0, k1 = cfg.cell_bc_keys
    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = {"i0": int(i0), "i1": int(i1),
               k0: float(c.bc[0]), k1: float(c.bc[1]),
               "fitness": float(c.fitness),
               "genotype": [float(v) for v in c.genotype]}
        rec.update(c.info)
        cells.append(rec)

    if cfg.emitter != "cma":
        proposals = emitter_telemetry["ask_calls"] * cfg.popsize
        emitter_telemetry["clipped_candidate_rate"] = (
            float(emitter_telemetry["clipped_candidates"]) / proposals
            if proposals else 0.0)
        emitter_telemetry["clipped_value_rate"] = (
            float(emitter_telemetry["clipped_values"]) / (proposals * len(lo))
            if proposals else 0.0)

    meta = dict(experiment=cfg.label, evals=int(n), seed=int(cfg.seed),
                invalid=int(n_invalid), bc_bins=list(cfg.bc_bins),
                bc_bounds=[list(b) for b in cfg.bc_bounds],
                bc_axes=[descriptor_registry.axis_label(d) for d in cfg.descriptors],
                fitness=fitness_registry.META_DESCRIPTION.get(cfg.fitness, cfg.fitness),
                focus_w=problem.focus_w,
                emitter=cfg.emitter,
                emitter_count=int(cfg.emitters), popsize=int(cfg.popsize),
                optimizer_sigma0=float(cfg.sigma0),
                # Compatibility with consumers of existing CMA archive metadata.
                cma_sigma0=float(cfg.sigma0), param_scaling=cfg.param_scaling,
                evotf_context_len=(int(cfg.evotf_context_len)
                                   if cfg.emitter != "cma" else None),
                emitter_telemetry=(emitter_telemetry
                                   if cfg.emitter != "cma" else None),
                evotf_checkpoint=cfg.evotf_checkpoint,
                optical_adapters=(list(cfg.optical_adapters)
                                  if cfg.emitter == "optical_evotf" else []),
                optical_categorical_mode=(cfg.optical_categorical_mode
                                          if cfg.emitter == "optical_evotf" else None),
                requested_candidate_budget=int(cfg.evals),
                engine_calls=int(n_engine_calls),
                seed_engine_calls=int(n_seed_engine_calls),
                optimizer_transition_schema=(
                    "nslens.optimizer_transition.v1"
                    if cfg.optimizer_transitions else None),
                field_aggregate=cfg.field_aggregate, fields=list(cfg.fields or ()),
                intended_sequence=cfg.intended_sequence or None,
                intended_path_gates=(dict(
                    min_root_support=float(cfg.min_intended_root_support),
                    min_mean_delivered_energy_share=float(
                        cfg.min_intended_delivered_energy_share),
                    require_dominant_all_fields=bool(
                        cfg.require_intended_dominant_all_fields),
                    min_critical_margin_deg=cfg.min_intended_critical_margin_deg,
                ) if cfg.intended_sequence else None),
                convergence_gate=(cfg.convergence_gate
                                  if cfg.require_monotone_convergence else None),
                resumed_cells=int(n_resumed), resume_from=cfg.resume_from,
                initial_genotype=(None if initial is None
                                  else [float(value) for value in initial]),
                param_names=list(problem.param_names),
                wall_seconds=float(time.time() - t0))
    meta.update(cfg.meta_extra)
    payload = dict(meta=meta, history=history, cells=cells)
    (out_dir / "archive.json").write_text(json.dumps(payload, indent=2))
    log(f"[{cfg.name}] wrote {out_dir/'archive.json'} ({len(cells)} cells, {n} evals, "
        f"{payload['meta']['wall_seconds']:.0f}s)")
    return payload
