"""Verified local repair of continuous-boundary, 2-D freeform imaging elements.

This is a bounded experimental environment, not a 3-D or folded-optics claim.
It uses the existing non-sequential spline tracer with Fresnel energy accounting.
The learner receives physical requirements, current geometry, and measured ray errors;
neither the hidden solvability witness nor the corruption vector is a policy input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from nslens.freeform_body.codec import FreeformBodyCodec
from nslens.freeform_body.trace import InputRay, project_exits_to_x, trace_bundle

SCHEMA = "nslens.freeform_repair.v1"
ACTIVE_GENES = (5, 6, 9, 10)  # radial cosine/sine modes 2 and 3
BOUND = 0.65
MAX_STEP = 0.30
SEVERITIES = (0.025, 0.075, 0.16, 0.28)


@dataclass(frozen=True)
class ImagingSpec:
    family_id: str
    split: str
    radius_x: float = 8.0
    radius_y: float = 20.0
    index: float = 1.60
    focal_distance: float = 45.0
    half_field_deg: float = 2.0
    chief_deg: float = 0.0
    pupil_half: float = 3.0
    max_rms: float = 0.03
    max_field_rms: float = 0.05
    min_ray_energy: float = 0.75

    def __post_init__(self):
        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.isfinite(numeric).all():
            raise ValueError("specification must be finite")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("unknown split")
        if min(self.radius_x, self.radius_y, self.focal_distance, self.half_field_deg,
               self.pupil_half, self.max_rms, self.max_field_rms) <= 0:
            raise ValueError("lengths, field extent, and tolerances must be positive")
        if not 1.5 <= self.index <= 1.7 or not 0 < self.min_ray_energy <= 1:
            raise ValueError("invalid index or energy floor")

    def codec(self):
        return FreeformBodyCodec(center=(60.0, 0.0),
                                 base_radii=(self.radius_x, self.radius_y),
                                 n=self.index, control_count=16, harmonic_count=2)

    def context(self):
        return np.array([self.radius_x / 8, self.radius_y / 20,
                         (self.index - 1.6) / .1, self.focal_distance / 45,
                         self.half_field_deg / 2, self.chief_deg / 3,
                         self.pupil_half / 3])


@dataclass
class Verdict:
    residual: list[float]
    rms: float
    worst_field_rms: float
    min_ray_energy: float
    throughput: float
    route_fraction: float
    geometry_valid: bool
    physical_valid: bool
    success: bool
    closure_error: float
    unresolved_energy_fraction: float
    ghost_energy_fraction: float
    reason: str
    rays: int


def evaluate(spec: ImagingSpec, x, fidelity="coarse") -> Verdict:
    """Score *every* launched pupil ray, without a target-window clipping reward.

    The declared first domain is a transmitting imaging element: each primary ray
    must realize two transmissions. Ghosts are measured separately. Missing rays or
    low energy cannot improve the score by disappearing from the RMS calculation.
    Dense and audit grids include additional fields and shifted interior pupil rays.
    """
    grids = {"coarse": (3, 5, .002), "dense": (7, 17, .0002),
             "audit": (9, 31, .00002)}
    nf, npupil, floor = grids[fidelity]
    x = np.asarray(x, dtype=float)
    nrays = nf * npupil

    def failure(reason, geometry=False):
        return Verdict([20.] * nrays, 20., 20., 0., 0., 0., geometry, False,
                       False, 0., 1., 0., reason, nrays)

    if x.shape != (4,) or not np.isfinite(x).all() or np.any(np.abs(x) > BOUND + 1e-12):
        return failure("invalid freeform coefficients")
    genes = np.zeros(13)
    genes[list(ACTIVE_GENES)] = x
    body = spec.codec().decode(genes)
    geometry = body.geometry_report()
    if not geometry.valid:
        return failure("; ".join(geometry.reasons))
    fields = spec.chief_deg + np.linspace(-spec.half_field_deg, spec.half_field_deg, nf)
    pupil = np.linspace(-1., 1., npupil)
    if fidelity != "coarse":
        # Preserve the physical aperture endpoints; move interior samples off the
        # optimization grid. The audit uses a different shift from dense admission.
        pupil[1:-1] += (.27 if fidelity == "dense" else -.31) * 2 / (npupil - 1)
    rays, target = [], []
    for angle in fields:
        theta = np.deg2rad(angle)
        for y in pupil * spec.pupil_half:
            rays.append(InputRay((0., float(y - 60. * np.tan(theta))),
                                 (float(np.cos(theta)), float(np.sin(theta))),
                                 source_id=len(rays)))
            target.append(spec.focal_distance * np.tan(theta))
    trace = trace_bundle(body, rays, energy_floor=floor, max_interactions=12,
                         intersection_scan=512 if fidelity == "coarse" else 1024)
    records = project_exits_to_x(trace, 60. + spec.focal_distance)
    groups: dict[int, list] = {}
    for record in records:
        groups.setdefault(record["source_id"], []).append(record)
    residual, energy, routes, ghost = [], [], [], 0.
    for i, wanted in enumerate(target):
        group = groups.get(i, [])
        if not group:
            residual.append(20.)
            energy.append(0.)
            routes.append(False)
            continue
        main = max(group, key=lambda row: row["energy"])
        residual.append(float(main["y"] - wanted))
        energy.append(float(main["energy"]))
        routes.append(tuple(event.kind for event in main["events"]) ==
                      ("refract", "refract"))
        ghost += sum(row["energy"] for row in group) - main["energy"]
    residual = np.array(residual)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    worst = float(np.sqrt(np.mean(residual.reshape(nf, npupil) ** 2, axis=1)).max())
    unresolved = (trace.depth_limited_energy + trace.branch_limited_energy) / nrays
    closure = abs(trace.closure_residual) / nrays
    valid = (min(energy) >= spec.min_ray_energy and all(routes) and
             unresolved < 1e-5 and closure < 1e-9 and np.isfinite(residual).all())
    success = valid and rms <= spec.max_rms and worst <= spec.max_field_rms
    return Verdict(residual.tolist(), rms, worst, min(energy), float(np.mean(energy)),
                   float(np.mean(routes)), True, bool(valid), bool(success), closure,
                   unresolved, ghost / nrays, "" if valid else "energy/route/trace gate",
                   nrays)


def merit(v: Verdict):
    # Feasibility cannot be traded for a smaller RMS.
    return (not v.physical_valid, v.rms)


def bounded_action(action):
    action = np.nan_to_num(np.asarray(action, dtype=float), nan=0., posinf=0., neginf=0.)
    return action * min(1., MAX_STEP / max(float(np.linalg.norm(action)), 1e-12))


class Recorder:
    """Every attempted evaluation, including failed geometry and rejected steps."""
    def __init__(self, spec, task_id, stage):
        self.spec, self.task_id, self.stage = spec, task_id, stage
        self.records = []

    def __call__(self, x, fidelity="coarse", source="evaluation", parent=None):
        v = evaluate(self.spec, x, fidelity)
        self.records.append(dict(schema_version=SCHEMA, task_id=self.task_id,
                                 family_id=self.spec.family_id, split=self.spec.split,
                                 stage=self.stage, fidelity=fidelity, source=source,
                                 x=np.asarray(x).tolist(), parent=parent,
                                 verdict=asdict(v)))
        return v


def teacher_step(x, initial: Verdict, measure: Callable):
    """One finite-difference Gauss-Newton step, exactly eight candidate calls.

    Includes four Jacobian probes and four trust-region line-search candidates.
    All proposals are retained; only a physically valid improvement is imitated.
    """
    x = np.asarray(x)
    cols = []
    best_x, best_v = x.copy(), initial
    for j in range(4):
        probe = x.copy()
        h = .001 if x[j] < BOUND - .001 else -.001
        probe[j] += h
        v = measure(probe, source="jacobian", parent=x.tolist())
        cols.append((np.array(v.residual) - initial.residual) / h)
        if merit(v) < merit(best_v):
            best_x, best_v = probe.copy(), v
    jac = np.array(cols).T
    delta = np.linalg.solve(jac.T @ jac + .01 * np.eye(4),
                            -jac.T @ np.array(initial.residual))
    delta = bounded_action(delta)
    for scale in (1., .5, .25, .125):
        candidate = np.clip(x + scale * delta, -BOUND, BOUND)
        v = measure(candidate, source="teacher", parent=x.tolist())
        if merit(v) < merit(best_v):
            best_x, best_v = candidate.copy(), v
    return best_x, best_v


def features(spec, x, v):
    if len(v.residual) != 15:
        raise ValueError("policy observations must use the 3-field, 5-pupil coarse grid")
    # These interactions let a small model learn how an error-to-edit map changes
    # with geometry and requirements. No hidden witness, task ID, or severity input.
    context = np.r_[spec.context(), np.asarray(x)]
    r = np.clip(v.residual, -20., 20.)
    return np.r_[context, r, np.outer(r, context).ravel()]


class RepairPolicy:
    def __init__(self):
        self.model = None

    def fit(self, examples):
        if not examples:
            return self
        if any(e["split"] != "train" for e in examples):
            raise ValueError("held-out examples cannot train a repair policy")
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        self.model.fit(np.array([e["features"] for e in examples]),
                       np.array([e["action"] for e in examples]))
        return self

    def action(self, spec, x, v):
        if self.model is None:
            return np.zeros(4)
        return bounded_action(self.model.predict(features(spec, x, v)[None])[0])


class ResponsePolicy:
    """Learn the measured surface-edit -> ray-error Jacobian, then solve for an edit.

    The local response is smoother than a teacher's clipped accepted action. This
    makes the reusable training label a physical intervention, not an optimizer's
    trust-region convention. Finite-difference probes are made only during training;
    inference predicts the Jacobian without additional ray calls.
    """
    def __init__(self):
        self.model = None

    def fit(self, examples):
        if any(e["split"] != "train" for e in examples):
            raise ValueError("held-out examples cannot train a response policy")
        rows = [e for e in examples if e.get("jacobian") is not None]
        if rows:
            self.model = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(alpha=1.))
            self.model.fit(np.array([e["response_features"] for e in rows]),
                           np.array([e["jacobian"] for e in rows]))
        return self

    def action(self, spec, x, v):
        if self.model is None:
            return np.zeros(4)
        if len(v.residual) != 15:
            raise ValueError("response policy requires the coarse observation grid")
        jac = self.model.predict(np.r_[spec.context(), x][None])[0].reshape(15, 4)
        delta = np.linalg.solve(jac.T @ jac + .01 * np.eye(4),
                                -jac.T @ np.asarray(v.residual))
        return bounded_action(delta)


def solve(spec, x, measure, *, policy=None, algorithm="learned", budget=8, seed=0):
    """Same initial evaluation and candidate allowance for every method.

    Search decisions use only the coarse verifier. Dense/audit checks happen later.
    """
    if budget < 1:
        raise ValueError("candidate budget must be positive")
    x = np.array(x, dtype=float)
    initial = current = measure(x, source="initial")
    rng = np.random.default_rng(seed)
    if algorithm == "teacher":
        if budget % 8:
            raise ValueError("teacher budget must be a multiple of eight")
        for _ in range(budget // 8):
            x, current = teacher_step(x, current, measure)
    else:
        for step in range(budget):
            if algorithm == "learned":
                delta = policy.action(spec, x, current)
                # Backtracking is driven only by measured rejection.
                scale = 1. if step == 0 or improved else max(.125, scale * .5)
                candidate = np.clip(x + scale * delta, -BOUND, BOUND)
            elif algorithm == "random":
                candidate = np.clip(x + rng.normal(0., (.02, .06, .15)[step % 3], 4),
                                    -BOUND, BOUND)
            elif algorithm == "reset":
                # Detect a trivial 'undo all freeform coefficients' solution.
                candidate = np.zeros(4)
            else:
                raise ValueError(f"unknown algorithm {algorithm}")
            v = measure(candidate, source=algorithm, parent=x.tolist())
            improved = merit(v) < merit(current)
            if improved:
                x, current = candidate, v
    return x, initial, current


def curriculum_weights(strategy, successes, attempts, round_index, rounds):
    """Coverage floor plus a learnability heuristic, never test-set feedback."""
    if strategy == "uniform":
        return np.full(4, .25)
    if strategy == "fixed":
        focus = min(3, int(4 * round_index / max(rounds, 1)))
        w = np.full(4, .1)
        w[focus] += .6
        return w
    if strategy != "adaptive":
        raise ValueError("unknown curriculum")
    # A beta(1,1) posterior avoids hard 0/1 decisions from tiny probes. Preference
    # for mixed outcomes approximates learnability; it is not measured learning gain.
    p = (np.asarray(successes) + 1) / (np.asarray(attempts) + 2)
    w = .15 + 4 * p * (1 - p)
    return w / w.sum()
