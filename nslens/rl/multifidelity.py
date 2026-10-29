"""A controlled fidelity ladder for the hardened local optical verifier.

The ladder changes only pupil sampling and field sampling.  Every level keeps the same
geometry, material, detector, focus window, and measurement code as ``rl.reward``.  This
is important: a useful proxy experiment needs paired views of the *same* design, not a
collection of unrelated objectives that happen to have similar names.

``pencil_onaxis`` is intentionally not treated as a physical acceptance test.  One ray
cannot establish convergence, beam width, or robustness.  It supplies route/topology and
coarse throughput supervision only.  ``full_multifov`` remains the local authority.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple

from .env_config import RLEnv, field_set


@dataclass(frozen=True)
class Fidelity:
    """One point on the pupil x field fidelity ladder."""

    name: str
    n_rays: int
    n_fields: int
    acceptance_authority: bool = False

    def fields(self, fov: float) -> Tuple[float, ...]:
        if self.n_fields == 1:
            return (0.0,)
        return field_set(fov, self.n_fields)

    @property
    def nominal_engine_calls(self) -> int:
        # The bridge traces one scene per field.  Geometry rejects use zero calls, but the
        # nominal value is the stable budget unit for planning a dataset before tracing it.
        return self.n_fields


FIDELITIES: Tuple[Fidelity, ...] = (
    Fidelity("pencil_onaxis", n_rays=1, n_fields=1),
    Fidelity("beam_onaxis", n_rays=17, n_fields=1),
    Fidelity("sparse_multifov", n_rays=5, n_fields=3),
    Fidelity("full_multifov", n_rays=17, n_fields=9, acceptance_authority=True),
)

_BY_NAME: Dict[str, Fidelity] = {f.name: f for f in FIDELITIES}


def get_fidelity(name: str) -> Fidelity:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown fidelity {name!r}; known: {sorted(_BY_NAME)}") from exc


def with_fidelity(env: RLEnv, fidelity: Fidelity | str) -> RLEnv:
    """Return an env view at ``fidelity`` without mutating the cached base env."""
    fidelity = get_fidelity(fidelity) if isinstance(fidelity, str) else fidelity
    problem = replace(env.problem, n_rays=fidelity.n_rays)
    fields = fidelity.fields(env.fov)
    cfg = replace(
        env.cfg,
        problem=problem,
        fields=fields,
        name=f"{env.cfg.name}__{fidelity.name}",
    )
    return replace(env, cfg=cfg)


def ladder(env: RLEnv, names: Iterable[str] | None = None
           ) -> Tuple[Tuple[Fidelity, RLEnv], ...]:
    selected = FIDELITIES if names is None else tuple(get_fidelity(n) for n in names)
    return tuple((f, with_fidelity(env, f)) for f in selected)


def validate_ladder() -> None:
    """Fail if an edit makes the declared ladder cheaper only in name."""
    last = (0, 0)
    authority = []
    for fidelity in FIDELITIES:
        current = (fidelity.n_fields, fidelity.n_rays)
        if current[0] < last[0]:
            raise ValueError("fidelity field counts must be non-decreasing")
        if fidelity.n_fields == last[0] and current[1] < last[1]:
            raise ValueError("ray count must not fall at a fixed field count")
        last = current
        if fidelity.acceptance_authority:
            authority.append(fidelity.name)
    if authority != ["full_multifov"]:
        raise ValueError("full_multifov must be the sole local acceptance authority")


validate_ladder()


__all__ = [
    "Fidelity", "FIDELITIES", "get_fidelity", "with_fidelity", "ladder",
]
