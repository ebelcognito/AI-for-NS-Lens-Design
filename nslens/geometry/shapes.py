"""Face shapes.

The hierarchy is not invented -- it is read off the existing code:

  Flat      a straight edge; contributes no arc control point to the element path.
  Arc       a circular face with signed sag, positive = convex OUTWARD (away from the
            element body). This is what fold_demo/arvr_prism emit and what the
            ray-optics JS engine understands natively.
  Freeform  a conic + polynomial deviation. ns_autodiff/element_freeform.arc_to_freeform
            shows an Arc is exactly a Freeform with c = -1/r, k = 0 and no polynomial
            terms, so Freeform is a strict superset and Arc.upgraded() is lossless.

The JS engine's scene format can express Flat and Arc only. Freeform is traced by the
torch backend (ns_autodiff/geometry_freeform.py). Asking for a JS scene dict from a
Freeform face is therefore an error, not a silent approximation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple


class Shape:
    """Base for face shapes. Subclasses are frozen value objects."""

    #: does this shape contribute an arc control vertex to the element boundary path?
    emits_control_point: bool = False

    def sag_value(self) -> float:
        """Signed outward bulge at the face centre, in world units. 0.0 for Flat."""
        return 0.0


@dataclass(frozen=True)
class Flat(Shape):
    """A straight edge between its two corners."""

    emits_control_point: bool = field(default=False, init=False, repr=False)


@dataclass(frozen=True)
class Arc(Shape):
    """Circular face. `sag` > 0 bulges outward, away from the element centre."""

    sag: float
    emits_control_point: bool = field(default=True, init=False, repr=False)

    def sag_value(self) -> float:
        return float(self.sag)

    def upgraded(self, powers: Sequence[int] = (),
                 coeffs: Sequence[float] | None = None) -> "Freeform":
        """Lossless promotion to Freeform: with no polynomial terms the traced surface
        is identical to this arc (see element_freeform.arc_to_freeform)."""
        return Freeform(sag=self.sag, powers=tuple(powers),
                        coeffs=tuple(coeffs) if coeffs is not None else None)


@dataclass(frozen=True)
class Freeform(Shape):
    """Conic base (initialised from the arc) plus an optional polynomial deviation.

    `powers`/`coeffs` are the transverse polynomial terms refined by the autodiff
    backend; empty means "exactly the base arc", which is how refinement starts.
    """

    sag: float
    powers: Tuple[int, ...] = ()
    coeffs: Tuple[float, ...] | None = None
    emits_control_point: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.coeffs is not None and len(self.coeffs) != len(self.powers):
            raise ValueError(
                f"freeform powers/coeffs length mismatch: "
                f"{len(self.powers)} powers vs {len(self.coeffs)} coeffs")

    def sag_value(self) -> float:
        return float(self.sag)

    @property
    def is_pure_arc(self) -> bool:
        """True when no polynomial deviation is active, i.e. still exactly the arc."""
        return not self.powers or self.coeffs is None or all(c == 0.0 for c in self.coeffs)
