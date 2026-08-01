"""nslens -- non-sequential lens design.

A parametric lens element, an engine-agnostic scene, and pluggable fitness and
descriptor functions. See MODULARIZATION_PLAN.md for how this replaces the script
sprawl in ns_probe/ and ns_autodiff/.
"""
from .geometry import Arc, Element, Face, Flat, Freeform

__all__ = ["Element", "Face", "Flat", "Arc", "Freeform"]
__version__ = "0.1.0"
