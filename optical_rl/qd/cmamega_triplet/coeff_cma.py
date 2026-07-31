"""Backwards-compat shim. The canonical CMA-ES engine lives in
optical_rl/qd/_cma.py and is shared by every CMA-MEGA runner."""

from optical_rl.qd._cma import GradientCoefficientCMA  # noqa: F401
