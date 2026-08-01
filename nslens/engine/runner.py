"""The ray-optics engine process.

Thin re-export of the persistent Node.js bridge under a stable name. The bridge itself
still lives in optical_rl/utils while phase 2 is in progress; this module is the seam,
so downstream code imports `nslens.engine.Runner` and never has to know that.

    from nslens.engine import Runner
    with Runner(timeout_s=20) as runner:
        result, _ = runner.simulate(layout.to_scene_dict())
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from optical_rl.utils.ray_optics_bridge_persistent import (  # noqa: E402
    PersistentRayOpticsRunner as Runner)

__all__ = ["Runner"]
