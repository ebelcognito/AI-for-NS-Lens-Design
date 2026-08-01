r"""DEPRECATED shim -- the real module is nslens.search.cma_me.

Kept so the ~13 existing `from cma_me import ...` call sites keep working while phase 2
migrates them. Delete once nothing imports it (phase 4).

    grep -rl "import cma_me\|from cma_me" --include="*.py" .
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nslens.search.cma_me import CMA_ME, CMAES, Cell, GridArchive  # noqa: F401,E402

__all__ = ["CMAES", "Cell", "GridArchive", "CMA_ME"]
