r"""DEPRECATED shim -- the real module is nslens.engine.signatures.

Kept so the ~20 existing `from sequence_tools import ...` call sites keep working while
phase 2 migrates them. Delete this file once nothing imports it (phase 4).

    grep -rl "import sequence_tools\|from sequence_tools" --include="*.py" .
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nslens.engine.signatures import *          # noqa: F401,F403,E402
from nslens.engine.signatures import (          # noqa: E402  (explicit: `*` skips these)
    Signature, Vertex, aggregate_by_signature, audit_ray_energy, dominance_report,
    format_signature, ledger_closure, make_single_element_face_classifier, matches,
    parse_target, print_report, signature_of)

__all__ = ["Signature", "Vertex", "parse_target", "signature_of", "matches",
           "aggregate_by_signature", "format_signature", "ledger_closure",
           "audit_ray_energy", "dominance_report", "print_report",
           "make_single_element_face_classifier"]
