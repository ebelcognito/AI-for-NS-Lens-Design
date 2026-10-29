"""Model output -> genotype, strictly, with a logged failure taxonomy.

Strict on purpose. A lenient parser that repairs the model's JSON hides the format-failure
rate, and doc 18 §4.2 makes that rate a go/no-go gate on the SFT warm start (">= 70% of
samples parse and are geometrically valid"). A parser that quietly fixes things cannot
measure it.

The taxonomy is the point of this module as much as the parsing is: `ParseResult.failure`
names WHICH way the output was wrong, so the training log can separate "the model does not
know the output format" from "the model knows the format and proposes absurd numbers".
Those need different fixes and look identical in a scalar reward.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .env_config import RLEnv

#: reward for output that does not parse at all
R_MALFORMED = 0.00
#: reward for a well-formed design block whose numbers fall outside the genome box.
#: Not in doc 18's ladder, which jumps 0.00 -> 0.05 at "parses, in-bounds". Splitting it
#: keeps the two failures distinguishable in the gradient: a model that has learned the
#: format and not the ranges is in a very different place from one emitting prose, and
#: collapsing them to the same number throws away the only signal that says so.
R_OUT_OF_BOUNDS = 0.02

_DESIGN_RE = re.compile(r"<design>(.*?)</design>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

FAILURES = ("no_design_block", "bad_json", "not_an_object", "missing_keys",
            "unknown_keys", "non_numeric", "out_of_bounds")


@dataclass
class ParseResult:
    ok: bool
    genotype: Optional[np.ndarray] = None
    params: Dict[str, float] = field(default_factory=dict)
    failure: str = ""
    detail: str = ""
    #: which parameters were out of range, as {name: (value, lo, hi)}
    violations: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    @property
    def reward_floor(self) -> float:
        """What this output is worth before the engine is asked anything."""
        if self.ok:
            return 0.05
        return R_OUT_OF_BOUNDS if self.failure == "out_of_bounds" else R_MALFORMED


def extract_block(text: str) -> Optional[str]:
    """The LAST <design> block. Last, not first: a model that reasons out loud often
    writes a candidate mid-thought and revises it, and the revision is the answer."""
    blocks = _DESIGN_RE.findall(text or "")
    return blocks[-1] if blocks else None


def parse(text: str, env: RLEnv) -> ParseResult:
    """Model output -> in-bounds genotype in the env's parameter order."""
    raw = extract_block(text)
    if raw is None:
        return ParseResult(False, failure="no_design_block",
                           detail="no <design>...</design> in the output")
    raw = _FENCE_RE.sub("", raw).strip()
    try:
        obj = json.loads(raw)
    except Exception as exc:
        return ParseResult(False, failure="bad_json", detail=str(exc))
    if not isinstance(obj, dict):
        return ParseResult(False, failure="not_an_object",
                           detail=f"design block is a {type(obj).__name__}, not an object")

    names = list(env.param_names)
    missing = [n for n in names if n not in obj]
    if missing:
        return ParseResult(False, failure="missing_keys",
                           detail=f"missing {missing}")
    unknown = [k for k in obj if k not in names]
    if unknown:
        # not repaired by dropping them: an unknown key usually means the model is
        # designing a different element than the one it was asked for
        return ParseResult(False, failure="unknown_keys", detail=f"unknown {unknown}")

    vals: List[float] = []
    for n in names:
        v = obj[n]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return ParseResult(False, failure="non_numeric",
                               detail=f"{n}={v!r} is {type(v).__name__}")
        v = float(v)
        if not np.isfinite(v):
            return ParseResult(False, failure="non_numeric", detail=f"{n}={v!r}")
        vals.append(v)

    bounds = env.bounds()
    violations = {n: (v, *bounds[n]) for n, v in zip(names, vals)
                  if not (bounds[n][0] <= v <= bounds[n][1])}
    params = dict(zip(names, vals))
    if violations:
        return ParseResult(False, params=params, failure="out_of_bounds",
                           detail=", ".join(f"{n}={v:g} not in [{lo:g}, {hi:g}]"
                                            for n, (v, lo, hi) in violations.items()),
                           violations=violations)
    return ParseResult(True, genotype=np.array(vals, float), params=params)


def format_design(env: RLEnv, x: Sequence[float], *, digits: int = 4) -> str:
    """The inverse, for building SFT targets and few-shot examples from real genotypes."""
    body = json.dumps({n: round(float(v), digits)
                       for n, v in zip(env.param_names, x)})
    return f"<design>{body}</design>"


def failure_histogram(results: Sequence[ParseResult]) -> Dict[str, int]:
    """Counts by taxonomy label, `ok` included. The training-log view."""
    out = {k: 0 for k in ("ok",) + FAILURES}
    for r in results:
        out["ok" if r.ok else r.failure] += 1
    return out
