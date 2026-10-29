"""Task supply: where the specs the policy trains on come from.

The archive cannot be the task distribution, and that is a measured fact rather than a
preference: 149 archives on disk hold 695 unique (experiment, cell) specs, of which only
~143 are backed by a design good enough to prove the spec was worth asking for. A model
trained on 143 goal specs memorises them. Doc 18 §0.3.

So specs are generated and then **certified** -- a short CMA-ME probe actually reaches a
score in the cell before any spec claims it is reachable. That step is what prevents the
failure this program has already hit once: the Tier A V1 benchmark spec was unsatisfiable,
and only 2 of its 5 fields could ever close. Never ship an uncertified spec.
"""
from __future__ import annotations

__all__ = ["generate", "certify", "splits"]
