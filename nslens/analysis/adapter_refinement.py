"""Bounded freeform-state evaluation and local refinement for three mirrors."""
from __future__ import annotations

import numpy as np

from nslens.analysis import final_freeform as ff
from nslens.analysis.optical_refinement import geometry_crossing
from nslens.search.cma_me import CMAES

_MIRRORS = 3
_STATE_SIZE = 13


def _validated_state(state) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    if values.shape != (_STATE_SIZE,) or not np.all(np.isfinite(values)):
        raise ValueError("state must contain 13 finite normalized parameters")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("state parameters must lie in [-1, 1]")
    return values.copy()


def evaluate_state(state, task):
    """Trace one three-mirror state (9 genes, then four coeffs scaled by 2)."""
    x = _validated_state(state)
    genes = x[:9]
    coefficients = 2.0 * x[9:13]
    row = ff.trace(genes, _MIRRORS, coefficients, fields=5, pupils=21, **task)
    crossing = geometry_crossing(genes, _MIRRORS, coefficients, 65)
    row["crossing"] = bool(crossing)
    row["optical_loss"] = row["loss"]
    row["loss"] = float(row["loss"] + 30000.0 * crossing)
    row["state"] = x.tolist()
    return row


def refine_state(state, task, seed, budget=65):
    """Refine only the final curvature and four freeform coefficients.

    ``budget`` counts all trace calls, including the initial incumbent evaluation.
    """
    x = _validated_state(state)
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("budget must be a positive integer")

    calls = 0

    def evaluate(candidate):
        nonlocal calls
        row = evaluate_state(candidate, task)
        calls += 1
        return row

    initial = evaluate(x)
    best = initial
    if budget > 1:
        lower = np.full(5, -1.0)
        upper = np.full(5, 1.0)
        cma = CMAES(x[[8, 9, 10, 11, 12]], 0.03, popsize=8, seed=seed,
                    bounds=(lower, upper), boundary_handling="reflect")
        while calls < budget:
            population = cma.ask()
            count = min(len(population), budget - calls)
            rows = []
            for candidate in population[:count]:
                proposed = x.copy()
                proposed[[8, 9, 10, 11, 12]] = candidate
                row = evaluate(proposed)
                rows.append(row)
                if row["loss"] < best["loss"]:
                    best = row
            if count == len(population):
                cma.tell(np.argsort([row["loss"] for row in rows]))
    assert calls == int(budget)
    np.testing.assert_array_equal(np.asarray(best["state"])[0:8], x[0:8])
    return {"initial": initial, "coarse": best, "state": best["state"],
            "coarse_calls": calls}
