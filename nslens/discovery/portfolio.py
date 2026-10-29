"""Retain measured stepping stones and geometric alternatives for restarts.

This module does not relax physical acceptance. Selection uses only search-budget
measurements, never dense audit outcomes or previously completed experiments.
"""
from __future__ import annotations

import numpy as np


def constraint_rank(row):
    """Lexicographic minimax: reduce the worst violation, then the next worst, etc."""
    if row["status"] != "measured":
        return (0,)
    if row["feasible"]:
        return (2, row["quality"], row["support"])
    violations = sorted(row["violations"].values(), reverse=True)
    return (1, *(-v for v in violations), *row["progress"])


def select_diverse(rows, capacity, ranking=constraint_rank):
    """Keep the best row, then farthest geometry among a bounded quality shortlist.

    The shortlist is the best 2*capacity rows; within it farthest-first retention
    has no arbitrary distance cutoff. Distances are in normalized gene space.
    """
    unique = {}
    for row in rows:
        key = tuple(row["genes"])
        if key not in unique or ranking(row) > ranking(unique[key]):
            unique[key] = row
    shortlist = sorted(unique.values(), key=ranking, reverse=True)[:2*capacity]
    if not shortlist or capacity < 1:
        return []
    selected = [shortlist.pop(0)]
    while shortlist and len(selected) < capacity:
        chosen = max(range(len(shortlist)), key=lambda i: (
            min(float(np.mean((np.asarray(shortlist[i]["genes"])-np.asarray(s["genes"]))**2))
                for s in selected), ranking(shortlist[i])))
        selected.append(shortlist.pop(chosen))
    return selected


class RestartPool:
    def __init__(self, capacity=12, ranking=constraint_rank):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("positive pool capacity required")
        self.capacity = capacity
        self.ranking = ranking
        self.rows = []
        self.visits = {}

    def update(self, rows):
        measured = [r for r in rows if r["status"] == "measured"]
        self.rows = select_diverse([*self.rows, *measured], self.capacity, self.ranking)

    def start(self):
        if not self.rows:
            return None
        # Prefer untried retained basins; rank resolves ties. A start is counted even
        # when its subsequent population fails, so one bad basin cannot monopolize.
        row = max(self.rows, key=lambda r: (-self.visits.get(tuple(r["genes"]), 0), self.ranking(r)))
        key = tuple(row["genes"])
        self.visits[key] = self.visits.get(key, 0) + 1
        return np.asarray(row["genes"])
