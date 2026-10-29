"""Cheap, explicit evidence for an intended interaction path.

The historical QD objective inferred a design's topology from whichever delivered
signature happened to carry the most energy.  That is useful for discovery, but it is
not a sufficient contract once an environment names the path it is meant to realise:
a different path can dominate at another field, and Fresnel descendants can make a
handful of source rays look like a supported bundle.

This module is the inner-loop counterpart of the dense interaction audit.  It uses the
ordinary QD ray count, but preserves the audit's two important semantics: energy is
assigned to an exact typed path, and support counts distinct source roots rather than
terminal branches.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

from ..engine.signatures import (Signature, aggregate_by_signature, matches,
                                 signature_of)
from .observation import Observation


def root_ray_ids(records: Sequence[Mapping]) -> dict[int, int]:
    """Map every recorded branch id to its primary source-ray id.

    A cycle is treated as its own root rather than looping forever.  The simulator never
    intentionally emits one, but keeping this helper total makes malformed diagnostic
    output fail the support threshold instead of hanging an optimisation run.
    """
    parent = {int(record["rayId"]): record.get("parentRayId")
              for record in records if record.get("rayId") is not None}
    roots: dict[int, int] = {}
    for ray_id in parent:
        current = ray_id
        seen: set[int] = set()
        while parent.get(current) is not None and current not in seen:
            seen.add(current)
            current = int(parent[current])
        roots[ray_id] = current
    return roots


def target_path_evidence(result: Mapping, face_of: Callable, target: Signature,
                         n_source_rays: int) -> dict:
    """Measure target-path support and energy for one field trace.

    Only forward, first-crossing sensor records contribute delivered energy.  Multiple
    reflected/refracted descendants of one primary ray contribute their physical energy
    but count once toward ``root_support``.
    """
    if n_source_rays <= 0:
        raise ValueError("n_source_rays must be positive")
    sequences = list(result.get("raySequences", []) or [])
    terminal = list(result.get("rayTerminations", []) or [])
    roots = root_ray_ids([*terminal, *sequences])
    by_signature = aggregate_by_signature(sequences, face_of)
    target_energy = sum(float(energy) for signature, energy in by_signature.items()
                        if matches(signature, target))
    biggest_other = max(
        (float(energy) for signature, energy in by_signature.items()
         if not matches(signature, target)), default=0.0)

    matching_roots = set()
    for record in sequences:
        if record.get("dirx", 0.0) <= 0.0:
            continue
        # aggregate_by_signature already applies the first-crossing rule.  Reuse the
        # signature it calculated through this small lookup rather than interpreting
        # terminal histories, which also contain escaped and absorbed branches.
        signature = signature_of(record, face_of)
        if signature is not None and matches(signature, target):
            ray_id = record.get("rayId")
            if ray_id is not None and int(ray_id) in roots:
                matching_roots.add(roots[int(ray_id)])

    emitted = float((result.get("energyLedger") or {}).get("emitted", 0.0))
    return {
        "root_support": min(1.0, len(matching_roots) / float(n_source_rays)),
        "delivered_energy_share": target_energy / emitted if emitted > 0.0 else 0.0,
        "target_energy": target_energy,
        "biggest_other_class": biggest_other,
        "target_is_dominant": bool(target_energy > 0.0
                                   and target_energy > biggest_other),
    }


def target_records(obs: Observation, target: Signature) -> list[dict]:
    """Forward sensor records whose complete typed signature matches ``target``."""
    return [record for record in obs.records
            if record.get("_sig") is not None and matches(record["_sig"], target)]


def target_focused_throughput(obs: Observation, target: Signature,
                              focus_w: float) -> float:
    """Input-energy fraction on the intended path and inside its own focus window."""
    records = target_records(obs, target)
    if not records or obs.emitted <= 0.0:
        return 0.0
    y = np.asarray([record["y"] for record in records], dtype=float)
    b = np.asarray([record["b"] for record in records], dtype=float)
    if b.sum() <= 0.0:
        return 0.0
    centroid = float(np.average(y, weights=b))
    return float(b[np.abs(y - centroid) <= focus_w].sum() / obs.emitted)


def target_critical_margin(obs: Observation, target: Signature,
                           n_glass: float) -> float | None:
    """Minimum TIR incidence margin over every delivered target-path ray.

    ``None`` means that no measurable TIR event was present.  An intended fold config
    treats that as a failed margin gate; it must never be confused with a large margin.
    """
    theta_c = math.degrees(math.asin(min(1.0 / float(n_glass), 1.0)))
    margins = []
    for record in target_records(obs, target):
        seq = record["seq"]
        for index in range(1, len(seq) - 1):
            if seq[index].get("t") != "tir":
                continue
            incoming = np.asarray([
                seq[index]["x"] - seq[index - 1]["x"],
                seq[index]["y"] - seq[index - 1]["y"],
            ], dtype=float)
            outgoing = np.asarray([
                seq[index + 1]["x"] - seq[index]["x"],
                seq[index + 1]["y"] - seq[index]["y"],
            ], dtype=float)
            denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
            if denominator <= 1e-12:
                continue
            cosine = float(np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0))
            incidence = 90.0 - math.degrees(math.acos(cosine)) / 2.0
            margins.append(incidence - theta_c)
    return float(min(margins)) if margins else None


__all__ = [
    "root_ray_ids",
    "target_critical_margin",
    "target_focused_throughput",
    "target_path_evidence",
    "target_records",
]
