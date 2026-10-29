"""Experiment 1: dense bundle audit of the historical fold catalogue.

This is deliberately an audit, not another optimizer.  It replays both accepted and
rejected catalogue candidates with denser field/pupil sampling, retains all terminal
fates, fingerprints every exact and aggregate branch, and tests the historically claimed
interaction sequence under small geometry perturbations.

Every invocation refuses an existing output directory.  Experimental outputs therefore
cannot silently overwrite either a previous run or source data.

Example::

    python -m nslens.analysis.interaction_audit \
      --catalogue runs/rl_datasets/hardened_catalogue_v1 \
      --out runs/interaction_audit_20260903_v1 --workers 12
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..engine import (Runner, fingerprint_record, make_single_element_contact_coordinate,
                      terminal_ledger_residuals)
from ..engine.signatures import format_signature, parse_target
from ..qd.fixed_environments import problem_from_environment_spec
from ..rl import env_config
from ..rl.reward import geometry_verdict

AUDIT_SCHEMA = "nslens.interaction_audit.v1"
FINGERPRINT_AGGREGATE_SCHEMA = "nslens.fingerprint_aggregate.v1"

DEFAULT_THRESHOLDS = {
    # A contact is bundle-supported if descendants from this fraction of source rays
    # share the claimed prefix. Root IDs prevent Fresnel branching from inflating counts.
    "bundle_support": 0.80,
    "spurious_bundle_support": 0.35,
    "field_persistence": 0.80,
    "perturbation_persistence": 0.80,
    # At least this input-energy fraction must arrive at the detector on the complete
    # claimed route before a whole candidate can be called intentional.
    "functional_delivered_energy": 0.05,
    "spurious_delivered_energy": 0.01,
    # Dimensionless fraction of a face's clear span between repeated contacts.
    "repeat_separation": 0.10,
    "tir_critical_margin_deg": 0.50,
    "tir_grazing_margin_deg": 0.50,
    "ledger_tolerance": 1e-9,
}

_RUNNER = None
_ENV = None
_BASE_PROBLEM = None
_FIXED_ENVIRONMENT_ID = None
_DENSE_PROBLEM = None
_PERTURB_PROBLEM = None
_DENSE_FIELDS: tuple[float, ...] = ()
_PERTURB_FIELDS: tuple[float, ...] = ()
_PERTURBATIONS = 0
_PERTURB_SCALE = 0.0
_MAX_FINGERPRINTS = 0
_GLOBAL_SEED = 0


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _topology_key(fingerprint: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple((event["interaction"], event.get("mechanism"), event["surface"],
                  int(event["repeat_index"])) for event in fingerprint["events"])


def _topology_id(key: Sequence[Sequence[Any]]) -> str:
    return "topo:" + hashlib.sha256(_canonical_json(key).encode()).hexdigest()[:20]


def _topology_text(key: Sequence[Sequence[Any]]) -> str:
    pretty = {"refract": "T", "tir": "TIR", "partial_reflect": "PR",
              "mirror": "M", "beam_split": "BS", "absorbed": "ABS"}
    parts = []
    for interaction, mechanism, surface, repeat in key:
        token = pretty.get(interaction, interaction)
        if mechanism is not None:
            token += f"[{mechanism}]"
        parts.append(f"{token}@{surface}#{repeat}")
    return " -> ".join(parts)


def _optical_vertices(record: Mapping[str, Any], face_of) -> list[tuple[tuple[str, str], dict]]:
    out = []
    for vertex in record.get("seq", []):
        interaction = str(vertex.get("t") or "other")
        if interaction in {"source", "sensor", "depth_limit"}:
            continue
        out.append(((interaction, str(face_of(vertex))), vertex))
    return out


def _root_ids(records: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    parent = {int(record["rayId"]): record.get("parentRayId")
              for record in records if record.get("rayId") is not None}
    roots = {}
    for ray_id in parent:
        current = ray_id
        seen = set()
        while parent.get(current) is not None and current not in seen:
            seen.add(current)
            current = int(parent[current])
        roots[ray_id] = current
    return roots


def _weighted_quantile(values: Sequence[tuple[float, float]], q: float) -> float | None:
    clean = sorted((float(value), max(0.0, float(weight)))
                   for value, weight in values if math.isfinite(float(value)))
    if not clean:
        return None
    total = sum(weight for _, weight in clean)
    if total <= 0:
        return clean[int(round(q * (len(clean) - 1)))][0]
    target = q * total
    cumulative = 0.0
    for value, weight in clean:
        cumulative += weight
        if cumulative >= target:
            return value
    return clean[-1][0]


def _turning_angle(vertex: Mapping[str, Any]) -> float | None:
    values = (vertex.get("dirInX"), vertex.get("dirInY"),
              vertex.get("dirOutX"), vertex.get("dirOutY"))
    if any(value is None for value in values):
        return None
    ix, iy, ox, oy = map(float, values)
    denominator = math.hypot(ix, iy) * math.hypot(ox, oy)
    if denominator <= 0:
        return None
    cosine = max(-1.0, min(1.0, (ix * ox + iy * oy) / denominator))
    return math.degrees(math.acos(cosine))


def _target_signature(text: str) -> tuple[tuple[str, str], ...]:
    return parse_target(text.replace("->", " ")) if text else ()


def _analyse_trace(result: Mapping[str, Any], problem, params: Mapping[str, float],
                   target: Sequence[tuple[str, str]], n_source_rays: int) -> dict:
    face_of = problem.face_of(params)
    coordinate_of = make_single_element_contact_coordinate(
        params["cx"], params["thick"], params["tilt_a"], params["tilt_b"],
        problem.half_h)
    terminal = list(result.get("rayTerminations", []))
    exact = [record for record in terminal
             if record.get("historyComplete") is True and not record.get("aggregate")]
    roots = _root_ids(terminal)
    enriched = []
    fingerprint_groups: dict[str, dict[str, Any]] = {}
    topology_groups: dict[str, dict[str, Any]] = {}

    for record in terminal:
        fp = fingerprint_record(record, face_of, coordinate_of)
        key = _topology_key(fp)
        tid = _topology_id(key)
        fhash = fp["fingerprint_hash"]
        fgroup = fingerprint_groups.setdefault(fhash, {
            "fingerprint_hash": fhash, "path_hash": fp["path_hash"],
            "topology_id": tid, "topology": _topology_text(key),
            "events": fp["events"], "terminal": fp["terminal"],
            "energy": 0.0, "ray_count": 0,
        })
        fgroup["energy"] += float(record.get("b", 0.0))
        fgroup["ray_count"] += 1

        optical = _optical_vertices(record, face_of)
        item = {
            "record": record,
            "fingerprint": fp,
            "topology_id": tid,
            "topology_key": key,
            "legacy": tuple(event for event, _ in optical),
            "vertices": tuple(vertex for _, vertex in optical),
            "root": roots.get(record.get("rayId")),
            "exact": record.get("historyComplete") is True and not record.get("aggregate"),
        }
        enriched.append(item)

        tgroup = topology_groups.setdefault(tid, {
            "topology_id": tid, "topology": _topology_text(key), "energy": 0.0,
            "delivered_energy": 0.0, "exact_roots": set(), "delivered_roots": set(),
            "fingerprint_classes": set(), "outcomes": defaultdict(float),
        })
        energy = float(record.get("b", 0.0))
        tgroup["energy"] += energy
        tgroup["outcomes"][str(record.get("outcome", "unknown"))] += energy
        tgroup["fingerprint_classes"].add(fhash)
        if item["root"] is not None and item["exact"]:
            tgroup["exact_roots"].add(item["root"])
        if record.get("outcome") == "delivered" and item["exact"]:
            tgroup["delivered_energy"] += energy
            if item["root"] is not None:
                tgroup["delivered_roots"].add(item["root"])

    emitted = float((result.get("energyLedger") or {}).get("emitted", 0.0))
    dense_topologies = []
    for group in topology_groups.values():
        dense_topologies.append({
            "topology_id": group["topology_id"], "topology": group["topology"],
            "energy_share": group["energy"] / emitted if emitted else 0.0,
            "delivered_energy_share": group["delivered_energy"] / emitted if emitted else 0.0,
            "exact_root_support": min(1.0, len(group["exact_roots"]) / n_source_rays),
            "delivered_root_support": min(1.0, len(group["delivered_roots"]) / n_source_rays),
            "fingerprint_classes": len(group["fingerprint_classes"]),
            "outcome_energy_share": {name: value / emitted if emitted else 0.0
                                     for name, value in sorted(group["outcomes"].items())},
        })
    dense_topologies.sort(key=lambda group: (-group["delivered_energy_share"],
                                              -group["energy_share"],
                                              group["topology_id"]))
    dominant = (dense_topologies[0]["topology_id"]
                if dense_topologies and dense_topologies[0]["delivered_energy_share"] > 0
                else None)

    contacts = []
    for index, expected in enumerate(target):
        matches = [item for item in enriched
                   if item["exact"] and item["legacy"][:index + 1] == tuple(target[:index + 1])]
        matching_roots = {item["root"] for item in matches if item["root"] is not None}
        energy = sum(float(item["record"].get("b", 0.0)) for item in matches)
        # One representative per source root prevents descendants created after this
        # contact from biasing angular and footprint statistics.
        representative = {}
        for item in matches:
            root = item["root"]
            if root is None:
                continue
            if root not in representative or item["record"].get("b", 0) > \
                    representative[root]["record"].get("b", 0):
                representative[root] = item

        tir_values = []
        separation_values = []
        for item in representative.values():
            vertex = item["vertices"][index]
            weight = float(item["record"].get("b", 0.0))
            if expected[0] == "tir":
                turn = _turning_angle(vertex)
                if turn is not None:
                    incidence = 90.0 - turn / 2.0
                    critical = math.degrees(math.asin(min(1.0 / float(params["n"]), 1.0)))
                    tir_values.append((incidence - critical, 1.0))
            prior = [j for j in range(index) if target[j][1] == expected[1]]
            if prior:
                earlier = item["vertices"][prior[-1]]
                q_now = coordinate_of(vertex, expected[1])
                q_before = coordinate_of(earlier, expected[1])
                if q_now is not None and q_before is not None:
                    separation_values.append((abs(q_now - q_before), max(weight, 1e-15)))

        contacts.append({
            "index": index, "interaction": expected[0], "surface": expected[1],
            "repeat_index": sum(1 for event in target[:index] if event[1] == expected[1]),
            "root_support": min(1.0, len(matching_roots) / n_source_rays),
            "terminal_energy_share": energy / emitted if emitted else 0.0,
            "tir_critical_margin_p10_deg": _weighted_quantile(tir_values, 0.10),
            "tir_grazing_margin_p10_deg": (
                None if not tir_values else _weighted_quantile(
                    [(90.0 - (value + math.degrees(math.asin(
                        min(1.0 / float(params["n"]), 1.0)))), weight)
                     for value, weight in tir_values], 0.10)),
            "repeat_separation_p10": _weighted_quantile(separation_values, 0.10),
            "repeat_separation_median": _weighted_quantile(separation_values, 0.50),
            "_tir_values": tir_values,
            "_separation_values": separation_values,
        })

    full_matches = [item for item in enriched if item["exact"]
                    and item["legacy"] == tuple(target)]
    full_delivered = [item for item in full_matches
                      if item["record"].get("outcome") == "delivered"]
    full_delivered_roots = {item["root"] for item in full_delivered
                            if item["root"] is not None}
    full_delivered_energy = sum(float(item["record"].get("b", 0.0))
                                for item in full_delivered)

    fingerprints = sorted(fingerprint_groups.values(),
                          key=lambda group: (-group["energy"], group["fingerprint_hash"]))
    retained = fingerprints[:_MAX_FINGERPRINTS] if _MAX_FINGERPRINTS > 0 else fingerprints
    omitted = fingerprints[len(retained):]
    for group in retained:
        group["energy_share"] = group["energy"] / emitted if emitted else 0.0

    ledger = result.get("energyLedger") or {}
    residuals = terminal_ledger_residuals(result)
    fate_shares = {name: float(ledger.get(name, 0.0)) / emitted if emitted else 0.0
                   for name in ("delivered", "escaped", "absorbed", "truncated", "unfinished")}
    return {
        "emitted": emitted,
        "fate_shares": fate_shares,
        "history_exact_energy_share": (sum(float(record.get("b", 0.0)) for record in exact)
                                       / emitted if emitted else 0.0),
        "ledger_residuals": residuals,
        "terminal_records": len(terminal), "exact_terminal_records": len(exact),
        "dominant_delivered_topology_id": dominant,
        "topologies": dense_topologies[:12],
        "historical_contacts": contacts,
        "historical_full_delivered_root_support": min(
            1.0, len(full_delivered_roots) / n_source_rays),
        "historical_full_delivered_energy_share": (
            full_delivered_energy / emitted if emitted else 0.0),
        "historical_is_dominant": bool(dense_topologies and any(
            item["topology_id"] == dominant for item in full_delivered)),
        "fingerprints": retained,
        "fingerprint_classes_total": len(fingerprints),
        "fingerprint_energy_share_omitted": (
            sum(group["energy"] for group in omitted) / emitted if emitted else 0.0),
    }


def _worker_close() -> None:
    global _RUNNER
    if _RUNNER is not None:
        _RUNNER.__exit__(None, None, None)
        _RUNNER = None


def _worker_init(dense_fields, dense_rays, perturb_fields, perturb_rays,
                 perturbations, perturb_scale, max_fingerprints, seed,
                 fixed_environment_spec=None) -> None:
    global _RUNNER, _ENV, _BASE_PROBLEM, _FIXED_ENVIRONMENT_ID
    global _DENSE_PROBLEM, _PERTURB_PROBLEM, _DENSE_FIELDS
    global _PERTURB_FIELDS, _PERTURBATIONS, _PERTURB_SCALE, _MAX_FINGERPRINTS
    global _GLOBAL_SEED
    if fixed_environment_spec is None:
        _ENV = env_config.get("fold", fov=8.0, focus_w=8.0)
        _BASE_PROBLEM = _ENV.problem
        _FIXED_ENVIRONMENT_ID = None
    else:
        _ENV = None
        _BASE_PROBLEM = problem_from_environment_spec(
            fixed_environment_spec, n_rays=int(dense_rays))
        _FIXED_ENVIRONMENT_ID = fixed_environment_spec["environment_id"]
    _DENSE_PROBLEM = replace(_BASE_PROBLEM, n_rays=int(dense_rays))
    _PERTURB_PROBLEM = replace(_BASE_PROBLEM, n_rays=int(perturb_rays))
    _DENSE_FIELDS = tuple(float(v) for v in dense_fields)
    _PERTURB_FIELDS = tuple(float(v) for v in perturb_fields)
    _PERTURBATIONS = int(perturbations)
    _PERTURB_SCALE = float(perturb_scale)
    _MAX_FINGERPRINTS = int(max_fingerprints)
    _GLOBAL_SEED = int(seed)
    _RUNNER = Runner(timeout_s=60)
    _RUNNER.__enter__()
    atexit.register(_worker_close)


def _geometry_failure(x: Sequence[float]) -> str | None:
    """Apply the matching pretrace gate in historical or fixed-environment mode."""
    if _ENV is not None:
        bad = geometry_verdict(_ENV, x)
        return None if bad is None else f"{bad.stage}: {bad.reason}"
    params = _BASE_PROBLEM.vec_to_params(x)
    return (None if _BASE_PROBLEM.is_valid(params)
            else "invalid_geometry: fixed-environment geometry/clearance gate failed")


def _trace(x: Sequence[float], problem, field_deg: float, target) -> dict:
    params = problem.vec_to_params(x)
    result, _ = _RUNNER.simulate(problem.layout(params, field_deg=field_deg).to_scene_dict())
    return _analyse_trace(result, problem, params, target, problem.n_rays)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _classify_contact(contact: Mapping[str, Any], thresholds: Mapping[str, float]) -> tuple[str, list[str]]:
    reasons = []
    if contact["mean_root_support"] < thresholds["spurious_bundle_support"]:
        reasons.append("claimed contact is supported by too few source rays")
        return "spurious", reasons
    separation = contact.get("repeat_separation_p10")
    if contact["repeat_index"] > 0 and (separation is None or
                                        separation < thresholds["repeat_separation"]):
        reasons.append("reused-surface footprints are not reliably separated")
        return "spurious", reasons
    if contact["field_persistence"] < thresholds["field_persistence"]:
        reasons.append("claimed contact is field-dependent")
        return "field-dependent", reasons
    if contact["perturbation_persistence"] < thresholds["perturbation_persistence"]:
        reasons.append("claimed contact changes under small geometry perturbations")
        return "unstable", reasons
    critical = contact.get("tir_critical_margin_p10_deg")
    grazing = contact.get("tir_grazing_margin_p10_deg")
    if critical is not None and critical < thresholds["tir_critical_margin_deg"]:
        reasons.append("TIR contact is too close to the critical angle")
        return "unstable", reasons
    if grazing is not None and grazing < thresholds["tir_grazing_margin_deg"]:
        reasons.append("TIR contact is nearly tangent")
        return "unstable", reasons
    if contact["mean_root_support"] >= thresholds["bundle_support"]:
        return "intentional", reasons
    reasons.append("contact support lies between intentional and spurious thresholds")
    return "ambiguous", reasons


def _audit_one(source: Mapping[str, Any]) -> dict:
    started = time.time()
    design_id = source["design_id"]
    x = np.asarray([source["design"][name] for name in _BASE_PROBLEM.param_names],
                   dtype=float)
    historical_text = source.get("historical_sequence") \
        or (source.get("hardened_verdict") or {}).get("full_signature") \
        or (source.get("provenance") or {}).get("historical_signature", "")
    try:
        target = _target_signature(historical_text)
    except Exception as exc:
        return {"schema_version": AUDIT_SCHEMA, "design_id": design_id,
                "fixed_environment_id": _FIXED_ENVIRONMENT_ID,
                "source_split": source.get("split"), "audit_label": "ambiguous",
                "reasons": [f"unparseable historical sequence: {exc}"],
                "historical_sequence": historical_text, "engine_seconds": 0.0,
                "fingerprints": []}

    geometry_failure = _geometry_failure(x)
    if geometry_failure is not None:
        return {
            "schema_version": AUDIT_SCHEMA, "design_id": design_id,
            "fixed_environment_id": _FIXED_ENVIRONMENT_ID,
            "source_split": source.get("split"), "audit_label": "pretrace-invalid",
            "reasons": [geometry_failure],
            "historical_sequence": historical_text,
            "source_hardened_verdict": source.get("hardened_verdict"),
            "design": source.get("design"), "source_provenance": source.get("provenance"),
            "dense_fields": [], "perturbations": [], "contacts": [],
            "engine_seconds": 0.0, "fingerprints": [],
        }

    try:
        fields = []
        fingerprints = []
        for field_deg in _DENSE_FIELDS:
            analysis = _trace(x, _DENSE_PROBLEM, field_deg, target)
            for group in analysis.pop("fingerprints"):
                fingerprints.append({
                    "schema_version": FINGERPRINT_AGGREGATE_SCHEMA,
                    "design_id": design_id, "field_deg": field_deg, **group,
                })
            analysis["field_deg"] = field_deg
            fields.append(analysis)

        # Stable seed per candidate, independent of multiprocessing order.
        seed_text = f"{_GLOBAL_SEED}|{design_id}".encode()
        local_seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "little")
        rng = np.random.default_rng(local_seed)
        lo, hi = np.asarray(_BASE_PROBLEM.lo, float), np.asarray(_BASE_PROBLEM.hi, float)
        geometry_indices = (np.arange(len(x)) if _FIXED_ENVIRONMENT_ID is not None
                            else np.array([2, 3, 4, 5, 6, 7]))
        perturb_summaries = []
        perturb_contact_support = [[] for _ in target]
        global_topology = Counter(
            field["dominant_delivered_topology_id"] for field in fields
            if field["dominant_delivered_topology_id"] is not None).most_common(1)
        global_topology_id = global_topology[0][0] if global_topology else None
        global_topology_perturb_hits = []

        for perturb_index in range(_PERTURBATIONS):
            xp = x.copy()
            delta = np.zeros_like(xp)
            delta[geometry_indices] = (rng.normal(size=len(geometry_indices))
                                       * _PERTURB_SCALE * (hi - lo)[geometry_indices])
            xp = np.clip(xp + delta, lo, hi)
            perturb_failure = _geometry_failure(xp)
            summary = {
                "index": perturb_index,
                "effective_normalized_l2": float(np.linalg.norm((xp - x) / (hi - lo))),
                "valid_geometry": perturb_failure is None, "fields": [],
            }
            if perturb_failure is not None:
                summary["failure"] = perturb_failure
                for supports in perturb_contact_support:
                    supports.extend([0.0] * len(_PERTURB_FIELDS))
                global_topology_perturb_hits.extend([False] * len(_PERTURB_FIELDS))
            else:
                for field_deg in _PERTURB_FIELDS:
                    pert = _trace(xp, _PERTURB_PROBLEM, field_deg, target)
                    supports = [contact["root_support"]
                                for contact in pert["historical_contacts"]]
                    for index, support in enumerate(supports):
                        perturb_contact_support[index].append(support)
                    hit = pert["dominant_delivered_topology_id"] == global_topology_id
                    global_topology_perturb_hits.append(hit)
                    summary["fields"].append({
                        "field_deg": field_deg,
                        "dominant_delivered_topology_id": pert["dominant_delivered_topology_id"],
                        "historical_contact_root_support": supports,
                        "historical_full_delivered_energy_share":
                            pert["historical_full_delivered_energy_share"],
                        "global_topology_persists": hit,
                    })
            perturb_summaries.append(summary)

        contacts = []
        for index, expected in enumerate(target):
            field_contacts = [field["historical_contacts"][index] for field in fields]
            root_support = [contact["root_support"] for contact in field_contacts]
            energy_support = [contact["terminal_energy_share"] for contact in field_contacts]
            all_tir = [value for contact in field_contacts for value in contact["_tir_values"]]
            all_sep = [value for contact in field_contacts
                       for value in contact["_separation_values"]]
            pert_support = perturb_contact_support[index]
            aggregate = {
                "index": index, "interaction": expected[0], "surface": expected[1],
                "repeat_index": sum(1 for event in target[:index] if event[1] == expected[1]),
                "mean_root_support": _mean(root_support),
                "minimum_root_support": min(root_support, default=0.0),
                "mean_terminal_energy_share": _mean(energy_support),
                "field_persistence": (_mean(value >= DEFAULT_THRESHOLDS["bundle_support"]
                                            for value in root_support)),
                "perturbation_persistence": (
                    _mean(value >= DEFAULT_THRESHOLDS["bundle_support"]
                          for value in pert_support) if pert_support else 1.0),
                "tir_critical_margin_p10_deg": _weighted_quantile(all_tir, 0.10),
                "tir_critical_margin_min_deg": (
                    min((value for value, _weight in all_tir), default=None)),
                "tir_grazing_margin_p10_deg": None,
                "tir_grazing_margin_min_deg": None,
                "repeat_separation_p10": _weighted_quantile(all_sep, 0.10),
                "repeat_separation_median": _weighted_quantile(all_sep, 0.50),
            }
            if all_tir:
                n_glass = float(_DENSE_PROBLEM.vec_to_params(x)["n"])
                critical = math.degrees(math.asin(min(1.0 / n_glass, 1.0)))
                grazing_values = [
                    (90.0 - (margin + critical), weight)
                    for margin, weight in all_tir
                ]
                aggregate["tir_grazing_margin_p10_deg"] = _weighted_quantile(
                    grazing_values, 0.10)
                aggregate["tir_grazing_margin_min_deg"] = min(
                    value for value, _weight in grazing_values)
            label, reasons = _classify_contact(aggregate, DEFAULT_THRESHOLDS)
            aggregate["label"], aggregate["reasons"] = label, reasons
            contacts.append(aggregate)

        historical_energy = [field["historical_full_delivered_energy_share"]
                             for field in fields]
        historical_dominance = _mean(field["historical_is_dominant"] for field in fields)
        labels = {contact["label"] for contact in contacts}
        reasons = [reason for contact in contacts for reason in contact["reasons"]]
        if any(abs(value) > DEFAULT_THRESHOLDS["ledger_tolerance"]
               for field in fields for name, value in field["ledger_residuals"].items()
               if name != "overflow_energy"):
            label = "audit-invalid"
            reasons.append("terminal records do not close against the energy ledger")
        elif _mean(historical_energy) < DEFAULT_THRESHOLDS["spurious_delivered_energy"]:
            label = "spurious"
            reasons.append("claimed full path delivers negligible energy")
        elif "spurious" in labels or historical_dominance < 0.5:
            label = "spurious"
            if historical_dominance < 0.5:
                reasons.append("claimed path is not dominant at most dense-sampled fields")
        elif "field-dependent" in labels:
            label = "field-dependent"
        elif "unstable" in labels:
            label = "unstable"
        elif labels == {"intentional"} and _mean(historical_energy) >= \
                DEFAULT_THRESHOLDS["functional_delivered_energy"]:
            label = "intentional"
        else:
            label = "ambiguous"
            reasons.append("evidence falls between the declared audit thresholds")

        for field in fields:
            for contact in field["historical_contacts"]:
                contact.pop("_tir_values", None)
                contact.pop("_separation_values", None)

        return {
            "schema_version": AUDIT_SCHEMA, "design_id": design_id,
            "fixed_environment_id": _FIXED_ENVIRONMENT_ID,
            "source_split": source.get("split"), "audit_label": label,
            "reasons": sorted(set(reasons)), "historical_sequence": historical_text,
            "source_hardened_verdict": source.get("hardened_verdict"),
            "design": source.get("design"), "source_provenance": source.get("provenance"),
            "global_topology_id": global_topology_id,
            "global_topology_field_persistence": (
                global_topology[0][1] / len(fields) if global_topology else 0.0),
            "global_topology_perturbation_persistence": _mean(global_topology_perturb_hits),
            "historical_dense_dominance_fraction": historical_dominance,
            "historical_full_delivered_energy_share_mean": _mean(historical_energy),
            "historical_full_delivered_energy_share_min": min(historical_energy, default=0.0),
            "contacts": contacts, "dense_fields": fields,
            "perturbations": perturb_summaries,
            "engine_seconds": time.time() - started, "fingerprints": fingerprints,
        }
    except Exception as exc:
        return {
            "schema_version": AUDIT_SCHEMA, "design_id": design_id,
            "fixed_environment_id": _FIXED_ENVIRONMENT_ID,
            "source_split": source.get("split"), "audit_label": "engine-error",
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "historical_sequence": historical_text,
            "source_hardened_verdict": source.get("hardened_verdict"),
            "design": source.get("design"), "source_provenance": source.get("provenance"),
            "dense_fields": [], "perturbations": [], "contacts": [],
            "engine_seconds": time.time() - started, "fingerprints": [],
        }


def _load_catalogue(directory: Path) -> tuple[list[dict], list[dict]]:
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL catalogue partitions in {directory}")
    records = []
    sources = []
    seen = set()
    for path in files:
        count = 0
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                design_id = record["design_id"]
                if design_id in seen:
                    raise ValueError(f"duplicate design_id {design_id} in {path}")
                seen.add(design_id)
                records.append(record)
                count += 1
        sources.append({"path": str(path), "rows": count, "sha256": _sha256_file(path)})
    records.sort(key=lambda record: record["design_id"])
    return records, sources


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def run(args) -> dict:
    catalogue = Path(args.catalogue).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {output}")
    records, source_files = _load_catalogue(catalogue)
    if args.limit:
        records = records[:args.limit]
    output.mkdir(parents=True)

    dense_fields = tuple(float(v) for v in np.linspace(-args.fov, args.fov,
                                                        args.dense_fields))
    perturb_fields = (-float(args.fov), 0.0, float(args.fov))
    fixed_environment_id = getattr(args, "fixed_environment_id", None)
    fixed_environment_spec = None
    fixed_specs_path = None
    if fixed_environment_id:
        fixed_specs_path = Path(args.environment_specs).resolve()
        payload = json.loads(fixed_specs_path.read_text())
        matches = [row for row in payload.get("environments", [])
                   if row.get("environment_id") == fixed_environment_id
                   and row.get("status") == "accepted"]
        if len(matches) != 1:
            raise ValueError(
                f"expected one accepted spec for {fixed_environment_id!r} in "
                f"{fixed_specs_path}, found {len(matches)}")
        fixed_environment_spec = matches[0]
    initializer_args = (dense_fields, args.dense_rays, perturb_fields,
                        args.perturb_rays, args.perturbations, args.perturb_scale,
                        args.max_fingerprints_per_field, args.seed,
                        fixed_environment_spec)
    workers = max(1, min(int(args.workers), len(records)))
    started = time.time()
    audit_path = output / "records.jsonl"
    fingerprint_path = output / "fingerprints.jsonl"
    audit_tmp = audit_path.with_name(audit_path.name + ".tmp")
    fingerprint_tmp = fingerprint_path.with_name(fingerprint_path.name + ".tmp")
    labels = Counter()
    fingerprints_written = 0

    with audit_tmp.open("w") as audit_fh, fingerprint_tmp.open("w") as fingerprint_fh:
        if workers == 1:
            _worker_init(*initializer_args)
            iterator = map(_audit_one, records)
        else:
            pool = Pool(workers, initializer=_worker_init, initargs=initializer_args)
            iterator = pool.imap(_audit_one, records, chunksize=1)
        try:
            for index, audit in enumerate(iterator, 1):
                fingerprints = audit.pop("fingerprints", [])
                for group in fingerprints:
                    fingerprint_fh.write(_canonical_json(group) + "\n")
                    fingerprints_written += 1
                audit_fh.write(_canonical_json(audit) + "\n")
                labels[audit["audit_label"]] += 1
                if index == 1 or index % 25 == 0 or index == len(records):
                    print(f"[interaction-audit] {index}/{len(records)} "
                          f"labels={dict(sorted(labels.items()))}", flush=True)
        finally:
            if workers == 1:
                _worker_close()
            else:
                pool.close()
                pool.join()
    os.replace(audit_tmp, audit_path)
    os.replace(fingerprint_tmp, fingerprint_path)

    manifest = {
        "schema_version": AUDIT_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "catalogue": str(catalogue), "source_files": source_files,
        "fixed_environment_id": fixed_environment_id,
        "environment_specs": (str(fixed_specs_path) if fixed_specs_path else None),
        "environment_specs_sha256": (
            _sha256_file(fixed_specs_path) if fixed_specs_path else None),
        "records_audited": len(records), "label_counts": dict(sorted(labels.items())),
        "config": {
            "dense_fields_deg": dense_fields, "dense_rays_per_field": args.dense_rays,
            "perturbation_fields_deg": perturb_fields,
            "perturbation_rays_per_field": args.perturb_rays,
            "geometry_perturbations": args.perturbations,
            "perturbation_scale_fraction_of_gene_range": args.perturb_scale,
            "perturbed_gene_names": ["cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b"],
            "fixed_during_perturbations": ["ray_y", "ang", "n"],
            "fingerprint_classes_retained_per_field": args.max_fingerprints_per_field,
            "seed": args.seed, "workers": workers,
        },
        "thresholds": DEFAULT_THRESHOLDS,
        "outputs": {
            "records": {"path": str(audit_path), "sha256": _sha256_file(audit_path)},
            "fingerprints": {"path": str(fingerprint_path),
                             "sha256": _sha256_file(fingerprint_path),
                             "rows": fingerprints_written},
        },
        "wall_seconds": time.time() - started,
    }
    _atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _atomic_text(output / "README.md", (
        "# Interaction audit v1\n\n"
        "Dense nominal field/pupil replay plus small geometry perturbations. Labels are "
        "evidence-based screening decisions under the exact thresholds in `manifest.json`; "
        "they are not CODE V validation or manufacturing certification. `records.jsonl` "
        "contains candidate/contact decisions, and `fingerprints.jsonl` contains aggregated "
        "canonical terminal-path classes.\n"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixed-environment-id",
                        help="audit six-DOF designs under this accepted fixed environment")
    parser.add_argument("--environment-specs",
                        default="runs/environment_pilot_20260903_v2/environment_specs.json")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fov", type=float, default=8.0)
    parser.add_argument("--dense-fields", type=int, default=17)
    parser.add_argument("--dense-rays", type=int, default=101)
    parser.add_argument("--perturbations", type=int, default=4)
    parser.add_argument("--perturb-rays", type=int, default=33)
    parser.add_argument("--perturb-scale", type=float, default=0.005)
    parser.add_argument("--max-fingerprints-per-field", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.dense_fields < 3 or args.dense_fields % 2 == 0:
        parser.error("--dense-fields must be odd and at least 3 so it contains field 0")
    if min(args.dense_rays, args.perturb_rays) < 3:
        parser.error("ray counts must be at least 3")
    if args.perturbations < 0 or args.perturb_scale < 0:
        parser.error("perturbation count and scale must be non-negative")
    manifest = run(args)
    print(json.dumps({"output": str(Path(args.out).resolve()),
                      "label_counts": manifest["label_counts"],
                      "wall_seconds": round(manifest["wall_seconds"], 3)}, indent=2))


if __name__ == "__main__":
    main()
