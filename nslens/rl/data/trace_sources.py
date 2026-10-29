"""Resolve trace genomes from their recorded archive metadata.

A nine-number vector is not self describing. In particular, the same ``fsag`` value is
millimetres in the old runs and a fraction of a face chord in the packing runs. Dataset
builders use this module so a directory name can never silently choose the interpretation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ...qd import experiments
from .. import env_config


@dataclass(frozen=True)
class TraceSource:
    trace_path: Path
    archive_path: Path
    run: str
    seed: int
    experiment_key: str
    experiment_label: str
    parameter_names: Tuple[str, ...]
    lo: Tuple[float, ...]
    hi: Tuple[float, ...]
    meta: dict

    @property
    def scale(self) -> np.ndarray:
        return np.asarray(self.hi, float) - np.asarray(self.lo, float)


def _label_registry() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key in experiments.names():
        label = experiments.get(key).label
        out.setdefault(label, []).append(key)
    return out


def compatible_sources(root: Path, *, require_relative_sag: bool = True,
                       require_normalized: bool = True,
                       canonical_layout: bool = True
                       ) -> Tuple[List[TraceSource], List[dict]]:
    """Return safe fold trace sources and explicit exclusions.

    The first policy/refiner version deliberately uses the current nine-parameter fold
    layout only. Ten-gene surfaces, perpendicular prisms, absolute-sag runs, and labels no
    longer present in the registry need separate adapters; mixing them would turn one token
    into several incompatible physical meanings.
    """
    root = Path(root)
    labels = _label_registry()
    wanted = tuple(env_config.get("fold").param_names)
    sources: List[TraceSource] = []
    excluded: List[dict] = []
    for trace in sorted(root.glob("*/seed*/trace.jsonl")):
        archive = trace.with_name("archive.json")
        reason = None
        if not archive.exists():
            reason = "missing_archive_metadata"
            meta = {}
        else:
            try:
                meta = json.loads(archive.read_text()).get("meta", {})
            except Exception:
                meta = {}
                reason = "unreadable_archive_metadata"
        label = meta.get("experiment")
        keys = labels.get(label) or []
        # A legacy tilt ablation deliberately shares a label. Prefer a unique registry
        # interpretation; ambiguous labels are excluded unless every candidate has the
        # same parameter semantics.
        key = keys[0] if keys else None
        cfg = experiments.get(key) if key else None
        if len(keys) > 1:
            signatures = {(tuple(experiments.get(k).problem.param_names),
                           bool(getattr(experiments.get(k).problem,
                                        "relative_sag", False))) for k in keys}
            if len(signatures) > 1:
                key, cfg = None, None
        names = tuple(meta.get("param_names") or ())
        if reason is None and cfg is None:
            reason = "experiment_not_in_registry"
        elif reason is None and tuple(cfg.problem.param_names) != names:
            reason = "registry_and_archive_layout_disagree"
        elif reason is None and canonical_layout and names != wanted:
            reason = "not_current_nine_parameter_fold_layout"
        elif reason is None and require_relative_sag \
                and not bool(getattr(cfg.problem, "relative_sag", False)):
            reason = "absolute_sag_semantics"
        elif reason is None and require_normalized \
                and meta.get("param_scaling") != "normalized":
            reason = "non_normalized_search"
        if reason is not None:
            excluded.append({
                "trace": str(trace), "experiment": label, "reason": reason,
                "parameter_names": list(names),
            })
            continue
        lo = tuple(float(v) for v in cfg.problem.lo)
        hi = tuple(float(v) for v in cfg.problem.hi)
        if len(lo) != len(names) or np.any(np.asarray(hi) <= np.asarray(lo)):
            excluded.append({
                "trace": str(trace), "experiment": label,
                "reason": "invalid_registry_bounds", "parameter_names": list(names),
            })
            continue
        sources.append(TraceSource(
            trace_path=trace, archive_path=archive,
            run=trace.parents[1].name, seed=int(meta.get("seed", trace.parent.name[4:])),
            experiment_key=str(key), experiment_label=str(label),
            parameter_names=names, lo=lo, hi=hi, meta=meta,
        ))
    return sources, excluded
