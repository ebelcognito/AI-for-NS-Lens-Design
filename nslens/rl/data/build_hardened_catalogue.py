"""Select diverse historical designs and re-score them under one hardened yardstick.

Historical trace fitness is used only to *select candidates within an observed signature*.
Every exported quality label is freshly measured at nine fields under the current fold RL
environment. This turns mixed old runs into one comparable catalogue without CODE V.

    python -m nslens.rl.data.build_hardened_catalogue --runs runs \
      --out runs/rl_datasets/hardened_catalogue_v1 --workers 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ...qd.descriptors import tir_face_label
from .. import env_config
from ..server import RewardPool
from ..reward import hardened_pass
from .schema import HARDENED_DESIGN_SCHEMA, validate_hardened_design
from .trace_sources import TraceSource, compatible_sources


def _hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def yardstick_payload() -> dict:
    env = env_config.get("fold", fov=8.0, focus_w=8.0)
    cfg = env.cfg
    payload = {
        "env": "fold", "fov_half_angle_deg": env.fov,
        "fields_deg": list(env.fields), "focus_window_half_width": 8.0,
        "n_rays": int(env.problem.n_rays),
        "parameter_names": list(env.param_names), "parameter_bounds": env.bounds(),
        "relative_sag": bool(env.problem.relative_sag),
        "max_sag_ratio": float(env.problem.max_sag_ratio),
        "detector_clearance": float(env.problem.det_clearance),
        "source_clearance": float(env.source_clearance),
        "min_critical_margin_deg": float(env.min_crit_margin),
        "field_aggregate": cfg.field_aggregate,
        "convergence_gate": cfg.convergence_gate,
        "gate_all_fields": bool(cfg.gate_all_fields),
        "require_monotone_convergence": bool(cfg.require_monotone_convergence),
        "require_exit_convergence": bool(cfg.require_exit_convergence),
        "descriptors": list(cfg.descriptors), "bc_bins": list(cfg.bc_bins),
        "bc_bounds": [list(v) for v in cfg.bc_bounds],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["config_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def _historical_key(row: dict) -> Tuple[int, float]:
    status = row.get("status", row.get("historical_status"))
    tier = {"none": 1, "improved": 2, "new": 3}.get(status, 0)
    value = float(row.get("delta", row.get("historical_delta"))
                  or row.get("fitness", row.get("historical_fitness")) or 0.0)
    return tier, value


def select_candidates(sources: Sequence[TraceSource], *, per_signature: int = 12,
                      max_candidates: int = 6000) -> Tuple[List[dict], List[dict]]:
    """Dedupe six-decimal genotypes, then sample round-robin across ray signatures."""
    unique: Dict[tuple, dict] = {}
    audits = []
    for i, source in enumerate(sources, 1):
        print(f"[catalogue:scan] {i}/{len(sources)} {source.run}/seed{source.seed}",
              flush=True)
        digest = hashlib.sha256()
        seen = kept = malformed = 0
        with source.trace_path.open("rb") as fh:
            for raw in fh:
                digest.update(raw)
                try:
                    row = json.loads(raw)
                except Exception:
                    malformed += 1
                    continue
                seen += 1
                info = row.get("info") or {}
                signature = info.get("dom_signature")
                genotype = row.get("genotype")
                if not signature or row.get("fitness") is None \
                        or len(genotype or ()) != len(source.parameter_names):
                    continue
                key = tuple(round(float(v), 6) for v in genotype)
                rec = {
                    "genotype": list(key), "historical_signature": str(signature),
                    "historical_fitness": float(row["fitness"]),
                    "historical_delta": (float(row["delta"])
                                         if row.get("delta") is not None else None),
                    "historical_status": row.get("status"),
                    "source_run": source.run, "source_seed": source.seed,
                    "source_experiment": source.experiment_label,
                    "source_trace": str(source.trace_path),
                    "source_eval": int(row.get("eval", -1)),
                }
                if key not in unique or _historical_key(row) > _historical_key(unique[key]):
                    unique[key] = rec
                kept += 1
        audits.append({
            "trace": str(source.trace_path), "sha256": digest.hexdigest(),
            "rows": seen, "historically_scored_rows": kept, "malformed": malformed,
        })

    by_signature = defaultdict(list)
    winners = []
    for rec in unique.values():
        by_signature[rec["historical_signature"]].append(rec)
        if rec["historical_status"] in ("new", "improved"):
            winners.append(rec)
    for rows in by_signature.values():
        rows.sort(key=lambda r: -r["historical_fitness"])

    # Start with every archive-changing action, then fill in a round-robin over signatures
    # so common paths cannot crowd rare paths out of the expensive hardened re-score.
    chosen: Dict[tuple, dict] = {
        tuple(r["genotype"]): r for r in sorted(
            winners, key=lambda r: (-_historical_key(r)[0],
                                    -r["historical_fitness"]))
    }
    signature_order = sorted(by_signature, key=lambda s: _hash("signature", s))
    for depth in range(per_signature):
        for signature in signature_order:
            rows = by_signature[signature]
            if depth < len(rows):
                rec = rows[depth]
                chosen.setdefault(tuple(rec["genotype"]), rec)
                if max_candidates > 0 and len(chosen) >= max_candidates:
                    break
        if max_candidates > 0 and len(chosen) >= max_candidates:
            break
    selected = list(chosen.values())
    if max_candidates > 0:
        selected = selected[:max_candidates]
    return selected, audits


def _topology_splits(records: Sequence[dict], seed: int = 0) -> Dict[int, str]:
    sizes = Counter(r["hardened_verdict"]["topology_index"] for r in records)
    ordered = sorted(sizes, key=lambda t: _hash("topology-split", seed, t))
    total = sum(sizes.values())

    def take(target: float, available: List[int]) -> List[int]:
        out, n = [], 0
        for topology in available:
            if n >= target and out:
                break
            out.append(topology)
            n += sizes[topology]
        return out

    test = take(0.15 * total, ordered)
    remaining = [t for t in ordered if t not in test]
    validation = take(0.10 * total, remaining)
    return {t: ("test_topology" if t in test else
                "validation_topology" if t in validation else "train") for t in sizes}


def _record(candidate: dict, verdict, yardstick: dict, env, *, min_fitness: float) -> dict:
    design = {n: float(v) for n, v in zip(env.param_names, candidate["genotype"])}
    gates = _jsonable(verdict.gates or {})
    eligible = bool(hardened_pass(verdict) and verdict.fitness >= min_fitness)
    hv = {
        "stage": verdict.stage, "reason": verdict.reason,
        "fitness": float(verdict.fitness),
        "per_field": [float(v) for v in verdict.per_field],
        "bc": list(verdict.bc) if verdict.bc is not None else None,
        "topology_index": verdict.topology,
        "topology_label": (tir_face_label(verdict.topology)
                           if verdict.topology is not None else None),
        "thinness_bin": verdict.thinness_bin, "thinness_T": verdict.thinness_T,
        "full_signature": verdict.signature, "spot": verdict.spot,
        "critical_margin_deg": verdict.crit_margin,
        "gates": gates, "info": _jsonable(verdict.info or {}),
        "training_eligible": eligible,
    }
    rec = {
        "schema_version": HARDENED_DESIGN_SCHEMA,
        "design_id": "hard:" + _hash(candidate["genotype"],
                                        yardstick["config_fingerprint"])[:24],
        "split": "rejected", "design": design,
        "parameter_bounds": env.bounds(), "yardstick": yardstick,
        "hardened_verdict": hv,
        "provenance": {
            "selection": "diverse historical signature candidate; all labels remeasured",
            **candidate,
        },
    }
    return rec


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def build(runs_root: Path, out_dir: Path, *, workers: int | None = None,
          per_signature: int = 12, max_candidates: int = 6000,
          min_fitness: float = 0.05, seed: int = 0) -> dict:
    sources, excluded = compatible_sources(runs_root)
    candidates, source_audit = select_candidates(
        sources, per_signature=per_signature, max_candidates=max_candidates)
    if not candidates:
        raise RuntimeError("no compatible, historically scored candidates")
    env = env_config.get("fold", fov=8.0, focus_w=8.0)
    yardstick = yardstick_payload()
    items = [(('fold', 8.0, 8.0), c["genotype"]) for c in candidates]
    print(f"[catalogue:rescore] {len(items)} candidates, {workers or 'auto'} workers",
          flush=True)
    with RewardPool(workers=workers) as pool:
        verdicts = pool.verdicts(items)
        pool_stats = {
            "unique_designs": pool.stats.designs,
            "engine_seconds": pool.stats.engine_seconds,
            "designs_per_second": pool.stats.designs_per_second,
        }
    records = [_record(c, v, yardstick, env, min_fitness=min_fitness)
               for c, v in zip(candidates, verdicts)]
    eligible = [r for r in records if r["hardened_verdict"]["training_eligible"]]
    split_of = _topology_splits(eligible, seed=seed) if eligible else {}
    for rec in eligible:
        rec["split"] = split_of[rec["hardened_verdict"]["topology_index"]]
    for rec in records:
        errors = validate_hardened_design(rec)
        if errors:
            raise ValueError(f"invalid hardened design {rec['design_id']}: {errors}")

    by_split = defaultdict(list)
    for rec in records:
        by_split[rec["split"]].append(rec)
    counts = {split: _write_jsonl(out_dir / f"{split}.jsonl", rows)
              for split, rows in by_split.items()}
    topologies = {split: sorted({r["hardened_verdict"]["topology_index"]
                                 for r in rows
                                 if r["hardened_verdict"]["topology_index"] is not None})
                  for split, rows in by_split.items()}
    train_top = set(topologies.get("train", []))
    held_top = set(topologies.get("validation_topology", [])) \
        | set(topologies.get("test_topology", []))
    if train_top & held_top:
        raise AssertionError("topology leakage in hardened catalogue")
    rejection_reasons = Counter()
    for rec in records:
        hv = rec["hardened_verdict"]
        if hv["training_eligible"]:
            continue
        if hv["stage"] != "scored":
            rejection_reasons[f"stage:{hv['stage']}"] += 1
            continue
        if float(hv["fitness"]) < min_fitness:
            rejection_reasons["fitness_below_threshold"] += 1
        for gate in ("clean_all", "exit_ok_all", "margin_ok"):
            if not (hv.get("gates") or {}).get(gate):
                rejection_reasons[f"gate:{gate}"] += 1
    fit_values = [float(r["hardened_verdict"]["fitness"]) for r in eligible]
    manifest = {
        "schema_version": HARDENED_DESIGN_SCHEMA,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "yardstick": yardstick, "source_root": str(runs_root),
        "source_traces_included": len(sources),
        "source_traces_excluded": len(excluded),
        "candidate_selection": {
            "per_historical_signature": per_signature,
            "max_candidates": max_candidates,
            "selected": len(candidates),
            "archive_changing_candidates_prioritized": True,
        },
        "training_gate": {
            "minimum_hardened_fitness": min_fitness,
            "required": ["stage=scored", "clean_all", "exit_ok_all", "margin_ok"],
        },
        "counts": counts,
        "verdict_stages": dict(Counter(v.stage for v in verdicts)),
        "rejection_reasons_nonexclusive": dict(rejection_reasons),
        "eligible_fitness_percentiles": ({
            str(q): float(np.percentile(fit_values, q))
            for q in (0, 10, 25, 50, 75, 90, 100)} if fit_values else {}),
        "eligible_topologies": topologies,
        "eligible_full_signatures": len({
            r["hardened_verdict"]["full_signature"] for r in eligible}),
        "pool_stats": pool_stats,
        "checks": {
            "single_hardened_config": True,
            "config_fingerprint": yardstick["config_fingerprint"],
            "train_heldout_topology_overlap": sorted(train_top & held_top),
            "all_records_validated": True,
        },
        "source_audit": source_audit,
        "excluded_source_reasons": dict(Counter(e["reason"] for e in excluded)),
        "limitations": [
            "This is a local 2D ray-tracer label, not a CODE V validation label.",
            "Only the current nine-parameter relative-sag fold family is represented.",
            "Historical scores select candidates but never become exported quality labels.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/rl_datasets/hardened_catalogue_v1")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--per-signature", type=int, default=12)
    ap.add_argument("--max-candidates", type=int, default=6000)
    ap.add_argument("--min-fitness", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.runs), Path(a.out), workers=a.workers,
                           per_signature=a.per_signature,
                           max_candidates=a.max_candidates,
                           min_fitness=a.min_fitness, seed=a.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
