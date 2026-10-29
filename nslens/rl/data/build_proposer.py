"""Build Model-A datasets from certified tasks and random valid geometry.

The directed, unconditioned, and format corpora are deliberately kept separate.

    python -m nslens.rl.data.build_proposer --certified \
        runs/rl_tasks/certified_20k.json --out \
        runs/rl_datasets/proposer_v1 --n-format 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .. import env_config, prompts
from ..reward import geometry_verdict, hardened_pass
from ..server import RewardPool
from ..tasks import certify as certify_mod
from ..tasks import splits as split_mod
from .schema import PROPOSER_SCHEMA, validate_proposer


def _hash(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts).encode()
    return hashlib.sha256(raw).hexdigest()


def _seen_validation(spec_id: str, seed: int, fraction: float = 0.1) -> bool:
    bucket = int(_hash("seen-val", seed, spec_id)[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def _anchor_id(cert: certify_mod.Certified) -> str:
    """Physical-design lineage shared by specs that reused the same warm-start elite."""
    return _hash("anchor", *(round(float(v), 6) for v in cert.genotype))[:20]


def _design(env, genotype: Sequence[float], digits: int = 4) -> Dict[str, float]:
    return {n: round(float(v), digits) for n, v in zip(env.param_names, genotype)}


def _record(*, example_id: str, kind: str, split: str, payload: dict,
            design: dict, labels: dict, provenance: dict) -> dict:
    block = "<design>" + json.dumps(design, separators=(",", ":")) + "</design>"
    rec = {
        "schema_version": PROPOSER_SCHEMA,
        "example_id": example_id,
        "dataset_kind": kind,
        "split": split,
        "messages": prompts.messages(payload, block),
        "input": payload,
        "target": {"design": design, "design_block": block},
        "labels": labels,
        "provenance": provenance,
    }
    errors = validate_proposer(rec)
    if errors:
        raise ValueError(f"invalid generated record {example_id}: {errors}")
    return rec


def _split_names(certs: Sequence[certify_mod.Certified], seed: int) -> Dict[str, str]:
    partition = split_mod.split(certs, seed=seed)
    return {c.spec_id: name for name, rows in (
        ("train", partition.train), ("test_cell", partition.test_cell),
        ("test_env", partition.test_env)) for c in rows}


def certified_records(certs: Sequence[certify_mod.Certified], *, split_seed: int = 0,
                      verdicts: Dict[str, object] | None = None
                      ) -> Dict[str, List[dict]]:
    """Directed and unconditioned examples with topology-safe held-out splits."""
    split_of = _split_names(certs, split_seed)
    out = {"directed": [], "unconditioned": []}
    for cert in certs:
        spec = cert.to_spec()
        env = env_config.get(cert.env, fov=cert.fov, focus_w=cert.focus_w)
        split = split_of[cert.spec_id]
        anchor_id = _anchor_id(cert)
        if split == "train" and _seen_validation(anchor_id, split_seed):
            split = "validation_seen"
        design = _design(env, cert.genotype)
        rounded = np.asarray([design[n] for n in env.param_names], float)
        bad = geometry_verdict(env, rounded)
        if bad is not None:
            continue
        observed = (verdicts or {}).get(cert.spec_id)
        labels = {
            "topology_index": int(observed.topology if observed is not None
                                  else cert.topology),
            "topology_label": spec.topology_label,
            "full_signature": (observed.signature if observed is not None
                               else cert.signature),
            "thinness_bin": int(observed.thinness_bin if observed is not None
                                else cert.thinness_bin),
            "certified_achievable": float(cert.achievable),
            "critical_margin_deg": float(observed.crit_margin if observed is not None
                                         else cert.crit_margin),
            "difficulty": float(cert.difficulty),
        }
        if observed is not None:
            labels.update(observed_fitness=float(observed.fitness),
                          observed_per_field=list(observed.per_field))
        provenance = {
            "source": "certified_cma_me_probe",
            "spec_id": cert.spec_id,
            "anchor_id": anchor_id,
            "from_seed": bool(cert.from_seed),
            "seeds_hit": int(cert.seeds_hit),
            "seeds_run": int(cert.seeds_run),
            "serialized_target_retraced": observed is not None,
        }
        for mode, kind in (("directed", "directed_seed"),
                           ("unconditioned", "unconditioned_archive_seed")):
            payload = prompts.spec_payload(spec, env, mode=mode)
            out[mode].append(_record(
                example_id=f"{mode}:{cert.spec_id}:{_hash(design)[:12]}",
                kind=kind, split=split, payload=payload, design=design,
                labels=labels, provenance=provenance))
    return out


def verify_certified_anchors(certs: Sequence[certify_mod.Certified], *,
                             workers: int | None = None):
    """Re-trace the exact four-decimal anchor that will appear in SFT.

    Certification happened on higher-precision genotypes. Geometry-only checking after
    serialization is not enough: rounding could in principle change a marginal TIR path,
    exit focus, or requested cell. Future task files therefore cannot enter a generated
    corpus unless the serialized target itself clears every hardened gate and still lands
    in its certified topology/thinness cell.
    """
    items = []
    for cert in certs:
        env = env_config.get(cert.env, fov=cert.fov, focus_w=cert.focus_w)
        design = _design(env, cert.genotype)
        x = [design[n] for n in env.param_names]
        items.append(((cert.env, cert.fov, cert.focus_w), x))
    with RewardPool(workers=workers) as pool:
        verdict_list = pool.verdicts(items)
    kept, verdicts = [], {}
    reasons = Counter()
    for cert, verdict in zip(certs, verdict_list):
        if not hardened_pass(verdict):
            reasons[f"hard_gate:{verdict.stage}"] += 1
            continue
        if verdict.topology != cert.topology:
            reasons["requested_topology_changed_after_rounding"] += 1
            continue
        if verdict.thinness_bin != cert.thinness_bin:
            reasons["thinness_cell_changed_after_rounding"] += 1
            continue
        kept.append(cert)
        verdicts[cert.spec_id] = verdict
    return kept, verdicts, {
        "input": len(certs), "kept": len(kept), "dropped": len(certs) - len(kept),
        "drop_reasons": dict(reasons), "serialization_digits": 4,
        "required": ["geometry", "requested_topology", "requested_thinness_cell",
                     "clean_all", "exit_ok_all", "margin_ok"],
    }


def perturbation_records(certs: Sequence[certify_mod.Certified], *,
                         per_cert: int = 8, split_seed: int = 0,
                         sample_seed: int = 0, workers: int | None = None,
                         min_relative_fitness: float = prompts.TRAIN_TARGET_FRACTION
                         ) -> Dict[str, List[dict]]:
    """Ray-traced basin augmentation around certified anchors.

    Targets are rounded before tracing, so the JSON emitted to SFT is exactly the vector
    that earned the label. A perturbation is kept only when it stays in the requested
    topology and thinness cell, clears every hardened gate, and meets the prompt's quality
    threshold. These are independent training examples, not duplicate copies of an elite.
    """
    if per_cert <= 0:
        return {"directed": [], "unconditioned": []}
    split_of = _split_names(certs, split_seed)
    rng = np.random.default_rng(sample_seed)
    sigmas = (0.002, 0.005, 0.010)
    pending = []
    items = []
    for cert in certs:
        spec = cert.to_spec()
        env = env_config.get(cert.env, fov=cert.fov, focus_w=cert.focus_w)
        lo = np.asarray(env.problem.lo, float)
        hi = np.asarray(env.problem.hi, float)
        anchor = np.asarray(cert.genotype, float)
        for j in range(per_cert):
            sigma = sigmas[j % len(sigmas)]
            x = np.clip(anchor + rng.normal(size=len(anchor)) * sigma * (hi - lo), lo, hi)
            design = _design(env, x)
            rounded = np.asarray([design[n] for n in env.param_names], float)
            if geometry_verdict(env, rounded) is not None:
                continue
            pending.append((cert, spec, env, design, rounded, sigma, j))
            items.append(((cert.env, cert.fov, cert.focus_w), rounded))

    with RewardPool(workers=workers) as pool:
        verdicts = pool.verdicts(items)

    out = {"directed": [], "unconditioned": []}
    seen = set()
    for meta, verdict in zip(pending, verdicts):
        cert, spec, env, design, rounded, sigma, j = meta
        if not verdict.scored:
            continue
        gates = verdict.gates or {}
        if verdict.topology != cert.topology or verdict.thinness_bin != cert.thinness_bin:
            continue
        if verdict.fitness < min_relative_fitness * cert.achievable:
            continue
        if not hardened_pass(verdict):
            continue
        key = tuple(float(v) for v in rounded)
        if key in seen:
            continue
        seen.add(key)
        split = split_of[cert.spec_id]
        anchor_id = _anchor_id(cert)
        if split == "train" and _seen_validation(anchor_id, split_seed):
            split = "validation_seen"
        labels = {
            "topology_index": int(verdict.topology),
            "topology_label": spec.topology_label,
            "full_signature": verdict.signature,
            "thinness_bin": int(verdict.thinness_bin),
            "certified_achievable": float(cert.achievable),
            "observed_fitness": float(verdict.fitness),
            "observed_per_field": list(verdict.per_field),
            "critical_margin_deg": float(verdict.crit_margin),
            "difficulty": float(cert.difficulty),
        }
        provenance = {
            "source": "certified_anchor_local_perturbation",
            "spec_id": cert.spec_id,
            "anchor_id": anchor_id,
            "anchor_from_seed": bool(cert.from_seed),
            "perturbation_sigma_fraction_of_range": sigma,
            "perturbation_index": j,
            "sample_seed": sample_seed,
        }
        for mode, kind in (("directed", "directed_seed"),
                           ("unconditioned", "unconditioned_archive_seed")):
            payload = prompts.spec_payload(spec, env, mode=mode)
            out[mode].append(_record(
                example_id=f"{mode}:perturb:{cert.spec_id}:{_hash(key)[:12]}",
                kind=kind, split=split, payload=payload, design=design,
                labels=labels, provenance=provenance))
    return out


def format_records(n: int, *, seed: int = 0, env_name: str = "fold") -> List[dict]:
    """Uniform-random, topology-free, buildable geometry records (no engine calls)."""
    env = env_config.get(env_name)
    lo = np.asarray(env.problem.lo, float)
    hi = np.asarray(env.problem.hi, float)
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    attempts = 0
    max_attempts = max(1000, n * 100)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(lo, hi)
        design = _design(env, x)
        key = tuple(design[n] for n in env.param_names)
        if key in seen:
            continue
        rounded = np.asarray(key, float)
        if geometry_verdict(env, rounded) is not None:
            continue
        seen.add(key)
        ident = _hash("format", seed, key)
        bucket = int(ident[:8], 16) % 20
        split = "test_format" if bucket == 0 else \
            ("validation_seen" if bucket == 1 else "train")
        out.append(_record(
            example_id=f"format:{ident[:20]}", kind="format_validity", split=split,
            payload=prompts.format_payload(env), design=design,
            labels={"geometry_buildable": True, "topology_label_supplied": False,
                    "performance_label_supplied": False},
            provenance={"source": "uniform_random_geometry", "random_seed": seed,
                        "attempt": attempts}))
    if len(out) != n:
        raise RuntimeError(f"generated only {len(out)}/{n} format examples in "
                           f"{attempts} attempts")
    return out


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def build(certified_path: Path, out_dir: Path, *, n_format: int = 2000,
          augment_per_cert: int = 0, workers: int | None = None,
          seed: int = 0) -> dict:
    certs = certify_mod.load(str(certified_path))
    certs, anchor_verdicts, anchor_audit = verify_certified_anchors(
        certs, workers=workers)
    corpora = certified_records(certs, split_seed=seed, verdicts=anchor_verdicts)
    augmented = perturbation_records(
        certs, per_cert=augment_per_cert, split_seed=seed,
        sample_seed=seed + 1000, workers=workers)
    corpora["directed"].extend(augmented["directed"])
    corpora["unconditioned"].extend(augmented["unconditioned"])
    corpora["format"] = format_records(n_format, seed=seed)

    # Perturbations around one certified design are correlated measurements of one basin,
    # not independent discoveries. Give every certified lineage unit total mass so a cell
    # with eight surviving perturbations cannot outweigh a cell whose basin is narrower.
    lineage_counts: Dict[str, Counter] = {}
    for corpus, rows in corpora.items():
        keys = [str((r.get("provenance") or {}).get("spec_id") or r["example_id"])
                for r in rows]
        lineage_counts[corpus] = Counter(keys)
        for r, key in zip(rows, keys):
            r["sample_weight"] = 1.0 / lineage_counts[corpus][key]
            errors = validate_proposer(r)
            if errors:
                raise ValueError(f"weighted record {r['example_id']} invalid: {errors}")

    counts: Dict[str, Dict[str, int]] = {}
    topology_by_split = defaultdict(set)
    for corpus, rows in corpora.items():
        counts[corpus] = dict(Counter(r["split"] for r in rows))
        for r in rows:
            topo = (r.get("labels") or {}).get("topology_index")
            if topo is not None:
                topology_by_split[r["split"]].add(int(topo))
        by_split = defaultdict(list)
        for r in rows:
            by_split[r["split"]].append(r)
        for split, part in by_split.items():
            _write_jsonl(out_dir / corpus / f"{split}.jsonl", part)

    train_topo = topology_by_split.get("train", set()) | \
        topology_by_split.get("validation_seen", set())
    test_topo = topology_by_split.get("test_cell", set())
    overlap = sorted(train_topo & test_topo)
    if overlap:
        raise AssertionError(f"topology leakage into test_cell: {overlap}")

    source_bytes = certified_path.read_bytes()
    manifest = {
        "schema_version": PROPOSER_SCHEMA,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(certified_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "split_seed": int(seed),
        "augmentation": {
            "requested_per_certified_anchor": int(augment_per_cert),
            "kept_directed_records": len(augmented["directed"]),
            "quality_fraction_of_certified_achievable":
                prompts.TRAIN_TARGET_FRACTION,
            "hardened_gates_required": ["clean_all", "exit_ok_all", "margin_ok"],
        },
        "certified_anchor_revalidation": anchor_audit,
        "counts": counts,
        "effective_lineages": {k: len(v) for k, v in lineage_counts.items()},
        "physical_anchor_lineages": {
            corpus: len({str((r.get("provenance") or {}).get("anchor_id")
                            or (r.get("provenance") or {}).get("spec_id")
                            or r["example_id"]) for r in rows})
            for corpus, rows in corpora.items()},
        "sample_weight_sums": {
            corpus: sum(float(r["sample_weight"]) for r in rows)
            for corpus, rows in corpora.items()},
        "topologies": {k: sorted(v) for k, v in topology_by_split.items()},
        "checks": {"train_test_cell_topology_overlap": overlap,
                   "all_records_validated": True},
        "intended_use": {
            "format": "syntax and cheap geometry-validity scaffold; no topology labels",
            "directed": "requested known-topology starting-point generation",
            "unconditioned": "archive distillation; not evidence of topology emergence",
        },
        "limitations": [
            "Certified records currently cover one fixed-length fold environment.",
            "No CODE V outcomes are present in this dataset.",
            "Every four-decimal certified anchor and augmentation is re-traced before it "
            "becomes a positive target.",
            "validation_seen shares topology classes with train and is not an OOD test.",
            "Perturbations share a certified lineage; sample_weight makes each lineage "
            "contribute unit total mass.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--certified", default="runs/rl_tasks/certified_20k.json")
    ap.add_argument("--out", default="runs/rl_datasets/proposer_v1")
    ap.add_argument("--n-format", type=int, default=2000)
    ap.add_argument("--augment-per-cert", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    manifest = build(Path(args.certified), Path(args.out),
                     n_format=args.n_format,
                     augment_per_cert=args.augment_per_cert,
                     workers=args.workers, seed=args.seed)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
