"""Build paired proxy/full-beam traces around certified optical designs.

The generator deliberately does not require thousands of high-quality designs.  It uses
certified designs as basin anchors, creates deterministic perturbations at several radii,
labels every sample with cheap fidelities, and spends full nine-field labels only on a
small training subset plus the held-out evaluation set.

Example::

    python3 -m nslens.rl.data.build_multifidelity \
        --n 512 --hf-train 64 --out runs/multifidelity_pilot_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ...engine import Runner
from .. import env_config, reward
from ..multifidelity import FIDELITIES, Fidelity, ladder
from ..tasks.certify import Certified, load as load_certified
from ..tasks.splits import split as split_certified

SCHEMA = "nslens.multifidelity_trace.v1"
DEFAULT_CERTIFIED = "runs/rl_tasks/certified_20k.json"
DEFAULT_SCALES = (0.0, 0.003, 0.01, 0.03, 0.10)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(*values) -> str:
    return hashlib.sha256("|".join(_canonical(v) for v in values).encode()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value)
    os.replace(tmp, path)


def _label(verdict: reward.Verdict, fidelity: Fidelity) -> dict:
    # Gate values from a one-ray trace are diagnostic only.  ``acceptance_authority`` is
    # explicit so downstream code cannot silently train a hard-pass classifier from them.
    return {
        "stage": verdict.stage,
        "fitness": float(verdict.fitness),
        "topology_index": verdict.topology,
        "thinness_bin": verdict.thinness_bin,
        "signature": verdict.signature,
        "per_field": list(verdict.per_field),
        "hard_pass": (bool(reward.hardened_pass(verdict))
                      if fidelity.acceptance_authority else None),
        "acceptance_authority": fidelity.acceptance_authority,
        "gates": verdict.gates,
    }


def _normalise(x: Sequence[float], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (np.asarray(x, float) - lo) / (hi - lo)


def _sample(anchor: Certified, index: int, rng: np.random.Generator,
            scales: Sequence[float], uniform_fraction: float) -> tuple[np.ndarray, dict]:
    env = env_config.get(anchor.env, fov=anchor.fov, focus_w=anchor.focus_w)
    lo, hi = np.asarray(env.problem.lo, float), np.asarray(env.problem.hi, float)
    if rng.random() < uniform_fraction:
        z = rng.uniform(0.0, 1.0, len(lo))
        source = {"kind": "uniform", "normalized_radius": None}
    else:
        scale = float(scales[index % len(scales)])
        z0 = _normalise(anchor.genotype, lo, hi)
        z = np.clip(z0 + rng.normal(0.0, scale, len(lo)), 0.0, 1.0)
        source = {"kind": "anchor_perturbation", "normalized_radius": scale}
    return lo + z * (hi - lo), source


def make_samples(certs: Sequence[Certified], n: int, *, seed: int,
                 scales: Sequence[float] = DEFAULT_SCALES,
                 uniform_fraction: float = 0.10) -> list[dict]:
    """Create deterministic sample descriptions before spending any engine calls."""
    if n < 1:
        raise ValueError("n must be positive")
    if not certs:
        raise ValueError("no certified anchors")
    if not scales or any(s < 0 for s in scales):
        raise ValueError("scales must contain non-negative values")
    if not 0.0 <= uniform_fraction <= 1.0:
        raise ValueError("uniform_fraction must lie in [0, 1]")

    partition = split_certified(certs, seed=seed)
    groups = {
        "train": partition.train,
        "test_cell": partition.test_cell,
        "test_env": partition.test_env,
    }
    nonempty = [(name, rows) for name, rows in groups.items() if rows]
    # 80/20 when only train/test_cell exist; preserve any test_env anchors by allocating
    # groups evenly inside the held-out fifth.  Sampling within a split is round-robin so
    # rare variants are not erased by their anchor count.
    weights = {"train": 0.8}
    held = [name for name, rows in nonempty if name != "train"]
    for name in held:
        weights[name] = 0.2 / len(held)
    if not groups["train"]:
        raise ValueError("certified split has no training anchors")

    counts = {name: int(round(n * weights.get(name, 0.0))) for name, _ in nonempty}
    counts["train"] += n - sum(counts.values())
    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    global_index = 0
    for split_name, anchors in nonempty:
        ordered = sorted(anchors, key=lambda c: c.spec_id)
        rng.shuffle(ordered)
        for local_index in range(counts[split_name]):
            anchor = ordered[local_index % len(ordered)]
            x, source = _sample(anchor, global_index, rng, scales, uniform_fraction)
            env = env_config.get(anchor.env, fov=anchor.fov, focus_w=anchor.focus_w)
            lo, hi = np.asarray(env.problem.lo, float), np.asarray(env.problem.hi, float)
            sample_id = "mf:" + _hash(
                seed, split_name, global_index, anchor.spec_id,
                [round(float(v), 12) for v in x])[:24]
            samples.append({
                "sample_id": sample_id,
                "split": split_name,
                "anchor": anchor,
                "genotype": x,
                "normalized_genotype": _normalise(x, lo, hi),
                "source": {**source, "anchor_spec_id": anchor.spec_id},
            })
            global_index += 1
    return samples


def _trace_one(runner: Runner, sample: Mapping, include_full: bool) -> dict:
    anchor: Certified = sample["anchor"]
    base = env_config.get(anchor.env, fov=anchor.fov, focus_w=anchor.focus_w)
    labels = {}
    for fidelity, fenv in ladder(base):
        if fidelity.acceptance_authority and not include_full:
            continue
        labels[fidelity.name] = _label(
            reward.evaluate(runner, fenv, sample["genotype"]), fidelity)
    return {
        "schema_version": SCHEMA,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "environment": {
            "name": anchor.env,
            "fov_half_angle_deg": float(anchor.fov),
            "focus_window_half_width_mm": float(anchor.focus_w),
            "parameter_names": list(base.param_names),
        },
        "genotype": [float(v) for v in sample["genotype"]],
        "normalized_genotype": [float(v) for v in sample["normalized_genotype"]],
        "source": sample["source"],
        "labels": labels,
    }


def build(certs: Sequence[Certified], *, n: int, hf_train: int, seed: int,
          scales: Sequence[float] = DEFAULT_SCALES,
          uniform_fraction: float = 0.10, log=print) -> tuple[list[dict], dict]:
    samples = make_samples(certs, n, seed=seed, scales=scales,
                           uniform_fraction=uniform_fraction)
    train_ids = [s["sample_id"] for s in samples if s["split"] == "train"]
    if hf_train < 1 or hf_train > len(train_ids):
        raise ValueError(f"hf_train must be in [1, {len(train_ids)}]")
    # Random, but frozen by seed.  All held-out rows receive authority labels because they
    # are the yardstick; only ``hf_train`` training rows spend those labels.
    rng = np.random.default_rng(seed + 1)
    hf_ids = set(rng.choice(train_ids, size=hf_train, replace=False).tolist())

    rows = []
    t0 = time.time()
    with Runner(timeout_s=20) as runner:
        for i, sample in enumerate(samples, 1):
            include_full = sample["split"] != "train" or sample["sample_id"] in hf_ids
            rows.append(_trace_one(runner, sample, include_full))
            if i % 100 == 0 or i == len(samples):
                log(f"[multifidelity] traced {i}/{len(samples)} samples")

    label_counts = Counter(name for row in rows for name in row["labels"])
    full_rows = [r for r in rows if "full_multifov" in r["labels"]]
    manifest = {
        "schema_version": "nslens.multifidelity_manifest.v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "n": len(rows),
        "hf_train": hf_train,
        "splits": dict(Counter(r["split"] for r in rows)),
        "label_counts": dict(label_counts),
        "fidelities": [asdict(f) for f in FIDELITIES],
        "sampling": {"normalized_scales": list(scales),
                     "uniform_fraction": uniform_fraction},
        "nominal_engine_calls": sum(
            f.nominal_engine_calls * label_counts[f.name] for f in FIDELITIES),
        "full_hard_passes": sum(
            bool(r["labels"]["full_multifov"]["hard_pass"]) for r in full_rows),
        "elapsed_s": time.time() - t0,
    }
    return rows, manifest


def write(rows: Iterable[dict], manifest: dict, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(out_dir / "traces.jsonl", "".join(_canonical(r) + "\n" for r in rows))
    _atomic_text(out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def main(argv=None) -> int:  # pragma: no cover - exercised through the CLI smoke run
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--certified", default=DEFAULT_CERTIFIED)
    ap.add_argument("--out", default="runs/multifidelity_pilot_v1")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--hf-train", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    ap.add_argument("--uniform-fraction", type=float, default=0.10)
    args = ap.parse_args(argv)
    certs = load_certified(args.certified)
    rows, manifest = build(certs, n=args.n, hf_train=args.hf_train, seed=args.seed,
                           scales=args.scales, uniform_fraction=args.uniform_fraction)
    write(rows, manifest, Path(args.out))
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
