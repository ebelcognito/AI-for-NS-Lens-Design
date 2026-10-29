"""Train a dedicated same-parent action ranker for fixed-environment refinement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from .fixed_surrogate import feature_vector

SCHEMA = "nslens.fixed_pairwise_policy.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_features(state: dict, first: dict, second: dict) -> np.ndarray:
    """Ordered pair encoding; swapping candidates changes the comparison example."""
    environment = state["environment"]
    first_features = feature_vector(environment, first["genotype"])
    second_features = feature_vector(environment, second["genotype"])
    return np.concatenate([
        first_features[:6], second_features[:6], first_features[6:],
        np.asarray(state["parent_normalized"], np.float32),
        np.asarray([state["sigma"]], np.float32),
    ]).astype(np.float32)


def _load(path: Path, augment: bool) -> tuple[list[dict], np.ndarray, np.ndarray,
                                               np.ndarray]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    features, labels, weights = [], [], []
    for row in rows:
        features.append(pair_features(row["state"], row["chosen"], row["rejected"]))
        labels.append(1)
        weights.append(float(row["sample_weight"]))
        if augment:
            features.append(pair_features(row["state"], row["rejected"], row["chosen"]))
            labels.append(0)
            weights.append(float(row["sample_weight"]))
    return rows, np.stack(features), np.asarray(labels), np.asarray(weights)


def _metrics(rows: list[dict], model) -> dict:
    features = np.stack([pair_features(row["state"], row["chosen"], row["rejected"])
                         for row in rows])
    probability = model.predict_proba(features)[:, 1]
    hit = probability > 0.5
    by_environment = defaultdict(list)
    by_type = defaultdict(list)
    for row, success in zip(rows, hit):
        by_environment[row["state"]["environment"]["environment_id"]].append(success)
        by_type[row["preference_type"]].append(success)
    return {
        "count": len(rows), "accuracy": float(hit.mean()),
        "mean_chosen_probability": float(probability.mean()),
        "by_environment": {key: {"count": len(values),
                                 "accuracy": float(np.mean(values))}
                           for key, values in sorted(by_environment.items())},
        "by_type": {key: {"count": len(values), "accuracy": float(np.mean(values))}
                    for key, values in sorted(by_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--leaf-sizes", default="3,5,10,20")
    parser.add_argument("--max-features", default="0.5,0.8,1.0")
    args = parser.parse_args()
    data = Path(args.data).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    train_rows, train_x, train_y, train_weights = _load(
        data / "preferences" / "train.jsonl", augment=True)
    val_rows, val_x, _val_y, _val_weights = _load(
        data / "preferences" / "validation_env.jsonl", augment=False)
    test_rows, _test_x, _test_y, _test_weights = _load(
        data / "preferences" / "test_env.jsonl", augment=False)
    search = []
    fitted = {}
    started = time.time()
    for leaf_size in [int(value) for value in args.leaf_sizes.split(",")]:
        for max_features in [float(value) for value in args.max_features.split(",")]:
            model = ExtraTreesClassifier(
                n_estimators=args.trees, min_samples_leaf=leaf_size,
                max_features=max_features, n_jobs=-1, random_state=args.seed,
                class_weight="balanced")
            model.fit(train_x, train_y, sample_weight=train_weights)
            validation_accuracy = float((model.predict_proba(val_x)[:, 1] > 0.5).mean())
            key = (leaf_size, max_features)
            fitted[key] = model
            search.append({"leaf_size": leaf_size, "max_features": max_features,
                           "validation_accuracy": validation_accuracy})
    search.sort(key=lambda row: (-row["validation_accuracy"], row["leaf_size"],
                                 row["max_features"]))
    selected = search[0]
    model = fitted[(selected["leaf_size"], selected["max_features"])]
    metrics = {
        "train": _metrics(train_rows, model),
        "validation_env": _metrics(val_rows, model),
        "test_env": _metrics(test_rows, model),
    }
    model_path = output / "model.joblib"
    joblib.dump({"schema_version": SCHEMA, "model": model,
                 "feature_contract": (
                     "first_geometry_norm[6],second_geometry_norm[6],context[5],"
                     "parent_norm[6],sigma[1]")}, model_path, compress=3)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": str(data), "data_manifest_sha256": _sha256(data / "manifest.json"),
        "selection_basis": "highest validation-environment ordered-pair accuracy",
        "search": search, "selected": selected, "metrics": metrics,
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "config": {"seed": args.seed, "trees": args.trees},
        "training_pairs": len(train_rows),
        "training_examples_after_swap_augmentation": len(train_y),
        "preference_type_counts": dict(Counter(
            row["preference_type"] for row in train_rows)),
        "wall_seconds": time.time() - started,
        "limitations": [
            "The policy ranks a supplied same-parent action batch; it does not generate "
            "moves without an explicit proposal distribution.",
            "Fresh-seed online evaluation is required before claiming improvement over CMA.",
            "All labels are from the local ray engine, not CODE V.",
        ],
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "manifest.json")
    print(json.dumps({"output": str(output), "selected": selected,
                      "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
