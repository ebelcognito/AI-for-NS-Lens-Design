"""Train a tabular fixed-environment ranker selected only on validation environments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from .fixed_surrogate import FEATURE_NAMES, feature_vector

SCHEMA = "nslens.fixed_tree_surrogate.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    features = np.stack([feature_vector(row["environment"], row["genotype"])
                         for row in rows])
    feasible = np.asarray([bool(row["hard_pass"]) for row in rows])
    quality = np.asarray([float(row["fitness"]) if row["hard_pass"] else 0.0
                          for row in rows])
    return rows, features, feasible, quality


def _score(classifier, regressor, x: np.ndarray, mode: str) -> np.ndarray:
    probability = classifier.predict_proba(x)[:, 1]
    if mode == "feasibility":
        return probability
    quality = regressor.predict(x)
    if mode == "feasibility_x_quality":
        return probability * quality
    if mode == "threshold_quality":
        return (probability >= 0.5) * quality
    raise ValueError(mode)


def _pool_metrics(rows: list[dict], utility: np.ndarray, score: np.ndarray,
                  pool_size: int) -> tuple[dict, float]:
    environments = np.asarray([row["environment"]["environment_id"] for row in rows])
    algorithms = np.asarray([row["source"]["algorithm"] for row in rows])
    evaluations = np.asarray([int(row["source"]["eval"]) for row in rows])
    result = {}
    lifts = []
    for environment_id in sorted(set(environments)):
        indices = np.where((environments == environment_id) & (algorithms == "random"))[0]
        indices = indices[np.argsort(evaluations[indices])]
        selected, random_values, oracle = [], [], []
        selected_pass = []
        for start in range(0, len(indices) - pool_size + 1, pool_size):
            block = indices[start:start + pool_size]
            choice = block[int(np.argmax(score[block]))]
            selected.append(float(utility[choice]))
            selected_pass.append(utility[choice] > 0)
            random_values.extend(float(value) for value in utility[block])
            oracle.append(float(utility[block].max()))
        selected_mean = float(np.mean(selected))
        random_mean = float(np.mean(random_values))
        lift = selected_mean / max(random_mean, 1e-12)
        lifts.append(lift)
        result[environment_id] = {
            "pools": len(selected), "pool_size": pool_size,
            "selected_hard_pass_rate": float(np.mean(selected_pass)),
            "selected_mean_utility": selected_mean,
            "uniform_random_mean_utility": random_mean,
            "oracle_mean_utility": float(np.mean(oracle)), "lift_over_random": lift,
        }
    geometric_lift = float(math.exp(np.mean(np.log(np.maximum(lifts, 1e-12)))))
    return result, geometric_lift


def _metrics(rows: list[dict], x: np.ndarray, feasible: np.ndarray,
             quality: np.ndarray, classifier, regressor, mode: str,
             pool_size: int) -> dict:
    probability = classifier.predict_proba(x)[:, 1]
    predicted_quality = regressor.predict(x)
    environments = np.asarray([row["environment"]["environment_id"] for row in rows])
    per_environment = {}
    for environment_id in sorted(set(environments)):
        mask = environments == environment_id
        positive = mask & feasible
        rho = spearmanr(quality[positive], predicted_quality[positive]).statistic
        per_environment[environment_id] = {
            "count": int(mask.sum()), "positive_fraction": float(feasible[mask].mean()),
            "roc_auc": float(roc_auc_score(feasible[mask], probability[mask])),
            "average_precision": float(average_precision_score(
                feasible[mask], probability[mask])),
            "quality_mae_on_hard_pass": float(np.mean(np.abs(
                quality[positive] - predicted_quality[positive]))),
            "quality_spearman_on_hard_pass": (
                None if not math.isfinite(float(rho)) else float(rho)),
        }
    score = _score(classifier, regressor, x, mode)
    pools, geometric_lift = _pool_metrics(rows, quality, score, pool_size)
    return {"per_environment": per_environment,
            "random_pool_ranking": pools,
            "geometric_mean_lift_over_random": geometric_lift}


def _preference_accuracy(path: Path, classifier, regressor, mode: str) -> dict:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        return {"count": 0, "accuracy": None, "by_type": {}}
    features = []
    for row in rows:
        environment = row["state"]["environment"]
        features.extend(feature_vector(environment, row[name]["genotype"])
                        for name in ("chosen", "rejected"))
    scores = _score(classifier, regressor, np.stack(features), mode).reshape(-1, 2)
    correct = 0
    by_type = defaultdict(lambda: [0, 0])
    for row, score in zip(rows, scores):
        hit = bool(score[0] > score[1])
        correct += int(hit)
        bucket = by_type[row["preference_type"]]
        bucket[0] += int(hit)
        bucket[1] += 1
    return {"count": len(rows), "accuracy": correct / len(rows),
            "by_type": {key: {"count": total, "accuracy": hits / total}
                        for key, (hits, total) in sorted(by_type.items())}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--leaf-sizes", default="1,3,5,10,20")
    parser.add_argument("--score-modes",
                        default="feasibility,feasibility_x_quality,threshold_quality")
    parser.add_argument("--pool-size", type=int, default=20)
    args = parser.parse_args()
    data = Path(args.data).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    leaf_sizes = [int(value) for value in args.leaf_sizes.split(",")]
    score_modes = [value.strip() for value in args.score_modes.split(",") if value.strip()]
    loaded = {split: _load(data / "outcomes" / f"{split}.jsonl")
              for split in ("train", "validation_env", "test_env")}
    train_rows, train_x, train_y, train_quality = loaded["train"]
    val_rows, val_x, _val_y, val_quality = loaded["validation_env"]
    candidates = []
    fitted = {}
    started = time.time()
    for leaf_size in leaf_sizes:
        classifier = ExtraTreesClassifier(
            n_estimators=args.trees, min_samples_leaf=leaf_size,
            max_features=1.0, class_weight="balanced", n_jobs=-1,
            random_state=args.seed)
        regressor = ExtraTreesRegressor(
            n_estimators=args.trees, min_samples_leaf=leaf_size,
            max_features=1.0, n_jobs=-1, random_state=args.seed)
        classifier.fit(train_x, train_y)
        regressor.fit(train_x[train_y], train_quality[train_y])
        fitted[leaf_size] = (classifier, regressor)
        for mode in score_modes:
            score = _score(classifier, regressor, val_x, mode)
            _pools, lift = _pool_metrics(
                val_rows, val_quality, score, args.pool_size)
            candidates.append({"leaf_size": leaf_size, "score_mode": mode,
                               "validation_geometric_mean_lift": lift})
    candidates.sort(key=lambda row: (-row["validation_geometric_mean_lift"],
                                     row["leaf_size"], row["score_mode"]))
    selected = candidates[0]
    classifier, regressor = fitted[selected["leaf_size"]]
    metrics = {}
    for split, (rows, x, feasible, quality) in loaded.items():
        metrics[split] = _metrics(rows, x, feasible, quality, classifier, regressor,
                                  selected["score_mode"], args.pool_size)
        if split != "train":
            metrics[split]["refinement_preference"] = _preference_accuracy(
                data / "preferences" / f"{split}.jsonl", classifier, regressor,
                selected["score_mode"])

    model_path = output / "model.joblib"
    joblib.dump({"schema_version": SCHEMA, "classifier": classifier,
                 "regressor": regressor, "score_mode": selected["score_mode"],
                 "features": FEATURE_NAMES}, model_path, compress=3)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": str(data), "data_manifest_sha256": _sha256(data / "manifest.json"),
        "selection_basis": (
            "maximum geometric-mean utility lift over uniform random across validation "
            "environments; test environments are not used for selection"),
        "search": candidates, "selected": selected, "metrics": metrics,
        "config": {"seed": args.seed, "trees": args.trees,
                   "pool_size": args.pool_size, "features": list(FEATURE_NAMES)},
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "wall_seconds": time.time() - started,
        "limitations": [
            "Environment-level sample size is small; online fresh-seed evaluation is required.",
            "This model ranks local ray-engine candidates and has no CODE V supervision.",
        ],
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "manifest.json")
    print(json.dumps({"output": str(output), "selected": selected,
                      "validation": metrics["validation_env"],
                      "test": metrics["test_env"]}, indent=2))


if __name__ == "__main__":
    main()
