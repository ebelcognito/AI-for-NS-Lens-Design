"""Train and evaluate the fixed-environment feasibility/quality ranker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .fixed_surrogate import FEATURE_NAMES, build_model, feature_vector

SCHEMA = "nslens.fixed_surrogate_training.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _arrays(rows: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(rows)
    x = np.stack([feature_vector(row["environment"], row["genotype"])
                  for row in rows])
    feasible = np.asarray([float(row["hard_pass"]) for row in rows], np.float32)
    quality = np.asarray([
        float(row["fitness"]) if row["hard_pass"] else 0.0 for row in rows
    ], np.float32)
    return x, feasible, quality


def _classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 precision_recall_fscore_support, roc_auc_score)

    pred = probability >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return {
        "positive_fraction": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def _regression_metrics(y: np.ndarray, prediction: np.ndarray,
                        mask: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    actual = y[mask]
    predicted = prediction[mask]
    if len(actual) == 0:
        return {"count": 0}
    residual = actual - predicted
    denominator = float(((actual - actual.mean()) ** 2).sum())
    rho = spearmanr(actual, predicted).statistic if len(actual) > 1 else math.nan
    return {
        "count": int(len(actual)), "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt((residual ** 2).mean())),
        "r2": (float(1.0 - (residual ** 2).sum() / denominator)
               if denominator > 0 else None),
        "spearman": None if not math.isfinite(float(rho)) else float(rho),
    }


def _predict(model, x: np.ndarray, device: str, batch_size: int = 4096
             ) -> tuple[np.ndarray, np.ndarray]:
    import torch

    probabilities, qualities = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            logits, quality = model(xb)
            probabilities.append(logits.sigmoid().cpu().numpy())
            qualities.append(quality.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(qualities)


def _ranking_metrics(rows: list[dict], model, device: str, pool_size: int) -> dict:
    """Offline ranking on disjoint blocks of uniform-random held-environment proposals."""
    by_environment = defaultdict(list)
    for row in rows:
        if row["source"]["algorithm"] == "random":
            by_environment[row["environment"]["environment_id"]].append(row)
    output = {}
    for environment_id, candidates in sorted(by_environment.items()):
        candidates.sort(key=lambda row: int(row["source"]["eval"]))
        picked, oracle, random_utility = [], [], []
        for start in range(0, len(candidates) - pool_size + 1, pool_size):
            block = candidates[start:start + pool_size]
            x, _, _ = _arrays(block)
            probability, quality = _predict(model, x, device)
            index = int(np.argmax(probability * quality))
            utilities = np.asarray([
                float(row["fitness"]) if row["hard_pass"] else 0.0 for row in block])
            picked.append((bool(block[index]["hard_pass"]), float(utilities[index])))
            oracle.append(float(utilities.max()))
            random_utility.extend(float(value) for value in utilities)
        output[environment_id] = {
            "pools": len(picked), "pool_size": pool_size,
            "selected_hard_pass_rate": float(np.mean([value[0] for value in picked])),
            "selected_mean_utility": float(np.mean([value[1] for value in picked])),
            "uniform_random_mean_utility": float(np.mean(random_utility)),
            "oracle_mean_utility": float(np.mean(oracle)),
        }
    return output


def _preference_accuracy(path: Path, model, device: str) -> dict:
    rows = _read_jsonl(path)
    if not rows:
        return {"count": 0, "accuracy": None}
    correct = 0
    by_type = defaultdict(lambda: [0, 0])
    for row in rows:
        environment = row["state"]["environment"]
        x = np.stack([
            feature_vector(environment, row["chosen"]["genotype"]),
            feature_vector(environment, row["rejected"]["genotype"]),
        ])
        probability, quality = _predict(model, x, device)
        success = bool(probability[0] * quality[0] > probability[1] * quality[1])
        correct += int(success)
        bucket = by_type[row["preference_type"]]
        bucket[0] += int(success)
        bucket[1] += 1
    return {
        "count": len(rows), "accuracy": correct / len(rows),
        "by_type": {key: {"count": total, "accuracy": hits / total}
                    for key, (hits, total) in sorted(by_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--quality-weight", type=float, default=5.0)
    parser.add_argument("--ranking-pool-size", type=int, default=20)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional

    data = Path(args.data).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = {split: _read_jsonl(data / "outcomes" / f"{split}.jsonl")
            for split in ("train", "validation_env", "test_env")}
    arrays = {split: _arrays(part) for split, part in rows.items()}
    train_x, train_y, train_q = arrays["train"]
    val_x, val_y, val_q = arrays["validation_env"]
    model = build_model(args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    positive_weight = torch.tensor(
        [(len(train_y) - train_y.sum()) / train_y.sum()], device=device)
    rng = np.random.default_rng(args.seed)
    best_loss = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()

    val_xt = torch.from_numpy(val_x).to(device)
    val_yt = torch.from_numpy(val_y).to(device)
    val_qt = torch.from_numpy(val_q).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        train_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            xb = torch.from_numpy(train_x[indices]).to(device)
            yb = torch.from_numpy(train_y[indices]).to(device)
            qb = torch.from_numpy(train_q[indices]).to(device)
            logits, quality = model(xb)
            feasibility_loss = functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=positive_weight)
            mask = yb > 0.5
            quality_loss = functional.mse_loss(quality[mask], qb[mask])
            loss = feasibility_loss + args.quality_weight * quality_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * len(indices)
        model.eval()
        with torch.no_grad():
            logits, quality = model(val_xt)
            feasibility_loss = functional.binary_cross_entropy_with_logits(
                logits, val_yt, pos_weight=positive_weight)
            mask = val_yt > 0.5
            quality_loss = functional.mse_loss(quality[mask], val_qt[mask])
            val_loss = float(feasibility_loss + args.quality_weight * quality_loss)
        history.append({"epoch": epoch, "train_loss": train_loss / len(train_x),
                        "validation_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)

    metrics = {}
    for split, (x, y, quality) in arrays.items():
        probability, prediction = _predict(model, x, device)
        metrics[split] = {
            "classification": _classification_metrics(y, probability),
            "quality_on_hard_pass": _regression_metrics(
                quality, prediction, y > 0.5),
            "ranking_random_pools": _ranking_metrics(
                rows[split], model, device, args.ranking_pool_size),
        }
        if split != "train":
            metrics[split]["refinement_preference"] = _preference_accuracy(
                data / "preferences" / f"{split}.jsonl", model, device)

    checkpoint_path = output / "checkpoint.pt"
    checkpoint = {
        "schema_version": SCHEMA, "model_state": best_state,
        "config": {"hidden": args.hidden, "features": list(FEATURE_NAMES),
                   "seed": args.seed},
    }
    torch.save(checkpoint, checkpoint_path)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": str(data),
        "data_manifest_sha256": _sha256(data / "manifest.json"),
        "device": device, "seed": args.seed,
        "config": {key: getattr(args, key) for key in (
            "hidden", "epochs", "patience", "batch_size", "learning_rate",
            "quality_weight", "ranking_pool_size")},
        "selection": {"criterion": "minimum validation_env joint loss",
                      "best_epoch": best_epoch, "best_validation_loss": best_loss,
                      "epochs_ran": len(history)},
        "metrics": metrics, "history": history,
        "checkpoint": {"path": str(checkpoint_path),
                       "sha256": _sha256(checkpoint_path)},
        "wall_seconds": time.time() - started,
        "limitations": [
            "The test split is used only after checkpoint selection on validation environments.",
            "Offline random-pool ranking is diagnostic; the final claim requires online "
            "equal-engine-budget evaluation.",
            "This is a local ray-engine surrogate and has no CODE V labels.",
        ],
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "manifest.json")
    print(json.dumps({
        "output": str(output), "device": device,
        "best_epoch": best_epoch, "epochs_ran": len(history),
        "validation": metrics["validation_env"],
        "test": metrics["test_env"],
    }, indent=2))


if __name__ == "__main__":
    main()
