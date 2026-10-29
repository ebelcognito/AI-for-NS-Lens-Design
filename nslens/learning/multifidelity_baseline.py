"""Measure whether cheap ray-trace supervision transfers to the full local verifier.

This is a deliberately small statistical gate, not the final design model.  At a fixed
number of full-beam training labels it compares:

* ``hf_only``: geometry/spec features -> full nine-field fitness;
* ``proxy_only``: geometry/spec features -> sparse proxy fitness;
* ``learned_proxy_delta``: many proxy labels plus a small paired full-label residual;
* ``measured_proxy_delta``: the same residual with the cheap proxy called at inference.

The last method is an upper bound on how valuable the proxy can be.  The learned variant
is the actual transfer test: its inference input is identical to the high-fidelity-only
baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

PROXY = "sparse_multifov"
AUTHORITY = "full_multifov"


def _target(row: dict, name: str) -> float:
    label = row["labels"][name]
    return float(label["fitness"]) if label["stage"] == "scored" else 0.0


def _features(rows: Sequence[dict], env_names: Sequence[str]) -> np.ndarray:
    out = []
    for row in rows:
        env = row["environment"]
        one_hot = [float(env["name"] == name) for name in env_names]
        out.append([
            *map(float, row["normalized_genotype"]),
            float(env["fov_half_angle_deg"]) / 12.0,
            float(env["focus_window_half_width_mm"]) / 12.0,
            *one_hot,
        ])
    return np.asarray(out, float)


class PolynomialRidge:
    """Small dependency-free nonlinear baseline.

    Pairwise products are enough to expose whether proxy supervision carries signal while
    keeping this gate runnable in the engine's minimal Python environment.  This is not
    intended to be the eventual model architecture.
    """

    def __init__(self, alpha: float | None = None, seed: int = 0):
        self.alpha = None if alpha is None else float(alpha)
        self.seed = int(seed)

    @staticmethod
    def _expand(x: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(x)), *[x[:, i] for i in range(x.shape[1])]]
        cols.extend(x[:, i] * x[:, j]
                    for i in range(x.shape[1]) for j in range(i, x.shape[1]))
        return np.column_stack(cols)

    def _fit_fixed(self, x: np.ndarray, y: np.ndarray, alpha: float) -> None:
        if len(x) < 1:
            raise ValueError("PolynomialRidge.fit requires at least one training row")
        phi = self._expand(np.asarray(x, float))
        self.mean_ = phi[:, 1:].mean(axis=0)
        self.scale_ = phi[:, 1:].std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        phi[:, 1:] = (phi[:, 1:] - self.mean_) / self.scale_
        penalty = np.eye(phi.shape[1]) * alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(phi.T @ phi + penalty, phi.T @ np.asarray(y, float))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialRidge":
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 2:
            raise ValueError(f"x must be a 2-D array, got shape {x.shape}")
        if y.ndim != 1 or len(x) != len(y):
            raise ValueError("x must have one target per row in y")
        # A single paired label cannot select regularisation or identify a residual.
        # Failing explicitly is preferable to the singular-matrix error produced by
        # the cross-validation path below, and keeps tiny scaling sweeps honest.
        if len(x) < 2:
            raise ValueError("PolynomialRidge.fit requires at least two training rows")
        if self.alpha is None:
            # Pick regularisation using training labels only.  The feature expansion has
            # more columns than the 16/32-label regimes have rows, so an unregularised
            # fit is not a meaningful baseline.  Three deterministic folds avoid looking
            # at the held-out topology set while selecting the bias/variance tradeoff.
            candidates = (0.1, 1.0, 10.0, 100.0)
            order = np.random.default_rng(self.seed).permutation(len(x))
            folds = np.array_split(order, min(3, len(x)))
            losses = []
            for alpha in candidates:
                fold_losses = []
                for valid in folds:
                    train = np.setdiff1d(order, valid, assume_unique=True)
                    trial = PolynomialRidge(alpha=alpha)
                    trial._fit_fixed(x[train], y[train], alpha)
                    fold_losses.append(float(np.mean((trial.predict(x[valid]) - y[valid]) ** 2)))
                losses.append(float(np.mean(fold_losses)))
            self.chosen_alpha_ = float(candidates[int(np.argmin(losses))])
        else:
            self.chosen_alpha_ = self.alpha
        self._fit_fixed(x, y, self.chosen_alpha_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        phi = self._expand(np.asarray(x, float))
        phi[:, 1:] = (phi[:, 1:] - self.mean_) / self.scale_
        return phi @ self.coef_


def _model(seed: int) -> PolynomialRidge:
    return PolynomialRidge(alpha=None, seed=seed)


def _metrics(y: np.ndarray, prediction: np.ndarray, hard: np.ndarray,
             top_k: int) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(y) == 0:
        raise ValueError("cannot score an empty test set")
    rho = spearmanr(y, prediction).statistic
    k = min(top_k, len(y))
    selected = np.argsort(prediction)[-k:]
    oracle = np.argsort(y)[-k:]
    selected_mean = float(np.mean(y[selected]))
    oracle_mean = float(np.mean(y[oracle]))
    return {
        "mae": float(np.mean(np.abs(y - prediction))),
        "r2": (float(1.0 - np.sum((y - prediction) ** 2)
                     / max(np.sum((y - np.mean(y)) ** 2), 1e-12))
               if len(y) > 1 else 0.0),
        "spearman": float(rho) if math.isfinite(float(rho)) else 0.0,
        "top_k": k,
        "top_k_full_fitness": selected_mean,
        "top_k_regret": oracle_mean - selected_mean,
        "hard_pass_precision_at_k": float(np.mean(hard[selected])),
    }


def run(rows: Sequence[dict], *, proxy_sizes: Sequence[int],
        hf_sizes: Sequence[int], model_seeds: int = 3, top_k: int = 20) -> dict:
    proxy_sizes = tuple(int(n) for n in proxy_sizes)
    hf_sizes = tuple(int(n) for n in hf_sizes)
    if not proxy_sizes or not hf_sizes:
        raise ValueError("proxy_sizes and hf_sizes must each contain at least one size")
    if any(n < 2 for n in proxy_sizes + hf_sizes):
        raise ValueError("proxy_sizes and hf_sizes must all be at least 2")
    if model_seeds < 1:
        raise ValueError("model_seeds must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    train = [r for r in rows if r["split"] == "train" and PROXY in r["labels"]]
    hf_train = [r for r in train if AUTHORITY in r["labels"]]
    test = [r for r in rows if r["split"] != "train" and AUTHORITY in r["labels"]]
    if not train or not hf_train or not test:
        raise ValueError("dataset needs proxy train, paired full train, and held-out full rows")
    if max(proxy_sizes) > len(train):
        raise ValueError(f"proxy size exceeds {len(train)} available training rows")
    if max(hf_sizes) > len(hf_train):
        raise ValueError(f"HF size exceeds {len(hf_train)} paired training rows")

    env_names = sorted({r["environment"]["name"] for r in rows})
    x_test = _features(test, env_names)
    y_test = np.asarray([_target(r, AUTHORITY) for r in test])
    p_test_measured = np.asarray([_target(r, PROXY) for r in test])
    hard_test = np.asarray([
        bool(r["labels"][AUTHORITY]["hard_pass"]) for r in test
    ], bool)

    results = []
    for seed in range(model_seeds):
        rng = np.random.default_rng(10_000 + seed)
        train_order = rng.permutation(len(train))
        hf_order = rng.permutation(len(hf_train))
        for proxy_n in sorted(set(proxy_sizes)):
            proxy_rows = [train[i] for i in train_order[:proxy_n]]
            x_proxy = _features(proxy_rows, env_names)
            y_proxy = np.asarray([_target(r, PROXY) for r in proxy_rows])
            proxy_model = _model(seed * 1000 + proxy_n).fit(x_proxy, y_proxy)
            learned_proxy_test = proxy_model.predict(x_test)

            for hf_n in sorted(set(hf_sizes)):
                paired = [hf_train[i] for i in hf_order[:hf_n]]
                x_hf = _features(paired, env_names)
                y_hf = np.asarray([_target(r, AUTHORITY) for r in paired])
                p_hf_measured = np.asarray([_target(r, PROXY) for r in paired])
                p_hf_learned = proxy_model.predict(x_hf)

                hf_model = _model(seed * 1000 + hf_n + 100).fit(x_hf, y_hf)
                learned_delta = _model(seed * 1000 + hf_n + proxy_n + 200).fit(
                    np.column_stack([x_hf, p_hf_learned]), y_hf - p_hf_learned)
                measured_delta = _model(seed * 1000 + hf_n + proxy_n + 300).fit(
                    np.column_stack([x_hf, p_hf_measured]), y_hf - p_hf_measured)

                predictions = {
                    "hf_only": hf_model.predict(x_test),
                    "proxy_only": learned_proxy_test,
                    "learned_proxy_delta": learned_proxy_test + learned_delta.predict(
                        np.column_stack([x_test, learned_proxy_test])),
                    "measured_proxy_delta": p_test_measured + measured_delta.predict(
                        np.column_stack([x_test, p_test_measured])),
                }
                for method, prediction in predictions.items():
                    results.append({
                        "seed": seed,
                        "proxy_train": proxy_n,
                        "hf_train": hf_n,
                        "method": method,
                        **_metrics(y_test, prediction, hard_test, top_k),
                    })

    aggregates = []
    buckets = defaultdict(list)
    for result in results:
        buckets[(result["proxy_train"], result["hf_train"], result["method"])].append(result)
    metric_names = ("mae", "r2", "spearman", "top_k_full_fitness",
                    "top_k_regret", "hard_pass_precision_at_k")
    for (proxy_n, hf_n, method), values in sorted(buckets.items()):
        aggregates.append({
            "proxy_train": proxy_n,
            "hf_train": hf_n,
            "method": method,
            **{name: float(np.mean([v[name] for v in values])) for name in metric_names},
            **{name + "_std": float(np.std([v[name] for v in values]))
               for name in metric_names},
        })

    return {
        "schema_version": "nslens.multifidelity_baseline.v1",
        "proxy_label": PROXY,
        "authority_label": AUTHORITY,
        "counts": {"train_proxy": len(train), "train_hf": len(hf_train),
                   "test_hf": len(test)},
        "feature_dimension": int(x_test.shape[1]),
        "model_seeds": model_seeds,
        "top_k": min(top_k, len(test)),
        "aggregates": aggregates,
        "runs": results,
    }


def _sizes(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main(argv=None) -> int:  # pragma: no cover - exercised through the CLI smoke run
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="runs/multifidelity_pilot_v1/traces.jsonl")
    ap.add_argument("--out", default="runs/multifidelity_pilot_v1/baseline.json")
    ap.add_argument("--proxy-sizes", type=_sizes, default=_sizes("64,128,256"))
    ap.add_argument("--hf-sizes", type=_sizes, default=_sizes("16,32,64"))
    ap.add_argument("--model-seeds", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.data).open() if line.strip()]
    payload = run(rows, proxy_sizes=args.proxy_sizes, hf_sizes=args.hf_sizes,
                  model_seeds=args.model_seeds, top_k=args.top_k)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["counts"], indent=2))
    print(f"wrote {out} ({len(payload['aggregates'])} aggregate rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
