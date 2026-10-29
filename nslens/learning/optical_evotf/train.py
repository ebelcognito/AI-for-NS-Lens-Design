"""Train numeric-only EvoTF or zero-initialized Optical-EvoTF adapters.

Both arms consume the same generation records and hindsight targets. The numeric arm
updates the official EvoTransformer parameters using only official numeric features.
The optical arm freezes that released core by default and trains context adapters; its
warmup phase combines CMA update matching with validity/rank/route auxiliary losses.
No test split is opened during model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from evosax.algorithms import EvoTF_ES
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from ...search.evotf_emitter import verify_official_runtime
from .checkpoint import load_checkpoint, save_checkpoint, verify_runtime
from .data import batch_examples, build_examples, read_records
from .embeddings import ADAPTER_NAMES
from .model import OpticalEvoTFModel


def _gather_generation(values, indices):
    batch_indices = jnp.arange(values.shape[0])
    return values[batch_indices, indices]


def loss_for_batch(
    model: OpticalEvoTFModel,
    core_variables,
    adapter_variables,
    batch,
    *,
    phase: str,
):
    prediction, extra = model.apply(
        core_params=core_variables,
        adapter_variables=adapter_variables,
        solution_features=batch["solution_features"],
        fitness_features=batch["fitness_features"],
        distribution_features=batch["distribution_features"],
        parameter=batch["parameter"],
        environment=batch["environment"],
        optical=batch["optical"],
        generation_mask=batch["generation_mask"],
        mask=model.strategy.la_mask,
        train=True,
    )
    indices = batch["index"].astype(jnp.int32)
    pred_mean = _gather_generation(prediction[0], indices)
    pred_log_std = _gather_generation(prediction[1], indices)
    teacher_mean = jnp.mean(jnp.square(pred_mean - batch["teacher_mean_update"]))
    teacher_scale = jnp.mean(
        jnp.square(pred_log_std - batch["teacher_log_std_update"]))
    hindsight_error = jnp.mean(
        jnp.square(pred_mean - batch["hindsight_mean_update"]), axis=-1)
    hindsight_count = jnp.maximum(jnp.sum(batch["hindsight_mask"]), 1.0)
    hindsight = jnp.sum(hindsight_error * batch["hindsight_mask"]) / hindsight_count

    auxiliary = extra["auxiliary"]
    validity_logits = _gather_generation(auxiliary["validity_logits"], indices)
    rank_prediction = _gather_generation(auxiliary["rank_prediction"], indices)
    route_logits = _gather_generation(auxiliary["route_presence_logits"], indices)
    validity = jnp.mean(optax.sigmoid_binary_cross_entropy(
        validity_logits, batch["validity_target"]))
    rank = jnp.mean(jnp.square(rank_prediction - batch["rank_target"]))
    route = jnp.mean(optax.sigmoid_binary_cross_entropy(
        route_logits, batch["route_target"]))
    auxiliary_loss = validity + rank + route

    auxiliary_weight = 0.1 if model.enabled else 0.0
    if phase == "warmup":
        total = teacher_mean + teacher_scale + auxiliary_weight * auxiliary_loss
    elif phase == "hindsight":
        total = hindsight + 0.25 * teacher_scale + 0.5 * auxiliary_weight * auxiliary_loss
    else:
        raise ValueError("phase must be 'warmup' or 'hindsight'")
    return total, {
        "loss": total,
        "teacher_mean": teacher_mean,
        "teacher_scale": teacher_scale,
        "hindsight": hindsight,
        "auxiliary": auxiliary_loss,
    }


def _mean_metrics(rows):
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in rows[0]}


def _data_sources(paths):
    return [{
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    } for path in paths]


def _final_core_gradient_mask(core_variables):
    """Select only the last temporal transformer block and update heads."""
    flat = flatten_dict(unfreeze(core_variables))
    temporal_blocks = sorted({
        part for path in flat for part in path if part.startswith("transformer_")
        and "DimBatchedTransformer" in path
    })
    final_block = temporal_blocks[-1] if temporal_blocks else None
    selected = {}
    for path in flat:
        selected[path] = bool(
            "DistributionUpdate" in path
            or (final_block is not None
                and "DimBatchedTransformer" in path and final_block in path))
    if not any(selected.values()):
        raise ValueError("could not identify final EvoTF block/update heads")
    return freeze(unflatten_dict(selected))


def train(
    *,
    train_paths: list[Path],
    validation_paths: list[Path],
    output: Path,
    arm: str,
    phase: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    core_learning_rate: float,
    seed: int,
    population_size: int,
    dimensions: int,
    context_len: int,
    hidden_dim: int,
    enabled_adapters: tuple[str, ...],
    categorical_mode: str,
    init_checkpoint: Path | None = None,
    unfreeze_core: bool = False,
) -> dict:
    verify_official_runtime()
    verify_runtime()
    if arm not in ("numeric", "optical"):
        raise ValueError("arm must be 'numeric' or 'optical'")
    if phase not in ("warmup", "hindsight"):
        raise ValueError("phase must be 'warmup' or 'hindsight'")
    if arm == "numeric" and unfreeze_core:
        raise ValueError("numeric training already updates the complete EvoTF core")
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if learning_rate <= 0 or core_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    enabled = () if arm == "numeric" else tuple(enabled_adapters)
    strategy = EvoTF_ES(
        population_size=population_size,
        solution=jnp.zeros(dimensions, dtype=jnp.float32),
        max_context_len=context_len,
    )
    model = OpticalEvoTFModel(
        strategy, enabled=enabled, hidden_dim=hidden_dim,
        categorical_mode=categorical_mode)
    adapter_variables = model.init_adapter(jax.random.key(seed ^ 0x51A7))
    core_variables = strategy.default_params.params
    if init_checkpoint:
        loaded = load_checkpoint(init_checkpoint)
        manifest_model = loaded["manifest"]["model"]
        prior_training = loaded["manifest"].get("training") or {}
        if tuple(manifest_model["enabled_adapters"]) != enabled:
            raise ValueError("init checkpoint adapters do not match this training arm")
        if int(manifest_model["max_context_len"]) != context_len:
            raise ValueError("init checkpoint context length does not match")
        if int(manifest_model["hidden_dim"]) != hidden_dim:
            raise ValueError("init checkpoint hidden dimension does not match")
        if manifest_model["categorical_mode"] != categorical_mode:
            raise ValueError("init checkpoint categorical mode does not match")
        if int(prior_training.get("population_size", population_size)) != population_size:
            raise ValueError("init checkpoint population size does not match")
        if int(prior_training.get("dimensions", dimensions)) != dimensions:
            raise ValueError("init checkpoint search dimension does not match")
        adapter_variables = loaded["adapter"]
        core_variables = loaded.get("core", core_variables)

    train_examples = build_examples(read_records(train_paths), strategy)
    validation_examples = build_examples(read_records(validation_paths), strategy)
    if not train_examples or not validation_examples:
        raise ValueError("training and validation must each contain at least one example")
    if phase == "hindsight" and not any(
            float(example["hindsight_mask"]) > 0 for example in train_examples):
        raise ValueError(
            "hindsight phase has no usable fixed-horizon targets in the training split")
    rng = np.random.default_rng(seed)

    adapter_optimizer = optax.adam(learning_rate)
    core_optimizer = optax.adam(core_learning_rate if arm == "optical" else learning_rate)
    adapter_state = adapter_optimizer.init(adapter_variables)
    core_state = core_optimizer.init(core_variables)
    final_core_mask = (_final_core_gradient_mask(core_variables)
                       if arm == "optical" and unfreeze_core else None)

    if arm == "numeric":
        @jax.jit
        def update(core, state, batch):
            (loss, metrics), gradients = jax.value_and_grad(
                lambda variables: loss_for_batch(
                    model, variables, adapter_variables, batch, phase=phase),
                has_aux=True)(core)
            updates, state = core_optimizer.update(gradients, state, core)
            return optax.apply_updates(core, updates), state, loss, metrics
    elif unfreeze_core:
        @jax.jit
        def update_both(adapter, adapter_opt_state, core, core_opt_state, batch):
            def objective(adapter_value, core_value):
                return loss_for_batch(
                    model, core_value, adapter_value, batch, phase=phase)
            (loss, metrics), (adapter_grad, core_grad) = jax.value_and_grad(
                objective, argnums=(0, 1), has_aux=True)(adapter, core)
            core_grad = jax.tree.map(
                lambda gradient, selected: gradient * selected,
                core_grad, final_core_mask)
            adapter_updates, adapter_opt_state = adapter_optimizer.update(
                adapter_grad, adapter_opt_state, adapter)
            core_updates, core_opt_state = core_optimizer.update(
                core_grad, core_opt_state, core)
            return (optax.apply_updates(adapter, adapter_updates), adapter_opt_state,
                    optax.apply_updates(core, core_updates), core_opt_state, loss, metrics)
    else:
        @jax.jit
        def update_adapter(adapter, state, batch):
            (loss, metrics), gradients = jax.value_and_grad(
                lambda variables: loss_for_batch(
                    model, core_variables, variables, batch, phase=phase),
                has_aux=True)(adapter)
            updates, state = adapter_optimizer.update(gradients, state, adapter)
            return optax.apply_updates(adapter, updates), state, loss, metrics

    train_trace = []
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(train_examples), size=batch_size)
        batch = batch_examples([train_examples[int(index)] for index in indices])
        # Keep string IDs for audit/debugging, but never send them through a JIT
        # boundary: the loss consumes only array-valued model inputs and targets.
        batch.pop("transition_ids", None)
        if arm == "numeric":
            core_variables, core_state, loss, metrics = update(
                core_variables, core_state, batch)
        elif unfreeze_core:
            (adapter_variables, adapter_state, core_variables, core_state,
             loss, metrics) = update_both(
                adapter_variables, adapter_state, core_variables, core_state, batch)
        else:
            adapter_variables, adapter_state, loss, metrics = update_adapter(
                adapter_variables, adapter_state, batch)
        if step == 1 or step == steps or step % max(steps // 10, 1) == 0:
            row = {key: float(value) for key, value in metrics.items()}
            row["step"] = step
            train_trace.append(row)

    validation_rows = []
    for start in range(0, len(validation_examples), batch_size):
        batch = batch_examples(validation_examples[start:start + batch_size])
        batch.pop("transition_ids", None)
        _, metrics = loss_for_batch(
            model, core_variables, adapter_variables, batch, phase=phase)
        validation_rows.append({key: float(value) for key, value in metrics.items()})
    validation_metrics = _mean_metrics(validation_rows)
    training = {
        "arm": arm,
        "phase": phase,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "core_learning_rate": core_learning_rate if unfreeze_core else None,
        "unfreeze_core": bool(unfreeze_core),
        "core_trainable_scope": (
            "all" if arm == "numeric" else
            "final_temporal_block_and_update_heads" if unfreeze_core else "frozen"),
        "seed": seed,
        "population_size": population_size,
        "dimensions": dimensions,
        "max_context_len": context_len,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_trace": train_trace,
        "validation_metrics": validation_metrics,
        "model_selection_split": "validation",
        "test_metrics_read": False,
        "data": {
            "train": _data_sources(train_paths),
            "validation": _data_sources(validation_paths),
            "test": [],
        },
    }
    manifest = save_checkpoint(
        output, adapter_variables=adapter_variables, enabled=enabled,
        hidden_dim=hidden_dim, categorical_mode=categorical_mode,
        max_context_len=context_len, training=training,
        core_variables=(core_variables if arm == "numeric" or unfreeze_core else None),
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--view", default="held_seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("numeric", "optical"), required=True)
    parser.add_argument("--phase", choices=("warmup", "hindsight"), default="warmup")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--core-learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--dimensions", type=int, default=9)
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--enabled-adapters", nargs="+", choices=ADAPTER_NAMES,
                        default=list(ADAPTER_NAMES))
    parser.add_argument("--categorical-mode", choices=("learned", "one_hot"),
                        default="learned")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--unfreeze-core", action="store_true")
    args = parser.parse_args(argv)
    train_path = args.dataset / args.view / "train.jsonl"
    validation_path = args.dataset / args.view / "validation.jsonl"
    if not validation_path.exists() or validation_path.stat().st_size == 0:
        parser.error(
            "the selected view has no validation split; collect at least three search seeds")
    manifest = train(
        train_paths=[train_path], validation_paths=[validation_path],
        output=args.output, arm=args.arm, phase=args.phase, steps=args.steps,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        core_learning_rate=args.core_learning_rate, seed=args.seed,
        population_size=args.population_size, dimensions=args.dimensions,
        context_len=args.context_len, hidden_dim=args.hidden_dim,
        enabled_adapters=tuple(args.enabled_adapters),
        categorical_mode=args.categorical_mode,
        init_checkpoint=args.init_checkpoint, unfreeze_core=args.unfreeze_core)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
