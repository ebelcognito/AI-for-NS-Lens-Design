"""Padded chronological EvoTF feature batches from optimizer-transition records."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from ...rl.data.schema import validate_optimizer_transition
from .metadata import (ENV_CONTINUOUS_SIZE, encode_environment,
                       encode_observations, parameter_metadata)


def read_records(paths: Sequence[Path]) -> list[dict]:
    records = []
    for path in paths:
        with Path(path).open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                errors = validate_optimizer_transition(record)
                if errors:
                    raise ValueError(f"{path}:{line_number}: {errors}")
                if "teacher_target" not in record or "hindsight_target" not in record:
                    raise ValueError(
                        f"{path}:{line_number}: build_optimizer_trajectories targets missing")
                records.append(record)
    return records


def _push(history: np.ndarray, value: np.ndarray, position: int) -> tuple[np.ndarray, int]:
    length = history.shape[0]
    if position < length:
        history[position] = value
        return history, position
    history[:-1] = history[1:]
    history[-1] = value
    return history, length - 1


def build_examples(records: Sequence[dict], strategy) -> list[dict]:
    """Recreate official numeric feature histories and aligned optical histories."""
    by_lifecycle = defaultdict(list)
    for record in records:
        key = record.get("source_lifecycle_id", record["lifecycle_id"])
        by_lifecycle[key].append(record)
    examples = []
    timesteps = strategy.max_context_len
    population = strategy.population_size
    dimensions = strategy.num_dims
    for lifecycle in by_lifecycle.values():
        lifecycle.sort(key=lambda row: row["generation"])
        sf_state, ff_state, df_state = strategy.sf.init(), strategy.ff.init(), strategy.df.init()
        solution_history = np.zeros(
            (timesteps, population, dimensions, strategy.sf.num_features), np.float32)
        fitness_history = np.zeros(
            (timesteps, population, strategy.ff.num_features), np.float32)
        distribution_history = np.zeros(
            (timesteps, dimensions, strategy.df.num_features), np.float32)
        optical_template = encode_observations(
            [], population_size=population, num_dims=dimensions)
        optical_history = {
            key: np.zeros((timesteps, *value.shape), dtype=value.dtype)
            for key, value in optical_template.items()
        }
        environment_history = {
            "continuous": np.zeros((timesteps, ENV_CONTINUOUS_SIZE), np.float32),
            "categorical": np.zeros((timesteps, 2), np.int32),
        }
        generation_mask = np.zeros(timesteps, np.float32)
        for position, record in enumerate(lifecycle):
            if len(record["candidates"]) != population:
                raise ValueError("transition population does not match strategy.population_size")
            if len(record["parameter_names"]) != dimensions:
                raise ValueError("transition dimensions do not match strategy.num_dims")
            before = record["state_before"]
            mean = jnp.asarray(before["mean"], dtype=jnp.float32)
            covariance = np.asarray(before["covariance"], dtype=np.float64)
            std_np = float(before["sigma"]) * np.sqrt(np.diag(covariance))
            std = jnp.asarray(std_np, dtype=jnp.float32)
            candidates = sorted(record["candidates"], key=lambda row: row["candidate_index"])
            candidate_x = jnp.asarray(
                [row["search_coordinates"] for row in candidates], dtype=jnp.float32)
            fitness = jnp.asarray(
                [row["rank_percentile"] for row in candidates], dtype=jnp.float32)
            sfeatures, sf_state = strategy.sf.featurize(
                candidate_x, fitness, mean, std, sf_state)
            ffeatures, ff_state = strategy.ff.featurize(candidate_x, fitness, ff_state)
            dfeatures, df_state = strategy.df.featurize(
                candidate_x, fitness, mean, std, df_state)
            solution_history, index = _push(
                solution_history, np.asarray(sfeatures), position)
            fitness_history, _ = _push(fitness_history, np.asarray(ffeatures), position)
            distribution_history, _ = _push(
                distribution_history, np.asarray(dfeatures), position)

            encoded_optical = encode_observations(
                candidates, population_size=population, num_dims=dimensions)
            for key in optical_history:
                optical_history[key], _ = _push(
                    optical_history[key], encoded_optical[key], position)
            environment_input = {
                **record["context"], "fingerprints": record["fingerprints"]}
            encoded_environment = encode_environment(environment_input)
            for key in environment_history:
                environment_history[key], _ = _push(
                    environment_history[key], encoded_environment[key], position)
            generation_mask, _ = _push(generation_mask, np.asarray(1.0), position)

            teacher = record["teacher_target"]
            next_mean = np.asarray(teacher["next_mean"], dtype=np.float64)
            next_std = np.asarray(teacher["next_diagonal_std"], dtype=np.float64)
            teacher_mean_update = (next_mean - np.asarray(mean)) / np.maximum(std_np, 1e-12)
            teacher_log_std_update = 2.0 * np.log(
                np.maximum(next_std, 1e-12) / np.maximum(std_np, 1e-12))
            hindsight = record["hindsight_target"]
            if hindsight.get("mask") and hindsight.get("displacement") is not None:
                hindsight_update = np.asarray(
                    hindsight["displacement"], dtype=np.float64) / np.maximum(std_np, 1e-12)
            else:
                hindsight_update = np.zeros(dimensions, dtype=np.float64)
            examples.append({
                "transition_id": record["transition_id"],
                "parameter": parameter_metadata(record["parameter_names"]),
                "solution_features": solution_history.copy(),
                "fitness_features": fitness_history.copy(),
                "distribution_features": distribution_history.copy(),
                "environment": {key: value.copy() for key, value in environment_history.items()},
                "optical": {key: value.copy() for key, value in optical_history.items()},
                "generation_mask": generation_mask.copy(),
                "index": index,
                "teacher_mean_update": teacher_mean_update.astype(np.float32),
                "teacher_log_std_update": teacher_log_std_update.astype(np.float32),
                "hindsight_mean_update": hindsight_update.astype(np.float32),
                "hindsight_mask": float(bool(hindsight.get("mask"))),
                "hindsight_truncated": float(bool(hindsight.get("truncated"))),
                "validity_target": np.asarray(
                    [row["status"] != "invalid" for row in candidates], np.float32),
                "rank_target": np.asarray(
                    [row["rank_percentile"] for row in candidates], np.float32),
                "route_target": np.asarray(
                    [bool((row.get("route") or {}).get("events")) for row in candidates],
                    np.float32),
            })
    return examples


def batch_examples(examples: Sequence[dict]) -> dict:
    if not examples:
        raise ValueError("cannot batch an empty example list")
    first_parameter = examples[0]["parameter"]
    for example in examples[1:]:
        if any(not np.array_equal(example["parameter"][key], first_parameter[key])
               for key in first_parameter):
            raise ValueError("one batch cannot mix parameter representations")
    stack_keys = (
        "solution_features", "fitness_features", "distribution_features",
        "generation_mask", "index", "teacher_mean_update",
        "teacher_log_std_update", "hindsight_mean_update", "hindsight_mask",
        "hindsight_truncated", "validity_target", "rank_target", "route_target",
    )
    batch = {key: jnp.asarray(np.stack([example[key] for example in examples]))
             for key in stack_keys}
    batch["parameter"] = {key: jnp.asarray(value) for key, value in first_parameter.items()}
    batch["environment"] = {
        key: jnp.asarray(np.stack([example["environment"][key] for example in examples]))
        for key in examples[0]["environment"]}
    batch["optical"] = {
        key: jnp.asarray(np.stack([example["optical"][key] for example in examples]))
        for key in examples[0]["optical"]}
    batch["transition_ids"] = [example["transition_id"] for example in examples]
    return batch


__all__ = ["batch_examples", "build_examples", "read_records"]
