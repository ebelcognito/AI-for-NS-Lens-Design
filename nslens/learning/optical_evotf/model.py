"""Composition of zero-context adapters with the unchanged official EvoTransformer."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .embeddings import ADAPTER_NAMES, OpticalContextAdapters
from .metadata import (CANDIDATE_CONTINUOUS_SIZE, ENV_CONTINUOUS_SIZE,
                       MAX_ROUTE_EVENTS, ROUTE_CONTINUOUS_SIZE)


class OpticalEvoTFModel:
    """Add context in feature space, then invoke the exact upstream model and heads."""

    def __init__(
        self,
        strategy,
        *,
        enabled: tuple[str, ...] = ADAPTER_NAMES,
        hidden_dim: int = 32,
        categorical_mode: str = "learned",
    ):
        self.strategy = strategy
        self.enabled = tuple(enabled)
        unknown = set(self.enabled) - set(ADAPTER_NAMES)
        if unknown:
            raise ValueError(f"unknown optical adapters: {sorted(unknown)}")
        self.adapter = OpticalContextAdapters(
            solution_feature_dim=strategy.sf.num_features,
            fitness_feature_dim=strategy.ff.num_features,
            distribution_feature_dim=strategy.df.num_features,
            hidden_dim=hidden_dim,
            enabled=self.enabled,
            categorical_mode=categorical_mode,
        )
        self.hidden_dim = int(hidden_dim)
        self.categorical_mode = categorical_mode
        self.infer = jax.jit(self.apply, static_argnames=("train",))

    def dummy_inputs(self, batch_size: int = 1):
        timesteps = self.strategy.max_context_len
        population = self.strategy.population_size
        dimensions = self.strategy.num_dims
        parameter = {key: jnp.zeros((dimensions,), dtype=jnp.int32)
                     for key in ("name", "kind", "group", "units")}
        environment = {
            "continuous": jnp.zeros(
                (batch_size, timesteps, ENV_CONTINUOUS_SIZE), dtype=jnp.float32),
            "categorical": jnp.zeros((batch_size, timesteps, 2), dtype=jnp.int32),
        }
        optical = {
            "status": jnp.zeros((batch_size, timesteps, population), dtype=jnp.int32),
            "failure": jnp.zeros((batch_size, timesteps, population), dtype=jnp.int32),
            "continuous": jnp.zeros(
                (batch_size, timesteps, population, CANDIDATE_CONTINUOUS_SIZE),
                dtype=jnp.float32),
            "clipping": jnp.zeros(
                (batch_size, timesteps, population, dimensions), dtype=jnp.float32),
            "route_kind": jnp.zeros(
                (batch_size, timesteps, population, MAX_ROUTE_EVENTS), dtype=jnp.int32),
            "route_face": jnp.zeros(
                (batch_size, timesteps, population, MAX_ROUTE_EVENTS), dtype=jnp.int32),
            "route_repeat": jnp.zeros(
                (batch_size, timesteps, population, MAX_ROUTE_EVENTS), dtype=jnp.int32),
            "route_continuous": jnp.zeros(
                (batch_size, timesteps, population, MAX_ROUTE_EVENTS,
                 ROUTE_CONTINUOUS_SIZE), dtype=jnp.float32),
            "route_mask": jnp.zeros(
                (batch_size, timesteps, population, MAX_ROUTE_EVENTS), dtype=jnp.float32),
        }
        generation_mask = jnp.zeros((batch_size, timesteps), dtype=jnp.float32)
        return parameter, environment, optical, generation_mask

    def init_adapter(self, key, batch_size: int = 1):
        return self.adapter.init(key, *self.dummy_inputs(batch_size))

    def apply(
        self,
        *,
        core_params,
        adapter_variables,
        solution_features,
        fitness_features,
        distribution_features,
        parameter,
        environment,
        optical,
        generation_mask,
        mask,
        train: bool = False,
    ):
        adapter_output = self.adapter.apply(
            adapter_variables, parameter, environment, optical, generation_mask)
        prediction, attention = self.strategy.model.apply(
            core_params,
            solution_features + adapter_output["solution_delta"],
            fitness_features + adapter_output["fitness_delta"],
            distribution_features + adapter_output["distribution_delta"],
            rngs={"dropout": jax.random.key(0)},
            train=train,
            mask=mask,
        )
        return prediction, {
            "attention": attention,
            "auxiliary": adapter_output["auxiliary"],
        }


_MODEL_CACHE = {}


def shared_optical_model(
    strategy, *, enabled=ADAPTER_NAMES, hidden_dim=32, categorical_mode="learned"
):
    """Reuse one static model/JIT object across emitter restart lifecycles."""
    key = (id(strategy), tuple(enabled), int(hidden_dim), str(categorical_mode))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = OpticalEvoTFModel(
            strategy, enabled=tuple(enabled), hidden_dim=hidden_dim,
            categorical_mode=categorical_mode)
    return _MODEL_CACHE[key]


__all__ = ["OpticalEvoTFModel", "shared_optical_model"]
