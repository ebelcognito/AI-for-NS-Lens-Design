"""Online Optical-EvoTF emitter using official features, core, and update heads."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...search.evotf_emitter import EvoTFEmitter, stable_rank_fitness
from .checkpoint import load_checkpoint
from .embeddings import ADAPTER_NAMES
from .metadata import encode_environment, encode_observations, parameter_metadata
from .model import shared_optical_model


class OpticalEvoTFEmitter(EvoTFEmitter):
    """Released EvoTF with narrow additive optical-context input adapters."""

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: float,
        *,
        popsize: Optional[int] = None,
        seed: int = 0,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_context_len: int = 100,
        parameter_names: Sequence[str],
        environment_context: Mapping[str, Any],
        checkpoint_path: Optional[Path] = None,
        enabled_adapters: tuple[str, ...] = ADAPTER_NAMES,
        hidden_dim: int = 32,
        categorical_mode: str = "learned",
    ):
        super().__init__(
            x0, sigma0, popsize=popsize, seed=seed, bounds=bounds,
            max_context_len=max_context_len)
        if len(parameter_names) != self.N:
            raise ValueError("parameter_names must match the search dimension")

        loaded = load_checkpoint(checkpoint_path) if checkpoint_path else None
        if loaded:
            model_manifest = loaded["manifest"]["model"]
            training_manifest = loaded["manifest"].get("training") or {}
            if int(model_manifest.get("max_context_len", max_context_len)) != max_context_len:
                raise ValueError("max_context_len must match the selected checkpoint")
            if int(training_manifest.get("population_size", self.lam)) != self.lam:
                raise ValueError("population size must match the selected checkpoint")
            if int(training_manifest.get("dimensions", self.N)) != self.N:
                raise ValueError("search dimension must match the selected checkpoint")
            manifest_enabled = tuple(model_manifest["enabled_adapters"])
            if tuple(enabled_adapters) != manifest_enabled:
                raise ValueError(
                    "enabled_adapters must match the selected checkpoint manifest")
            hidden_dim = int(model_manifest["hidden_dim"])
            manifest_categorical = str(model_manifest["categorical_mode"])
            if categorical_mode != manifest_categorical:
                raise ValueError(
                    "categorical_mode must match the selected checkpoint manifest")
        self.optical_model = shared_optical_model(
            self._strategy, enabled=tuple(enabled_adapters), hidden_dim=hidden_dim,
            categorical_mode=categorical_mode)
        adapter_key = self._jax.random.key(int(seed) ^ 0x45A1D3)
        self.adapter_variables = self.optical_model.init_adapter(adapter_key)
        self.core_variables = self._params.params
        self.checkpoint_path = str(Path(checkpoint_path).resolve()) if checkpoint_path else None
        if loaded:
            self.adapter_variables = loaded["adapter"]
            self.core_variables = loaded.get("core", self.core_variables)

        self.enabled_adapters = tuple(enabled_adapters)
        self.hidden_dim = int(hidden_dim)
        self.categorical_mode = categorical_mode
        self.parameter_context = {
            key: self._jnp.asarray(value)
            for key, value in parameter_metadata(parameter_names).items()
        }
        encoded_environment = encode_environment(environment_context)
        self.environment_context = {
            "continuous": self._jnp.repeat(
                self._jnp.asarray(encoded_environment["continuous"])[None, None, :],
                self.max_context_len, axis=1),
            "categorical": self._jnp.repeat(
                self._jnp.asarray(encoded_environment["categorical"])[None, None, :],
                self.max_context_len, axis=1),
        }
        _, _, dummy_optical, _ = self.optical_model.dummy_inputs()
        self.optical_context = dummy_optical
        self.generation_mask = self._jnp.zeros((1, self.max_context_len), dtype=self._jnp.float32)

    def _append_optical_context(self, observations, generation: int) -> int:
        encoded = encode_observations(
            observations or [], population_size=self.lam, num_dims=self.N)
        shift = generation >= self.max_context_len
        index = self.max_context_len - 1 if shift else generation
        updated = {}
        for key, history in self.optical_context.items():
            if shift:
                history = history.at[0, :-1].set(history[0, 1:])
            updated[key] = history.at[0, index].set(self._jnp.asarray(encoded[key]))
        generation_mask = self.generation_mask
        if shift:
            generation_mask = generation_mask.at[0, :-1].set(generation_mask[0, 1:])
        self.generation_mask = generation_mask.at[0, index].set(1.0)
        self.optical_context = updated
        return index

    @staticmethod
    def _append_numeric(history, features, generation: int, max_context_len: int):
        shift = generation >= max_context_len
        index = max_context_len - 1 if shift else generation
        if shift:
            history = history.at[0, :-1].set(history[0, 1:])
        return history.at[0, index].set(features), index

    def tell(
        self,
        order: Sequence[int],
        observations: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        if self._last_population is None:
            raise RuntimeError("tell() called before ask()")
        fitness = self._jnp.asarray(stable_rank_fitness(order, self.lam))
        started = time.perf_counter()
        # Match EvoTFEmitter's PRNG protocol even though inference is deterministic.
        # Without consuming the official tell key, the next ask would sample a
        # different population and zero-adapter equivalence would stop after one step.
        self._key, _tell_key = self._jax.random.split(self._key)
        state = self._state
        generation = int(np.asarray(state.generation_counter))

        # This is the public evosax tell wrapper's best-state update, followed by the
        # official EvoTF featurizers. Only model inputs below receive adapter deltas.
        from evosax.algorithms.base import update_best_solution_and_fitness
        best_solution, best_fitness = update_best_solution_and_fitness(
            self._last_population, fitness, state.best_solution, state.best_fitness)
        state = state.replace(best_solution=best_solution, best_fitness=best_fitness)
        sfeatures, sf_state = self._strategy.sf.featurize(
            self._last_population, fitness, state.mean, state.std, state.sf_state)
        ffeatures, ff_state = self._strategy.ff.featurize(
            self._last_population, fitness, state.ff_state)
        dfeatures, df_state = self._strategy.df.featurize(
            self._last_population, fitness, state.mean, state.std, state.df_state)

        solution_context, index = self._append_numeric(
            state.solution_context, sfeatures, generation, self.max_context_len)
        fitness_context, _ = self._append_numeric(
            state.fitness_context, ffeatures, generation, self.max_context_len)
        distribution_context, _ = self._append_numeric(
            state.distribution_context, dfeatures, generation, self.max_context_len)
        optical_index = self._append_optical_context(observations, generation)
        if optical_index != index:
            raise AssertionError("numeric and optical context windows are misaligned")

        prediction, _ = self.optical_model.infer(
            core_params=self.core_variables,
            adapter_variables=self.adapter_variables,
            solution_features=solution_context,
            fitness_features=fitness_context,
            distribution_features=distribution_context,
            parameter=self.parameter_context,
            environment=self.environment_context,
            optical=self.optical_context,
            generation_mask=self.generation_mask,
            mask=self._strategy.la_mask,
            train=False,
        )
        mean = state.mean + self._params.lr_mean * state.std * prediction[0, 0, index]
        std = state.std * self._jnp.exp(
            self._params.lr_std / 2 * prediction[1, 0, index])
        std = self._jnp.clip(std, 1e-8)
        self._state = state.replace(
            mean=mean, std=std, sf_state=sf_state, ff_state=ff_state,
            df_state=df_state, solution_context=solution_context,
            fitness_context=fitness_context, distribution_context=distribution_context,
            generation_counter=state.generation_counter + 1,
        )
        self._jax.block_until_ready(self._state.mean)
        self._last_population = None
        self._telemetry.tell_calls += 1
        self._telemetry.inference_seconds += time.perf_counter() - started
        if not self._state_is_finite():
            self._mark_numerical_failure("non-finite Optical-EvoTF distribution state")

    def state_summary(self) -> dict[str, Any]:
        summary = super().state_summary()
        summary.update({
            "algorithm": ("optical_evotf_es" if self.enabled_adapters
                          else "trained_numeric_evotf_es"),
            "enabled_adapters": list(self.enabled_adapters),
            "categorical_mode": self.categorical_mode,
            "adapter_hidden_dim": self.hidden_dim,
            "checkpoint_path": self.checkpoint_path,
        })
        return summary


__all__ = ["OpticalEvoTFEmitter"]
