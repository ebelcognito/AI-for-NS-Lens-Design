"""Zero-initialized optical adapters for EvoTF's three native feature streams."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from .metadata import (AGGREGATE_IDS, CANDIDATE_CONTINUOUS_SIZE,
                       ENV_CONTINUOUS_SIZE, FACE_IDS, FAILURE_IDS, GROUP_IDS,
                       KIND_IDS, MAX_ROUTE_EVENTS, NAME_IDS,
                       REPRESENTATION_IDS, ROUTE_CONTINUOUS_SIZE,
                       ROUTE_KIND_IDS, STATUS_IDS, UNIT_IDS)


ADAPTER_NAMES = ("parameter_role", "environment", "status_failure", "route", "archive")


class OpticalContextAdapters(nn.Module):
    """Produce additive deltas while leaving upstream EvoTF parameters untouched."""

    solution_feature_dim: int
    fitness_feature_dim: int
    distribution_feature_dim: int
    hidden_dim: int = 32
    max_route_events: int = MAX_ROUTE_EVENTS
    enabled: tuple[str, ...] = ADAPTER_NAMES
    categorical_mode: str = "learned"

    def setup(self):
        if self.categorical_mode not in ("learned", "one_hot"):
            raise ValueError("categorical_mode must be 'learned' or 'one_hot'")

    def _embed(self, values, size: int, name: str):
        if self.categorical_mode == "one_hot":
            return nn.Dense(self.hidden_dim, name=f"{name}_one_hot")(
                jax.nn.one_hot(values, size))
        return nn.Embed(size, self.hidden_dim, name=name)(values)

    def parameter_encoding(self, parameter):
        if self.categorical_mode == "one_hot":
            pieces = (
                jax.nn.one_hot(parameter["name"], len(NAME_IDS) + 1),
                jax.nn.one_hot(parameter["kind"], len(KIND_IDS) + 1),
                jax.nn.one_hot(parameter["group"], len(GROUP_IDS) + 1),
                jax.nn.one_hot(parameter["units"], len(UNIT_IDS) + 1),
            )
            return nn.tanh(nn.Dense(self.hidden_dim, name="role_one_hot")(
                jnp.concatenate(pieces, axis=-1)))
        encoded = (
            self._embed(parameter["name"], len(NAME_IDS) + 1, "parameter_name")
            + self._embed(parameter["kind"], len(KIND_IDS) + 1, "parameter_kind")
            + self._embed(parameter["group"], len(GROUP_IDS) + 1, "physical_group")
            + self._embed(parameter["units"], len(UNIT_IDS) + 1, "units_normalization")
        )
        return nn.tanh(encoded)

    def route_encoding(self, optical):
        kind = self._embed(
            optical["route_kind"], len(ROUTE_KIND_IDS), "route_kind")
        face = self._embed(optical["route_face"], len(FACE_IDS), "route_face")
        repeat = self._embed(
            optical["route_repeat"], self.max_route_events + 1, "route_repeat")
        continuous = nn.Dense(self.hidden_dim, name="route_continuous")(
            optical["route_continuous"])
        positions = jnp.arange(self.max_route_events, dtype=jnp.int32)
        position = nn.Embed(
            self.max_route_events, self.hidden_dim, name="route_position")(positions)
        event = nn.tanh(kind + face + repeat + continuous + position)
        mask = optical["route_mask"][..., None]
        return jnp.sum(event * mask, axis=-2) / jnp.maximum(
            jnp.sum(mask, axis=-2), 1.0)

    @nn.compact
    def __call__(self, parameter, environment, optical, generation_mask):
        # Shapes: environment [B,T,*], candidate [B,T,P,*], parameter [D].
        batch, timesteps, population = optical["status"].shape
        dimensions = parameter["name"].shape[0]
        generation_mask = generation_mask[..., None]

        role = self.parameter_encoding(parameter)
        if "parameter_role" not in self.enabled:
            role = jnp.zeros_like(role)

        environment_continuous = nn.tanh(nn.Dense(
            self.hidden_dim, name="environment_continuous_1")(
                environment["continuous"]))
        environment_continuous = nn.tanh(nn.Dense(
            self.hidden_dim, name="environment_continuous_2")(
                environment_continuous))
        environment_categorical = (
            self._embed(environment["categorical"][..., 0],
                        len(REPRESENTATION_IDS), "representation")
            + self._embed(environment["categorical"][..., 1],
                          len(AGGREGATE_IDS), "field_aggregate")
        )
        environment_encoded = nn.tanh(
            environment_continuous + environment_categorical)
        if "environment" not in self.enabled:
            environment_encoded = jnp.zeros_like(environment_encoded)

        candidate_continuous = nn.tanh(nn.Dense(
            self.hidden_dim, name="candidate_continuous_1")(
                optical["continuous"][..., :17]))
        candidate_continuous = nn.tanh(nn.Dense(
            self.hidden_dim, name="candidate_continuous_2")(
                candidate_continuous))
        candidate_categorical = (
            self._embed(optical["status"], len(STATUS_IDS), "archive_status")
            + self._embed(optical["failure"], len(FAILURE_IDS), "failure_reason")
        )
        candidate_encoded = nn.tanh(candidate_continuous + candidate_categorical)
        if "status_failure" not in self.enabled:
            candidate_encoded = jnp.zeros_like(candidate_encoded)

        route_encoded = self.route_encoding(optical)
        if "route" not in self.enabled:
            route_encoded = jnp.zeros_like(route_encoded)

        archive_encoded = nn.tanh(nn.Dense(
            self.hidden_dim, name="archive_context_1")(
                optical["continuous"][..., 17:CANDIDATE_CONTINUOUS_SIZE]))
        archive_encoded = nn.tanh(nn.Dense(
            self.hidden_dim, name="archive_context_2")(archive_encoded))
        if "archive" not in self.enabled:
            archive_encoded = jnp.zeros_like(archive_encoded)

        zero = nn.initializers.zeros
        role_solution = nn.Dense(
            self.solution_feature_dim, kernel_init=zero, bias_init=zero,
            name="role_to_solution")(role)
        role_distribution = nn.Dense(
            self.distribution_feature_dim, kernel_init=zero, bias_init=zero,
            name="role_to_distribution")(role)
        candidate_solution = nn.Dense(
            self.solution_feature_dim, kernel_init=zero, bias_init=zero,
            name="candidate_to_solution")(candidate_encoded + route_encoded)
        candidate_fitness = nn.Dense(
            self.fitness_feature_dim, kernel_init=zero, bias_init=zero,
            name="candidate_to_fitness")(candidate_encoded + route_encoded)
        clip_solution = nn.Dense(
            self.solution_feature_dim, kernel_init=zero, bias_init=zero,
            name="clipping_to_solution")(optical["clipping"][..., None])
        generation_distribution = nn.Dense(
            self.distribution_feature_dim, kernel_init=zero, bias_init=zero,
            name="environment_to_distribution")(environment_encoded)
        archive_distribution = nn.Dense(
            self.distribution_feature_dim, kernel_init=zero, bias_init=zero,
            name="archive_to_distribution")(jnp.mean(archive_encoded, axis=2))

        # Gate projections after their Dense layers.  Gating only the encoded input
        # would still let a disabled branch learn its output bias, contaminating the
        # cumulative embedding ladder with an unlabelled generic adapter.
        if "parameter_role" not in self.enabled:
            role_solution = jnp.zeros_like(role_solution)
            role_distribution = jnp.zeros_like(role_distribution)
        if not ({"status_failure", "route"} & set(self.enabled)):
            candidate_solution = jnp.zeros_like(candidate_solution)
            candidate_fitness = jnp.zeros_like(candidate_fitness)
        if "status_failure" not in self.enabled:
            clip_solution = jnp.zeros_like(clip_solution)
        if "environment" not in self.enabled:
            generation_distribution = jnp.zeros_like(generation_distribution)
        if "archive" not in self.enabled:
            archive_distribution = jnp.zeros_like(archive_distribution)

        solution_delta = (
            role_solution[None, None, None, :, :]
            + candidate_solution[:, :, :, None, :]
            + clip_solution
        )
        fitness_delta = candidate_fitness
        distribution_delta = (
            role_distribution[None, None, :, :]
            + generation_distribution[:, :, None, :]
            + archive_distribution[:, :, None, :]
        )
        solution_delta *= generation_mask[:, :, None, None, :]
        fitness_delta *= generation_mask[:, :, None, :]
        distribution_delta *= generation_mask[:, :, None, :]
        auxiliary = {
            "validity_logits": nn.Dense(1, name="validity_head")(
                candidate_encoded).squeeze(-1),
            "rank_prediction": nn.Dense(1, name="rank_head")(
                candidate_encoded).squeeze(-1),
            "route_presence_logits": nn.Dense(1, name="route_presence_head")(
                route_encoded).squeeze(-1),
        }
        return {
            "solution_delta": solution_delta,
            "fitness_delta": fitness_delta,
            "distribution_delta": distribution_delta,
            "auxiliary": auxiliary,
        }


__all__ = ["ADAPTER_NAMES", "OpticalContextAdapters"]
