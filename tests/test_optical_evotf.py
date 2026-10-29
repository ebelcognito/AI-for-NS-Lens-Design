"""Optical-EvoTF equivalence, symmetry, mask, gradient, and checkpoint tests."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from importlib import metadata
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.learning.optical_evotf.metadata import (  # noqa: E402
    encode_observations, parameter_metadata)
from nslens.search.evotf_emitter import EVOSAX_VERSION  # noqa: E402


NAMES = ["ray_y", "ang", "cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b", "n"]


def _runtime():
    try:
        return (importlib.util.find_spec("jax") is not None
                and importlib.util.find_spec("evosax") is not None
                and metadata.version("evosax") == EVOSAX_VERSION)
    except metadata.PackageNotFoundError:
        return False


def _observations(population=4):
    statuses = ("new", "improved", "none", "invalid")
    rows = []
    for index in range(population):
        rows.append({
            "candidate_index": index,
            "status": statuses[index % len(statuses)],
            "rank_percentile": index / max(population - 1, 1),
            "fitness": 0.5 - index * 0.05 if index % 4 != 3 else None,
            "archive_improvement": 0.1 if index % 4 == 1 else None,
            "clipping_mask": [int(gene == index % 9) for gene in range(9)],
            "failure": ({"stage": "invalid_geometry", "reason": "test"}
                        if index % 4 == 3 else None),
            "optical_metrics": {"per_field": [0.2, 0.3], "efficiency": 0.4,
                                "spot": 2.0, "crit_margin": 1.5},
            "archive_context": {"coverage": 0.2, "filled": 4,
                                "cell_was_empty": index == 0},
            "route": {"events": [
                {"kind": "transmission_refraction", "face": "A", "repeat_index": 0},
                {"kind": "tir", "face": "T", "repeat_index": 0},
                {"kind": "transmission_refraction", "face": "B", "repeat_index": 0},
            ]},
        })
    return rows


class MetadataTests(unittest.TestCase):
    def test_parameter_roles_follow_permutation(self):
        permutation = np.array([2, 0, 1, 3, 4, 5, 8, 7, 6])
        original = parameter_metadata(NAMES)
        permuted = parameter_metadata([NAMES[index] for index in permutation])
        for key in original:
            np.testing.assert_array_equal(permuted[key], original[key][permutation])

    def test_absent_route_is_all_padding(self):
        encoded = encode_observations(
            [{"status": "invalid", "clipping_mask": [0] * 9}],
            population_size=1, num_dims=9)
        self.assertFalse(np.any(encoded["route_mask"]))
        self.assertFalse(np.any(encoded["route_kind"]))


@unittest.skipUnless(_runtime(), "optional evosax/JAX runtime not installed")
class OpticalModelTests(unittest.TestCase):
    def _pair(self, *, names=NAMES, seed=12):
        from nslens.learning.optical_evotf.emitter import OpticalEvoTFEmitter
        from nslens.search.evotf_emitter import EvoTFEmitter

        x = np.full(9, 0.5)
        bounds = (np.zeros(9), np.ones(9))
        released = EvoTFEmitter(
            x, 0.12, popsize=4, seed=seed, bounds=bounds, max_context_len=4)
        optical = OpticalEvoTFEmitter(
            x, 0.12, popsize=4, seed=seed, bounds=bounds, max_context_len=4,
            parameter_names=names, environment_context={})
        return released, optical

    @staticmethod
    def _activate_final_projections(emitter):
        from flax.core import freeze, unfreeze
        from flax.traverse_util import flatten_dict, unflatten_dict

        variables = unfreeze(emitter.adapter_variables)
        flat = flatten_dict(variables)
        for path, value in list(flat.items()):
            if path[-1] == "kernel" and any("_to_" in part for part in path):
                flat[path] = np.full_like(value, 0.01)
        emitter.adapter_variables = freeze(unflatten_dict(flat))

    def test_disabled_adapter_projection_biases_cannot_leak(self):
        import jax
        import jax.numpy as jnp
        from evosax.algorithms import EvoTF_ES
        from flax.core import freeze, unfreeze
        from flax.traverse_util import flatten_dict, unflatten_dict
        from nslens.learning.optical_evotf.model import OpticalEvoTFModel

        strategy = EvoTF_ES(
            population_size=4, solution=jnp.zeros(9), max_context_len=2)
        model = OpticalEvoTFModel(strategy, enabled=(), hidden_dim=8)
        variables = unfreeze(model.init_adapter(jax.random.key(9)))
        flat = flatten_dict(variables)
        for path, value in list(flat.items()):
            if any("_to_" in part for part in path):
                flat[path] = np.full_like(value, 0.1)
        variables = freeze(unflatten_dict(flat))
        parameter, environment, optical, mask = model.dummy_inputs()
        mask = mask.at[0, 0].set(1.0)
        output = model.adapter.apply(
            variables, parameter, environment, optical, mask)
        for name in ("solution_delta", "fitness_delta", "distribution_delta"):
            np.testing.assert_array_equal(np.asarray(output[name]), 0.0)

    def test_one_hot_categorical_control_initializes_at_zero(self):
        import jax
        import jax.numpy as jnp
        from evosax.algorithms import EvoTF_ES
        from nslens.learning.optical_evotf.model import OpticalEvoTFModel

        strategy = EvoTF_ES(
            population_size=4, solution=jnp.zeros(9), max_context_len=2)
        model = OpticalEvoTFModel(
            strategy, categorical_mode="one_hot", hidden_dim=8)
        inputs = model.dummy_inputs()
        output = model.adapter.apply(
            model.init_adapter(jax.random.key(10)), *inputs)
        for name in ("solution_delta", "fitness_delta", "distribution_delta"):
            np.testing.assert_array_equal(np.asarray(output[name]), 0.0)

    def test_zero_adapters_reproduce_official_update(self):
        released, optical = self._pair()
        # Cross the four-generation context boundary as well as checking one update.
        for generation in range(6):
            released_population, optical_population = released.ask(), optical.ask()
            if generation == 0:
                np.testing.assert_array_equal(released_population, optical_population)
            else:
                np.testing.assert_allclose(
                    released_population, optical_population, atol=1e-6, rtol=1e-6)
            order = np.roll(np.array([2, 0, 3, 1]), generation)
            released.tell(order)
            optical.tell(order, _observations())
            np.testing.assert_allclose(
                optical.mean, released.mean, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(
                optical.std, released.std, atol=1e-6, rtol=1e-6)

    def test_population_permutation_invariance_with_active_context(self):
        import jax.numpy as jnp

        _, emitter = self._pair(seed=21)
        self._activate_final_projections(emitter)
        parameter, environment, optical, generation_mask = (
            emitter.optical_model.dummy_inputs())
        rng = np.random.default_rng(21)
        solution = jnp.asarray(rng.normal(size=(
            1, 4, 4, 9, emitter._strategy.sf.num_features)), dtype=jnp.float32)
        fitness = jnp.asarray(rng.normal(size=(
            1, 4, 4, emitter._strategy.ff.num_features)), dtype=jnp.float32)
        distribution = jnp.asarray(rng.normal(size=(
            1, 4, 9, emitter._strategy.df.num_features)), dtype=jnp.float32)
        observations = encode_observations(
            _observations(), population_size=4, num_dims=9)
        optical = {
            key: value.at[0, 0].set(jnp.asarray(observations[key]))
            for key, value in optical.items()
        }
        generation_mask = generation_mask.at[0, 0].set(1.0)
        permutation = np.array([2, 0, 3, 1])

        def infer(solution_features, fitness_features, optical_features):
            prediction, _ = emitter.optical_model.infer(
                core_params=emitter.core_variables,
                adapter_variables=emitter.adapter_variables,
                solution_features=solution_features,
                fitness_features=fitness_features,
                distribution_features=distribution,
                parameter=parameter,
                environment=environment,
                optical=optical_features,
                generation_mask=generation_mask,
                mask=emitter._strategy.la_mask,
                train=False,
            )
            return np.asarray(prediction)

        left = infer(solution, fitness, optical)
        permuted_optical = {
            key: value[:, :, permutation] for key, value in optical.items()
        }
        right = infer(
            solution[:, :, permutation], fitness[:, :, permutation],
            permuted_optical)
        np.testing.assert_allclose(left, right, atol=2e-6, rtol=2e-6)

    def test_gene_metadata_equivariance_with_active_context(self):
        permutation = np.array([2, 0, 1, 5, 4, 3, 8, 6, 7])
        _, left = self._pair(seed=31)
        _, right = self._pair(
            seed=31, names=[NAMES[index] for index in permutation])
        self._activate_final_projections(left)
        self._activate_final_projections(right)
        left.ask()
        right.ask()
        right._last_population = left._last_population[:, permutation]
        observations = _observations()
        permuted = []
        for row in observations:
            row = json.loads(json.dumps(row))
            row["clipping_mask"] = np.asarray(row["clipping_mask"])[permutation].tolist()
            permuted.append(row)
        order = np.array([0, 1, 2, 3])
        left.tell(order, observations)
        right.tell(order, permuted)
        np.testing.assert_allclose(right.mean, left.mean[permutation], atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(right.std, left.std[permutation], atol=2e-6, rtol=2e-6)

    def test_route_order_sensitive_and_padding_mask_exact(self):
        import jax.numpy as jnp

        _, emitter = self._pair(seed=41)
        original = _observations()
        reversed_rows = json.loads(json.dumps(original))
        reversed_rows[0]["route"]["events"].reverse()
        absent = json.loads(json.dumps(original))
        absent[0]["route"]["events"] = []

        def route(rows):
            encoded = encode_observations(rows, population_size=4, num_dims=9)
            parameter, environment, optical, mask = emitter.optical_model.dummy_inputs()
            optical = {
                key: value.at[0, 0].set(jnp.asarray(encoded[key]))
                for key, value in optical.items()
            }
            mask = mask.at[0, 0].set(1.0)
            output = emitter.optical_model.adapter.apply(
                emitter.adapter_variables, parameter, environment, optical, mask)
            return np.asarray(output["auxiliary"]["route_presence_logits"])

        route_original = route(original)
        route_reversed = route(reversed_rows)
        self.assertFalse(np.allclose(route_original[0, 0, 0], route_reversed[0, 0, 0]))
        route_absent = route(absent)
        padded = json.loads(json.dumps(absent))
        padded[0]["route"]["events"] = [
            {"kind": "tir", "face": "D", "repeat_index": 7, "masked": True}]
        # A masked event is represented by arbitrary padded values with a zero mask.
        encoded_padding = encode_observations(absent, population_size=4, num_dims=9)
        encoded_padding["route_kind"][0, 0] = 2
        encoded_padding["route_face"][0, 0] = 4
        parameter, environment, optical, mask = emitter.optical_model.dummy_inputs()
        optical = {
            key: value.at[0, 0].set(jnp.asarray(encoded_padding[key]))
            for key, value in optical.items()
        }
        mask = mask.at[0, 0].set(1.0)
        route_padding = np.asarray(emitter.optical_model.adapter.apply(
            emitter.adapter_variables, parameter, environment, optical, mask
        )["auxiliary"]["route_presence_logits"])
        np.testing.assert_array_equal(route_absent[0, 0, 0], route_padding[0, 0, 0])

    def test_finite_gradients_tiny_overfit_and_checkpoint_roundtrip(self):
        import jax
        import jax.numpy as jnp
        import optax
        from nslens.learning.optical_evotf.checkpoint import (
            RUNTIME_VERSIONS, load_checkpoint, save_checkpoint)

        _, emitter = self._pair(seed=51)
        parameter, environment, optical, mask = emitter.optical_model.dummy_inputs()
        environment = dict(environment)
        optical = dict(optical)
        mask = mask.at[0, 0].set(1.0)
        optical["status"] = optical["status"].at[0, 0].set(
            jnp.asarray([1, 2, 3, 4]))
        target = jnp.full(
            (1, 4, 9, emitter._strategy.sf.num_features), 0.05)

        def loss(variables):
            output = emitter.optical_model.adapter.apply(
                variables, parameter, environment, optical, mask)
            return jnp.mean(jnp.square(output["solution_delta"][:, :1] - target))

        optimizer = optax.adam(0.005)
        variables = emitter.adapter_variables
        state = optimizer.init(variables)
        initial = float(loss(variables))

        @jax.jit
        def train_step(current, optimizer_state):
            value, gradients = jax.value_and_grad(loss)(current)
            updates, optimizer_state = optimizer.update(
                gradients, optimizer_state, current)
            return (optax.apply_updates(current, updates), optimizer_state,
                    value, gradients)

        for _ in range(100):
            variables, state, value, gradients = train_step(variables, state)
            self.assertTrue(all(np.all(np.isfinite(np.asarray(leaf)))
                                for leaf in jax.tree.leaves(gradients)))
        self.assertLess(float(loss(variables)), initial * 0.2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_checkpoint(
                path, adapter_variables=variables,
                enabled=tuple(emitter.enabled_adapters), hidden_dim=emitter.hidden_dim,
                categorical_mode=emitter.categorical_mode, max_context_len=4,
                training={"validation_metrics": {"loss": float(value)},
                          "test_metrics_read": False})
            loaded = load_checkpoint(path)
            self.assertEqual(loaded["manifest"]["runtime"], RUNTIME_VERSIONS)
            left = jax.tree.leaves(variables)
            right = jax.tree.leaves(loaded["adapter"])
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


if __name__ == "__main__":
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        print("ALL OPTICAL EVOTF CHECKS PASS")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
