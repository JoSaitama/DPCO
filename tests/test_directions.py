import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.methods import create_method
from src.methods.base import aggregate, mean_loss
from src.methods.random_flip import RandomFlip
from src.training.randomness import training_key
from tests.helpers import QuadraticTask, scalar_batch


class DirectionTests(unittest.TestCase):
    def test_batch_mean_and_loo_have_distinct_analytic_directions(self):
        task = QuadraticTask()
        params = {"x": jnp.zeros(1)}
        expected = {"sgd": 1/6, "sage": 7/6, "dtv": 2/3,
                    "sage_loo": -1/6, "dtv_loo": 0.}
        for name, value in expected.items():
            with self.subTest(method=name):
                result = create_method(name).compute_direction(task, params, scalar_batch())
                np.testing.assert_allclose(result.direction["x"], [value], atol=1e-6)
                self.assertEqual(result.model_state["calls"], 1)
                self.assertEqual(int(result.metrics["valid_batch_size"]), 3)

    def test_filter_retains_original_batch_denominator(self):
        # [1,1,-1.5] -> filter last -> (1+1)/3, not (1+1)/2.
        method = create_method("dtv")
        result = method.compute_direction(QuadraticTask(), {"x": jnp.zeros(1)}, scalar_batch())
        self.assertAlmostEqual(float(result.direction["x"][0]), 2/3, places=6)

    def test_padding_and_frozen_weights(self):
        mask = jnp.array([True, False, True])
        gradients = {"x": jnp.array([[2.], [jnp.nan], [4.]])}
        np.testing.assert_allclose(aggregate(gradients, jnp.ones(3), mask)["x"], [3.])
        derivative = jax.grad(lambda w: jnp.sum(aggregate(gradients, w, mask)["x"]))(jnp.ones(3))
        np.testing.assert_array_equal(derivative, jnp.zeros(3))
        derivative = jax.grad(lambda w: mean_loss(jnp.array([2., 99., 4.]), w, mask))(jnp.ones(3))
        np.testing.assert_array_equal(derivative, jnp.zeros(3))

    def test_raw_gradient_strength_is_not_unit_normalized(self):
        gradients = {"x": jnp.array([[4., 0.], [0., 2.]])}
        direction = aggregate(gradients, jnp.ones(2), jnp.array([True, True]))
        np.testing.assert_allclose(direction["x"], [2., 1.])

    def test_random_endpoints_flip_all_samples_not_only_negative(self):
        task, params, batch = QuadraticTask(), {"x": jnp.zeros(1)}, scalar_batch()
        base = create_method("sgd").compute_direction(task, params, batch).direction["x"]
        for p, sign in ((0., 1.), (1., -1.)):
            method = create_method("random_flip", reference_name="unused", flip_probability=p)
            result = method.compute_direction(task, params, batch, rng_key=training_key(7, 0))
            np.testing.assert_allclose(result.direction["x"], sign * base, atol=1e-6)
        self.assertFalse(np.allclose(-base, create_method("sage").compute_direction(task, params, batch).direction["x"]))

    def test_random_probability_and_same_seed_reproducibility(self):
        method = RandomFlip(0.25)
        mask = jnp.ones(6, dtype=jnp.bool_)
        keys = jax.random.split(training_key(13, 0), 4096)
        weights = jax.vmap(lambda key: method.weights(mask, key))(keys)
        weights_again = jax.vmap(lambda key: method.weights(mask, key))(keys)
        np.testing.assert_array_equal(weights, weights_again)
        proportion = float(jnp.mean(weights < 0))
        self.assertLess(abs(proportion - 0.25), 0.015)
        counts = np.asarray(jnp.sum(weights < 0, axis=1))
        self.assertGreater(len(np.unique(counts)), 2)
        # Probability applies to every position; no score/reference is supplied.
        np.testing.assert_allclose(jnp.mean(weights < 0, axis=0), jnp.full(6, .25), atol=.035)

    def test_random_requires_key_and_ignores_padding(self):
        method = RandomFlip(1.)
        with self.assertRaises(ValueError):
            method.weights(jnp.array([True, True]), None)
        np.testing.assert_array_equal(
            method.weights(jnp.array([True, False]), training_key(1, 0)), [-1., 0.]
        )

    def test_step_and_stream_keys_are_explicit(self):
        same = (training_key(5, 3), training_key(5, 3))
        np.testing.assert_array_equal(jax.random.key_data(same[0]), jax.random.key_data(same[1]))
        self.assertFalse(np.array_equal(jax.random.key_data(same[0]), jax.random.key_data(training_key(5, 4))))
        self.assertFalse(np.array_equal(jax.random.key_data(same[0]), jax.random.key_data(training_key(5, 3, stream=1))))

    def test_invalid_reference_is_only_relevant_to_alignment_methods(self):
        self.assertIsNotNone(create_method("sgd", reference_name="not_needed"))
        self.assertIsNotNone(create_method("random_flip", reference_name="not_needed"))
        with self.assertRaises(ValueError):
            create_method("sage", reference_name="not_implemented")
