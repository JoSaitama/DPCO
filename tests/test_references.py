import unittest

import jax.numpy as jnp
import numpy as np

from src.methods import create_method
from src.references import create_reference
from src.signals.gradient_alignment import gradient_alignment
from tests.helpers import QuadraticTask


class ReferenceTests(unittest.TestCase):
    def test_full_batch_mean_and_self_exclusion(self):
        gradients = {"x": jnp.array([[1.], [1.], [-1.5]])}
        mask = jnp.ones(3, dtype=jnp.bool_)
        reference = create_reference("loss")
        common = reference.build(gradients, mask, {})
        np.testing.assert_allclose(common.direction["x"], [1/6], atol=1e-6)
        np.testing.assert_allclose(gradient_alignment(gradients, common, mask), [1/6, 1/6, -.25], atol=1e-6)
        loo = reference.build(gradients, mask, {}, exclude_self=True)
        np.testing.assert_allclose(loo.direction["x"], [[-.25], [-.25], [1.]])
        np.testing.assert_allclose(gradient_alignment(gradients, loo, mask), [-.25, -.25, -1.5])

    def test_error_reference_uses_surrogate_gradients_and_excludes_only_selected_self(self):
        gradients = {"x": jnp.array([[2.], [4.], [8.]])}
        mask = jnp.ones(3, dtype=jnp.bool_)
        feedback = {"error_mask": jnp.array([True, False, True])}
        reference = create_reference("error")
        common = reference.build(gradients, mask, feedback)
        np.testing.assert_allclose(common.direction["x"], [5.])
        loo = reference.build(gradients, mask, feedback, exclude_self=True)
        np.testing.assert_allclose(loo.direction["x"], [[8.], [5.], [2.]])
        np.testing.assert_array_equal(loo.selected_count, [1., 2., 1.])

    def test_reference_ignores_nan_padding(self):
        gradients = {"x": jnp.array([[2.], [jnp.nan], [8.]])}
        mask = jnp.array([True, False, True])
        result = create_reference("loss").build(gradients, mask, {}, exclude_self=True)
        np.testing.assert_allclose(result.direction["x"], [[8.], [5.], [2.]])

    def test_empty_error_reference_keeps_existing_no_correction_fallback(self):
        task = QuadraticTask()
        params = {"x": jnp.zeros(1)}
        batch = (jnp.array([[-1.], [2.]]), jnp.ones(2, dtype=jnp.bool_), jnp.zeros(2, dtype=jnp.bool_))
        expected = create_method("sgd").compute_direction(task, params, batch).direction
        for name in ("sage", "sage_loo", "dtv", "dtv_loo"):
            actual = create_method(name, reference_name="error").compute_direction(task, params, batch).direction
            np.testing.assert_allclose(actual["x"], expected["x"])

    def test_singleton_loo_is_flagged_and_retains_uncorrected_contribution(self):
        gradients = {"x": jnp.array([[2.]])}
        reference = create_reference("loss").build(gradients, jnp.array([True]), {}, exclude_self=True)
        np.testing.assert_array_equal(reference.has_reference, [False])
        np.testing.assert_array_equal(reference.selected_count, [0.])

    def test_error_reference_requires_explicit_feedback(self):
        with self.assertRaises(ValueError):
            create_reference("error").build({"x": jnp.ones((2, 1))}, jnp.array([True, True]), {})
