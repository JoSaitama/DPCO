import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.data import classification_batch
from src.methods import create_method
from src.tasks.classification import ClassificationTask
from src.training import TrainState, Trainer
from src.training.updates import SGDUpdate


def linear_model(params, images, state):
    next_state = {"calls": (0 if state is None else state["calls"]) + 1}
    return images @ params["w"], next_state


class ClassificationTests(unittest.TestCase):
    def test_cross_entropy_direction_matches_analytic_softmax_gradient(self):
        task = ClassificationTask(linear_model, 3)
        batch = classification_batch(jnp.array([[1., 0.], [0., 2.]]), jnp.array([0, 2]))
        params = {"w": jnp.zeros((2, 3))}
        result = create_method("sgd").compute_direction(task, params, batch)
        expected = jnp.array([[-1/3, 1/6, 1/6], [1/3, 1/3, -2/3]])
        np.testing.assert_allclose(result.direction["w"], expected, atol=1e-6)
        evaluated = task.evaluate(params, batch)
        np.testing.assert_allclose(evaluated.losses, jnp.full(2, jnp.log(3.)), atol=1e-6)
        np.testing.assert_array_equal(evaluated.feedback["error_mask"], [False, True])

    def test_padding_does_not_change_valid_batch_direction(self):
        task = ClassificationTask(linear_model, 3)
        params = {"w": jnp.zeros((2, 3))}
        short = classification_batch(jnp.array([[1., 0.], [0., 2.]]), jnp.array([0, 2]))
        padded = classification_batch(jnp.array([[1., 0.], [jnp.nan, jnp.nan], [0., 2.]]),
                                      jnp.array([0, 999, 2]), jnp.array([True, False, True]))
        for name in ("sgd", "dpco", "dpco_loo", "dtv", "dtv_loo"):
            method = create_method(name)
            left = method.compute_direction(task, params, short).direction
            right = method.compute_direction(task, params, padded).direction
            np.testing.assert_allclose(left["w"], right["w"], atol=1e-6)

    def test_different_class_counts_share_task_and_trainer_interfaces(self):
        for count in (100, 1000):
            task = ClassificationTask(linear_model, count)
            batch = classification_batch(jnp.ones((2, 2)), jnp.array([0, count-1]))
            state = TrainState({"w": jnp.zeros((2, count))})
            trainer = Trainer(task, create_method("sgd"), SGDUpdate(.1))
            result, _ = jax.jit(trainer.step)(state, batch)
            self.assertEqual(result.params["w"].shape, (2, count))
            self.assertTrue(bool(jnp.all(jnp.isfinite(result.params["w"]))))

    def test_batch_and_model_shapes_are_checked(self):
        with self.assertRaises(ValueError):
            classification_batch(jnp.ones((2, 2)), jnp.array([1]))
        with self.assertRaises(ValueError):
            classification_batch(jnp.ones((2, 2)), jnp.array([0, 1]), jnp.array([False, False]))
        task = ClassificationTask(linear_model, 3)
        batch = classification_batch(jnp.ones((2, 2)), jnp.array([0, 1]))
        with self.assertRaises(ValueError):
            task.evaluate({"w": jnp.zeros((2, 2))}, batch)
