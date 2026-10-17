import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.methods import create_method
from src.training import TrainState, Trainer
from src.training.randomness import training_key
from src.training.updates import SGDUpdate
from tests.helpers import QuadraticTask, scalar_batch


class TrainerTests(unittest.TestCase):
    def test_learning_rate_applied_once_without_second_batch_division(self):
        trainer = Trainer(QuadraticTask(), create_method("sage"), SGDUpdate(.1))
        initial = TrainState({"x": jnp.zeros(1)}, {"calls": 0})
        result, metrics = trainer.step(initial, scalar_batch())
        np.testing.assert_allclose(result.params["x"], [-.1 * 7/6], atol=1e-6)
        self.assertEqual(result.step, 1)
        self.assertEqual(result.model_state["calls"], 1)
        self.assertEqual(initial.model_state["calls"], 0)
        np.testing.assert_array_equal(initial.params["x"], [0.])
        self.assertEqual(metrics, {})

    def test_all_six_methods_use_same_trainer_and_task(self):
        for name in ("sgd", "sage", "sage_loo", "dtv", "dtv_loo", "random_flip"):
            with self.subTest(method=name):
                trainer = Trainer(QuadraticTask(), create_method(name), SGDUpdate(.1))
                initial = TrainState({"x": jnp.zeros(1)}, {"calls": 0})
                result, _ = trainer.step(initial, scalar_batch(), training_key(2, 0))
                self.assertEqual(result.step, 1)
                self.assertEqual(result.model_state["calls"], 1)

    def test_step_is_jittable_and_matches_eager(self):
        for name in ("sgd", "sage_loo", "dtv", "random_flip"):
            with self.subTest(method=name):
                trainer = Trainer(QuadraticTask(), create_method(name), SGDUpdate(.1))
                initial = TrainState({"x": jnp.zeros(1)}, {"calls": 0})
                key = training_key(3, 0)
                eager, _ = trainer.step(initial, scalar_batch(), key)
                compiled, _ = jax.jit(trainer.step)(initial, scalar_batch(), key)
                np.testing.assert_allclose(compiled.params["x"], eager.params["x"], atol=1e-6)
                self.assertEqual(int(compiled.model_state["calls"]), 1)

    def test_step_schedule_and_state_are_explicit(self):
        trainer = Trainer(QuadraticTask(), create_method("sgd"), SGDUpdate(lambda t: .1/(t+1)))
        state = TrainState({"x": jnp.zeros(1)}, {"calls": 0}, {"token": 7}, None, 1)
        result, _ = trainer.step(state, scalar_batch())
        np.testing.assert_allclose(result.params["x"], [-.05/6], atol=1e-6)
        self.assertEqual(result.method_state, {"token": 7})
