"""Schedules, momentum recurrence, state restoration and zero-step probes."""

from dataclasses import replace
import csv
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.config import ProbeConfig, TrainingConfig
from src.training.experiment import ExperimentRunner
from src.training.learning_rates import CosineLearningRate, learning_rate_for
from src.training.updates import MomentumState, SGDUpdate, make_updater
from tests.test_observation import ROOT, fixture_components, fixture_config


class OptimizerOptionTests(unittest.TestCase):
    def test_constant_mode_retains_scalar_and_original_update_graph(self):
        training = TrainingConfig(learning_rate=.1)
        self.assertEqual(learning_rate_for(training), .1)
        updater = make_updater(training)
        params = {"w": jnp.ones((2, 2))}
        direction = {"w": jnp.full((2, 2), 3.)}
        original = lambda p, d: jax.tree_util.tree_map(lambda x, g: x - .1 * g, p, d)
        actual = lambda p, d: updater.apply(p, d, 0).params
        self.assertEqual(str(jax.make_jaxpr(original)(params, direction)),
                         str(jax.make_jaxpr(actual)(params, direction)))
        self.assertIsNone(updater.init(params))

    def test_cosine_endpoints_floors_and_host_device_agreement(self):
        for floor in (0., .001, .1):
            with self.subTest(floor=floor):
                schedule = CosineLearningRate(.1, floor, 5)
                host = np.array([schedule(t) for t in range(7)])
                device = np.array(jax.jit(jax.vmap(schedule))(jnp.arange(7)))
                np.testing.assert_allclose(host, device, atol=1e-8)
                np.testing.assert_allclose(host[[0, 2, 4, 6]], [.1, (.1 + floor) / 2, floor, floor])
                self.assertTrue(np.all(np.diff(host) <= 0))
        single = CosineLearningRate(.1, 0., 1)
        np.testing.assert_allclose([single(0), single(1)], [.1, 0.])

    def test_momentum_and_nesterov_match_independent_multistep_recurrence(self):
        for nesterov in (False, True):
            with self.subTest(nesterov=nesterov):
                training = TrainingConfig(learning_rate=.1, total_steps=4, momentum=.9,
                    nesterov=nesterov, l2_coefficient=.2, lr_schedule="cosine", min_learning_rate=.001)
                updater = make_updater(training)
                params = {"w": jnp.full((2, 2), 2.), "b": jnp.ones(2)}
                state = updater.init(params)
                expected = jax.tree_util.tree_map(np.array, params)
                velocity = jax.tree_util.tree_map(np.zeros_like, expected)
                apply = jax.jit(updater.apply)
                for step in range(4):
                    direction = {"w": jnp.full((2, 2), step - 1.), "b": jnp.full(2, .5)}
                    regularized = {k: np.asarray(direction[k]) + (.2 * p if p.ndim > 1 else 0.)
                                   for k, p in expected.items()}
                    velocity = {k: .9 * velocity[k] + regularized[k] for k in velocity}
                    used = ({k: regularized[k] + .9 * velocity[k] for k in velocity}
                            if nesterov else velocity)
                    eta = updater.learning_rate(step)
                    expected = {k: expected[k] - eta * used[k] for k in expected}
                    params, state = apply(params, direction, step, state)
                    for k in expected:
                        np.testing.assert_allclose(params[k], expected[k], rtol=2e-6, atol=2e-7)
                        np.testing.assert_allclose(state.velocity[k], velocity[k], rtol=2e-6, atol=2e-7)

    def test_invalid_optimizer_options_and_momentum_probes_fail_validation(self):
        with TemporaryDirectory() as tmp:
            c = fixture_config(tmp)
            invalid = ({"momentum": -1.}, {"momentum": 1.}, {"momentum": float("nan")},
                       {"momentum": True}, {"nesterov": True}, {"nesterov": 1},
                       {"lr_schedule": "unknown"}, {"min_learning_rate": -.1},
                       {"min_learning_rate": .001}, {"lr_schedule": "cosine", "min_learning_rate": 1.},
                       {"lr_schedule": "cosine", "min_learning_rate": float("nan")})
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    replace(c, training=replace(c.training, **options)).validate()
            with self.assertRaisesRegex(ValueError, "momentum=0"):
                replace(c, training=replace(c.training, momentum=.9), probes=ProbeConfig(enabled=True)).validate()
        for beta in (-.1, 1., float("inf")):
            with self.assertRaises(ValueError):
                SGDUpdate(.1, momentum=beta)

    def test_momentum_cosine_and_l2_resume_exactly_and_log_applied_rates(self):
        for nesterov in (False, True):
            with self.subTest(nesterov=nesterov), TemporaryDirectory() as tmp:
                c = fixture_config(tmp)
                c = replace(c, training=replace(c.training, momentum=.9, nesterov=nesterov,
                    l2_coefficient=.0001, lr_schedule="cosine", min_learning_rate=.0001))
                trainer, source, evaluation, initial, _ = fixture_components(c)
                runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="options")
                final = runner.run(initial)
                self.assertIsInstance(runner.store.load("initial")["train_state"].optimizer_state, MomentumState)
                self.assertIsInstance(final.optimizer_state, MomentumState)
                with (runner.paths.files / "train_steps.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                expected_rates = [trainer.updater.learning_rate(t) for t in range(4)]
                np.testing.assert_allclose([float(r["learning_rate"]) for r in rows], expected_rates)
                _, source2, evaluation2, _, _ = fixture_components(c)
                resumed = ExperimentRunner(c, trainer, source2, evaluation2, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="options", resume=True)
                restored = resumed.run(initial, resume_slot="p50")
                for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(restored)):
                    np.testing.assert_array_equal(a, b)
                for field, value in (("momentum", .8), ("lr_schedule", "constant")):
                    different = replace(c.training, **{field: value})
                    if field == "lr_schedule":
                        different = replace(different, min_learning_rate=0.)
                    with self.assertRaisesRegex(ValueError, "configuration differs"):
                        ExperimentRunner(replace(c, training=different), trainer, source2, evaluation2,
                            base_dir=ROOT, train_sample_count=5, metadata={}, console=False,
                            run_name="options", resume=True)

    def test_zero_rate_probe_has_no_division_by_zero_or_false_strict_gain(self):
        with TemporaryDirectory() as tmp:
            c = fixture_config(tmp, probes=True)
            _, _, _, state, probes = fixture_components(c)
            rows = probes.run(state, step=4, learning_rate=0., config=c.probes,
                              cumulative={})
            comparisons = [row for stream, row in rows if stream == "theorem2"]
            self.assertTrue(comparisons)
            for row in comparisons:
                self.assertEqual(row["difference"], 0.)
                self.assertIsNone(row["difference_over_lr"])
                self.assertIsNone(row["first_order_residual"])
                self.assertFalse(row["strict_gain_eligible"])
                self.assertTrue(row["finite"])


if __name__ == "__main__":
    unittest.main()
