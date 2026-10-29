"""Optional L2 must preserve the unregularized computation when disabled."""

from dataclasses import replace
import json
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.config import load_config, ProbeConfig
from src.training import Trainer
from src.training.experiment import ExperimentRunner
from src.training.updates import SGDUpdate
from tests.test_observation import ROOT, fixture_components, fixture_config


class L2Tests(unittest.TestCase):
    def test_disabled_update_has_identical_jaxpr_and_compiled_results(self):
        params = {"w": jnp.arange(6, dtype=jnp.float32).reshape(2, 3), "b": jnp.ones(3)}
        direction = jax.tree_util.tree_map(lambda p: p + .25, params)

        def original(p, d):
            return jax.tree_util.tree_map(lambda x, g: x - .1 * g, p, d)

        for coefficient in (0., -0.):
            updater = SGDUpdate(.1, l2_coefficient=coefficient)
            updated = lambda p, d: updater.apply(p, d, 0).params
            self.assertEqual(str(jax.make_jaxpr(original)(params, direction)),
                             str(jax.make_jaxpr(updated)(params, direction)))
            old, new = jax.jit(original)(params, direction), jax.jit(updated)(params, direction)
            for a, b in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new)):
                np.testing.assert_array_equal(a, b)
            self.assertIsNone(updater.apply(params, direction, 0).optimizer_state)

    def test_enabled_l2_uses_kernel_mask_and_preserves_optimizer_state(self):
        params = {"conv": jnp.full((1, 1, 2, 3), 2.), "dense": jnp.full((2, 3), -2.),
                  "bias": jnp.ones(3), "bn": {"scale": jnp.ones(3), "offset": jnp.zeros(3)}}
        direction = jax.tree_util.tree_map(jnp.ones_like, params)
        updater = SGDUpdate(lambda step: .1 / (step + 1), l2_coefficient=.2)
        output = jax.jit(updater.apply)(params, direction, 1, {"sentinel": jnp.array(3)})
        expected = jax.tree_util.tree_map(
            lambda p, d: np.asarray(p) - .05 * (np.asarray(d) + (.2 * np.asarray(p) if p.ndim > 1 else 0)),
            params, direction)
        for a, b in zip(jax.tree_util.tree_leaves(output.params), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(a, b, atol=1e-7)
        self.assertEqual(int(output.optimizer_state["sentinel"]), 3)

    def test_enabled_l2_is_applied_after_sgd_or_dpco_direction_construction(self):
        for method, reference in (("sgd", "loss"), ("dpco", "loss"), ("dpco", "error")):
            with self.subTest(method=method, reference=reference), TemporaryDirectory() as tmp:
                c = fixture_config(tmp)
                c = replace(c, method=replace(c.method, name=method),
                            reference=replace(c.reference, name=reference))
                trainer, source, _, initial, _ = fixture_components(c)
                batch = next(source)
                base, metrics = jax.jit(trainer.step)(initial, batch)
                regularized_trainer = Trainer(trainer.task, trainer.method,
                    SGDUpdate(c.training.learning_rate, l2_coefficient=.1))
                regularized, regularized_metrics = jax.jit(regularized_trainer.step)(initial, batch)
                np.testing.assert_allclose(regularized.params["w"],
                    base.params["w"] - c.training.learning_rate * .1 * initial.params["w"], atol=1e-7)
                np.testing.assert_array_equal(regularized.params["b"], base.params["b"])
                for key in metrics:
                    np.testing.assert_array_equal(metrics[key], regularized_metrics[key])

    def test_invalid_coefficients_and_regularized_theorem_probes_are_rejected(self):
        c = load_config(ROOT / "configs/tpu/imagenet100_resnet50_dpco_e.toml")
        self.assertEqual(c.training.l2_coefficient, 0.)
        for value in (-.1, float("inf"), float("nan"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SGDUpdate(.1, l2_coefficient=value)
                with self.assertRaises(ValueError):
                    replace(c, training=replace(c.training, l2_coefficient=value)).validate()
        with self.assertRaisesRegex(ValueError, "theorem probes"):
            replace(c, training=replace(c.training, l2_coefficient=.0001),
                    probes=ProbeConfig(enabled=True)).validate()

    def test_regularized_training_resumes_exactly_and_rejects_coefficient_changes(self):
        for coefficient in (0., .0001):
            with self.subTest(coefficient=coefficient), TemporaryDirectory() as tmp:
                c = fixture_config(tmp)
                c = replace(c, training=replace(c.training, l2_coefficient=coefficient))
                trainer, source, evaluation, initial, _ = fixture_components(c)
                trainer.updater = SGDUpdate(c.training.learning_rate, l2_coefficient=coefficient)
                runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="l2")
                final = runner.run(initial)
                metadata_path = runner.paths.logs / "metadata.json"
                if coefficient == 0:
                    # Pre-feature runs did not store this field. Off stays compatible.
                    metadata = json.loads(metadata_path.read_text())
                    for key in ("l2_coefficient", "momentum", "nesterov", "lr_schedule", "min_learning_rate",
                                "optimizer", "adam_beta1", "adam_beta2", "adam_epsilon"):
                        del metadata["config"]["training"][key]
                    del metadata["config"]["method"]["negative_weight"]
                    del metadata["config"]["reference"]["beta"]
                    for key in ("noise_type", "is_human", "noise_seed", "noise_percent", "download"):
                        del metadata["config"]["data"][key]
                    metadata_path.write_text(json.dumps(metadata))
                original_metadata = metadata_path.read_bytes()
                _, source2, evaluation2, _, _ = fixture_components(c)
                resumed = ExperimentRunner(c, trainer, source2, evaluation2, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="l2", resume=True)
                final2 = resumed.run(initial, resume_slot="p50")
                for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(final2)):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(metadata_path.read_bytes(), original_metadata)
                changed = replace(c, training=replace(c.training, l2_coefficient=.01))
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    ExperimentRunner(changed, trainer, source2, evaluation2, base_dir=ROOT,
                        train_sample_count=5, metadata={}, console=False, run_name="l2", resume=True)


if __name__ == "__main__":
    unittest.main()
