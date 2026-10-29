"""CPU unit checks for continuous correct-reference DPCO; no TPU launches."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.config import MethodConfig, ProbeConfig, ReferenceConfig, load_config
from src.data.batch import classification_batch
from src.methods import create_method
from src.probes.classification import ClassificationProbes, ProbeBatch
from src.probes.training_diagnostics import independent_directions
from src.references import create_reference
from src.training import TrainState
from src.training.launch import resolve_config
from src.training.results import make_run_name
from tests.helpers import QuadraticTask, scalar_batch
from tests.test_observation import fixture_config, fixture_components, read_rows
from tests.test_training_diagnostics import analytic_observer, observe
from tests.test_tpu_mechanisms import components, rows_for


ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/tpu/imagenet100_resnet50_dpco_e.toml"


class DPCORTests(unittest.TestCase):
    def test_correct_mean_masks_padding_and_requires_feedback(self):
        reference = create_reference("correct")
        valid = jnp.array([True, True, True, False])
        feedback = {"error_mask": jnp.array([False, True, False, False])}
        gradients = {"x": jnp.array([[2.], [-3.], [-1.], [jnp.nan]])}
        np.testing.assert_array_equal(reference.selection_mask(valid, feedback), [True, False, True, False])
        result = reference.build(gradients, valid, feedback)
        self.assertEqual(float(result.selected_count), 2.)
        np.testing.assert_allclose(result.direction["x"], [.5])
        with self.assertRaisesRegex(ValueError, "error_mask"):
            reference.selection_mask(valid, {})
        with self.assertRaisesRegex(ValueError, "same shape"):
            reference.selection_mask(valid, {"error_mask": jnp.zeros(2, bool)})

    def test_negative_scores_flip_across_entire_batch_with_original_B(self):
        # Correct gradients are [2,-1], reference=.5. Both the wrong gradient
        # -3 and the correct gradient -1 must flip. Correctness is not the rule.
        task, params = QuadraticTask(), {"x": jnp.zeros(1)}
        batch = (jnp.array([[-2.], [3.], [1.], [999.]]),
                 jnp.array([True, True, True, False]), jnp.array([False, True, False, False]))
        for backend in ("audit", "jvp"):
            method = replace(create_method("dpco", reference_name="correct", gradient_backend=backend), record_theory=True)
            result = jax.jit(lambda p: method.compute_direction(task, p, batch))(params)
            np.testing.assert_allclose(result.direction["x"], [2.], atol=1e-6)
            self.assertEqual(int(result.metrics["theory_flip_count"]), 2)
            self.assertEqual(int(result.model_state["calls"]), 1)
            self.assertAlmostEqual(float(result.metrics["theory_delta"]), 4/3, places=6)
        independent = independent_directions(task, "correct", params, batch, None)
        np.testing.assert_allclose(independent.reference["x"], [.5])
        np.testing.assert_allclose(independent.weights, [1., -1., -1., 0.])
        np.testing.assert_allclose(independent.dpco["x"], [2.])

    def test_empty_correct_falls_back_to_sgd_and_all_correct_equals_loss(self):
        task, params = QuadraticTask(), {"x": jnp.zeros(1)}
        targets, mask, _ = scalar_batch()
        for backend in ("audit", "jvp"):
            for errors, baseline in ((jnp.ones(3, bool), "sgd"), (jnp.zeros(3, bool), "dpco")):
                batch = targets, mask, errors
                actual = create_method("dpco", reference_name="correct", gradient_backend=backend).compute_direction(task, params, batch)
                expected = create_method(baseline, reference_name="loss", gradient_backend=backend).compute_direction(task, params, batch)
                for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
                    np.testing.assert_allclose(a, b, atol=1e-6)

    def test_real_classification_correct_selection_and_full_jacobian_parity(self):
        from src.tasks.classification import ClassificationTask
        task = ClassificationTask(lambda p, x, state: (x @ p["w"], state), 2)
        params = {"w": jnp.array([[1., -1.]])}
        batch = classification_batch(np.array([[1.], [-1.], [2.], [-2.]], np.float32),
                                     np.array([0, 0, 1, 1], np.int32))
        result = task.evaluate(params, batch)
        mask = create_reference("correct").selection_mask(result.valid_mask, result.feedback)
        np.testing.assert_array_equal(mask, [True, False, False, True])
        directions = [create_method("dpco", reference_name="correct", gradient_backend=b)
                      .compute_direction(task, params, batch).direction for b in ("audit", "jvp")]
        np.testing.assert_allclose(directions[0]["w"], directions[1]["w"], atol=1e-6)

    def test_resnet_batchnorm_parity_and_one_state_commit(self):
        from src.models.resnet import ResNet, initialize, model_adapter
        from src.tasks.classification import ClassificationTask
        model = ResNet((1,), 3, num_filters=2)
        params, buffers = initialize(model, 3, 8)
        task = ClassificationTask(model_adapter(model, train=True), 3)
        images = np.random.default_rng(4).normal(size=(3, 8, 8, 3)).astype(np.float32)
        predictions = np.asarray(jnp.argmax(task.model_fn(params, images, buffers)[0], axis=-1)).copy()
        labels = predictions.copy(); labels[1] = (labels[1]+1) % 3
        batch = classification_batch(images, labels.astype(np.int32))
        results = []
        for backend in ("audit", "jvp"):
            method = create_method("dpco", reference_name="correct", gradient_backend=backend)
            results.append(jax.jit(lambda p, s: method.compute_direction(task, p, batch, s))(params, buffers))
        for a, b in zip(jax.tree_util.tree_leaves(results[0].direction), jax.tree_util.tree_leaves(results[1].direction)):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=2e-6)
        expected = task.evaluate(params, batch, buffers).model_state
        for a, b in zip(jax.tree_util.tree_leaves(results[1].model_state), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_config_alias_identity_and_continuous_only_scope(self):
        config = resolve_config(PRESET, mode="train", method="dpco_r")
        self.assertEqual(config.method.name, "dpco")
        self.assertEqual(config.reference.name, "correct")
        self.assertIn("_dpco_r_", make_run_name(config))
        self.assertEqual(config.method.dpco_warmup_percent, 0.)
        for method in (MethodConfig("dpco_loo"), MethodConfig("dtv"),
                       MethodConfig("dpco_policy", policy_mode="half"),
                       MethodConfig("dpco", dpco_warmup_percent=10)):
            with self.assertRaisesRegex(ValueError, "continuous dpco_r only"):
                replace(config, method=method).validate()
        old = load_config(PRESET)
        for label, reference in (("dpco_e", "error"), ("dpco_l", "loss"), ("sgd", "loss")):
            resolved = resolve_config(PRESET, mode="train", method=label)
            self.assertEqual(resolved.reference.name, reference)
            self.assertEqual(resolved.training, old.training)

    def test_all_tpu_suites_forward_dpco_r_without_launching_children(self):
        env = {**os.environ, "DPCO_PYTHON": sys.executable}
        for script in ("run_dpco_suite.sh", "run_inaturalist_suite.sh", "run_dpco_mechanism_suite.sh"):
            result = subprocess.check_output(["bash", str(ROOT/script), "--dry-run", "--methods", "dpco_r",
                "--seeds", "2", "5", "--steps", "5", "--batch-size", "8", "--no-checkpoints",
                "--mechanism-windows"],
                cwd=ROOT, env=env, text=True)
            plans = json.loads(result)
            self.assertEqual(len(plans), 2)
            for plan in plans:
                self.assertEqual(plan["reference"], "correct")
                self.assertEqual(plan["method"], "dpco")
                self.assertIn("_dpco_r_", plan["stdout"])
                self.assertEqual(plan["backend"], "tpu_only")
                command = plan["command"]
                self.assertEqual(command[command.index("--method")+1], "dpco_r")
                self.assertFalse(plan["checkpoints_enabled"])

    def test_observers_use_frozen_correct_objective_and_actual_update(self):
        observer, trainer, before, batch = analytic_observer(reference="correct")
        after, _ = trainer.compile_step()(before, batch, jax.random.key(2))
        rows = observe(observer, before, after, batch)
        update = next(r for s, r in rows if s == "diagnostic_update")
        self.assertLess(update["parameter_l2_error"], 1e-6)
        self.assertLess(update["model_state_max_abs_error"], 1e-6)
        objectives = {r["objective"] for s, r in rows if s == "diagnostic_paired"}
        self.assertEqual(objectives, {"full_batch_ce", "frozen_correct_ce"})
        probes = ClassificationProbes(trainer.task, create_reference("correct"), [ProbeBatch("fixed", batch)],
            reference_name="correct", objective_mode="train")
        rows = probes.run(before, step=0, learning_rate=.01,
                          config=ProbeConfig(enabled=True, theorem3=False, batches_per_node=1), cumulative={})
        t1 = next(r for s, r in rows if s == "theorem1")
        self.assertAlmostEqual(float(t1["identity_residual"]), 0., places=6)
        self.assertEqual({r["objective"] for s, r in rows if s == "theorem2"},
                         {"full_batch_ce", "frozen_correct_ce"})

    def test_tpu_mechanism_correct_reference_pairs_and_bank_moments(self):
        with TemporaryDirectory() as tmp:
            _, trainer, source, _, before, observer = components(tmp, reference="correct")
            _, rows = rows_for(observer, trainer, before, next(source))
            self.assertEqual(observer.trajectory_label, "dpco_r")
            t1 = [r for s, r in rows if s == "mechanism_t1" and r["reference"] == "correct"]
            self.assertEqual(len(t1), 1)
            self.assertAlmostEqual(t1[0]["identity_residual"], 0., places=6)
            pairs = [r for s, r in rows if s == "mechanism_pairs" and r["reference"] == "correct"]
            self.assertIn("source_frozen_correct_ce", {r["objective"] for r in pairs})
            self.assertTrue(any(r["candidate_kind"] == "actual_pair" and r["actual_endpoint"] == "dpco" for r in pairs))
            moments = [r for s, r in rows if s == "mechanism_moments" and r["reference"] == "correct"]
            self.assertEqual(len(moments), 1)
            self.assertLess(moments[0]["mean_gd_h_l2_error"], 2e-6)
        # Old trajectories do not pay for a newly added reference diagnostic.
        with TemporaryDirectory() as tmp:
            _, _, _, _, _, old = components(tmp, reference="error")
            self.assertEqual(old.reference_names, ("loss", "error"))

    def test_recording_identity_and_checkpoint_resume(self):
        from src.training.experiment import ExperimentRunner
        with TemporaryDirectory() as tmp:
            config = replace(fixture_config(tmp), method=MethodConfig("dpco"), reference=ReferenceConfig("correct"))
            trainer, source, evaluation, state, _ = fixture_components(config)
            trainer = trainer.with_method(create_method("dpco", reference_name="correct", gradient_backend="jvp"))
            runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, console=False)
            final = runner.run(state)
            self.assertIn("_dpco_r_", runner.paths.name)
            metadata = json.loads((runner.paths.logs/"metadata.json").read_text())
            self.assertEqual(metadata["config"]["reference"]["name"], "correct")
            self.assertEqual(len(read_rows(runner.paths.files/"train_steps.csv")), 4)
            _, source2, evaluation2, _, _ = fixture_components(config)
            resumed = ExperimentRunner(config, trainer, source2, evaluation2, base_dir=ROOT,
                train_sample_count=5, metadata={}, console=False, resume=True, run_name=runner.paths.name)
            restored = resumed.run(state, resume_slot="p50")
            for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(restored)):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(len(read_rows(runner.paths.files/"train_steps.csv")), 4)


if __name__ == "__main__":
    unittest.main()
