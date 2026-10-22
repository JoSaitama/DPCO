"""Independent oracles, fault injection, isolation and real-update recording."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import NamedTuple
import unittest
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np

from src.config import DiagnosticConfig
from src.data.diagnostic_batches import TrainingAuditBatch, training_audit_batches
from src.data.images import ImageBatchSource
from src.data.imagenet_hf import DataPlan, prepare
from src.methods import create_method
from src.probes.training_diagnostics import TrainingDiagnostics, independent_directions
from src.training import TrainState, Trainer
from src.training.experiment import ExperimentRunner
from src.training.launch import diagnostic_overrides, resolve_config
from src.training.recording import SCHEMAS
from src.training.updates import SGDUpdate
from tests.helpers import QuadraticTask, scalar_batch
from tests.test_observation import ROOT, fixture_components, fixture_config, read_rows
from tests.test_probe_memory import coupled_fixture, dense_statistics
from src.references import create_reference
from tests.test_tpu_pipeline import fake_stream


class AnalyticBatch(NamedTuple):
    targets: object
    valid_mask: object
    errors: object


def analytic_observer(*, reference="loss", trajectory="sage", config=None, audits=()):
    config = config or DiagnosticConfig(enabled=True, transfer=False)
    task = QuadraticTask()
    method = create_method(trajectory, reference_name=reference, gradient_backend="jvp")
    observer = TrainingDiagnostics(task, method, config, reference_name=reference,
        trajectory_method=trajectory, audit_batches=audits)
    state = TrainState({"x": jnp.array([0.])}, {"calls": jnp.asarray(0)})
    batch = AnalyticBatch(*scalar_batch())
    return observer, Trainer(task, method, SGDUpdate(.01)), state, batch


def observe(observer, before, after, batch, *, eta=.01):
    rows = observer.run(before, after, batch, learning_rate=eta,
        rng_key=jax.random.key(2), context={"batch_id": "actual_train", "sample_indices": [0, 1, 2]},
        cumulative_eta_before=0.)
    # Every emitted field must have an explicitly registered common outlet.
    for stream, row in rows:
        assert set(row) <= set(SCHEMAS[stream]), (stream, set(row) - set(SCHEMAS[stream]))
    return rows


class TrainingDiagnosticTests(unittest.TestCase):
    def test_scalar_loss_route_matches_dense_oracle_with_global_batch_coupling(self):
        for errors in ("none", "all", "mixed"):
            task, state, batch = coupled_fixture(errors)
            for reference in ("loss", "error"):
                with self.subTest(errors=errors, reference=reference):
                    actual = jax.jit(lambda p, b, s: independent_directions(task, reference, p, b, s))(
                        state.params, batch, state.model_state)
                    oracle = dense_statistics(task, create_reference(reference), state.params,
                        batch, state.model_state, include_energy=False)
                    runtime = TrainingDiagnostics(task, create_method("sage", reference_name=reference, gradient_backend="jvp"),
                        DiagnosticConfig(enabled=True, transfer=False), reference_name=reference,
                        trajectory_method="sage")._independent(state.params, batch, state.model_state)
                    for name in ("reference", "scores", "sgd", "sage"):
                        for result in (actual, runtime):
                            for a, b in zip(jax.tree_util.tree_leaves(getattr(result, name)),
                                            jax.tree_util.tree_leaves(getattr(oracle, name))):
                                np.testing.assert_allclose(a, b, rtol=4e-5, atol=3e-6)

    def test_actual_update_and_single_state_commit_are_verified_for_both_methods(self):
        for trajectory in ("sage", "sgd"):
            observer, trainer, before, batch = analytic_observer(trajectory=trajectory)
            frozen = jax.tree_util.tree_map(np.array, before)
            after, _ = trainer.compile_step()(before, batch, jax.random.key(2))
            rows = observe(observer, before, after, batch)
            update = next(r for s, r in rows if s == "diagnostic_update")
            self.assertEqual((update["step"], update["update_step"]), (0, 1))
            for key in ("direction_l2_error", "parameter_l2_error", "replayed_parameter_l2_error",
                        "model_state_max_abs_error", "replayed_model_state_max_abs_error"):
                self.assertLess(update[key], 1e-6)
            for a, b in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(frozen)):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(int(after.model_state["calls"]), 1)

    def test_wrong_lr_and_double_state_update_are_detected_not_hidden_by_reconstruction(self):
        observer, trainer, before, batch = analytic_observer()
        # Fault injection deliberately updates with twice the declared learning rate.
        wrong = Trainer(trainer.task, trainer.method, SGDUpdate(.02))
        after, _ = wrong.compile_step()(before, batch, jax.random.key(2))
        after = after._replace(model_state={"calls": jnp.asarray(2)})
        rows = observe(observer, before, after, batch)
        update = next(r for s, r in rows if s == "diagnostic_update")
        self.assertGreater(update["parameter_l2_error"], .001)
        self.assertEqual(update["model_state_max_abs_error"], 1.)
        pair = next(r for s, r in rows if s == "diagnostic_paired" and r["lr_scale"] == 1.)
        actual_loss = jnp.mean(trainer.task.evaluate(after.params, batch, before.model_state).losses)
        self.assertAlmostEqual(pair["j_sage"], float(actual_loss), places=6)
        self.assertTrue(pair["actual_update_in_pair"])

    def test_quadratic_pair_has_exact_known_finite_step_remainder_and_transfer_sign(self):
        audit = TrainingAuditBatch("other_training_images",
            AnalyticBatch(jnp.ones((3, 1)), jnp.ones(3, bool), jnp.zeros(3, bool)), (2, 8, 9))
        config = DiagnosticConfig(enabled=True, audit_batches=1, lr_scales=(1., .5))
        observer, trainer, before, batch = analytic_observer(config=config, audits=(audit,))
        after, _ = trainer.compile_step()(before, batch, jax.random.key(2))
        with patch("jax.jacrev", side_effect=AssertionError("Dense runtime Jacobian forbidden")):
            rows = observe(observer, before, after, batch)
        for stream, row in rows:
            if stream == "diagnostic_paired":
                eta = row["probe_lr"]
                expected = -eta * row["objective_gain"] + .5 * eta**2 * (row["norm_sage_sq"] - row["norm_sgd_sq"])
                self.assertAlmostEqual(row["difference"], expected, places=6)
                self.assertLess(row["difference"], 0)
            elif stream == "diagnostic_transfer":
                self.assertEqual(row["overlap_sample_count"], 1)
                self.assertAlmostEqual(row["objective_gain"], -1., places=5)
                self.assertGreater(row["difference"], 0)
                self.assertAlmostEqual(row["transfer_difference"], row["objective_gain"] - row["reference_delta"], places=6)

    def test_error_objective_uses_frozen_mask_even_after_predictions_change(self):
        c = fixture_config("unused")
        trainer, source, _, before, _ = fixture_components(c)
        batch = source.batches[0]
        observer = TrainingDiagnostics(trainer.task, trainer.method,
            DiagnosticConfig(enabled=True, transfer=False, lr_scales=(1.,)),
            reference_name="error", trajectory_method="sage")
        trainer = Trainer(trainer.task, trainer.method, SGDUpdate(2.))
        after, _ = trainer.compile_step()(before, batch, jax.random.key(2))
        rows = observe(observer, before, after, batch, eta=2.)
        pair = next(r for s, r in rows if s == "diagnostic_paired" and r["objective"] == "frozen_error_ce")
        previous = trainer.task.evaluate(before.params, batch).feedback["error_mask"]
        following = trainer.task.evaluate(after.params, batch)
        self.assertFalse(np.array_equal(previous, following.feedback["error_mask"]))
        expected = jnp.sum(jnp.where(previous, following.losses, 0.)) / jnp.sum(previous)
        self.assertAlmostEqual(pair["j_sage"], float(expected), places=6)

    def test_empty_error_reference_and_zero_lr_are_explicit_and_finite(self):
        observer, trainer, before, batch = analytic_observer(reference="error")
        batch = batch._replace(errors=jnp.zeros_like(batch.errors))
        after, _ = trainer.compile_step()(before, batch)
        rows = observe(observer, before, after, batch)
        update = next(r for s, r in rows if s == "diagnostic_update")
        self.assertFalse(update["reference_valid"])
        self.assertEqual(update["reference_delta"], 0.)
        self.assertTrue(all(r["objective"] == "full_batch_ce" for s, r in rows if s == "diagnostic_paired"))
        after_zero, _ = Trainer(trainer.task, trainer.method, SGDUpdate(lambda t: 0.)).compile_step()(before, batch)
        rows = observe(observer, before, after_zero, batch, eta=0.)
        for stream, row in rows:
            if stream == "diagnostic_paired":
                self.assertEqual(row["difference"], 0.)
                self.assertIsNone(row["difference_over_lr"])
                self.assertIsNone(row["first_order_residual"])
                self.assertTrue(row["finite"])

    def test_checks_are_independently_disabled(self):
        for name in ("update_check", "paired_loss", "transfer"):
            config = DiagnosticConfig(enabled=True, update_check=name == "update_check",
                paired_loss=name == "paired_loss", transfer=name == "transfer", audit_batches=1, lr_scales=(1.,))
            audit = TrainingAuditBatch("audit", AnalyticBatch(*scalar_batch()))
            observer, trainer, before, batch = analytic_observer(config=config,
                audits=(audit,) if name == "transfer" else ())
            after, _ = trainer.compile_step()(before, batch)
            rows = observe(observer, before, after, batch)
            streams = {s for s, _ in rows}
            expected = {"update_check": "diagnostic_update", "paired_loss": "diagnostic_paired", "transfer": "diagnostic_transfer"}
            self.assertEqual(streams, {"diagnostic_context", expected[name]})

    def test_fixed_audit_bank_and_source_context_are_reproducible_and_do_not_advance_training(self):
        with TemporaryDirectory() as tmp:
            images = prepare(DataPlan("fixture", "a"*40, {0: 0, 1: 1}, "fixture", 12, 3),
                             Path(tmp) / "data", stream_factory=fake_stream)
            try:
                source = ImageBatchSource(images, 2, 8, 2)
                next(source)
                snapshot = source.snapshot()
                context = source.diagnostic_context()
                first = training_audit_batches(images, 2, 8, 2, 42)
                second = training_audit_batches(images, 2, 8, 2, 42)
                self.assertEqual(source.snapshot(), snapshot)
                self.assertEqual(context["sample_indices"], list(source.order[:2]))
                for a, b in zip(first, second):
                    self.assertEqual(a.identity, b.identity)
                    self.assertEqual(a.indices, b.indices)
                    np.testing.assert_array_equal(a.batch.images, b.batch.images)
                self.assertEqual(len(set(first[0].indices + first[1].indices)), 4)
                other = training_audit_batches(images, 2, 8, 2, 43)
                self.assertNotEqual(first[0].identity, other[0].identity)
            finally:
                images.close()

    def test_runner_diagnostics_leave_trajectory_identical_and_resume_without_duplicate_rows(self):
        for method in ("sage", "sgd"):
            with self.subTest(method=method), TemporaryDirectory() as tmp:
                base = fixture_config(tmp)
                base = replace(base, method=replace(base.method, name=method))
                finals = []
                for enabled in (False, True):
                    config = replace(base, diagnostics=DiagnosticConfig(enabled=enabled,
                        steps=(0, 1, 3, 100), audit_batches=1, lr_scales=(1., .5)))
                    trainer, source, evaluation, initial, _ = fixture_components(config)
                    observer = TrainingDiagnostics(trainer.task, trainer.method, config.diagnostics,
                        reference_name=config.reference.name, trajectory_method=method,
                        audit_batches=(TrainingAuditBatch("audit", source.batches[0]),)) if enabled else Mock()
                    runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                        train_sample_count=5, metadata={}, diagnostics=observer, run_name=f"{method}_{enabled}", console=False)
                    finals.append(runner.run(initial))
                    if not enabled:
                        observer.metadata.assert_not_called()
                        observer.run.assert_not_called()
                        self.assertFalse(list(runner.paths.files.glob("diagnostic_*.csv")))
                    else:
                        context_rows = read_rows(runner.paths.files / "diagnostic_context.csv")
                        self.assertEqual([(int(r["step"]), int(r["update_step"])) for r in context_rows], [(0, 1), (1, 2), (3, 4)])
                        trainer2, source2, eval2, _, _ = fixture_components(config)
                        restored = ExperimentRunner(config, trainer2, source2, eval2, base_dir=ROOT,
                            train_sample_count=5, metadata={}, diagnostics=observer, run_name=runner.paths.name,
                            resume=True, console=False).run(initial, resume_slot="p50")
                        for a, b in zip(jax.tree_util.tree_leaves(restored), jax.tree_util.tree_leaves(finals[-1])):
                            np.testing.assert_array_equal(a, b)
                        self.assertEqual(read_rows(runner.paths.files / "diagnostic_context.csv"), context_rows)
                for a, b in zip(jax.tree_util.tree_leaves(finals[0]), jax.tree_util.tree_leaves(finals[1])):
                    np.testing.assert_array_equal(a, b)

    def test_validation_and_suite_parent_child_flags(self):
        from src.training.launch import add_overrides
        import argparse
        parser = argparse.ArgumentParser(allow_abbrev=False)
        add_overrides(parser)
        args = parser.parse_args(["--diagnostics", "--diagnostic-steps", "0", "25", "100",
            "--no-diagnostic-transfer", "--diagnostic-audit-batches", "1", "--no-theorem-probes"])
        c = resolve_config(ROOT / "configs/tpu/imagenet100_resnet50_sage_e.toml", mode="train",
                           steps=200, **diagnostic_overrides(args))
        self.assertTrue(c.diagnostics.enabled)
        self.assertFalse(c.diagnostics.transfer)
        self.assertEqual(list(c.diagnostics.steps), [0, 25, 100])
        for options in ({"steps": [True]}, {"steps": [0, 0]}, {"steps": [-1]},
                        {"lr_scales": [float("nan")]}, {"audit_seed": -1}, {"audit_batches": 0},
                        {"update_check": False, "paired_loss": False, "transfer": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                replace(c, diagnostics=replace(c.diagnostics, **options)).validate()
        with self.assertRaisesRegex(ValueError, "momentum=0"):
            replace(c, training=replace(c.training, momentum=.9)).validate()

    def test_shell_grid_forwards_all_switches_to_each_child(self):
        import argparse
        from src.training.launch import add_overrides
        output = subprocess.check_output([str(ROOT / "run_sage_suite.sh"), "--dry-run",
            "--mode", "train", "--methods", "sage_l", "sage_e", "sgd", "--seeds", "2", "5",
            "--steps", "101", "--diagnostics", "--diagnostic-steps", "0", "25", "100", "150",
            "--no-diagnostic-transfer", "--diagnostic-update-check", "--diagnostic-paired-loss",
            "--diagnostic-lr-scales", "1", ".25", "--diagnostic-audit-seed", "17",
            "--diagnostic-audit-batches", "1", "--no-theorem-probes"],
            cwd="/private/tmp", env={**os.environ, "SAGE_PYTHON": sys.executable}, text=True)
        plans = json.loads(output)
        self.assertEqual(len(plans), 6)
        parser = argparse.ArgumentParser(allow_abbrev=False)
        add_overrides(parser)
        for plan in plans:
            flags, _ = parser.parse_known_args(plan["command"][4:])
            resolved = resolve_config(ROOT / "configs/tpu/imagenet100_resnet50_sage_e.toml",
                mode="train", steps=101, method=flags.method, **diagnostic_overrides(flags))
            self.assertEqual(json.loads(json.dumps(resolved.as_dict()["diagnostics"])), plan["diagnostics"])
            self.assertEqual(plan["active_diagnostic_state_steps"], [0, 25, 100])
            self.assertFalse(plan["probes_enabled"])
        disabled = resolve_config(ROOT / "configs/tpu/imagenet100_resnet50_sage_e.toml", mode="train",
            diagnostics=False, momentum=.9)
        self.assertFalse(disabled.diagnostics.enabled)
        # An explicit legacy override also permits diagnostics on an SGD trajectory.
        self.assertFalse(resolve_config(ROOT / "configs/tpu/imagenet100_resnet50_sage_e.toml",
            mode="validate", method="sgd", diagnostics=True, theorem_probes=False).probes.enabled)

    def test_legacy_metadata_without_diagnostic_config_resumes_only_on_disabled_path(self):
        with TemporaryDirectory() as tmp:
            config = fixture_config(tmp)
            trainer, source, evaluation, initial, _ = fixture_components(config)
            runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, run_name="legacy", console=False)
            final = runner.run(initial)
            path = runner.paths.logs / "metadata.json"
            metadata = json.loads(path.read_text())
            del metadata["config"]["diagnostics"]
            path.write_text(json.dumps(metadata))
            trainer2, source2, eval2, _, _ = fixture_components(config)
            restored = ExperimentRunner(config, trainer2, source2, eval2, base_dir=ROOT,
                train_sample_count=5, metadata={}, run_name="legacy", console=False, resume=True).run(initial, resume_slot="p50")
            for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(restored)):
                np.testing.assert_array_equal(a, b)
            self.assertNotIn("diagnostics", json.loads(path.read_text())["config"])

    def test_legacy_probes_and_real_update_diagnostics_coexist_without_switching_method(self):
        with TemporaryDirectory() as tmp:
            config = replace(fixture_config(tmp, probes=True),
                diagnostics=DiagnosticConfig(enabled=True, transfer=False, steps=(0, 1, 3), lr_scales=(1.,)))
            trainer, source, evaluation, initial, probes = fixture_components(config)
            observer = TrainingDiagnostics(trainer.task, trainer.method, config.diagnostics,
                reference_name="error", trajectory_method="sage")
            runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, probes=probes, diagnostics=observer,
                run_name="both", console=False)
            runner.run(initial)
            self.assertEqual(len(read_rows(runner.paths.files / "diagnostic_update.csv")), 3)
            self.assertTrue((runner.paths.files / "theorem2.csv").is_file())
            self.assertEqual(runner.trainer.method.__class__, trainer.method.__class__)

    def test_real_entry_uses_tiny_resnet_bn_and_records_all_three_checks(self):
        from scripts.train_classification import run
        from src.config import ResultsConfig
        from src.models.resnet import ResNet

        with TemporaryDirectory() as tmp:
            images = prepare(DataPlan("fixture", "a"*40, {0: 0, 1: 1}, "fixture", 8, 3),
                             Path(tmp) / "data", stream_factory=fake_stream)
            config = resolve_config(ROOT / "configs/tpu/imagenet100_resnet50_sage_e.toml", mode="smoke",
                steps=2, batch_size=2, diagnostics=True, diagnostic_steps=[0, 1],
                diagnostic_audit_batches=1, diagnostic_lr_scales=[1., .5])
            config = replace(config, data=replace(config.data, image_size=8),
                             results=ResultsConfig(str(Path(tmp) / "results")))
            args = SimpleNamespace(mode="smoke", run_name="real_diagnostics", resume=False,
                                   reserved=False, resume_slot="latest")
            with patch("src.data.classification.open_training_images", return_value=(images, {})), \
                 patch("src.models.resnet.create_resnet", return_value=ResNet((1,), 100, num_filters=2)), \
                 redirect_stdout(StringIO()):
                run(args, config, jax.local_devices(), status=None)
            files = Path(tmp) / "results/files/real_diagnostics"
            self.assertEqual(len(read_rows(files / "diagnostic_update.csv")), 2)
            self.assertEqual(len(read_rows(files / "diagnostic_transfer.csv")), 4)
            for row in read_rows(files / "diagnostic_update.csv"):
                self.assertLess(float(row["direction_relative_error"]), 1e-4)
                self.assertLess(float(row["model_state_max_abs_error"]), 1e-5)
            metadata = json.loads((Path(tmp) / "results/logs/real_diagnostics/metadata.json").read_text())
            self.assertEqual(metadata["training_diagnostics"]["sample_energy"], "not measured; no per-example energy loop")
            source = json.loads(read_rows(files / "diagnostic_context.csv")[0]["source_context_json"])
            self.assertEqual(len(source["sample_indices"]), 2)


if __name__ == "__main__":
    unittest.main()
