"""Policy algebra, conditional execution, real updates and legacy isolation."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from src.config import DiagnosticConfig, MethodConfig, ProbeConfig, ResultsConfig
from src.data.diagnostic_batches import TrainingAuditBatch
from src.methods import create_method
from src.methods.sage_policy import PolicyState, policy_alpha
from src.probes.training_diagnostics import TrainingDiagnostics
from src.training import Trainer, TrainState
from src.training.experiment import ExperimentRunner
from src.training.launch import POLICY_METHODS, resolve_config
from src.training.recording import SCHEMAS
from src.training.results import make_run_name
from src.training.updates import SGDUpdate
from tests.helpers import QuadraticTask, scalar_batch
from tests.test_observation import ROOT, fixture_config, fixture_components, read_rows
from tests.test_probe_memory import coupled_fixture, dense_statistics
from tests.test_training_diagnostics import AnalyticBatch, observe
from src.references import create_reference


class SAGEPolicyTests(unittest.TestCase):
    def assert_tree_close(self, left, right, *, exact=False):
        self.assertEqual(jax.tree_util.tree_structure(left), jax.tree_util.tree_structure(right))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
            if exact:
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-6)

    def method(self, mode, reference="loss", backend="jvp"):
        return create_method("sage_policy", reference_name=reference,
                             policy_mode=mode, gradient_backend=backend)

    def test_continuous_and_half_match_dense_oracle_with_global_bn_and_empty_errors(self):
        for errors in ("mixed", "all", "none"):
            task, state, batch = coupled_fixture(errors)
            for reference in ("loss", "error"):
                oracle = dense_statistics(task, create_reference(reference), state.params,
                                          batch, state.model_state, include_energy=False)
                for mode, expected in (("continuous", oracle.sage), ("half",
                        jax.tree_util.tree_map(lambda d, s: .5*(d+s), oracle.sgd, oracle.sage))):
                    for backend in ("audit", "jvp"):
                        method = self.method(mode, reference, backend)
                        result = jax.jit(lambda p, b, s: method.compute_direction(task, p, b, s,
                            method.initial_state()))(state.params, batch, state.model_state)
                        self.assert_tree_close(result.direction, expected)
                        baseline_name = "sage" if mode == "continuous" else "dtv"
                        original = create_method(baseline_name, reference_name=reference,
                            gradient_backend=backend)
                        baseline = jax.jit(lambda p, b, s: original.compute_direction(task, p, b, s))(
                            state.params, batch, state.model_state)
                        self.assert_tree_close(result.direction, baseline.direction, exact=True)
                        self.assert_tree_close(result.model_state, baseline.model_state, exact=True)
                        self.assertEqual(float(result.metrics["policy_alpha"]), 1. if mode == "continuous" else .5)

    def test_alternating_compiled_rounds_match_manual_sgd_sage_and_commit_state_once(self):
        method = self.method("alternating")
        trainer = Trainer(QuadraticTask(), method, SGDUpdate(.01))
        state = TrainState({"x": jnp.zeros(1)}, {"calls": jnp.asarray(0)}, method.initial_state())
        expected = state._replace(method_state=None)
        compiled = trainer.compile_step()
        for t in range(6):
            reference = Trainer(trainer.task, create_method("sage" if t % 2 else "sgd",
                gradient_backend="jvp"), trainer.updater)
            expected, _ = reference.compile_step()(expected, scalar_batch(), jax.random.key(t))
            state, metrics = compiled(state, scalar_batch(), jax.random.key(t))
            self.assert_tree_close(state.params, expected.params)
            self.assert_tree_close(state.model_state, expected.model_state)
            self.assertEqual(int(state.model_state["calls"]), t+1)
            self.assertEqual(int(state.method_state.updates), t+1)
            self.assertEqual(float(metrics["policy_alpha"]), float(t % 2))

    def test_sgd_round_does_not_execute_the_sage_branch(self):
        from src.methods.sage import SAGE
        executed = []
        class MarkedSAGE(SAGE):
            def compute_direction(self, *args, **kwargs):
                result = super().compute_direction(*args, **kwargs)
                jax.debug.callback(lambda: executed.append("sage"), ordered=True)
                return result
        method = self.method("alternating")
        state = TrainState({"x": jnp.zeros(1)}, {"calls": jnp.asarray(0)}, method.initial_state())
        with patch("src.methods.sage_policy.SAGE", MarkedSAGE):
            step = Trainer(QuadraticTask(), method, SGDUpdate(.01)).compile_step()
            for t in range(4):
                state, _ = step(state, scalar_batch(), jax.random.key(t))
                jax.block_until_ready(state)
                self.assertEqual(len(executed), (t+1)//2)

    def test_policy_diagnostics_separate_actual_direction_from_full_sage_counterfactual(self):
        task, batch = QuadraticTask(), AnalyticBatch(*scalar_batch())
        for mode in ("continuous", "alternating", "half"):
            method = self.method(mode)
            state = TrainState({"x": jnp.zeros(1)}, {"calls": jnp.asarray(0)}, method.initial_state())
            trainer = Trainer(task, method, SGDUpdate(.01))
            observer = TrainingDiagnostics(task, method,
                DiagnosticConfig(enabled=True, lr_scales=(1., .5), audit_batches=1),
                reference_name="loss", trajectory_method="sage_policy",
                audit_batches=(TrainingAuditBatch("audit", batch),))
            for t in range(2):
                before = state
                state, _ = trainer.compile_step()(state, batch, jax.random.key(t))
                rows = observe(observer, before, state, batch)
                update = next(r for s, r in rows if s == "diagnostic_update")
                self.assertLess(update["parameter_l2_error"], 1e-6)
                full = [r for s, r in rows if s == "diagnostic_paired"]
                actual = [r for s, r in rows if s == "diagnostic_policy"]
                alpha = float(policy_alpha(mode, before.method_state))
                self.assertEqual(len(actual), 4)
                for row in actual:
                    self.assertEqual(row["policy_alpha"], alpha)
                    if row["objective"] == "full_batch_ce":
                        other = next(r for r in full if r["lr_scale"] == row["lr_scale"])
                        self.assertAlmostEqual(row["objective_gain"], alpha * other["objective_gain"], places=6)
                    self.assertEqual(row["actual_update_in_pair"], row["lr_scale"] == 1.)
                self.assertEqual(full[0]["actual_update_in_pair"], alpha in (0., 1.))
                if mode == "half":
                    self.assertNotEqual(actual[0]["j_policy"], full[0]["j_sage"])
                self.assertEqual(int(state.method_state.updates), t+1)

    def test_runner_resume_and_diagnostic_switch_preserve_each_policy_trajectory(self):
        for mode in ("continuous", "alternating", "half"):
            with TemporaryDirectory() as tmp:
                finals = []
                for enabled in (False, True):
                    config = replace(fixture_config(tmp), method=MethodConfig("sage_policy", policy_mode=mode),
                        diagnostics=DiagnosticConfig(enabled=enabled, steps=(0, 1, 2, 3),
                            audit_batches=1, lr_scales=(1., .5)))
                    trainer, source, evaluation, initial, _ = fixture_components(replace(config, method=MethodConfig("sage")))
                    method = self.method(mode, reference="error", backend="audit")
                    trainer = trainer.with_method(method)
                    observer = TrainingDiagnostics(trainer.task, method, config.diagnostics,
                        reference_name="error", trajectory_method="sage_policy",
                        audit_batches=(TrainingAuditBatch("audit", source.batches[0]),)) if enabled else None
                    runner = ExperimentRunner(
                        config, trainer, source, evaluation, base_dir=ROOT, train_sample_count=5,
                        metadata={}, diagnostics=observer, run_name=f"{mode}_{enabled}", console=False)
                    final = runner.run(initial)
                    finals.append(final)
                    policy = read_rows(runner.paths.files / "policy_steps.csv")
                    self.assertEqual(len(policy), 4)
                    expected_alphas = [t % 2 if mode == "alternating" else (1 if mode == "continuous" else .5) for t in range(4)]
                    self.assertEqual([float(r["policy_alpha"]) for r in policy], expected_alphas)
                    self.assertEqual(int(final.method_state.updates), 4)
                    self.assertEqual(len(read_rows(runner.paths.parent.logs / "run_status.tsv")), 2 if not enabled else 6)
                    # Restore an odd completed step, where alternation must next use SAGE.
                    source.restore({"index": 0})
                    restored = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                        train_sample_count=5, metadata={}, diagnostics=observer, run_name=runner.paths.name,
                        resume=True, console=False).run(initial, resume_slot="p25")
                    self.assert_tree_close(restored, final, exact=True)
                    self.assertEqual(read_rows(runner.paths.files / "policy_steps.csv"), policy)
                self.assert_tree_close(finals[0], finals[1], exact=True)

    def test_suite_aliases_round_trip_and_isolate_run_names(self):
        env = {**os.environ, "SAGE_PYTHON": sys.executable}
        output = subprocess.check_output([str(ROOT / "run_sage_suite.sh"), "--dry-run", "--methods",
            *POLICY_METHODS, "sgd", "sage_l", "--seeds", "2", "5", "--diagnostics"],
            cwd=gettempdir(), env=env, text=True)
        plans = json.loads(output)
        self.assertEqual(len(plans), 16)
        for plan in plans:
            cmd = plan["command"]
            get = lambda flag: cmd[cmd.index(flag)+1]
            config = resolve_config(get("--config"), mode=get("--mode"), seed=int(get("--seed")),
                method=get("--method"), diagnostics=True)
            self.assertEqual(config.method.name, plan["method"])
            if config.method.name == "sage_policy":
                self.assertEqual(config.method.policy_mode, plan["policy_mode"])
                self.assertIn(f"sage_{'e' if config.reference.name == 'error' else 'l'}_{config.method.policy_mode}", plan["stdout"])
        config = resolve_config(ROOT / "configs/tpu/imagenet100_resnet50_sgd.toml", mode="train")
        with self.assertRaises(ValueError):
            replace(config, method=MethodConfig("sage_policy", policy_mode="invalid")).validate()
        with self.assertRaises(ValueError):
            replace(config, method=MethodConfig("sage", policy_mode="half")).validate()
        with self.assertRaises(ValueError):
            replace(config, method=MethodConfig("sage_policy"), probes=ProbeConfig(enabled=True)).validate()
        output = subprocess.check_output([str(ROOT / "run_inaturalist_suite.sh"), "--dry-run",
            "--methods", "sage_l_alternating", "sage_l_half", "--seed", "5"],
            cwd=gettempdir(), env=env, text=True)
        self.assertEqual([p["policy_mode"] for p in json.loads(output)], ["alternating", "half"])

    def test_policy_diagnostics_handle_zero_lr_and_reject_wrong_phase(self):
        method = self.method("half")
        before = TrainState({"x": jnp.zeros(1)}, {"calls": jnp.asarray(0)}, method.initial_state())
        trainer = Trainer(QuadraticTask(), method, SGDUpdate(lambda t: 0.))
        batch = AnalyticBatch(*scalar_batch())
        after, _ = trainer.compile_step()(before, batch, jax.random.key(0))
        observer = TrainingDiagnostics(trainer.task, method,
            DiagnosticConfig(enabled=True, transfer=False, lr_scales=(1.,)),
            reference_name="loss", trajectory_method="sage_policy")
        rows = observe(observer, before, after, batch, eta=0.)
        row = next(r for s, r in rows if s == "diagnostic_policy")
        self.assertEqual(row["difference"], 0.)
        self.assertIsNone(row["first_order_residual"])
        with self.assertRaisesRegex(ValueError, "phase"):
            observe(observer, before, after._replace(method_state=PolicyState(jnp.asarray(9))), batch)

    def test_legacy_resume_accepts_missing_policy_field_without_rewriting_metadata(self):
        for name in ("sgd", "sage"):
            with TemporaryDirectory() as tmp:
                config = replace(fixture_config(tmp), method=MethodConfig(name))
                trainer, source, evaluation, initial, _ = fixture_components(config)
                runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, run_name="legacy", console=False)
                final = runner.run(initial)
                path = runner.paths.logs / "metadata.json"
                metadata = json.loads(path.read_text())
                del metadata["config"]["method"]["policy_mode"]
                path.write_text(json.dumps(metadata))
                restored = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, run_name="legacy", resume=True,
                    console=False).run(initial, resume_slot="p50")
                self.assert_tree_close(restored, final, exact=True)
                self.assertNotIn("policy_mode", json.loads(path.read_text())["config"]["method"])
                self.assertFalse((runner.paths.files / "policy_steps.csv").exists())

    def test_real_training_entry_initializes_policy_state_for_all_modes(self):
        from scripts.train_classification import run
        from src.data.imagenet_hf import DataPlan, prepare
        from src.models.resnet import ResNet
        from tests.test_tpu_pipeline import fake_stream
        for mode, first in (("continuous", "sgd"), ("alternating", "sgd"),
                            ("alternating", "sage"), ("half", "sgd")):
            with TemporaryDirectory() as tmp:
                images = prepare(DataPlan("fixture", "a"*40, {0: 0, 1: 1}, "fixture", 8, 3),
                    Path(tmp)/"data", stream_factory=fake_stream)
                config = resolve_config(ROOT / "configs/tpu/imagenet100_resnet50_sgd.toml", mode="smoke",
                    steps=2, batch_size=2, method=f"sage_l_{mode}", alternating_first=first, diagnostics=True,
                    diagnostic_steps=[0, 1], diagnostic_transfer=False, diagnostic_lr_scales=[1.])
                config = replace(config, data=replace(config.data, image_size=8),
                    results=ResultsConfig(str(Path(tmp)/"results")))
                args = SimpleNamespace(mode="smoke", run_name=mode, resume=False, reserved=False, resume_slot="latest")
                with patch("src.data.classification.open_training_images", return_value=(images, {})), \
                     patch("src.models.resnet.create_resnet", return_value=ResNet((1,), 100, num_filters=2)), \
                     redirect_stdout(StringIO()):
                    run(args, config, jax.local_devices(), status=None)
                files = Path(tmp)/"results/files"/mode
                self.assertEqual(len(read_rows(files/"policy_steps.csv")), 2)
                if mode == "alternating":
                    self.assertEqual(read_rows(files/"policy_steps.csv")[0]["executed_method"], first)
                self.assertEqual(len(read_rows(files/"diagnostic_policy.csv")), 2)
                for row in read_rows(files/"diagnostic_update.csv"):
                    self.assertLess(float(row["direction_relative_error"]), 1e-4)


if __name__ == "__main__":
    unittest.main()
