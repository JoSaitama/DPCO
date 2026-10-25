"""SGD-prefix boundaries, exact recovery, observers and configuration forwarding."""

from dataclasses import replace
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

from src.config import DiagnosticConfig, MechanismConfig, MethodConfig, ProbeConfig
from src.data.diagnostic_batches import TrainingAuditBatch
from src.methods import create_method
from src.methods.sage import SAGE
from src.methods.sage_policy import PolicyState
from src.probes.tpu_mechanisms import TPUMechanisms
from src.probes.training_diagnostics import TrainingDiagnostics
from src.training import Trainer, TrainState
from src.training.experiment import ExperimentRunner
from src.training.launch import resolve_config
from src.training.results import make_run_name
from src.training.schedules import sage_warmup_steps
from src.training.updates import SGDUpdate
from tests.helpers import QuadraticTask, scalar_batch
from tests.test_observation import ROOT, fixture_config, fixture_components, read_rows
from tests.test_probe_memory import coupled_fixture
from tests.test_training_diagnostics import AnalyticBatch


PRESET = ROOT / "configs/tpu/imagenet100_resnet50_sage_e.toml"


class SAGEWarmupTests(unittest.TestCase):
    def assert_tree_close(self, a, b, *, exact=False):
        self.assertEqual(jax.tree_util.tree_structure(a), jax.tree_util.tree_structure(b))
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            if exact:
                np.testing.assert_array_equal(x, y)
            else:
                np.testing.assert_allclose(x, y, rtol=3e-5, atol=3e-6)

    def test_config_rounding_disabled_path_and_invalid_values(self):
        for total, percent, expected in ((5000, 10, 500), (7, 10, 1), (7, 25, 2),
                                         (7, 0, 0), (7, 100, 7)):
            c = resolve_config(PRESET, mode="train", steps=total, sage_warmup_percent=percent)
            self.assertEqual(sage_warmup_steps(c), expected)
        baseline = resolve_config(PRESET, mode="train")
        self.assertEqual(baseline.method.sage_warmup_percent, 0)
        self.assertIsInstance(create_method("sage", warmup_steps=0), SAGE)
        self.assertNotIn("warmup", make_run_name(baseline))
        for bad in (-1, 101, float("nan"), float("inf"), True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_config(PRESET, mode="train", sage_warmup_percent=bad)
        with self.assertRaisesRegex(ValueError, "legacy theorem probes off"):
            resolve_config(PRESET, mode="train", sage_warmup_percent=10, theorem_probes=True)
        with self.assertRaises(ValueError):
            replace(baseline, method=MethodConfig("sgd", sage_warmup_percent=10)).validate()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "warmup.toml"
            path.write_text(PRESET.read_text().replace('[method]\nname = "sage"',
                '[method]\nname = "sage"\nsage_warmup_percent = 10.0'))
            self.assertEqual(sage_warmup_steps(resolve_config(path, mode="train", steps=5000)), 500)

    def test_all_tpu_shell_suites_forward_parent_child_and_preserve_other_modes(self):
        env = {**os.environ, "SAGE_PYTHON": sys.executable}
        for suite in ("run_sage_suite.sh", "run_inaturalist_suite.sh", "run_sage_mechanism_suite.sh"):
            methods = ["sage_e", "sage_l", "sgd"]
            output = subprocess.check_output(["bash", str(ROOT/suite), "--dry-run", "--methods", *methods,
                "--seed", "5", "--steps", "5000", "--sage-warmup-percent", "10"],
                cwd=gettempdir(), env=env, text=True)
            for plan in json.loads(output):
                expected = 0 if plan["method"] == "sgd" else 500
                self.assertEqual(plan["sage_warmup_steps"], expected)
                self.assertEqual("warmup10pct" in plan["stdout"], expected > 0)
                command = plan["command"]
                value = lambda flag: command[command.index(flag)+1]
                child = resolve_config(value("--config"), mode=value("--mode"), method=value("--method"),
                    steps=int(value("--steps")), sage_warmup_percent=float(value("--sage-warmup-percent")))
                self.assertEqual(sage_warmup_steps(child), expected)
        for method in ("sage_l_half", "sage_e_alternating"):
            unchanged = resolve_config(PRESET, mode="train", method=method)
            configured = resolve_config(PRESET, mode="train", method=method, sage_warmup_percent=10)
            self.assertEqual(unchanged, configured)

    def test_compiled_transition_matches_manual_methods_and_global_bn(self):
        for reference in ("loss", "error"):
            task, initial, batch = coupled_fixture("mixed")
            method = create_method("sage", reference_name=reference, gradient_backend="jvp", warmup_steps=2)
            trainer = Trainer(task, method, SGDUpdate(.01))
            state = initial._replace(method_state=method.initial_state())
            expected = initial
            step = trainer.compile_step()
            for t in range(4):
                manual = create_method("sgd" if t < 2 else "sage", reference_name=reference,
                                       gradient_backend="jvp")
                expected, _ = Trainer(task, manual, trainer.updater).compile_step()(expected, batch)
                state, metrics = step(state, batch)
                self.assert_tree_close(state.params, expected.params)
                self.assert_tree_close(state.model_state, expected.model_state)
                self.assertEqual(float(metrics["policy_alpha"]), float(t >= 2))
                self.assertEqual(int(state.method_state.updates), t+1)

    def test_prefix_does_not_execute_alignment_and_state_is_committed_once(self):
        from src.methods.sage_warmup import SAGEWarmup
        executed = []
        class MarkedSAGE(SAGE):
            def compute_direction(self, *args, **kwargs):
                result = super().compute_direction(*args, **kwargs)
                jax.debug.callback(lambda: executed.append("sage"), ordered=True)
                return result
        method = SAGEWarmup(create_method("sage").reference, 2, "jvp")
        state = TrainState({"x": jnp.zeros(1)}, {"calls": jnp.asarray(0)}, method.initial_state())
        with patch("src.methods.sage_warmup.SAGE", MarkedSAGE):
            compiled = Trainer(QuadraticTask(), method, SGDUpdate(.01)).compile_step()
            for t in range(4):
                state, _ = compiled(state, scalar_batch())
                jax.block_until_ready(state)
                self.assertEqual(len(executed), max(0, t-1))
                self.assertEqual(int(state.model_state["calls"]), t+1)

    def test_real_update_diagnostics_use_executed_phase_not_full_sage(self):
        task, batch = QuadraticTask(), AnalyticBatch(*scalar_batch())
        for reference in ("loss", "error"):
            method = create_method("sage", reference_name=reference, gradient_backend="jvp", warmup_steps=1)
            state = TrainState({"x": jnp.zeros(1)}, {"calls": jnp.asarray(0)}, method.initial_state())
            trainer = Trainer(task, method, SGDUpdate(.01))
            observer = TrainingDiagnostics(task, method, DiagnosticConfig(enabled=True, transfer=False,
                lr_scales=(1.,)), reference_name=reference, trajectory_method="sage")
            for t in range(2):
                before = state
                state, _ = trainer.compile_step()(state, batch)
                rows = observer.run(before, state, batch, learning_rate=.01, rng_key=jax.random.key(t),
                    context={}, cumulative_eta_before=t*.01)
                update = next(r for s,r in rows if s == "diagnostic_update")
                self.assertLess(update["parameter_l2_error"], 1e-6)
                policy = next(r for s,r in rows if s == "diagnostic_policy")
                self.assertEqual(policy["policy_alpha"], float(t >= 1))
                if t == 0:
                    self.assertAlmostEqual(policy["difference"], 0., delta=1e-6)

    def test_mechanisms_label_actual_sgd_then_actual_sage_endpoints(self):
        with TemporaryDirectory() as tmp:
            c = fixture_config(tmp)
            trainer, source, _, state, _ = fixture_components(c)
            method = create_method("sage", reference_name="error", gradient_backend="jvp", warmup_steps=1)
            trainer = trainer.with_method(method)
            state = state._replace(method_state=method.initial_state())
            cfg = MechanismConfig(enabled=True, steps=(0, 1), moment_steps=(), bank_moments=False,
                absolute_lrs=(.01,), lr_scales=(), bank_batches=1)
            observer = TPUMechanisms(trainer.task, method, cfg, reference_name="error", trajectory_method="sage",
                bank_batches=(TrainingAuditBatch("fixed", source.batches[0]),), fixed_buffers=state.model_state)
            for t in range(2):
                batch, before = next(source), state
                state, _ = trainer.compile_step()(state, batch)
                rows = observer.run(before, state, batch, learning_rate=.01, rng_key=jax.random.key(t),
                    context={}, cumulative_eta_before=t*.01)
                check = next(r for s,r in rows if s == "mechanism_update")
                self.assertTrue(check["reliable"])
                pairs = [r for s,r in rows if s == "mechanism_pairs" and r["candidate_kind"] == "actual_pair"]
                self.assertTrue(pairs)
                self.assertEqual({r["actual_endpoint"] for r in pairs}, {"sgd" if t == 0 else "sage"})
                if t == 1:
                    self.assertEqual({r["reference"] for r in pairs}, {"error"})

    def test_resume_before_at_and_after_switch_and_observation_isolation(self):
        for reference in ("loss", "error"):
            with TemporaryDirectory() as tmp:
                finals = []
                for enabled in (False, True):
                    c = fixture_config(tmp)
                    c = replace(c, method=replace(c.method, sage_warmup_percent=50),
                        reference=replace(c.reference, name=reference),
                        diagnostics=DiagnosticConfig(enabled=enabled, transfer=False, steps=(0,1,2,3), lr_scales=(1.,)))
                    trainer, source, evaluation, initial, _ = fixture_components(c)
                    method = create_method("sage", reference_name=reference, gradient_backend="jvp", warmup_steps=2)
                    trainer = trainer.with_method(method)
                    observer = TrainingDiagnostics(trainer.task, method, c.diagnostics, reference_name=reference,
                        trajectory_method="sage") if enabled else None
                    runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT, train_sample_count=5,
                        metadata={}, diagnostics=observer, console=False, run_name=f"warmup_{enabled}")
                    final = runner.run(initial)
                    finals.append(final)
                    policy = read_rows(runner.paths.files/"policy_steps.csv")
                    self.assertEqual([r["executed_method"] for r in policy], ["sgd", "sgd", "sage", "sage"])
                    for slot in ("p75", "p50", "p25"):
                        restored = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                            train_sample_count=5, metadata={}, diagnostics=observer, console=False,
                            run_name=runner.paths.name, resume=True).run(initial, resume_slot=slot)
                        self.assert_tree_close(final, restored, exact=True)
                        self.assertEqual(policy, read_rows(runner.paths.files/"policy_steps.csv"))
                    changed = replace(c, method=replace(c.method, sage_warmup_percent=25))
                    with self.assertRaisesRegex(ValueError, "configuration differs"):
                        ExperimentRunner(changed, trainer, source, evaluation, base_dir=ROOT,
                            train_sample_count=5, metadata={}, diagnostics=observer, console=False,
                            run_name=runner.paths.name, resume=True)
                self.assert_tree_close(finals[0], finals[1], exact=True)

    def test_real_tpu_entry_constructs_warmup_without_changing_model_or_data(self):
        from scripts.train_classification import run
        from src.data.imagenet_hf import DataPlan, prepare
        from src.models.resnet import ResNet
        from tests.test_tpu_pipeline import fake_stream
        for reference in ("error", "loss"):
            with TemporaryDirectory() as tmp:
                c = fixture_config(tmp)
                c = replace(c, method=replace(c.method, sage_warmup_percent=50),
                    reference=replace(c.reference, name=reference),
                    data=replace(c.data, image_size=8), global_batch_size=2,
                    execution=replace(c.execution, gradient_backend="jvp"))
                args = SimpleNamespace(mode="smoke", run_name="entry", resume=False, reserved=False, resume_slot="latest")
                images = prepare(DataPlan("fixture", "a"*40, {0: 0, 1: 1}, "fixture", 8, 3),
                    Path(tmp)/"data", stream_factory=fake_stream)
                # No TPU launch: call the existing entry with a tiny CPU fixture.
                with patch("src.data.classification.open_training_images", return_value=(images, {})), \
                     patch("src.models.resnet.create_resnet", return_value=ResNet((1,), 100, num_filters=2)):
                    run(args, c, jax.local_devices(), status=None)
                rows = read_rows(Path(tmp)/"files/entry/policy_steps.csv")
                self.assertEqual([r["executed_method"] for r in rows], ["sgd", "sgd", "sage", "sage"])


if __name__ == "__main__":
    unittest.main()
