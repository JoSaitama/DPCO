"""Predetermined inverse-square-root rates, forwarding and checkpoint indexing.

Only tiny CPU unit fixtures and configuration-only subprocesses run here.
No production training child, data preparation or accelerator launch is needed.
"""

import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.config import TrainingConfig, load_config
from src.training.experiment import ExperimentRunner
from src.training.launch import resolve_config
from src.training.learning_rates import InverseSqrtLearningRate, learning_rate_for
from src.training.results import make_run_name
from src.training.updates import make_updater
from tests.test_observation import ROOT, fixture_components, fixture_config


PRESET = ROOT / "configs/tpu/imagenet100_resnet50_sgd.toml"


class InverseSqrtTests(unittest.TestCase):
    def test_new_decay_run_names_include_schedule_without_changing_constant_names(self):
        config = load_config(PRESET)
        config = replace(config, seed=5, training=replace(config.training, learning_rate=.05))
        timestamp = datetime(2026, 10, 25, 12, 30, 45, tzinfo=timezone.utc)
        prefix = "imagenet100_clean_resnet50_sgd_5_0.05"
        for schedule in ("constant", "cosine", "inverse_sqrt"):
            with self.subTest(schedule=schedule):
                selected = replace(config, training=replace(config.training, lr_schedule=schedule))
                suffix = "" if schedule == "constant" else "_" + schedule
                self.assertEqual(make_run_name(selected, timestamp=timestamp),
                                 prefix + suffix + "_20261025_123045")

    def test_host_compiled_and_large_absolute_steps_match_formula(self):
        schedule = InverseSqrtLearningRate(.01)
        steps = np.array([0, 1, 3, 99, 4999, 1_000_000], dtype=np.int32)
        expected = .01 / np.sqrt(steps.astype(np.float64) + 1.)
        host = np.array([schedule(int(t)) for t in steps])
        device = np.asarray(jax.jit(jax.vmap(schedule))(jnp.asarray(steps)))
        np.testing.assert_allclose(host, expected, rtol=1e-15)
        np.testing.assert_allclose(device, expected, rtol=2e-7)
        self.assertEqual(schedule(0), .01)
        self.assertTrue(np.all(np.diff(host) < 0))
        # The restored absolute step, not steps since restart or total budget,
        # determines the rate. There is no special final-step zero or floor.
        for total in (5, 5000):
            training = TrainingConfig(learning_rate=.01, lr_schedule="inverse_sqrt", total_steps=total)
            self.assertEqual(learning_rate_for(training)(4999), schedule(4999))
        for initial in (0., -.01, float("inf"), float("nan")):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                InverseSqrtLearningRate(initial)

    def test_toml_cli_and_nonzero_floor_validation(self):
        original = load_config(PRESET)
        self.assertEqual(original.training.lr_schedule, "constant")
        self.assertIsInstance(learning_rate_for(original.training), float)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.toml"
            path.write_text(PRESET.read_text().replace("[training]",
                '[training]\nlr_schedule = "inverse_sqrt"'))
            configured = load_config(path)
            self.assertIsInstance(learning_rate_for(configured.training), InverseSqrtLearningRate)
            self.assertEqual(resolve_config(path, mode="train", lr_schedule="constant").training.lr_schedule,
                             "constant")
        for schedule in ("constant", "inverse_sqrt"):
            with self.subTest(schedule=schedule), self.assertRaisesRegex(ValueError, "requires lr_schedule=cosine"):
                resolve_config(PRESET, mode="train", lr_schedule=schedule, min_learning_rate=.001)

    def test_compiled_update_uses_absolute_step_rate_once(self):
        updater = make_updater(TrainingConfig(learning_rate=.01, lr_schedule="inverse_sqrt"))
        params = {"w": jnp.zeros(3)}
        direction = {"w": jnp.array([1., -2., 4.])}
        apply = jax.jit(updater.apply)
        for step in (0, 3, 4999):
            result = apply(params, direction, jnp.asarray(step))
            np.testing.assert_allclose(result.params["w"],
                -.01 / math.sqrt(step + 1) * np.asarray(direction["w"]), rtol=2e-7)
            self.assertIsNone(result.optimizer_state)

    def test_tiny_fixture_resume_and_recorded_rates_for_all_original_methods(self):
        for method, reference in (("sgd", "loss"), ("dpco", "error"), ("dpco", "loss")):
            with self.subTest(method=method, reference=reference), TemporaryDirectory() as tmp:
                config = fixture_config(tmp)
                config = replace(config, method=replace(config.method, name=method),
                    reference=replace(config.reference, name=reference),
                    training=replace(config.training, lr_schedule="inverse_sqrt"))
                trainer, source, evaluation, initial, _ = fixture_components(config)
                runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="inverse")
                final = runner.run(initial)
                _, source2, evaluation2, _, _ = fixture_components(config)
                resumed = ExperimentRunner(config, trainer, source2, evaluation2, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="inverse", resume=True)
                restored = resumed.run(initial, resume_slot="p50")
                for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(restored)):
                    np.testing.assert_array_equal(a, b)
                with (runner.paths.files / "train_steps.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([int(row["step"]) for row in rows], [1, 2, 3, 4])
                np.testing.assert_allclose([float(row["learning_rate"]) for row in rows],
                    [config.training.learning_rate / math.sqrt(t + 1) for t in range(4)], rtol=1e-15)
                for schedule in ("constant", "cosine"):
                    changed = replace(config, training=replace(config.training, lr_schedule=schedule))
                    with self.assertRaisesRegex(ValueError, "configuration differs"):
                        ExperimentRunner(changed, trainer, source2, evaluation2, base_dir=ROOT,
                            train_sample_count=5, metadata={}, console=False, run_name="inverse", resume=True)

    def test_all_shell_suites_forward_schedule_in_configuration_only_dry_run(self):
        env = {**os.environ, "DPCO_PYTHON": sys.executable}
        for entry in ("run_dpco_suite.sh", "run_inaturalist_suite.sh", "run_dpco_mechanism_suite.sh"):
            with self.subTest(entry=entry):
                output = subprocess.check_output(["bash", str(ROOT / entry), "--dry-run",
                    "--methods", "sgd", "dpco_e", "dpco_l", "--seeds", "2", "--steps", "5000",
                    "--lr", ".01", "--lr-schedule", "inverse_sqrt", "--min-lr", "0"],
                    cwd=gettempdir(), env=env, text=True)
                plans = json.loads(output)
                self.assertEqual(len(plans), 3)
                for plan in plans:
                    self.assertEqual(plan["lr_schedule"], "inverse_sqrt")
                    self.assertIn("_inverse_sqrt_", plan["stdout"])
                    command = plan["command"]
                    value = lambda flag: command[command.index(flag) + 1]
                    self.assertEqual(value("--lr-schedule"), "inverse_sqrt")
                    child = resolve_config(value("--config"), mode=value("--mode"),
                        lr_schedule=value("--lr-schedule"), learning_rate=float(value("--lr")),
                        min_learning_rate=float(value("--min-lr")), steps=int(value("--steps")))
                    self.assertEqual(learning_rate_for(child.training)(99), .001)

    def test_cpu_theory_cli_accepts_schedule_without_running_experiments(self):
        output = subprocess.check_output([sys.executable, str(ROOT / "scripts/validate_dpco_theory.py"),
            "--dry-run", "--schedules", "inverse_sqrt", "--methods", "sgd", "--lrs", ".01"], text=True)
        self.assertEqual(json.loads(output)["grid"][0]["schedule"], "inverse_sqrt")

    def test_cpu_theory_unit_fixture_uses_same_schedule_beyond_cli_validation(self):
        from src.probes.theory_validation import SCHEMAS, ExperimentSpec, LogisticSystem, run_experiment, synthetic_data
        from src.training.recording import RunRecorder
        from src.training.results import ResultsPaths
        with TemporaryDirectory() as tmp:
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("three_step_unit")
            paths.create()
            recorder = RunRecorder(paths, schemas=SCHEMAS, console=False)
            try:
                system = LogisticSystem(*synthetic_data(4, 1))
                _, summary = run_experiment(system, ExperimentSpec(method="sgd", schedule="inverse_sqrt",
                    lr=.01, steps=3, batch_size=2, probes=0, probe_steps=()), recorder)
            finally:
                recorder.close()
            self.assertTrue(summary["cumulative_bound_pass"])
            with (paths.files / "theory_train.csv").open() as handle:
                rates = [float(row["learning_rate"]) for row in csv.DictReader(handle)]
            np.testing.assert_allclose(rates, [.01 / math.sqrt(t + 1) for t in range(3)], rtol=1e-15)


if __name__ == "__main__":
    unittest.main()
