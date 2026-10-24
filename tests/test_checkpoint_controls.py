"""Checkpoint opt-out must preserve training and diagnostic evidence."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import jax
import numpy as np

from src.config import DiagnosticConfig
from src.probes.training_diagnostics import TrainingDiagnostics
from src.training.experiment import ExperimentRunner
from src.training.launch import add_overrides, resolve_config
from tests.test_observation import ROOT, fixture_components, fixture_config, read_rows


class CheckpointControlTests(unittest.TestCase):
    def test_disabled_saves_nothing_but_keeps_identical_training_and_probes(self):
        results = []
        for enabled in (True, False):
            with self.subTest(enabled=enabled), TemporaryDirectory() as tmp:
                config = fixture_config(tmp, probes=True)
                config = replace(config, checkpoints=replace(config.checkpoints, enabled=enabled),
                    diagnostics=DiagnosticConfig(enabled=True, transfer=False,
                        steps=(0, 1, 3), lr_scales=(1.,)))
                trainer, source, evaluation, initial, probes = fixture_components(config)
                observer = TrainingDiagnostics(trainer.task, trainer.method, config.diagnostics,
                    reference_name="error", trajectory_method="sage")
                runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, probes=probes, diagnostics=observer,
                    console=False, run_name="checkpoint_control")
                if enabled:
                    final = runner.run(initial)
                    self.assertTrue((runner.paths.runs / "index.json").exists())
                else:
                    # Even transient serialization/index writes are forbidden.
                    with patch.object(runner.store, "save", side_effect=AssertionError("checkpoint write")):
                        final = runner.run(initial)
                    self.assertEqual(list(runner.paths.runs.iterdir()), [])
                    self.assertFalse(any(r["event"] == "checkpoint" for r in
                        read_rows(runner.paths.files / "timing.csv")))
                self.assertEqual(len(read_rows(runner.paths.files / "diagnostic_update.csv")), 3)
                self.assertTrue((runner.paths.files / "theorem2.csv").exists())
                rows = read_rows(runner.paths.files / "train_steps.csv")
                results.append((jax.device_get(final), source.index,
                    [(r["step"], r["train_ce"], r["train_top1"]) for r in rows]))
                metadata = json.loads((runner.paths.logs / "metadata.json").read_text())
                self.assertEqual(metadata["config"]["checkpoints"]["enabled"], enabled)
        self.assertEqual(results[0][1:], results[1][1:])
        for a, b in zip(jax.tree_util.tree_leaves(results[0][0]), jax.tree_util.tree_leaves(results[1][0])):
            np.testing.assert_array_equal(a, b)

    def test_resume_disabled_is_rejected_and_legacy_enabled_metadata_is_compatible(self):
        with TemporaryDirectory() as tmp:
            config = fixture_config(tmp)
            trainer, source, evaluation, initial, _ = fixture_components(config)
            disabled = replace(config, checkpoints=replace(config.checkpoints, enabled=False))
            with self.assertRaisesRegex(ValueError, "Resume requires checkpoints enabled"):
                ExperimentRunner(disabled, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, run_name="missing", resume=True)
            runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, run_name="legacy", console=False)
            final = runner.run(initial)
            path = runner.paths.logs / "metadata.json"
            metadata = json.loads(path.read_text())
            del metadata["config"]["checkpoints"]["enabled"]
            path.write_text(json.dumps(metadata))
            trainer, source, evaluation, _, _ = fixture_components(config)
            restored = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, run_name="legacy", resume=True, console=False)
            actual = restored.run(initial, resume_slot="p50")
            for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(actual)):
                np.testing.assert_array_equal(a, b)
            self.assertNotIn("enabled", json.loads(path.read_text())["config"]["checkpoints"])

    def test_both_suites_forward_opt_out_and_default_remains_on(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        add_overrides(parser)
        path = ROOT / "configs/tpu/imagenet100_resnet50_sgd.toml"
        self.assertTrue(resolve_config(path, mode="train").checkpoints.enabled)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            replace(fixture_config("unused"), checkpoints=replace(
                fixture_config("unused").checkpoints, enabled=0)).validate()
        env = {**os.environ, "SAGE_PYTHON": sys.executable}
        for suite in ("run_sage_suite.sh", "run_inaturalist_suite.sh"):
            for flag, enabled in (("--no-checkpoints", False), ("--checkpoints", True)):
                with self.subTest(suite=suite, flag=flag):
                    output = subprocess.check_output(["bash", str(ROOT / suite), "--config", str(path),
                        "--methods", "sgd", "sage_e_half", "--steps", "5", flag,
                        "--no-theorem-probes", "--dry-run"], cwd=ROOT, env=env, text=True)
                    for plan in json.loads(output):
                        self.assertEqual(plan["checkpoints_enabled"], enabled)
                        flags, _ = parser.parse_known_args(plan["command"][4:])
                        child = resolve_config(path, mode="train", method=flags.method,
                            checkpoints=flags.checkpoints, theorem_probes=False)
                        self.assertEqual(child.checkpoints.enabled, enabled)
        failure = subprocess.run([sys.executable, str(ROOT / "scripts/launch_tpu.py"),
            "--config", str(path), "--resume", "old", "--no-checkpoints", "--dry-run"],
            cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("Resume requires checkpoints enabled", failure.stderr)
