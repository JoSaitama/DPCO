"""Validate actual evaluation triggers, deduplication, state isolation and resume."""

from dataclasses import replace
import json
from tempfile import TemporaryDirectory
import unittest

import jax
import numpy as np

from src.config import EvaluationConfig
from src.training.experiment import ExperimentRunner
from tests.test_observation import ROOT, fixture_components, fixture_config, read_rows


class ValidationControlTests(unittest.TestCase):
    def test_step_and_fractional_epoch_cadences_include_initial_and_final_once(self):
        cases = (("steps", 3, 7, [0, 3, 6, 7]),
                 ("steps", 3, 6, [0, 3, 6]),
                 ("steps", 100, 5, [0, 5]),
                 ("epochs", 1., 5, [0, 2, 4, 5]),
                 ("epochs", .5, 4, [0, 1, 2, 3, 4]),
                 ("epochs", .25, 4, [0, 1, 2, 3, 4]))
        for mode, interval, steps, expected in cases:
            with self.subTest(mode=mode, interval=interval, steps=steps), TemporaryDirectory() as tmp:
                c = fixture_config(tmp)
                c = replace(c, training=replace(c.training, total_steps=steps),
                            evaluation=EvaluationConfig(mode, interval))
                trainer, source, evaluation, initial, _ = fixture_components(c)
                runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="validation")
                runner.run(initial)
                rows = read_rows(runner.paths.files / "evaluation.csv")
                self.assertEqual([int(row["step"]) for row in rows], expected)
                self.assertTrue(all(int(row["eval_sample_count"]) == 5 for row in rows))
                metadata = json.loads((runner.paths.logs / "metadata.json").read_text())
                self.assertEqual(metadata["config"]["evaluation"], {"mode": mode, "interval": interval})

    def test_evaluation_frequency_does_not_change_training_parameters_or_data_cursor(self):
        results = []
        for interval in (1, 100):
            with TemporaryDirectory() as tmp:
                c = fixture_config(tmp)
                c = replace(c, evaluation=EvaluationConfig("steps", interval))
                trainer, source, evaluation, initial, _ = fixture_components(c)
                runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="isolation")
                final = runner.run(initial)
                results.append((jax.device_get(final), source.index))
        self.assertEqual(results[0][1], results[1][1])
        for a, b in zip(jax.tree_util.tree_leaves(results[0][0]), jax.tree_util.tree_leaves(results[1][0])):
            np.testing.assert_array_equal(a, b)

    def test_resume_continues_evaluation_schedule_without_duplicate_rows(self):
        with TemporaryDirectory() as tmp:
            c = fixture_config(tmp)
            c = replace(c, training=replace(c.training, total_steps=8), evaluation=EvaluationConfig("steps", 3))
            trainer, source, evaluation, initial, _ = fixture_components(c)
            runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, console=False, run_name="resume")
            final = runner.run(initial)
            _, source2, evaluation2, _, _ = fixture_components(c)
            resumed = ExperimentRunner(c, trainer, source2, evaluation2, base_dir=ROOT,
                train_sample_count=5, metadata={}, console=False, run_name="resume", resume=True)
            restored = resumed.run(initial, resume_slot="p50")
            rows = read_rows(runner.paths.files / "evaluation.csv")
            self.assertEqual([int(row["step"]) for row in rows], [0, 3, 6, 8])
            for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(restored)):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
