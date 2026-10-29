"""Mathematical oracles and observational isolation, not a DPCO win condition."""

from dataclasses import replace
from itertools import product
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

from src.data.batch import classification_batch
from src.probes.theory_validation import (SCHEMAS, ExperimentSpec, LogisticSystem,
    analytic_cases, logistic_oracle, moment_rows, oracle_directions, paired_rows,
    run_experiment, synthetic_data)
from src.training.recording import RunRecorder, SCHEMAS as LEGACY_SCHEMAS
from src.training.results import ResultsPaths
from src.training.learning_rates import CosineLearningRate
from tests.test_observation import read_rows


ROOT = Path(__file__).resolve().parents[1]


class TheoryValidationTests(unittest.TestCase):
    def test_reference_masks_jvp_oracle_valid_denominator_and_ties(self):
        system = LogisticSystem(np.array([[-.01], [1.], [100.]], np.float32), np.array([1, 1, 0]))
        params = {"theta": jnp.array([1.])}
        batch = classification_batch(system.x, system.y, [True, True, False])
        for reference in ("loss", "error"):
            info = system.directions(params, batch, reference)
            gd, gs, _, _ = oracle_directions([1.], system.x, system.y, reference, [True, True, False])
            np.testing.assert_allclose(info["gd"], gd, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(info["gs"], gs, rtol=1e-5, atol=1e-7)
            self.assertAlmostEqual(info["identity_residual"], 0., delta=2e-7)
            self.assertAlmostEqual(info["transfer_residual"], 0., delta=1e-12)
            self.assertGreaterEqual(info["energy_slack"], -2e-7)
        # Correctly classified contributions remain eligible: sample 2 flips in E.
        self.assertEqual(system.directions(params, batch, "error")["flip_count"], 1)
        balanced = LogisticSystem(np.array([[1.], [1.]], np.float32), np.array([0, 1]))
        info = balanced.directions({"theta": jnp.array([0.])}, balanced.full_batch, "loss")
        self.assertEqual(info["flip_count"], 0)
        np.testing.assert_array_equal(info["gs"], info["gd"])
        # The vector gradient is orthogonal to a zero-feature sample: q=0 stays +1.
        self.assertEqual(float(balanced.methods["dpco_l"].weights(jnp.array([0.]), True, jnp.array([True]))[0]), 1.)

    def test_empty_error_fallback_and_all_error_equals_loss(self):
        for labels, all_error in (([1, 0], False), ([0, 1], True)):
            system = LogisticSystem(np.array([[1.], [-1.]], np.float32), np.array(labels))
            params = {"theta": jnp.array([1.])}
            de = system.directions(params, system.full_batch, "error")
            target = system.kernels["dpco_l" if all_error else "sgd"](params, system.full_batch)
            np.testing.assert_allclose(de["gs"], target, atol=1e-7)
            self.assertEqual(de["selected_count"], 2 if all_error else 0)

    def test_all_three_analytic_cases_and_conditional_expectation(self):
        cases = analytic_cases(2048)
        json.dumps(cases, allow_nan=False)
        self.assertTrue(cases["all_checks_pass"], cases)
        self.assertEqual(cases["A"]["first_crossover"], 16)
        self.assertAlmostEqual(cases["A"]["same_state_H"], .0002375, places=8)
        self.assertGreater(cases["B"]["delta"], 0)
        self.assertLess(cases["B"]["h_dot_Gs"], 0)
        self.assertEqual(cases["C"]["ordered_batch_count"], 9)
        for row in cases["C"]["moments"]:
            if row["estimator"].startswith("exact"):
                self.assertEqual(row["probability_sum"], 1.)
                self.assertIsNone(row["standard_error"])

    def test_frozen_error_objective_zero_lr_and_unsigned_ce(self):
        system = LogisticSystem(np.array([[-.01], [1.]], np.float32), np.array([1, 1]))
        params = {"theta": jnp.array([1.])}
        rows = list(paired_rows(system, params, np.arange(2), "error", [0., .01, 1e-8], "source"))
        self.assertEqual({r["objective"] for r in rows},
            {"source_full_batch_ce", "source_frozen_error_ce", "fixed_full_training_ce"})
        error = next(r for r in rows if r["objective"] == "source_frozen_error_ce" and r["probe_lr"] == .01)
        self.assertAlmostEqual(error["objective_gain"], error["reference_delta"], places=7)
        full = next(r for r in rows if r["objective"] == "fixed_full_training_ce" and r["probe_lr"] == .01)
        self.assertGreater(full["difference"], 0)
        self.assertLess(full["objective_gain"], 0)
        for row in rows:
            if row["probe_lr"] == 0:
                self.assertIsNone(row["first_order_residual"])
                self.assertIsNone(row["difference_over_lr"])
        correct = LogisticSystem(np.array([[1.], [-1.]], np.float32), np.array([1, 0]))
        rows = list(paired_rows(correct, params, np.arange(2), "error", [.01], "correct"))
        self.assertNotIn("source_frozen_error_ce", {r["objective"] for r in rows})

    def test_full_gradient_smoothness_and_near_zero_ratio(self):
        x, y = synthetic_data(8, 2, 42)
        system = LogisticSystem(x, y)
        theta = np.array([.4, -.2])
        _, gradients, _ = logistic_oracle(theta, x, y)
        for j in range(2):
            e = np.eye(2)[j] * 1e-5
            difference = (logistic_oracle(theta+e, x, y)[0].mean()
                - logistic_oracle(theta-e, x, y)[0].mean()) / 2e-5
            self.assertAlmostEqual(difference, gradients.mean(0)[j], places=8)
        z = x.astype(np.float64) @ theta
        p = 1/(1+np.exp(-z))
        hessian = x.T @ ((p*(1-p))[:, None]*x) / len(x)
        self.assertLessEqual(np.linalg.eigvalsh(hessian)[-1], system.smoothness+1e-12)
        batches = np.array(list(product(range(8), repeat=1)))
        rows = list(moment_rows(system, {"theta": jnp.zeros(2)}, batches, "loss", exact=True))
        ratio = next(r for r in rows if r["statistic"] == "alignment_ratio")
        self.assertFalse(ratio["defined"])
        self.assertIsNone(ratio["mean"])

    def test_probe_count_isolation_actual_schedules_and_cumulative_bound(self):
        x, y = synthetic_data(8, 2)
        system = LogisticSystem(x, y)
        for method in ("sgd", "dpco_e", "dpco_l"):
            for schedule in ("constant", "cosine"):
                with self.subTest(method=method, schedule=schedule):
                    results = []
                    for count in (0, 3, 7):
                        with TemporaryDirectory() as tmp:
                            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("isolated")
                            paths.create()
                            recorder = RunRecorder(paths, schemas=SCHEMAS, console=False)
                            spec = ExperimentSpec(method=method, schedule=schedule, steps=5, batch_size=3,
                                probes=count, probe_steps=(0, 2, 4), probe_lrs=(0., .01), lr=.05)
                            final, summary = run_experiment(system, spec, recorder)
                            recorder.close()
                            rows = read_rows(paths.files / "theory_train.csv")
                            self.assertTrue(summary["cumulative_bound_pass"])
                            expected = [.05]*5 if schedule == "constant" else [CosineLearningRate(.05, 0, 5)(t) for t in range(5)]
                            np.testing.assert_allclose([float(r["learning_rate"]) for r in rows], expected, atol=1e-10)
                            results.append((final, summary["training_indices_sha256"],
                                [(r["full_train_ce_after"], r["batch_indices_json"]) for r in rows]))
                    for result in results[1:]:
                        np.testing.assert_array_equal(result[0].params["theta"], results[0][0].params["theta"])
                        self.assertEqual(result[1:], results[0][1:])

    def test_private_schemas_do_not_mutate_ordinary_cursors(self):
        before = dict(LEGACY_SCHEMAS)
        with TemporaryDirectory() as tmp:
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("private")
            paths.create()
            recorder = RunRecorder(paths, schemas=SCHEMAS, console=False)
            self.assertEqual(set(recorder.cursors()), set(SCHEMAS))
            recorder.record("theory_train", step=1)
            cursor = recorder.cursors()
            recorder.record("theory_train", step=2)
            recorder.flush()
            recorder.rewind(cursor)
            self.assertEqual([r["step"] for r in read_rows(paths.files/"theory_train.csv")], ["1"])
            recorder.close()
            ordinary = RunRecorder(paths, console=False, resume=True)
            self.assertEqual(set(ordinary.cursors()), set(before))
            ordinary.close()
        self.assertEqual(LEGACY_SCHEMAS, before)

    def test_shell_grid_and_invalid_cli_are_cpu_only(self):
        env = {**os.environ, "DPCO_PYTHON": sys.executable, "JAX_PLATFORMS": "tpu"}
        cmd = ["bash", str(ROOT / "run_dpco_theory_suite.sh"), "--dry-run", "--seeds", "2", "5",
            "--lrs", ".01", ".05", "--schedules", "constant", "cosine"]
        plan = json.loads(subprocess.check_output(cmd, cwd=ROOT, env=env, text=True))
        self.assertEqual(plan["backend"], "cpu_only")
        self.assertEqual(len(plan["grid"]), 24)
        failed = subprocess.run(cmd+["--lrs", "nan"], cwd=ROOT, env=env, text=True, capture_output=True)
        self.assertNotEqual(failed.returncode, 0)

    def test_failed_grid_member_is_retained_and_does_not_stop_later_lr(self):
        with TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_dpco_theory.py"),
                "--methods", "sgd", "--lrs", "1e300", ".01", "--steps", "1",
                "--no-probes", "--case-mc-samples", "256", "--output", tmp],
                cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            suite = next((Path(tmp)/"logs").glob("theory_validation_*"))
            results = json.loads((suite/"grid_results.json").read_text())
            self.assertEqual([r["status"] for r in results], ["failed", "completed"])
            self.assertTrue((Path(tmp)/"logs"/results[0]["run_name"]/"failure.txt").exists())


if __name__ == "__main__":
    unittest.main()
