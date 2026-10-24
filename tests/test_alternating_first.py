"""First-step selection must survive CLI forwarding, observation and recovery."""

from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.config import DiagnosticConfig, MethodConfig, ReferenceConfig
from src.data.diagnostic_batches import TrainingAuditBatch
from src.methods import create_method
from src.probes.training_diagnostics import TrainingDiagnostics
from src.training import Trainer, TrainState
from src.training.experiment import ExperimentRunner
from src.training.launch import resolve_config
from src.training.updates import SGDUpdate
from tests.helpers import QuadraticTask, scalar_batch
from tests.test_observation import ROOT, fixture_config, fixture_components, read_rows


class AlternatingFirstTests(unittest.TestCase):
    def test_both_first_choices_match_explicit_sgd_sage_updates_for_each_reference(self):
        for first in ("sgd", "sage"):
            for reference in ("loss", "error"):
                method = create_method("sage_policy", reference_name=reference,
                    policy_mode="alternating", alternating_first=first, gradient_backend="jvp")
                state = TrainState({"x": jnp.zeros(1)}, {"calls": jnp.asarray(0)}, method.initial_state())
                expected = state._replace(method_state=None)
                trainer = Trainer(QuadraticTask(), method, SGDUpdate(.01))
                compiled = trainer.compile_step()
                for t in range(4):
                    alpha = (t + int(first == "sage")) % 2
                    baseline = Trainer(trainer.task, create_method("sage" if alpha else "sgd",
                        reference_name=reference, gradient_backend="jvp"), trainer.updater)
                    expected, _ = baseline.compile_step()(expected, scalar_batch(), jax.random.key(t))
                    state, metrics = compiled(state, scalar_batch(), jax.random.key(t))
                    self.assertEqual(float(metrics["policy_alpha"]), alpha)
                    self.assertEqual(int(state.model_state["calls"]), t+1)
                    for a, b in zip(jax.tree_util.tree_leaves(expected),
                                    jax.tree_util.tree_leaves(state._replace(method_state=None))):
                        np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-7)

    def test_shared_suite_forwards_first_choice_and_leaves_other_methods_unchanged(self):
        env = {**os.environ, "SAGE_PYTHON": sys.executable}
        labels = ("sage_l_alternating", "sage_e_alternating", "sgd", "sage_l", "sage_e_half")
        for script in ("run_sage_suite.sh", "run_inaturalist_suite.sh"):
            for first in ("sgd", "sage"):
                output = subprocess.check_output([str(ROOT/script), "--dry-run", "--methods",
                    *labels, "--alternating-first", first, "--seed", "5"],
                    cwd=gettempdir(), env=env, text=True)
                plans = json.loads(output)
                for label, plan in zip(labels, plans):
                    cmd = plan["command"]
                    get = lambda flag: cmd[cmd.index(flag)+1]
                    self.assertEqual(get("--alternating-first"), first)
                    config = resolve_config(get("--config"), mode=get("--mode"), method=get("--method"),
                        seed=int(get("--seed")), alternating_first=get("--alternating-first"))
                    if label.endswith("alternating"):
                        self.assertEqual(config.method.alternating_first, first)
                        self.assertEqual(plan["alternating_first"], first)
                        self.assertEqual("_first_sage" in plan["stdout"], first == "sage")
                    else:
                        self.assertEqual(config.method.alternating_first, "sgd")
                        self.assertIsNone(plan["alternating_first"])
        path = ROOT/"configs/tpu/imagenet100_resnet50_sgd.toml"
        default = resolve_config(path, mode="train", method="sage_l_alternating")
        self.assertEqual(default.method.alternating_first, "sgd")
        for bad in ("invalid", "SGD", True):
            with self.assertRaises(ValueError):
                resolve_config(path, mode="train", method="sage_l_alternating", alternating_first=bad)
        with self.assertRaises(ValueError):
            replace(default, method=MethodConfig("sage_policy", policy_mode="half", alternating_first="sage")).validate()

    def test_resume_observer_and_status_keep_the_selected_first_method(self):
        for first in ("sgd", "sage"):
            for reference in ("loss", "error"):
                with TemporaryDirectory() as tmp:
                    config = replace(fixture_config(tmp), reference=ReferenceConfig(reference),
                        method=MethodConfig("sage_policy", policy_mode="alternating", alternating_first=first),
                        diagnostics=DiagnosticConfig(enabled=True, steps=(0, 1, 2, 3),
                            audit_batches=1, lr_scales=(1.,)))
                    trainer, source, evaluation, initial, _ = fixture_components(
                        replace(config, method=MethodConfig("sage")))
                    method = create_method("sage_policy", reference_name=reference,
                        policy_mode="alternating", alternating_first=first, gradient_backend="jvp")
                    trainer = trainer.with_method(method)
                    observer = TrainingDiagnostics(trainer.task, method, config.diagnostics,
                        reference_name=reference, trajectory_method="sage_policy",
                        audit_batches=(TrainingAuditBatch("audit", source.batches[0]),))
                    runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                        train_sample_count=5, metadata={}, diagnostics=observer, console=False)
                    final = runner.run(initial)
                    phases = read_rows(runner.paths.files/"policy_steps.csv")
                    self.assertEqual([float(r["policy_alpha"]) for r in phases],
                        [(t+int(first == "sage")) % 2 for t in range(4)])
                    diagnostics = read_rows(runner.paths.files/"diagnostic_policy.csv")
                    for row in diagnostics:
                        self.assertEqual(float(row["policy_alpha"]), (int(row["step"])+int(first == "sage")) % 2)
                    for row in read_rows(runner.paths.files/"diagnostic_update.csv"):
                        self.assertLess(float(row["parameter_l2_error"]), 1e-6)
                    with (runner.paths.parent.logs/"run_status.tsv").open(newline="") as stream:
                        statuses = list(csv.DictReader(stream, delimiter="\t"))
                    for row in statuses:
                        self.assertEqual(row["method"].endswith("_first_sage"), first == "sage")
                    self.assertEqual(observer.metadata()["alternating_first"], first)
                    # An odd completed step must not restart the initial phase.
                    restored = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                        train_sample_count=5, metadata={}, diagnostics=observer, console=False,
                        run_name=runner.paths.name, resume=True).run(initial, resume_slot="p25")
                    for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(restored)):
                        np.testing.assert_array_equal(a, b)
                    self.assertEqual(read_rows(runner.paths.files/"policy_steps.csv"), phases)
                    changed = replace(config, method=replace(config.method,
                        alternating_first="sage" if first == "sgd" else "sgd"))
                    with self.assertRaisesRegex(ValueError, "configuration differs"):
                        ExperimentRunner(changed, trainer, source, evaluation, base_dir=ROOT,
                            train_sample_count=5, metadata={}, diagnostics=observer, console=False,
                            run_name=runner.paths.name, resume=True)

    def test_old_policy_metadata_missing_first_choice_defaults_to_sgd_without_rewrite(self):
        with TemporaryDirectory() as tmp:
            config = replace(fixture_config(tmp),
                method=MethodConfig("sage_policy", policy_mode="alternating"))
            trainer, source, evaluation, initial, _ = fixture_components(
                replace(config, method=MethodConfig("sage")))
            trainer = trainer.with_method(create_method("sage_policy", reference_name="error",
                policy_mode="alternating"))
            runner = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, run_name="old_policy", console=False)
            final = runner.run(initial)
            path = runner.paths.logs/"metadata.json"
            metadata = json.loads(path.read_text())
            del metadata["config"]["method"]["alternating_first"]
            path.write_text(json.dumps(metadata))
            restored = ExperimentRunner(config, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, run_name=runner.paths.name, resume=True,
                console=False).run(initial, resume_slot="p25")
            for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(restored)):
                np.testing.assert_array_equal(a, b)
            self.assertNotIn("alternating_first", json.loads(path.read_text())["config"]["method"])


if __name__ == "__main__":
    unittest.main()
