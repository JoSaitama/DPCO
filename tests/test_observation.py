"""Event, retention, recovery, and statistical checks for the recording contract."""

import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.config import (CheckpointConfig, EvaluationConfig, LoggingConfig,
                        ProbeConfig, ResultsConfig, load_config)
from src.data.batch import ClassificationBatch, classification_batch
from src.methods import create_method
from src.probes.classification import ClassificationProbes, ProbeBatch
from src.references import create_reference
from src.tasks.classification import ClassificationTask
from src.training import TrainState, Trainer
from src.training.checkpoints import CheckpointStore
from src.training.evaluation import EvaluationSpec
from src.training.experiment import ExperimentRunner
from src.training.recording import RunRecorder
from src.training.results import ResultsPaths, make_run_name
from src.training.schedules import EventSchedule, milestone_steps
from src.training.status import RunStatus
from src.training.updates import SGDUpdate


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class FixedSource:
    """Host arrays, with a deliberately unequal padded tail batch."""
    def __init__(self):
        self.index = 0
        self.batches = [
            ClassificationBatch(np.array([[1.], [-2.], [-2.]], np.float32),
                                np.array([0, 0, 0], np.int32), np.ones(3, bool)),
            ClassificationBatch(np.array([[-2.], [1.], [100.]], np.float32),
                                np.array([0, 0, 1], np.int32), np.array([True, True, False])),
        ]

    def __next__(self):
        batch = self.batches[self.index % 2]
        self.index += 1
        return batch

    def snapshot(self):
        return {"index": self.index}

    def restore(self, state):
        self.index = state["index"]


def fixture_config(root, *, probes=False):
    original = load_config(ROOT / "configs/imagenet100_resnet50_sage_e.toml")
    return replace(original, results=ResultsConfig(str(root)),
        training=replace(original.training, total_steps=4),
        logging=LoggingConfig("steps", 2, 3, 3),
        evaluation=EvaluationConfig("epochs", .5),
        checkpoints=CheckpointConfig(2, True),
        probes=ProbeConfig(enabled=probes, batches_per_node=1))


def fixture_components(config):
    def model_fn(params, images, state):
        return images @ params["w"] + params["b"], state
    task = ClassificationTask(model_fn, 2)
    state = TrainState({"w": jnp.array([[0., 1.]]), "b": jnp.zeros(2)})
    source = FixedSource()
    evaluation = EvaluationSpec(task, lambda: iter(source.batches), 5, "deterministic_linear")
    trainer = Trainer(task, create_method(config.method.name, reference_name=config.reference.name),
                      SGDUpdate(config.training.learning_rate))
    probe_batch = classification_batch(*source.batches[0])
    def fixed_audit(p):
        return jnp.mean(task.evaluate(p, probe_batch).losses)
    probes = ClassificationProbes(task, create_reference("error"),
        [ProbeBatch("fixed_batch_0", probe_batch)], reference_name="error",
        objective_mode="deterministic_linear", audit_objective=fixed_audit,
        audit_name="fixed_training_audit")
    return trainer, source, evaluation, state, probes


class ObservationTests(unittest.TestCase):
    def test_names_isolate_runs_and_reject_overwrites(self):
        with TemporaryDirectory() as tmp:
            config = fixture_config(tmp)
            name = make_run_name(config, timestamp=datetime(2026, 10, 17, 14, 30, 25, tzinfo=timezone.utc))
            self.assertEqual(name, "imagenet100_clean_resnet50_sage_e_42_0.01_20261017_143025")
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment(name)
            paths.create()
            with self.assertRaises(FileExistsError):
                paths.create()
            paths.create(resume=True)
            with self.assertRaises(ValueError):
                ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("../outside")

    def test_fractional_epochs_percentage_and_threshold_rounding(self):
        schedule = EventSchedule("epochs", .25, total_steps=11, train_sample_count=10)
        self.assertFalse(schedule.due(1, 2))
        self.assertTrue(schedule.due(2, 4))
        self.assertFalse(schedule.due(2, 4))
        percent = EventSchedule("percent", 1, total_steps=11, train_sample_count=10)
        self.assertTrue(percent.due(1, 2))
        self.assertEqual(milestone_steps(3), {0, 1, 2, 3})

    def test_buffered_rows_and_immediate_log_are_independent(self):
        with TemporaryDirectory() as tmp:
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("test")
            paths.create()
            recorder = RunRecorder(paths, metadata={"seed": 1}, console=False)
            recorder.record("train_steps", step=1, train_ce=.5)
            recorder.log("train", step=1)
            self.assertFalse((paths.files / "train_steps.csv").exists())
            self.assertIn('"step": 1', (paths.logs / "train.log").read_text())
            recorder.flush()
            cursor = recorder.cursors()
            recorder.record("train_steps", step=2)
            recorder.flush()
            recorder.rewind(cursor)
            recorder.close()
            self.assertEqual([row["step"] for row in read_rows(paths.files / "train_steps.csv")], ["1"])

    def test_checkpoint_slots_share_state_and_prune_unreferenced_snapshots(self):
        with TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp))
            store.save(0, {"value": 0}, ["initial", "best", "latest"])
            store.save(1, {"value": 1}, ["latest"])
            store.save(2, {"value": 2}, ["latest"])
            self.assertFalse((Path(tmp) / "step_000000001.pkl").exists())
            self.assertEqual(store.load("initial")["value"], 0)
            self.assertEqual(store.load()["value"], 2)
            self.assertEqual(len(list(Path(tmp).glob("*.pkl"))), 2)

    def test_validation_weights_unequal_batches_and_does_not_commit_state(self):
        c = fixture_config("unused")
        _, source, evaluation, state, _ = fixture_components(c)
        result = evaluation.evaluate(state)
        logits = np.concatenate([np.asarray(batch.images)[batch.valid_mask] for batch in source.batches]) @ np.asarray(state.params["w"])
        ce = np.logaddexp(logits[:, 0], logits[:, 1]) - logits[:, 0]
        self.assertAlmostEqual(result["val_ce"], float(ce.mean()), places=6)
        self.assertEqual(result["eval_sample_count"], 5)
        self.assertAlmostEqual(result["val_top1"], .6)
        self.assertIsNone(state.model_state)

    def test_full_recording_pipeline_and_exact_resume(self):
        with TemporaryDirectory() as tmp:
            c = fixture_config(tmp, probes=True)
            trainer, source, evaluation, state, probes = fixture_components(c)
            runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={"execution_kind": "synthetic", "actual_model": "linear_2class"},
                probes=probes, console=False, run_name="integration")
            final = runner.run(state)
            files = runner.paths.files
            rows = read_rows(files / "train_steps.csv")
            self.assertEqual([int(row["step"]) for row in rows], [1, 2, 3, 4])
            self.assertEqual([int(row["samples_seen"]) for row in rows], [3, 5, 8, 10])
            t1 = read_rows(files / "theorem1.csv")
            self.assertEqual(sum(row["phase"] == "train" for row in t1), 4)
            self.assertTrue(all(abs(float(row["identity_residual"])) < 1e-5 for row in t1 if row["phase"] == "node"))
            self.assertEqual(len(read_rows(files / "theorem2.csv")), 5 * 2 * 5)
            self.assertEqual(len(read_rows(files / "evaluation.csv")), 5)
            self.assertLessEqual(len(list(runner.paths.runs.glob("*.pkl"))), 7)
            t3_train = [row for row in read_rows(files / "theorem3.csv") if row["phase"] == "train"]
            t1_train = [row for row in t1 if row["phase"] == "train"]
            self.assertAlmostEqual(float(t3_train[-1]["cumulative_reference_gain"]),
                                   sum(float(row["eta_delta"]) for row in t1_train))
            original_metrics = [(row["train_ce"], row["train_top1"]) for row in rows]
            trainer2, source2, evaluation2, _, probes2 = fixture_components(c)
            resumed = ExperimentRunner(c, trainer2, source2, evaluation2, base_dir=ROOT,
                train_sample_count=5, metadata={}, probes=probes2, console=False,
                run_name="integration", resume=True)
            final2 = resumed.run(state, resume_slot="p50")
            for a, b in zip(jax.tree_util.tree_leaves(final.params), jax.tree_util.tree_leaves(final2.params)):
                np.testing.assert_array_equal(a, b)
            rows2 = read_rows(files / "train_steps.csv")
            self.assertEqual([int(row["step"]) for row in rows2], [1, 2, 3, 4])
            self.assertEqual(original_metrics, [(row["train_ce"], row["train_top1"]) for row in rows2])
            self.assertEqual(source2.index, 4)
            with (runner.paths.parent.logs / "run_status.tsv").open(newline="") as handle:
                status = list(csv.DictReader(handle, delimiter="\t"))
            attempts = {row["attempt_id"] for row in status}
            self.assertEqual(len(attempts), 2)
            for attempt in attempts:
                rows = [row for row in status if row["attempt_id"] == attempt]
                self.assertEqual(rows[-1]["status"], "completed")
                self.assertEqual([row["event"] for row in rows], ["start", "end"])
            self.assertEqual(len(status), 4)
            self.assertTrue(any(row["event"] == "start" and row["resume"] == "True" for row in status))

    def test_disabled_probes_do_not_execute_or_change_the_training_trajectory(self):
        with TemporaryDirectory() as tmp:
            c = fixture_config(tmp)
            trainer, source, evaluation, state, _ = fixture_components(c)
            class ForbiddenProbes:
                def run(self, *args, **kwargs):
                    raise AssertionError("Disabled probes must not run")
            runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, probes=ForbiddenProbes(), console=False,
                run_name="off")
            final = runner.run(state)
            expected = state
            source2 = FixedSource()
            for _ in range(4):
                expected, _ = trainer.step(expected, next(source2))
            for a, b in zip(jax.tree_util.tree_leaves(final.params), jax.tree_util.tree_leaves(expected.params)):
                np.testing.assert_allclose(a, b, atol=1e-7)
            self.assertFalse(any(runner.paths.files.glob("theorem*.csv")))

    def test_nonfinite_run_records_failure(self):
        with TemporaryDirectory() as tmp:
            c = fixture_config(tmp)
            trainer, source, evaluation, state, _ = fixture_components(c)
            source.batches[0].images[0, 0] = np.nan
            runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, console=False, run_name="failed")
            with self.assertRaises(FloatingPointError):
                runner.run(state)
            summary = read_rows(runner.paths.files / "summary.csv")
            self.assertEqual(summary[-1]["status"], "failed")
            with (runner.paths.parent.logs / "run_status.tsv").open(newline="") as handle:
                status = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(status[-1]["status"], "failed")

    def test_recording_reuses_the_launcher_attempt_without_duplicate_failure(self):
        with TemporaryDirectory() as tmp:
            c = fixture_config(tmp)
            trainer, source, evaluation, state, _ = fixture_components(c)
            paths = ResultsPaths.resolve(tmp, base_dir=ROOT).experiment("shared_attempt")
            status = RunStatus(paths, c, mode="smoke")
            status.record("started")
            source.batches[0].images[0, 0] = np.nan
            with self.assertRaises(FloatingPointError), status.monitor():
                runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False,
                    run_name=paths.name, status=status)
                runner.run(state)
            with status.path.open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual({row["attempt_id"] for row in rows}, {status.attempt_id})
            self.assertEqual(sum(row["status"] == "failed" for row in rows), 1)

    def test_window_buffering_percent_logs_and_final_evaluation_deduplicate(self):
        with TemporaryDirectory() as tmp:
            c = fixture_config(tmp)
            c = replace(c, training=replace(c.training, total_steps=7),
                logging=LoggingConfig("percent", 50, 3, 3),
                evaluation=EvaluationConfig("steps", 7), checkpoints=CheckpointConfig(100))
            trainer, source, evaluation, state, _ = fixture_components(c)
            runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, console=False, run_name="percent")
            runner.run(state)
            logs = [json.loads(line) for line in (runner.paths.logs / "train.log").read_text().splitlines()]
            self.assertEqual([row["step"] for row in logs if row["event"] == "train"], [4, 7])
            rows = read_rows(runner.paths.files / "train_steps.csv")
            self.assertEqual([int(row["step"]) for row in rows], list(range(1, 8)))
            self.assertEqual(len(read_rows(runner.paths.files / "evaluation.csv")), 2)
            windows = [row for row in read_rows(runner.paths.files / "timing.csv") if row["event"] == "train_window"]
            self.assertEqual(sum(int(row["window_steps"]) for row in windows), 7)
            self.assertTrue(any(int(row["window_steps"]) > 1 for row in windows))

    def test_random_flip_resume_reconstructs_the_step_key_stream(self):
        from src.config import MethodConfig
        with TemporaryDirectory() as tmp:
            c = replace(fixture_config(tmp), method=MethodConfig("random_flip", .5))
            trainer, source, evaluation, state, _ = fixture_components(c)
            trainer = Trainer(trainer.task, create_method("random_flip", flip_probability=.5), trainer.updater)
            runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, console=False, run_name="random")
            final = runner.run(state)
            source2 = FixedSource()
            resumed = ExperimentRunner(c, trainer, source2, evaluation, base_dir=ROOT,
                train_sample_count=5, metadata={}, console=False, run_name="random", resume=True)
            final2 = resumed.run(state, resume_slot="p25")
            for a, b in zip(jax.tree_util.tree_leaves(final.params), jax.tree_util.tree_leaves(final2.params)):
                np.testing.assert_array_equal(a, b)

    def test_probe_state_is_read_only_and_objective_masks_are_frozen(self):
        c = fixture_config("unused", probes=True)
        _, _, _, state, probes = fixture_components(c)
        before = jax.tree_util.tree_map(np.array, state.params)
        rows = probes.run(state, step=0, learning_rate=.1, config=c.probes,
                          cumulative=dict(cumulative_reference_gain=0., cumulative_eta=0., cumulative_eta_sq=0.))
        for key in before:
            np.testing.assert_array_equal(before[key], state.params[key])
        reference_rows = [row for stream, row in rows if stream == "theorem2" and row["objective"] == "frozen_error_ce"]
        self.assertTrue(all(row["selected_count"] == 1 for row in reference_rows))
        self.assertLess(abs(reference_rows[-1]["first_order_residual"]), abs(reference_rows[0]["first_order_residual"]))
