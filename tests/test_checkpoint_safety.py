"""Checkpoint cadence and publication must preserve a usable recovery point."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.config import CheckpointConfig, EvaluationConfig
from src.training.checkpoints import CheckpointStore
from src.training.experiment import ExperimentRunner
from tests.test_observation import ROOT, fixture_components, fixture_config


class CheckpointSafetyTests(unittest.TestCase):
    def test_new_snapshot_and_index_are_readable_before_old_snapshot_is_deleted(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            store = CheckpointStore(directory)
            store.save(0, {"step": 0}, ["initial"])
            store.save(10, {"step": 10}, ["latest"])
            old = directory / store.index["latest"]
            unlink = Path.unlink
            observed = []

            def checked_unlink(path, *args, **kwargs):
                if path == old:
                    # Verify on-disk publication too, rather than only memory.
                    fresh = CheckpointStore(directory)
                    self.assertEqual(fresh.load("latest"), {"step": 20})
                    self.assertEqual(fresh.load("initial"), {"step": 0})
                    observed.append(path.name)
                return unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", checked_unlink):
                store.save(20, {"step": 20}, ["latest"])
            self.assertEqual(observed, [old.name])
            self.assertFalse(old.exists())
            self.assertEqual(len(list(directory.glob("step_*.pkl"))), 2)

    def test_snapshot_write_failure_preserves_old_index_and_recovery_point(self):
        with TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp))
            store.save(10, {"step": 10}, ["latest"])
            before = store.index_path.read_bytes()

            def failed_dump(payload, handle, **kwargs):
                handle.write(b"partial snapshot")
                raise OSError("simulated full disk")

            with patch("src.training.checkpoints.pickle.dump", side_effect=failed_dump):
                with self.assertRaises(OSError):
                    store.save(20, {"step": 20}, ["latest"])
            self.assertEqual(store.index_path.read_bytes(), before)
            self.assertEqual(CheckpointStore(Path(tmp)).load("latest"), {"step": 10})

    def test_index_publication_failure_does_not_delete_old_snapshot(self):
        with TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp))
            store.save(10, {"step": 10}, ["latest"])
            before = store.index_path.read_bytes()
            with patch("src.training.checkpoints.atomic_json", side_effect=OSError("index write failed")):
                with self.assertRaises(OSError):
                    store.save(20, {"step": 20}, ["latest"])
            self.assertEqual(store.index_path.read_bytes(), before)
            self.assertEqual(CheckpointStore(Path(tmp)).load("latest"), {"step": 10})
            self.assertTrue((Path(tmp) / "step_000000020.pkl").exists())

    def test_cleanup_failure_leaves_new_checkpoint_published_and_recoverable(self):
        with TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp))
            store.save(10, {"step": 10}, ["latest"])
            with patch.object(Path, "unlink", side_effect=OSError("cleanup failed")):
                with self.assertRaises(OSError):
                    store.save(20, {"step": 20}, ["latest"])
            self.assertEqual(CheckpointStore(Path(tmp)).load("latest"), {"step": 20})
            self.assertTrue((Path(tmp) / "step_000000010.pkl").exists())

    def test_latest_cadence_and_milestones_remain_independent(self):
        for interval in (2, 5):
            with self.subTest(interval=interval), TemporaryDirectory() as tmp:
                c = fixture_config(tmp)
                c = replace(c, training=replace(c.training, total_steps=12),
                    evaluation=EvaluationConfig("steps", 100), checkpoints=CheckpointConfig(interval))
                trainer, source, evaluation, initial, _ = fixture_components(c)
                runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, console=False, run_name="cadence")
                with patch.object(runner.store, "save", wraps=runner.store.save) as save:
                    runner.run(initial)
                latest = [call.args[0] for call in save.call_args_list if "latest" in call.args[2]]
                self.assertEqual(latest, sorted(set([0, *range(interval, 13, interval), 12])))
                for slot, step in (("initial", 0), ("p25", 3), ("p50", 6), ("p75", 9), ("final", 12)):
                    self.assertEqual(runner.store.load(slot)["train_state"].step, step)
                self.assertEqual(runner.store.load("latest")["train_state"].step, 12)
                self.assertLessEqual(len(list(runner.paths.runs.glob("step_*.pkl"))), 7)
                metadata = json.loads((runner.paths.logs / "metadata.json").read_text())
                self.assertEqual(metadata["config"]["checkpoints"]["latest_every_steps"], interval)


if __name__ == "__main__":
    unittest.main()
