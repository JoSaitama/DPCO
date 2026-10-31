"""Offline integrity, restart, label isolation and actual training-entry checks."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import BytesIO, StringIO
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from src.config import ResultsConfig, load_config
from src.data.classification import open_training_images
from src.data.images import ImageBatchSource
from src.data.imagenet_hf import DataPlan, label_mapping
from src.data.imagenet_clip_noise import (canonical, score_images, publish_noise, verify_noise,
    export_audit, prediction_summary, selection_order, source_description)
from src.models.resnet import ResNet
from src.tasks.classification import ClassificationTask
from src.training import TrainState
from src.training.checkpoints import CheckpointStore
from src.training.evaluation import noisy_classification_evaluation
from src.training.launch import resolve_config
from tests.test_dpco_history import assert_tree

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/tpu/dpco/imagenet100_resnet50.toml"


def fixture(root, n=20):
    config = load_config(PRESET)
    mapping, digest = label_mapping(config.data, ROOT)
    plan = DataPlan(config.data.hf_repo_id, config.data.hf_revision, mapping, digest)
    path = Path(root) / plan.fingerprint[:24] / "images.sqlite"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        db.execute("CREATE TABLE samples(split TEXT,id INTEGER,label INTEGER,image BLOB,source_index INTEGER,PRIMARY KEY(split,id))")
        for key, value in (("identity", plan.identity), ("complete", True),
                           ("original_label_names", [f"fixture_class_{i}" for i in range(1000)])):
            db.execute("INSERT INTO meta VALUES(?,?)", (key, canonical(value).decode()))
        for split, count in (("train", n), ("validation", 3)):
            for i in range(count):
                stream = BytesIO()
                Image.new("RGB", (12, 14), (i, i, i)).save(stream, format="PNG")
                db.execute("INSERT INTO samples VALUES(?,?,?,?,?)", (split, i, i % 3, stream.getvalue(), i + 100))
    return path


class FakePredictor:
    identity = {"model": "synthetic_fixture_not_CLIP", "checkpoint_sha256": "fixture"}

    def __call__(self, images):
        logits = np.full((len(images), 100), -5., dtype=np.float64)
        for row, image in enumerate(images):
            i = image.getpixel((0, 0))[0]
            label = i % 3
            logits[row, label] = 2
            logits[row, (label + 1) % 3] = -2 + i * .3
        return logits


def artifact(root, percent=20, seed=7, output="noise"):
    db = fixture(Path(root) / "data")
    scores = Path(root) / "scores"
    score_images(db, scores, FakePredictor(), batch_size=4)
    manifest = publish_noise(scores, Path(root) / output, percent=percent, seed=seed)
    return db, scores, manifest


class CLIPNoiseTests(unittest.TestCase):
    def test_summary_candidates_and_invalid_controls(self):
        wrong, true, false, top1 = prediction_summary([[10., 2., 1.], [1., 2., 10.]], [0, 1])
        np.testing.assert_array_equal(wrong, [1, 2])
        np.testing.assert_array_equal(top1, [0, 2])
        self.assertLess(false[0] - true[0], 0)
        self.assertGreater(false[1] - true[1], 0)
        for logits in ([[float("nan"), 0]], [[1]]):
            with self.assertRaises(ValueError):
                prediction_summary(logits, [0])
        for seed, tau in ((True, .2), (-1, .2), (0, 0), (0, float("nan"))):
            with self.assertRaises(ValueError):
                selection_order([.1], seed, tau)

    def test_exact_quota_nested_seeds_and_artifact_checksums(self):
        with TemporaryDirectory() as tmp:
            db, scores, manifest = artifact(tmp)
            first, a, _ = verify_noise(manifest)
            self.assertEqual(first["corrupted_count"], 4)
            for percent in (0., 20., 33.3, 40., 100.):
                path = publish_noise(scores, Path(tmp) / str(percent), percent=percent, seed=7)
                m, b, _ = verify_noise(path)
                self.assertEqual(np.count_nonzero(b["is_corrupted"]), int(np.floor(20 * percent / 100 + .5)))
                np.testing.assert_array_equal(a["original_label"], b["original_label"])
                if percent == 20:
                    np.testing.assert_array_equal(a["noisy_label"], b["noisy_label"])
                if percent == 40:
                    self.assertTrue(np.all(b["is_corrupted"][a["is_corrupted"]]))
            other = publish_noise(scores, Path(tmp) / "seed8", seed=8)
            self.assertFalse(np.array_equal(a["noisy_label"], verify_noise(other)[1]["noisy_label"]))
            with self.assertRaises(FileExistsError):
                publish_noise(scores, manifest.parent)
            (manifest.parent / "labels.csv").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_noise(manifest)

    def test_scoring_commits_batches_resumes_and_rejects_identity_changes(self):
        with TemporaryDirectory() as tmp:
            db = fixture(Path(tmp) / "data")
            scores = Path(tmp) / "scores"
            calls = []

            class Interrupted(FakePredictor):
                def __call__(self, images):
                    calls.append(1)
                    if len(calls) == 2:
                        raise RuntimeError("interrupted")
                    return super().__call__(images)

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                score_images(db, scores, Interrupted(), batch_size=4)
            with sqlite3.connect(scores / "scores.sqlite") as cache:
                self.assertEqual(cache.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 4)
            predictor = FakePredictor()
            score_images(db, scores, predictor, batch_size=4)
            with patch.object(FakePredictor, "__call__", side_effect=AssertionError("must reuse")):
                score_images(db, scores, predictor, batch_size=4)
            predictor.identity = {"model": "different"}
            with self.assertRaisesRegex(ValueError, "identity differs"):
                score_images(db, scores, predictor, batch_size=4)

    def test_uniform_audit_and_source_image_integrity(self):
        with TemporaryDirectory() as tmp:
            db, _, manifest = artifact(tmp)
            for name in ("audit1", "audit2"):
                page = export_audit(db, manifest, Path(tmp) / name, count=3, seed=8)
                self.assertTrue(page.exists())
                self.assertIn("Original:", page.read_text())
            one = json.loads((Path(tmp) / "audit1/audit.json").read_bytes())
            two = json.loads((Path(tmp) / "audit2/audit.json").read_bytes())
            self.assertEqual(one, two)
            self.assertEqual(len(one["sample_ids"]), 6)
            with self.assertRaises(FileExistsError):
                export_audit(db, manifest, Path(tmp) / "audit1")
            with sqlite3.connect(db) as source:
                source.execute("UPDATE samples SET image=? WHERE split='train'", (b"changed",))
            with self.assertRaisesRegex(ValueError, "image bytes changed"):
                export_audit(db, manifest, Path(tmp) / "audit3", count=1)

    def test_reader_truth_isolation_shuffle_restore_and_group_metrics(self):
        with TemporaryDirectory() as tmp:
            _, _, manifest = artifact(tmp)
            c = resolve_config(PRESET, mode="train", data_dir=str(Path(tmp) / "data"), noise_manifest=str(manifest))
            images, identity = open_training_images(c.data, ROOT)
            try:
                self.assertEqual(identity["selection"]["noise_seed"], 7)
                source = ImageBatchSource(images, 4, 8, 42)
                next(source)
                snap = source.snapshot()
                expected = next(source)
                source.restore(snap)
                assert_tree(self, expected, next(source), exact=True)
                self.assertEqual(expected._fields, ("images", "labels", "valid_mask"))
                for i in range(20):
                    self.assertEqual(images.get("train", i)[1], images.observed_labels[i])
                    self.assertEqual(images.label_view("original").get("train", i)[1], i % 3)
                alternate = publish_noise(Path(tmp) / "scores", Path(tmp) / "alternate", seed=8)
                other_config = resolve_config(PRESET, mode="train", data_dir=str(Path(tmp) / "data"), noise_manifest=str(alternate))
                other, _ = open_training_images(other_config.data, ROOT)
                try:
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        ImageBatchSource(other, 4, 8, 42).restore(snap)
                finally:
                    other.close()
                task = ClassificationTask(lambda p, x, s: (jnp.tile(p["logits"], (len(x), 1)), s), 100)
                logits = jnp.linspace(-1, 1, 100)
                spec = noisy_classification_evaluation(images, task, 4, 8, "fixture", include_all_original=True)
                rows = spec.evaluate_groups(TrainState({"logits": logits}, {}))
                self.assertEqual([r["group"] for r in rows], ["clean", "corrupted_noisy", "corrupted_original", "all_original"])
                self.assertEqual(rows[-1]["sample_count"], 20)
                loss = -np.array(jax.nn.log_softmax(logits))[images.original_labels].mean()
                self.assertAlmostEqual(rows[-1]["ce"], float(loss), delta=1e-6)
                self.assertEqual(spec.evaluate(TrainState({"logits": logits}, {}))["eval_sample_count"], 3)
            finally:
                images.close()
            for kw in ({"noise_seed": 7}, {"label_noise_percent": 20}, {"theorem_probes": True}, {"data_mode": "clean"}):
                with self.subTest(kw=kw), self.assertRaises(ValueError):
                    resolve_config(PRESET, mode="train", noise_manifest=str(manifest), **kw)

    def test_empty_partition_endpoints_still_produce_full_original_metrics(self):
        with TemporaryDirectory() as tmp:
            _, scores, _ = artifact(tmp)
            for percent in (0, 100):
                manifest = publish_noise(scores, Path(tmp) / f"endpoint{percent}", percent=percent)
                config = resolve_config(PRESET, mode="train", data_dir=str(Path(tmp) / "data"), noise_manifest=str(manifest))
                images, _ = open_training_images(config.data, ROOT)
                try:
                    task = ClassificationTask(lambda p, x, s: (jnp.zeros((len(x), 100)), s), 100)
                    spec = noisy_classification_evaluation(images, task, 4, 8, "fixture", include_all_original=True)
                    rows = spec.evaluate_groups(TrainState({}, {}))
                    self.assertEqual(rows[-1]["sample_count"], 20)
                    self.assertTrue(np.isfinite(rows[-1]["ce"]))
                    empty = rows[2] if percent == 0 else rows[0]
                    self.assertEqual(empty["sample_count"], 0)
                    self.assertIsNone(empty["ce"])
                finally:
                    images.close()

    def test_cli_cached_generation_audit_verify_and_suite_passthrough(self):
        with TemporaryDirectory() as tmp:
            db, scores, _ = artifact(tmp)
            output = Path(tmp) / "cli_noise"
            base = [sys.executable, "scripts/prepare_imagenet_clip_noise.py", "--output-dir", str(output)]
            subprocess.check_output([*base, "--stage", "generate", "--scores-dir", str(scores), "--label-noise-percent", "40", "--noise-seed", "3"], cwd=ROOT)
            subprocess.check_output([*base, "--stage", "verify"], cwd=ROOT)
            subprocess.check_output([*base, "--stage", "audit", "--image-db", str(db), "--audit-count", "2"], cwd=ROOT)
            manifest = output / "noise_manifest.json"
            plans = json.loads(subprocess.check_output([sys.executable, "scripts/run_suite.py", "--config", str(PRESET),
                "--dry-run", "--noise-manifest", str(manifest), "--methods", "adam", "dpco_filter", "dpco_flip"], cwd=ROOT))
            for plan in plans:
                self.assertEqual(plan["noise_manifest"], str(manifest))
                self.assertEqual(plan["noise_type"], "clip_idn")
                self.assertIn("--noise-manifest", plan["command"])
                self.assertIn("clip_idn", plan["stdout"])
            clean = load_config(PRESET)
            self.assertNotIn("noise_manifest", clean.as_dict()["data"])

    def test_actual_three_method_training_entry_and_checkpoint_resume(self):
        from scripts.train_classification import run
        with TemporaryDirectory() as tmp:
            _, _, manifest = artifact(tmp)
            for method in ("adam", "dpco_filter", "dpco_flip"):
                config = resolve_config(PRESET, mode="train", steps=2, batch_size=2,
                    method=method, data_dir=str(Path(tmp) / "data"), noise_manifest=str(manifest),
                    eval_every_steps=1, checkpoint_every_steps=1)
                config = replace(config, data=replace(config.data, image_size=8),
                    results=ResultsConfig(str(Path(tmp) / "results")))
                args = SimpleNamespace(mode="train", run_name=method, resume=False, reserved=False, resume_slot="latest")
                with patch("src.models.classification.create_model", return_value=ResNet((1,), 100, num_filters=2)), redirect_stdout(StringIO()):
                    run(args, config, jax.local_devices(), status=None)
                    store = CheckpointStore(Path(tmp) / "results/runs" / method)
                    final = store.load("final")["train_state"]
                    self.assertIn("best", store.index)
                    args.resume, args.resume_slot = True, "p50"
                    run(args, config, jax.local_devices(), status=None)
                    assert_tree(self, final, CheckpointStore(store.directory).load("final")["train_state"], exact=True)
                groups = (Path(tmp) / "results/files" / method / "group_evaluation.csv").read_text()
                self.assertIn("all_original", groups)
                self.assertIn("corrupted_original", groups)
                metadata = json.loads((Path(tmp) / "results/logs" / method / "metadata.json").read_bytes())
                self.assertEqual(metadata["experiment"]["data_plan"]["selection"]["noise_seed"], 7)


if __name__ == "__main__":
    unittest.main()
