"""CIFAR-100N provenance, labels, native transforms, and shared training checks."""

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from src.config import ResultsConfig, EvaluationConfig, load_config
from src.data.cifar100n import CIFAR100NImages, MEAN, STD, synthetic_labels, transform
from src.data.cifar100n_prepare import (ASSETS, REVISION, align_official_labels, download_asset,
    canonical, file_digest, publish_arrays, verify_prepared)
from src.data.classification import open_training_images
from src.data.images import ImageBatchSource, evaluation_batches
from src.models.cifar_resnet import CifarResNet, create_cifar_resnet
from src.models.resnet import initialize, model_adapter
from src.tasks.classification import ClassificationTask
from src.training import TrainState, Trainer
from src.training.checkpoints import CheckpointStore
from src.training.evaluation import cifar100n_evaluation
from src.training.launch import add_overrides, cifar_overrides, dpco_overrides, resolve_config
from src.training.experiment import ExperimentRunner
from src.training.results import make_run_name
from tests.test_observation import ROOT, read_rows
from tests.test_dpco_history import assert_tree

PRESET = ROOT / "configs/tpu/cifar100n/dpco_adam.toml"


def fixture(root, *, same=False):
    pixels = np.random.default_rng(17).integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    clean = np.array([0, 1, 0, 1, 0, 1, 0, 1], np.int32)
    human = clean.copy()
    if not same:
        human[[1, 2, 6]] = 1 - human[[1, 2, 6]]
    return publish_arrays(root, pixels, clean, human, pixels[:3], clean[:3],
        {"dataset": "synthetic_fixture_not_official", "official_revision": "fixture"})


def config(root, *, method="dpco_flip", noise_type="noisy100", seed=42):
    c = resolve_config(PRESET, mode="smoke", method=method, steps=2, batch_size=2,
        data_dir=str(Path(root) / "data"), noise_type=noise_type,
        noise_percent=37.5 if noise_type == "noisy100" else 0., seed=seed,
        eval_every_steps=1)
    return replace(c, results=ResultsConfig(str(Path(root) / "results")))


class CIFARPreparationTests(unittest.TestCase):
    def test_official_inputs_are_pinned_to_a_commit_and_digest(self):
        self.assertEqual(len(REVISION), 40)
        for name, (url, algorithm, digest) in ASSETS.items():
            self.assertTrue(url.startswith("https://"))
            if name.endswith(".npy"):
                self.assertIn(REVISION, url)
                self.assertEqual(algorithm, "sha256")
                self.assertEqual(len(digest), 64)

    def test_official_tfds_to_python_permutation_and_label_validation(self):
        clean = np.array([0, 1, 2, 3])
        human = np.array([1, 1, 0, 3])
        order = np.array([2, 0, 3, 1])
        ordered = {"clean_label": clean[order], "noise_label": human[order]}
        np.testing.assert_array_equal(align_official_labels(clean, ordered, order), human)
        for bad in (np.array([0, 0, 2, 3]), np.array([0, 1, 2]), np.array([0., 1., 2., 3.])):
            with self.assertRaises(ValueError):
                align_official_labels(clean, ordered, bad)
        with self.assertRaisesRegex(ValueError, "do not match"):
            align_official_labels(clean[::-1], ordered, order)
        bad = {**ordered, "noise_label": np.array([100, 0, 1, 2])}
        with self.assertRaises(ValueError):
            align_official_labels(clean, bad, order)

    def test_download_publishes_only_verified_bytes_and_reuses_valid_sources(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.npy"
            content = b"official-source-fixture"
            digest = hashlib.sha256(content).hexdigest()
            with patch("urllib.request.urlopen", return_value=BytesIO(content)) as request:
                download_asset("https://fixture/source.npy", path, "sha256", digest)
                self.assertEqual(path.read_bytes(), content)
                request.assert_called_once()
            with patch("urllib.request.urlopen", side_effect=AssertionError("should reuse")):
                download_asset("https://fixture/source.npy", path, "sha256", digest)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                download_asset("https://fixture/source.npy", path, "sha256", digest)
            self.assertEqual(path.read_bytes(), b"tampered")

    def test_bad_download_is_not_published(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "source"
            with patch("urllib.request.urlopen", return_value=BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    download_asset("https://fixture", path, "sha256", "0"*64)
            self.assertFalse(path.exists())
            self.assertTrue(path.with_suffix(".partial").exists())

    def test_prepared_cache_detects_tampering_and_never_overwrites(self):
        with TemporaryDirectory() as tmp:
            fixture(tmp)
            manifest, digest = verify_prepared(tmp)
            self.assertEqual(manifest["human_noise_percent"], 37.5)
            with self.assertRaises(FileExistsError):
                fixture(tmp)
            path = Path(tmp) / "prepared/labels.npz"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                CIFAR100NImages(tmp)

    def test_entire_preparation_download_order_conversion_and_cache_reuse(self):
        from src.data.cifar100n_prepare import prepare
        # Small balanced fixture drives the exact archive/NPY preparation path.
        clean = np.tile(np.arange(100), 2).astype(np.int32)
        human = clean.copy(); human[::2] = (human[::2] + 1) % 100
        order = np.random.default_rng(5).permutation(200)
        archive_bytes = BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            for split, labels in (("train", clean), ("test", np.arange(100))):
                pixels = np.tile(np.concatenate((np.full(1024, 10), np.full(1024, 20), np.full(1024, 30))),
                                 (len(labels), 1)).astype(np.uint8)
                content = pickle.dumps({"data": pixels, "fine_labels": labels.tolist()})
                info = tarfile.TarInfo("cifar-100-python/" + split); info.size = len(content)
                archive.addfile(info, BytesIO(content))
            # An unrelated archive member must never be extracted.
            info = tarfile.TarInfo("../../outside"); info.size = 1
            archive.addfile(info, BytesIO(b"x"))
        annotations = BytesIO(); permutation = BytesIO()
        np.save(annotations, {"clean_label": clean[order].tolist(), "noise_label": human[order].tolist()})
        np.save(permutation, order, allow_pickle=False)
        payloads = {"cifar-100-python.tar.gz": archive_bytes.getvalue(),
                    "CIFAR-100_human_ordered.npy": annotations.getvalue(),
                    "image_order_c100.npy": permutation.getvalue()}
        assets = {name: ("https://fixture/" + name, "sha256", hashlib.sha256(data).hexdigest())
                  for name, data in payloads.items()}
        with TemporaryDirectory() as tmp, patch("src.data.cifar100n_prepare.ASSETS", assets), \
             patch("src.data.cifar100n_prepare.TRAIN_COUNT", 200), \
             patch("src.data.cifar100n_prepare.TEST_COUNT", 100), \
             patch("src.data.cifar100n_prepare.REVISION", "fixture"):
            with patch("urllib.request.urlopen", side_effect=lambda url, **kw: BytesIO(payloads[url.rsplit('/', 1)[1]])) as request:
                first = prepare(tmp, download=True)
                self.assertEqual(request.call_count, 3)
            with patch("urllib.request.urlopen", side_effect=AssertionError("cache must not download")):
                self.assertEqual(first, prepare(tmp, download=True))
            images = CIFAR100NImages(tmp, noise_percent=50.)
            try:
                np.testing.assert_array_equal(images.original_labels, clean)
                np.testing.assert_array_equal(images.human_labels, human)
                np.testing.assert_array_equal(np.asarray(images.get("train", 0)[0])[0, 0], [10, 20, 30])
                self.assertEqual(images.identity["official_revision"], "fixture")
            finally:
                images.close()

    def test_prepare_plan_imports_no_accelerator_or_torch(self):
        code = '''import builtins, runpy, sys
+original = builtins.__import__
+def guarded(name, *a, **kw):
+    if name.split('.')[0] in ('jax', 'torch', 'tensorflow', 'flax'):
+        raise AssertionError('runtime import: ' + name)
+    return original(name, *a, **kw)
+builtins.__import__ = guarded
+sys.argv = ['scripts/prepare_cifar100n.py', '--stage', 'plan']
+runpy.run_path(sys.argv[0], run_name='__main__')
+'''.replace('\n+', '\n')
        result = json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True))
        self.assertEqual(result["official_revision"], REVISION)


class CIFARDataTests(unittest.TestCase):
    def test_human_clean_and_synthetic_modes_and_fixed_control_partition(self):
        with TemporaryDirectory() as tmp:
            fixture(tmp)
            human = CIFAR100NImages(tmp, noise_percent=37.5)
            clean = CIFAR100NImages(tmp, noise_type="clean100", noise_percent=0.)
            synthetic = CIFAR100NImages(tmp, is_human=False)
            try:
                self.assertEqual(human.actual_noise_percent, 37.5)
                self.assertEqual(human.actual_noise_rate, .375)
                self.assertFalse(clean.noise_or_not.any())
                self.assertEqual(clean.actual_noise_percent, 0.)
                np.testing.assert_array_equal(human.corrupted, clean.corrupted)
                np.testing.assert_array_equal(clean.observed_labels, clean.original_labels)
                np.testing.assert_array_equal(human.observed_labels, human.human_labels)
                expected, _ = synthetic_labels(human.original_labels, human.human_labels)
                np.testing.assert_array_equal(synthetic.observed_labels, expected)
                self.assertNotEqual(human.fingerprint, clean.fingerprint)
                example = human.example(1)
                self.assertEqual(example["label"], 1)
                self.assertEqual(example["noise_label"], 0)
                self.assertEqual(human.get("train", 1)[1], 0)
                self.assertEqual(human.get("validation", 1)[1], 1)
                with self.assertRaises(ValueError):
                    human.observed_labels[0] = 99
            finally:
                human.close(); clean.close(); synthetic.close()

    def test_expected_percentage_does_not_change_labels_and_smoke_checks_full_rate(self):
        with TemporaryDirectory() as tmp:
            fixture(tmp)
            for percentage in (20., 40., float("nan"), True, -1.):
                with self.assertRaisesRegex(ValueError, "checks the actual rate"):
                    CIFAR100NImages(tmp, noise_percent=percentage)
            limited = CIFAR100NImages(tmp, limit=2, noise_percent=37.5)
            try:
                self.assertEqual(limited.counts, {"train": 2, "validation": 2})
                self.assertEqual(limited.identity["actual_noise_percent"], 37.5)
                self.assertEqual(limited.identity["group_counts"], {"clean": 1, "corrupted": 1})
                with self.assertRaises(IndexError):
                    limited.get("train", 2)
            finally:
                limited.close()

    def test_synthetic_sampling_matches_official_randomstate_multinomial(self):
        clean = np.repeat(np.arange(100), 5)
        human = clean.copy()
        human[::3] = (human[::3] + 1) % 100
        labels, transition = synthetic_labels(clean, human, 0)
        counts = np.zeros((100, 100))
        for a, b in zip(clean, human):
            counts[a, b] += 1
        # Exactly the official expression, valid for balanced CIFAR-100 rows.
        matrix = counts / counts.sum(axis=1)
        rng = np.random.RandomState(0)
        expected = np.array([np.where(rng.multinomial(1, matrix[y], 1)[0] == 1)[0][0] for y in clean])
        np.testing.assert_array_equal(labels, expected)
        np.testing.assert_array_equal(transition, matrix)

    def test_transform_matches_native_eval_and_zero_padded_crop(self):
        pixels = np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3)
        image = Image.fromarray(pixels)
        np.testing.assert_array_equal(transform(image), (pixels.astype(np.float32)/255 - MEAN)/STD)
        rng = np.random.default_rng(4)
        top, left = rng.integers(0, 9, size=2)
        expected = np.pad(pixels, ((4, 4), (4, 4), (0, 0)))[top:top+32, left:left+32]
        if rng.random() < .5:
            expected = expected[:, ::-1]
        np.testing.assert_array_equal(transform(image, rng=np.random.default_rng(4)),
                                      (expected.astype(np.float32)/255 - MEAN)/STD)
        with self.assertRaises(ValueError):
            transform(image, 224)

    def test_source_resume_and_batch_interface_exclude_audit_truth(self):
        with TemporaryDirectory() as tmp:
            fixture(tmp)
            images = CIFAR100NImages(tmp)
            try:
                source = ImageBatchSource(images, 3, 32, 7)
                next(source)
                cursor = source.snapshot()
                expected = next(source)
                source.restore(cursor)
                actual = next(source)
                assert_tree(self, expected, actual, exact=True)
                self.assertEqual(expected._fields, ("images", "labels", "valid_mask"))
                self.assertEqual(expected.images.shape, (3, 32, 32, 3))
                other = CIFAR100NImages(tmp, noise_type="clean100")
                try:
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        ImageBatchSource(other, 3, 32, 7).restore(cursor)
                finally:
                    other.close()
                tail = list(evaluation_batches(images, 2, 32))[-1]
                np.testing.assert_array_equal(tail.valid_mask, [True, False])
            finally:
                images.close()


class CIFAREvaluationTests(unittest.TestCase):
    def test_fixed_group_ce_matches_independent_numpy_and_does_not_commit_state(self):
        with TemporaryDirectory() as tmp:
            fixture(tmp)
            images = CIFAR100NImages(tmp)
            try:
                def model(params, pixels, state):
                    logits = jnp.tile(params["logits"], (pixels.shape[0], 1))
                    return logits, {"counter": state["counter"] + 1}
                task = ClassificationTask(model, 100)
                state = TrainState({"logits": jnp.linspace(-1, 1, 100)}, {"counter": jnp.array(0)})
                evaluation = cifar100n_evaluation(images, task, 3)
                rows = evaluation.evaluate_groups(state)
                log_probs = np.asarray(jax.nn.log_softmax(state.params["logits"]))
                self.assertEqual([r["sample_count"] for r in rows], [5, 3, 3])
                for row, group, labels in zip(rows, ("clean", "corrupted", "corrupted"),
                                             (images.original_labels, images.reference_labels, images.original_labels)):
                    expected = -log_probs[labels[images.group_indices[group]]].mean()
                    self.assertAlmostEqual(row["ce"], float(expected), places=6)
                self.assertEqual(int(state.model_state["counter"]), 0)
                self.assertEqual(evaluation.evaluate(state)["eval_sample_count"], 3)
                self.assertFalse(evaluation.select_best)
            finally:
                images.close()

    def test_empty_corrupted_group_records_missing_not_nan(self):
        with TemporaryDirectory() as tmp:
            fixture(tmp, same=True)
            images = CIFAR100NImages(tmp)
            try:
                task = ClassificationTask(lambda p, x, s: (jnp.zeros((len(x), 100)), s), 100)
                rows = cifar100n_evaluation(images, task, 3).evaluate_groups(TrainState({}))
                self.assertEqual(rows[0]["sample_count"], 8)
                self.assertIsNone(rows[1]["ce"])
                self.assertEqual(rows[1]["sample_count"], 0)
                self.assertIsNone(rows[2]["ce"])
            finally:
                images.close()


class CIFARIntegrationTests(unittest.TestCase):
    def test_all_three_real_training_entry_groups_and_exact_checkpoint_resume(self):
        from scripts.train_classification import run
        with TemporaryDirectory() as tmp:
            fixture(Path(tmp) / "data")
            for method in ("adam", "dpco_filter", "dpco_flip"):
                c = config(tmp, method=method)
                args = SimpleNamespace(mode="smoke", run_name=method, resume=False,
                                       reserved=False, resume_slot="latest")
                # Real loader, shared entry/mesh/Adam/ref; only shrink the backbone.
                with patch("src.models.cifar_resnet.create_cifar_resnet",
                           return_value=CifarResNet((1,), 100, num_filters=2)), redirect_stdout(StringIO()):
                    run(args, c, jax.local_devices(), status=None)
                    store = CheckpointStore(Path(tmp)/"results/runs"/method)
                    final = store.load("final")["train_state"]
                    self.assertNotIn("best", store.index)
                    args.resume, args.resume_slot = True, "p50"
                    run(args, c, jax.local_devices(), status=None)
                    final2 = CheckpointStore(store.directory).load("final")["train_state"]
                assert_tree(self, final, final2, exact=True)
                rows = read_rows(Path(tmp)/"results/files"/method/"group_evaluation.csv")
                self.assertEqual(len(rows), 9)
                self.assertEqual([int(r["step"]) for r in rows], [0]*3 + [1]*3 + [2]*3)
                self.assertEqual(int(final.optimizer_state.count), 2)
                if method != "adam":
                    self.assertEqual(int(final.method_state.updates), 2)
                metadata = json.loads((Path(tmp)/"results/logs"/method/"metadata.json").read_bytes())
                self.assertFalse(metadata["evaluation_selects_best"])
                self.assertEqual(metadata["experiment"]["data_plan"]["actual_noise_percent"], 37.5)

    def test_models_have_native_stem_and_distinct_depths(self):
        for name, stages in (("cifar_resnet18", (2, 2, 2, 2)), ("cifar_resnet34", (3, 4, 6, 3))):
            model = create_cifar_resnet(name)
            self.assertEqual(model.stage_sizes, stages)
            shapes = jax.eval_shape(lambda key: model.init(key, jnp.zeros((1, 32, 32, 3)), train=True), jax.random.key(0))
            self.assertEqual(shapes["params"]["conv_init"]["kernel"].shape, (3, 3, 3, 64))
            self.assertEqual(shapes["params"]["head"]["bias"].shape, (100,))

    def test_config_labels_percentage_and_dataset_switching(self):
        c = resolve_config(PRESET, mode="train")
        self.assertEqual(c.data.noise_percent, 40.2)
        clean = resolve_config(PRESET, mode="train", noise_type="clean100")
        self.assertEqual((clean.data.mode, clean.data.noise_percent), ("clean", 0.))
        synthetic = resolve_config(PRESET, mode="train", is_human=False)
        self.assertFalse(synthetic.data.is_human)
        self.assertIsNone(synthetic.data.noise_percent)
        old = ROOT / "configs/tpu/dpco/imagenet100_resnet50.toml"
        switched = resolve_config(old, mode="train", dataset="cifar100n")
        self.assertEqual(switched.data.image_size, 32)
        self.assertEqual(switched.model.name, "cifar_resnet34")
        restored = resolve_config(PRESET, mode="train", dataset="imagenet100")
        self.assertEqual(restored.data.image_size, 224)
        self.assertIsNone(restored.data.noise_type)
        self.assertNotEqual(make_run_name(c), make_run_name(clean))
        self.assertNotEqual(make_run_name(c), make_run_name(synthetic))
        for kwargs in ({"noise_percent": 101.}, {"noise_percent": float("nan")}, {"noise_seed": 1},
                       {"model": "resnet50"}, {"data_mode": "clean"}, {"theorem_probes": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_config(PRESET, mode="train", **kwargs)

    def test_shell_dry_run_parent_child_options_and_no_accelerator_import(self):
        env = {**os.environ, "DPCO_PYTHON": sys.executable}
        output = subprocess.check_output([str(ROOT/"run_cifar100n_suite.sh"), "--dry-run", "--noise-type", "clean100"],
                                         cwd="/private/tmp", env=env, text=True)
        plans = json.loads(output)
        self.assertEqual(len(plans), 3)
        self.assertEqual([p["negative_weight"] for p in plans], [None, 0., -1.])
        import argparse
        parser = argparse.ArgumentParser()
        add_overrides(parser)
        for plan in plans:
            self.assertEqual(plan["noise_type"], "clean100")
            command = plan["command"]
            args = parser.parse_args(command[command.index("--seed"):])
            child = resolve_config(PRESET, mode="train", method=args.method, seed=args.seed,
                                   **cifar_overrides(args), **dpco_overrides(args))
            self.assertEqual(child.data.noise_percent, 0.)
            self.assertEqual(child.training.optimizer, "adam")
        code = '''import builtins, runpy, sys
+original=builtins.__import__
+def guarded(name,*a,**kw):
+    if name.split('.')[0] in ('jax','flax','torch','tensorflow','numpy'):
+        raise AssertionError(name)
+    return original(name,*a,**kw)
+builtins.__import__=guarded
+sys.argv=['scripts/launch_tpu.py','--config','configs/tpu/cifar100n/dpco_adam.toml','--dry-run']
+runpy.run_path(sys.argv[0],run_name='__main__')
+'''.replace('\n+','\n')
        json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True))


if __name__ == "__main__":
    unittest.main()
