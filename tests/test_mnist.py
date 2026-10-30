"""MNIST integrity/corruption and real entry-point training/recovery checks."""

from contextlib import redirect_stdout
from dataclasses import replace
import gzip
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from src.config import ResultsConfig, load_config
from src.data.images import ImageBatchSource, evaluation_batches
from src.data.mnist import MNISTImages, corrupt_labels
from src.data.mnist_prepare import ASSETS, prepare, publish_arrays, read_idx, verify_prepared
from src.models.classification import create_model
from src.models.resnet import initialize, model_adapter
from src.tasks.classification import ClassificationTask
from src.training import TrainState
from src.training.checkpoints import CheckpointStore
from src.training.evaluation import noisy_classification_evaluation
from src.training.launch import add_overrides, cifar_overrides, dpco_overrides, resolve_config
from src.training.results import make_run_name
from tests.test_dpco_history import assert_tree
from tests.test_observation import ROOT, read_rows

PRESET = ROOT / "configs/tpu/mnist/dpco_adam.toml"
CIFAR_PRESET = ROOT / "configs/tpu/cifar100n/dpco_adam.toml"


def fixture(root):
    rng = np.random.default_rng(19)
    return publish_arrays(root, rng.integers(0, 256, (20, 28, 28), dtype=np.uint8),
        np.arange(20, dtype=np.int32) % 10,
        rng.integers(0, 256, (3, 28, 28), dtype=np.uint8), np.array([0, 1, 2], np.int32),
        {"dataset": "synthetic_fixture_not_official"})


class MNISTDataTests(unittest.TestCase):
    def test_idx_bounds_headers_and_labels(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "idx.gz"
            valid = struct.pack(">II", 2049, 3) + bytes([0, 4, 9])
            path.write_bytes(gzip.compress(valid))
            np.testing.assert_array_equal(read_idx(path, 3, images=False), [0, 4, 9])
            for invalid in (valid[:-1], valid + b"x", valid[:8] + bytes([0, 4, 10]), b"invalid"):
                path.write_bytes(gzip.compress(invalid))
                with self.assertRaises(ValueError):
                    read_idx(path, 3, images=False)

    def test_full_download_preparation_validates_sources_and_reuses_cache(self):
        assets, content = {}, {}
        for name in ASSETS:
            is_images = "images" in name
            count = 60000 if name.startswith("train") else 10000
            header = struct.pack(">IIII", 2051, count, 28, 28) if is_images else struct.pack(">II", 2049, count)
            payload = bytes(count * (28 * 28 if is_images else 1))
            body = gzip.compress(header + payload)
            url = "https://fixture/" + name
            assets[name] = (url, "md5", hashlib.md5(body).hexdigest())
            content[url] = body
        with TemporaryDirectory() as tmp, patch("src.data.mnist_prepare.ASSETS", assets):
            with patch("urllib.request.urlopen", side_effect=lambda url, **kw: BytesIO(content[url])) as request:
                manifest, digest = prepare(tmp, download=True)
                self.assertEqual(request.call_count, 4)
            self.assertEqual(manifest["counts"], {"train": 60000, "validation": 10000})
            with patch("urllib.request.urlopen", side_effect=AssertionError("must reuse cache")):
                self.assertEqual(prepare(tmp, download=True)[1], digest)
            reader = MNISTImages(tmp, label_noise_percent=20.)
            self.assertEqual(reader.identity["active_mismatch_count"], 12000)
            self.assertEqual(reader.identity["actual_noise_percent"], 20.)
            reader.close()

    def test_fixed_exact_count_corruption_determinism_and_no_original_reselection(self):
        clean = np.arange(100, dtype=np.int32) % 10
        for percent in (0., 20., 40.2, 100.):
            actual = corrupt_labels(clean, percent, 5)
            self.assertEqual(int(np.count_nonzero(actual != clean)), int(np.floor(percent + .5)))
            np.testing.assert_array_equal(actual, corrupt_labels(clean, percent, 5))
        self.assertFalse(np.array_equal(corrupt_labels(clean, 20, 5), corrupt_labels(clean, 20, 6)))
        for percent in (-1, 101, float("nan"), True):
            with self.assertRaises(ValueError):
                corrupt_labels(clean, percent, 0)

    def test_reader_audit_truth_is_not_in_batches_and_gray_padding_restores(self):
        with TemporaryDirectory() as tmp:
            fixture(tmp)
            images = MNISTImages(tmp, noise_percent=20.)
            try:
                source = ImageBatchSource(images, 4, 28, 42)
                next(source)
                snapshot = source.snapshot()
                expected = next(source)
                source.restore(snapshot)
                actual = next(source)
                assert_tree(self, actual, expected, exact=True)
                self.assertEqual(actual._fields, ("images", "labels", "valid_mask"))
                self.assertEqual(actual.images.shape, (4, 28, 28, 1))
                tail = list(evaluation_batches(images, 2, 28))[-1]
                self.assertEqual(tail.images.shape, (2, 28, 28, 1))
                np.testing.assert_array_equal(tail.valid_mask, [True, False])
                for i in range(20):
                    np.testing.assert_array_equal(images.transform(images.get("train", i)[0]),
                        images.transform(images.get("train", i)[0], rng=np.random.default_rng(i)))
                other = MNISTImages(tmp, label_noise_percent=0.)
                try:
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        ImageBatchSource(other, 4, 28, 42).restore(snapshot)
                finally:
                    other.close()
                with self.assertRaisesRegex(ValueError, "checks actual rate"):
                    MNISTImages(tmp, noise_percent=19.)
            finally:
                images.close()
            path = Path(tmp) / "prepared/labels.npz"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_prepared(tmp)
            with self.assertRaises(FileExistsError):
                fixture(tmp)

    def test_group_losses_match_numpy_and_empty_groups_are_explicit(self):
        with TemporaryDirectory() as tmp:
            fixture(tmp)
            for percent in (0., 20., 100.):
                images = MNISTImages(tmp, label_noise_percent=percent)
                try:
                    logits = jnp.linspace(-1, 1, 10)
                    task = ClassificationTask(lambda p, x, s: (jnp.tile(p["logits"], (len(x), 1)), s), 10)
                    state = TrainState({"logits": logits}, {})
                    spec = noisy_classification_evaluation(images, task, 4, 28, "test")
                    self.assertEqual(spec.evaluate(state)["eval_sample_count"], 3)
                    for row, partition, labels in zip(spec.evaluate_groups(state),
                            ("clean", "corrupted", "corrupted"),
                            (images.original_labels, images.observed_labels, images.original_labels)):
                        ids = images.group_indices[partition]
                        if len(ids):
                            expected = -np.asarray(jax.nn.log_softmax(logits))[labels[ids]].mean()
                            self.assertAlmostEqual(row["ce"], float(expected), places=6)
                        else:
                            self.assertIsNone(row["ce"])
                finally:
                    images.close()


class MNISTIntegrationTests(unittest.TestCase):
    def test_three_groups_real_entry_and_exact_checkpoint_resume(self):
        from scripts.train_classification import run
        with TemporaryDirectory() as tmp:
            fixture(Path(tmp) / "data")
            for method in ("adam", "dpco_filter", "dpco_flip"):
                c = resolve_config(PRESET, mode="smoke", steps=2, batch_size=2, method=method,
                    data_dir=str(Path(tmp) / "data"), eval_every_steps=1, checkpoint_every_steps=1)
                c = replace(c, results=ResultsConfig(str(Path(tmp) / "results")))
                args = SimpleNamespace(mode="train", run_name=method, resume=False,
                                       reserved=False, resume_slot="latest")
                with redirect_stdout(StringIO()):
                    run(args, c, jax.local_devices(), status=None)
                    store = CheckpointStore(Path(tmp) / "results/runs" / method)
                    final = store.load("final")["train_state"]
                    args.resume, args.resume_slot = True, "p50"
                    run(args, c, jax.local_devices(), status=None)
                    resumed = CheckpointStore(store.directory).load("final")["train_state"]
                assert_tree(self, final, resumed, exact=True)
                self.assertEqual(final.model_state, {})
                self.assertEqual(int(final.optimizer_state.count), 2)
                if method != "adam":
                    self.assertEqual(int(final.method_state.updates), 2)
                files = Path(tmp) / "results/files" / method
                self.assertEqual(len(read_rows(files / "train_steps.csv")), 2)
                self.assertEqual(len(read_rows(files / "group_evaluation.csv")), 9)
                metadata = json.loads((Path(tmp) / "results/logs" / method / "metadata.json").read_text())
                self.assertEqual(metadata["experiment"]["input_channels"], 1)
                self.assertEqual(metadata["experiment"]["data_plan"]["actual_noise_percent"], 20.)
                self.assertNotIn("best", CheckpointStore(store.directory).index)

    def test_models_and_cli_parent_child_passthrough(self):
        for name in ("mnist_cnn", "mnist_mlp"):
            model = create_model(name, 10)
            params, state = initialize(model, 42, 28, channels=1)
            task = ClassificationTask(model_adapter(model, train=True), 10)
            logits, state2 = task.model_fn(params, jnp.ones((2, 28, 28, 1)), state)
            self.assertEqual(logits.shape, (2, 10))
            self.assertEqual(state2, {})
        output = subprocess.check_output([sys.executable, "scripts/run_suite.py", "--config", str(PRESET),
            "--dry-run", "--model", "mnist_mlp", "--label-noise-percent", "40", "--noise-seed", "7",
            "--methods", "adam", "dpco_filter", "dpco_flip"], cwd=ROOT, text=True)
        plans = json.loads(output)
        # Verify generated options are accepted/resolved by the child's shared parser.
        import argparse
        parser = argparse.ArgumentParser()
        add_overrides(parser)
        for p in plans:
            command = p["command"]
            self.assertEqual(p["model"], "mnist_mlp")
            self.assertEqual(p["label_noise_percent"], 40.)
            self.assertIn("symmetric40_noise_s7", p["stdout"])
            self.assertEqual(command[command.index("--label-noise-percent") + 1], "40.0")
            self.assertEqual(command[command.index("--noise-seed") + 1], "7")
            args = parser.parse_args(command[command.index("--seed"):])
            child = resolve_config(PRESET, mode="train", seed=args.seed, method=args.method, model=args.model,
                                   **cifar_overrides(args), **dpco_overrides(args))
            self.assertEqual(child.model.name, p["model"])
            self.assertEqual(child.data.label_noise_percent, p["label_noise_percent"])
            self.assertEqual(child.reference.name, p["reference"])

    def test_switching_datasets_and_noise_semantics(self):
        c = load_config(PRESET)
        self.assertNotEqual(make_run_name(c), make_run_name(resolve_config(PRESET, mode="train", label_noise_percent=0)))
        clean = resolve_config(PRESET, mode="train", data_mode="clean")
        self.assertEqual((clean.data.mode, clean.data.label_noise_percent), ("clean", 0.))
        switched = resolve_config(CIFAR_PRESET, mode="train", dataset="mnist")
        self.assertEqual((switched.model.name, switched.data.image_size, switched.data.label_noise_percent),
                         ("mnist_cnn", 28, 20.))
        self.assertIsNone(switched.data.noise_percent)
        for dataset in ("cifar100n", "imagenet100"):
            other = resolve_config(PRESET, mode="train", dataset=dataset)
            self.assertEqual(other.data.label_noise_percent, 0.)
            self.assertTrue(other.data.is_human)
        for kw in ({"label_noise_percent": -1}, {"label_noise_percent": float("nan")},
                   {"is_human": True}, {"noise_type": "noisy100"}, {"theorem_probes": True}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                resolve_config(PRESET, mode="train", **kw)
        with self.assertRaises(ValueError):
            resolve_config(CIFAR_PRESET, mode="train", label_noise_percent=20)


if __name__ == "__main__":
    unittest.main()
