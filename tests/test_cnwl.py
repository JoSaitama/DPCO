"""CNWL split fidelity, natural-noise auditing and shared training integration.

All small datasets below are synthetic fixtures, never research results.
"""

import argparse
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from io import BytesIO, StringIO
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from PIL import Image

from src.config import ResultsConfig, load_config
from src.data import cnwl_prepare as cp
from src.data.cnwl import CNWLImages
from src.data.images import ImageBatchSource, evaluation_batches
from src.training import TrainState
from src.tasks.classification import ClassificationTask
from src.training.evaluation import cnwl_evaluation
from src.training.checkpoints import CheckpointStore
from src.training.launch import add_overrides, cifar_overrides, dpco_overrides, resolve_config
from src.training.results import make_run_name
from tests.test_dpco_history import assert_tree
from tests.test_observation import ROOT, read_rows

PRESET = ROOT / "configs/tpu/cnwl/dpco_adam.toml"


def jpeg(value=50):
    output = BytesIO()
    Image.new("RGB", (15, 21), (value, 80, 120)).save(output, format="JPEG")
    return output.getvalue()


def fixture(manual):
    manual = Path(manual)
    release = manual / "dataset_no_images"
    splits = release / "mini-imagenet/split"
    splits.mkdir(parents=True)
    for folder in ("clean_images", "noisy_images", "validation_all"):
        (manual / folder).mkdir()
    blue, red, validation, annotations = [], [], [], []
    for label in range(100):
        synset = f"n{label:08d}"
        for rank in (1, 2):
            name = f"{synset}{rank:08d}.jpg"
            blue.append(f"{name} {label}\n")
            (manual / "clean_images" / name).write_bytes(jpeg(20 + label))
            if label < 20:
                noise = f"web_{label}_{rank}"
                annotations.append([{"image/id": noise, "image/uri": f"https://fixture.invalid/{noise}.jpg",
                    "image/class/label": label, "image/class/label/is_clean": 0}])
                (manual / "noisy_images" / (noise + ".jpg")).write_bytes(jpeg(180))
                red.append(f"{noise}.jpg {label}\n")
            else:
                red.append(blue[-1])
        name = f"ILSVRC2012_val_{label + 1:08d}.jpg"
        validation.append(f"{name} {label}\n")
        (manual / "validation_all" / name).write_bytes(jpeg(label))
    (release / "mini-imagenet/class_names.txt").write_text("\n".join(f"class{i}" for i in range(100)) + "\n")
    (release / "mini-imagenet-annotations.json").write_text(json.dumps({"data": annotations}))
    (splits / "blue_noise_nl_0.0").write_text("".join(blue))
    (splits / "clean_validation").write_text("".join(validation))
    for percent in cp.LEVELS:
        (splits / f"red_noise_nl_{percent / 100}").write_text("".join(blue if percent == 0 else red))
    return release


def fixture_counts():
    stack = ExitStack()
    stack.enter_context(patch.object(cp, "RED_COUNTS", {p: 200 for p in cp.LEVELS}))
    stack.enter_context(patch.object(cp, "BLUE_COUNT", 200))
    stack.enter_context(patch.object(cp, "VALIDATION_COUNT", 100))
    return stack


class CNWLDataTests(unittest.TestCase):
    def test_release_order_groups_truth_exclusion_and_cache_integrity(self):
        with TemporaryDirectory() as tmp, fixture_counts():
            root, manual = Path(tmp) / "cache", Path(tmp) / "manual"
            fixture(manual)
            manifest, digest = cp.prepare(root, manual_dir=manual)
            self.assertEqual(manifest["counts"], {"train": 200, "validation": 100})
            self.assertEqual(manifest["actual_noise_percent"], 20.)
            with patch.object(cp, "read_release", side_effect=AssertionError("must reuse complete cache")):
                self.assertEqual(cp.prepare(root, manual_dir=manual)[1], digest)
            images = CNWLImages(root, noise_percent=20.)
            try:
                self.assertEqual({k: len(v) for k, v in images.group_indices.items()}, {"clean": 160, "corrupted": 40})
                self.assertFalse(hasattr(images, "original_labels"))
                self.assertEqual(images.get("train", 0)[1], 0)
                self.assertEqual(images.get("train", 0)[0].size, (84, 84))
                source = ImageBatchSource(images, 4, 84, 42)
                next(source)
                snapshot = source.snapshot()
                expected = next(source)
                source.restore(snapshot)
                assert_tree(self, next(source), expected, exact=True)
                self.assertEqual(expected._fields, ("images", "labels", "valid_mask"))
                self.assertEqual(expected.images.shape, (4, 84, 84, 3))
                tail = list(evaluation_batches(images, 128, 84))[-1]
                self.assertEqual(int(tail.valid_mask.sum()), 100)
                with self.assertRaisesRegex(ValueError, "checks actual rate"):
                    CNWLImages(root, noise_percent=40.)
                cp.prepare(root, manual_dir=manual, percent=0)
                other = CNWLImages(root, cnwl_noise_percent=0)
                try:
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        ImageBatchSource(other, 4, 84, 42).restore(snapshot)
                finally:
                    other.close()
                # Fixture files deliberately share membership across nonzero
                # levels: prove the selector and measured-rate assertion differ.
                cp.prepare(root, manual_dir=manual, percent=40)
                selected = CNWLImages(root, cnwl_noise_percent=40, noise_percent=20.)
                try:
                    self.assertEqual(selected.identity["cnwl_noise_percent"], 40)
                    self.assertEqual(selected.actual_noise_percent, 20.)
                finally:
                    selected.close()
            finally:
                images.close()
            database = cp.cache_dir(root, 20) / "images.sqlite"
            with database.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                cp.verify_prepared(root, 20)

    def test_failures_never_drop_samples_and_retry_reuses_completed_rows(self):
        with TemporaryDirectory() as tmp, fixture_counts():
            root, manual = Path(tmp) / "cache", Path(tmp) / "manual"
            fixture(manual)
            missing = manual / "noisy_images/web_0_1.jpg"
            payload = missing.read_bytes()
            missing.unlink()
            report = Path(tmp) / "report.json"
            with self.assertRaisesRegex(RuntimeError, "no samples dropped"):
                cp.prepare(root, manual_dir=manual, report_path=report)
            self.assertEqual(json.loads(report.read_text())["failures"][0]["image_id"], missing.name)
            self.assertFalse(cp.cache_dir(root, 20).exists())
            partial = root / "prepared/.red_20.building/images.sqlite"
            with sqlite3.connect(partial) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 299)
            missing.write_bytes(payload)
            with patch.object(cp, "resize_image", wraps=cp.resize_image) as resize:
                cp.prepare(root, manual_dir=manual)
                self.assertEqual(resize.call_count, 1)

    def test_released_web_uri_download_and_missing_metadata_http_failure(self):
        with TemporaryDirectory() as tmp, fixture_counts():
            root, manual = Path(tmp) / "cache", Path(tmp) / "manual"
            fixture(manual)
            path = manual / "noisy_images/web_0_1.jpg"
            data = path.read_bytes()
            path.unlink()
            with patch("urllib.request.urlopen", return_value=BytesIO(data)) as request:
                cp.prepare(root, manual_dir=manual, download=True)
                self.assertEqual(request.call_args.args[0], "https://fixture.invalid/web_0_1.jpg")
            with patch("urllib.request.urlopen", side_effect=OSError("HTTP 403")), self.assertRaisesRegex(OSError, "403"):
                cp.metadata_directory(Path(tmp) / "no_sources", download=True)

    def test_metadata_zip_and_reject_modified_split_or_label(self):
        with TemporaryDirectory() as tmp, fixture_counts():
            manual = Path(tmp) / "manual"
            release = fixture(manual)
            zip_dir = Path(tmp) / "zip"
            zip_dir.mkdir()
            with zipfile.ZipFile(zip_dir / "dataset_no_images.zip", "w") as archive:
                for path in release.rglob("*"):
                    if path.is_file():
                        archive.write(path, "dataset_no_images/" + str(path.relative_to(release)))
                archive.writestr("../../unsafe", "not extracted")
            extracted = cp.metadata_directory(zip_dir)
            cp.read_release(extracted, 20)
            self.assertFalse((Path(tmp) / "unsafe").exists())
            # An interrupted metadata extraction must be repairable from zip.
            (extracted / "mini-imagenet/split/red_noise_nl_0.2").unlink()
            cp.read_release(cp.metadata_directory(zip_dir), 20)
            split = release / "mini-imagenet/split/red_noise_nl_0.2"
            original = split.read_text()
            split.write_text(original.replace("web_0_1.jpg 0", "web_0_1.jpg 1"))
            with self.assertRaisesRegex(ValueError, "annotation"):
                cp.read_release(release, 20)
            split.write_text(original + original.splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                cp.read_release(release, 20)
            split.write_text("../escape.jpg 0\n")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                cp.read_split(split)

    def test_numeric_rank_mapping_matches_official_builder_not_lexical_or_suffix(self):
        with TemporaryDirectory() as tmp:
            body = BytesIO()
            with tarfile.open(fileobj=body, mode="w") as inner:
                for number in (10, 2, 1):
                    payload = str(number).encode()
                    entry = tarfile.TarInfo(f"n00000001_{number}.JPEG")
                    entry.size = len(payload)
                    inner.addfile(entry, BytesIO(payload))
            path = Path(tmp) / "train.tar"
            with tarfile.open(path, "w") as outer:
                entry = tarfile.TarInfo("n00000001.tar")
                entry.size = len(body.getvalue())
                outer.addfile(entry, BytesIO(body.getvalue()))
            self.assertEqual(dict(cp.train_archive_images(path, {"n0000000100000002.jpg"})),
                             {"n0000000100000002.jpg": b"2"})
            with self.assertRaisesRegex(ValueError, "ordinal"):
                list(cp.train_archive_images(path, {"n0000000100000004.jpg"}))

    def test_full_raw_archive_preparation_has_identical_decoded_images_and_labels(self):
        with TemporaryDirectory() as tmp, fixture_counts():
            manual = Path(tmp) / "manual"
            fixture(manual)
            flat_manifest, _ = cp.prepare(Path(tmp) / "flat", manual_dir=manual)
            with tarfile.open(manual / "ILSVRC2012_img_train.tar", "w") as outer:
                for label in range(100):
                    body = BytesIO()
                    with tarfile.open(fileobj=body, mode="w") as inner:
                        for rank, number in ((1, 3), (2, 12)):
                            path = manual / "clean_images" / f"n{label:08d}{rank:08d}.jpg"
                            payload = path.read_bytes()
                            entry = tarfile.TarInfo(f"n{label:08d}_{number}.JPEG")
                            entry.size = len(payload)
                            inner.addfile(entry, BytesIO(payload))
                            path.unlink()
                    entry = tarfile.TarInfo(f"n{label:08d}.tar")
                    entry.size = len(body.getvalue())
                    outer.addfile(entry, BytesIO(body.getvalue()))
            with tarfile.open(manual / "ILSVRC2012_img_val.tar", "w") as archive:
                for path in (manual / "validation_all").iterdir():
                    payload = path.read_bytes()
                    entry = tarfile.TarInfo(path.stem + ".JPEG")
                    entry.size = len(payload)
                    archive.addfile(entry, BytesIO(payload))
                    path.unlink()
            raw_manifest, _ = cp.prepare(Path(tmp) / "raw", manual_dir=manual)
            self.assertEqual(raw_manifest["counts"], flat_manifest["counts"])
            flat, raw = CNWLImages(Path(tmp) / "flat"), CNWLImages(Path(tmp) / "raw")
            try:
                for split, count in raw.counts.items():
                    for index in range(count):
                        actual, actual_label = raw.get(split, index)
                        expected, expected_label = flat.get(split, index)
                        self.assertEqual(actual_label, expected_label)
                        np.testing.assert_array_equal(actual, expected)
            finally:
                flat.close()
                raw.close()

    def test_groups_use_only_available_labels_and_match_numpy(self):
        with TemporaryDirectory() as tmp, fixture_counts():
            fixture(Path(tmp) / "manual")
            cp.prepare(tmp, manual_dir=Path(tmp) / "manual")
            images = CNWLImages(tmp)
            try:
                logits = jnp.linspace(-1, 1, 100)
                task = ClassificationTask(lambda p, x, s: (jnp.tile(p["logits"], (len(x), 1)), s), 100)
                spec = cnwl_evaluation(images, task, 32)
                state = TrainState({"logits": logits}, {})
                self.assertEqual(spec.evaluate(state)["eval_sample_count"], 100)
                rows = spec.evaluate_groups(state)
                self.assertEqual([r["group"] for r in rows], ["clean", "corrupted_noisy"])
                self.assertEqual([r["label_source"] for r in rows], ["original", "observed"])
                for row, partition in zip(rows, ("clean", "corrupted")):
                    labels = [images.get("train", i)[1] for i in images.group_indices[partition]]
                    expected = -np.asarray(jax.nn.log_softmax(logits))[labels].mean()
                    np.testing.assert_allclose(row["ce"], float(expected), atol=1e-6, rtol=2e-7)
            finally:
                images.close()


class TinyClassifier(nn.Module):
    classes: int

    @nn.compact
    def __call__(self, images, *, train=False):
        return nn.Dense(self.classes)(images.mean(axis=(1, 2)))


class CNWLIntegrationTests(unittest.TestCase):
    def test_suite_parent_child_and_model_noise_controls(self):
        output = subprocess.check_output([sys.executable, "scripts/run_suite.py", "--config", str(PRESET),
            "--dry-run", "--cnwl-noise-percent", "40", "--model", "cifar_resnet50",
            "--methods", "adam", "dpco_filter", "dpco_flip"], cwd=ROOT, text=True)
        parser = argparse.ArgumentParser()
        add_overrides(parser)
        for plan in json.loads(output):
            self.assertEqual(plan["dataset"], "cnwl")
            self.assertEqual(plan["cnwl_noise_percent"], 40)
            self.assertIn("red40_official_split", plan["stdout"])
            command = plan["command"]
            args = parser.parse_args(command[command.index("--seed"):])
            child = resolve_config(PRESET, mode="train", seed=args.seed, method=args.method, model=args.model,
                                   **cifar_overrides(args), **dpco_overrides(args))
            self.assertEqual(child.data.cnwl_noise_percent, 40)
            self.assertEqual(child.model.name, "cifar_resnet50")
            self.assertEqual(child.method.negative_weight if child.reference.name == "ema" else None, plan["negative_weight"])
        self.assertNotEqual(make_run_name(load_config(PRESET)), make_run_name(resolve_config(PRESET, mode="train", cnwl_noise_percent=40)))
        self.assertEqual(resolve_config(PRESET, mode="train", cnwl_noise_percent=0).data.mode, "clean")
        for dataset in ("mnist", "cifar100n", "imagenet100", "imagenet1k"):
            switched = resolve_config(PRESET, mode="train", dataset=dataset)
            self.assertNotIn("cnwl_noise_percent", switched.as_dict()["data"])
        switched = resolve_config(ROOT / "configs/tpu/mnist/dpco_adam.toml", mode="train", dataset="cnwl")
        self.assertEqual((switched.data.image_size, switched.data.noise_type, switched.model.name), (84, "red", "resnet50"))
        for kwargs in ({"cnwl_noise_percent": 25}, {"noise_seed": 1}, {"is_human": False},
                       {"label_noise_percent": 20}, {"noise_type": "symmetric"}, {"theorem_probes": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_config(PRESET, mode="train", **kwargs)

    def test_shared_training_csv_checkpoint_and_exact_resume_for_three_methods(self):
        from scripts.train_classification import run
        # Test the real runner/updaters/EMA/checkpoints on CPU with a small model;
        # accelerator performance and the ResNet training recipe are not tested.
        with TemporaryDirectory() as tmp, fixture_counts(), patch(
                "src.models.classification.create_model", side_effect=lambda name, classes, precision: TinyClassifier(classes)):
            fixture(Path(tmp) / "manual")
            cp.prepare(Path(tmp) / "data", manual_dir=Path(tmp) / "manual")
            for method in ("adam", "dpco_filter", "dpco_flip"):
                config = resolve_config(PRESET, mode="train", steps=2, batch_size=2, method=method,
                    data_dir=str(Path(tmp) / "data"), eval_every_steps=1, checkpoint_every_steps=1)
                config = replace(config, results=ResultsConfig(str(Path(tmp) / "results")))
                args = SimpleNamespace(mode="train", run_name=method, resume=False, reserved=False, resume_slot="latest")
                with redirect_stdout(StringIO()):
                    run(args, config, jax.local_devices(), status=None)
                    store = CheckpointStore(Path(tmp) / "results/runs" / method)
                    expected = store.load("final")["train_state"]
                    args.resume, args.resume_slot = True, "p50"
                    run(args, config, jax.local_devices(), status=None)
                    actual = CheckpointStore(store.directory).load("final")["train_state"]
                assert_tree(self, actual, expected, exact=True)
                files = Path(tmp) / "results/files" / method
                self.assertEqual(len(read_rows(files / "train_steps.csv")), 2)
                self.assertEqual(len(read_rows(files / "group_evaluation.csv")), 6)
                self.assertEqual({r["group"] for r in read_rows(files / "group_evaluation.csv")}, {"clean", "corrupted_noisy"})
                self.assertEqual(int(actual.optimizer_state.count), 2)
                if method != "adam":
                    self.assertEqual(int(actual.method_state.updates), 2)
                self.assertNotIn("best", CheckpointStore(store.directory).index)


if __name__ == "__main__":
    unittest.main()
